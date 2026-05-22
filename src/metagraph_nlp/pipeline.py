"""Оркестрация стадий pipeline: text → sentences → parse → clauses → graph → metagraph.

Каждая стадия — явный вызов с явным входом и выходом (CLAUDE.md §9.6).
Морфо-синтаксический парсер инжектируется через аргумент `parser` — это
позволяет подменять реализацию в тестах (fake) и в продакшене (natasha).

Pipeline разделён на две фазы:
- Phase 1 (run_phase1): text → semantic graph. Тяжёлая, кэшируется в UI.
- Phase 2 (run_phase2): semantic graph → metagraph. Лёгкая, перезапускается
  мгновенно при смене настроек агрегации.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

from metagraph_nlp.logging_config import setup_logging

logger = logging.getLogger("metagraph_nlp.pipeline")

from metagraph_nlp.aggregators import (
    aggregate_clauses_to_metanodes,
    aggregate_clauses_to_paragraphs,
    aggregate_entity_centric,
    aggregate_entity_clusters,
    aggregate_predicate_class_clusters,
    build_shared_entity_metaedges,
    build_topic_overlap_metaedges,
)
from metagraph_nlp.config import Config
from metagraph_nlp.domain import (
    AnaphoraResolution,
    Clause,
    Document,
    IdFactory,
    Metagraph,
    Provenance,
    SemanticGraph,
    Sentence,
)
from metagraph_nlp.graph_builders import build_semantic_graph
from metagraph_nlp.graph_builders.np_collapse import collapse_noun_phrases
from metagraph_nlp.io import write_pipeline_artifacts
from metagraph_nlp.parsers import (
    extract_clauses,
    normalize_text,
    resolve_anaphora,
    split_sentences,
)
from metagraph_nlp.parsers.predicate_lexicon import (
    load_predicate_classes,
    load_predicate_hierarchy,
)
from metagraph_nlp.parsers.morphsyntax import MorphSyntaxParser, ParsedSentence
from metagraph_nlp.profiling import PipelineMetrics, measure_stage
from metagraph_nlp.provenance import AuditLog


@dataclass
class Phase1Result:
    """Результат Phase 1: text → semantic graph. Кэшируется в UI."""

    document: Document
    sentences: list[Sentence]
    parsed_sentences: dict[str, ParsedSentence]
    clauses: list[Clause]
    graph: SemanticGraph
    audit: AuditLog
    config: Config
    metrics: PipelineMetrics
    id_snapshot: dict[str, int] = field(default_factory=dict)
    anaphora_resolutions: list[AnaphoraResolution] | None = None


@dataclass
class PipelineResult:
    document: Document
    sentences: list[Sentence]
    parsed_sentences: dict[str, ParsedSentence]
    clauses: list[Clause]
    graph: SemanticGraph
    metagraph: Metagraph
    audit: AuditLog
    config: Config
    metrics: PipelineMetrics | None = None
    anaphora_resolutions: list[AnaphoraResolution] | None = None


def get_default_parser(cfg: Config | None = None) -> MorphSyntaxParser:
    parser_name = cfg.morphsyntax.parser if cfg else "natasha"

    if parser_name == "natasha":
        from metagraph_nlp.parsers.morphsyntax.natasha_adapter import get_natasha_parser
        return get_natasha_parser()

    if parser_name == "maltparser":
        return _build_maltparser_treetagger(cfg)

    raise ValueError(f"Unknown parser: {parser_name}")


def _build_maltparser_treetagger(cfg: Config | None) -> MorphSyntaxParser:
    """Собрать цепочку razdel → TreeTagger → MaltParser из конфига."""
    from metagraph_nlp.parsers.morphsyntax.maltparser_adapter import TreeTaggerMaltParser
    from metagraph_nlp.parsers.morphsyntax.treetagger_adapter import TreeTaggerMorph
    from metagraph_nlp.parsers.morphsyntax.treetagger_tagmap import (
        MsdTagMapper,
        default_tagset_path,
    )

    if cfg is None:
        raise ValueError("maltparser requires a full Config (not None)")
    ms = cfg.morphsyntax
    missing: list[str] = []
    if not ms.tree_tagger_bin:
        missing.append("tree_tagger_bin")
    if not ms.tree_tagger_param:
        missing.append("tree_tagger_param")
    if not ms.malt_jar:
        missing.append("malt_jar")
    if not ms.malt_model:
        missing.append("malt_model")
    if missing:
        raise ValueError(
            "maltparser parser requires morphsyntax fields: "
            + ", ".join(missing)
        )

    tagmap = MsdTagMapper(default_tagset_path(ms.tree_tagger_tagset))
    morph = TreeTaggerMorph(
        bin_path=Path(ms.tree_tagger_bin),
        param_path=Path(ms.tree_tagger_param),
        tagmap=tagmap,
    )

    deprel_mapping: dict[str, str] | None = None
    if ms.malt_deprel_mapping:
        import yaml as _yaml

        with open(ms.malt_deprel_mapping, encoding="utf-8") as f:
            loaded = _yaml.safe_load(f) or {}
        if isinstance(loaded, dict):
            deprel_mapping = {str(k): str(v) for k, v in loaded.items()}

    return TreeTaggerMaltParser(
        malt_jar=Path(ms.malt_jar),
        model_path=Path(ms.malt_model),
        morph=morph,
        java_bin=ms.java_bin,
        deprel_mapping=deprel_mapping,
    )


def _default_parser() -> MorphSyntaxParser:
    return get_default_parser()


_WORKER_PARSER: MorphSyntaxParser | None = None


def _worker_init(morphsyntax_payload: dict) -> None:
    """Pickle-safe init: payload — dict из `MorphSyntaxConfig.model_dump()`."""
    from metagraph_nlp.config import MorphSyntaxConfig

    global _WORKER_PARSER
    cfg = Config(morphsyntax=MorphSyntaxConfig(**morphsyntax_payload))
    _WORKER_PARSER = get_default_parser(cfg)


def _worker_parse(text: str) -> ParsedSentence:
    assert _WORKER_PARSER is not None
    return _WORKER_PARSER.parse(text)


def _parse_sentences_parallel(
    texts: list[str],
    cfg: Config,
    workers: int,
) -> list[ParsedSentence]:
    from concurrent.futures import ProcessPoolExecutor

    with ProcessPoolExecutor(
        max_workers=workers,
        initializer=_worker_init,
        initargs=(cfg.morphsyntax.model_dump(),),
    ) as pool:
        return list(pool.map(_worker_parse, texts))


def run_phase1(
    raw_text: str,
    *,
    config: Config | None = None,
    source_path: str | None = None,
    parser: MorphSyntaxParser | None = None,
) -> Phase1Result:
    """Phase 1: text → semantic graph. Тяжёлая фаза, кэшируется в UI."""
    cfg = config or Config()
    setup_logging(cfg.log_level)
    cfg_hash = cfg.hash()
    ids = IdFactory()
    audit = AuditLog()
    metrics = PipelineMetrics()

    with measure_stage("normalize", metrics) as sm:
        normalized = normalize_text(raw_text)
        doc_id = ids.doc()
        document = Document(
            id=doc_id,
            source_path=source_path,
            raw_text=raw_text,
            normalized_text=normalized,
            provenance=Provenance(
                rule="normalize_text",
                stage="normalize",
                inputs=[],
                document_id=doc_id,
                config_hash=cfg_hash,
            ),
        )
        sm.output_count = 1
    audit.record("normalize", "normalize_text", outputs=[doc_id])
    logger.info("normalize: doc_id=%s, len=%d", doc_id, len(normalized))

    with measure_stage("segment", metrics) as sm:
        sentences = split_sentences(document, ids)
        sm.output_count = len(sentences)
    audit.record(
        "segment",
        "razdel.sentenize",
        inputs=[doc_id],
        outputs=[s.id for s in sentences],
    )
    logger.info("segment: %d sentences", len(sentences))

    with measure_stage("parse", metrics) as sm:
        parsed_sentences: dict[str, ParsedSentence] = {}
        use_parallel = (
            parser is None
            and cfg.morphsyntax.workers > 1
            and len(sentences) >= cfg.morphsyntax.parallel_threshold
        )
        if use_parallel:
            texts = [s.span.text for s in sentences]
            results = _parse_sentences_parallel(
                texts, cfg, cfg.morphsyntax.workers
            )
            for s, parsed in zip(sentences, results):
                parsed_sentences[s.id] = parsed
            logger.info(
                "parse: %d sentences parsed in parallel (workers=%d)",
                len(parsed_sentences),
                cfg.morphsyntax.workers,
            )
        else:
            active_parser = parser or _default_parser()
            for s in sentences:
                parsed_sentences[s.id] = active_parser.parse(s.span.text)
            logger.info("parse: %d sentences parsed sequentially", len(parsed_sentences))
        sm.output_count = len(parsed_sentences)
    audit.record(
        "parse",
        f"morphsyntax:{cfg.morphsyntax.parser}",
        inputs=[s.id for s in sentences],
        outputs=[f"parsed:{s.id}" for s in sentences],
        params=(
            {"workers": str(cfg.morphsyntax.workers)}
            if use_parallel else None
        ),
    )

    with measure_stage("clauses", metrics) as sm:
        clauses = extract_clauses(
            sentences,
            ids,
            parsed_sentences=parsed_sentences,
            strategy=cfg.clauses.strategy,
        )
        sm.output_count = len(clauses)
    audit.record(
        "clauses",
        cfg.clauses.strategy,
        inputs=[s.id for s in sentences],
        outputs=[c.id for c in clauses],
    )
    logger.info("clauses: %d clauses extracted", len(clauses))

    predicate_classes = load_predicate_classes(cfg.aggregation.predicate_classes_path)

    with measure_stage("graph_builder", metrics) as sm:
        graph = build_semantic_graph(
            document, clauses, parsed_sentences, ids,
            predicate_classes=predicate_classes,
        )
        sm.output_count = len(graph.nodes) + len(graph.edges)
    audit.record(
        "graph_builder",
        cfg.graph.builder,
        inputs=[c.id for c in clauses],
        outputs=[n.id for n in graph.nodes] + [e.id for e in graph.edges],
    )
    logger.info("graph: %d nodes, %d edges", len(graph.nodes), len(graph.edges))

    anaphora_resolutions: list[AnaphoraResolution] | None = None
    if cfg.anaphora.enabled:
        with measure_stage("anaphora_resolution", metrics) as sm:
            graph, anaphora_resolutions = resolve_anaphora(
                graph,
                clauses,
                sentences,
                parsed_sentences,
                ids,
                search_window_sentences=cfg.anaphora.search_window_sentences,
                require_animacy_match=cfg.anaphora.require_animacy_match,
                pronoun_types=cfg.anaphora.pronoun_types,
                salience_weights=cfg.anaphora.salience_weights,
            )
            sm.output_count = len(anaphora_resolutions)
        audit.record(
            "anaphora_resolution",
            "anaphora_resolution_v1",
            inputs=[r.pronoun_node_id for r in anaphora_resolutions],
            outputs=[r.antecedent_node_id for r in anaphora_resolutions],
        )
        logger.info(
            "anaphora: %d pronouns replaced, %d nodes total",
            len(anaphora_resolutions),
            len(graph.nodes),
        )

    if cfg.aggregation.np_collapse_enabled:
        with measure_stage("np_collapse", metrics) as sm:
            graph = collapse_noun_phrases(
                graph,
                parsed_sentences,
                clauses,
                ids,
                include_deprels=set(cfg.aggregation.np_collapse_deprels),
            )
            sm.output_count = len(graph.nodes) + len(graph.edges)
        audit.record(
            "np_collapse",
            "np_collapse_v0",
            inputs=[n.id for n in graph.nodes],
            outputs=[n.id for n in graph.nodes],
        )
        logger.info("np_collapse: %d nodes, %d edges after collapse", len(graph.nodes), len(graph.edges))

    logger.info("phase1 complete: %.3fs", metrics.total_wall_seconds)

    return Phase1Result(
        document=document,
        sentences=sentences,
        parsed_sentences=parsed_sentences,
        clauses=clauses,
        graph=graph,
        audit=audit,
        config=cfg,
        metrics=metrics,
        id_snapshot=ids.snapshot(),
        anaphora_resolutions=anaphora_resolutions,
    )


def run_phase2(
    phase1: Phase1Result,
    *,
    config: Config | None = None,
) -> PipelineResult:
    """Phase 2: semantic graph → metagraph. Лёгкая фаза, перезапускается мгновенно."""
    cfg = config or phase1.config
    ids = IdFactory.from_snapshot(phase1.id_snapshot)
    audit = AuditLog(events=list(phase1.audit.events))
    metrics = PipelineMetrics(
        stages=list(phase1.metrics.stages),
        total_wall_seconds=phase1.metrics.total_wall_seconds,
    )

    graph = phase1.graph
    clauses = phase1.clauses
    sentences = phase1.sentences

    with measure_stage("aggregate_L1", metrics) as sm:
        metagraph = aggregate_clauses_to_metanodes(graph, clauses, ids)
        sm.output_count = len(metagraph.meta_nodes)
    audit.record(
        "aggregate",
        "clause_as_metanode_v0",
        inputs=[c.id for c in clauses],
        outputs=[mn.id for mn in metagraph.meta_nodes],
    )
    logger.info("aggregate L1: %d metanodes", len(metagraph.meta_nodes))

    if cfg.aggregation.shared_entity_enabled:
        with measure_stage("aggregate_L1_edges", metrics) as sm:
            new_medges = build_shared_entity_metaedges(
                metagraph,
                graph,
                ids,
                min_lemma_len=cfg.aggregation.shared_entity_min_lemma_len,
                exclude_upos=frozenset(cfg.aggregation.shared_entity_exclude_upos),
            )
            sm.output_count = len(new_medges)
        audit.record(
            "aggregate",
            "shared_entity_by_lemma_v0",
            inputs=[mn.id for mn in metagraph.meta_nodes],
            outputs=[me.id for me in new_medges],
        )
        logger.info("aggregate L1-edges: %d shared_entity metaedges", len(new_medges))

    if cfg.aggregation.paragraph_enabled:
        with measure_stage("aggregate_L2_paragraph", metrics) as sm:
            new_para_nodes = aggregate_clauses_to_paragraphs(
                metagraph, graph, clauses, sentences, ids
            )
            sm.output_count = len(new_para_nodes)
        audit.record(
            "aggregate",
            "paragraph_clauses_v0",
            inputs=[c.id for c in clauses],
            outputs=[mn.id for mn in new_para_nodes],
            params={"L2_strategy": "paragraph"},
        )
        logger.info(
            "aggregate L2 paragraph: %d metanodes", len(new_para_nodes)
        )

    if cfg.aggregation.entity_cluster_enabled:
        with measure_stage("aggregate_L2_entity_cluster", metrics) as sm:
            new_entity_nodes = aggregate_entity_clusters(
                metagraph,
                graph,
                ids,
                min_cluster_size=cfg.aggregation.entity_cluster_min_size,
            )
            sm.output_count = len(new_entity_nodes)
        audit.record(
            "aggregate",
            "entity_cluster_v0",
            inputs=[mn.id for mn in metagraph.meta_nodes if mn.level == 1],
            outputs=[mn.id for mn in new_entity_nodes],
            params={"L2_strategy": "entity_cluster"},
        )
        logger.info(
            "aggregate L2 entity_cluster: %d metanodes", len(new_entity_nodes)
        )

    if cfg.aggregation.entity_centric_enabled:
        with measure_stage("aggregate_L2_entity_centric", metrics) as sm:
            new_ec_nodes = aggregate_entity_centric(
                metagraph,
                graph,
                ids,
                min_freq=cfg.aggregation.entity_centric_min_freq,
                propn_always=cfg.aggregation.entity_centric_propn_always,
            )
            sm.output_count = len(new_ec_nodes)
        audit.record(
            "aggregate",
            "entity_centric_v0",
            inputs=[mn.id for mn in metagraph.meta_nodes if mn.level == 1],
            outputs=[mn.id for mn in new_ec_nodes],
            params={"L2_strategy": "entity_centric"},
        )
        logger.info(
            "aggregate L2 entity_centric: %d metanodes", len(new_ec_nodes)
        )

    if cfg.aggregation.predicate_class_cluster_enabled:
        predicate_hierarchy = load_predicate_hierarchy(
            cfg.aggregation.predicate_classes_path
        )
        with measure_stage("aggregate_L2_predicate_class", metrics) as sm:
            new_pred_nodes = aggregate_predicate_class_clusters(
                metagraph,
                graph,
                ids,
                min_cluster_size=cfg.aggregation.predicate_class_cluster_min_size,
                hierarchy=predicate_hierarchy,
                levels=cfg.aggregation.predicate_class_cluster_levels,
                create_containment_edges=(
                    cfg.aggregation.predicate_hierarchy_edges_enabled
                    and predicate_hierarchy is not None
                ),
            )
            sm.output_count = len(new_pred_nodes)
        audit.record(
            "aggregate",
            "predicate_class_cluster_v0",
            inputs=[mn.id for mn in metagraph.meta_nodes if mn.level == 1],
            outputs=[mn.id for mn in new_pred_nodes],
            params={
                "L2_strategy": "predicate_class",
                "hierarchy": "v1" if predicate_hierarchy is not None else "v0",
                "levels": ",".join(cfg.aggregation.predicate_class_cluster_levels),
                "containment_edges": str(
                    cfg.aggregation.predicate_hierarchy_edges_enabled
                    and predicate_hierarchy is not None
                ),
            },
        )
        logger.info(
            "aggregate L2 predicate_class: %d metanodes "
            "(hierarchy=%s, levels=%s)",
            len(new_pred_nodes),
            "v1" if predicate_hierarchy is not None else "v0",
            cfg.aggregation.predicate_class_cluster_levels,
        )

    if cfg.aggregation.topic_overlap_enabled:
        with measure_stage("aggregate_L2_edges", metrics) as sm:
            new_l2_medges = build_topic_overlap_metaedges(
                metagraph,
                ids,
                min_overlap=cfg.aggregation.topic_overlap_min_overlap,
            )
            sm.output_count = len(new_l2_medges)
        audit.record(
            "aggregate",
            "topic_overlap_v0",
            inputs=[mn.id for mn in metagraph.meta_nodes if mn.level == 2],
            outputs=[me.id for me in new_l2_medges],
        )
        logger.info("aggregate L2-edges: %d topic_overlap metaedges", len(new_l2_medges))

    logger.info("phase2 complete: %.3fs total", metrics.total_wall_seconds)

    return PipelineResult(
        document=phase1.document,
        sentences=sentences,
        parsed_sentences=phase1.parsed_sentences,
        clauses=clauses,
        graph=graph,
        metagraph=metagraph,
        audit=audit,
        config=cfg,
        metrics=metrics,
        anaphora_resolutions=phase1.anaphora_resolutions,
    )


def run(
    raw_text: str,
    *,
    config: Config | None = None,
    source_path: str | None = None,
    parser: MorphSyntaxParser | None = None,
) -> PipelineResult:
    cfg = config or Config()
    p1 = run_phase1(raw_text, config=cfg, source_path=source_path, parser=parser)
    return run_phase2(p1, config=cfg)


def run_from_file(
    input_path: Path,
    out_dir: Path,
    config: Config | None = None,
    viz: bool = False,
    parser: MorphSyntaxParser | None = None,
) -> PipelineResult:
    raw = input_path.read_text(encoding="utf-8")
    result = run(raw, config=config, source_path=str(input_path), parser=parser)
    write_pipeline_artifacts(
        out_dir,
        document=result.document,
        sentences=result.sentences,
        clauses=result.clauses,
        graph=result.graph,
        metagraph=result.metagraph,
        audit=result.audit,
        config=result.config,
        viz=viz,
        metrics=result.metrics,
        anaphora_resolutions=result.anaphora_resolutions,
    )
    return result
