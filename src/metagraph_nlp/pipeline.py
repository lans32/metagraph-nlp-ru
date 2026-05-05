"""Оркестрация стадий pipeline: text → sentences → parse → clauses → graph → metagraph.

Каждая стадия — явный вызов с явным входом и выходом (CLAUDE.md §9.6).
Морфо-синтаксический парсер инжектируется через аргумент `parser` — это
позволяет подменять реализацию в тестах (fake) и в продакшене (natasha).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from metagraph_nlp.logging_config import setup_logging

logger = logging.getLogger("metagraph_nlp.pipeline")

from metagraph_nlp.aggregators import (
    aggregate_clauses_to_metanodes,
    aggregate_clauses_to_paragraphs,
    aggregate_coref_clusters,
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
from metagraph_nlp.parsers.morphsyntax import MorphSyntaxParser, ParsedSentence
from metagraph_nlp.profiling import PipelineMetrics, measure_stage
from metagraph_nlp.provenance import AuditLog


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
        from metagraph_nlp.parsers.morphsyntax.maltparser_adapter import MaltParserAdapter
        if not cfg or not cfg.morphsyntax.malt_jar or not cfg.morphsyntax.malt_model:
            raise ValueError("maltparser requires morphsyntax.malt_jar and morphsyntax.malt_model in config")
        return MaltParserAdapter(
            malt_jar=Path(cfg.morphsyntax.malt_jar),
            model_path=Path(cfg.morphsyntax.malt_model),
        )

    raise ValueError(f"Unknown parser: {parser_name}")


def _default_parser() -> MorphSyntaxParser:
    return get_default_parser()


def run(
    raw_text: str,
    *,
    config: Config | None = None,
    source_path: str | None = None,
    parser: MorphSyntaxParser | None = None,
) -> PipelineResult:
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
        active_parser = parser or _default_parser()
        parsed_sentences: dict[str, ParsedSentence] = {}
        for s in sentences:
            parsed_sentences[s.id] = active_parser.parse(s.span.text)
        sm.output_count = len(parsed_sentences)
    audit.record(
        "parse",
        f"morphsyntax:{cfg.morphsyntax.parser}",
        inputs=[s.id for s in sentences],
        outputs=[f"parsed:{s.id}" for s in sentences],
    )
    logger.info("parse: %d sentences parsed", len(parsed_sentences))

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

    with measure_stage("graph_builder", metrics) as sm:
        graph = build_semantic_graph(document, clauses, parsed_sentences, ids)
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
            )
            sm.output_count = len(anaphora_resolutions)
        audit.record(
            "anaphora_resolution",
            "anaphora_resolution_v0",
            inputs=[r.pronoun_node_id for r in anaphora_resolutions],
            outputs=[r.antecedent_node_id for r in anaphora_resolutions],
        )
        logger.info(
            "anaphora: %d pronouns resolved, %d nodes remain",
            len(anaphora_resolutions),
            len(graph.nodes),
        )

    if cfg.aggregation.np_collapse_enabled:
        with measure_stage("np_collapse", metrics) as sm:
            graph = collapse_noun_phrases(graph, parsed_sentences, clauses, ids)
            sm.output_count = len(graph.nodes) + len(graph.edges)
        audit.record(
            "np_collapse",
            "np_collapse_v0",
            inputs=[n.id for n in graph.nodes],
            outputs=[n.id for n in graph.nodes],
        )
        logger.info("np_collapse: %d nodes, %d edges after collapse", len(graph.nodes), len(graph.edges))

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

    if cfg.aggregation.coref_cluster_enabled:
        with measure_stage("aggregate_coref_clusters", metrics) as sm:
            new_coref_nodes = aggregate_coref_clusters(
                metagraph,
                ids,
                min_cluster_size=cfg.aggregation.coref_cluster_min_size,
            )
            sm.output_count = len(new_coref_nodes)
        audit.record(
            "aggregate",
            "coref_cluster_v0",
            inputs=[mn.id for mn in metagraph.meta_nodes if mn.level == 1],
            outputs=[mn.id for mn in new_coref_nodes],
        )
        logger.info("aggregate coref_clusters: %d L2 metanodes", len(new_coref_nodes))

    if cfg.aggregation.paragraph_enabled:
        with measure_stage("aggregate_L2_nodes", metrics) as sm:
            new_l2_nodes = aggregate_clauses_to_paragraphs(
                metagraph, clauses, sentences, ids
            )
            sm.output_count = len(new_l2_nodes)
        audit.record(
            "aggregate",
            "paragraph_clauses_v0",
            inputs=[c.id for c in clauses],
            outputs=[mn.id for mn in new_l2_nodes],
        )
        logger.info("aggregate L2-nodes: %d paragraph metanodes", len(new_l2_nodes))

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

    logger.info("pipeline complete: %.3fs total", metrics.total_wall_seconds)

    return PipelineResult(
        document=document,
        sentences=sentences,
        parsed_sentences=parsed_sentences,
        clauses=clauses,
        graph=graph,
        metagraph=metagraph,
        audit=audit,
        config=cfg,
        metrics=metrics,
        anaphora_resolutions=anaphora_resolutions,
    )


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
