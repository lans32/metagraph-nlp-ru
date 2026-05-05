"""Оркестрация стадий pipeline: text → sentences → parse → clauses → graph → metagraph.

Каждая стадия — явный вызов с явным входом и выходом (CLAUDE.md §9.6).
Морфо-синтаксический парсер инжектируется через аргумент `parser` — это
позволяет подменять реализацию в тестах (fake) и в продакшене (natasha).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from metagraph_nlp.aggregators import (
    aggregate_clauses_to_metanodes,
    aggregate_clauses_to_paragraphs,
    build_shared_entity_metaedges,
    build_topic_overlap_metaedges,
)
from metagraph_nlp.config import Config
from metagraph_nlp.domain import (
    Clause,
    Document,
    IdFactory,
    Metagraph,
    Provenance,
    SemanticGraph,
    Sentence,
)
from metagraph_nlp.graph_builders import build_semantic_graph
from metagraph_nlp.io import write_pipeline_artifacts
from metagraph_nlp.parsers import extract_clauses, normalize_text, split_sentences
from metagraph_nlp.parsers.morphsyntax import MorphSyntaxParser, ParsedSentence
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


def _default_parser() -> MorphSyntaxParser:
    from metagraph_nlp.parsers.morphsyntax.natasha_adapter import get_natasha_parser

    return get_natasha_parser()


def run(
    raw_text: str,
    *,
    config: Config | None = None,
    source_path: str | None = None,
    parser: MorphSyntaxParser | None = None,
) -> PipelineResult:
    cfg = config or Config()
    cfg_hash = cfg.hash()
    ids = IdFactory()
    audit = AuditLog()

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
    audit.record("normalize", "normalize_text", outputs=[doc_id])

    sentences = split_sentences(document, ids)
    audit.record(
        "segment",
        "razdel.sentenize",
        inputs=[doc_id],
        outputs=[s.id for s in sentences],
    )

    # Морфо-синтаксический разбор каждого предложения. Ленивый fallback
    # на natasha, если парсер не передали извне.
    active_parser = parser or _default_parser()
    parsed_sentences: dict[str, ParsedSentence] = {}
    for s in sentences:
        parsed_sentences[s.id] = active_parser.parse(s.span.text)
    audit.record(
        "parse",
        f"morphsyntax:{cfg.morphsyntax.parser}",
        inputs=[s.id for s in sentences],
        outputs=[f"parsed:{s.id}" for s in sentences],
    )

    clauses = extract_clauses(
        sentences,
        ids,
        parsed_sentences=parsed_sentences,
        strategy=cfg.clauses.strategy,
    )
    audit.record(
        "clauses",
        cfg.clauses.strategy,
        inputs=[s.id for s in sentences],
        outputs=[c.id for c in clauses],
    )

    graph = build_semantic_graph(document, clauses, parsed_sentences, ids)
    audit.record(
        "graph_builder",
        cfg.graph.builder,
        inputs=[c.id for c in clauses],
        outputs=[n.id for n in graph.nodes] + [e.id for e in graph.edges],
    )

    metagraph = aggregate_clauses_to_metanodes(graph, clauses, ids)
    audit.record(
        "aggregate",
        "clause_as_metanode_v0",
        inputs=[c.id for c in clauses],
        outputs=[mn.id for mn in metagraph.meta_nodes],
    )

    if cfg.aggregation.shared_entity_enabled:
        new_medges = build_shared_entity_metaedges(
            metagraph,
            graph,
            ids,
            min_lemma_len=cfg.aggregation.shared_entity_min_lemma_len,
            exclude_upos=frozenset(cfg.aggregation.shared_entity_exclude_upos),
        )
        audit.record(
            "aggregate",
            "shared_entity_by_lemma_v0",
            inputs=[mn.id for mn in metagraph.meta_nodes],
            outputs=[me.id for me in new_medges],
        )

    if cfg.aggregation.paragraph_enabled:
        new_l2_nodes = aggregate_clauses_to_paragraphs(
            metagraph, clauses, sentences, ids
        )
        audit.record(
            "aggregate",
            "paragraph_clauses_v0",
            inputs=[c.id for c in clauses],
            outputs=[mn.id for mn in new_l2_nodes],
        )

    if cfg.aggregation.topic_overlap_enabled:
        new_l2_medges = build_topic_overlap_metaedges(
            metagraph,
            ids,
            min_overlap=cfg.aggregation.topic_overlap_min_overlap,
        )
        audit.record(
            "aggregate",
            "topic_overlap_v0",
            inputs=[mn.id for mn in metagraph.meta_nodes if mn.level == 2],
            outputs=[me.id for me in new_l2_medges],
        )

    return PipelineResult(
        document=document,
        sentences=sentences,
        parsed_sentences=parsed_sentences,
        clauses=clauses,
        graph=graph,
        metagraph=metagraph,
        audit=audit,
        config=cfg,
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
    )
    return result
