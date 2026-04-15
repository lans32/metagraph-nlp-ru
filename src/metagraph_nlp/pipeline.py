"""Оркестрация стадий pipeline: text → sentences → clauses → graph → metagraph.

Каждая стадия — явный вызов с явным входом и выходом (CLAUDE.md §9.6).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from metagraph_nlp.aggregators import aggregate_clauses_to_metanodes
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
from metagraph_nlp.provenance import AuditLog


@dataclass
class PipelineResult:
    document: Document
    sentences: list[Sentence]
    clauses: list[Clause]
    graph: SemanticGraph
    metagraph: Metagraph
    audit: AuditLog
    config: Config


def run(
    raw_text: str,
    *,
    config: Config | None = None,
    source_path: str | None = None,
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

    clauses = extract_clauses(sentences, ids)
    audit.record(
        "clauses",
        "sentence_as_clause_v0",
        inputs=[s.id for s in sentences],
        outputs=[c.id for c in clauses],
        notes="MVP stub: one clause per sentence",
    )

    graph = build_semantic_graph(document, clauses, ids)
    audit.record(
        "graph_builder",
        "naive_head_dep_v0",
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

    return PipelineResult(
        document=document,
        sentences=sentences,
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
) -> PipelineResult:
    raw = input_path.read_text(encoding="utf-8")
    result = run(raw, config=config, source_path=str(input_path))
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
