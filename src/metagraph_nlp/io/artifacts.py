"""Запись промежуточных артефактов в JSON/JSONL (CLAUDE.md §11.1, §11.2)."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel

if TYPE_CHECKING:
    from metagraph_nlp.config import Config
    from metagraph_nlp.domain import (
        Clause,
        Document,
        Metagraph,
        SemanticGraph,
        Sentence,
    )
    from metagraph_nlp.provenance import AuditLog


def _write_json(path: Path, model: BaseModel) -> None:
    path.write_text(model.model_dump_json(indent=2), encoding="utf-8")


def _write_jsonl(path: Path, items: list[BaseModel]) -> None:
    lines = [m.model_dump_json(exclude_none=True) for m in items]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_pipeline_artifacts(
    out_dir: Path,
    *,
    document: Document,
    sentences: list[Sentence],
    clauses: list[Clause],
    graph: SemanticGraph,
    metagraph: Metagraph,
    audit: AuditLog,
    config: Config,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    _write_json(out_dir / "document.json", document)
    _write_jsonl(out_dir / "sentences.jsonl", list(sentences))
    _write_jsonl(out_dir / "clauses.jsonl", list(clauses))
    _write_json(out_dir / "semantic_graph.json", graph)
    _write_json(out_dir / "metagraph.json", metagraph)

    (out_dir / "audit.jsonl").write_text(audit.to_jsonl(), encoding="utf-8")

    snapshot = {
        "config_hash": config.hash(),
        "config": config.model_dump(),
    }
    (out_dir / "config.snapshot.yaml").write_text(
        yaml.safe_dump(snapshot, allow_unicode=True, sort_keys=False),
        encoding="utf-8",
    )
