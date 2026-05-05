"""Конфигурация pipeline: загрузка YAML + pydantic-схема (CLAUDE.md §13)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import yaml
from pydantic import BaseModel, Field


class SegmentationConfig(BaseModel):
    sentence_splitter: str = "razdel"


class ClauseConfig(BaseModel):
    strategy: str = Field(
        default="ud_subtree_clauses_v0",
        description=(
            "Стратегия выделения клауз. Варианты: "
            "`sentence_as_clause_v0` (одно предложение = клауза), "
            "`ud_subtree_clauses_v0` (поддерево финитного предиката)."
        ),
    )


class GraphConfig(BaseModel):
    builder: str = Field(
        default="ud_roles_v0",
        description="Билдер графа: `ud_roles_v0` (UD-роли) или `naive_head_dep_v0`.",
    )


class MorphSyntaxConfig(BaseModel):
    parser: str = Field(
        default="natasha",
        description="Имя реализации `MorphSyntaxParser`: сейчас только `natasha`.",
    )


class AggregationConfig(BaseModel):
    linguistic_enabled: bool = True
    structural_enabled: bool = False
    semantic_enabled: bool = False
    shared_entity_enabled: bool = Field(
        default=True,
        description="Создавать метарёбра shared_entity по общим леммам.",
    )
    shared_entity_min_lemma_len: int = 3
    shared_entity_exclude_upos: list[str] = Field(
        default_factory=lambda: ["PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART"],
    )
    paragraph_enabled: bool = Field(
        default=True,
        description="Создавать L2-метавершины по параграфам (paragraph_clauses_v0).",
    )
    topic_overlap_enabled: bool = Field(
        default=True,
        description=(
            "Создавать L2-метарёбра topic_overlap между L2-метавершинами с "
            "пересекающимися L1-фрагментами."
        ),
    )
    topic_overlap_min_overlap: int = 1


class PathsConfig(BaseModel):
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


class Config(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    morphsyntax: MorphSyntaxConfig = Field(default_factory=MorphSyntaxConfig)
    clauses: ClauseConfig = Field(default_factory=ClauseConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    log_level: str = "INFO"

    def hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def load_config(path: str | Path | None) -> Config:
    if path is None:
        return Config()
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return Config.model_validate(data)
