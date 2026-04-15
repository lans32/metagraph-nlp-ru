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
        default="sentence_as_clause",
        description="Временная стратегия: каждое предложение = одна клауза.",
    )


class GraphConfig(BaseModel):
    builder: str = "naive_head_dep"


class AggregationConfig(BaseModel):
    linguistic_enabled: bool = True
    structural_enabled: bool = False
    semantic_enabled: bool = False


class PathsConfig(BaseModel):
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


class Config(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
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
