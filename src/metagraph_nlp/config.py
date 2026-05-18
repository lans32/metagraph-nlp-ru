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
        description="Имя реализации `MorphSyntaxParser`: natasha | maltparser.",
    )
    malt_jar: str | None = Field(default=None, description="Путь к maltparser.jar")
    malt_model: str | None = Field(default=None, description="Путь к модели MaltParser")
    workers: int = Field(
        default=1,
        ge=1,
        description=(
            "Число параллельных процессов для парсинга предложений. "
            "1 — последовательно; >1 — ProcessPool. Активируется при "
            "количестве предложений ≥ parallel_threshold."
        ),
    )
    parallel_threshold: int = Field(
        default=16,
        ge=2,
        description=(
            "Минимальное число предложений для активации параллельного "
            "парсинга. На малых документах накладные расходы spawn'а "
            "превышают выигрыш."
        ),
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
    entity_cluster_enabled: bool = Field(
        default=True,
        description=(
            "Создавать L2-метавершины — тематические кластеры по графу "
            "shared_entity (entity_cluster_v0). Включает холархию: одна "
            "L1-клауза может попасть и в свой параграф, и в entity_cluster."
        ),
    )
    entity_cluster_min_size: int = 2
    predicate_class_cluster_enabled: bool = Field(
        default=True,
        description=(
            "Создавать L2-метавершины — кластеры клауз по классам "
            "предикатов из словаря (predicate_class_cluster_v0)."
        ),
    )
    predicate_class_cluster_min_size: int = 2
    predicate_classes_path: str | None = Field(
        default=None,
        description=(
            "Путь к YAML-словарю predicate-классов. None → "
            "встроенный configs/predicate_classes.yaml."
        ),
    )
    np_collapse_enabled: bool = Field(
        default=False,
        description="Сворачивать именные группы (NP) в один узел графа перед агрегацией.",
    )
    entity_centric_enabled: bool = Field(
        default=False,
        description=(
            "Создавать L2-метавершины — по одной на каждую значимую сущность "
            "(entity_centric_v0). Одна L1-клауза может входить в несколько "
            "entity-метавершин (холархия)."
        ),
    )
    entity_centric_min_freq: int = Field(
        default=2,
        ge=1,
        description="Минимальное число клауз с леммой для создания entity-метавершины.",
    )
    entity_centric_propn_always: bool = Field(
        default=True,
        description="PROPN-леммы (имена собственные) включаются при freq >= 1.",
    )


class SalienceWeights(BaseModel):
    """Веса упрощённого Lappin–Leass-скоринга кандидатов в антецеденты.

    Лучший кандидат = argmax(score). Hard constraints (Number, Gender,
    Animacy) применяются как фильтры до скоринга — несовместимые
    кандидаты в скоринг не попадают вовсе.
    """

    subj: int = Field(default=80, description="Бонус за роль nsubj/nsubj:pass у кандидата.")
    obj: int = Field(default=50, description="Бонус за роль obj у кандидата.")
    oblique: int = Field(
        default=20,
        description="Бонус за obl/iobj/nmod у кандидата.",
    )
    propn: int = Field(default=50, description="Бонус, если кандидат — PROPN (именованная сущность).")
    recency_per_sent: int = Field(
        default=-10,
        description=(
            "Штраф за расстояние в предложениях между PRON и кандидатом. "
            "Должен быть отрицательным (или 0)."
        ),
    )
    thematic: int = Field(
        default=20,
        description="Бонус за тематическую позицию (кандидат в начале предложения).",
    )
    thematic_token_threshold: int = Field(
        default=3,
        ge=1,
        description="Какие token_id_in_sent считаются началом предложения (≤ threshold).",
    )
    repeat_mention: int = Field(
        default=30,
        description=(
            "Бонус за каждое предыдущее разрешение, в котором этот кандидат "
            "уже выступал антецедентом (учёт salience через повтор)."
        ),
    )


class AnaphoraConfig(BaseModel):
    enabled: bool = Field(
        default=False,
        description=(
            "Включить разрешение анафоры (anaphora_resolution_v1): "
            "заменять местоимения на найденные антецеденты с сохранением узла."
        ),
    )
    search_window_sentences: int = Field(
        default=2,
        ge=1,
        description="Сколько предложений назад от местоимения искать антецедент.",
    )
    require_animacy_match: bool = Field(
        default=True,
        description="Требовать совпадения Animacy (Anim/Inan) PRON-токена и антецедента.",
    )
    pronoun_types: list[str] = Field(
        default_factory=lambda: ["personal_3p", "possessive_3p", "reflexive"],
        description=(
            "Типы покрываемых местоимений: personal_3p (он/она/оно/они), "
            "possessive_3p (его/её/их с Poss=Yes), reflexive (себя, свой)."
        ),
    )
    salience_weights: SalienceWeights = Field(
        default_factory=SalienceWeights,
        description="Веса salience-скоринга кандидатов (упрощённый Lappin–Leass).",
    )


class PathsConfig(BaseModel):
    data_dir: str = "data"
    artifacts_dir: str = "artifacts"


class Config(BaseModel):
    paths: PathsConfig = Field(default_factory=PathsConfig)
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    morphsyntax: MorphSyntaxConfig = Field(default_factory=MorphSyntaxConfig)
    clauses: ClauseConfig = Field(default_factory=ClauseConfig)
    graph: GraphConfig = Field(default_factory=GraphConfig)
    anaphora: AnaphoraConfig = Field(default_factory=AnaphoraConfig)
    aggregation: AggregationConfig = Field(default_factory=AggregationConfig)
    log_level: str = "INFO"

    def hash(self) -> str:
        blob = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]

    def phase1_hash(self) -> str:
        phase1_fields = {
            "morphsyntax": self.morphsyntax.model_dump(),
            "clauses": self.clauses.model_dump(),
            "graph": self.graph.model_dump(),
            "anaphora": self.anaphora.model_dump(),
            "np_collapse_enabled": self.aggregation.np_collapse_enabled,
            "predicate_classes_path": self.aggregation.predicate_classes_path,
        }
        blob = json.dumps(phase1_fields, sort_keys=True, ensure_ascii=False).encode("utf-8")
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
