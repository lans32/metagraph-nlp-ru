"""Пресеты агрегации: предустановленные конфигурации переключателей агрегации."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from metagraph_nlp.config import Config


@dataclass
class Preset:
    key: str
    name: str
    description: str
    _apply: Callable[[Config], Config] | None = None

    def apply(self, config: Config) -> Config:
        if self._apply is None:
            return config
        return self._apply(config)


def _apply_clauses_only(config: Config) -> Config:
    agg = config.aggregation.model_copy(
        update={
            "linguistic_enabled": True,
            "structural_enabled": False,
            "semantic_enabled": False,
            "shared_entity_enabled": False,
            "paragraph_enabled": False,
            "topic_overlap_enabled": False,
            "entity_cluster_enabled": False,
            "predicate_class_cluster_enabled": False,
            "np_collapse_enabled": False,
            "entity_centric_enabled": False,
        },
    )
    return config.model_copy(update={"aggregation": agg})


def _apply_entities(config: Config) -> Config:
    agg = config.aggregation.model_copy(
        update={
            "linguistic_enabled": True,
            "structural_enabled": False,
            "semantic_enabled": False,
            "shared_entity_enabled": True,
            "paragraph_enabled": False,
            "topic_overlap_enabled": False,
            "entity_cluster_enabled": False,
            "predicate_class_cluster_enabled": False,
            "np_collapse_enabled": False,
            "entity_centric_enabled": True,
        },
    )
    return config.model_copy(update={"aggregation": agg})


def _apply_paragraphs(config: Config) -> Config:
    agg = config.aggregation.model_copy(
        update={
            "linguistic_enabled": True,
            "structural_enabled": False,
            "semantic_enabled": False,
            "shared_entity_enabled": True,
            "paragraph_enabled": True,
            "topic_overlap_enabled": True,
            "entity_cluster_enabled": False,
            "predicate_class_cluster_enabled": False,
            "np_collapse_enabled": False,
            "entity_centric_enabled": False,
        },
    )
    return config.model_copy(update={"aggregation": agg})


def _apply_predicates(config: Config) -> Config:
    agg = config.aggregation.model_copy(
        update={
            "linguistic_enabled": True,
            "structural_enabled": False,
            "semantic_enabled": False,
            "shared_entity_enabled": True,
            "paragraph_enabled": False,
            "topic_overlap_enabled": True,
            "entity_cluster_enabled": False,
            "predicate_class_cluster_enabled": True,
            "predicate_class_cluster_levels": ["leaf", "mid", "root"],
            "predicate_hierarchy_edges_enabled": True,
            "np_collapse_enabled": False,
            "entity_centric_enabled": False,
        },
    )
    return config.model_copy(update={"aggregation": agg})


def _apply_full(config: Config) -> Config:
    agg = config.aggregation.model_copy(
        update={
            "linguistic_enabled": True,
            "structural_enabled": False,
            "semantic_enabled": False,
            "shared_entity_enabled": True,
            "paragraph_enabled": True,
            "topic_overlap_enabled": True,
            "entity_cluster_enabled": True,
            "predicate_class_cluster_enabled": True,
            "predicate_class_cluster_levels": ["leaf", "mid", "root"],
            "predicate_hierarchy_edges_enabled": True,
            "np_collapse_enabled": False,
            "entity_centric_enabled": False,
        },
    )
    return config.model_copy(update={"aggregation": agg})


_PRESETS: list[Preset] = [
    Preset(
        key="clauses_only",
        name="Только клаузы",
        description="L1-метавершины без дополнительной агрегации",
        _apply=_apply_clauses_only,
    ),
    Preset(
        key="entities",
        name="По сущностям",
        description="Отдельная метавершина для каждой значимой сущности",
        _apply=_apply_entities,
    ),
    Preset(
        key="paragraphs",
        name="По параграфам",
        description="Группировка клауз по абзацам текста",
        _apply=_apply_paragraphs,
    ),
    Preset(
        key="predicates",
        name="По семантике предикатов",
        description="Кластеры клауз по классам глаголов",
        _apply=_apply_predicates,
    ),
    Preset(
        key="full",
        name="Полный",
        description="Все стратегии агрегации включены",
        _apply=_apply_full,
    ),
    Preset(
        key="custom",
        name="Пользовательский",
        description="Ручная настройка всех параметров",
    ),
]

PRESET_KEYS: list[str] = [p.key for p in _PRESETS]


def get_presets() -> list[Preset]:
    return list(_PRESETS)


def get_preset_by_key(key: str) -> Preset | None:
    for p in _PRESETS:
        if p.key == key:
            return p
    return None
