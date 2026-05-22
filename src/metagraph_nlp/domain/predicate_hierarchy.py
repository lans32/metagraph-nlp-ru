"""Иерархия классов предикатов из RuWordNet (CLAUDE.md §5.1, §4.4).

Структура — дерево классов с parent/level/anchor_synset_id плюс
обратный индекс «лемма → список путей от leaf к root». Используется
агрегатором `predicate_class_cluster_v0` для:

- multi-level кластеризации (создание L2 на каждом уровне иерархии);
- создания containment-метарёбер между parent ↔ child кластерами;
- расширенного provenance (трассировка до конкретного synset RuWordNet).

Формируется loader'ом `parsers.predicate_lexicon.load_predicate_hierarchy`
из YAML version=1. Для legacy YAML version=0 loader возвращает None.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class PredicateHierarchy(BaseModel):
    """Дерево predicate-классов с обратным индексом по леммам.

    Attributes:
        parent_of: class_slug → parent_slug | None (root-классы имеют None).
        level_of: class_slug → "root" | "mid" | "leaf".
        anchor_of: class_slug → synset_id RuWordNet (для трассировки).
        label_of: class_slug → русский label (title синсета).
        lemma_paths: lemma → список путей [leaf_slug, ..., root_slug].
            Одна лемма может встречаться в нескольких ветвях (полисемия).
        metadata: метаданные YAML (source, built_at, max_depth, и т.п.).
    """

    parent_of: dict[str, str | None] = Field(default_factory=dict)
    level_of: dict[str, str] = Field(default_factory=dict)
    anchor_of: dict[str, str] = Field(default_factory=dict)
    label_of: dict[str, str] = Field(default_factory=dict)
    lemma_paths: dict[str, list[list[str]]] = Field(default_factory=dict)
    metadata: dict[str, object] = Field(default_factory=dict)

    def roots(self) -> list[str]:
        """Список корневых классов (parent is None)."""
        return sorted(
            slug for slug, parent in self.parent_of.items() if parent is None
        )

    def children_of(self, slug: str) -> list[str]:
        """Список прямых детей класса (parent == slug)."""
        return sorted(
            child for child, parent in self.parent_of.items() if parent == slug
        )

    def classes_at_level(self, level: str) -> list[str]:
        """Список классов на заданном уровне (root / mid / leaf)."""
        return sorted(slug for slug, lvl in self.level_of.items() if lvl == level)
