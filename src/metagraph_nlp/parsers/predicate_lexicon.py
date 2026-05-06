"""Загрузка словаря семантических классов предикатов (CLAUDE.md §5.5).

Словарь — простой YAML вида::

    version: 0
    classes:
        motion: [идти, ехать, ...]
        communication: [говорить, сказать, ...]

Эта функция превращает прямой индекс ``class → [lemma]`` в обратный
``lemma → frozenset(classes)`` для O(1) lookup при построении графа
(`graph_builders/from_clause.py`).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

_DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "predicate_classes.yaml"
)


def load_predicate_classes(
    path: str | Path | None = None,
) -> dict[str, frozenset[str]]:
    """Загрузить словарь predicate-классов из YAML.

    Если ``path is None``, читается встроенный
    ``configs/predicate_classes.yaml``. Возвращает словарь
    ``lemma → frozenset(classes)``.
    """

    target = Path(path) if path is not None else _DEFAULT_LEXICON_PATH
    if not target.exists():
        raise FileNotFoundError(f"Predicate lexicon not found: {target}")

    with target.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}

    classes_section = data.get("classes") or {}
    inverted: dict[str, set[str]] = defaultdict(set)
    for class_name, lemmas in classes_section.items():
        if not isinstance(lemmas, list):
            continue
        for lemma in lemmas:
            if isinstance(lemma, str) and lemma:
                inverted[lemma].add(class_name)

    return {lemma: frozenset(classes) for lemma, classes in inverted.items()}
