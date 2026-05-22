"""Загрузка словаря семантических классов предикатов (CLAUDE.md §5.5).

Поддерживаются две версии формата YAML:

**version=0 (legacy, рукописный)** — плоский словарь, ~100 лемм::

    version: 0
    classes:
        motion: [идти, ехать, ...]
        communication: [говорить, сказать, ...]

**version=1 (RuWordNet, иерархический)** — дерево классов + пути::

    version: 1
    metadata: {source: ruwordnet-2.0, ...}
    hierarchy:
        motion:
            parent: null
            level: root
            anchor_synset_id: "106587-V"
            label_ru: "движение, перемещение"
        motion_walking:
            parent: motion
            level: mid
            ...
    lemmas:
        идти:
            - [motion_walking, motion]
            - [change_of_state]   # переносное значение

Loader детектирует версию через ``data.get("version", 0)`` и возвращает
обратный индекс ``lemma → frozenset(classes)`` (для v1 — union всех
уровней всех путей). Для расширенной структуры с иерархией используйте
:func:`load_predicate_hierarchy` — она возвращает ``PredicateHierarchy``
для v1 и ``None`` для v0.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from metagraph_nlp.domain import PredicateHierarchy

_DEFAULT_LEXICON_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "predicate_classes.yaml"
)

# Кэш загруженного YAML по path+mtime — избегаем двойного парсинга,
# когда load_predicate_classes и load_predicate_hierarchy вызываются
# на один и тот же файл в Phase 1.
_YAML_CACHE: dict[tuple[Path, float], dict] = {}


def _resolve_path(path: str | Path | None) -> Path:
    target = Path(path) if path is not None else _DEFAULT_LEXICON_PATH
    if not target.exists():
        raise FileNotFoundError(f"Predicate lexicon not found: {target}")
    return target


def _load_yaml(target: Path) -> dict:
    mtime = target.stat().st_mtime
    key = (target.resolve(), mtime)
    if key in _YAML_CACHE:
        return _YAML_CACHE[key]
    with target.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    _YAML_CACHE[key] = data
    return data


def load_predicate_classes(
    path: str | Path | None = None,
) -> dict[str, frozenset[str]]:
    """Загрузить словарь predicate-классов из YAML.

    Возвращает обратный индекс ``lemma → frozenset(classes)``. Контракт
    одинаков для v0 (плоский) и v1 (иерархический): во frozenset для v1
    попадают все классы пути от leaf к root по всем ветвям.

    Если ``path is None``, читается встроенный
    ``configs/predicate_classes.yaml``.
    """
    target = _resolve_path(path)
    data = _load_yaml(target)
    version = int(data.get("version", 0))

    inverted: dict[str, set[str]] = defaultdict(set)

    if version == 0:
        # Legacy: classes: {class_name: [lemma, ...]}
        classes_section = data.get("classes") or {}
        for class_name, lemmas in classes_section.items():
            if not isinstance(lemmas, list):
                continue
            for lemma in lemmas:
                if isinstance(lemma, str) and lemma:
                    inverted[lemma].add(class_name)
    elif version == 1:
        # Hierarchical: lemmas: {lemma: [[leaf, ..., root], ...]}
        lemmas_section = data.get("lemmas") or {}
        for lemma, paths in lemmas_section.items():
            if not isinstance(lemma, str) or not lemma:
                continue
            if not isinstance(paths, list):
                continue
            for path_list in paths:
                if not isinstance(path_list, list):
                    continue
                for class_name in path_list:
                    if isinstance(class_name, str) and class_name:
                        inverted[lemma].add(class_name)
    else:
        raise ValueError(
            f"Unsupported predicate lexicon version: {version} (file: {target})"
        )

    return {lemma: frozenset(classes) for lemma, classes in inverted.items()}


def load_predicate_hierarchy(
    path: str | Path | None = None,
) -> PredicateHierarchy | None:
    """Загрузить иерархию predicate-классов (v1 only).

    Возвращает ``PredicateHierarchy`` для v1 формата и ``None`` для v0
    (legacy без иерархии). Использует тот же YAML-кэш, что и
    :func:`load_predicate_classes` — двойной парсинг исключён.
    """
    target = _resolve_path(path)
    data = _load_yaml(target)
    version = int(data.get("version", 0))

    if version != 1:
        return None

    hierarchy_section = data.get("hierarchy") or {}
    parent_of: dict[str, str | None] = {}
    level_of: dict[str, str] = {}
    anchor_of: dict[str, str] = {}
    label_of: dict[str, str] = {}

    for class_slug, meta in hierarchy_section.items():
        if not isinstance(class_slug, str) or not isinstance(meta, dict):
            continue
        parent_of[class_slug] = meta.get("parent")
        level_of[class_slug] = str(meta.get("level", "leaf"))
        anchor_of[class_slug] = str(meta.get("anchor_synset_id", ""))
        label_of[class_slug] = str(meta.get("label_ru", class_slug))

    lemmas_section = data.get("lemmas") or {}
    lemma_paths: dict[str, list[list[str]]] = {}
    for lemma, paths in lemmas_section.items():
        if not isinstance(lemma, str) or not isinstance(paths, list):
            continue
        normalized: list[list[str]] = []
        for path_list in paths:
            if not isinstance(path_list, list):
                continue
            cleaned = [c for c in path_list if isinstance(c, str) and c]
            if cleaned:
                normalized.append(cleaned)
        if normalized:
            lemma_paths[lemma] = normalized

    metadata = data.get("metadata") or {}

    return PredicateHierarchy(
        parent_of=parent_of,
        level_of=level_of,
        anchor_of=anchor_of,
        label_of=label_of,
        lemma_paths=lemma_paths,
        metadata=metadata,
    )
