"""Slow интеграционный тест: build_predicate_lexicon.py против RuWordNet.

Запуск: ``pytest -m slow tests/integration/test_build_predicate_lexicon.py``.

Тест запускается только при установленном ``ruwordnet`` пакете и
загруженной SQLite-базе. Проверяет:
- скрипт корректно собирает YAML v1 из 2 маленьких anchor'ов;
- иерархия читается обратно loader'ом;
- ожидаемые леммы попадают в правильные ветви.
"""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.slow

# Скрипт лежит в scripts/, а тесты — в tests/. Импорт через path-вставку.
REPO_ROOT = Path(__file__).resolve().parents[2]


def _import_builder():
    pytest.importorskip("ruwordnet")
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import build_predicate_lexicon as builder
    finally:
        sys.path.pop(0)
    return builder


def _build(tmp_path: Path) -> Path:
    """Вызвать build_predicate_lexicon с маленькими anchors для двух классов."""
    builder = _import_builder()

    anchors_path = tmp_path / "test_anchors.yaml"
    anchors_path.write_text(
        # Два узких якоря для скорости теста (motion и volition не пересекаются).
        yaml.safe_dump(
            {
                "version": 1,
                "anchors": {
                    "motion": {
                        "seed_lemma": "перемещаться",
                        "label_ru": "движение",
                        "resolved_synset_id": "106587-V",
                        "priority": 10,
                    },
                    "volition": {
                        "seed_lemma": "хотеть",
                        "label_ru": "желание",
                        "resolved_synset_id": "119928-V",
                        "priority": 20,
                    },
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "lex.yaml"
    lexicon, _prune = builder.build_lexicon(anchors_path, max_depth=2)

    with output_path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(lexicon, f, allow_unicode=True, sort_keys=False)

    return output_path


def test_build_creates_valid_v1_yaml(tmp_path):
    output = _build(tmp_path)
    data = yaml.safe_load(output.read_text(encoding="utf-8"))

    assert data["version"] == 1
    assert data["metadata"]["max_depth"] == 2
    assert "motion" in data["hierarchy"]
    assert "volition" in data["hierarchy"]
    assert data["hierarchy"]["motion"]["parent"] is None
    assert data["hierarchy"]["motion"]["level"] == "root"
    assert data["hierarchy"]["motion"]["anchor_synset_id"] == "106587-V"

    # leaf-классы существуют (по крайней мере один из 2 деревьев должен дать).
    leaves = [
        slug for slug, meta in data["hierarchy"].items()
        if meta["level"] == "leaf"
    ]
    assert leaves, "ожидаются leaf-классы при max_depth=2"

    # Леммы из motion должны попасть в motion-ветку.
    # `идти` — широко представленный motion-глагол в RuWordNet.
    lemmas = data["lemmas"]
    if "идти" in lemmas:
        any_motion = any("motion" in path for path in lemmas["идти"])
        assert any_motion, "идти должно попадать в motion-ветку"


def test_built_yaml_loads_back_via_loader(tmp_path):
    """Артефакт корректно читается through load_predicate_hierarchy."""
    output = _build(tmp_path)
    from metagraph_nlp.parsers.predicate_lexicon import (
        load_predicate_classes,
        load_predicate_hierarchy,
    )

    lex = load_predicate_classes(output)
    h = load_predicate_hierarchy(output)

    assert h is not None
    assert "motion" in h.parent_of
    assert h.parent_of["motion"] is None
    assert h.level_of["motion"] == "root"

    # У как минимум одной леммы из motion-ветви есть путь, содержащий "motion".
    motion_lemmas = [
        lemma for lemma, classes in lex.items() if "motion" in classes
    ]
    assert motion_lemmas, "ожидается хотя бы одна лемма с классом motion"


def test_build_lexicon_prunes_overlap(tmp_path):
    """Synset 106915-V (perception root) не должен оказаться в emotion-ветке.

    emotion (`116921-V` "пережить") содержит synset `106915-V`
    "ощущать, воспринимать" как потомка — это же корень `perception`.
    При priority=10 для perception и priority=110 для emotion якорь
    perception обрабатывается первым и резервирует 106915-V; BFS emotion
    должен пропустить его вместе с поддеревом.
    """
    builder = _import_builder()

    anchors_path = tmp_path / "overlap_anchors.yaml"
    anchors_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "anchors": {
                    "perception": {
                        "seed_lemma": "воспринимать",
                        "label_ru": "ощущать, воспринимать",
                        "resolved_synset_id": "106915-V",
                        "priority": 10,
                    },
                    "emotion": {
                        "seed_lemma": "переживать",
                        "label_ru": "пережить, испытать (эмоцию, состояние)",
                        "resolved_synset_id": "116921-V",
                        "priority": 110,
                    },
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    lexicon, prune_report = builder.build_lexicon(anchors_path, max_depth=2)

    # Оба root-класса присутствуют, perception идёт первым в priority_order.
    assert "perception" in lexicon["hierarchy"]
    assert "emotion" in lexicon["hierarchy"]
    assert lexicon["hierarchy"]["perception"]["level"] == "root"
    assert lexicon["hierarchy"]["emotion"]["level"] == "root"
    assert lexicon["metadata"]["priority_order"][0] == "perception"
    assert lexicon["metadata"]["priority_order"][-1] == "emotion"

    # Ни один slug в emotion_*-поддереве не должен ссылаться на synset
    # 106915-V (он принадлежит perception).
    overlapping_synset = "106915-V"
    emotion_slugs_with_overlap = [
        slug
        for slug, meta in lexicon["hierarchy"].items()
        if slug.startswith("emotion") and meta["anchor_synset_id"] == overlapping_synset
    ]
    assert not emotion_slugs_with_overlap, (
        f"synset {overlapping_synset} не должен быть в emotion-ветке, "
        f"найдено: {emotion_slugs_with_overlap}"
    )

    # Лемма "воспринимать" — только в путях, оканчивающихся на perception
    # (root-класс пути идёт ПОСЛЕДНИМ в path-listе по контракту v1:
    # paths = [leaf, ..., root]).
    paths_for_lemma = lexicon["lemmas"].get("воспринимать", [])
    assert paths_for_lemma, "лемма 'воспринимать' должна быть в perception-дереве"
    roots = {path[-1] for path in paths_for_lemma}
    assert roots == {"perception"}, (
        f"'воспринимать' попала в неожиданные корни: {roots}"
    )

    # Prune-отчёт зафиксировал отрезанные synset-ы у emotion.
    assert lexicon["metadata"]["pruned_synset_count"] >= 1
    emotion_prunes = [r for r in prune_report if r["anchor"] == "emotion"]
    assert emotion_prunes, "ожидается prune-запись для emotion"
    pruned_ids = {p["synset_id"] for p in emotion_prunes[0]["pruned"]}
    assert overlapping_synset in pruned_ids


def test_build_lexicon_orders_by_priority(tmp_path):
    """priority определяет порядок обхода, а не алфавит ключей."""
    builder = _import_builder()

    anchors_path = tmp_path / "prio_anchors.yaml"
    anchors_path.write_text(
        yaml.safe_dump(
            {
                "version": 1,
                "anchors": {
                    "motion": {
                        "seed_lemma": "перемещаться",
                        "label_ru": "движение",
                        "resolved_synset_id": "106587-V",
                        "priority": 30,
                    },
                    "volition": {
                        "seed_lemma": "хотеть",
                        "label_ru": "желание",
                        "resolved_synset_id": "119928-V",
                        "priority": 10,
                    },
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    lexicon, _ = builder.build_lexicon(anchors_path, max_depth=1)
    assert lexicon["metadata"]["priority_order"] == ["volition", "motion"]
