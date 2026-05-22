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


def _build(tmp_path: Path) -> Path:
    """Вызвать build_predicate_lexicon с маленькими anchors для двух классов."""
    pytest.importorskip("ruwordnet")
    import sys

    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    try:
        import build_predicate_lexicon as builder
    finally:
        sys.path.pop(0)

    anchors_path = tmp_path / "test_anchors.yaml"
    anchors_path.write_text(
        # Два узких якоря для скорости теста.
        yaml.safe_dump(
            {
                "version": 0,
                "anchors": {
                    "motion": {
                        "seed_lemma": "перемещаться",
                        "label_ru": "движение",
                        "resolved_synset_id": "106587-V",
                    },
                    "volition": {
                        "seed_lemma": "хотеть",
                        "label_ru": "желание",
                        "resolved_synset_id": "119928-V",
                    },
                },
            },
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    output_path = tmp_path / "lex.yaml"
    lexicon = builder.build_lexicon(anchors_path, max_depth=2)

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
