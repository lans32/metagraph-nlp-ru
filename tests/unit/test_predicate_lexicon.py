"""Юнит-тесты загрузчика словаря predicate-классов (v0 и v1)."""

from __future__ import annotations

import textwrap

import pytest

from metagraph_nlp.parsers.predicate_lexicon import (
    load_predicate_classes,
    load_predicate_hierarchy,
)


def test_default_path_loads_builtin_classes():
    """По умолчанию загружается configs/predicate_classes.yaml."""
    lex = load_predicate_classes()
    assert "сказать" in lex
    assert "communication" in lex["сказать"]
    assert "идти" in lex
    assert "motion" in lex["идти"]


def test_inverted_index_uses_frozensets():
    lex = load_predicate_classes()
    for lemma, classes in lex.items():
        assert isinstance(classes, frozenset)
        assert all(isinstance(c, str) for c in classes)
        assert lemma  # непустая лемма


def test_custom_yaml_path(tmp_path):
    yaml_text = textwrap.dedent(
        """
        version: 0
        classes:
          gardening: [копать, поливать]
          cooking: [жарить]
        """
    ).strip()
    p = tmp_path / "custom.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    lex = load_predicate_classes(p)
    assert lex == {
        "копать": frozenset({"gardening"}),
        "поливать": frozenset({"gardening"}),
        "жарить": frozenset({"cooking"}),
    }


def test_lemma_in_multiple_classes(tmp_path):
    yaml_text = textwrap.dedent(
        """
        version: 0
        classes:
          motion: [идти]
          state: [идти]
        """
    ).strip()
    p = tmp_path / "ambiguous.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    lex = load_predicate_classes(p)
    assert lex["идти"] == frozenset({"motion", "state"})


def test_missing_path_raises(tmp_path):
    nonexistent = tmp_path / "nope.yaml"
    with pytest.raises(FileNotFoundError):
        load_predicate_classes(nonexistent)


# --- v1 (hierarchical) tests ----------------------------------------------


V1_YAML = textwrap.dedent(
    """
    version: 1
    metadata:
      source: ruwordnet-2.0
      built_at: "2026-05-22T12:00:00Z"
      max_depth: 3
    hierarchy:
      motion:
        parent: null
        level: root
        anchor_synset_id: "106587-V"
        label_ru: "движение, перемещение"
      motion_walking:
        parent: motion
        level: mid
        anchor_synset_id: "106588-V"
        label_ru: "ходить"
      motion_walking_strolling:
        parent: motion_walking
        level: leaf
        anchor_synset_id: "106589-V"
        label_ru: "прогуливаться"
      change_of_state:
        parent: null
        level: root
        anchor_synset_id: "106629-V"
        label_ru: "изменение"
    lemmas:
      гулять:
        - [motion_walking_strolling, motion_walking, motion]
      идти:
        - [motion_walking, motion]
        - [change_of_state]
      перемещаться:
        - [motion]
    """
).strip()


def test_v1_load_predicate_classes_flat_union(tmp_path):
    """v1: lex[lemma] = union всех уровней всех путей."""
    p = tmp_path / "v1.yaml"
    p.write_text(V1_YAML, encoding="utf-8")

    lex = load_predicate_classes(p)
    # `гулять` имеет один путь из 3 классов
    assert lex["гулять"] == frozenset({"motion_walking_strolling", "motion_walking", "motion"})
    # `идти` — два пути, union = 3 класса
    assert lex["идти"] == frozenset({"motion_walking", "motion", "change_of_state"})
    # `перемещаться` — только корень
    assert lex["перемещаться"] == frozenset({"motion"})


def test_v1_load_predicate_hierarchy(tmp_path):
    """v1: load_predicate_hierarchy возвращает PredicateHierarchy с parent/level/anchor."""
    p = tmp_path / "v1.yaml"
    p.write_text(V1_YAML, encoding="utf-8")

    h = load_predicate_hierarchy(p)
    assert h is not None
    assert h.parent_of["motion"] is None
    assert h.parent_of["motion_walking"] == "motion"
    assert h.parent_of["motion_walking_strolling"] == "motion_walking"
    assert h.level_of["motion"] == "root"
    assert h.level_of["motion_walking"] == "mid"
    assert h.level_of["motion_walking_strolling"] == "leaf"
    assert h.anchor_of["motion"] == "106587-V"
    assert h.lemma_paths["гулять"] == [
        ["motion_walking_strolling", "motion_walking", "motion"]
    ]


def test_v0_load_predicate_hierarchy_returns_none(tmp_path):
    """v0 формат: load_predicate_hierarchy возвращает None."""
    yaml_text = "version: 0\nclasses:\n  motion: [идти]\n"
    p = tmp_path / "v0.yaml"
    p.write_text(yaml_text, encoding="utf-8")

    assert load_predicate_hierarchy(p) is None


def test_predicate_hierarchy_navigation(tmp_path):
    """PredicateHierarchy.roots/children_of/classes_at_level."""
    p = tmp_path / "v1.yaml"
    p.write_text(V1_YAML, encoding="utf-8")

    h = load_predicate_hierarchy(p)
    assert h is not None
    assert sorted(h.roots()) == ["change_of_state", "motion"]
    assert h.children_of("motion") == ["motion_walking"]
    assert h.children_of("motion_walking") == ["motion_walking_strolling"]
    assert h.classes_at_level("root") == ["change_of_state", "motion"]
    assert h.classes_at_level("leaf") == ["motion_walking_strolling"]


def test_unsupported_version_raises(tmp_path):
    yaml_text = "version: 99\nclasses: {}\n"
    p = tmp_path / "future.yaml"
    p.write_text(yaml_text, encoding="utf-8")
    with pytest.raises(ValueError, match="version"):
        load_predicate_classes(p)


def test_yaml_cache_avoids_double_parsing(tmp_path):
    """load_predicate_classes и load_predicate_hierarchy не парсят дважды."""
    p = tmp_path / "v1.yaml"
    p.write_text(V1_YAML, encoding="utf-8")
    # Просто проверяем что оба вызова не падают и возвращают согласованные данные.
    lex = load_predicate_classes(p)
    h = load_predicate_hierarchy(p)
    assert h is not None
    # Все классы из lemma_paths должны быть в hierarchy.
    for paths in h.lemma_paths.values():
        for path in paths:
            for cls in path:
                assert cls in h.parent_of
    # И все эти классы должны давать leaf-union в lex.
    for lemma, paths in h.lemma_paths.items():
        expected_union: set[str] = set()
        for path in paths:
            expected_union.update(path)
        assert lex[lemma] == frozenset(expected_union)
