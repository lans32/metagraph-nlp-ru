"""Юнит-тесты загрузчика словаря predicate-классов."""

from __future__ import annotations

import textwrap

import pytest

from metagraph_nlp.parsers.predicate_lexicon import load_predicate_classes


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
