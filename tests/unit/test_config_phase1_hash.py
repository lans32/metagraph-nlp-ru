"""Юнит-тесты Config.phase1_hash — учёт содержимого predicate-словаря."""

from __future__ import annotations

import textwrap

import pytest

from metagraph_nlp.config import Config


def _config_with_lexicon(path) -> Config:
    cfg = Config()
    cfg.aggregation.predicate_classes_path = str(path)
    return cfg


def test_phase1_hash_changes_when_lexicon_content_changes(tmp_path):
    """Смена содержимого YAML без смены пути → разный phase1_hash."""
    lex_path = tmp_path / "lex.yaml"
    lex_path.write_text(
        "version: 0\nclasses:\n  motion: [идти]\n",
        encoding="utf-8",
    )

    cfg = _config_with_lexicon(lex_path)
    hash_before = cfg.phase1_hash()

    # Перезаписать YAML с другим содержимым
    lex_path.write_text(
        "version: 0\nclasses:\n  motion: [идти, ехать]\n",
        encoding="utf-8",
    )

    cfg2 = _config_with_lexicon(lex_path)
    hash_after = cfg2.phase1_hash()

    assert hash_before != hash_after


def test_phase1_hash_stable_for_same_content(tmp_path):
    """Один и тот же файл → одинаковый hash при повторных вычислениях."""
    lex_path = tmp_path / "lex.yaml"
    lex_path.write_text(
        "version: 0\nclasses:\n  motion: [идти]\n",
        encoding="utf-8",
    )

    cfg = _config_with_lexicon(lex_path)
    assert cfg.phase1_hash() == cfg.phase1_hash()


def test_phase1_hash_distinguishes_v0_and_v1(tmp_path):
    """v0 и v1 для одних и тех же лемм дают разный hash (разная структура)."""
    v0 = tmp_path / "v0.yaml"
    v0.write_text(
        "version: 0\nclasses:\n  motion: [идти]\n",
        encoding="utf-8",
    )
    v1 = tmp_path / "v1.yaml"
    v1.write_text(
        textwrap.dedent(
            """
            version: 1
            hierarchy:
              motion:
                parent: null
                level: root
                anchor_synset_id: "X"
                label_ru: M
            lemmas:
              идти:
                - [motion]
            """
        ).strip(),
        encoding="utf-8",
    )

    cfg_v0 = _config_with_lexicon(v0)
    cfg_v1 = _config_with_lexicon(v1)
    assert cfg_v0.phase1_hash() != cfg_v1.phase1_hash()


def test_phase1_hash_includes_default_lexicon_when_path_none():
    """С None должен браться встроенный configs/predicate_classes.yaml.

    Это даёт стабильный hash и инвалидирует Phase 1 кэш при обновлении
    словаря в репозитории.
    """
    cfg = Config()
    cfg.aggregation.predicate_classes_path = None
    # Главное — не падает и возвращает не пустую строку
    h = cfg.phase1_hash()
    assert isinstance(h, str)
    assert len(h) == 16
