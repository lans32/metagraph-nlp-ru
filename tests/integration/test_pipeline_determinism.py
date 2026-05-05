"""Тест: два прогона pipeline на одинаковом тексте дают идентичный результат (ПМИ §7.4.3)."""

from __future__ import annotations

import json

import pytest

from metagraph_nlp.config import Config
from metagraph_nlp.pipeline import run

pytestmark = pytest.mark.slow

SAMPLE = (
    "Студент читает книгу в библиотеке. "
    "Преподаватель объясняет студенту теорему на лекции. "
    "Исследователь анализирует данные эксперимента."
)


def _strip_timestamps(obj):
    """Рекурсивно убирает поле timestamp из словарей для сравнения."""
    if isinstance(obj, dict):
        return {k: _strip_timestamps(v) for k, v in obj.items() if k != "timestamp"}
    if isinstance(obj, list):
        return [_strip_timestamps(i) for i in obj]
    return obj


def test_pipeline_determinism():
    r1 = run(SAMPLE, config=Config())
    r2 = run(SAMPLE, config=Config())

    g1 = json.loads(r1.graph.model_dump_json())
    g2 = json.loads(r2.graph.model_dump_json())
    assert g1 == g2

    m1 = _strip_timestamps(json.loads(r1.metagraph.model_dump_json()))
    m2 = _strip_timestamps(json.loads(r2.metagraph.model_dump_json()))
    assert m1 == m2

    c1 = [(c.head_lemma, c.clause_type) for c in r1.clauses]
    c2 = [(c.head_lemma, c.clause_type) for c in r2.clauses]
    assert c1 == c2

    assert [s.span.text for s in r1.sentences] == [s.span.text for s in r2.sentences]
