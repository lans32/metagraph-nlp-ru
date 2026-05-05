"""End-to-end pipeline с реальным natasha-парсером.

Эти тесты нагружают полный UD-стек, поэтому помечены `slow` и не
запускаются в быстром прогоне. Запустить: `pytest -m slow`.
"""

from pathlib import Path

import pytest

from metagraph_nlp.config import Config
from metagraph_nlp.pipeline import run, run_from_file

pytestmark = pytest.mark.slow

SAMPLE = (
    "Студент читает книгу в библиотеке. "
    "Преподаватель объясняет студенту теорему на лекции. "
    "Исследователь анализирует данные эксперимента."
)


def test_thin_slice_in_memory():
    result = run(SAMPLE, config=Config())

    assert result.document.normalized_text
    assert len(result.sentences) == 3
    assert len(result.clauses) >= 3
    assert {c.sentence_id for c in result.clauses} == {s.id for s in result.sentences}

    assert len(result.graph.nodes) >= 3
    assert len(result.graph.edges) >= 1

    # Лемматизация: в графе должна быть начальная форма "студент", не "Студент".
    labels = {n.label for n in result.graph.nodes}
    assert "студент" in labels

    l1_nodes = [mn for mn in result.metagraph.meta_nodes if mn.level == 1]
    l2_nodes = [mn for mn in result.metagraph.meta_nodes if mn.level == 2]
    assert len(l1_nodes) == len(result.clauses)
    for mn in l1_nodes:
        assert mn.type == "clause"

    # SAMPLE — одно-параграфный: ровно одна L2-метавершина типа "paragraph".
    assert len(l2_nodes) == 1
    assert l2_nodes[0].type == "paragraph"
    assert l2_nodes[0].label == "paragraph_0"
    assert set(l2_nodes[0].fragment.meta_node_ids) == {mn.id for mn in l1_nodes}

    rule_names = {e.rule for e in result.audit.events}
    assert "normalize_text" in rule_names
    assert "razdel.sentenize" in rule_names
    assert "ud_subtree_clauses_v0" in rule_names
    assert "ud_roles_v0" in rule_names
    assert "paragraph_clauses_v0" in rule_names
    assert "topic_overlap_v0" in rule_names

    # «студент» встречается в первой и второй клаузах — должно быть
    # хотя бы одно метаребро shared_entity.
    shared = [me for me in result.metagraph.meta_edges if me.type == "shared_entity"]
    assert shared, "ожидается хотя бы одно метаребро shared_entity"

    # На параграфной стратегии L2-метавершины не пересекаются по L1-клаузам,
    # поэтому topic_overlap рёбер быть не должно.
    topic = [me for me in result.metagraph.meta_edges if me.type == "topic_overlap"]
    assert topic == []


def test_thin_slice_writes_artifacts(tmp_path: Path):
    input_path = tmp_path / "input.txt"
    input_path.write_text(SAMPLE, encoding="utf-8")
    out_dir = tmp_path / "run1"

    run_from_file(input_path, out_dir, config=Config())

    expected = [
        "document.json",
        "sentences.jsonl",
        "clauses.jsonl",
        "semantic_graph.json",
        "metagraph.json",
        "audit.jsonl",
        "config.snapshot.yaml",
    ]
    for name in expected:
        p = out_dir / name
        assert p.exists(), f"missing artifact: {name}"
        assert p.stat().st_size > 0


def test_thin_slice_writes_viz_artifacts(tmp_path: Path):
    input_path = tmp_path / "input.txt"
    input_path.write_text(SAMPLE, encoding="utf-8")
    out_dir = tmp_path / "run_viz"

    run_from_file(input_path, out_dir, config=Config(), viz=True)

    viz_files = [
        "semantic_graph.html",
        "metagraph.html",
        "semantic_graph.dot",
        "metagraph.dot",
    ]
    for name in viz_files:
        p = out_dir / name
        assert p.exists(), f"missing viz artifact: {name}"
        assert p.stat().st_size > 0
