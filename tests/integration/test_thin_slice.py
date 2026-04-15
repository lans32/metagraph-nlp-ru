from pathlib import Path

from metagraph_nlp.config import Config
from metagraph_nlp.pipeline import run, run_from_file

SAMPLE = (
    "Студент читает книгу в библиотеке. "
    "Преподаватель объясняет теорему на лекции. "
    "Исследователь анализирует данные эксперимента."
)


def test_thin_slice_in_memory():
    result = run(SAMPLE, config=Config())

    assert result.document.normalized_text
    assert len(result.sentences) == 3
    assert len(result.clauses) == len(result.sentences)
    assert {c.sentence_id for c in result.clauses} == {s.id for s in result.sentences}

    assert len(result.graph.nodes) >= 2
    assert len(result.graph.edges) >= 1

    assert len(result.metagraph.meta_nodes) == len(result.clauses)
    for mn in result.metagraph.meta_nodes:
        assert mn.type == "clause"
        assert mn.level == 1
        assert mn.provenance.clause_id is not None

    rule_names = {e.rule for e in result.audit.events}
    assert {"normalize_text", "razdel.sentenize", "sentence_as_clause_v0"} <= rule_names


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
