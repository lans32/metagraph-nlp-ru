"""Тесты CLI batch subcommand (без реального natasha — мокаем run_from_file)."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

from metagraph_nlp.cli import main


def _mock_run_from_file(input_path: Path, out_dir: Path, **kwargs):
    out_dir.mkdir(parents=True, exist_ok=True)
    result = MagicMock()
    result.document.id = "doc_001"
    result.sentences = [MagicMock()] * 2
    result.clauses = [MagicMock()] * 3
    result.graph.nodes = [MagicMock()] * 5
    result.graph.edges = [MagicMock()] * 4
    result.metagraph.meta_nodes = [MagicMock()] * 3
    result.metagraph.meta_edges = [MagicMock()] * 2
    result.metrics = None
    return result


@patch("metagraph_nlp.cli.run_from_file", side_effect=_mock_run_from_file)
@patch("metagraph_nlp.cli.load_config", return_value=MagicMock())
@patch("metagraph_nlp.pipeline.get_default_parser", return_value=MagicMock())
def test_batch_processes_txt_files(mock_parser, mock_config, mock_run, tmp_path: Path):
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    (in_dir / "a.txt").write_text("Текст 1.", encoding="utf-8")
    (in_dir / "b.txt").write_text("Текст 2.", encoding="utf-8")
    (in_dir / "readme.md").write_text("ignore me", encoding="utf-8")

    out_dir = tmp_path / "output"
    ret = main(["batch", "--input-dir", str(in_dir), "--out-dir", str(out_dir)])

    assert ret == 0
    assert mock_run.call_count == 2


@patch("metagraph_nlp.cli.run_from_file", side_effect=_mock_run_from_file)
@patch("metagraph_nlp.cli.load_config", return_value=MagicMock())
@patch("metagraph_nlp.pipeline.get_default_parser", return_value=MagicMock())
def test_batch_writes_summary_json(mock_parser, mock_config, mock_run, tmp_path: Path):
    in_dir = tmp_path / "input"
    in_dir.mkdir()
    (in_dir / "a.txt").write_text("Текст.", encoding="utf-8")

    out_dir = tmp_path / "output"
    main(["batch", "--input-dir", str(in_dir), "--out-dir", str(out_dir)])

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    data = json.loads(summary_path.read_text(encoding="utf-8"))
    assert data["total_files"] == 1
    assert data["successful"] == 1
    assert data["failed"] == 0
    assert len(data["files"]) == 1
    assert data["files"][0]["sentences"] == 2


@patch("metagraph_nlp.cli.run_from_file", side_effect=_mock_run_from_file)
@patch("metagraph_nlp.cli.load_config", return_value=MagicMock())
@patch("metagraph_nlp.pipeline.get_default_parser", return_value=MagicMock())
def test_batch_empty_dir(mock_parser, mock_config, mock_run, tmp_path: Path):
    in_dir = tmp_path / "empty"
    in_dir.mkdir()
    out_dir = tmp_path / "output"

    ret = main(["batch", "--input-dir", str(in_dir), "--out-dir", str(out_dir)])

    assert ret == 0
    data = json.loads((out_dir / "summary.json").read_text(encoding="utf-8"))
    assert data["total_files"] == 0
    assert mock_run.call_count == 0
