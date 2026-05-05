"""CLI: `python -m metagraph_nlp process --input ... --out ...`."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from metagraph_nlp.config import load_config
from metagraph_nlp.pipeline import run_from_file


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="metagraph-nlp",
        description="Построение метаграфового представления русскоязычного текста.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("process", help="Прогнать один текстовый файл через pipeline.")
    p.add_argument("--input", required=True, type=Path, help="Путь к входному .txt")
    p.add_argument("--out", required=True, type=Path, help="Каталог для артефактов")
    p.add_argument("--config", type=Path, default=None, help="YAML-конфиг (опционально)")
    p.add_argument(
        "--viz",
        action="store_true",
        help="Дополнительно сохранить HTML и DOT визуализации графа и метаграфа.",
    )

    b = sub.add_parser("batch", help="Прогнать все .txt файлы из каталога через pipeline.")
    b.add_argument("--input-dir", required=True, type=Path, help="Каталог с входными .txt файлами")
    b.add_argument("--out-dir", required=True, type=Path, help="Каталог для артефактов")
    b.add_argument("--config", type=Path, default=None, help="YAML-конфиг (опционально)")
    b.add_argument("--viz", action="store_true", help="Сохранить визуализации для каждого файла.")

    return parser


def _run_process(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    result = run_from_file(args.input, args.out, config=config, viz=args.viz)
    print(f"[ok] document={result.document.id}")
    print(f"[ok] sentences={len(result.sentences)}")
    print(f"[ok] clauses={len(result.clauses)}")
    print(f"[ok] nodes={len(result.graph.nodes)} edges={len(result.graph.edges)}")
    print(f"[ok] meta_nodes={len(result.metagraph.meta_nodes)}")
    print(f"[ok] artifacts -> {args.out}")
    if args.viz:
        print(f"[ok] viz -> {args.out}")
    return 0


def _run_batch(args: argparse.Namespace) -> int:
    from metagraph_nlp.pipeline import get_default_parser

    config = load_config(args.config)
    parser = get_default_parser()
    txt_files = sorted(args.input_dir.glob("*.txt"))

    args.out_dir.mkdir(parents=True, exist_ok=True)
    file_stats: list[dict] = []
    success_count = 0

    for f in txt_files:
        sub_out = args.out_dir / f.stem
        entry: dict = {"filename": f.name, "status": "ok", "error": None}
        try:
            result = run_from_file(f, sub_out, config=config, viz=args.viz, parser=parser)
            entry["sentences"] = len(result.sentences)
            entry["clauses"] = len(result.clauses)
            entry["nodes"] = len(result.graph.nodes)
            entry["edges"] = len(result.graph.edges)
            entry["meta_nodes"] = len(result.metagraph.meta_nodes)
            entry["meta_edges"] = len(result.metagraph.meta_edges)
            success_count += 1
            print(f"[ok] {f.name}: {entry['sentences']} sent, {entry['clauses']} clauses")
        except Exception as exc:
            entry["status"] = "error"
            entry["error"] = str(exc)
            print(f"[err] {f.name}: {exc}")
        file_stats.append(entry)

    summary = {
        "total_files": len(txt_files),
        "successful": success_count,
        "failed": len(txt_files) - success_count,
        "files": file_stats,
    }
    summary_path = args.out_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(f"[ok] batch: {success_count}/{len(txt_files)} files -> {args.out_dir}")
    print(f"[ok] summary -> {summary_path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "process":
        return _run_process(args)

    if args.command == "batch":
        return _run_batch(args)

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
