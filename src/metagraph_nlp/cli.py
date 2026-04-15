"""CLI: `python -m metagraph_nlp process --input ... --out ...`."""

from __future__ import annotations

import argparse
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

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "process":
        config = load_config(args.config)
        result = run_from_file(args.input, args.out, config=config)
        print(f"[ok] document={result.document.id}")
        print(f"[ok] sentences={len(result.sentences)}")
        print(f"[ok] clauses={len(result.clauses)}")
        print(f"[ok] nodes={len(result.graph.nodes)} edges={len(result.graph.edges)}")
        print(f"[ok] meta_nodes={len(result.metagraph.meta_nodes)}")
        print(f"[ok] artifacts -> {args.out}")
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
