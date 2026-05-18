"""Одноразовый прогрев natasha: скачать и инициализировать модели.

Запускается после установки зависимостей:

    .venv/Scripts/python.exe scripts/warmup_natasha.py

Скачивает веса Navec/Slovnet в ~/.natasha/ и прогоняет одно предложение,
чтобы убедиться что pipeline работает и кэш заполнен. После первого запуска
обычные вызовы парсера стартуют за ~1–2 секунды.
"""

from __future__ import annotations

import sys
import time

from metagraph_nlp.parsers.morphsyntax.natasha_adapter import get_natasha_parser


def main() -> int:
    print("[warmup] initialising natasha…", flush=True)
    t0 = time.perf_counter()
    parser = get_natasha_parser()
    t1 = time.perf_counter()
    print(f"[warmup] init done in {t1 - t0:.2f}s", flush=True)

    sample = "Студент читает книгу в библиотеке."
    print(f"[warmup] parsing: {sample!r}", flush=True)
    parsed = parser.parse(sample)
    for tok in parsed.tokens:
        print(
            f"  {tok.id_in_sent:>2} {tok.text:<12} "
            f"lemma={tok.lemma:<12} pos={tok.pos:<6} "
            f"head={tok.head} deprel={tok.deprel}",
            flush=True,
        )
    print("[warmup] ok", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
