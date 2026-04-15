# metagraph-nlp-ru

Построение метаграфового представления русскоязычного текста.
Исследовательский проект. Основной pipeline:

```
raw text → normalize → sentences → clauses → semantic graph → metagraph
```

Подробная спецификация архитектуры и инвариантов — в [CLAUDE.md](CLAUDE.md).
Дневник разработки — в [docs/journal/](docs/journal/).

## Установка (dev)

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS:    source .venv/bin/activate
pip install -e ".[dev]"
```

## Запуск thin slice

```bash
python -m metagraph_nlp process \
  --input data/samples/short.txt \
  --out artifacts/run1 \
  --config configs/default.yaml
```

В `artifacts/run1/` появятся `document.json`, `sentences.jsonl`,
`clauses.jsonl`, `semantic_graph.json`, `metagraph.json`, `audit.jsonl`,
`config.snapshot.yaml`.

## Тесты

```bash
pytest
```

## Статус

MVP-каркас. Ряд стадий — осознанные заглушки (clause=sentence, наивный
билдер графа); см. [docs/journal/2026-04-15-bootstrap.md](docs/journal/2026-04-15-bootstrap.md).
