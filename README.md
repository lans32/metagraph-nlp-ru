# metagraph-nlp-ru

Построение метаграфового представления русскоязычного текста.
Исследовательский проект для дипломной работы. Основной pipeline:

```
raw text → normalize → sentences → parse (UD) → clauses → semantic graph
  → [anaphora resolution] → [NP collapse]
  → metagraph L1 (clause metanodes + shared_entity metaedges)
  → [coref clusters] → metagraph L2 (paragraphs + topic_overlap)
```

Стадии в квадратных скобках — опциональные, управляются конфигурацией.

Подробная спецификация архитектуры и инвариантов — в [CLAUDE.md](CLAUDE.md).
Дневник разработки — в [docs/journal/](docs/journal/).

## Возможности

- **Типизация клауз**: main, coord, compl, xcompl, adverbial, relative, participial — по UD-deprel предиката.
- **Многоуровневый метаграф**: L0 (узлы/рёбра) → L1 (клаузы, shared_entity) → L2 (параграфы, coref clusters, topic_overlap).
- **NP collapse**: свёртка именных групп (NOUN + amod/nmod/det) в один узел с составной леммой.
- **Разрешение анафоры**: rule-based замена личных местоимений 3-го лица (он/она/оно/они) на ближайший антецедент с согласованием по Gender / Number / Animacy.
- **Три формата визуализации**: pyvis HTML, GraphViz DOT, Cytoscape.js с compound nodes и инспектором.
- **Пакетная обработка**: CLI-команда `batch` для каталога `.txt` файлов.
- **Профилирование**: wall time + peak memory на каждую стадию pipeline.
- **Streamlit веб-интерфейс**: ввод текста, просмотр графа/метаграфа, экспорт JSON.
- **Два UD-парсера**: natasha (по умолчанию) и MaltParser (через subprocess).
- **Audit trail**: каждый элемент имеет provenance с правилом, стадией и UTC-timestamp.

## Установка

```bash
python -m venv .venv
# Windows:  .venv\Scripts\activate
# macOS:    source .venv/bin/activate
pip install -e ".[dev]"
```

Для веб-интерфейса:

```bash
pip install -e ".[web]"
```

## Запуск

### Обработка одного файла

```bash
python -m metagraph_nlp process \
  --input data/samples/short.txt \
  --out artifacts/run1 \
  --config configs/default.yaml \
  --viz
```

В `artifacts/run1/` появятся: `document.json`, `sentences.jsonl`,
`clauses.jsonl`, `semantic_graph.json`, `metagraph.json`, `audit.jsonl`,
`config.snapshot.yaml`, `profiling.json`, а при `--viz` — ещё
`semantic_graph.html`, `metagraph.html`, `metagraph_cytoscape.html`,
`semantic_graph.dot`, `metagraph.dot`.

### Пакетная обработка

```bash
python -m metagraph_nlp batch \
  --input-dir data/samples/ \
  --out-dir artifacts/batch_run \
  --config configs/default.yaml \
  --viz
```

### Веб-интерфейс

```bash
streamlit run src/metagraph_nlp/web/app.py
```

## Тесты

```bash
pytest                    # 78 fast unit-тестов
pytest -m slow            # 6 integration-тестов (требуют natasha)
pytest -m 'slow or not slow'  # все 84 теста
```

## Структура модулей

| Модуль | Назначение |
|--------|-----------|
| `domain/` | Доменная модель: Document, Sentence, Clause, Node, Edge, MetaNode, MetaEdge, SemanticGraph, Metagraph |
| `parsers/` | Нормализация, сегментация, клаузы, адаптеры morphsyntax (natasha, MaltParser) |
| `graph_builders/` | Семантический граф из UD-ролей, NP collapse |
| `aggregators/` | L1: clause metanodes, shared_entity metaedges; L2: paragraph metanodes, coref clusters, topic_overlap metaedges |
| `viz/` | pyvis HTML, GraphViz DOT, Cytoscape.js с compound nodes |
| `io/` | Сериализация артефактов (JSON, JSONL, YAML) |
| `provenance/` | AuditLog с timestamps |
| `profiling/` | Таймеры и memory tracking |
| `web/` | Streamlit-приложение |

## Конфигурация

Все параметры управляются через YAML-конфиг (см. `configs/default.yaml`).
Ключевые опции агрегации:

| Параметр | Default | Описание |
|----------|---------|----------|
| `aggregation.shared_entity_enabled` | `true` | L1-метарёбра по общим леммам |
| `aggregation.paragraph_enabled` | `true` | L2-метавершины по параграфам |
| `aggregation.topic_overlap_enabled` | `true` | L2-метарёбра по пересечению фрагментов |
| `aggregation.coref_cluster_enabled` | `false` | L2-метавершины по кластерам shared_entity |
| `aggregation.np_collapse_enabled` | `false` | Свёртка именных групп перед агрегацией |
| `anaphora.enabled` | `false` | Разрешение анафоры (личные местоимения 3-го лица) |
| `anaphora.search_window_sentences` | `2` | Окно поиска антецедента в предложениях |
| `anaphora.require_animacy_match` | `true` | Требовать совпадения Animacy PRON и антецедента |
| `morphsyntax.parser` | `"natasha"` | UD-парсер: `natasha` или `maltparser` |

## Дневник разработки

- [2026-04-15 — Bootstrap](docs/journal/2026-04-15-bootstrap.md)
- [2026-04-15 — Визуализация](docs/journal/2026-04-15-viz.md)
- [2026-04-16 — UD-разбор, клаузы, метарёбра](docs/journal/2026-04-16-ud-and-metaedges.md)
- [2026-04-19 — L2-параграфная агрегация](docs/journal/2026-04-19-l2-paragraph-aggregation.md)
- [2026-05-05 — Девять задач для защиты](docs/journal/2026-05-05-diploma-features.md)
- [2026-05-06 — Разрешение анафоры (v0)](docs/journal/2026-05-06-anaphora-resolution-v0.md)
