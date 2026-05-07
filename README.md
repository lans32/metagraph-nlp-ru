# metagraph-nlp-ru

Построение метаграфового представления русскоязычного текста.
Исследовательский проект для дипломной работы. Основной pipeline:

```
raw text → normalize → sentences → parse (UD) → clauses → semantic graph
  → [anaphora resolution] → [NP collapse]
  → metagraph L1 (clause metanodes + shared_entity metaedges)
  → metagraph L2 (paragraphs + entity_clusters + predicate_class_clusters
                  + topic_overlap)
```

Стадии в квадратных скобках — опциональные, управляются конфигурацией.

Подробная спецификация архитектуры и инвариантов — в [CLAUDE.md](CLAUDE.md).
Дневник разработки — в [docs/journal/](docs/journal/).

## Возможности

- **Типизация клауз**: main, coord, compl, xcompl, adverbial, relative, participial — по UD-deprel предиката.
- **Многоуровневый метаграф (холархия)**: L0 (узлы/рёбра) → L1 (клаузы, shared_entity) → L2 (параграфы, entity-кластеры по общим сущностям, predicate-class-кластеры по таксономии глаголов, topic_overlap). Одна L1-клауза может одновременно входить в несколько L2-метавершин (например, в свой параграф и в тематический кластер).
- **Содержательные labels у L2-метавершин**: самая частая значимая лемма во фрагменте (для paragraph и entity_cluster) либо имя семантического класса (для predicate_class_cluster).
- **Минимальная таксономия глаголов**: ручной YAML-словарь `configs/predicate_classes.yaml` (motion, communication, cognition, perception, possession, creation, change_of_state, causation), поле `Edge.predicate_class`, агрегатор `predicate_class_cluster_v0`.
- **NP collapse**: свёртка именных групп (NOUN + amod/nmod/det) в один узел с составной леммой.
- **Разрешение анафоры (v1)**: rule-based замена-в-узле для личных, притяжательных 3-го лица и возвратных местоимений (он/она/оно/они, его/её/их с Poss=Yes, себя/свой). PRON-узел остаётся в графе с обновлёнными `lemma`/`upos`/`label` от антецедента; исходные значения сохраняются в `original_lemma`/`original_upos`, добавляется `antecedent_node_id`. Кандидаты ранжируются упрощённым Lappin–Leass salience-скорингом (subj / obj / obl / propn / recency / thematic / repeat_mention). Hard constraints — Gender / Number / Animacy. Возвратные берут subject текущей клаузы.
- **Три формата визуализации**: pyvis HTML, GraphViz DOT, Cytoscape.js с compound nodes и инспектором.
- **Пакетная обработка**: CLI-команда `batch` для каталога `.txt` файлов.
- **Профилирование**: wall time + peak memory на каждую стадию pipeline.
- **Streamlit веб-интерфейс**: ввод текста, структурированная панель конфигурации (морфосинтаксис, клаузы, граф, анафора, агрегация), просмотр графа/метаграфа, экспорт JSON.
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
`config.snapshot.yaml`, `profiling.json`. При `anaphora.enabled: true`
дополнительно появляется `anaphora_resolutions.jsonl` с записями о
заменах местоимений на антецеденты. При `--viz` — ещё
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

Удобный перезапуск с очисткой кэша (cross-platform, без `psutil`):

```bash
.venv/Scripts/python.exe scripts/restart_streamlit.py
```

## Тесты

```bash
pytest                          # быстрые тесты (118: 116 unit + 2 integration)
pytest -m slow                  # slow-тесты (6: 2 unit + 4 integration; требуют natasha/pymorphy3)
pytest -m 'slow or not slow'    # все тесты
```

## Структура модулей

| Модуль | Назначение |
|--------|-----------|
| `domain/` | Доменная модель: Document, Sentence, Clause, Node, Edge, MetaNode, MetaEdge, SemanticGraph, Metagraph |
| `parsers/` | Нормализация, сегментация, клаузы, адаптеры morphsyntax (natasha, MaltParser) |
| `graph_builders/` | Семантический граф из UD-ролей, NP collapse |
| `aggregators/` | L1: clause metanodes, shared_entity metaedges; L2: paragraph metanodes, entity_cluster metanodes, predicate_class_cluster metanodes, topic_overlap metaedges |
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
| `aggregation.entity_cluster_enabled` | `true` | L2-метавершины — тематические кластеры по графу shared_entity (правило `entity_cluster_v0`) |
| `aggregation.entity_cluster_min_size` | `2` | Минимальный размер тематического кластера |
| `aggregation.predicate_class_cluster_enabled` | `true` | L2-метавершины — кластеры клауз по классам предикатов из словаря (`predicate_class_cluster_v0`) |
| `aggregation.predicate_class_cluster_min_size` | `2` | Минимальный размер predicate-класса |
| `aggregation.predicate_classes_path` | `null` | Путь к YAML-словарю классов; `null` → встроенный `configs/predicate_classes.yaml` |
| `aggregation.topic_overlap_enabled` | `true` | L2-метарёбра по пересечению фрагментов |
| `aggregation.np_collapse_enabled` | `false` | Свёртка именных групп перед агрегацией |
| `anaphora.enabled` | `false` | Разрешение анафоры (`anaphora_resolution_v1`, замена-в-узле) |
| `anaphora.search_window_sentences` | `2` | Окно поиска антецедента в предложениях |
| `anaphora.require_animacy_match` | `true` | Требовать совпадения Animacy PRON и антецедента |
| `anaphora.pronoun_types` | `["personal_3p", "possessive_3p", "reflexive"]` | Покрываемые типы местоимений |
| `anaphora.salience_weights` | см. `SalienceWeights` | Веса упрощённого Lappin–Leass-скоринга кандидатов |
| `morphsyntax.parser` | `"natasha"` | UD-парсер: `natasha` или `maltparser` |

## Дневник разработки

- [2026-04-15 — Bootstrap](docs/journal/2026-04-15-bootstrap.md)
- [2026-04-15 — Визуализация](docs/journal/2026-04-15-viz.md)
- [2026-04-16 — UD-разбор, клаузы, метарёбра](docs/journal/2026-04-16-ud-and-metaedges.md)
- [2026-04-19 — L2-параграфная агрегация](docs/journal/2026-04-19-l2-paragraph-aggregation.md)
- [2026-05-05 — Девять задач для защиты](docs/journal/2026-05-05-diploma-features.md)
- [2026-05-06 — Разрешение анафоры (v0)](docs/journal/2026-05-06-anaphora-resolution-v0.md)
- [2026-05-06 — Тематические кластеры и таксономия глаголов](docs/journal/2026-05-06-l2-entity-cluster-and-verb-taxonomy.md)
- [2026-05-07 — Разрешение анафоры v1: замена-в-узле, salience-скоринг](docs/journal/2026-05-07-anaphora-resolution-v1.md)
