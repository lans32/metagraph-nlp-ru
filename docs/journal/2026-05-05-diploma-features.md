# 2026-05-05 — Девять задач для дипломной защиты

## Задача
Реализовать блок функциональности, необходимый для защиты диплома:
типизация клауз, пакетная обработка, логирование с timestamps,
профилирование, укрепление тестов, Streamlit веб-интерфейс, адаптер
MaltParser, расширенная агрегация (coref clusters + NP collapse),
Cytoscape.js визуализация с compound nodes.

## Гипотеза
Если каждая задача реализуется изолированно (отдельный модуль, отдельные
тесты, отдельный конфиг-флаг), интеграция пройдёт без разрушения
существующего pipeline. Существующие агрегаторы и визуализация продолжат
работать, а новые стратегии подключаются через конфигурацию.

## Что сделано

### 1. Типизация клауз (`clause_type`)
- Новое поле `Clause.clause_type: str | None` в доменной модели.
- Функция `_determine_clause_type(pred, parsed)` в `parsers/clauses.py`
  определяет тип по UD-deprel предиката: `main`, `coord`, `compl`,
  `xcompl`, `adverbial`, `relative`, `participial`, `other`.
- Экстрактор `_collect_predicates` расширен: причастия (`VerbForm=Part`
  с deprel `acl`/`acl:relcl`) теперь тоже собираются как предикаты клауз.
- `MetaNode.type` изменён с `"clause"` на `f"clause:{clause_type}"` —
  это прокидывает лингвистическую классификацию на уровень метаграфа.
- Фильтр в `paragraph_metanodes` обновлён: `mn.type.startswith("clause")`.
- 9 юнит-тестов (было 6): добавлены `adverbial`, `relative`, `participial`.

### 2. Пакетная обработка (`batch`)
- CLI-команда `metagraph-nlp batch --input-dir DIR --out-dir DIR`.
- Обработка всех `.txt` файлов в каталоге; парсер инициализируется один
  раз (экономия ~2 с на документ).
- Генерация `summary.json` (total, successful, failed, per-file статус).
- `pipeline.get_default_parser(cfg)` — публичная фабрика с поддержкой
  natasha и maltparser.
- 3 юнит-теста (mock pipeline).

### 3. Логирование с timestamps
- `AuditEvent.timestamp: str | None` — UTC ISO 8601.
- Модуль `logging_config.py`: `setup_logging(level)` настраивает stderr
  handler с `%(asctime)s` форматом.
- `pipeline.run()` логирует каждую стадию через `logging.getLogger`.
- 2 юнит-теста на timestamp.

### 4. Профилирование (`profiling/metrics.py`)
- `StageMetrics` (stage, wall_seconds, peak_memory_kb, output_count).
- `PipelineMetrics` с `to_dict()` и `total_wall_seconds`.
- Контекстный менеджер `measure_stage(name, metrics)` — `time.perf_counter()`
  + `tracemalloc`.
- Каждая стадия `pipeline.run()` обёрнута в `measure_stage`.
- `profiling.json` записывается в артефакты.
- 3 юнит-теста.

### 5. Укрепление тестов
- `tests/conftest.py`: `FakeParser` (Protocol-совместимый), фикстуры
  `fake_parser()`, `ids()`.
- `tests/integration/test_pipeline_determinism.py` — два прогона pipeline
  → сравнение JSON графа и метаграфа.
- 78 unit-тестов + 6 slow integration-тестов. Все зелёные.

### 6. Streamlit веб-интерфейс (`web/app.py`)
- Sidebar: загрузка YAML-конфига.
- Tabs: «Ввод текста» / «Загрузка файла».
- Метрики: предложения, клаузы, узлы, рёбра, метавершины.
- Визуализация: pyvis-embed (граф + метаграф) + Cytoscape.js (метаграф).
- Expanders: таблица клауз с `clause_type`, профилирование, audit log.
- Кнопки экспорта JSON.
- Optional dependency: `pip install -e ".[web]"`.

### 7. Адаптер MaltParser (`maltparser_adapter.py`)
- `parse_conllu(text, sentence_text)` — парсер CoNLL-U формата с
  восстановлением start/end офсетов.
- `MaltParserAdapter(MorphSyntaxParser)` — вызов Java jar через
  `subprocess.run`.
- Конфиг: `morphsyntax.parser = "maltparser"`, `morphsyntax.malt_jar`,
  `morphsyntax.malt_model`.
- 6 юнит-тестов.

### 8. Расширенная агрегация
#### 8a. Coref cluster metanodes (`coref_cluster_metanodes.py`)
- L2-метавершины типа `"coref_cluster"` через connected components
  (union-find) по `shared_entity` метарёбрам.
- Одна L1-метавершина может попасть в несколько кластеров — это делает
  `topic_overlap` нетривиальным (в отличие от параграфной стратегии).
- Конфиг: `aggregation.coref_cluster_enabled`, `.coref_cluster_min_size`.
- 6 юнит-тестов.

#### 8b. NP collapse (`graph_builders/np_collapse.py`)
- Свёртка NOUN + модификаторы (amod, nmod, det) в один узел с составной
  леммой. Запускается после `build_semantic_graph`, до агрегации.
- Возвращает новый `SemanticGraph` — исходный не мутируется.
- Конфиг: `aggregation.np_collapse_enabled`.
- Исправлен баг: merged-узлы, добавленные в `new_nodes` до обнаружения
  merge, теперь корректно фильтруются.
- 6 юнит-тестов.

### 9. Cytoscape.js визуализация (`viz/cytoscape_export.py`)
- `metagraph_to_cytoscape_elements(metagraph, graph, clauses)` — JSON
  в формате Cytoscape.js elements с compound nodes.
- L0-узлы вложены в L1-метавершины через `parent`, L1 — в L2.
- Self-contained HTML: Cytoscape.js + fcose layout из CDN.
- Боковая панель Inspector: нажатие на элемент → таблица свойств (id,
  type, lemma, upos, rule, stage, clause_text).
- Чекбоксы фильтрации по уровням (L0 / L1 / L2).
- Интеграция: `metagraph_cytoscape.html` в артефактах; третий таб в
  Streamlit.
- 7 юнит-тестов.

## Результат
Pipeline полностью функционален с 84 тестами (78 fast + 6 slow).
Стадии pipeline:

```
normalize → segment → parse → clauses → graph_builder
  → [np_collapse] → aggregate_L1 → [shared_entity_L1_edges]
  → [coref_clusters] → [paragraph_L2_nodes] → [topic_overlap_L2_edges]
```

Стадии в квадратных скобках — опциональные, управляются конфигом.

Три формата визуализации: pyvis HTML (граф, метаграф), GraphViz DOT,
Cytoscape.js HTML с compound nodes.

## Ограничения
1. **MaltParser не тестирован end-to-end** — требует Java и скачанную
   модель. Протестирован на уровне CoNLL-U парсера и фабрики.
2. **NP collapse — эвристика по леммам**. Если natasha назначает разные
   леммы одному слову в разных контекстах, collapse может не сработать.
3. **Cytoscape.js CDN** — HTML требует интернета при первом открытии.
4. **Streamlit не тестирован как UI** — только проверка импортов.
5. **Coref clusters при параграфной стратегии** — обе L2-стратегии
   работают параллельно, но coref-кластеры появляются только при наличии
   shared_entity связей между L1-метавершинами.

## Следующий шаг
1. Подготовка корпуса из 5–10 текстов для демонстрации на защите.
2. Сравнительный анализ natasha vs. MaltParser на тестовом корпусе.
3. Содержательный `label` для L2-метавершин (главная лемма фрагмента).
4. Экспорт метаграфа в формат для NetworkX / igraph для аналитики.
