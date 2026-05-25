# metagraph-nlp-ru

Построение метаграфового представления русскоязычного текста.
Исследовательский проект для дипломной работы. Основной pipeline:

```
raw text → normalize → sentences → parse (UD) → clauses → semantic graph
  → [anaphora resolution]
  → metagraph L1 (clause metanodes + shared_entity metaedges)
  → metagraph L2 (paragraphs + entity_clusters + predicate_class_clusters
                  + topic_overlap)
```

Стадии в квадратных скобках — опциональные, управляются конфигурацией.

## Возможности

- **Двухфазный pipeline (`run_phase1` / `run_phase2`)**: тяжёлая фаза `text → semantic graph` отделена от лёгкой фазы агрегации. В Streamlit-UI Phase 1 кэшируется в `session_state`, Phase 2 перезапускается мгновенно при смене настроек или пресета — без перепарсинга.
- **Пресеты агрегации**: 6 предустановленных режимов (`Только клаузы`, `По сущностям`, `По параграфам`, `По семантике предикатов`, `Полный`, `Пользовательский`) — для демо и быстрого переключения между стратегиями без россыпи тоглов.
- **Типизация клауз**: main, coord, compl, xcompl, adverbial, relative, participial — по UD-deprel предиката.
- **Многоуровневый метаграф (холархия)**: L0 (узлы/рёбра) → L1 (клаузы, shared_entity) → L2 (параграфы, entity-кластеры по общим сущностям, **entity-centric метавершины** — по одной на значимую сущность, predicate-class-кластеры по таксономии глаголов, topic_overlap). Одна L1-клауза может одновременно входить в несколько L2-метавершин.
- **Содержательные labels у L2-метавершин**: самая частая значимая лемма во фрагменте (для paragraph и entity_cluster), сама лемма (для entity_centric), имя семантического класса (для predicate_class_cluster).
- **Минимальная таксономия глаголов**: ручной YAML-словарь `configs/predicate_classes.yaml` (motion, communication, cognition, perception, possession, creation, change_of_state, causation), поле `Edge.predicate_class`, агрегатор `predicate_class_cluster_v0`.
- **Разрешение анафоры (v1)**: rule-based замена-в-узле для личных, притяжательных 3-го лица и возвратных местоимений (он/она/оно/они, его/её/их с Poss=Yes, себя/свой). PRON-узел остаётся в графе с обновлёнными `lemma`/`upos`/`label` от антецедента; исходные значения сохраняются в `original_lemma`/`original_upos`, добавляется `antecedent_node_id`. Кандидаты ранжируются упрощённым Lappin–Leass salience-скорингом (subj / obj / obl / propn / recency / thematic / repeat_mention). Hard constraints — Gender / Number / Animacy. Возвратные берут subject текущей клаузы.
- **Параллельный парсинг (ProcessPool)**: распараллеливание `parse`-стадии по предложениям через `ProcessPoolExecutor` (`morphsyntax.workers`, `morphsyntax.parallel_threshold`). На больших текстах даёт 3-4x ускорение; на малых — sequential без накладных расходов на spawn.
- **Три формата визуализации**: pyvis HTML, GraphViz DOT, Cytoscape.js с compound nodes, инспектором и **toggle-фильтрами** по уровням (L0/L1/L2) и типам рёбер (base/shared_entity/topic_overlap/contains). Параллельные `shared_entity` рёбра объединяются в одно с подписью «N общих лемм».
- **Умная визуализация больших графов**: при превышении порога (default 500 элементов) автоматический рендер отключается; пользователь выбирает «Только L2 (быстро)» или «Полный вид (медленно)» — браузер не зависает.
- **Пакетная обработка**: CLI-команда `batch` для каталога `.txt` файлов.
- **Профилирование**: wall time + peak memory на каждую стадию pipeline.
- **Streamlit веб-интерфейс**: ввод текста, двухсекционная панель конфигурации (Phase 1 / Phase 2 — мгновенно), просмотр графа/метаграфа, экспорт JSON. Загрузка файлов с авто-определением кодировки (utf-8 / utf-8-sig / cp1251 / cp866 / koi8-r).
- **Два UD-парсера**: natasha (по умолчанию) и связка `razdel → TreeTagger → MaltParser` (через subprocess) как объяснимый rule/ML-baseline без эмбеддингов.
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

## Альтернативный парсер: MaltParser + TreeTagger

По умолчанию pipeline использует `natasha` — это самый быстрый путь старта. Для исследовательского
сравнения предусмотрен второй backend: классическая связка `razdel → TreeTagger → MaltParser`.
Оба компонента — статистические, но не нейросетевые: TreeTagger — decision-tree теггер
(Schmid, 1994) с MSD-Russian параметрами (Sharoff & Nivre), MaltParser — transition-based
dependency parser (Nivre, 2003). Получается полностью объяснимый baseline без эмбеддингов,
полезный как точка отсчёта для оценки natasha.

### Что нужно установить

Общие зависимости (одинаково на Windows и macOS):

- **Java ≥ 8** для MaltParser.
- **MaltParser** — `maltparser-1.9.2.jar` + обученная модель `.mco` для русского.
  Готовой публичной модели для UD-SynTagRus нет, её нужно обучить локально (см. ниже).
- **TreeTagger** — бинарь под платформу + параметр-файл `russian.par` (от Schmid/Sharoff).
- Python-зависимости (`pymorphy3`, `razdel`, `PyYAML`) уже подтянуты `pyproject.toml`.

### Windows

```powershell
# 1. TreeTagger
# Скачать https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-windows-3.2.5.zip
# Распаковать в C:\TreeTagger\ (или TreeTagger\ внутри проекта)
# Скачать https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/russian-par-linux-3.2-utf8.bin.gz
# (Windows использует тот же .par-файл; распаковать в TreeTagger\lib\russian.par)

# 2. Java
winget install Microsoft.OpenJDK.17
# либо choco install openjdk17

# 3. MaltParser
# Скачать https://www.maltparser.org/dist/maltparser-1.9.2.tar.gz
# Распаковать в maltparser\maltparser-1.9.2\

# 4. UD-treebank для обучения модели
git clone https://github.com/UniversalDependencies/UD_Russian-SynTagRus.git maltparser\UD_Russian-SynTagRus

# 5. Обучить .mco-модель (один раз, ~20-40 минут)
cd maltparser\maltparser-1.9.2
java -Xmx4g -jar maltparser-1.9.2.jar `
  -c ru_syntagrus `
  -m learn `
  -i ..\UD_Russian-SynTagRus\ru_syntagrus-ud-train-a.conllu `
  -if appdata\dataformat\conllu.xml `
  -a nivreeager
# → появится ru_syntagrus.mco в текущей папке

# 6. Прописать пути в configs/malt_treetagger.yaml (или скопировать пример)
# 7. Запустить через CLI:
python -m metagraph_nlp process --input data\samples\short.txt --out artifacts\malt --config configs\malt_treetagger.yaml
```

### macOS

```bash
# 1. Java
brew install openjdk@17
echo 'export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"' >> ~/.zshrc
source ~/.zshrc

# 2. TreeTagger (есть отдельная сборка под macOS, Intel + ARM через Rosetta)
mkdir -p TreeTagger && cd TreeTagger
curl -O https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/tree-tagger-MacOSX-3.2.5.tar.gz
tar -xzf tree-tagger-MacOSX-3.2.5.tar.gz
curl -O https://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/russian-par-linux-3.2-utf8.bin.gz
gunzip russian-par-linux-3.2-utf8.bin.gz
mv russian-par-linux-3.2-utf8.bin lib/russian.par
chmod +x bin/* cmd/*
cd ..

# 3. MaltParser
mkdir -p maltparser && cd maltparser
curl -LO https://www.maltparser.org/dist/maltparser-1.9.2.tar.gz
tar -xzf maltparser-1.9.2.tar.gz

# 4. UD-treebank
git clone https://github.com/UniversalDependencies/UD_Russian-SynTagRus.git

# 5. Обучить модель
cd maltparser-1.9.2
java -Xmx4g -jar maltparser-1.9.2.jar \
  -c ru_syntagrus \
  -m learn \
  -i ../UD_Russian-SynTagRus/ru_syntagrus-ud-train-a.conllu \
  -if appdata/dataformat/conllu.xml \
  -a nivreeager

# 6. Пути для macOS в configs/malt_treetagger.yaml:
#   tree_tagger_bin: TreeTagger/cmd/tree-tagger-russian
#   tree_tagger_param: TreeTagger/lib/russian.par
#   malt_jar: maltparser/maltparser-1.9.2/maltparser-1.9.2.jar
#   malt_model: maltparser/maltparser-1.9.2/ru_syntagrus.mco

# 7. Запуск
python -m metagraph_nlp process --input data/samples/short.txt --out artifacts/malt --config configs/malt_treetagger.yaml
```

### Smoke-проверка

```bash
python -c "from metagraph_nlp.config import Config, MorphSyntaxConfig; from metagraph_nlp.pipeline import get_default_parser; cfg = Config(morphsyntax=MorphSyntaxConfig(parser='maltparser', tree_tagger_bin='TreeTagger/bin/tree-tagger.exe', tree_tagger_param='TreeTagger/lib/russian.par', malt_jar='maltparser/maltparser-1.9.2/maltparser-1.9.2.jar', malt_model='maltparser/maltparser-1.9.2/ru_syntagrus.mco')); print(get_default_parser(cfg).parse('Студент читает книгу.'))"
```

В выводе должны быть непустые `lemma`, `pos`, `feats` и осмысленные `head`/`deprel`.

### Troubleshooting

- **`tree-tagger: command not found`** (macOS) — `chmod +x TreeTagger/bin/* TreeTagger/cmd/*`.
- **«not from identified developer»** на Apple Silicon — System Settings → Privacy & Security → разрешить запуск `tree-tagger` после первой попытки.
- **Кракозябры в lemma** — проверить, что `russian.par` именно UTF-8 (`russian-par-linux-3.2-utf8.bin.gz`), а не latin-1 версия.
- **`java: command not found`** — добавить openjdk в PATH (Mac: `export PATH="/opt/homebrew/opt/openjdk@17/bin:$PATH"`; Windows: PATH через `setx`).
- **`maltparser requires morphsyntax fields:`** — фабрика подсказывает, какие именно пути не заданы. Проверь `configs/malt_treetagger.yaml`.
- **MSD-теги распознаются как `X`** — TreeTagger выдал код, которого нет в `configs/treetagger_tagsets/msd_ru.yaml`. Можно дополнить YAML или подменить на свой через `morphsyntax.tree_tagger_tagset: /abs/path/to/custom.yaml`.

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
| `parsers/` | Нормализация, сегментация, клаузы, адаптеры morphsyntax (natasha, TreeTagger+MaltParser), маппер MSD→UD |
| `graph_builders/` | Семантический граф из UD-ролей |
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
| `aggregation.entity_cluster_enabled` | `true` | L2-метавершины — тематические кластеры по графу shared_entity (правило `entity_cluster_v0`, union-find) |
| `aggregation.entity_cluster_min_size` | `2` | Минимальный размер тематического кластера |
| `aggregation.entity_centric_enabled` | `false` | L2-метавершины — по одной на каждую значимую сущность (правило `entity_centric_v0`); альтернатива union-find кластерам |
| `aggregation.entity_centric_min_freq` | `2` | Минимальное число клауз с леммой для entity-метавершины |
| `aggregation.entity_centric_propn_always` | `true` | PROPN-леммы (имена собственные) включаются при freq ≥ 1 |
| `aggregation.predicate_class_cluster_enabled` | `true` | L2-метавершины — кластеры клауз по классам предикатов из словаря (`predicate_class_cluster_v0`) |
| `aggregation.predicate_class_cluster_min_size` | `2` | Минимальный размер predicate-класса |
| `aggregation.predicate_classes_path` | `null` | Путь к YAML-словарю классов; `null` → встроенный `configs/predicate_classes_ruwordnet.yaml` (v1). Для legacy v0 указать `configs/predicate_classes.yaml` |
| `aggregation.topic_overlap_enabled` | `true` | L2-метарёбра по пересечению фрагментов |
| `anaphora.enabled` | `false` | Разрешение анафоры (`anaphora_resolution_v1`, замена-в-узле) |
| `anaphora.search_window_sentences` | `2` | Окно поиска антецедента в предложениях |
| `anaphora.require_animacy_match` | `true` | Требовать совпадения Animacy PRON и антецедента |
| `anaphora.pronoun_types` | `["personal_3p", "possessive_3p", "reflexive"]` | Покрываемые типы местоимений |
| `anaphora.salience_weights` | см. `SalienceWeights` | Веса упрощённого Lappin–Leass-скоринга кандидатов |
| `morphsyntax.parser` | `"natasha"` | UD-парсер: `natasha` или `maltparser` |
| `morphsyntax.workers` | `1` | Число параллельных процессов для парсинга предложений (1 = последовательно) |
| `morphsyntax.parallel_threshold` | `16` | Минимум предложений для активации параллельного парсинга |
