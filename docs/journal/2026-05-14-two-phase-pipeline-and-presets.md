# 2026-05-14 — Двухфазный pipeline, пресеты, entity-centric агрегатор, параллельный парсинг

## Лабораторная заметка

**Дата:** 2026-05-14

**Задача:** Сделать интерфейс пригодным для демонстрации: переключение
стратегий агрегации без перепарсинга, осмысленные «темы как сущности»
вместо одной гигантской компоненты связности, отсутствие подвисания
браузера на больших графах, использование всех ядер CPU при тяжёлом
парсинге.

**Гипотеза:** Текущий pipeline можно разделить на две фазы по
естественной границе `semantic graph → metagraph`. Стадии Phase 1
(normalize → parse → clauses → graph → анафора → NP collapse) не зависят
от настроек агрегации L1/L2 — их результат можно кэшировать в
`session_state` и пересчитывать только Phase 2 при смене конфига
агрегации. Это даёт мгновенную смену представлений без ущерба для
объяснимости (CLAUDE.md §9.6 о границах стадий). Параллельно: правило
`entity_cluster_v0` (union-find по shared_entity) транзитивно склеивает
большую часть клауз в один кластер; альтернатива «по одной L2-метавершине
на каждую значимую лемму» должна давать несколько per-entity групп с
естественными пересечениями (одна клауза в нескольких темах = холархия).

**Что изменено:**

### Архитектура pipeline
- **Phase1Result + run_phase1() + run_phase2()** в
  [src/metagraph_nlp/pipeline.py](../../src/metagraph_nlp/pipeline.py):
  Phase 1 (text → semantic graph) возвращает `Phase1Result` с
  `id_snapshot: dict[str,int]`. Phase 2 принимает `Phase1Result` и
  свежий `Config`, создаёт `IdFactory.from_snapshot()` для продолжения
  нумерации, копирует Phase 1 audit/metrics и добавляет свои.
- **`run()`** сохранён как тонкий wrapper, вызывающий обе фазы — CLI
  (`process`, `batch`) и все 118 тестов работают без изменений.
- **`IdFactory.snapshot()` / `from_snapshot()`** в
  [src/metagraph_nlp/domain/ids.py](../../src/metagraph_nlp/domain/ids.py):
  сериализация/восстановление счётчиков идентификаторов между фазами.
- **`Config.phase1_hash()`** в
  [src/metagraph_nlp/config.py](../../src/metagraph_nlp/config.py):
  хэш только Phase 1 полей (morphsyntax / clauses / graph / anaphora /
  np_collapse / predicate_classes_path). Используется как часть ключа
  кэша в UI.
- Лексикон предикатных классов теперь загружается всегда (раньше — по
  условию `predicate_class_cluster_enabled`), `Edge.predicate_class`
  заполняется безусловно. Тогда `predicate_class_cluster_enabled` стал
  чистым Phase 2 тоглом.

### Entity-centric агрегатор
- **Новое правило `entity_centric_v0`** в
  [src/metagraph_nlp/aggregators/entity_centric_metanodes.py](../../src/metagraph_nlp/aggregators/entity_centric_metanodes.py):
  по одной L2-метавершине на каждую значимую лемму, содержащую все
  L1-клаузы с этой леммой во фрагменте. Использует те же фильтры, что
  `shared_entity_metaedges` (exclude PRON/DET/ADP/AUX/CCONJ/SCONJ/PART,
  предикаты, короткие леммы). PROPN-леммы при `propn_always=true`
  включаются при freq ≥ 1 — имена собственные всегда значимы.
- **Конфиг**: `entity_centric_enabled` (default false),
  `entity_centric_min_freq` (default 2), `entity_centric_propn_always`
  (default true).
- В отличие от `entity_cluster_v0` (union-find — одна большая
  компонента), даёт несколько per-entity групп; одна L1-клауза может
  входить в несколько entity-метавершин (холархия по §4.4 CLAUDE.md).

### Пресеты агрегации
- **Новый модуль** [src/metagraph_nlp/web/presets.py](../../src/metagraph_nlp/web/presets.py):
  6 пресетов — `clauses_only`, `entities` (L1 + shared_entity +
  entity_centric), `paragraphs`, `predicates`, `full`, `custom`. Каждый
  пресет применяется через `model_copy(update=...)` и трогает только
  поля `AggregationConfig`, не затрагивая Phase 1.

### UI рефакторинг
- [src/metagraph_nlp/web/app.py](../../src/metagraph_nlp/web/app.py)
  переписан вокруг session_state:
  - **Двухсекционный sidebar**: «Анализ текста» (Phase 1, требует
    пересчёта) и «Агрегация» (Phase 2, мгновенно).
  - Selectbox пресетов в Phase 2; ручные тоглы доступны при
    `custom`-режиме.
  - Phase 1 результат кэшируется по ключу `sha256(text)[:16] +
    config.phase1_hash()`. Phase 2 запускается при каждом Streamlit
    rerun, если кэш есть.
  - Загрузка файлов: пробует кодировки `utf-8 → utf-8-sig → cp1251 →
    cp866 → koi8-r` (раньше падала с `UnicodeDecodeError` на CP1251).
  - Слайдер `morphsyntax.workers` в sidebar; максимум = число CPU ядер.

### Параллельный парсинг
- `_worker_init` / `_worker_parse` / `_parse_sentences_parallel` в
  [pipeline.py](../../src/metagraph_nlp/pipeline.py): `ProcessPoolExecutor`
  с `initializer` загружает парсер один раз на воркер.
- Активируется при `morphsyntax.workers > 1` И `len(sentences) >=
  morphsyntax.parallel_threshold` (default 16) — на малых документах
  spawn-overhead превышает выигрыш.

### Cytoscape: фильтры и объединение
- [src/metagraph_nlp/viz/cytoscape_export.py](../../src/metagraph_nlp/viz/cytoscape_export.py):
  - **Объединение параллельных shared_entity рёбер**: между парой клауз
    с N общими леммами теперь одно ребро с label `«N общих: lemma1,
    lemma2, ...»`, списком всех лемм в инспекторе, толщиной
    пропорциональной N. Раньше было N параллельных рёбер.
  - **Toggle-фильтры рёбер**: 4 чекбокса (`base` / `shared_entity` /
    `topic_overlap` / `contains`) в дополнение к существующим level
    toggle (L0/L1/L2).
  - **Параметры `hidden_levels` / `hidden_etypes`** в
    `render_cytoscape_html()`: начальное состояние чекбоксов и
    соответствующих элементов задаётся при экспорте.

### Авто-упрощение визуализации
- При `total_elements > viz_limit` (default 500) **рендер не
  запускается автоматически** — пользователь нажимает «Только L2
  (быстро)» (`hidden_levels=[0,1]` + все рёбра скрыты) или «Полный вид
  (медленно)». Выбор привязан к signature результата — при смене
  пресета сбрасывается.

**Результат:**

- Тесты: **118 passed**, обратная совместимость `run()` сохранена.
- Параллельный парсинг измерен на 50 предложениях с 4 воркерами: parse
  ускоряется с 1.56s до 1.18s (~1.33x). На больших текстах ожидаемо
  3-4x — узкое место — однократная загрузка моделей natasha в каждом
  воркере.
- На тексте «Анна пошла в магазин. Вова ждал дома. Анна встретила
  Вову» пресет «По сущностям» создаёт **2 entity-centric метавершины**
  (Анна, Вова) с пересечением на клаузе встречи, вместо одной
  union-find компоненты.
- Переключение пресета «Полный» → «По сущностям» в UI триггерит только
  Phase 2: лог `phase2 complete: 1.344s total`, Phase 1 стадий нет —
  кэш сработал.

**Ограничения:**

- Параллельный парсинг: каждый воркер загружает модели natasha (~200MB
  RSS), для 8 воркеров ≈ 1.6GB памяти. На многоядерных Linux-серверах
  выигрыш будет больше; на маленьких документах (< 16 предложений)
  spawn-overhead делает sequential быстрее.
- Авто-упрощение визуализации привязано к `viz_limit` по сумме всех
  элементов — не учитывает топологическую сложность (длинные цепочки
  vs. плотные клики). При плотных графах < 500 элементов всё ещё
  возможны тормоза.
- `entity_centric_v0` создаёт метавершину для каждой леммы, проходящей
  фильтры — в больших текстах их может быть много (десятки-сотни). При
  необходимости можно поднять `entity_centric_min_freq` или ограничить
  топ-N по частоте.
- Кэш Phase 1 в session_state не персистится между перезапусками
  Streamlit; для production-ready решения нужно файловое кэширование
  или Redis.
- Пресет применяется поверх текущего конфига, не сохраняя в
  session_state выбор отдельных тоглов — при возврате в `custom`
  состояние пресета остаётся, но это ожидаемо.

**Следующий шаг:**

1. Сравнить `entity_cluster_v0` и `entity_centric_v0` на реальном
   корпусе по числу осмысленных групп и доле «склеек».
2. Добавить unit-тесты для `entity_centric_v0`, `phase1_hash()`,
   `IdFactory.snapshot()`, бандлинга shared_entity-рёбер в Cytoscape
   export.
3. Профилирование параллельного парсинга на документе 200+ предложений
   (фрагмент из реального корпуса).
4. Рассмотреть persistent кэш Phase 1 (диск) для повторных запусков на
   тех же текстах.
5. В viz: возможность фильтровать L2-метавершины по типу
   (`entity_centric` отдельно от `paragraph` / `predicate_class`).

⚠ Это изменение затрагивает **правила агрегации** (новое правило
`entity_centric_v0` как альтернатива `entity_cluster_v0`) и **границы
стадий pipeline** (введена явная двухфазность с промежуточным
`Phase1Result`-артефактом и кэшированием). Архитектурные решения
зафиксированы в [CLAUDE.md §2 (MVP scope), §5.1, §12.3, §12.10, §15.3](../../CLAUDE.md).
В дипломе стоит отразить отдельно: (а) сравнение двух стратегий
тематической агрегации (connected components vs. per-entity), (б)
двухфазная архитектура pipeline как способ совмещения детерминизма с
интерактивной демонстрацией.

## Затронутые файлы

### Pipeline и доменная модель
- [src/metagraph_nlp/pipeline.py](../../src/metagraph_nlp/pipeline.py) — `Phase1Result`, `run_phase1`, `run_phase2`, параллельный парсинг
- [src/metagraph_nlp/domain/ids.py](../../src/metagraph_nlp/domain/ids.py) — `snapshot` / `from_snapshot`
- [src/metagraph_nlp/config.py](../../src/metagraph_nlp/config.py) — `phase1_hash`, `entity_centric_*`, `morphsyntax.workers`, `morphsyntax.parallel_threshold`

### Агрегация
- [src/metagraph_nlp/aggregators/entity_centric_metanodes.py](../../src/metagraph_nlp/aggregators/entity_centric_metanodes.py) — новое правило
- [src/metagraph_nlp/aggregators/__init__.py](../../src/metagraph_nlp/aggregators/__init__.py) — экспорт `aggregate_entity_centric`

### UI и визуализация
- [src/metagraph_nlp/web/presets.py](../../src/metagraph_nlp/web/presets.py) — модуль пресетов
- [src/metagraph_nlp/web/app.py](../../src/metagraph_nlp/web/app.py) — двухфазный UI, session_state, авто-упрощение viz, кодировка
- [src/metagraph_nlp/viz/cytoscape_export.py](../../src/metagraph_nlp/viz/cytoscape_export.py) — фильтры рёбер, бандлинг параллельных shared_entity, `hidden_levels` / `hidden_etypes`

### Документация
- [CLAUDE.md](../../CLAUDE.md) — §2, §5.1, §12.3, §12.9, §12.10, §15.3
- [README.md](../../README.md) — раздел «Возможности», таблица конфигурации

---

## Фрагмент для диплома

В рамках доработки веб-интерфейса pipeline был разделён на две фазы по
естественной границе «семантический граф → метаграф». Phase 1 включает
все детерминированные лингвистические преобразования (нормализация,
сегментация, морфо-синтаксический разбор, выделение клауз, построение
семантического графа, опциональное разрешение анафоры и свёртка именных
групп), её результат не зависит от настроек агрегации и сохраняется как
артефакт `Phase1Result` с фиксацией состояния фабрики идентификаторов.
Phase 2 выполняет агрегацию L1/L2 и принимает `Phase1Result` вместе со
свежим конфигом; это позволяет переключать стратегии агрегации без
перепарсинга текста, сохраняя детерминизм и audit trail (audit-события
Phase 1 копируются в результат Phase 2 без перезаписи). Параллельно
введено новое правило агрегации `entity_centric_v0`, создающее по одной
L2-метавершине на каждую значимую лемму и содержащее все
L1-метавершины-клаузы с её упоминанием; в отличие от `entity_cluster_v0`,
основанного на поиске компонент связности по графу `shared_entity` и
страдающего от транзитивного склеивания, `entity_centric_v0` даёт
несколько per-entity групп с естественными пересечениями, реализуя
заявленную в архитектуре холархию — одна клауза может одновременно
входить в несколько тематических метавершин. Для тяжёлых текстов
дополнительно реализован параллельный парсинг через `ProcessPoolExecutor`
с одноразовой инициализацией парсера в каждом воркере, активируемый при
числе предложений выше порога; на 50 предложениях с четырьмя воркерами
получено ускорение стадии парсинга в 1.33 раза, при ожидаемом росте до
3–4 раз на текстах объёмом несколько сотен предложений (предел
определяется временем загрузки нейросетевых моделей анализатора в
каждый процесс).
