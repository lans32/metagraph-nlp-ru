# 2026-05-22 — Иерархическая семантика предикатов через RuWordNet

## Lab note

**Дата:** 2026-05-22.

**Задача.** Сильно расширить покрытие правила
`predicate_class_cluster_v0` и сделать L2-агрегацию по семантике
предикатов многоуровневой. До этой даты словарь
`configs/predicate_classes.yaml` содержал ручную разметку: 8 плоских
классов и ~100 лемм. Большинство глаголов реального текста не получали
`predicate_class` вовсе, поэтому семантическая ось L2-агрегации (одна
из трёх обещанных в lab-note от 2026-05-06) давала мало кластеров и
неинформативные метавершины.

**Гипотезы.**

1. Внешняя rule-based лексическая база ([RuWordNet 2.0](https://ruwordnet.ru/),
   7636 глагольных синсетов с отношениями hyponym/hypernym/cause) даёт
   достаточный объём и иерархию, чтобы качественно сменить уровень
   агрегации без отказа от инварианта объяснимости (CLAUDE.md §3).
2. Предкомпиляция в YAML — правильная стратегия интеграции:
   зависимость от пакета `ruwordnet` нужна только при сборке
   артефакта `configs/predicate_classes_ruwordnet.yaml`, а
   runtime pipeline продолжает работать с обычным YAML-словарём.
   Это сохраняет CPU-first и offline-friendly стек (CLAUDE.md §15).
3. Если на L2 создавать predicate-кластеры на нескольких уровнях
   иерархии одновременно (leaf → mid → root), холархия (CLAUDE.md §4.4)
   проявится не только между разными стратегиями (параграф / сущность /
   предикат), но и внутри одной стратегии — одна L1-клауза попадает в
   несколько predicate-кластеров на разных уровнях абстракции.
4. Опциональные `contains`-метарёбра между parent ↔ child predicate-
   кластерами делают дендрограмму (CLAUDE.md §4.4) явной в метаграфе
   (по §9.3), не превращая её в имплицитный артефакт визуализации.

**Что изменено.**

- `configs/predicate_anchors.yaml` (новый): 12 anchor-классов с
  `seed_lemma`, `label_ru`, `resolved_synset_id` и контрольным числом
  hyponyms. Покрывают исходные 8 классов (motion, communication,
  cognition, perception, possession, creation, change_of_state,
  causation) плюс 4 расширения (emotion, volition, existence,
  physical_action). Synset-ID зафиксированы из RuWordNet 2.0 (2021)
  для воспроизводимости.
- `scripts/build_predicate_lexicon.py` (новый): CLI-скрипт. BFS
  hyponyms от каждого anchor до `--max-depth`. Имя класса —
  детерминированный slug
  (русская транслитерация + ограничение длины). Записывает YAML v1
  с двумя секциями: `hierarchy` (parent / level / anchor_synset_id /
  label_ru) и `lemmas` (lemma → список путей `[leaf, mid, root]`).
- `configs/predicate_classes_ruwordnet.yaml` (новый, артефакт сборки):
  предкомпилированный словарь v1 — 3455 классов и 7978 лемм (на
  max_depth=3). Коммитится в репозиторий.
- `pyproject.toml`: новая опциональная группа
  `[project.optional-dependencies] lexicon = ["ruwordnet>=0.0.5"]` —
  устанавливается только для пересборки словаря, не для основного
  pipeline.
- `domain/predicate_hierarchy.py` (новый): pydantic-модель
  `PredicateHierarchy` с полями `parent_of`, `level_of`, `anchor_of`,
  `label_of`, `lemma_paths`, `metadata` и вспомогательными методами
  `roots()`, `children_of(slug)`, `classes_at_level(level)`.
- `parsers/predicate_lexicon.py`: расширен под версионирование. Loader
  детектирует `version: 0` (legacy плоский) или `version: 1`
  (иерархический) и возвращает одинаковый контракт
  `dict[str, frozenset[str]]` — для v1 во frozenset попадают **все**
  классы пути от leaf к root по всем ветвям. Новая функция
  `load_predicate_hierarchy(path)` возвращает `PredicateHierarchy` для
  v1 и `None` для v0. Внутри — кэш по `path+mtime` против двойного
  парсинга YAML в Phase 1.
- `aggregators/predicate_class_cluster.py`: сигнатура расширена
  параметрами `hierarchy`, `levels`, `create_containment_edges`.
  Поведение для v0 (hierarchy=None) полностью сохранено. Для v1:
  фильтр кластеров по уровню; provenance.notes хранит
  `level=...; parent=...; anchor_synset=...` для трассировки до
  RuWordNet (CLAUDE.md §10). При `create_containment_edges=True`
  правило создаёт L2-метарёбра типа `containment` с
  `relation="contains"` для каждой пары parent ↔ child, обе из которых
  стали L2-метавершинами.
- `config.py`: в `AggregationConfig` добавлены
  `predicate_class_cluster_levels: list[str]` (дефолт
  `["leaf", "mid", "root"]`) и `predicate_hierarchy_edges_enabled: bool`
  (дефолт `False`). Критическая правка `Config.phase1_hash`:
  раньше учитывался только **путь** к словарю — смена YAML без смены
  пути не инвалидировала Phase 1 кэш. Теперь добавлен byte-hash
  содержимого через утилиту `_predicate_classes_file_sha256` —
  смена словаря даёт другой `phase1_hash` и Phase 1 пересчитывается.
- `pipeline.py`: загрузка `predicate_hierarchy` в Phase 2 (только
  если `predicate_class_cluster_enabled`). Параметры
  `hierarchy/levels/create_containment_edges` пробрасываются в
  агрегатор. Audit log получает `params={"hierarchy": "v1"|"v0",
  "levels": "...", "containment_edges": "..."}`.
- `configs/default.yaml`: новые поля с дефолтами. Путь к словарю
  остаётся `null` (= встроенный v0) для backward compat.
- `web/presets.py`: пресеты `predicates` и `full` включают
  `predicate_hierarchy_edges_enabled: True` — для них иерархия
  работает «из коробки», когда пользователь указывает путь к v1
  YAML.
- Тесты:
  - `test_predicate_lexicon.py`: 6 новых тестов для v1 (union путей,
    PredicateHierarchy navigation, unsupported version, YAML-кэш).
  - `test_predicate_class_cluster_hierarchy.py` (новый, 9 тестов):
    фильтр по уровням (leaf / mid / root / все), containment-рёбра
    создаются ровно нужные, orphan-child без containment,
    `hierarchy=None` backward-compat (включая отсутствие containment-
    рёбер даже при включённом флаге).
  - `test_config_phase1_hash.py` (новый, 4 теста): byte-hash YAML
    меняет phase1_hash, стабильность при одинаковом контенте,
    отличие v0 vs v1.
  - `test_thin_slice.py`: новый slow-тест
    `test_thin_slice_predicate_hierarchy_multi_level` — end-to-end
    через natasha на 4-х предложениях, проверяет multi-level L2,
    наличие containment-рёбер, audit `params.hierarchy == "v1"`.
  - `tests/integration/test_build_predicate_lexicon.py` (новый,
    slow, 2 теста): прогон `build_predicate_lexicon.py` на двух
    маленьких anchor'ах, проверка валидности артефакта и обратной
    читаемости loader'ом.

**Результат.** На контрольном тексте «Студент читает книгу.
Преподаватель говорит студенту правду. Журналист пишет статью.
Слушатель слышит лекцию.»:

- Pipeline с RuWordNet-словарём создаёт predicate-кластеры на
  нескольких уровнях абстракции одновременно (наблюдаются и
  leaf-кластеры типа `communication_*_*_*`, и root-кластеры типа
  `communication`/`creation`).
- `audit.jsonl` в событии `predicate_class_cluster_v0` содержит
  `params.hierarchy=v1`, `params.levels=leaf,mid,root`,
  `params.containment_edges=True`.
- `containment`-метарёбра видны как отдельный тип в метаграфе и
  явно соединяют parent ↔ child predicate-кластеры.
- Размер артефакта: 1.3 МБ (3455 классов, 7978 лемм) на max_depth=3.
- Полный быстрый прогон: 137 unit/integration тестов зелёные;
  slow: predicate_hierarchy multi-level и build_predicate_lexicon —
  тоже зелёные.

**Ограничения.**

- Slug-имена классов получаются громоздкими на глубине 3
  (`communication_proiznesti_vygovorit_krichat_govorit_gomonit` и
  подобные). Это нормально для трассировки, но в визуализации стоит
  отображать `label_ru` из иерархии вместо slug — потенциальная
  правка в `viz/cytoscape_export.py` отложена до следующей итерации.
- Лицензия RuWordNet — non-commercial / research use. Для дипломной
  работы это нормально, для коммерческого деривата потребовалась бы
  альтернативная база (FrameBank GPL 3.0 или YARN MIT).
- Скрипт сборки не делает merge с ручным `configs/predicate_classes.yaml` —
  если пользователь хочет добавить специфичные для домена леммы поверх
  RuWordNet, нужен отдельный шаг merge-or-override. Не реализовано:
  следующая итерация, если в текстах будут глаголы, которых
  RuWordNet не покрывает.
- Возвратные пары (`-ся`) и видовые пары (сов./несов.) хранятся в
  RuWordNet как разные senses → попадают как разные ключи в `lemmas`.
  Согласовано с natasha-лемматизацией, но семантически одинаковые
  пары не объединяются. Это можно сделать на уровне нормализации
  лемм, но это отдельная задача (потеря объяснимости).
- `min_cluster_size` остаётся один общий для всех уровней. На
  практике для root-уровня хороший cutoff может быть выше, чем для
  leaf — `min_size_per_level: dict[str, int]` отложен до появления
  конкретного запроса.

**Следующие шаги.**

1. Заменить slug-label на `label_ru` в visualisation
   (`cytoscape_export.py` и `pyvis HTML`) — читаемость в Cytoscape
   ощутимо повысится.
2. Эксперимент: применить `aggregation-rule-reviewer` к новому
   правилу и собрать список граничных случаев (одиночные leaf-кластеры,
   глубокие ветви RuWordNet с малой плотностью лемм в реальных текстах).
3. Опциональный merge с ручным `predicate_classes.yaml` — если у
   домена есть специфические леммы (термины VKR-предметной области),
   добавить их через override-секцию в anchors.yaml.
4. L3-эксперимент: использовать RuWordNet-иерархию для
   проектирования следующего уровня абстракции (метакластер тем,
   объединяющий entity_centric и predicate_class root-уровня).

---

## Thesis paragraph

Переход от ручного словаря predicate-классов к иерархической
классификации через [тезаурус RuWordNet 2.0](https://ruwordnet.ru/)
сохраняет инвариант объяснимости (CLAUDE.md §3): источником семантики
остаётся внешняя rule-based лексическая база, а не нейросетевая модель.
Скрипт `scripts/build_predicate_lexicon.py` разово обходит
hyponym-дерево от 12 выбранных корневых синсетов (anchor'ов) до
заданной глубины и материализует расширенный YAML версии 1 с двумя
секциями: дерево классов (`hierarchy`) и обратный индекс лемма → пути
от leaf к root (`lemmas`). В отличие от исходной плоской разметки из
8 классов и ~100 лемм, новый словарь покрывает 3455 классов и 7978
лемм на глубине 3 — превышение почти в два порядка по покрытию.
Pipeline в runtime от пакета `ruwordnet` не зависит: предкомпилированный
артефакт коммитится в репозиторий, что обеспечивает воспроизводимость
и совместимость с CPU-first / offline стеком (CLAUDE.md §15). На уровне
агрегации иерархия проявляется как multi-level L2-кластеризация: одна
L1-клауза одновременно принадлежит leaf-кластеру предиката, его
mid-родителю и root-классу — холархия (CLAUDE.md §4.4), реализованная
не межосной (между параграфом / сущностью / предикатом, как в
2026-05-06), а внутри одной стратегии. Опциональные метарёбра типа
`containment` явно фиксируют родительско-дочерние отношения между
predicate-кластерами (CLAUDE.md §9.3), превращая имплицитную
дендрограмму в первоклассный объект модели. Phase 1 hash учитывает
byte-hash содержимого YAML — смена словаря инвалидирует кэш и
гарантирует пересчёт графа, что важно для исследовательских
экспериментов с разными версиями anchors. Provenance каждой L2-
метавершины хранит `anchor_synset_id` RuWordNet, что обеспечивает
обратимую трассировку до источника (CLAUDE.md §10) — необходимое
условие для дипломной защиты.
