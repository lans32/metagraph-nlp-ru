# 2026-05-07 — Разрешение анафоры v1: замена-в-узле, расширение покрытия, salience-веса

## Задача

Версия `anaphora_resolution_v0` ([2026-05-06](2026-05-06-anaphora-resolution-v0.md))
покрывала только личные местоимения 3-го лица и удаляла PRON-узлы из графа,
перенаправляя рёбра на антецедент. Это давало три проблемы:

1. **Прозрачность.** На визуализации местоимение полностью исчезало —
   нельзя было увидеть, что в этой позиции вообще что-то стояло.
2. **Узкое покрытие.** Притяжательные («его/её/их») и возвратные
   («себя/свой») оставались в графе как обычные PRON, разрывая
   связность так же, как раньше разрывали личные.
3. **Качество.** Лексикографический скоринг
   `(propn_score, subj_score, recency)` плохо работал, когда и
   PROPN-кандидат, и nsubj-кандидат были не одним и тем же словом —
   ближайший по позиции выигрывал, даже если был «случайным».

## Гипотеза

Три параллельных улучшения дадут лучшую связность графа и понятную
визуализацию без отказа от детерминированного rule-based ядра:

- **A.** Если PRON-узел остаётся в графе с обновлёнными лексическими
  атрибутами (`label`/`lemma`/`upos` берутся у антецедента), а исходные
  значения сохраняются в `original_lemma`/`original_upos`, то связность
  возникает естественно через `shared_entity_by_lemma_v0`: два узла с
  лемма «иван» (настоящий и бывший «он») попадают в одну группу. Рёбра
  трогать не нужно — топология не ломается.
- **B.** Притяжательные 3-го лица (Poss=Yes+Person=3+Reflex≠Yes)
  разрешаются той же логикой, что и личные. Возвратные (Reflex=Yes)
  тривиально берут subject текущей клаузы и не требуют поиска по окну.
- **C.** Численный salience-скоринг (упрощённый Lappin–Leass) с весами
  за роль, POS, расстояние, тематическую позицию и повторное
  упоминание точнее ранжирует кандидатов, чем лексикографическое
  сравнение трёх флагов.

## Что изменено

### Доменная модель

- **`Node`** ([src/metagraph_nlp/domain/graph.py](../../src/metagraph_nlp/domain/graph.py)):
  три новых опциональных поля — `original_lemma`, `original_upos`,
  `antecedent_node_id`. Заполняются только для узлов, заменённых
  анафорой; для остальных — None.
- **`AnaphoraResolution`** ([src/metagraph_nlp/domain/anaphora.py](../../src/metagraph_nlp/domain/anaphora.py)):
  добавлены `pronoun_type` (`personal_3p` / `possessive_3p` /
  `reflexive`), `resolution_strategy` (`search` / `clause_subject`),
  `salience_score` (None для возвратных).

### Конфигурация

- **`SalienceWeights`** ([src/metagraph_nlp/config.py](../../src/metagraph_nlp/config.py)):
  новый pydantic-класс с восемью весами. Дефолты:
  `subj=80, obj=50, oblique=20, propn=50, recency_per_sent=-10,
  thematic=20 (≤3 токена), repeat_mention=30`.
- **`AnaphoraConfig.pronoun_types`**: дефолт расширен до
  `["personal_3p", "possessive_3p", "reflexive"]`. Можно отключать
  отдельные типы через конфиг или Streamlit.
- **`AnaphoraConfig.salience_weights`** добавлено как поле.

### Логика разрешения

[src/metagraph_nlp/parsers/anaphora.py](../../src/metagraph_nlp/parsers/anaphora.py)
полностью переписан:

1. **Классификация**: новая функция `_classify_pronoun(token)`
   определяет тип в порядке приоритета: reflexive → possessive_3p →
   personal_3p (порядок важен, чтобы «свой» попал в reflexive, а не в
   possessive).
2. **Маршрутизация**: для возвратных вызывается `_find_clause_subject`
   (subject текущей клаузы), для остальных — `_find_antecedent_by_search`
   (поиск по окну с salience-скорингом). Hard constraints
   (Number/Gender/Animacy) применяются как фильтры до скоринга.
3. **Salience**: `_salience_score(cand_tok, cand_pos, pron_pos,
   repeat_count, weights)` возвращает int — сумму весов. Лучший =
   `argmax(score)`; tie-breaker — recency.
4. **Repeat mention**: счётчик `prior_antecedent_counts` увеличивается
   после каждого разрешения; кандидат, ранее уже бывший антецедентом,
   получает бонус `weights.repeat_mention` за каждое повторение —
   реализация salience через повтор.
5. **Замена-в-узле** (главное отличие от v0): PRON-узел остаётся в
   `final_nodes`, его атрибуты обновляются через `n.model_copy(update={...})`.
   Логика дедупликации рёбер и схлопывания петель удалена — топология
   графа не меняется.
6. **Provenance**: `_RULE = "anaphora_resolution_v1"`. У заменённого
   узла в `provenance.inputs` добавляется `antecedent_id`, в `notes` —
   `anaphora_replaced: original_lemma=...; antecedent=...`.

### Виз-слой

В трёх рендерерах ([html_pyvis.py](../../src/metagraph_nlp/viz/html_pyvis.py),
[dot.py](../../src/metagraph_nlp/viz/dot.py),
[cytoscape_export.py](../../src/metagraph_nlp/viz/cytoscape_export.py))
добавлен helper `_node_display_label(node)`: для узлов с
`antecedent_node_id != None` метка становится `«Иван ←Он»`. Тултип/data
дополнен полями `original_lemma`, `original_upos`, `antecedent_node_id`.

### Streamlit

[src/metagraph_nlp/web/app.py](../../src/metagraph_nlp/web/app.py):
в секции «Разрешение анафоры» добавлен `multiselect` типов местоимений с
русскими лейблами. Таблица результатов обогащена колонками
`Тип`, `Стратегия`, `Salience`.

### Pipeline

[src/metagraph_nlp/pipeline.py](../../src/metagraph_nlp/pipeline.py):
вызов `resolve_anaphora` теперь передаёт `pronoun_types` и
`salience_weights` из конфига. В `audit.record` имя правила обновлено на
`anaphora_resolution_v1`.

## Результат

- **Тесты**: 17 для анафоры (10 unit + 2 integration старых, обновлённых
  под новую семантику; +5 новых для possessive/reflexive/salience/v1-rule
  /pronoun_types-фильтра). Полный набор проходит: 118 passed.
- **Связность**: на тестовом тексте «Иван пришёл домой. Он устал.»
  PRON-узел «Он» остаётся в графе с lemma=«иван», upos=PROPN,
  surface=«Он». `shared_entity_by_lemma_v0` создаёт метаребро между
  cl-1 («Иван пришёл домой») и cl-2 («Он устал») по общей лемме
  «иван» — связь, которая в v0 рисовалась через перенаправленное
  ребро.
- **Visual**: в pyvis/cytoscape узел отображается как «иван ←Он» — и
  кого подставили, и где стояло местоимение.

## Ограничения v1

- **Указательные** («этот, тот, такой») и **относительные** («который»)
  по-прежнему вне scope. Указательные особенно сложны: в безличных
  конструкциях («это очевидно», «то, что…») у них нет антецедента-сущности —
  нужен отдельный фильтр контекста.
- **Притяжательные**: дискриминация «его-личное (Case=Gen)» от
  «его-притяжательное (Poss=Yes)» полагается на полноту natasha-feats.
  Если natasha не выставила Poss, форма попадёт в personal_3p ветку,
  что обычно тоже даёт корректный результат.
- **Гендер для PROPN**: natasha не всегда выставляет Gender для имён
  собственных. Простая морфологическая эвристика по окончанию (Иван →
  Masc, Анна → Fem) не реализована; при отсутствии feats у обоих
  стороны фильтр Gender просто не применяется.
- **Salience веса**: дефолты подобраны эмпирически на нескольких
  тестовых текстах, не калиброваны на размеченном корпусе. На реальных
  данных может потребоваться тюнинг.
- **Цепочки замен**: если антецедент сам был ранее заменён анафорой,
  цепочка следует к финальной сущности (поведение v0 сохранено), но
  promenance-цепочка фиксируется только для прямой ссылки на
  непосредственный антецедент в момент разрешения.

## Следующий шаг

1. Прогон на реальном корпусе с natasha — оценить покрытие и долю
   ошибок на возвратных и притяжательных.
2. Указательные «этот/тот»: рассмотреть с фильтром безличных
   конструкций (nsubj=это+VerbForm=Inf или nsubj=это+отсутствие
   ясного семантического antecedent).
3. Относительные «который»: антецедент = head существительного, к
   которому крепится придаточное (deprel=acl:relcl head). Структурно
   определимо без эвристик — высокая ожидаемая точность.
4. Гендер-эвристика для PROPN по окончанию (краткий справочник
   мужских/женских русских имён).
5. Калибровка `SalienceWeights` на размеченном фрагменте.

## Затронутые файлы

### Анафора (v1)
- [src/metagraph_nlp/parsers/anaphora.py](../../src/metagraph_nlp/parsers/anaphora.py) (полностью переписан)
- [src/metagraph_nlp/domain/graph.py](../../src/metagraph_nlp/domain/graph.py) (`Node` + 3 поля)
- [src/metagraph_nlp/domain/anaphora.py](../../src/metagraph_nlp/domain/anaphora.py) (`AnaphoraResolution` + 3 поля)
- [src/metagraph_nlp/config.py](../../src/metagraph_nlp/config.py) (`SalienceWeights`, `AnaphoraConfig.pronoun_types`)
- [src/metagraph_nlp/pipeline.py](../../src/metagraph_nlp/pipeline.py) (передача новых параметров, обновлённый audit)
- [src/metagraph_nlp/viz/html_pyvis.py](../../src/metagraph_nlp/viz/html_pyvis.py) (`_node_display_label`)
- [src/metagraph_nlp/viz/dot.py](../../src/metagraph_nlp/viz/dot.py) (`_node_display_label`, `_anaphora_tooltip_extra`)
- [src/metagraph_nlp/viz/cytoscape_export.py](../../src/metagraph_nlp/viz/cytoscape_export.py) (display label + новые data-поля)
- [src/metagraph_nlp/web/app.py](../../src/metagraph_nlp/web/app.py) (multiselect типов, расширенная таблица)
- [tests/unit/test_anaphora_resolution.py](../../tests/unit/test_anaphora_resolution.py) (15 тестов)
- [tests/integration/test_anaphora_pipeline.py](../../tests/integration/test_anaphora_pipeline.py) (обновлены ассерты)
