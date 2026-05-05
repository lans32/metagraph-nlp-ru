# 2026-05-06 — Разрешение анафоры (anaphora_resolution_v0)

## Задача

Личные местоимения 3-го лица (он, она, оно, они и их падежные формы) до
сих пор попадали в семантический граф как самостоятельные PRON-узлы и
затем фильтровались из `shared_entity_metaedges` через `exclude_upos`.
Это приводило к разрыву связности: фрагменты графа из соседних
предложений не связывались, даже когда речь идёт об одной и той же
сущности. Цель — добавить детерминированный rule-based шаг, который
заменяет местоимение на ближайший согласованный антецедент.

## Гипотеза

Для русского языка достаточно простого согласования по
**Gender / Number / Animacy** + recency-эвристики (ближайший подходящий
NOUN/PROPN в окне в 1–2 предложения), чтобы заметно поднять связность
графа без нейросетевого ядра. Особенно важно учитывать `Animacy`: без
него «он» легко цепляется к неодушевлённому существительному (например,
к «дому»), что разрушает осмысленность графа.

## Что изменено

### Pipeline

Между `build_semantic_graph` и `np_collapse` добавлена опциональная
стадия `anaphora_resolution`. Управляется флагом `anaphora.enabled`
(по умолчанию off). Стадия:

1. находит PRON-узлы графа, у которых соответствующий UD-токен имеет
   `PronType=Prs` и `Person=3` (или лемма в `{он, она, оно, они}`);
2. для каждого ищет ближайший предшествующий NOUN/PROPN-узел в окне
   `search_window_sentences` с совпадающими Gender / Number / Animacy;
3. перенаправляет все рёбра PRON-узла на узел-антецедент, удаляет
   PRON-узел, схлопывает дубликаты и self-loops;
4. в `Provenance` каждого затронутого ребра дописывает правило
   `anaphora_resolution_v0` и исходный pronoun_id (инвариант §9.4 «no
   silent collapse»);
5. формирует список `AnaphoraResolution` (pronoun_id, antecedent_id,
   matched feats, distance) для audit и сериализации в
   `anaphora_resolutions.jsonl`.

### Доменная модель

- `Node.token_id_in_sent: int | None` — якорь к UD-токену для надёжного
  доступа к `feats`. Проставляется в `from_clause.py` при создании
  каждого узла.
- `AnaphoraResolution` (pydantic) — запись о произведённой замене для
  сериализации и инспекции.

### Конфигурация

```yaml
anaphora:
  enabled: false
  search_window_sentences: 2
  require_animacy_match: true
  pronoun_types: ["personal_3p"]
```

## Результат

- Pipeline проходит на тестовом тексте «Иван пришёл домой. Он устал.»:
  узел «Он» исчезает, ребро `устать → Иван` заменяет `устать → он`,
  shared_entity-метарёбра становятся возможны там, где раньше связь
  терялась через PRON.
- 7 unit-тестов покрывают: базовую замену, согласование по Gender,
  фильтр Animacy, plural-местоимение `они`, неразрешимое местоимение в
  начале документа, отсечение по `search_window_sentences`,
  иммутабельность исходного графа.
- 2 интеграционных теста через `pipeline.run()` с кастомным парсером:
  включённый/выключенный режим.
- Полный набор тестов: 87 passed.

## Ограничения v0

- Только личные местоимения 3-го лица. Притяжательные (его, её, их,
  свой), возвратные (себя), указательные (этот, тот, такой) — вне
  scope.
- Антецедент — только NOUN или PROPN. Узлы-метавершины и узлы из
  предыдущих coref-кластеров пока не рассматриваются (но цепочка
  замен внутри одного запуска поддержана через `merge_map`).
- Качество разрешения сильно зависит от полноты feats у natasha
  (Animacy для «он/она/оно» проставляется не всегда — в этом случае
  фильтр Animacy просто не применяется к данной паре).
- Salience-веса (subject > object > oblique) пока не используются,
  только recency и согласование. При неоднозначностях с одинаковыми
  feats берётся ближайший по позиции кандидат.

## Следующий шаг

1. Прогнать на реальном корпусе с natasha и оценить долю успешных
   разрешений на размеченном фрагменте.
2. Добавить притяжательные `его / её / их / свой` (отдельный модуль —
   у них семантика отношения «обладатель», не «референт»).
3. Возвратное `себя` — антецедент = subject текущей клаузы, реализуется
   тривиально, но требует отдельной ветки в правилах.
4. Если recency окажется недостаточно — попробовать упрощённый
   Lappin–Leass: + веса за роль (nsubj > obj > obl) и за повторное
   упоминание.

## Затронутые файлы

- [src/metagraph_nlp/parsers/anaphora.py](../../src/metagraph_nlp/parsers/anaphora.py) (новый)
- [src/metagraph_nlp/domain/anaphora.py](../../src/metagraph_nlp/domain/anaphora.py) (новый)
- [src/metagraph_nlp/domain/graph.py](../../src/metagraph_nlp/domain/graph.py) (`Node.token_id_in_sent`)
- [src/metagraph_nlp/graph_builders/from_clause.py](../../src/metagraph_nlp/graph_builders/from_clause.py) (проставление token_id_in_sent)
- [src/metagraph_nlp/config.py](../../src/metagraph_nlp/config.py) (`AnaphoraConfig`)
- [src/metagraph_nlp/pipeline.py](../../src/metagraph_nlp/pipeline.py) (новая стадия + поле в `PipelineResult`)
- [src/metagraph_nlp/io/artifacts.py](../../src/metagraph_nlp/io/artifacts.py) (запись `anaphora_resolutions.jsonl`)
- [tests/unit/test_anaphora_resolution.py](../../tests/unit/test_anaphora_resolution.py) (новый, 7 тестов)
- [tests/integration/test_anaphora_pipeline.py](../../tests/integration/test_anaphora_pipeline.py) (новый, 2 теста)
