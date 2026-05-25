"""Содержательные labels для L2-метавершин.

Утилита ``dominant_lemma_label`` выбирает самую частую значимую лемму
во фрагменте L2 (та же фильтрация значимых узлов, что и в правиле
``shared_entity_by_lemma_v0``). Используется агрегаторами L2, у
которых нет естественного семантического имени (paragraph, entity_cluster).

Для агрегатора ``predicate_class_cluster_v0`` label — это сам класс
(motion, communication, ...), и эта утилита там не нужна.

Детерминизм:
- частоты считаются по графу значимых узлов;
- при равной частоте побеждает лексикографически меньшая лемма;
- при пустом значимом множестве fallback на label первой L1-метавершины
  во фрагменте (например, head_text клаузы).
"""

from __future__ import annotations

from collections import Counter

from metagraph_nlp.domain import Metagraph, SemanticGraph

_DEFAULT_EXCLUDE_UPOS: frozenset[str] = frozenset(
    {"PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}
)


def dominant_lemma_label(
    l1_meta_node_ids: list[str],
    metagraph: Metagraph,
    graph: SemanticGraph,
    *,
    min_lemma_len: int = 3,
    exclude_upos: frozenset[str] = _DEFAULT_EXCLUDE_UPOS,
) -> str:
    """Самая частая значимая лемма во фрагменте L2 (детерминированно).

    ``l1_meta_node_ids`` — идентификаторы L1-метавершин, формирующих фрагмент.
    Их ``fragment.node_ids`` указывают на узлы ``graph.nodes``, по леммам
    которых считается частота. Возвращает строку (никогда не пустую).
    """

    l1_by_id = {mn.id: mn for mn in metagraph.meta_nodes if mn.level == 1}

    target_node_ids: set[str] = set()
    for l1_id in l1_meta_node_ids:
        l1 = l1_by_id.get(l1_id)
        if l1 is None:
            continue
        target_node_ids.update(l1.fragment.node_ids)

    counter: Counter[str] = Counter()
    for n in graph.nodes:
        if n.id not in target_node_ids:
            continue
        if n.kind == "predicate":
            continue
        if n.upos in exclude_upos:
            continue
        lemma = n.lemma
        if lemma is None or len(lemma) < min_lemma_len:
            continue
        counter[lemma] += 1

    if counter:
        # Тай-брейк: выше частота → меньше лемма лексикографически.
        winner = min(counter.items(), key=lambda kv: (-kv[1], kv[0]))
        return winner[0]

    # Запасной вариант: label первой L1-метавершины в отсортированном порядке.
    fallback_ids = sorted(l1_meta_node_ids)
    for l1_id in fallback_ids:
        l1 = l1_by_id.get(l1_id)
        if l1 and l1.label:
            return l1.label

    # Крайний случай: ни одной L1 с label. Возвращаем литерал, чтобы
    # не нарушать инвариант non-empty label.
    return "(empty)"
