"""Агрегация уровня 2: метарёбра ``topic_overlap`` между L2-метавершинами.

Правило ``topic_overlap_v0`` (CLAUDE.md §5.5, §9.3). Для каждой пары
метавершин уровня 2 считается пересечение их L1-фрагментов. Если в
пересечении есть хотя бы ``min_overlap`` общих L1-метавершин, между ними
создаётся ``MetaEdge`` типа ``topic_overlap``.

На параграфной стратегии L2-метавершины партиционируют L1-клаузы
непересекающимся образом, поэтому правило детерминированно возвращает
пустой список — это осознанное поведение. Правило становится значимым,
когда появятся другие L2-стратегии (кластеризация по shared_entity,
агрегация по протагонисту), при которых L2-метавершины могут делить
L1-клаузы (аннотируемая модель этому не препятствует: анти-аннотируемость
не выполняется — см. материалы Гапанюка).
"""

from __future__ import annotations

from metagraph_nlp.domain import IdFactory, Metagraph, MetaEdge, Provenance

_STAGE = "aggregate"
_RULE = "topic_overlap_v0"


def build_topic_overlap_metaedges(
    metagraph: Metagraph,
    ids: IdFactory,
    *,
    min_overlap: int = 1,
) -> list[MetaEdge]:
    """Создать метарёбра topic_overlap между пересекающимися L2-метавершинами."""

    l2_nodes = [mn for mn in metagraph.meta_nodes if mn.level == 2]
    l2_sorted = sorted(l2_nodes, key=lambda mn: mn.id)
    doc_id: str | None = None
    for mn in l2_sorted:
        if mn.provenance.document_id is not None:
            doc_id = mn.provenance.document_id
            break

    created: list[MetaEdge] = []
    for i, a in enumerate(l2_sorted):
        a_children = set(a.fragment.meta_node_ids)
        if not a_children:
            continue
        for b in l2_sorted[i + 1 :]:
            shared = sorted(a_children & set(b.fragment.meta_node_ids))
            if len(shared) < min_overlap:
                continue
            created.append(
                MetaEdge(
                    id=ids.medge(),
                    source=a.id,
                    target=b.id,
                    relation=f"overlap_{len(shared)}",
                    type="topic_overlap",
                    level=2,
                    provenance=Provenance(
                        rule=_RULE,
                        stage=_STAGE,
                        inputs=shared,
                        document_id=doc_id,
                        notes=f"overlap_count={len(shared)}",
                    ),
                )
            )

    metagraph.meta_edges.extend(created)
    return created
