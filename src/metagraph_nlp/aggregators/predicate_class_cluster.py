"""Агрегация уровня 2: кластеры клауз по классам предикатов.

Правило ``predicate_class_cluster_v0`` (CLAUDE.md §5.5). Опирается на
поле ``Edge.predicate_class``, заполненное в ``graph_builders.from_clause``
по словарю ``configs/predicate_classes.yaml``.

Алгоритм:
1. Для каждой L1-метавершины (клаузы) собрать множество классов
   предикатов её рёбер.
2. Перевернуть индекс: класс → [l1_mnode_id, ...].
3. Для каждого класса с size ≥ min_cluster_size создать L2-метавершину
   ``type="predicate_class"``, ``label=<имя_класса>``.

Холархия: одна L1-клауза с двумя глаголами разных классов попадает в
обе соответствующие L2-метавершины (поддерживается моделью, см.
CLAUDE.md §4.4 «дендрограмма»).

Инварианты:
- L1-клаузы без предикатов из словаря не попадают ни в один кластер;
- ``label`` совпадает с именем семантического класса, что обеспечивает
  читаемость метаграфа без дополнительных утилит;
- ``provenance.notes`` хранит список конкретных лемм, попавших в кластер.
"""

from __future__ import annotations

from collections import defaultdict

from metagraph_nlp.domain import (
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Provenance,
    SemanticGraph,
)

_STAGE = "aggregate"
_RULE = "predicate_class_cluster_v0"


def aggregate_predicate_class_clusters(
    metagraph: Metagraph,
    graph: SemanticGraph,
    ids: IdFactory,
    *,
    min_cluster_size: int = 2,
) -> list[MetaNode]:
    """Создать L2-метавершины из кластеров клауз по predicate_class.

    Мутирует ``metagraph.meta_nodes`` и возвращает созданные метавершины.
    """

    edge_index = {e.id: e for e in graph.edges}

    # class → set[l1_mnode_id]; class → set[lemma] (для provenance.notes).
    by_class: dict[str, set[str]] = defaultdict(set)
    lemmas_by_class: dict[str, set[str]] = defaultdict(set)

    doc_id: str | None = None
    for mn in metagraph.meta_nodes:
        if mn.level != 1:
            continue
        if mn.provenance.document_id and doc_id is None:
            doc_id = mn.provenance.document_id
        for eid in mn.fragment.edge_ids:
            edge = edge_index.get(eid)
            if edge is None or not edge.predicate_class:
                continue
            for cls in edge.predicate_class:
                by_class[cls].add(mn.id)
                # `relation` хранит лемму предиката (возможно с предлогом
                # через подчёркивание); для notes берём первую часть.
                base_lemma = edge.relation.split("_", 1)[0]
                lemmas_by_class[cls].add(base_lemma)

    created: list[MetaNode] = []
    for cls in sorted(by_class.keys()):
        members = sorted(by_class[cls])
        if len(members) < min_cluster_size:
            continue
        lemmas = sorted(lemmas_by_class[cls])
        created.append(
            MetaNode(
                id=ids.mnode(),
                type="predicate_class",
                level=2,
                label=cls,
                fragment=GraphFragment(meta_node_ids=members),
                provenance=Provenance(
                    rule=_RULE,
                    stage=_STAGE,
                    inputs=members,
                    document_id=doc_id,
                    notes=f"predicate_class={cls}; lemmas={lemmas}",
                ),
            )
        )

    metagraph.meta_nodes.extend(created)
    return created
