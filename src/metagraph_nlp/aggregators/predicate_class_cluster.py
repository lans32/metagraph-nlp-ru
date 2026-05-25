"""Агрегация уровня 2: кластеры клауз по классам предикатов.

Правило ``predicate_class_cluster_v0``. Опирается на
поле ``Edge.predicate_class``, заполненное в ``graph_builders.from_clause``
по словарю ``configs/predicate_classes*.yaml``.

Алгоритм:
1. Для каждой L1-метавершины (клаузы) собрать множество классов
   предикатов её рёбер.
2. Перевернуть индекс: класс → [l1_mnode_id, ...].
3. Если задана иерархия (``hierarchy is not None``) и ``levels``,
   фильтровать классы по ``hierarchy.level_of[cls] ∈ levels``.
4. Для каждого класса с size ≥ ``min_cluster_size`` создать L2-метавершину
   ``type="predicate_class"``, ``label=<имя_класса>``.
5. При ``create_containment_edges=True`` — добавить L2-метарёбра
   ``containment`` (``relation="contains"``) между parent ↔ child
   predicate-кластерами, обе из которых стали L2-метавершинами.
   Это делает дендрограмму явной.

Холархия: одна L1-клауза с глаголом, чья лемма попадает
в несколько ветвей RuWordNet, оказывается в нескольких L2-метавершинах
одновременно — на разных уровнях абстракции (leaf / mid / root).

Инварианты:
- L1-клаузы без предикатов из словаря не попадают ни в один кластер;
- ``label`` совпадает с именем семантического класса для читаемости;
- ``provenance.notes`` хранит class, level, parent, anchor_synset_id и
  список конкретных лемм кластера (трассировка до RuWordNet).
"""

from __future__ import annotations

from collections import defaultdict

from metagraph_nlp.domain import (
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaEdge,
    MetaNode,
    PredicateHierarchy,
    Provenance,
    SemanticGraph,
)

_STAGE = "aggregate"
_RULE = "predicate_class_cluster_v0"
_CONTAINMENT_TYPE = "containment"
_CONTAINMENT_RELATION = "contains"


def aggregate_predicate_class_clusters(
    metagraph: Metagraph,
    graph: SemanticGraph,
    ids: IdFactory,
    *,
    min_cluster_size: int = 2,
    hierarchy: PredicateHierarchy | None = None,
    levels: list[str] | None = None,
    create_containment_edges: bool = False,
) -> list[MetaNode]:
    """Создать L2-метавершины из кластеров клауз по predicate_class.

    Параметры:
        metagraph: текущий метаграф (мутируется — meta_nodes и meta_edges
            расширяются).
        graph: семантический граф документа.
        ids: фабрика идентификаторов.
        min_cluster_size: минимальное число L1 в кластере (по умолчанию 2).
        hierarchy: иерархия классов для multi-level фильтра и containment
            edges. None — поведение v0 (все классы как leaf, без рёбер).
        levels: какие уровни иерархии материализовать в L2 (например
            ["leaf", "mid", "root"]). None — все. Применяется только
            когда hierarchy задана.
        create_containment_edges: создавать ли L2-метарёбра contains
            между parent ↔ child predicate-кластерами.

    Возвращает список созданных метавершин (метарёбра — побочный эффект
    в ``metagraph.meta_edges``).
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

    # Фильтр по уровням иерархии (только когда hierarchy задана).
    if hierarchy is not None and levels:
        allowed_levels = set(levels)
        by_class = {
            cls: members
            for cls, members in by_class.items()
            if hierarchy.level_of.get(cls, "leaf") in allowed_levels
        }

    # class_slug → созданная mnode (нужно для containment-рёбер).
    created_by_class: dict[str, MetaNode] = {}

    for cls in sorted(by_class.keys()):
        members = sorted(by_class[cls])
        if len(members) < min_cluster_size:
            continue
        lemmas = sorted(lemmas_by_class[cls])

        # Расширенный provenance для трассировки до RuWordNet.
        if hierarchy is not None:
            level = hierarchy.level_of.get(cls, "leaf")
            parent = hierarchy.parent_of.get(cls)
            anchor_synset = hierarchy.anchor_of.get(cls, "")
            notes = (
                f"predicate_class={cls}; level={level}; parent={parent}; "
                f"anchor_synset={anchor_synset}; lemmas={lemmas}"
            )
        else:
            notes = f"predicate_class={cls}; lemmas={lemmas}"

        node = MetaNode(
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
                notes=notes,
            ),
        )
        created_by_class[cls] = node

    created = list(created_by_class.values())
    metagraph.meta_nodes.extend(created)

    # Метарёбра вложенности (containment): parent_class L2 → child_class L2 (если обе созданы).
    if create_containment_edges and hierarchy is not None and created_by_class:
        new_edges: list[MetaEdge] = []
        for child_cls, child_node in created_by_class.items():
            parent_cls = hierarchy.parent_of.get(child_cls)
            if not parent_cls:
                continue
            parent_node = created_by_class.get(parent_cls)
            if parent_node is None:
                continue
            new_edges.append(
                MetaEdge(
                    id=ids.medge(),
                    source=parent_node.id,
                    target=child_node.id,
                    relation=_CONTAINMENT_RELATION,
                    type=_CONTAINMENT_TYPE,
                    level=2,
                    provenance=Provenance(
                        rule=_RULE,
                        stage=_STAGE,
                        inputs=[parent_node.id, child_node.id],
                        document_id=doc_id,
                        notes=(
                            f"containment parent={parent_cls} → child={child_cls}"
                        ),
                    ),
                )
            )
        metagraph.meta_edges.extend(new_edges)

    return created
