"""Агрегация уровня 2: кластеризация L1-метавершин по shared entities.

Правило ``coref_cluster_v0`` (CLAUDE.md §5.2, §5.5). В отличие от
paragraph_clauses_v0, одна L1-метавершина может попасть в несколько L2-кластеров,
что делает topic_overlap метарёбра нетривиальными.

Алгоритм:
1. Построить граф связности L1-метавершин по shared_entity метарёбрам.
2. Найти connected components (union-find).
3. Каждый компонент ≥ min_cluster_size → L2-метавершина типа "coref_cluster".

Инварианты:
- L1-метавершины, не связанные shared_entity метарёбрами, не кластеризуются;
- provenance хранит список входных L1-метавершин и shared лемм.
"""

from __future__ import annotations

from collections import defaultdict

from metagraph_nlp.domain import (
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Provenance,
)

_STAGE = "aggregate"
_RULE = "coref_cluster_v0"


def _union_find(edges: list[tuple[str, str]]) -> dict[str, set[str]]:
    """Простой union-find для нахождения connected components."""
    parent: dict[str, str] = {}

    def find(x: str) -> str:
        while parent.get(x, x) != x:
            parent[x] = parent.get(parent[x], parent[x])
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb

    for a, b in edges:
        union(a, b)

    components: dict[str, set[str]] = defaultdict(set)
    all_nodes = {n for edge in edges for n in edge}
    for node in all_nodes:
        components[find(node)].add(node)

    return components


def aggregate_coref_clusters(
    metagraph: Metagraph,
    ids: IdFactory,
    *,
    min_cluster_size: int = 2,
) -> list[MetaNode]:
    """Создать L2-метавершины, группируя L1-клаузы по shared_entity связям.

    Мутирует ``metagraph.meta_nodes`` и возвращает созданные L2-метавершины.
    """
    l1_ids = {mn.id for mn in metagraph.meta_nodes if mn.level == 1}

    shared_edges: list[tuple[str, str]] = []
    shared_lemmas: dict[tuple[str, str], list[str]] = defaultdict(list)
    for me in metagraph.meta_edges:
        if me.type != "shared_entity":
            continue
        if me.source in l1_ids and me.target in l1_ids:
            pair = (min(me.source, me.target), max(me.source, me.target))
            shared_edges.append(pair)
            shared_lemmas[pair].append(me.relation)

    if not shared_edges:
        return []

    components = _union_find(shared_edges)

    doc_id = None
    for mn in metagraph.meta_nodes:
        if mn.provenance.document_id:
            doc_id = mn.provenance.document_id
            break

    created: list[MetaNode] = []
    for _root, members in sorted(components.items(), key=lambda kv: sorted(kv[1])):
        if len(members) < min_cluster_size:
            continue
        sorted_members = sorted(members)

        lemma_set: set[str] = set()
        for i, a in enumerate(sorted_members):
            for b in sorted_members[i + 1:]:
                pair = (min(a, b), max(a, b))
                lemma_set.update(shared_lemmas.get(pair, []))

        fragment = GraphFragment(meta_node_ids=sorted_members)
        created.append(
            MetaNode(
                id=ids.mnode(),
                type="coref_cluster",
                level=2,
                label=f"cluster_{len(created)}",
                fragment=fragment,
                provenance=Provenance(
                    rule=_RULE,
                    stage=_STAGE,
                    inputs=sorted_members,
                    document_id=doc_id,
                    notes=f"shared_lemmas={sorted(lemma_set)}",
                ),
            )
        )

    metagraph.meta_nodes.extend(created)
    return created
