"""Лингвистическая агрегация: каждая клауза → MetaNode уровня 1.

Это базовая стратегия из CLAUDE.md §4.1 и §5.1. Для каждой клаузы собираются
все узлы и рёбра графа, привязанные к этой клаузе (по clause_id), и
оборачиваются в GraphFragment внутри MetaNode. Метарёбра на первом проходе
не создаются — это задача следующей итерации (structural/linguistic).

Инварианты:
- каждая клауза отражена ровно одной метавершиной уровня 1;
- ни один узел/ребро не теряется «молча» (§9.4): если к клаузе не привязано
  ничего, создаётся пустой фрагмент с явной пометкой в provenance.
"""

from __future__ import annotations

from collections import defaultdict

from metagraph_nlp.domain import (
    Clause,
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Provenance,
    SemanticGraph,
)

_STAGE = "aggregate"
_RULE = "clause_as_metanode_v0"


def aggregate_clauses_to_metanodes(
    graph: SemanticGraph,
    clauses: list[Clause],
    ids: IdFactory,
) -> Metagraph:
    nodes_by_clause: dict[str, list[str]] = defaultdict(list)
    edges_by_clause: dict[str, list[str]] = defaultdict(list)

    for n in graph.nodes:
        if n.clause_id is not None:
            nodes_by_clause[n.clause_id].append(n.id)
    for e in graph.edges:
        if e.clause_id is not None:
            edges_by_clause[e.clause_id].append(e.id)

    mg = Metagraph(document_id=graph.document_id)

    for clause in clauses:
        fragment = GraphFragment(
            node_ids=nodes_by_clause.get(clause.id, []),
            edge_ids=edges_by_clause.get(clause.id, []),
        )
        empty = not fragment.node_ids and not fragment.edge_ids
        mg.meta_nodes.append(
            MetaNode(
                id=ids.mnode(),
                type="clause",
                level=1,
                label=clause.head_text or clause.span.text[:60],
                fragment=fragment,
                provenance=Provenance(
                    rule=_RULE,
                    stage=_STAGE,
                    inputs=[clause.id, *fragment.node_ids, *fragment.edge_ids],
                    document_id=clause.document_id,
                    sentence_id=clause.sentence_id,
                    clause_id=clause.id,
                    notes="empty fragment" if empty else None,
                ),
            )
        )

    return mg
