"""Юнит-тесты predicate_class_cluster_v0."""

from __future__ import annotations

from metagraph_nlp.aggregators.predicate_class_cluster import (
    aggregate_predicate_class_clusters,
)
from metagraph_nlp.domain import (
    Edge,
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Provenance,
    SemanticGraph,
)

DOC = "doc-1"
_PROV = Provenance(rule="t", stage="t", document_id=DOC)


def _l1(id: str, edge_ids: list[str]) -> MetaNode:
    return MetaNode(
        id=id, type="clause:main", level=1, label=id,
        fragment=GraphFragment(edge_ids=edge_ids),
        provenance=_PROV,
    )


def _edge(id: str, relation: str, classes: list[str] | None) -> Edge:
    return Edge(
        id=id, source="s", target="t", relation=relation,
        kind="predicate", clause_id="c",
        predicate_class=classes,
        provenance=_PROV,
    )


def test_two_clauses_with_same_class_form_cluster():
    mg = Metagraph(document_id=DOC, meta_nodes=[
        _l1("mn-1", ["e1"]),
        _l1("mn-2", ["e2"]),
    ])
    graph = SemanticGraph(document_id=DOC, edges=[
        _edge("e1", "сказать", ["communication"]),
        _edge("e2", "ответить", ["communication"]),
    ])
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(mg, graph, ids, min_cluster_size=2)

    assert len(created) == 1
    cluster = created[0]
    assert cluster.level == 2
    assert cluster.type == "predicate_class"
    assert cluster.label == "communication"
    assert set(cluster.fragment.meta_node_ids) == {"mn-1", "mn-2"}
    assert "сказать" in cluster.provenance.notes
    assert "ответить" in cluster.provenance.notes


def test_single_clause_does_not_form_cluster():
    mg = Metagraph(document_id=DOC, meta_nodes=[_l1("mn-1", ["e1"])])
    graph = SemanticGraph(document_id=DOC, edges=[
        _edge("e1", "сказать", ["communication"]),
    ])
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(mg, graph, ids, min_cluster_size=2)
    assert created == []


def test_three_classes_three_clusters():
    mg = Metagraph(document_id=DOC, meta_nodes=[
        _l1("mn-1", ["e1a", "e1b"]),
        _l1("mn-2", ["e2"]),
        _l1("mn-3", ["e3"]),
        _l1("mn-4", ["e4"]),
    ])
    graph = SemanticGraph(document_id=DOC, edges=[
        _edge("e1a", "сказать", ["communication"]),
        _edge("e1b", "идти", ["motion"]),
        _edge("e2", "ответить", ["communication"]),
        _edge("e3", "бежать", ["motion"]),
        _edge("e4", "помнить", ["cognition"]),
    ])
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(mg, graph, ids, min_cluster_size=2)

    by_label = {mn.label: mn for mn in created}
    assert "communication" in by_label
    assert "motion" in by_label
    assert "cognition" not in by_label  # размер 1
    assert set(by_label["communication"].fragment.meta_node_ids) == {"mn-1", "mn-2"}
    assert set(by_label["motion"].fragment.meta_node_ids) == {"mn-1", "mn-3"}


def test_holarchy_one_l1_in_multiple_clusters():
    mg = Metagraph(document_id=DOC, meta_nodes=[
        _l1("mn-1", ["e1a", "e1b"]),
        _l1("mn-2", ["e2"]),
        _l1("mn-3", ["e3"]),
    ])
    graph = SemanticGraph(document_id=DOC, edges=[
        _edge("e1a", "сказать", ["communication"]),
        _edge("e1b", "идти", ["motion"]),
        _edge("e2", "ответить", ["communication"]),
        _edge("e3", "бежать", ["motion"]),
    ])
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(mg, graph, ids, min_cluster_size=2)

    in_clusters = [
        cluster.label for cluster in created
        if "mn-1" in cluster.fragment.meta_node_ids
    ]
    # mn-1 попадает и в communication, и в motion (холархия).
    assert set(in_clusters) == {"communication", "motion"}


def test_edges_without_predicate_class_ignored():
    mg = Metagraph(document_id=DOC, meta_nodes=[
        _l1("mn-1", ["e1"]),
        _l1("mn-2", ["e2"]),
    ])
    graph = SemanticGraph(document_id=DOC, edges=[
        _edge("e1", "редактировать", None),
        _edge("e2", "клацать", None),
    ])
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(mg, graph, ids, min_cluster_size=2)
    assert created == []


def test_metagraph_meta_nodes_mutated():
    mg = Metagraph(document_id=DOC, meta_nodes=[
        _l1("mn-1", ["e1"]),
        _l1("mn-2", ["e2"]),
    ])
    graph = SemanticGraph(document_id=DOC, edges=[
        _edge("e1", "сказать", ["communication"]),
        _edge("e2", "ответить", ["communication"]),
    ])
    ids = IdFactory()
    before = len(mg.meta_nodes)
    created = aggregate_predicate_class_clusters(mg, graph, ids, min_cluster_size=2)
    assert len(mg.meta_nodes) == before + len(created)
