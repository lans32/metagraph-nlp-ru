"""Тесты для coref_cluster_metanodes: L2-кластеризация по shared_entity."""

from __future__ import annotations

from metagraph_nlp.aggregators.coref_cluster_metanodes import (
    aggregate_coref_clusters,
)
from metagraph_nlp.domain import (
    GraphFragment,
    IdFactory,
    MetaEdge,
    Metagraph,
    MetaNode,
    Provenance,
)

_PROV = Provenance(rule="test", stage="test", inputs=[], document_id="doc-1")


def _mn(id: str, level: int = 1, type: str = "clause:main") -> MetaNode:
    return MetaNode(
        id=id, type=type, level=level, label=id,
        fragment=GraphFragment(node_ids=[f"n-{id}"]),
        provenance=_PROV,
    )


def _me(id: str, src: str, tgt: str, relation: str = "лемма") -> MetaEdge:
    return MetaEdge(
        id=id, source=src, target=tgt,
        relation=relation, type="shared_entity", level=1,
        provenance=_PROV,
    )


def test_two_connected_l1_nodes_form_one_cluster():
    mg = Metagraph(
        document_id="doc-1",
        meta_nodes=[_mn("mn-1"), _mn("mn-2")],
        meta_edges=[_me("me-1", "mn-1", "mn-2", "кот")],
    )
    ids = IdFactory()
    created = aggregate_coref_clusters(mg, ids, min_cluster_size=2)

    assert len(created) == 1
    cluster = created[0]
    assert cluster.level == 2
    assert cluster.type == "coref_cluster"
    assert set(cluster.fragment.meta_node_ids) == {"mn-1", "mn-2"}
    assert len(mg.meta_nodes) == 3  # 2 L1 + 1 L2


def test_separate_components_form_separate_clusters():
    mg = Metagraph(
        document_id="doc-1",
        meta_nodes=[_mn("mn-1"), _mn("mn-2"), _mn("mn-3"), _mn("mn-4")],
        meta_edges=[
            _me("me-1", "mn-1", "mn-2", "кот"),
            _me("me-2", "mn-3", "mn-4", "дом"),
        ],
    )
    ids = IdFactory()
    created = aggregate_coref_clusters(mg, ids, min_cluster_size=2)

    assert len(created) == 2
    member_sets = [set(c.fragment.meta_node_ids) for c in created]
    assert {"mn-1", "mn-2"} in member_sets
    assert {"mn-3", "mn-4"} in member_sets


def test_single_node_not_clustered():
    mg = Metagraph(
        document_id="doc-1",
        meta_nodes=[_mn("mn-1"), _mn("mn-2")],
        meta_edges=[],
    )
    ids = IdFactory()
    created = aggregate_coref_clusters(mg, ids, min_cluster_size=2)
    assert created == []


def test_min_cluster_size_filters_small_components():
    mg = Metagraph(
        document_id="doc-1",
        meta_nodes=[_mn("mn-1"), _mn("mn-2"), _mn("mn-3")],
        meta_edges=[
            _me("me-1", "mn-1", "mn-2", "кот"),
        ],
    )
    ids = IdFactory()
    created = aggregate_coref_clusters(mg, ids, min_cluster_size=3)
    assert created == []


def test_chain_of_three_forms_one_cluster():
    mg = Metagraph(
        document_id="doc-1",
        meta_nodes=[_mn("mn-1"), _mn("mn-2"), _mn("mn-3")],
        meta_edges=[
            _me("me-1", "mn-1", "mn-2", "кот"),
            _me("me-2", "mn-2", "mn-3", "кот"),
        ],
    )
    ids = IdFactory()
    created = aggregate_coref_clusters(mg, ids, min_cluster_size=2)

    assert len(created) == 1
    assert set(created[0].fragment.meta_node_ids) == {"mn-1", "mn-2", "mn-3"}


def test_non_shared_entity_edges_ignored():
    mg = Metagraph(
        document_id="doc-1",
        meta_nodes=[_mn("mn-1"), _mn("mn-2")],
        meta_edges=[
            MetaEdge(
                id="me-other", source="mn-1", target="mn-2",
                relation="topic", type="topic_overlap", level=2,
                provenance=_PROV,
            ),
        ],
    )
    ids = IdFactory()
    created = aggregate_coref_clusters(mg, ids, min_cluster_size=2)
    assert created == []
