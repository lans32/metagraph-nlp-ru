"""Юнит-тесты агрегатора topic_overlap_v0."""

from __future__ import annotations

from metagraph_nlp.aggregators import build_topic_overlap_metaedges
from metagraph_nlp.domain import (
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Provenance,
)

DOC_ID = "doc_test"


def _l2_metanode(ids: IdFactory, l1_ids: list[str], idx: int) -> MetaNode:
    return MetaNode(
        id=ids.mnode(),
        type="paragraph",
        level=2,
        label=f"p{idx}",
        fragment=GraphFragment(meta_node_ids=l1_ids),
        provenance=Provenance(
            rule="paragraph_clauses_v0",
            stage="aggregate",
            document_id=DOC_ID,
            notes=f"paragraph_index={idx}",
        ),
    )


def test_no_overlap_for_disjoint_l2_fragments():
    """Параграфная стратегия партиционирует клаузы — рёбер быть не должно."""
    ids = IdFactory()
    mg = Metagraph(document_id=DOC_ID)
    mg.meta_nodes.extend([
        _l2_metanode(ids, ["mnode_A1", "mnode_A2"], 0),
        _l2_metanode(ids, ["mnode_B1"], 1),
        _l2_metanode(ids, ["mnode_C1", "mnode_C2"], 2),
    ])

    created = build_topic_overlap_metaedges(mg, ids)

    assert created == []
    assert [me for me in mg.meta_edges if me.type == "topic_overlap"] == []


def test_overlap_creates_single_edge():
    ids = IdFactory()
    mg = Metagraph(document_id=DOC_ID)
    a = _l2_metanode(ids, ["mnode_001", "mnode_002"], 0)
    b = _l2_metanode(ids, ["mnode_002", "mnode_003"], 1)
    mg.meta_nodes.extend([a, b])

    created = build_topic_overlap_metaedges(mg, ids)

    assert len(created) == 1
    me = created[0]
    assert me.type == "topic_overlap"
    assert me.level == 2
    assert me.relation == "overlap_1"
    assert {me.source, me.target} == {a.id, b.id}
    assert me.provenance.rule == "topic_overlap_v0"
    assert me.provenance.inputs == ["mnode_002"]
    assert me.provenance.notes == "overlap_count=1"


def test_overlap_count_in_relation():
    ids = IdFactory()
    mg = Metagraph(document_id=DOC_ID)
    a = _l2_metanode(ids, ["mnode_001", "mnode_002", "mnode_003"], 0)
    b = _l2_metanode(ids, ["mnode_002", "mnode_003", "mnode_004"], 1)
    mg.meta_nodes.extend([a, b])

    created = build_topic_overlap_metaedges(mg, ids)

    assert len(created) == 1
    assert created[0].relation == "overlap_2"
    assert created[0].provenance.inputs == ["mnode_002", "mnode_003"]


def test_min_overlap_threshold():
    ids = IdFactory()
    mg = Metagraph(document_id=DOC_ID)
    a = _l2_metanode(ids, ["mnode_001", "mnode_002"], 0)
    b = _l2_metanode(ids, ["mnode_002", "mnode_003"], 1)
    mg.meta_nodes.extend([a, b])

    created = build_topic_overlap_metaedges(mg, ids, min_overlap=2)

    assert created == []


def test_l1_only_metagraph_produces_nothing():
    ids = IdFactory()
    mg = Metagraph(document_id=DOC_ID)
    mg.meta_nodes.append(
        MetaNode(
            id=ids.mnode(),
            type="clause",
            level=1,
            fragment=GraphFragment(node_ids=["n1", "n2"]),
            provenance=Provenance(rule="test", stage="aggregate"),
        )
    )

    created = build_topic_overlap_metaedges(mg, ids)

    assert created == []


def test_mutates_metagraph_meta_edges():
    ids = IdFactory()
    mg = Metagraph(document_id=DOC_ID)
    a = _l2_metanode(ids, ["mnode_A"], 0)
    b = _l2_metanode(ids, ["mnode_A"], 1)
    mg.meta_nodes.extend([a, b])
    assert mg.meta_edges == []

    created = build_topic_overlap_metaedges(mg, ids)

    assert len(created) == 1
    assert mg.meta_edges == created
