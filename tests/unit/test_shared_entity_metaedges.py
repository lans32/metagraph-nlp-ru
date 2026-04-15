"""Юнит-тесты агрегатора shared_entity_metaedges."""

from __future__ import annotations

from metagraph_nlp.aggregators import build_shared_entity_metaedges
from metagraph_nlp.domain import (
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Node,
    Provenance,
    SemanticGraph,
)


def _make_fixture():
    ids = IdFactory()
    doc_id = ids.doc()

    cid_a = "clause_000001"
    cid_b = "clause_000002"
    cid_c = "clause_000003"

    prov = Provenance(rule="test", stage="test", document_id=doc_id)

    nodes = [
        Node(id=ids.node(), label="студент", lemma="студент", upos="NOUN", clause_id=cid_a, provenance=prov),
        Node(id=ids.node(), label="книга",   lemma="книга",   upos="NOUN", clause_id=cid_a, provenance=prov),
        Node(id=ids.node(), label="преподаватель", lemma="преподаватель", upos="NOUN", clause_id=cid_b, provenance=prov),
        Node(id=ids.node(), label="книга",   lemma="книга",   upos="NOUN", clause_id=cid_b, provenance=prov),
        Node(id=ids.node(), label="студент", lemma="студент", upos="NOUN", clause_id=cid_c, provenance=prov),
        Node(id=ids.node(), label="теорема", lemma="теорема", upos="NOUN", clause_id=cid_c, provenance=prov),
    ]
    graph = SemanticGraph(document_id=doc_id, nodes=nodes)

    mn_a = MetaNode(
        id=ids.mnode(), type="clause", level=1, fragment=GraphFragment(),
        provenance=Provenance(rule="test", stage="test", clause_id=cid_a),
    )
    mn_b = MetaNode(
        id=ids.mnode(), type="clause", level=1, fragment=GraphFragment(),
        provenance=Provenance(rule="test", stage="test", clause_id=cid_b),
    )
    mn_c = MetaNode(
        id=ids.mnode(), type="clause", level=1, fragment=GraphFragment(),
        provenance=Provenance(rule="test", stage="test", clause_id=cid_c),
    )
    metagraph = Metagraph(document_id=doc_id, meta_nodes=[mn_a, mn_b, mn_c])

    return metagraph, graph, ids, (mn_a, mn_b, mn_c)


def test_creates_expected_shared_entity_edges():
    metagraph, graph, ids, (mn_a, mn_b, mn_c) = _make_fixture()

    created = build_shared_entity_metaedges(metagraph, graph, ids)

    # Ожидаем два метаребра:
    # (A-B, книга) и (A-C, студент); B-C общих лемм не имеют.
    assert len(created) == 2

    by_pair = {(me.source, me.target, me.relation): me for me in created}
    assert (mn_a.id, mn_b.id, "книга") in by_pair
    assert (mn_a.id, mn_c.id, "студент") in by_pair

    # Нет B-C.
    assert not any(me.source == mn_b.id and me.target == mn_c.id for me in created)

    for me in created:
        assert me.type == "shared_entity"
        assert me.level == 1
        assert me.provenance.rule == "shared_entity_by_lemma_v0"
        assert me.provenance.stage == "aggregate"
        assert me.provenance.notes == f"lemma={me.relation}"
        assert me.provenance.inputs == sorted(me.provenance.inputs)
        assert len(me.provenance.inputs) >= 2


def test_pron_is_excluded():
    metagraph, graph, ids, (mn_a, mn_b, _mn_c) = _make_fixture()

    prov = Provenance(rule="test", stage="test")
    graph.nodes.append(
        Node(id=ids.node(), label="он", lemma="он", upos="PRON",
             clause_id="clause_000001", provenance=prov)
    )
    graph.nodes.append(
        Node(id=ids.node(), label="он", lemma="он", upos="PRON",
             clause_id="clause_000002", provenance=prov)
    )

    created = build_shared_entity_metaedges(metagraph, graph, ids)

    assert all(me.relation != "он" for me in created)


def test_short_lemmas_are_filtered():
    metagraph, graph, ids, _mns = _make_fixture()

    prov = Provenance(rule="test", stage="test")
    graph.nodes.append(
        Node(id=ids.node(), label="я", lemma="я", upos="NOUN",
             clause_id="clause_000001", provenance=prov)
    )
    graph.nodes.append(
        Node(id=ids.node(), label="я", lemma="я", upos="NOUN",
             clause_id="clause_000002", provenance=prov)
    )

    created = build_shared_entity_metaedges(metagraph, graph, ids, min_lemma_len=3)

    assert all(me.relation != "я" for me in created)


def test_determinism_across_runs():
    metagraph1, graph1, ids1, _ = _make_fixture()
    metagraph2, graph2, ids2, _ = _make_fixture()

    created1 = build_shared_entity_metaedges(metagraph1, graph1, ids1)
    created2 = build_shared_entity_metaedges(metagraph2, graph2, ids2)

    rels1 = [me.relation for me in created1]
    rels2 = [me.relation for me in created2]
    assert rels1 == rels2

    pairs1 = [(me.source, me.target, me.relation) for me in created1]
    pairs2 = [(me.source, me.target, me.relation) for me in created2]
    assert pairs1 == pairs2


def test_returned_edges_are_also_in_metagraph():
    metagraph, graph, ids, _ = _make_fixture()
    assert metagraph.meta_edges == []

    created = build_shared_entity_metaedges(metagraph, graph, ids)

    assert len(created) > 0
    assert metagraph.meta_edges == created
    # Повторный вызов возвращает только новые (а в метаграфе они добавляются).
    created2 = build_shared_entity_metaedges(metagraph, graph, ids)
    assert len(metagraph.meta_edges) == len(created) + len(created2)
