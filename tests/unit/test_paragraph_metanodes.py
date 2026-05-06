"""Юнит-тесты агрегатора paragraph_clauses_v0."""

from __future__ import annotations

from metagraph_nlp.aggregators import aggregate_clauses_to_paragraphs
from metagraph_nlp.domain import (
    Clause,
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Provenance,
    SemanticGraph,
    Sentence,
    TextSpan,
)

DOC_ID = "doc_test"


def _sent(ids: IdFactory, text: str, start: int, paragraph_index: int) -> Sentence:
    end = start + len(text)
    return Sentence(
        id=ids.sent(),
        document_id=DOC_ID,
        index=0,
        span=TextSpan(start=start, end=end, text=text),
        paragraph_index=paragraph_index,
        provenance=Provenance(rule="test", stage="segment", document_id=DOC_ID),
    )


def _clause(ids: IdFactory, sent_id: str, text: str, start: int) -> Clause:
    end = start + len(text)
    return Clause(
        id=ids.clause(),
        sentence_id=sent_id,
        document_id=DOC_ID,
        span=TextSpan(start=start, end=end, text=text),
        provenance=Provenance(
            rule="test",
            stage="clauses",
            document_id=DOC_ID,
            sentence_id=sent_id,
        ),
    )


def _l1_metanode(ids: IdFactory, clause: Clause, label: str = "head") -> MetaNode:
    return MetaNode(
        id=ids.mnode(),
        type="clause",
        level=1,
        label=label,
        fragment=GraphFragment(),
        provenance=Provenance(
            rule="clause_as_metanode_v0",
            stage="aggregate",
            document_id=DOC_ID,
            sentence_id=clause.sentence_id,
            clause_id=clause.id,
        ),
    )


def _empty_graph() -> SemanticGraph:
    return SemanticGraph(document_id=DOC_ID, nodes=[], edges=[])


def _fixture_three_paragraphs():
    ids = IdFactory()
    # p0: s0 (1 clause), s1 (1 clause); p1: s2 (2 clauses); p2: s3 (1 clause).
    s0 = _sent(ids, "s0.", 0, 0)
    s1 = _sent(ids, "s1.", 4, 0)
    s2 = _sent(ids, "s2.", 8, 1)
    s3 = _sent(ids, "s3.", 12, 2)

    c0 = _clause(ids, s0.id, "s0.", 0)
    c1 = _clause(ids, s1.id, "s1.", 4)
    c2a = _clause(ids, s2.id, "s2a", 8)
    c2b = _clause(ids, s2.id, "s2b", 10)
    c3 = _clause(ids, s3.id, "s3.", 12)
    clauses = [c0, c1, c2a, c2b, c3]

    mg = Metagraph(document_id=DOC_ID)
    for c in clauses:
        mg.meta_nodes.append(_l1_metanode(ids, c))

    return mg, clauses, [s0, s1, s2, s3], ids


def test_creates_one_l2_per_paragraph():
    mg, clauses, sentences, ids = _fixture_three_paragraphs()

    created = aggregate_clauses_to_paragraphs(
        mg, _empty_graph(), clauses, sentences, ids
    )

    assert len(created) == 3
    assert all(mn.level == 2 for mn in created)
    assert all(mn.type == "paragraph" for mn in created)
    # Label вида "§<index>: <topic>". При пустом графе topic — label первой L1.
    for i, mn in enumerate(created):
        assert mn.label.startswith(f"§{i}:")


def test_l2_fragment_references_l1_metanodes():
    mg, clauses, sentences, ids = _fixture_three_paragraphs()

    l1_ids_by_clause = {mn.provenance.clause_id: mn.id for mn in mg.meta_nodes}
    created = aggregate_clauses_to_paragraphs(
        mg, _empty_graph(), clauses, sentences, ids
    )

    mp0, mp1, mp2 = created
    assert set(mp0.fragment.meta_node_ids) == {
        l1_ids_by_clause[clauses[0].id],
        l1_ids_by_clause[clauses[1].id],
    }
    assert set(mp1.fragment.meta_node_ids) == {
        l1_ids_by_clause[clauses[2].id],
        l1_ids_by_clause[clauses[3].id],
    }
    assert set(mp2.fragment.meta_node_ids) == {
        l1_ids_by_clause[clauses[4].id],
    }
    for mp in created:
        assert mp.fragment.node_ids == []
        assert mp.fragment.edge_ids == []


def test_provenance_has_paragraph_index():
    mg, clauses, sentences, ids = _fixture_three_paragraphs()

    created = aggregate_clauses_to_paragraphs(
        mg, _empty_graph(), clauses, sentences, ids
    )

    for i, mp in enumerate(created):
        assert mp.provenance.rule == "paragraph_clauses_v0"
        assert mp.provenance.stage == "aggregate"
        assert mp.provenance.notes == f"paragraph_index={i}"
        assert mp.provenance.document_id == DOC_ID
        # inputs содержат L1-mnode ids + sentence ids.
        assert len(mp.provenance.inputs) >= 1


def test_single_paragraph_single_metanode():
    ids = IdFactory()
    s0 = _sent(ids, "s0.", 0, 0)
    s1 = _sent(ids, "s1.", 4, 0)
    clauses = [
        _clause(ids, s0.id, "s0.", 0),
        _clause(ids, s1.id, "s1.", 4),
    ]
    mg = Metagraph(document_id=DOC_ID)
    for c in clauses:
        mg.meta_nodes.append(_l1_metanode(ids, c))

    created = aggregate_clauses_to_paragraphs(
        mg, _empty_graph(), clauses, [s0, s1], ids
    )

    assert len(created) == 1
    assert len(created[0].fragment.meta_node_ids) == 2


def test_mutates_metagraph_meta_nodes():
    mg, clauses, sentences, ids = _fixture_three_paragraphs()
    l1_count = len(mg.meta_nodes)

    created = aggregate_clauses_to_paragraphs(
        mg, _empty_graph(), clauses, sentences, ids
    )

    assert len(mg.meta_nodes) == l1_count + len(created)
    assert mg.meta_nodes[-len(created):] == created


def test_determinism_across_runs():
    mg1, clauses1, sentences1, ids1 = _fixture_three_paragraphs()
    mg2, clauses2, sentences2, ids2 = _fixture_three_paragraphs()

    created1 = aggregate_clauses_to_paragraphs(
        mg1, _empty_graph(), clauses1, sentences1, ids1
    )
    created2 = aggregate_clauses_to_paragraphs(
        mg2, _empty_graph(), clauses2, sentences2, ids2
    )

    assert [mn.label for mn in created1] == [mn.label for mn in created2]
    assert [mn.fragment.meta_node_ids for mn in created1] == [
        mn.fragment.meta_node_ids for mn in created2
    ]


def test_label_uses_dominant_lemma_when_graph_has_significant_nodes():
    """Когда граф непустой — label включает доминирующую лемму."""
    from metagraph_nlp.domain import Node

    ids = IdFactory()
    s0 = _sent(ids, "s0.", 0, 0)
    c0 = _clause(ids, s0.id, "s0.", 0)
    c1 = _clause(ids, s0.id, "s1.", 4)
    clauses = [c0, c1]

    mg = Metagraph(document_id=DOC_ID)
    mn0 = _l1_metanode(ids, c0)
    mn1 = _l1_metanode(ids, c1)
    # Привяжем фрагменты L1 к узлам графа явно.
    mn0.fragment.node_ids = ["nA"]
    mn1.fragment.node_ids = ["nB"]
    mg.meta_nodes.extend([mn0, mn1])

    prov = Provenance(rule="t", stage="t", document_id=DOC_ID)
    nodes = [
        Node(id="nA", label="кот", lemma="кот", upos="NOUN",
             clause_id=c0.id, provenance=prov),
        Node(id="nB", label="кот", lemma="кот", upos="NOUN",
             clause_id=c1.id, provenance=prov),
    ]
    graph = SemanticGraph(document_id=DOC_ID, nodes=nodes, edges=[])

    created = aggregate_clauses_to_paragraphs(mg, graph, clauses, [s0], ids)

    assert len(created) == 1
    assert created[0].label == "§0: кот"
