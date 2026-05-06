"""Юнит-тесты dominant_lemma_label."""

from __future__ import annotations

from metagraph_nlp.aggregators._label_utils import dominant_lemma_label
from metagraph_nlp.domain import (
    GraphFragment,
    Metagraph,
    MetaNode,
    Node,
    Provenance,
    SemanticGraph,
)

_PROV = Provenance(rule="test", stage="test", document_id="doc-1")


def _l1(id: str, node_ids: list[str], label: str = "head") -> MetaNode:
    return MetaNode(
        id=id, type="clause:main", level=1, label=label,
        fragment=GraphFragment(node_ids=node_ids),
        provenance=_PROV,
    )


def _node(id: str, lemma: str, upos: str = "NOUN", kind: str = "concept") -> Node:
    return Node(
        id=id, label=lemma, lemma=lemma, upos=upos, kind=kind,
        provenance=_PROV,
    )


def test_dominant_lemma_picks_most_frequent():
    mg = Metagraph(document_id="doc-1", meta_nodes=[
        _l1("mn-1", ["n1", "n2"]),
        _l1("mn-2", ["n3", "n4"]),
    ])
    graph = SemanticGraph(document_id="doc-1", nodes=[
        _node("n1", "кот"),
        _node("n2", "кот"),
        _node("n3", "кот"),
        _node("n4", "дом"),
    ], edges=[])

    label = dominant_lemma_label(["mn-1", "mn-2"], mg, graph)
    assert label == "кот"


def test_lexicographic_tiebreak():
    mg = Metagraph(document_id="doc-1", meta_nodes=[_l1("mn-1", ["n1", "n2"])])
    graph = SemanticGraph(document_id="doc-1", nodes=[
        _node("n1", "яблоко"),
        _node("n2", "груша"),
    ], edges=[])

    # Обе леммы по 1 разу → побеждает лексикографически меньшая.
    label = dominant_lemma_label(["mn-1"], mg, graph)
    assert label == "груша"


def test_filters_pron_det_short_lemmas_and_predicates():
    mg = Metagraph(document_id="doc-1", meta_nodes=[
        _l1("mn-1", ["n_pron", "n_short", "n_pred", "n_ok"]),
    ])
    graph = SemanticGraph(document_id="doc-1", nodes=[
        _node("n_pron", "он", upos="PRON"),
        _node("n_short", "ия", upos="NOUN"),  # < min_lemma_len=3
        _node("n_pred", "идти", upos="VERB", kind="predicate"),
        _node("n_ok", "книга", upos="NOUN"),
    ], edges=[])

    label = dominant_lemma_label(["mn-1"], mg, graph)
    assert label == "книга"


def test_fallback_to_first_l1_label_when_no_significant_lemmas():
    mg = Metagraph(document_id="doc-1", meta_nodes=[
        _l1("mn-1", ["n_pron"], label="заглавие1"),
        _l1("mn-2", [], label="заглавие2"),
    ])
    graph = SemanticGraph(document_id="doc-1", nodes=[
        _node("n_pron", "он", upos="PRON"),
    ], edges=[])

    label = dominant_lemma_label(["mn-1", "mn-2"], mg, graph)
    # Сортировка по id: mn-1 первая.
    assert label == "заглавие1"


def test_aggregates_lemmas_across_multiple_l1():
    mg = Metagraph(document_id="doc-1", meta_nodes=[
        _l1("mn-1", ["n1"]),
        _l1("mn-2", ["n2"]),
        _l1("mn-3", ["n3"]),
    ])
    graph = SemanticGraph(document_id="doc-1", nodes=[
        _node("n1", "снег"),
        _node("n2", "снег"),
        _node("n3", "лес"),
    ], edges=[])

    label = dominant_lemma_label(["mn-1", "mn-2", "mn-3"], mg, graph)
    assert label == "снег"
