"""Тесты для NP collapse: свёртка именных групп в один узел графа."""

from __future__ import annotations

from metagraph_nlp.domain import (
    Clause,
    Edge,
    IdFactory,
    Node,
    Provenance,
    SemanticGraph,
)
from metagraph_nlp.domain.text import TextSpan
from metagraph_nlp.graph_builders.np_collapse import collapse_noun_phrases
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

_PROV = Provenance(rule="test", stage="test", inputs=[], document_id="doc-1")


def _make_parsed(tokens: list[Token], text: str) -> ParsedSentence:
    return ParsedSentence(text=text, tokens=tokens)


def test_adj_noun_collapsed():
    """'большой кот' → один узел 'большой кот'."""
    text = "большой кот"
    tokens = [
        Token(id_in_sent=1, text="большой", lemma="большой", pos="ADJ",
              feats={}, head=2, deprel="amod", start=0, end=7),
        Token(id_in_sent=2, text="кот", lemma="кот", pos="NOUN",
              feats={}, head=0, deprel="root", start=8, end=11),
    ]
    parsed = _make_parsed(tokens, text)

    clause = Clause(
        id="cl-1", sentence_id="s-1", document_id="doc-1",
        span=TextSpan(start=0, end=11, text=text),
        head_lemma="кот",
        provenance=_PROV,
    )
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            Node(id="n-adj", label="большой", kind="concept", lemma="большой",
                 surface="большой", upos="ADJ", clause_id="cl-1", provenance=_PROV),
            Node(id="n-noun", label="кот", kind="concept", lemma="кот",
                 surface="кот", upos="NOUN", clause_id="cl-1", provenance=_PROV),
        ],
        edges=[],
    )
    ids = IdFactory()
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [clause], ids)

    assert len(result.nodes) == 1
    collapsed = result.nodes[0]
    assert collapsed.id == "n-noun"
    assert "большой" in collapsed.lemma
    assert "кот" in collapsed.lemma


def test_no_modifiers_no_collapse():
    """Одиночный NOUN без модификаторов остаётся как есть."""
    text = "кот"
    tokens = [
        Token(id_in_sent=1, text="кот", lemma="кот", pos="NOUN",
              feats={}, head=0, deprel="root", start=0, end=3),
    ]
    parsed = _make_parsed(tokens, text)

    clause = Clause(
        id="cl-1", sentence_id="s-1", document_id="doc-1",
        span=TextSpan(start=0, end=3, text=text),
        head_lemma="кот",
        provenance=_PROV,
    )
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            Node(id="n-1", label="кот", kind="concept", lemma="кот",
                 surface="кот", upos="NOUN", clause_id="cl-1", provenance=_PROV),
        ],
        edges=[],
    )
    ids = IdFactory()
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [clause], ids)

    assert len(result.nodes) == 1
    assert result.nodes[0].id == "n-1"
    assert result.nodes[0].lemma == "кот"


def test_verb_node_not_collapsed():
    """VERB-узлы не участвуют в NP collapse."""
    text = "бежит кот"
    tokens = [
        Token(id_in_sent=1, text="бежит", lemma="бежать", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=0, end=5),
        Token(id_in_sent=2, text="кот", lemma="кот", pos="NOUN",
              feats={}, head=1, deprel="nsubj", start=6, end=9),
    ]
    parsed = _make_parsed(tokens, text)

    clause = Clause(
        id="cl-1", sentence_id="s-1", document_id="doc-1",
        span=TextSpan(start=0, end=9, text=text),
        head_lemma="бежать",
        provenance=_PROV,
    )
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            Node(id="n-v", label="бежать", kind="predicate", lemma="бежать",
                 surface="бежит", upos="VERB", clause_id="cl-1", provenance=_PROV),
            Node(id="n-n", label="кот", kind="concept", lemma="кот",
                 surface="кот", upos="NOUN", clause_id="cl-1", provenance=_PROV),
        ],
        edges=[
            Edge(id="e-1", source="n-v", target="n-n", relation="nsubj",
                 clause_id="cl-1", provenance=_PROV),
        ],
    )
    ids = IdFactory()
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [clause], ids)

    assert len(result.nodes) == 2
    assert len(result.edges) == 1


def test_edges_redirected_after_collapse():
    """Рёбра к merged node переподключаются к collapsed node."""
    text = "видит большой кот"
    tokens = [
        Token(id_in_sent=1, text="видит", lemma="видеть", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=0, end=5),
        Token(id_in_sent=2, text="большой", lemma="большой", pos="ADJ",
              feats={}, head=3, deprel="amod", start=6, end=13),
        Token(id_in_sent=3, text="кот", lemma="кот", pos="NOUN",
              feats={}, head=1, deprel="nsubj", start=14, end=17),
    ]
    parsed = _make_parsed(tokens, text)

    clause = Clause(
        id="cl-1", sentence_id="s-1", document_id="doc-1",
        span=TextSpan(start=0, end=17, text=text),
        head_lemma="видеть",
        provenance=_PROV,
    )
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            Node(id="n-v", label="видеть", kind="predicate", lemma="видеть",
                 surface="видит", upos="VERB", clause_id="cl-1", provenance=_PROV),
            Node(id="n-adj", label="большой", kind="concept", lemma="большой",
                 surface="большой", upos="ADJ", clause_id="cl-1", provenance=_PROV),
            Node(id="n-noun", label="кот", kind="concept", lemma="кот",
                 surface="кот", upos="NOUN", clause_id="cl-1", provenance=_PROV),
        ],
        edges=[
            Edge(id="e-1", source="n-v", target="n-adj", relation="видеть",
                 clause_id="cl-1", provenance=_PROV),
        ],
    )
    ids = IdFactory()
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [clause], ids)

    assert len(result.nodes) == 2
    edge = result.edges[0]
    assert edge.source == "n-v"
    assert edge.target == "n-noun"


def test_self_loop_edges_removed():
    """Если src и tgt оба merged в один node, ребро удаляется."""
    text = "большой кот"
    tokens = [
        Token(id_in_sent=1, text="большой", lemma="большой", pos="ADJ",
              feats={}, head=2, deprel="amod", start=0, end=7),
        Token(id_in_sent=2, text="кот", lemma="кот", pos="NOUN",
              feats={}, head=0, deprel="root", start=8, end=11),
    ]
    parsed = _make_parsed(tokens, text)

    clause = Clause(
        id="cl-1", sentence_id="s-1", document_id="doc-1",
        span=TextSpan(start=0, end=11, text=text),
        head_lemma="кот",
        provenance=_PROV,
    )
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            Node(id="n-adj", label="большой", kind="concept", lemma="большой",
                 surface="большой", upos="ADJ", clause_id="cl-1", provenance=_PROV),
            Node(id="n-noun", label="кот", kind="concept", lemma="кот",
                 surface="кот", upos="NOUN", clause_id="cl-1", provenance=_PROV),
        ],
        edges=[
            Edge(id="e-1", source="n-adj", target="n-noun", relation="amod",
                 clause_id="cl-1", provenance=_PROV),
        ],
    )
    ids = IdFactory()
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [clause], ids)

    assert len(result.nodes) == 1
    assert len(result.edges) == 0


def test_original_graph_not_mutated():
    """collapse_noun_phrases возвращает новый граф, не мутирует исходный."""
    text = "большой кот"
    tokens = [
        Token(id_in_sent=1, text="большой", lemma="большой", pos="ADJ",
              feats={}, head=2, deprel="amod", start=0, end=7),
        Token(id_in_sent=2, text="кот", lemma="кот", pos="NOUN",
              feats={}, head=0, deprel="root", start=8, end=11),
    ]
    parsed = _make_parsed(tokens, text)

    clause = Clause(
        id="cl-1", sentence_id="s-1", document_id="doc-1",
        span=TextSpan(start=0, end=11, text=text),
        head_lemma="кот",
        provenance=_PROV,
    )
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            Node(id="n-adj", label="большой", kind="concept", lemma="большой",
                 surface="большой", upos="ADJ", clause_id="cl-1", provenance=_PROV),
            Node(id="n-noun", label="кот", kind="concept", lemma="кот",
                 surface="кот", upos="NOUN", clause_id="cl-1", provenance=_PROV),
        ],
        edges=[],
    )
    ids = IdFactory()
    original_node_count = len(graph.nodes)
    collapse_noun_phrases(graph, {"s-1": parsed}, [clause], ids)

    assert len(graph.nodes) == original_node_count
