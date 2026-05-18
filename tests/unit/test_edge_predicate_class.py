"""Тесты заполнения Edge.predicate_class в build_semantic_graph."""

from __future__ import annotations

from metagraph_nlp.domain import (
    Clause,
    Document,
    IdFactory,
    Provenance,
    TextSpan,
)
from metagraph_nlp.graph_builders.from_clause import build_semantic_graph
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token


def _prov(stage: str = "test") -> Provenance:
    return Provenance(rule="test", stage=stage)


def _doc_clause(text: str, ids: IdFactory):
    doc = Document(
        id=ids.doc(),
        raw_text=text,
        normalized_text=text,
        provenance=_prov("normalize"),
    )
    sent_id = ids.sent()
    clause = Clause(
        id=ids.clause(),
        sentence_id=sent_id,
        document_id=doc.id,
        span=TextSpan(start=0, end=len(text), text=text),
        provenance=_prov("clauses"),
    )
    return doc, clause, sent_id


def _svo_parsed(text: str, predicate_lemma: str) -> ParsedSentence:
    return ParsedSentence(
        text=text,
        tokens=[
            Token(id_in_sent=1, text="Иван", lemma="иван", pos="PROPN",
                  head=2, deprel="nsubj", start=0, end=4),
            Token(id_in_sent=2, text=predicate_lemma, lemma=predicate_lemma,
                  pos="VERB", head=0, deprel="root", start=5, end=5 + len(predicate_lemma)),
            Token(id_in_sent=3, text="правду", lemma="правда", pos="NOUN",
                  head=2, deprel="obj", start=6 + len(predicate_lemma),
                  end=12 + len(predicate_lemma)),
        ],
    )


def test_known_predicate_lemma_gets_class():
    ids = IdFactory()
    text = "Иван сказать правду"
    doc, clause, sent_id = _doc_clause(text, ids)
    parsed = _svo_parsed(text, "сказать")

    graph = build_semantic_graph(
        doc, [clause], {sent_id: parsed}, ids,
        predicate_classes={"сказать": frozenset({"communication"})},
    )

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    assert edge.predicate_class == ["communication"]


def test_unknown_predicate_lemma_keeps_none():
    ids = IdFactory()
    text = "Иван редактировать правду"
    doc, clause, sent_id = _doc_clause(text, ids)
    parsed = _svo_parsed(text, "редактировать")

    graph = build_semantic_graph(
        doc, [clause], {sent_id: parsed}, ids,
        predicate_classes={"сказать": frozenset({"communication"})},
    )

    assert len(graph.edges) == 1
    assert graph.edges[0].predicate_class is None


def test_no_lexicon_means_all_edges_have_no_class():
    ids = IdFactory()
    text = "Иван сказать правду"
    doc, clause, sent_id = _doc_clause(text, ids)
    parsed = _svo_parsed(text, "сказать")

    graph = build_semantic_graph(doc, [clause], {sent_id: parsed}, ids)

    assert all(e.predicate_class is None for e in graph.edges)


def test_multiple_classes_sorted():
    ids = IdFactory()
    text = "Иван идти правду"
    doc, clause, sent_id = _doc_clause(text, ids)
    parsed = _svo_parsed(text, "идти")

    graph = build_semantic_graph(
        doc, [clause], {sent_id: parsed}, ids,
        predicate_classes={"идти": frozenset({"motion", "activity"})},
    )

    assert graph.edges[0].predicate_class == ["activity", "motion"]
