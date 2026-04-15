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


def _make_doc_clause(text: str, ids: IdFactory) -> tuple[Document, Clause, str]:
    doc = Document(
        id=ids.doc(),
        raw_text=text,
        normalized_text=text,
        provenance=_prov("normalize"),
    )
    sentence_id = ids.sent()
    span = TextSpan(start=0, end=len(text), text=text)
    clause = Clause(
        id=ids.clause(),
        sentence_id=sentence_id,
        document_id=doc.id,
        span=span,
        provenance=_prov("clauses"),
    )
    return doc, clause, sentence_id


def test_subject_verb_object_builds_single_edge():
    ids = IdFactory()
    text = "Студент читает книгу"
    doc, clause, sent_id = _make_doc_clause(text, ids)

    parsed = ParsedSentence(
        text=text,
        tokens=[
            Token(id_in_sent=1, text="Студент", lemma="студент", pos="NOUN",
                  head=2, deprel="nsubj", start=0, end=7),
            Token(id_in_sent=2, text="читает", lemma="читать", pos="VERB",
                  head=0, deprel="root", start=8, end=14),
            Token(id_in_sent=3, text="книгу", lemma="книга", pos="NOUN",
                  head=2, deprel="obj", start=15, end=20),
        ],
    )

    graph = build_semantic_graph(doc, [clause], {sent_id: parsed}, ids)

    assert len(graph.nodes) == 2
    labels = {n.label for n in graph.nodes}
    assert labels == {"студент", "книга"}

    assert len(graph.edges) == 1
    edge = graph.edges[0]
    idx = {n.id: n for n in graph.nodes}
    assert idx[edge.source].label == "студент"
    assert idx[edge.target].label == "книга"
    assert edge.relation == "читать"
    assert edge.kind == "predicate"
    assert edge.clause_id == clause.id


def test_oblique_with_preposition_gets_prefixed_relation():
    ids = IdFactory()
    text = "Студент читает книгу в библиотеке"
    doc, clause, sent_id = _make_doc_clause(text, ids)

    parsed = ParsedSentence(
        text=text,
        tokens=[
            Token(id_in_sent=1, text="Студент", lemma="студент", pos="NOUN",
                  head=2, deprel="nsubj", start=0, end=7),
            Token(id_in_sent=2, text="читает", lemma="читать", pos="VERB",
                  head=0, deprel="root", start=8, end=14),
            Token(id_in_sent=3, text="книгу", lemma="книга", pos="NOUN",
                  head=2, deprel="obj", start=15, end=20),
            Token(id_in_sent=4, text="в", lemma="в", pos="ADP",
                  head=5, deprel="case", start=21, end=22),
            Token(id_in_sent=5, text="библиотеке", lemma="библиотека", pos="NOUN",
                  head=2, deprel="obl", start=23, end=33),
        ],
    )

    graph = build_semantic_graph(doc, [clause], {sent_id: parsed}, ids)

    assert len(graph.nodes) == 3
    assert len(graph.edges) == 2

    by_target = {}
    idx = {n.id: n for n in graph.nodes}
    for e in graph.edges:
        by_target[idx[e.target].label] = e

    assert by_target["книга"].relation == "читать"
    assert by_target["библиотека"].relation == "читать_в"
    for e in graph.edges:
        assert idx[e.source].label == "студент"


def test_label_is_lemma_and_surface_is_preserved():
    ids = IdFactory()
    text = "Студенты читали книги"
    doc, clause, sent_id = _make_doc_clause(text, ids)

    parsed = ParsedSentence(
        text=text,
        tokens=[
            Token(id_in_sent=1, text="Студенты", lemma="студент", pos="NOUN",
                  head=2, deprel="nsubj", start=0, end=8),
            Token(id_in_sent=2, text="читали", lemma="читать", pos="VERB",
                  head=0, deprel="root", start=9, end=15),
            Token(id_in_sent=3, text="книги", lemma="книга", pos="NOUN",
                  head=2, deprel="obj", start=16, end=21),
        ],
    )

    graph = build_semantic_graph(doc, [clause], {sent_id: parsed}, ids)

    by_lemma = {n.lemma: n for n in graph.nodes}
    assert by_lemma["студент"].label == "студент"
    assert by_lemma["студент"].surface == "Студенты"
    assert by_lemma["студент"].upos == "NOUN"
    assert by_lemma["книга"].label == "книга"
    assert by_lemma["книга"].surface == "книги"

    for n in graph.nodes:
        assert n.provenance.rule == "ud_roles_v0"
        assert n.provenance.stage == "graph_builder"
        assert clause.id in n.provenance.inputs


def test_propn_subject_becomes_entity():
    ids = IdFactory()
    text = "Москва встречает гостей"
    doc, clause, sent_id = _make_doc_clause(text, ids)

    parsed = ParsedSentence(
        text=text,
        tokens=[
            Token(id_in_sent=1, text="Москва", lemma="москва", pos="PROPN",
                  head=2, deprel="nsubj", start=0, end=6),
            Token(id_in_sent=2, text="встречает", lemma="встречать", pos="VERB",
                  head=0, deprel="root", start=7, end=16),
            Token(id_in_sent=3, text="гостей", lemma="гость", pos="NOUN",
                  head=2, deprel="obj", start=17, end=23),
        ],
    )

    graph = build_semantic_graph(doc, [clause], {sent_id: parsed}, ids)

    by_lemma = {n.lemma: n for n in graph.nodes}
    assert by_lemma["москва"].kind == "entity"
    assert by_lemma["гость"].kind == "concept"
