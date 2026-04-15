from __future__ import annotations

from metagraph_nlp.domain import IdFactory, Provenance, Sentence
from metagraph_nlp.domain.text import TextSpan
from metagraph_nlp.parsers.clauses import extract_clauses
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token


def _prov(stage: str = "test") -> Provenance:
    return Provenance(rule="test", stage=stage, document_id="doc_test")


def _mk_sentence(ids: IdFactory, text: str, start: int = 0) -> Sentence:
    return Sentence(
        id=ids.sent(),
        document_id="doc_test",
        index=0,
        span=TextSpan(start=start, end=start + len(text), text=text),
        provenance=_prov("segment"),
    )


def _tok(
    i: int,
    text: str,
    lemma: str,
    pos: str,
    head: int,
    deprel: str,
    start: int,
    end: int,
    feats: dict[str, str] | None = None,
) -> Token:
    return Token(
        id_in_sent=i,
        text=text,
        lemma=lemma,
        pos=pos,
        feats=feats or {},
        head=head,
        deprel=deprel,
        start=start,
        end=end,
    )


def test_simple_sentence_single_clause():
    ids = IdFactory()
    text = "Студент читает книгу."
    sent = _mk_sentence(ids, text)
    parsed = ParsedSentence(
        text=text,
        tokens=[
            _tok(1, "Студент", "студент", "NOUN", head=2, deprel="nsubj", start=0, end=7),
            _tok(
                2,
                "читает",
                "читать",
                "VERB",
                head=0,
                deprel="root",
                start=8,
                end=14,
                feats={"VerbForm": "Fin"},
            ),
            _tok(3, "книгу", "книга", "NOUN", head=2, deprel="obj", start=15, end=20),
            _tok(4, ".", ".", "PUNCT", head=2, deprel="punct", start=20, end=21),
        ],
    )
    clauses = extract_clauses([sent], ids, {sent.id: parsed})
    assert len(clauses) == 1
    assert clauses[0].head_lemma == "читать"
    assert clauses[0].provenance.rule == "ud_subtree_clauses_v0"


def test_coordinated_verbs_two_clauses():
    ids = IdFactory()
    text = "Он читал, и она писала."
    sent = _mk_sentence(ids, text)
    parsed = ParsedSentence(
        text=text,
        tokens=[
            _tok(1, "Он", "он", "PRON", head=2, deprel="nsubj", start=0, end=2),
            _tok(
                2,
                "читал",
                "читать",
                "VERB",
                head=0,
                deprel="root",
                start=3,
                end=8,
                feats={"VerbForm": "Fin"},
            ),
            _tok(3, ",", ",", "PUNCT", head=5, deprel="punct", start=8, end=9),
            _tok(4, "и", "и", "CCONJ", head=5, deprel="cc", start=10, end=11),
            _tok(
                5,
                "писала",
                "писать",
                "VERB",
                head=2,
                deprel="conj",
                start=16,
                end=22,
                feats={"VerbForm": "Fin"},
            ),
            _tok(6, "она", "она", "PRON", head=5, deprel="nsubj", start=12, end=15),
        ],
    )
    clauses = extract_clauses([sent], ids, {sent.id: parsed})
    lemmas = sorted(c.head_lemma for c in clauses)
    assert lemmas == ["писать", "читать"]
    assert len(clauses) == 2
    # спаны у двух клауз не совпадают
    spans = {(c.span.start, c.span.end) for c in clauses}
    assert len(spans) == 2


def test_ccomp_nested_clause():
    ids = IdFactory()
    text = "Он сказал, что устал."
    sent = _mk_sentence(ids, text)
    parsed = ParsedSentence(
        text=text,
        tokens=[
            _tok(1, "Он", "он", "PRON", head=2, deprel="nsubj", start=0, end=2),
            _tok(
                2,
                "сказал",
                "сказать",
                "VERB",
                head=0,
                deprel="root",
                start=3,
                end=9,
                feats={"VerbForm": "Fin"},
            ),
            _tok(3, ",", ",", "PUNCT", head=5, deprel="punct", start=9, end=10),
            _tok(4, "что", "что", "SCONJ", head=5, deprel="mark", start=11, end=14),
            _tok(
                5,
                "устал",
                "устать",
                "VERB",
                head=2,
                deprel="ccomp",
                start=15,
                end=20,
                feats={"VerbForm": "Fin"},
            ),
        ],
    )
    clauses = extract_clauses([sent], ids, {sent.id: parsed})
    assert len(clauses) == 2
    by_lemma = {c.head_lemma: c for c in clauses}
    assert set(by_lemma) == {"сказать", "устать"}
    # Главная клауза "сказать" не должна захватывать токены ccomp-поддерева.
    main = by_lemma["сказать"]
    assert "устал" not in main.span.text
    assert "что" not in main.span.text
    nested = by_lemma["устать"]
    assert "устал" in nested.span.text


def test_fallback_when_no_verb():
    ids = IdFactory()
    text = "Тишина."
    sent = _mk_sentence(ids, text)
    parsed = ParsedSentence(
        text=text,
        tokens=[
            _tok(1, "Тишина", "тишина", "NOUN", head=0, deprel="root", start=0, end=6),
            _tok(2, ".", ".", "PUNCT", head=1, deprel="punct", start=6, end=7),
        ],
    )
    clauses = extract_clauses([sent], ids, {sent.id: parsed})
    assert len(clauses) == 1
    assert clauses[0].provenance.rule == "sentence_as_clause_v0"
    assert "fallback" in (clauses[0].provenance.notes or "")


def test_legacy_strategy_sentence_as_clause():
    ids = IdFactory()
    sentences = [
        _mk_sentence(ids, "Первое предложение."),
        _mk_sentence(ids, "Второе.", start=20),
    ]
    clauses = extract_clauses(sentences, ids, strategy="sentence_as_clause_v0")
    assert len(clauses) == 2
    for c, s in zip(clauses, sentences):
        assert c.sentence_id == s.id
        assert c.span == s.span
        assert c.provenance.rule == "sentence_as_clause_v0"
        assert c.head_lemma is None


def test_none_parsed_falls_back_to_legacy():
    ids = IdFactory()
    sent = _mk_sentence(ids, "Одна клауза.")
    clauses = extract_clauses([sent], ids, parsed_sentences=None)
    assert len(clauses) == 1
    assert clauses[0].provenance.rule == "sentence_as_clause_v0"
