"""Юнит-тесты сегментации: paragraph_index по границам \\n\\n."""

from __future__ import annotations

from metagraph_nlp.domain import Document, IdFactory, Provenance
from metagraph_nlp.parsers import split_sentences
from metagraph_nlp.parsers.normalize import normalize_text


def _doc(raw: str) -> Document:
    ids = IdFactory()
    normalized = normalize_text(raw)
    return Document(
        id=ids.doc(),
        raw_text=raw,
        normalized_text=normalized,
        provenance=Provenance(rule="test", stage="normalize"),
    )


def test_single_paragraph_all_zero():
    doc = _doc("Первое предложение. Второе предложение. Третье.")
    ids = IdFactory()

    sentences = split_sentences(doc, ids)

    assert len(sentences) == 3
    assert {s.paragraph_index for s in sentences} == {0}


def test_two_paragraphs_split_by_blank_line():
    raw = (
        "Первое предложение. Второе предложение.\n\n"
        "Третье предложение. Четвёртое предложение."
    )
    doc = _doc(raw)
    ids = IdFactory()

    sentences = split_sentences(doc, ids)

    assert len(sentences) == 4
    indices = [s.paragraph_index for s in sentences]
    assert indices == [0, 0, 1, 1]


def test_three_paragraphs():
    raw = (
        "Первое. Второе.\n\n"
        "Третье. Четвёртое.\n\n"
        "Пятое."
    )
    doc = _doc(raw)
    ids = IdFactory()

    sentences = split_sentences(doc, ids)

    assert [s.paragraph_index for s in sentences] == [0, 0, 1, 1, 2]


def test_multiple_blank_lines_collapse():
    """Нормализатор схлопывает 3+ \\n в ровно два — параграф не удваивается."""
    raw = "Первое предложение.\n\n\n\nВторое предложение."
    doc = _doc(raw)
    ids = IdFactory()

    sentences = split_sentences(doc, ids)

    assert [s.paragraph_index for s in sentences] == [0, 1]
