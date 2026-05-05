"""Общие фикстуры для тестов metagraph-nlp-ru."""

from __future__ import annotations

import pytest

from metagraph_nlp.domain import IdFactory
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token


class FakeParser:
    """Минимальный парсер для быстрых тестов (без natasha).

    Каждое слово → отдельный токен. Первый VERB — root, NOUN — nsubj/obj.
    """

    def parse(self, sentence_text: str) -> ParsedSentence:
        words = sentence_text.replace(".", " .").replace(",", " ,").split()
        tokens: list[Token] = []
        pos = 0
        root_id = 0
        for i, w in enumerate(words, start=1):
            start = sentence_text.index(w, pos)
            end = start + len(w)
            pos = end
            clean = w.strip(".,!?;:")
            if not clean:
                tokens.append(
                    Token(
                        id_in_sent=i, text=w, lemma=w, pos="PUNCT",
                        feats={}, head=root_id or 1, deprel="punct",
                        start=start, end=end,
                    )
                )
                continue

            is_verb = clean[0].isupper() is False and len(clean) > 3
            if is_verb and root_id == 0:
                root_id = i
                tokens.append(
                    Token(
                        id_in_sent=i, text=w, lemma=clean.lower(), pos="VERB",
                        feats={"VerbForm": "Fin"}, head=0, deprel="root",
                        start=start, end=end,
                    )
                )
            else:
                deprel = "nsubj" if not root_id else "obj"
                tokens.append(
                    Token(
                        id_in_sent=i, text=w, lemma=clean.lower(), pos="NOUN",
                        feats={}, head=root_id or 1, deprel=deprel,
                        start=start, end=end,
                    )
                )
        return ParsedSentence(text=sentence_text, tokens=tokens)


@pytest.fixture
def fake_parser() -> FakeParser:
    return FakeParser()


@pytest.fixture
def ids() -> IdFactory:
    return IdFactory()
