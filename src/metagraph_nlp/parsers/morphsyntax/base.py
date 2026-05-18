"""Контракт морфо-синтаксического парсера.

Любой UD-парсер, встроенный в pipeline, должен реализовать этот Protocol.
Это позволяет подменять реализацию (natasha, spacy, fake для тестов) без
изменений в верхних слоях.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence


@runtime_checkable
class MorphSyntaxParser(Protocol):
    def parse(self, sentence_text: str) -> ParsedSentence:
        """Разобрать одно предложение и вернуть UD-структуру.

        На вход подаётся строка **одного** предложения; разбиение на
        предложения делает razdel на предыдущей стадии.
        """
        ...
