"""Морфо-синтаксический разбор: адаптеры UD-парсеров.

Изолируем зависимость от конкретного парсера (natasha / spacy) за
внутренним контрактом `MorphSyntaxParser`. Код выше по pipeline работает
с нашими типами `Token` / `ParsedSentence`, а не с объектами парсера.
"""

from metagraph_nlp.parsers.morphsyntax.base import MorphSyntaxParser
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

__all__ = ["MorphSyntaxParser", "ParsedSentence", "Token"]
