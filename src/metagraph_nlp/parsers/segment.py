"""Сегментация на предложения.

MVP: используется razdel.sentenize — надёжный rule-based сегментатор для
русского языка. Возвращает список Sentence с точным span'ом в
нормализованном тексте документа. Параграфы размечаются по границам
``\\n\\n`` (нормализатор схлопывает множественные переносы строк ровно в две).
"""

from __future__ import annotations

from bisect import bisect_right

from razdel import sentenize

from metagraph_nlp.domain import Document, IdFactory, Provenance, Sentence, TextSpan

_STAGE = "segment"
_RULE = "razdel.sentenize"


def _paragraph_starts(text: str) -> list[int]:
    """Начала параграфов в нормализованном тексте (0-й всегда включён)."""
    starts = [0]
    i = 0
    while True:
        j = text.find("\n\n", i)
        if j == -1:
            break
        # Старт следующего параграфа — сразу после "\n\n".
        starts.append(j + 2)
        i = j + 2
    return starts


def split_sentences(document: Document, ids: IdFactory) -> list[Sentence]:
    starts = _paragraph_starts(document.normalized_text)
    sentences: list[Sentence] = []
    for i, piece in enumerate(sentenize(document.normalized_text)):
        span = TextSpan(start=piece.start, end=piece.stop, text=piece.text)
        paragraph_index = bisect_right(starts, piece.start) - 1
        sentences.append(
            Sentence(
                id=ids.sent(),
                document_id=document.id,
                index=i,
                span=span,
                paragraph_index=paragraph_index,
                provenance=Provenance(
                    rule=_RULE,
                    stage=_STAGE,
                    inputs=[document.id],
                    document_id=document.id,
                ),
            )
        )
    return sentences
