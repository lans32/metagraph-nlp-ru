"""Сегментация на предложения.

MVP: используется razdel.sentenize — надёжный rule-based сегментатор для
русского языка. Возвращает список Sentence с точным span'ом в
нормализованном тексте документа.
"""

from __future__ import annotations

from razdel import sentenize

from metagraph_nlp.domain import Document, IdFactory, Provenance, Sentence, TextSpan

_STAGE = "segment"
_RULE = "razdel.sentenize"


def split_sentences(document: Document, ids: IdFactory) -> list[Sentence]:
    sentences: list[Sentence] = []
    for i, piece in enumerate(sentenize(document.normalized_text)):
        span = TextSpan(start=piece.start, end=piece.stop, text=piece.text)
        sentences.append(
            Sentence(
                id=ids.sent(),
                document_id=document.id,
                index=i,
                span=span,
                provenance=Provenance(
                    rule=_RULE,
                    stage=_STAGE,
                    inputs=[document.id],
                    document_id=document.id,
                ),
            )
        )
    return sentences
