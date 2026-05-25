"""Текстовые сущности: документ, предложение, клауза, span.

Инвариант текстовой трассируемости: каждая сущность связана с исходным
текстом через span (начало/конец в нормализованном тексте).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from metagraph_nlp.domain.provenance import Provenance


class TextSpan(BaseModel):
    """Полуинтервал [start, end) в нормализованном тексте документа."""

    start: int = Field(ge=0)
    end: int = Field(ge=0)
    text: str

    def __len__(self) -> int:
        return self.end - self.start


class Document(BaseModel):
    id: str
    source_path: str | None = None
    raw_text: str
    normalized_text: str
    provenance: Provenance


class Sentence(BaseModel):
    id: str
    document_id: str
    index: int
    span: TextSpan
    provenance: Provenance
    paragraph_index: int = Field(
        default=0,
        ge=0,
        description="Индекс параграфа в документе (границы — пустые строки).",
    )


class Clause(BaseModel):
    """Клауза — базовая единица агрегации.

    На старте используется упрощение clause=sentence; предикативный центр
    будет заполняться позднее в отдельной задаче.
    """

    id: str
    sentence_id: str
    document_id: str
    span: TextSpan
    head_text: str | None = Field(
        default=None,
        description="Поверхностная форма предикативного центра (если определён).",
    )
    head_lemma: str | None = Field(
        default=None,
        description="Лемма предикативного центра.",
    )
    clause_type: str | None = Field(
        default=None,
        description="Лингвистический тип клаузы: main, coord, compl, xcompl, adverbial, relative, participial.",
    )
    provenance: Provenance
