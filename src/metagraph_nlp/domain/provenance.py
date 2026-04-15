"""Происхождение элементов (provenance) как объект первого класса.

CLAUDE.md §9.1, §10: у каждого значимого элемента должно быть прослеживаемое
происхождение — исходный документ/предложение/клауза, правило создания,
стадия pipeline, хэш конфига, ссылки на входы.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class Provenance(BaseModel):
    """Запись о происхождении элемента модели."""

    rule: str = Field(description="Имя правила/процедуры, создавшей элемент.")
    stage: str = Field(description="Стадия pipeline, на которой элемент появился.")
    inputs: list[str] = Field(
        default_factory=list,
        description="Идентификаторы элементов, послуживших входом.",
    )
    document_id: str | None = None
    sentence_id: str | None = None
    clause_id: str | None = None
    config_hash: str | None = None
    notes: str | None = None
