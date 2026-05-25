"""Результат разрешения анафоры.

Содержит запись о каждой замене местоимения на антецедент: какой PRON-узел
исчез, к какому узлу-сущности были перенаправлены его рёбра, по каким
морфологическим признакам прошло согласование и на каком расстоянии (в
предложениях) находился антецедент.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnaphoraResolution(BaseModel):
    pronoun_node_id: str = Field(
        description=(
            "ID PRON-узла, у которого обновлены лексические атрибуты. "
            "Узел остаётся в графе (anaphora_resolution_v1)."
        )
    )
    antecedent_node_id: str = Field(description="ID узла-антецедента.")
    sentence_id: str = Field(description="ID предложения, в котором стоит местоимение.")
    pronoun_surface: str = Field(description="Поверхностная форма местоимения.")
    pronoun_lemma: str
    antecedent_lemma: str
    pronoun_type: str = Field(
        description=(
            "Тип местоимения: personal_3p / possessive_3p / reflexive."
        )
    )
    resolution_strategy: str = Field(
        description=(
            "Как найден антецедент: 'search' (поиск по окну с salience-скорингом) "
            "или 'clause_subject' (subject текущей клаузы для возвратных)."
        )
    )
    matched_features: dict[str, str] = Field(
        default_factory=dict,
        description="Сопоставленные feats: Gender / Number / Animacy.",
    )
    distance_sentences: int = Field(
        ge=0,
        description="Расстояние между местоимением и антецедентом в предложениях.",
    )
    salience_score: int | None = Field(
        default=None,
        description=(
            "Итоговый salience-балл выбранного кандидата. None для "
            "resolution_strategy='clause_subject' — там скоринг не применяется."
        ),
    )
