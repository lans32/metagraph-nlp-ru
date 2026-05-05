"""Результат разрешения анафоры (CLAUDE.md §10).

Содержит запись о каждой замене местоимения на антецедент: какой PRON-узел
исчез, к какому узлу-сущности были перенаправлены его рёбра, по каким
морфологическим признакам прошло согласование и на каком расстоянии (в
предложениях) находился антецедент.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AnaphoraResolution(BaseModel):
    pronoun_node_id: str = Field(description="ID удалённого PRON-узла.")
    antecedent_node_id: str = Field(description="ID узла-антецедента.")
    sentence_id: str = Field(description="ID предложения, в котором стоит местоимение.")
    pronoun_surface: str = Field(description="Поверхностная форма местоимения.")
    pronoun_lemma: str
    antecedent_lemma: str
    matched_features: dict[str, str] = Field(
        default_factory=dict,
        description="Сопоставленные feats: Gender / Number / Animacy.",
    )
    distance_sentences: int = Field(
        ge=0,
        description="Расстояние между местоимением и антецедентом в предложениях.",
    )
