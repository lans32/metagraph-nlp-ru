"""Ориентированный семантический граф (CLAUDE.md §4.2)."""

from __future__ import annotations

from pydantic import BaseModel, Field

from metagraph_nlp.domain.provenance import Provenance


class Node(BaseModel):
    """Узел семантического графа — именной концепт или сущность.

    Правило: `label` — всегда начальная форма (лемма) после перехода на
    UD-парсер. Поле `surface` хранит исходную поверхностную форму для
    трассируемости к тексту (CLAUDE.md §9.1).
    """

    id: str
    label: str
    kind: str = Field(default="concept", description="Тип узла (concept, entity, ...).")
    lemma: str | None = None
    surface: str | None = Field(
        default=None,
        description="Исходная поверхностная форма слова в тексте.",
    )
    upos: str | None = Field(
        default=None,
        description="Универсальная часть речи (NOUN, PROPN, VERB, ...).",
    )
    clause_id: str | None = None
    token_id_in_sent: int | None = Field(
        default=None,
        description="id_in_sent исходного UD-токена (нумерация с 1) для якоря к feats.",
    )
    original_lemma: str | None = Field(
        default=None,
        description=(
            "Исходная лемма до замены анафорой. None для обычных узлов; "
            "для PRON-узлов после anaphora_resolution_v1 содержит лемму "
            "местоимения (например, 'он' при заменённом lemma='иван')."
        ),
    )
    original_upos: str | None = Field(
        default=None,
        description="Исходный UPOS до замены анафорой (обычно PRON или DET).",
    )
    antecedent_node_id: str | None = Field(
        default=None,
        description=(
            "ID узла-антецедента, если этот узел был заменён анафорой. "
            "Связь с антецедентом для трассировки и виз-слоя."
        ),
    )
    provenance: Provenance


class Edge(BaseModel):
    """Ориентированное ребро — отношение между узлами, обычно предикат."""

    id: str
    source: str
    target: str
    relation: str = Field(description="Лейбл отношения (обычно лемма предиката).")
    kind: str = Field(default="predicate")
    clause_id: str | None = None
    predicate_class: list[str] | None = Field(
        default=None,
        description=(
            "Семантические классы предиката (напр. motion, communication) "
            "из словаря predicate_classes.yaml. None если лемма не в словаре."
        ),
    )
    provenance: Provenance


class SemanticGraph(BaseModel):
    """Ориентированный граф для всего документа."""

    document_id: str
    nodes: list[Node] = Field(default_factory=list)
    edges: list[Edge] = Field(default_factory=list)

    def node_index(self) -> dict[str, Node]:
        return {n.id: n for n in self.nodes}
