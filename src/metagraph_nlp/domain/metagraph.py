"""Метаграф: метавершины, метарёбра, фрагменты графа (CLAUDE.md §4.3, §4.4, §9.2, §9.3).

Метавершина и метаребро — полноценные объекты модели, не флаги.
Метаграф многоуровневый: уровень 0 — исходные вершины/рёбра, уровень 1 —
клаузы как метавершины, далее рекурсивно (§4.4).
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from metagraph_nlp.domain.provenance import Provenance


class GraphFragment(BaseModel):
    """Подграф, попавший внутрь метавершины.

    Содержит ссылки (id) на входящие узлы и рёбра семантического графа,
    а также на вложенные метаэлементы, если агрегация идёт не с базового уровня.
    """

    node_ids: list[str] = Field(default_factory=list)
    edge_ids: list[str] = Field(default_factory=list)
    meta_node_ids: list[str] = Field(default_factory=list)
    meta_edge_ids: list[str] = Field(default_factory=list)


class MetaNode(BaseModel):
    """Метавершина — агрегат из вершин/рёбер/метаэлементов.

    Обязательные атрибуты (§9.2):
    - собственный id;
    - тип (откуда появилась: clause, pattern, cluster, ...);
    - вложенный фрагмент;
    - provenance;
    - уровень иерархии.
    """

    id: str
    type: str = Field(description="Роль метавершины: clause, pattern, cluster, ...")
    level: int = Field(ge=1, description="Уровень в дендрограмме, начиная с 1.")
    label: str | None = None
    fragment: GraphFragment
    provenance: Provenance


class MetaEdge(BaseModel):
    """Метаребро — самостоятельное отношение/процесс между метаэлементами (§9.3)."""

    id: str
    source: str = Field(description="ID источника (Node или MetaNode).")
    target: str = Field(description="ID цели (Node или MetaNode).")
    relation: str
    type: str = Field(default="relation", description="relation | process | ...")
    level: int = Field(ge=1)
    provenance: Provenance


class Metagraph(BaseModel):
    """Многоуровневый метаграф документа."""

    document_id: str
    meta_nodes: list[MetaNode] = Field(default_factory=list)
    meta_edges: list[MetaEdge] = Field(default_factory=list)

    def max_level(self) -> int:
        levels = [mn.level for mn in self.meta_nodes] + [me.level for me in self.meta_edges]
        return max(levels, default=0)
