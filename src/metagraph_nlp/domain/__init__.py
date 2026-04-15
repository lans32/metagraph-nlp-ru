"""Доменная модель проекта.

Явные типы первого класса для всех сущностей pipeline: текст, клаузы,
семантический граф, метаграф, provenance. См. CLAUDE.md §7.2, §9, §12.5.
"""

from metagraph_nlp.domain.graph import Edge, Node, SemanticGraph
from metagraph_nlp.domain.ids import IdFactory
from metagraph_nlp.domain.metagraph import GraphFragment, Metagraph, MetaEdge, MetaNode
from metagraph_nlp.domain.provenance import Provenance
from metagraph_nlp.domain.text import Clause, Document, Sentence, TextSpan

__all__ = [
    "Clause",
    "Document",
    "Edge",
    "GraphFragment",
    "IdFactory",
    "Metagraph",
    "MetaEdge",
    "MetaNode",
    "Node",
    "Provenance",
    "SemanticGraph",
    "Sentence",
    "TextSpan",
]
