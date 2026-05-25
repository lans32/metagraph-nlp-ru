"""Доменная модель проекта.

Явные типы первого класса для всех сущностей pipeline: текст, клаузы,
семантический граф, метаграф, provenance.
"""

from metagraph_nlp.domain.anaphora import AnaphoraResolution
from metagraph_nlp.domain.graph import Edge, Node, SemanticGraph
from metagraph_nlp.domain.ids import IdFactory
from metagraph_nlp.domain.metagraph import GraphFragment, Metagraph, MetaEdge, MetaNode
from metagraph_nlp.domain.predicate_hierarchy import PredicateHierarchy
from metagraph_nlp.domain.provenance import Provenance
from metagraph_nlp.domain.text import Clause, Document, Sentence, TextSpan

__all__ = [
    "AnaphoraResolution",
    "Clause",
    "Document",
    "Edge",
    "GraphFragment",
    "IdFactory",
    "Metagraph",
    "MetaEdge",
    "MetaNode",
    "Node",
    "PredicateHierarchy",
    "Provenance",
    "SemanticGraph",
    "Sentence",
    "TextSpan",
]
