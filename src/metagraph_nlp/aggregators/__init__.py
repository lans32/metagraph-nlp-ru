"""Агрегаторы семантического графа в метаграф (CLAUDE.md §5, §12.3)."""

from metagraph_nlp.aggregators.clause_as_metanode import aggregate_clauses_to_metanodes
from metagraph_nlp.aggregators.shared_entity_metaedges import (
    build_shared_entity_metaedges,
)

__all__ = [
    "aggregate_clauses_to_metanodes",
    "build_shared_entity_metaedges",
]
