"""Агрегаторы семантического графа в метаграф."""

from metagraph_nlp.aggregators.clause_as_metanode import aggregate_clauses_to_metanodes
from metagraph_nlp.aggregators.entity_centric_metanodes import aggregate_entity_centric
from metagraph_nlp.aggregators.entity_cluster_metanodes import aggregate_entity_clusters
from metagraph_nlp.aggregators.paragraph_metanodes import (
    aggregate_clauses_to_paragraphs,
)
from metagraph_nlp.aggregators.predicate_class_cluster import (
    aggregate_predicate_class_clusters,
)
from metagraph_nlp.aggregators.shared_entity_metaedges import (
    build_shared_entity_metaedges,
)
from metagraph_nlp.aggregators.topic_overlap_metaedges import (
    build_topic_overlap_metaedges,
)

__all__ = [
    "aggregate_clauses_to_metanodes",
    "aggregate_clauses_to_paragraphs",
    "aggregate_entity_centric",
    "aggregate_entity_clusters",
    "aggregate_predicate_class_clusters",
    "build_shared_entity_metaedges",
    "build_topic_overlap_metaedges",
]
