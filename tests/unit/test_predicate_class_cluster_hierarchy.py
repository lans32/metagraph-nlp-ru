"""Юнит-тесты иерархической агрегации predicate_class_cluster_v0.

Покрывают:
- фильтр L2 по уровням (leaf / mid / root) с PredicateHierarchy;
- containment-метарёбра между parent ↔ child L2-кластерами;
- backward-compat: hierarchy=None → поведение v0 (никаких contains-рёбер).
"""

from __future__ import annotations

import pytest

from metagraph_nlp.aggregators.predicate_class_cluster import (
    aggregate_predicate_class_clusters,
)
from metagraph_nlp.domain import (
    Edge,
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    PredicateHierarchy,
    Provenance,
    SemanticGraph,
)

DOC = "doc-1"
_PROV = Provenance(rule="t", stage="t", document_id=DOC)


def _l1(id: str, edge_ids: list[str]) -> MetaNode:
    return MetaNode(
        id=id, type="clause:main", level=1, label=id,
        fragment=GraphFragment(edge_ids=edge_ids),
        provenance=_PROV,
    )


def _edge(id: str, relation: str, classes: list[str] | None) -> Edge:
    return Edge(
        id=id, source="s", target="t", relation=relation,
        kind="predicate", clause_id="c",
        predicate_class=classes,
        provenance=_PROV,
    )


@pytest.fixture
def hierarchy() -> PredicateHierarchy:
    """Иерархия с 2 корнями и 2 уровнями глубины."""
    return PredicateHierarchy(
        parent_of={
            "motion": None,
            "motion_walk": "motion",
            "motion_walk_stroll": "motion_walk",
            "communication": None,
            "communication_speak": "communication",
        },
        level_of={
            "motion": "root",
            "motion_walk": "mid",
            "motion_walk_stroll": "leaf",
            "communication": "root",
            "communication_speak": "leaf",
        },
        anchor_of={
            "motion": "106587-V",
            "motion_walk": "106588-V",
            "motion_walk_stroll": "106589-V",
            "communication": "106876-V",
            "communication_speak": "106877-V",
        },
        label_of={
            "motion": "движение",
            "motion_walk": "ходьба",
            "motion_walk_stroll": "прогулка",
            "communication": "общение",
            "communication_speak": "речь",
        },
        lemma_paths={
            "гулять": [["motion_walk_stroll", "motion_walk", "motion"]],
            "сказать": [["communication_speak", "communication"]],
        },
    )


def _setup_two_motion_two_communication(hierarchy):
    """Метаграф: 4 L1, 2 motion-глагола + 2 communication.

    Каждое ребро размечено полным path. Все 4 L1 попадают и в leaf, и в
    mid (только motion), и в root.
    """
    mg = Metagraph(document_id=DOC, meta_nodes=[
        _l1("mn-1", ["e1"]),
        _l1("mn-2", ["e2"]),
        _l1("mn-3", ["e3"]),
        _l1("mn-4", ["e4"]),
    ])
    graph = SemanticGraph(document_id=DOC, edges=[
        # motion ветка: 2 L1 — оба leaf+mid+root
        _edge("e1", "гулять", ["motion_walk_stroll", "motion_walk", "motion"]),
        _edge("e2", "гулять", ["motion_walk_stroll", "motion_walk", "motion"]),
        # communication ветка: 2 L1 — leaf+root (mid отсутствует в этой ветке)
        _edge("e3", "сказать", ["communication_speak", "communication"]),
        _edge("e4", "сказать", ["communication_speak", "communication"]),
    ])
    return mg, graph


def test_levels_filter_only_leaf(hierarchy):
    mg, graph = _setup_two_motion_two_communication(hierarchy)
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(
        mg, graph, ids, min_cluster_size=2,
        hierarchy=hierarchy, levels=["leaf"],
    )

    labels = sorted(c.label for c in created)
    assert labels == ["communication_speak", "motion_walk_stroll"]


def test_levels_filter_only_root(hierarchy):
    mg, graph = _setup_two_motion_two_communication(hierarchy)
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(
        mg, graph, ids, min_cluster_size=2,
        hierarchy=hierarchy, levels=["root"],
    )

    labels = sorted(c.label for c in created)
    assert labels == ["communication", "motion"]


def test_levels_filter_only_mid(hierarchy):
    mg, graph = _setup_two_motion_two_communication(hierarchy)
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(
        mg, graph, ids, min_cluster_size=2,
        hierarchy=hierarchy, levels=["mid"],
    )

    labels = sorted(c.label for c in created)
    assert labels == ["motion_walk"]  # только motion имеет mid-уровень


def test_all_levels_creates_full_hierarchy(hierarchy):
    mg, graph = _setup_two_motion_two_communication(hierarchy)
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(
        mg, graph, ids, min_cluster_size=2,
        hierarchy=hierarchy, levels=["leaf", "mid", "root"],
    )

    labels = sorted(c.label for c in created)
    # 2 motion (leaf+mid+root) + 2 communication (leaf+root) = 5
    assert labels == [
        "communication", "communication_speak",
        "motion", "motion_walk", "motion_walk_stroll",
    ]


def test_provenance_notes_include_hierarchy_metadata(hierarchy):
    mg, graph = _setup_two_motion_two_communication(hierarchy)
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(
        mg, graph, ids, min_cluster_size=2,
        hierarchy=hierarchy, levels=["root"],
    )

    motion = next(c for c in created if c.label == "motion")
    notes = motion.provenance.notes
    assert "level=root" in notes
    assert "parent=None" in notes
    assert "anchor_synset=106587-V" in notes
    assert "lemmas=" in notes


def test_containment_edges_created(hierarchy):
    mg, graph = _setup_two_motion_two_communication(hierarchy)
    ids = IdFactory()

    before_medges = len(mg.meta_edges)
    aggregate_predicate_class_clusters(
        mg, graph, ids, min_cluster_size=2,
        hierarchy=hierarchy, levels=["leaf", "mid", "root"],
        create_containment_edges=True,
    )

    # Ожидаемые containment-ребра:
    #   motion → motion_walk (parent → child)
    #   motion_walk → motion_walk_stroll
    #   communication → communication_speak
    new_medges = mg.meta_edges[before_medges:]
    pairs = {
        (e.source, e.target, e.relation, e.type)
        for e in new_medges
    }
    # source/target — mnode IDs, проверим по relation/type и числу.
    assert len(new_medges) == 3
    for e in new_medges:
        assert e.relation == "contains"
        assert e.type == "containment"
        assert e.level == 2


def test_containment_disabled_by_default(hierarchy):
    mg, graph = _setup_two_motion_two_communication(hierarchy)
    ids = IdFactory()

    before_medges = len(mg.meta_edges)
    aggregate_predicate_class_clusters(
        mg, graph, ids, min_cluster_size=2,
        hierarchy=hierarchy, levels=["leaf", "mid", "root"],
    )

    assert len(mg.meta_edges) == before_medges  # ничего не добавлено


def test_hierarchy_none_backward_compat():
    """hierarchy=None → поведение v0: все классы как leaf, никаких рёбер."""
    mg = Metagraph(document_id=DOC, meta_nodes=[
        _l1("mn-1", ["e1"]),
        _l1("mn-2", ["e2"]),
    ])
    graph = SemanticGraph(document_id=DOC, edges=[
        _edge("e1", "сказать", ["communication"]),
        _edge("e2", "ответить", ["communication"]),
    ])
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(
        mg, graph, ids, min_cluster_size=2,
        hierarchy=None, levels=None, create_containment_edges=True,
    )

    assert len(created) == 1
    assert created[0].label == "communication"
    # При hierarchy=None containment рёбра не создаются.
    assert mg.meta_edges == []
    # provenance.notes без hierarchy-полей
    assert "level=" not in created[0].provenance.notes
    assert "anchor_synset=" not in created[0].provenance.notes


def test_orphan_child_no_containment(hierarchy):
    """Если parent-кластер не создан (мало членов), containment не возникает."""
    # Только 2 leaf-клаузы — leaf проходит min_cluster_size=2, но
    # для root всего 2 мнодов тоже хватает: создаются motion и
    # motion_walk_stroll. Уменьшим до 1 leaf — тогда нет motion_walk_stroll.
    mg = Metagraph(document_id=DOC, meta_nodes=[
        _l1("mn-1", ["e1"]),  # один leaf — недостаточно для cluster
    ])
    graph = SemanticGraph(document_id=DOC, edges=[
        _edge("e1", "гулять", ["motion_walk_stroll", "motion_walk", "motion"]),
    ])
    ids = IdFactory()

    created = aggregate_predicate_class_clusters(
        mg, graph, ids, min_cluster_size=2,
        hierarchy=hierarchy, levels=["leaf", "mid", "root"],
        create_containment_edges=True,
    )

    assert created == []
    assert mg.meta_edges == []
