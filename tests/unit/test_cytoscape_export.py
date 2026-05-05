"""Тесты для Cytoscape.js экспорта: JSON-структура, containment, HTML."""

from __future__ import annotations

import json
from pathlib import Path

from metagraph_nlp.domain import (
    Clause,
    Edge,
    GraphFragment,
    Metagraph,
    MetaEdge,
    MetaNode,
    Node,
    Provenance,
    SemanticGraph,
)
from metagraph_nlp.domain.text import TextSpan
from metagraph_nlp.viz.cytoscape_export import (
    metagraph_to_cytoscape_elements,
    render_cytoscape_html,
)

_PROV = Provenance(rule="test", stage="test", inputs=[], document_id="doc-1", clause_id="cl-1")


def _fixture():
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            Node(id="n-1", label="кот", kind="concept", lemma="кот",
                 upos="NOUN", clause_id="cl-1", provenance=_PROV),
            Node(id="n-2", label="сидеть", kind="predicate", lemma="сидеть",
                 upos="VERB", clause_id="cl-1", provenance=_PROV),
        ],
        edges=[
            Edge(id="e-1", source="n-2", target="n-1", relation="nsubj",
                 clause_id="cl-1", provenance=_PROV),
        ],
    )
    metagraph = Metagraph(
        document_id="doc-1",
        meta_nodes=[
            MetaNode(
                id="mn-1", type="clause:main", level=1, label="cl1",
                fragment=GraphFragment(node_ids=["n-1", "n-2"], edge_ids=["e-1"]),
                provenance=_PROV,
            ),
        ],
        meta_edges=[],
    )
    clauses = [
        Clause(
            id="cl-1", sentence_id="s-1", document_id="doc-1",
            span=TextSpan(start=0, end=10, text="кот сидит"),
            head_lemma="сидеть",
            provenance=_PROV,
        ),
    ]
    return graph, metagraph, clauses


def test_elements_contain_all_types():
    graph, metagraph, clauses = _fixture()
    elements = metagraph_to_cytoscape_elements(metagraph, graph, clauses)

    kinds = {e["data"]["kind"] for e in elements}
    assert "node" in kinds
    assert "metanode" in kinds
    assert "edge" in kinds


def test_l0_nodes_have_parent_set_to_l1():
    graph, metagraph, clauses = _fixture()
    elements = metagraph_to_cytoscape_elements(metagraph, graph, clauses)

    l0_nodes = [e for e in elements if e["data"]["kind"] == "node"]
    for n in l0_nodes:
        assert n["data"].get("parent") == "mn-1"


def test_metanode_has_level_and_type():
    graph, metagraph, clauses = _fixture()
    elements = metagraph_to_cytoscape_elements(metagraph, graph, clauses)

    mn_elems = [e for e in elements if e["data"]["kind"] == "metanode"]
    assert len(mn_elems) == 1
    data = mn_elems[0]["data"]
    assert data["level"] == 1
    assert data["type"] == "clause:main"


def test_metaedge_appears_in_elements():
    graph, metagraph, clauses = _fixture()
    metagraph.meta_nodes.append(
        MetaNode(
            id="mn-2", type="clause:coord", level=1, label="cl2",
            fragment=GraphFragment(node_ids=[]),
            provenance=_PROV,
        )
    )
    metagraph.meta_edges.append(
        MetaEdge(
            id="me-1", source="mn-1", target="mn-2",
            relation="лемма", type="shared_entity", level=1,
            provenance=_PROV,
        )
    )
    elements = metagraph_to_cytoscape_elements(metagraph, graph, clauses)

    me_elems = [e for e in elements if e["data"]["kind"] == "metaedge"]
    assert len(me_elems) == 1
    assert me_elems[0]["data"]["type"] == "shared_entity"


def test_l2_contains_l1_via_parent():
    graph, metagraph, clauses = _fixture()
    metagraph.meta_nodes.append(
        MetaNode(
            id="mn-l2", type="paragraph", level=2, label="para_0",
            fragment=GraphFragment(meta_node_ids=["mn-1"]),
            provenance=_PROV,
        )
    )
    elements = metagraph_to_cytoscape_elements(metagraph, graph, clauses)

    l1_elems = [e for e in elements if e["data"].get("id") == "mn-1"]
    assert len(l1_elems) == 1
    assert l1_elems[0]["data"]["parent"] == "mn-l2"


def test_render_html_creates_file(tmp_path: Path):
    graph, metagraph, clauses = _fixture()
    out = tmp_path / "test.html"
    render_cytoscape_html(metagraph, graph, clauses, out)

    assert out.exists()
    html = out.read_text(encoding="utf-8")
    assert "cytoscape" in html.lower()
    assert "mn-1" in html
    assert "__ELEMENTS_JSON__" not in html


def test_elements_json_is_valid_in_html(tmp_path: Path):
    graph, metagraph, clauses = _fixture()
    out = tmp_path / "test.html"
    render_cytoscape_html(metagraph, graph, clauses, out)

    html = out.read_text(encoding="utf-8")
    start = html.index("var elements = ") + len("var elements = ")
    end = html.index(";\n\nvar cy = cytoscape")
    json_str = html[start:end]
    parsed = json.loads(json_str)
    assert isinstance(parsed, list)
    assert len(parsed) > 0
