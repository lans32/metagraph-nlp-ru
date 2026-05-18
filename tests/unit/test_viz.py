from pathlib import Path

from metagraph_nlp.domain import (
    Clause,
    Edge,
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Node,
    Provenance,
    SemanticGraph,
    TextSpan,
)
from metagraph_nlp.viz import (
    graph_to_dot,
    metagraph_to_dot,
    render_graph_html,
    render_metagraph_html,
)


def _build_fixture() -> tuple[SemanticGraph, Metagraph, list[Clause]]:
    ids = IdFactory()
    doc_id = ids.doc()
    sent_id = ids.sent()
    clause_id = ids.clause()

    prov = Provenance(rule="test", stage="test", document_id=doc_id, clause_id=clause_id)
    clause = Clause(
        id=clause_id,
        sentence_id=sent_id,
        document_id=doc_id,
        span=TextSpan(start=0, end=11, text="Студент читает"),
        provenance=prov,
    )

    n1 = Node(id=ids.node(), label="Студент", clause_id=clause_id, provenance=prov)
    n2 = Node(id=ids.node(), label="книгу", clause_id=clause_id, provenance=prov)
    e1 = Edge(
        id=ids.edge(),
        source=n1.id,
        target=n2.id,
        relation="читает",
        clause_id=clause_id,
        provenance=prov,
    )
    graph = SemanticGraph(document_id=doc_id, nodes=[n1, n2], edges=[e1])

    mn = MetaNode(
        id=ids.mnode(),
        type="clause",
        level=1,
        label="Студент читает",
        fragment=GraphFragment(node_ids=[n1.id, n2.id], edge_ids=[e1.id]),
        provenance=prov,
    )
    metagraph = Metagraph(document_id=doc_id, meta_nodes=[mn])

    return graph, metagraph, [clause]


def test_graph_to_dot_contains_all_nodes_and_relations():
    graph, _, clauses = _build_fixture()
    dot = graph_to_dot(graph, clauses)

    assert dot.startswith("digraph")
    for n in graph.nodes:
        assert n.id in dot
        assert n.label in dot
    for e in graph.edges:
        assert e.relation in dot
        assert f'"{e.source}" -> "{e.target}"' in dot


def test_metagraph_to_dot_has_metanodes_and_dashed_contains_edges():
    graph, metagraph, clauses = _build_fixture()
    dot = metagraph_to_dot(metagraph, graph, clauses)

    assert dot.startswith("digraph")
    for mn in metagraph.meta_nodes:
        assert mn.id in dot
        # contains-ребро от метавершины к каждому её узлу.
        for nid in mn.fragment.node_ids:
            assert f'"{mn.id}" -> "{nid}"' in dot
    assert "dashed" in dot
    assert "contains" in dot


def test_render_graph_html_writes_nonempty_file(tmp_path: Path):
    graph, _, clauses = _build_fixture()
    out = tmp_path / "graph.html"
    render_graph_html(graph, clauses, out)

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    assert len(content) > 0
    assert "<html" in content.lower()
    for n in graph.nodes:
        assert n.id in content


def test_render_metagraph_html_writes_nonempty_file(tmp_path: Path):
    graph, metagraph, clauses = _build_fixture()
    out = tmp_path / "metagraph.html"
    render_metagraph_html(metagraph, graph, clauses, out)

    assert out.exists()
    content = out.read_text(encoding="utf-8")
    for mn in metagraph.meta_nodes:
        assert mn.id in content
    for n in graph.nodes:
        assert n.id in content
