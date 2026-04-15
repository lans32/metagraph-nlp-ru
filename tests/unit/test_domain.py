from metagraph_nlp.domain import (
    Clause,
    Document,
    Edge,
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Node,
    Provenance,
    SemanticGraph,
    Sentence,
    TextSpan,
)


def test_id_factory_is_stable_and_monotonic():
    ids = IdFactory()
    assert ids.doc() == "doc_000001"
    assert ids.doc() == "doc_000002"
    assert ids.sent() == "sent_000001"
    assert ids.node() == "node_000001"
    assert ids.node() == "node_000002"


def _prov(stage: str = "test") -> Provenance:
    return Provenance(rule="unit", stage=stage)


def test_domain_roundtrip_json():
    ids = IdFactory()
    doc = Document(
        id=ids.doc(),
        raw_text="Привет мир.",
        normalized_text="Привет мир.",
        provenance=_prov("normalize"),
    )
    sent = Sentence(
        id=ids.sent(),
        document_id=doc.id,
        index=0,
        span=TextSpan(start=0, end=11, text="Привет мир."),
        provenance=_prov("segment"),
    )
    clause = Clause(
        id=ids.clause(),
        sentence_id=sent.id,
        document_id=doc.id,
        span=sent.span,
        provenance=_prov("clauses"),
    )

    n1 = Node(id=ids.node(), label="Привет", clause_id=clause.id, provenance=_prov("graph"))
    n2 = Node(id=ids.node(), label="мир", clause_id=clause.id, provenance=_prov("graph"))
    e = Edge(
        id=ids.edge(),
        source=n1.id,
        target=n2.id,
        relation="REL",
        clause_id=clause.id,
        provenance=_prov("graph"),
    )
    g = SemanticGraph(document_id=doc.id, nodes=[n1, n2], edges=[e])
    mn = MetaNode(
        id=ids.mnode(),
        type="clause",
        level=1,
        label="Привет мир",
        fragment=GraphFragment(node_ids=[n1.id, n2.id], edge_ids=[e.id]),
        provenance=_prov("aggregate"),
    )
    mg = Metagraph(document_id=doc.id, meta_nodes=[mn])

    for obj in (doc, sent, clause, g, mg):
        blob = obj.model_dump_json()
        assert isinstance(blob, str) and len(blob) > 0
        type(obj).model_validate_json(blob)

    assert mg.max_level() == 1
