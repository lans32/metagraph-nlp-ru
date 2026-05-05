"""Свёртка именных групп (NP collapse): multi-token NP → один узел графа.

Цель: сделать семантический граф компактнее и осмысленнее, объединяя
именную вершину с её модификаторами (amod, nmod, det) в один узел с
составной леммой (CLAUDE.md §5.2).

Запускается после build_semantic_graph, до агрегации.
"""

from __future__ import annotations

from metagraph_nlp.domain import (
    Clause,
    Edge,
    IdFactory,
    Node,
    Provenance,
    SemanticGraph,
)
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

_STAGE = "np_collapse"
_RULE = "np_collapse_v0"

_NP_CHILD_DEPRELS = {"amod", "det", "nmod", "nmod:poss"}


def collapse_noun_phrases(
    graph: SemanticGraph,
    parsed_sentences: dict[str, ParsedSentence],
    clauses: list[Clause],
    ids: IdFactory,
) -> SemanticGraph:
    """Свернуть multi-token NP в единый узел графа.

    Для каждого NOUN-узла находит его amod/nmod/det потомков в UD-дереве,
    создаёт один узел с объединённой леммой, перенаправляет рёбра.

    Возвращает новый SemanticGraph (исходный не мутируется).
    """
    clause_to_parsed: dict[str, ParsedSentence] = {}
    for c in clauses:
        parsed = parsed_sentences.get(c.sentence_id)
        if parsed:
            clause_to_parsed[c.id] = parsed

    node_index = graph.node_index()
    merge_map: dict[str, str] = {}
    new_nodes: dict[str, Node] = {}
    removed_ids: set[str] = set()

    for node in graph.nodes:
        if node.id in removed_ids or node.id in merge_map:
            continue
        if node.upos != "NOUN" and node.upos != "PROPN":
            new_nodes[node.id] = node
            continue

        parsed = clause_to_parsed.get(node.clause_id) if node.clause_id else None
        if not parsed:
            new_nodes[node.id] = node
            continue

        head_token = _find_token_by_lemma(parsed, node.lemma, node.surface)
        if not head_token:
            new_nodes[node.id] = node
            continue

        modifier_tokens = [
            t for t in parsed.children_of(head_token.id_in_sent)
            if t.deprel in _NP_CHILD_DEPRELS and t.pos in ("ADJ", "DET", "NOUN", "PROPN")
        ]

        modifier_node_ids: list[str] = []
        for mt in modifier_tokens:
            for other_node in graph.nodes:
                if (
                    other_node.clause_id == node.clause_id
                    and other_node.lemma == mt.lemma
                    and other_node.id not in removed_ids
                    and other_node.id != node.id
                ):
                    modifier_node_ids.append(other_node.id)
                    break

        if not modifier_node_ids:
            new_nodes[node.id] = node
            continue

        mod_lemmas = [node_index[mid].lemma for mid in modifier_node_ids if mid in node_index]
        collapsed_lemma = " ".join(mod_lemmas + [node.lemma or ""])
        collapsed_surface = " ".join(
            [node_index[mid].surface or node_index[mid].lemma or "" for mid in modifier_node_ids]
            + [node.surface or node.label]
        )

        collapsed_node = Node(
            id=node.id,
            label=collapsed_lemma,
            kind=node.kind,
            lemma=collapsed_lemma,
            surface=collapsed_surface,
            upos=node.upos,
            clause_id=node.clause_id,
            provenance=Provenance(
                rule=_RULE,
                stage=_STAGE,
                inputs=[node.id, *modifier_node_ids],
                document_id=graph.document_id,
                clause_id=node.clause_id,
                notes=f"collapsed_np={collapsed_lemma}",
            ),
        )
        new_nodes[node.id] = collapsed_node

        for mid in modifier_node_ids:
            merge_map[mid] = node.id
            removed_ids.add(mid)

    final_nodes = [n for n in new_nodes.values() if n.id not in removed_ids]

    final_edges: list[Edge] = []
    for e in graph.edges:
        src = merge_map.get(e.source, e.source)
        tgt = merge_map.get(e.target, e.target)
        if src in removed_ids or tgt in removed_ids:
            continue
        if src == tgt:
            continue
        final_edges.append(
            Edge(
                id=e.id,
                source=src,
                target=tgt,
                relation=e.relation,
                kind=e.kind,
                clause_id=e.clause_id,
                provenance=e.provenance,
            )
        )

    return SemanticGraph(
        document_id=graph.document_id,
        nodes=final_nodes,
        edges=final_edges,
    )


def _find_token_by_lemma(
    parsed: ParsedSentence, lemma: str | None, surface: str | None
) -> Token | None:
    if not lemma:
        return None
    for t in parsed.tokens:
        if t.lemma == lemma:
            return t
    if surface:
        for t in parsed.tokens:
            if t.text == surface:
                return t
    return None
