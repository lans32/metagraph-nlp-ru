"""UD-based построение семантического графа из клауз.

Для каждой клаузы определяется корневой предикат, поддерево токенов
клаузы (с отсечением вложенных клауз), и прямые аргументы предиката
по UD-ролям (nsubj, obj/iobj, obl). Лейблы узлов — леммы; исходная
форма сохраняется в `surface` для трассируемости (CLAUDE.md §9.1).
"""

from __future__ import annotations

from metagraph_nlp.domain import (
    Clause,
    Document,
    Edge,
    IdFactory,
    Node,
    Provenance,
    SemanticGraph,
)
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

_STAGE = "graph_builder"
_RULE = "ud_roles_v0"

_CLAUSE_STOP_DEPRELS: set[str] = {"ccomp", "xcomp", "advcl", "acl", "acl:relcl"}

_SUBJ_DEPRELS = {"nsubj", "nsubj:pass"}
_OBJ_DEPRELS = {"obj", "iobj"}


def _is_obl(deprel: str) -> bool:
    return deprel == "obl" or deprel.startswith("obl:")


def _find_preposition(token: Token, parsed: ParsedSentence) -> str | None:
    for child in parsed.children_of(token.id_in_sent):
        if child.deprel == "case":
            return child.lemma
    return None


def _find_predicate(clause: Clause, parsed: ParsedSentence) -> Token | None:
    target_lemma = clause.head_lemma
    target_text = clause.head_text
    if target_lemma or target_text:
        for t in parsed.tokens:
            if t.pos != "VERB":
                continue
            if target_lemma and t.lemma == target_lemma:
                return t
            if target_text and t.text == target_text:
                return t
    return parsed.root()


def _node_kind(token: Token) -> str:
    return "entity" if token.pos == "PROPN" else "concept"


def _make_node(
    token: Token,
    clause: Clause,
    document: Document,
    ids: IdFactory,
) -> Node:
    prov = Provenance(
        rule=_RULE,
        stage=_STAGE,
        inputs=[clause.id],
        document_id=document.id,
        sentence_id=clause.sentence_id,
        clause_id=clause.id,
        notes=f"deprel={token.deprel}",
    )
    return Node(
        id=ids.node(),
        label=token.lemma,
        kind=_node_kind(token),
        lemma=token.lemma,
        surface=token.text,
        upos=token.pos,
        clause_id=clause.id,
        provenance=prov,
    )


def build_semantic_graph(
    document: Document,
    clauses: list[Clause],
    parsed_sentences: dict[str, ParsedSentence],
    ids: IdFactory,
) -> SemanticGraph:
    graph = SemanticGraph(document_id=document.id)

    for clause in clauses:
        parsed = parsed_sentences.get(clause.sentence_id)
        if parsed is None:
            continue

        predicate = _find_predicate(clause, parsed)
        if predicate is None:
            continue

        clause_ids = parsed.subtree_ids(
            predicate.id_in_sent,
            stop_deprels=_CLAUSE_STOP_DEPRELS,
        )

        subject_token: Token | None = None
        object_tokens: list[tuple[Token, str | None]] = []

        for child in parsed.children_of(predicate.id_in_sent):
            if child.id_in_sent not in clause_ids:
                continue
            if child.deprel in _SUBJ_DEPRELS and subject_token is None:
                subject_token = child
            elif child.deprel in _OBJ_DEPRELS:
                object_tokens.append((child, None))
            elif _is_obl(child.deprel):
                prep = _find_preposition(child, parsed)
                object_tokens.append((child, prep))

        subject_node: Node | None = None
        if subject_token is not None:
            subject_node = _make_node(subject_token, clause, document, ids)
            graph.nodes.append(subject_node)

        object_nodes: list[tuple[Node, str | None]] = []
        for tok, prep in object_tokens:
            node = _make_node(tok, clause, document, ids)
            graph.nodes.append(node)
            object_nodes.append((node, prep))

        if subject_node is None:
            continue

        for obj_node, prep in object_nodes:
            relation = predicate.lemma if prep is None else f"{predicate.lemma}_{prep}"
            edge_prov = Provenance(
                rule=_RULE,
                stage=_STAGE,
                inputs=[clause.id, subject_node.id, obj_node.id],
                document_id=document.id,
                sentence_id=clause.sentence_id,
                clause_id=clause.id,
                notes=f"predicate_lemma={predicate.lemma}",
            )
            graph.edges.append(
                Edge(
                    id=ids.edge(),
                    source=subject_node.id,
                    target=obj_node.id,
                    relation=relation,
                    kind="predicate",
                    clause_id=clause.id,
                    provenance=edge_prov,
                )
            )

    return graph
