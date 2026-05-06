"""UD-based построение семантического графа из клауз.

Для каждой клаузы определяется корневой предикат, поддерево токенов
клаузы (с отсечением вложенных клауз), и прямые аргументы предиката
по UD-ролям (nsubj, obj/iobj, obl). Лейблы узлов — леммы; исходная
форма сохраняется в `surface` для трассируемости (CLAUDE.md §9.1).

Дополнительно:

- для каждого аргумента раскрываются его ``nmod``-подгруппы
  (предложные группы при существительном) как отдельные узлы, связанные
  с родителем рёбрами вида ``<prep>`` / ``nmod``;
- если у предиката-``conj`` не найден собственный ``nsubj``, субъект
  наследуется у головы сочинения (shared subject across coordinated
  predicates).
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


def _is_nmod(deprel: str) -> bool:
    return deprel == "nmod" or deprel.startswith("nmod:")


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


def _find_shared_subject(
    predicate: Token, parsed: ParsedSentence
) -> Token | None:
    """Поднимаемся по цепочке ``conj`` до головы сочинения и берём её nsubj."""
    current = predicate
    visited: set[int] = set()
    while current.deprel == "conj" and current.id_in_sent not in visited:
        visited.add(current.id_in_sent)
        head = parsed.by_id(current.head)
        if head is None:
            return None
        for child in parsed.children_of(head.id_in_sent):
            if child.deprel in _SUBJ_DEPRELS:
                return child
        current = head
    return None


def _node_kind(token: Token) -> str:
    return "entity" if token.pos == "PROPN" else "concept"


def _make_node(
    token: Token,
    clause: Clause,
    document: Document,
    ids: IdFactory,
    extra_note: str | None = None,
) -> Node:
    notes = f"deprel={token.deprel}"
    if extra_note:
        notes = f"{notes}; {extra_note}"
    prov = Provenance(
        rule=_RULE,
        stage=_STAGE,
        inputs=[clause.id],
        document_id=document.id,
        sentence_id=clause.sentence_id,
        clause_id=clause.id,
        notes=notes,
    )
    return Node(
        id=ids.node(),
        label=token.lemma,
        kind=_node_kind(token),
        lemma=token.lemma,
        surface=token.text,
        upos=token.pos,
        clause_id=clause.id,
        token_id_in_sent=token.id_in_sent,
        provenance=prov,
    )


def _make_edge(
    source: Node,
    target: Node,
    relation: str,
    kind: str,
    clause: Clause,
    document: Document,
    ids: IdFactory,
    note: str,
) -> Edge:
    prov = Provenance(
        rule=_RULE,
        stage=_STAGE,
        inputs=[clause.id, source.id, target.id],
        document_id=document.id,
        sentence_id=clause.sentence_id,
        clause_id=clause.id,
        notes=note,
    )
    return Edge(
        id=ids.edge(),
        source=source.id,
        target=target.id,
        relation=relation,
        kind=kind,
        clause_id=clause.id,
        provenance=prov,
    )


def _expand_nmod(
    parent_token: Token,
    parent_node: Node,
    parsed: ParsedSentence,
    clause_ids: set[int],
    clause: Clause,
    document: Document,
    ids: IdFactory,
    graph: SemanticGraph,
) -> None:
    """Добавить nmod-подгруппы (предложные группы) как узлы+рёбра."""
    for child in parsed.children_of(parent_token.id_in_sent):
        if child.id_in_sent not in clause_ids:
            continue
        if not _is_nmod(child.deprel):
            continue
        prep = _find_preposition(child, parsed)
        nmod_node = _make_node(
            child, clause, document, ids, extra_note=f"nmod_of={parent_token.lemma}"
        )
        graph.nodes.append(nmod_node)
        relation = prep if prep is not None else "nmod"
        kind = "prepositional" if prep is not None else "nmod"
        graph.edges.append(
            _make_edge(
                parent_node,
                nmod_node,
                relation,
                kind,
                clause,
                document,
                ids,
                note=f"nmod expansion of {parent_token.lemma}",
            )
        )


def build_semantic_graph(
    document: Document,
    clauses: list[Clause],
    parsed_sentences: dict[str, ParsedSentence],
    ids: IdFactory,
    predicate_classes: dict[str, frozenset[str]] | None = None,
) -> SemanticGraph:
    graph = SemanticGraph(document_id=document.id)
    classes_index = predicate_classes or {}

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

        shared_subject = False
        if subject_token is None:
            shared = _find_shared_subject(predicate, parsed)
            if shared is not None:
                subject_token = shared
                shared_subject = True

        subject_node: Node | None = None
        if subject_token is not None:
            extra = "shared_subject_via_conj" if shared_subject else None
            subject_node = _make_node(
                subject_token, clause, document, ids, extra_note=extra
            )
            graph.nodes.append(subject_node)
            if not shared_subject:
                _expand_nmod(
                    subject_token, subject_node, parsed, clause_ids,
                    clause, document, ids, graph,
                )

        object_nodes: list[tuple[Node, Token, str | None]] = []
        for tok, prep in object_tokens:
            node = _make_node(tok, clause, document, ids)
            graph.nodes.append(node)
            object_nodes.append((node, tok, prep))

        if subject_node is None:
            # Без субъекта предикативных рёбер не создаём, но nmod-расширения
            # для объектов всё равно собираем — они самостоятельные связи.
            for obj_node, obj_tok, _prep in object_nodes:
                _expand_nmod(
                    obj_tok, obj_node, parsed, clause_ids,
                    clause, document, ids, graph,
                )
            continue

        pred_classes = classes_index.get(predicate.lemma)
        for obj_node, obj_tok, prep in object_nodes:
            relation = predicate.lemma if prep is None else f"{predicate.lemma}_{prep}"
            edge = _make_edge(
                subject_node, obj_node, relation, "predicate",
                clause, document, ids,
                note=f"predicate_lemma={predicate.lemma}",
            )
            if pred_classes:
                edge.predicate_class = sorted(pred_classes)
            graph.edges.append(edge)
            _expand_nmod(
                obj_tok, obj_node, parsed, clause_ids,
                clause, document, ids, graph,
            )

    return graph
