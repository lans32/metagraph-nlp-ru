"""Наивный билдер семантического графа из клауз.

MVP: на каждую клаузу — до двух узлов (грубо: левая именная часть + правая
именная часть относительно первого глагола) и одно ребро-предикат между ними.
Это заглушка, достаточная для сквозной проверки pipeline и формы артефактов.

Реальное построение графа (через UD-парсер, роли, предложные конструкции)
выносится в отдельную задачу под skill `semantic-graph-builder`.
"""

from __future__ import annotations

import re

from metagraph_nlp.domain import (
    Clause,
    Document,
    Edge,
    IdFactory,
    Node,
    Provenance,
    SemanticGraph,
)

_STAGE = "graph_builder"
_RULE = "naive_head_dep_v0"

# Грубый список глагольных окончаний для первичного поиска предиката без
# морфологического анализатора. Это заглушка, не лингвистическая модель.
_VERB_SUFFIXES = (
    "ет", "ёт", "ут", "ют", "ит", "ат", "ят",
    "ал", "ял", "ил", "ел", "ала", "яла", "ила", "ела",
    "али", "яли", "или", "ели",
    "ся", "сь",
)

_WORD_RE = re.compile(r"[А-Яа-яЁёA-Za-z][А-Яа-яЁёA-Za-z\-]*")


def _looks_verb(token: str) -> bool:
    t = token.lower()
    if len(t) < 3:
        return False
    return t.endswith(_VERB_SUFFIXES)


def _first_verb_index(tokens: list[str]) -> int | None:
    for i, tok in enumerate(tokens):
        if _looks_verb(tok):
            return i
    return None


def _pick_noun_like(tokens: list[str]) -> str | None:
    """Первое слово с большой буквы или самое длинное не-глагольное слово."""
    for tok in tokens:
        if tok and tok[0].isupper():
            return tok
    candidates = [t for t in tokens if not _looks_verb(t)]
    if not candidates:
        return None
    return max(candidates, key=len)


def build_semantic_graph(
    document: Document,
    clauses: list[Clause],
    ids: IdFactory,
) -> SemanticGraph:
    graph = SemanticGraph(document_id=document.id)

    for clause in clauses:
        text = clause.span.text
        tokens = _WORD_RE.findall(text)
        if not tokens:
            continue

        verb_idx = _first_verb_index(tokens)
        if verb_idx is None:
            relation = "REL"
            left_tokens, right_tokens = tokens[: len(tokens) // 2], tokens[len(tokens) // 2 :]
        else:
            relation = tokens[verb_idx].lower()
            left_tokens = tokens[:verb_idx]
            right_tokens = tokens[verb_idx + 1 :]

        left_label = _pick_noun_like(left_tokens)
        right_label = _pick_noun_like(right_tokens)

        if left_label is None and right_label is None:
            continue
        if left_label is None:
            left_label = "∅"
        if right_label is None:
            right_label = "∅"

        prov_node = Provenance(
            rule=_RULE,
            stage=_STAGE,
            inputs=[clause.id],
            document_id=document.id,
            clause_id=clause.id,
        )

        source = Node(
            id=ids.node(),
            label=left_label,
            kind="concept",
            clause_id=clause.id,
            provenance=prov_node,
        )
        target = Node(
            id=ids.node(),
            label=right_label,
            kind="concept",
            clause_id=clause.id,
            provenance=prov_node.model_copy(),
        )
        edge = Edge(
            id=ids.edge(),
            source=source.id,
            target=target.id,
            relation=relation,
            kind="predicate",
            clause_id=clause.id,
            provenance=Provenance(
                rule=_RULE,
                stage=_STAGE,
                inputs=[clause.id, source.id, target.id],
                document_id=document.id,
                clause_id=clause.id,
            ),
        )
        graph.nodes.extend([source, target])
        graph.edges.append(edge)

    return graph
