"""Свёртка именных групп (NP collapse) v1: multi-token NP → один узел графа.

Цель — сделать семантический граф компактнее и осмысленнее, объединяя
именную вершину с её модификаторами (amod, det, nummod, appos, flat,
nmod:poss) в один узел с составной леммой (CLAUDE.md §5.2). Запускается
после `build_semantic_graph`, до агрегации.

Версия v1 vs v0:

- читает модификаторы напрямую из UD-дерева `ParsedSentence` через
  `Node.token_id_in_sent`, а не ищет узлы в графе по лемме (раньше
  модификаторы-прилагательные просто не попадали в граф, и свёртка
  ничего не находила);
- сортирует head + модификаторы по `Token.id_in_sent` — порядок в
  составной лемме теперь соответствует тексту (раньше mods всегда
  ставились впереди, что для правого nmod давало «клауза анализ»
  вместо «анализ клауза»);
- собирает subtree модификаторов рекурсивно — вложенные NP («русские
  клаузы» внутри «анализ клауз») сворачиваются за один проход;
- список включаемых deprel параметризуем через `AggregationConfig.np_collapse_deprels`;
- сохраняет исходные значения в `Node.original_lemma` / `original_upos`
  (CLAUDE.md §9.4 «no silent collapse»).

Топология графа не меняется: модификаторы не были самостоятельными
узлами и не становятся ими; рёбра, узлы-предикаты и nmod-расширения
из `_expand_nmod` остаются нетронутыми.
"""

from __future__ import annotations

from collections import deque

from metagraph_nlp.domain import (
    Clause,
    IdFactory,
    Node,
    Provenance,
    SemanticGraph,
)
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

_STAGE = "np_collapse"
_RULE = "np_collapse_v1"

_DEFAULT_NP_CHILD_DEPRELS: frozenset[str] = frozenset(
    {
        "amod",
        "det",
        "nummod",
        "appos",
        "flat",
        "flat:name",
        "nmod:poss",
    }
)


def collapse_noun_phrases(
    graph: SemanticGraph,
    parsed_sentences: dict[str, ParsedSentence],
    clauses: list[Clause],
    ids: IdFactory,
    *,
    include_deprels: set[str] | frozenset[str] | None = None,
) -> SemanticGraph:
    """Свернуть multi-token NP в единый узел графа (v1).

    Для каждого NOUN/PROPN-узла находит UD-subtree его модификаторов
    через `Token.id_in_sent` (а не lookup по лемме). Собирает составную
    лемму и поверхностную форму в порядке token-id. Топология графа
    остаётся той же — обновляется только сам узел.

    Возвращает новый `SemanticGraph` (исходный не мутируется).
    """
    deprels: frozenset[str] = (
        frozenset(include_deprels)
        if include_deprels is not None
        else _DEFAULT_NP_CHILD_DEPRELS
    )

    clause_to_parsed: dict[str, ParsedSentence] = {}
    for c in clauses:
        parsed = parsed_sentences.get(c.sentence_id)
        if parsed:
            clause_to_parsed[c.id] = parsed

    new_nodes: list[Node] = []
    for node in graph.nodes:
        replacement = _try_collapse_node(node, clause_to_parsed, graph, deprels)
        new_nodes.append(replacement if replacement is not None else node)

    return SemanticGraph(
        document_id=graph.document_id,
        nodes=new_nodes,
        edges=list(graph.edges),
    )


def _try_collapse_node(
    node: Node,
    clause_to_parsed: dict[str, ParsedSentence],
    graph: SemanticGraph,
    deprels: frozenset[str],
) -> Node | None:
    """Если у NOUN/PROPN-узла есть UD-модификаторы — вернуть свёрнутый Node,
    иначе None (оставить как есть)."""
    if node.upos not in ("NOUN", "PROPN"):
        return None
    if node.token_id_in_sent is None:
        return None
    if node.clause_id is None:
        return None
    parsed = clause_to_parsed.get(node.clause_id)
    if parsed is None:
        return None

    head_token = parsed.by_id(node.token_id_in_sent)
    if head_token is None:
        return None

    modifier_tokens = _collect_modifiers(head_token, parsed, deprels)
    if not modifier_tokens:
        return None

    np_tokens = sorted(
        [head_token, *modifier_tokens],
        key=lambda t: t.id_in_sent,
    )
    collapsed_lemma = " ".join(
        (t.lemma if t.lemma else t.text).lower() for t in np_tokens
    )
    collapsed_surface = " ".join(t.text for t in np_tokens)

    modifier_ids_sorted = sorted(t.id_in_sent for t in modifier_tokens)
    return Node(
        id=node.id,
        label=collapsed_lemma,
        kind=node.kind,
        lemma=collapsed_lemma,
        surface=collapsed_surface,
        upos=node.upos,
        clause_id=node.clause_id,
        token_id_in_sent=node.token_id_in_sent,
        original_lemma=node.lemma,
        original_upos=node.upos,
        antecedent_node_id=node.antecedent_node_id,
        provenance=Provenance(
            rule=_RULE,
            stage=_STAGE,
            inputs=[node.id],
            document_id=graph.document_id,
            clause_id=node.clause_id,
            notes=(
                f"collapsed_np={collapsed_lemma}; "
                f"modifier_token_ids={modifier_ids_sorted}"
            ),
        ),
    )


def _collect_modifiers(
    head: Token,
    parsed: ParsedSentence,
    deprels: frozenset[str],
) -> list[Token]:
    """BFS subtree от head, включающий рекурсивно все токены с deprel из
    `deprels`. Возвращает плоский список модификаторов (head не включён).
    Гарантирует, что вложенные модификаторы того же типа тоже собираются
    за один проход (fixed-point не нужен).
    """
    collected: list[Token] = []
    queue: deque[int] = deque([head.id_in_sent])
    visited: set[int] = {head.id_in_sent}
    while queue:
        parent_id = queue.popleft()
        for child in parsed.children_of(parent_id):
            if child.id_in_sent in visited:
                continue
            if child.deprel not in deprels:
                continue
            visited.add(child.id_in_sent)
            collected.append(child)
            queue.append(child.id_in_sent)
    return collected
