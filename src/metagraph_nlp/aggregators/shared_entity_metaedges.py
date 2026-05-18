"""Агрегация уровня 1: метарёбра между клаузами-метавершинами по общим сущностям.

Идея (CLAUDE.md §5.1, §5.4, §9.3): если две клаузы упоминают один и тот же
концепт (определяемый по лемме значимых слов), между соответствующими
метавершинами уровня 1 создаётся метаребро типа ``shared_entity``.

Правило детерминированно: пары клауз и общие леммы обрабатываются в
отсортированном порядке, провенанс каждого метаребра включает идентификаторы
исходных узлов графа, на которых основано решение.

Это первый шаг к "сквизингу" графа: сами метавершины уже есть, теперь между
ними появляются объяснимые связи.
"""

from __future__ import annotations

from collections import defaultdict

from metagraph_nlp.domain import (
    IdFactory,
    Metagraph,
    MetaEdge,
    Provenance,
    SemanticGraph,
)

_STAGE = "aggregate"
_RULE = "shared_entity_by_lemma_v0"

_DEFAULT_EXCLUDE_UPOS: frozenset[str] = frozenset(
    {"PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}
)


def build_shared_entity_metaedges(
    metagraph: Metagraph,
    graph: SemanticGraph,
    ids: IdFactory,
    *,
    min_lemma_len: int = 3,
    exclude_upos: frozenset[str] = _DEFAULT_EXCLUDE_UPOS,
) -> list[MetaEdge]:
    """Создать метарёбра уровня 1 между клаузами, делящими лемму сущности.

    Мутирует ``metagraph.meta_edges`` (добавляет новые элементы) и возвращает
    именно список новых метарёбер, созданных в этом вызове.
    """

    # 1. clause_id -> mnode_id (только метавершины уровня 1 с clause_id в provenance).
    mnode_by_clause: dict[str, str] = {}
    for mn in metagraph.meta_nodes:
        if mn.level != 1:
            continue
        cid = mn.provenance.clause_id
        if cid is None:
            continue
        # На один clause_id ожидается одна метавершина уровня 1; если вдруг
        # их несколько, сохраняем первую в порядке обхода (детерминизм через
        # порядок в metagraph.meta_nodes).
        mnode_by_clause.setdefault(cid, mn.id)

    # 2. clause_id -> lemma -> [node_id, ...] (только значимые узлы).
    clause_lemmas: dict[str, dict[str, list[str]]] = defaultdict(lambda: defaultdict(list))
    for n in graph.nodes:
        if n.clause_id is None or n.clause_id not in mnode_by_clause:
            continue
        if n.kind == "predicate":
            continue
        if n.upos in exclude_upos:
            continue
        lemma = n.lemma
        if lemma is None or len(lemma) < min_lemma_len:
            continue
        clause_lemmas[n.clause_id][lemma].append(n.id)

    # 3. Пары клауз в детерминированном порядке.
    sorted_clauses = sorted(clause_lemmas.keys())
    created: list[MetaEdge] = []

    for i, clause_a in enumerate(sorted_clauses):
        lemmas_a = clause_lemmas[clause_a]
        for clause_b in sorted_clauses[i + 1 :]:
            lemmas_b = clause_lemmas[clause_b]
            common = sorted(set(lemmas_a.keys()) & set(lemmas_b.keys()))
            if not common:
                continue
            for lemma in common:
                input_ids = sorted(lemmas_a[lemma] + lemmas_b[lemma])
                me = MetaEdge(
                    id=ids.medge(),
                    source=mnode_by_clause[clause_a],
                    target=mnode_by_clause[clause_b],
                    relation=lemma,
                    type="shared_entity",
                    level=1,
                    provenance=Provenance(
                        rule=_RULE,
                        stage=_STAGE,
                        inputs=input_ids,
                        document_id=graph.document_id,
                        notes=f"lemma={lemma}",
                    ),
                )
                created.append(me)

    metagraph.meta_edges.extend(created)
    return created
