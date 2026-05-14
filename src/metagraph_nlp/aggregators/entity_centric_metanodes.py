"""Агрегация уровня 2: per-entity метавершины по значимым леммам.

Правило ``entity_centric_v0`` — альтернатива ``entity_cluster_v0``
(CLAUDE.md §5.1, §5.5). В отличие от union-find кластеризации, которая
склонна объединять всё в один компонент, здесь создаётся **по одной
L2-метавершине на каждую значимую лемму**, собирающую все L1-метавершины
(клаузы), упоминающие эту лемму. Одна L1-метавершина может входить в
несколько entity-метавершин (холархия).

Алгоритм:
1. Построить ``mnode_by_clause`` из L1-метавершин.
2. Для каждого узла графа собрать ``LemmaInfo``: множество clause_ids,
   список node_ids, флаг is_propn.
3. Отфильтровать леммы: исключить служебные upos, короткие, предикаты;
   оставить леммы с freq ≥ min_freq (для PROPN — ≥ 1 при propn_always).
4. Для каждой квалифицированной леммы создать L2-метавершину типа
   ``entity_centric``, содержащую L1-метавершины соответствующих клауз.

Инварианты:
- одна L1-метавершина может принадлежать нескольким L2-метавершинам;
- леммы обрабатываются в отсортированном порядке (детерминизм);
- provenance хранит лемму, частоту и флаг PROPN.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

from metagraph_nlp.domain import (
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Provenance,
    SemanticGraph,
)

_STAGE = "aggregate"
_RULE = "entity_centric_v0"

_DEFAULT_EXCLUDE_UPOS: frozenset[str] = frozenset(
    {"PRON", "DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART"}
)


@dataclass
class _LemmaInfo:
    clause_ids: set[str] = field(default_factory=set)
    node_ids: list[str] = field(default_factory=list)
    is_propn: bool = False


def aggregate_entity_centric(
    metagraph: Metagraph,
    graph: SemanticGraph,
    ids: IdFactory,
    *,
    min_freq: int = 2,
    min_lemma_len: int = 3,
    propn_always: bool = True,
    exclude_upos: frozenset[str] = _DEFAULT_EXCLUDE_UPOS,
) -> list[MetaNode]:
    """Создать L2-метавершины, по одной на каждую значимую лемму.

    Мутирует ``metagraph.meta_nodes`` и возвращает созданные L2-метавершины.
    """

    mnode_by_clause: dict[str, str] = {}
    for mn in metagraph.meta_nodes:
        if mn.level != 1:
            continue
        cid = mn.provenance.clause_id
        if cid is None:
            continue
        mnode_by_clause.setdefault(cid, mn.id)

    lemma_info: dict[str, _LemmaInfo] = defaultdict(_LemmaInfo)
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
        info = lemma_info[lemma]
        info.clause_ids.add(n.clause_id)
        info.node_ids.append(n.id)
        if n.upos == "PROPN":
            info.is_propn = True

    doc_id = graph.document_id

    created: list[MetaNode] = []
    for lemma in sorted(lemma_info.keys()):
        info = lemma_info[lemma]
        freq = len(info.clause_ids)

        if propn_always and info.is_propn:
            if freq < 1:
                continue
        else:
            if freq < min_freq:
                continue

        sorted_mnode_ids = sorted(
            mnode_by_clause[cid] for cid in info.clause_ids
        )

        fragment = GraphFragment(meta_node_ids=sorted_mnode_ids)
        created.append(
            MetaNode(
                id=ids.mnode(),
                type="entity_centric",
                level=2,
                label=lemma,
                fragment=fragment,
                provenance=Provenance(
                    rule=_RULE,
                    stage=_STAGE,
                    inputs=sorted_mnode_ids,
                    document_id=doc_id,
                    notes=f"lemma={lemma}, freq={freq}, propn={info.is_propn}",
                ),
            )
        )

    metagraph.meta_nodes.extend(created)
    return created
