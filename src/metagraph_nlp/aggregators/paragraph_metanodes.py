"""Лингвистическая агрегация уровня 2: параграф как метавершина.

Правило ``paragraph_clauses_v0`` (CLAUDE.md §4.4, §5.1, §5.5). Каждый параграф
исходного текста порождает одну метавершину уровня 2, во фрагменте которой
лежат идентификаторы метавершин уровня 1 (клауз) из этого параграфа.

Формально это соответствует аннотируемой метаграфовой модели Гапанюка:
``mv = ⟨{attr}, MG_f⟩``, где ``MG_f`` допускает вложение других метавершин
(см. материалы/Основные_положения_многомерно_метаграфовой_модели_данных_и_знаний).

Инварианты:
- каждая L1-клауза из документа попадает ровно в одну L2-метавершину
  (параграфы партиционируют клаузы);
- уровень зафиксирован как ``level=2``, тип — ``paragraph``;
- provenance хранит список входных L1-метавершин и sentence_id параграфа.
"""

from __future__ import annotations

from collections import defaultdict

from metagraph_nlp.domain import (
    Clause,
    GraphFragment,
    IdFactory,
    Metagraph,
    MetaNode,
    Provenance,
    Sentence,
)

_STAGE = "aggregate"
_RULE = "paragraph_clauses_v0"


def aggregate_clauses_to_paragraphs(
    metagraph: Metagraph,
    clauses: list[Clause],
    sentences: list[Sentence],
    ids: IdFactory,
) -> list[MetaNode]:
    """Создать метавершины уровня 2, группируя L1-клаузы по параграфу.

    Мутирует ``metagraph.meta_nodes`` (добавляет новые элементы) и возвращает
    список созданных L2-метавершин.
    """

    # 1. sentence_id -> paragraph_index.
    para_by_sent: dict[str, int] = {s.id: s.paragraph_index for s in sentences}

    # 2. clause_id -> (paragraph_index, sentence_id).
    clause_para: dict[str, int] = {}
    clause_sent: dict[str, str] = {}
    for c in clauses:
        if c.sentence_id not in para_by_sent:
            continue
        clause_para[c.id] = para_by_sent[c.sentence_id]
        clause_sent[c.id] = c.sentence_id

    # 3. paragraph_index -> список L1-метавершин (+ sentence_ids — для provenance).
    l1_by_para: dict[int, list[str]] = defaultdict(list)
    sents_by_para: dict[int, set[str]] = defaultdict(set)
    doc_id: str | None = None
    for mn in metagraph.meta_nodes:
        if mn.level != 1 or mn.type != "clause":
            continue
        cid = mn.provenance.clause_id
        if cid is None or cid not in clause_para:
            continue
        p = clause_para[cid]
        l1_by_para[p].append(mn.id)
        sents_by_para[p].add(clause_sent[cid])
        doc_id = doc_id or mn.provenance.document_id

    created: list[MetaNode] = []
    for p in sorted(l1_by_para.keys()):
        l1_ids = sorted(l1_by_para[p])
        sent_ids = sorted(sents_by_para[p])
        fragment = GraphFragment(meta_node_ids=l1_ids)
        created.append(
            MetaNode(
                id=ids.mnode(),
                type="paragraph",
                level=2,
                label=f"paragraph_{p}",
                fragment=fragment,
                provenance=Provenance(
                    rule=_RULE,
                    stage=_STAGE,
                    inputs=[*l1_ids, *sent_ids],
                    document_id=doc_id,
                    notes=f"paragraph_index={p}",
                ),
            )
        )

    metagraph.meta_nodes.extend(created)
    return created
