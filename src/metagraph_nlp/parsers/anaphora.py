"""Разрешение анафоры (anaphora_resolution_v0).

Rule-based замена личных местоимений 3-го лица на ближайший согласованный
антецедент (NOUN/PROPN). Согласование — по Gender, Number, Animacy
(см. CLAUDE.md §3, §5.1, §10).

Алгоритм:

1. PRON-узлы графа сматчиваются с UD-токенами через `Node.token_id_in_sent`
   и фильтруются: PronType=Prs, Person=3.
2. Для каждого PRON по порядку (слева направо, по предложениям) ищется
   ближайший предшествующий NOUN/PROPN в окне `search_window_sentences`,
   у которого совпадают Gender, Number, Animacy с местоимением.
3. Все рёбра PRON-узла перенаправляются на узел-антецедент; PRON
   удаляется. Дубликаты рёбер (после переноса совпавшие по
   source/target/relation/clause_id) схлопываются в одно.
4. Provenance перенаправленного ребра дополняется ссылкой на удалённый
   pronoun_id и правилом `anaphora_resolution_v0` (§9.4 «no silent collapse»).
"""

from __future__ import annotations

import logging

from metagraph_nlp.domain import (
    AnaphoraResolution,
    Clause,
    Edge,
    IdFactory,
    Node,
    Provenance,
    SemanticGraph,
    Sentence,
)
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

logger = logging.getLogger("metagraph_nlp.anaphora")

_STAGE = "anaphora_resolution"
_RULE = "anaphora_resolution_v0"

_PERSONAL_3P_LEMMAS = {"он", "она", "оно", "они"}


def _is_personal_3p(token: Token) -> bool:
    if token.pos != "PRON":
        return False
    if token.lemma in _PERSONAL_3P_LEMMAS:
        return True
    feats = token.feats
    return feats.get("PronType") == "Prs" and feats.get("Person") == "3"


def _agreement_ok(
    pronoun: Token, candidate: Token, *, require_animacy_match: bool
) -> dict[str, str] | None:
    """Проверить согласование PRON с кандидатом-антецедентом.

    Возвращает словарь сопоставленных feats либо None, если согласования нет.
    Правила:
    - Number: PRON.Number должен совпасть с кандидатом (если у обоих указан).
    - Gender: для единственного числа Gender должен совпасть; для множественного
      Gender у местоимения отсутствует, ограничение не применяется.
    - Animacy: при require_animacy_match — должен совпасть (если задан у обоих).
    """
    p_feats = pronoun.feats
    c_feats = candidate.feats
    matched: dict[str, str] = {}

    p_number = p_feats.get("Number")
    c_number = c_feats.get("Number")
    if p_number and c_number and p_number != c_number:
        return None
    if p_number:
        matched["Number"] = p_number

    if p_number != "Plur":
        p_gender = p_feats.get("Gender")
        c_gender = c_feats.get("Gender")
        if p_gender and c_gender and p_gender != c_gender:
            return None
        if p_gender:
            matched["Gender"] = p_gender

    if require_animacy_match:
        p_anim = p_feats.get("Animacy")
        c_anim = c_feats.get("Animacy")
        if p_anim and c_anim and p_anim != c_anim:
            return None
        if p_anim:
            matched["Animacy"] = p_anim

    return matched


def _build_token_index(
    nodes: list[Node],
    clauses: list[Clause],
    parsed_sentences: dict[str, ParsedSentence],
) -> dict[str, Token]:
    """node_id → Token (через clause.sentence_id и token_id_in_sent)."""
    clause_to_sentence: dict[str, str] = {c.id: c.sentence_id for c in clauses}
    result: dict[str, Token] = {}
    for n in nodes:
        if n.token_id_in_sent is None or n.clause_id is None:
            continue
        sent_id = clause_to_sentence.get(n.clause_id)
        if sent_id is None:
            continue
        parsed = parsed_sentences.get(sent_id)
        if parsed is None:
            continue
        tok = parsed.by_id(n.token_id_in_sent)
        if tok is not None:
            result[n.id] = tok
    return result


def resolve_anaphora(
    graph: SemanticGraph,
    clauses: list[Clause],
    sentences: list[Sentence],
    parsed_sentences: dict[str, ParsedSentence],
    ids: IdFactory,
    *,
    search_window_sentences: int = 2,
    require_animacy_match: bool = True,
) -> tuple[SemanticGraph, list[AnaphoraResolution]]:
    """Заменить личные местоимения 3-го лица на узлы-антецеденты.

    Возвращает новый SemanticGraph и список произведённых замен. Граф на
    входе не мутируется. `ids` принимается для совместимости сигнатур
    стадий и задела на будущее (новые рёбра при дедупликации сейчас не
    создаются — переиспользуются существующие).
    """
    del ids  # см. docstring

    sentence_by_id: dict[str, Sentence] = {s.id: s for s in sentences}
    clause_by_id: dict[str, Clause] = {c.id: c for c in clauses}
    token_by_node = _build_token_index(graph.nodes, clauses, parsed_sentences)

    def node_position(node: Node) -> tuple[int, int]:
        """Глобальная позиция узла: (sentence.index, token_id_in_sent)."""
        clause = clause_by_id.get(node.clause_id) if node.clause_id else None
        sent = sentence_by_id.get(clause.sentence_id) if clause else None
        sent_idx = sent.index if sent else -1
        return (sent_idx, node.token_id_in_sent or 0)

    pron_nodes: list[Node] = []
    cand_nodes: list[Node] = []
    for n in graph.nodes:
        tok = token_by_node.get(n.id)
        if tok is None:
            continue
        if _is_personal_3p(tok):
            pron_nodes.append(n)
        elif tok.pos in ("NOUN", "PROPN"):
            cand_nodes.append(n)

    pron_nodes.sort(key=node_position)
    cand_nodes.sort(key=node_position)

    merge_map: dict[str, str] = {}
    resolutions: list[AnaphoraResolution] = []

    for pron in pron_nodes:
        pron_tok = token_by_node[pron.id]
        pron_pos = node_position(pron)

        # Ранжирование кандидатов:
        # 1) PROPN (именованная сущность) предпочтительнее NOUN — типичный
        #    референт «он/она/они» в нарративе;
        # 2) субъект (nsubj/nsubj:pass) предпочтительнее косвенных аргументов
        #    (obl/obj/nmod) — субъекты salient-нее;
        # 3) при равенстве — recency (ближе к местоимению).
        #
        # Без этого Animacy-фильтр не спасает, когда natasha не выставила
        # Animacy на PRON: тогда «он» цепляет ближайшее существительное,
        # даже если это «дом», а не «Иван».
        best: tuple[Node, dict[str, str], int, tuple[int, int, tuple[int, int]]] | None = None
        for cand in cand_nodes:
            cand_pos = node_position(cand)
            if cand_pos >= pron_pos:
                continue
            sent_distance = pron_pos[0] - cand_pos[0]
            if sent_distance > search_window_sentences:
                continue
            cand_tok = token_by_node.get(cand.id)
            if cand_tok is None:
                continue
            matched = _agreement_ok(
                pron_tok, cand_tok, require_animacy_match=require_animacy_match
            )
            if matched is None:
                continue

            propn_score = 1 if cand_tok.pos == "PROPN" else 0
            subj_score = 1 if cand_tok.deprel in ("nsubj", "nsubj:pass") else 0
            score = (propn_score, subj_score, cand_pos)

            if best is None or score > best[3]:
                best = (cand, matched, sent_distance, score)

        if best is None:
            logger.debug("anaphora unresolved: pron_id=%s surface=%s", pron.id, pron.surface)
            continue

        antecedent, matched_feats, sent_distance, _ = best
        # если антецедент сам был ранее перенаправлен — следуем по цепочке
        target_id = antecedent.id
        while target_id in merge_map:
            target_id = merge_map[target_id]

        merge_map[pron.id] = target_id
        target_node = next(n for n in graph.nodes if n.id == target_id)

        clause = clause_by_id.get(pron.clause_id) if pron.clause_id else None
        sent_id = clause.sentence_id if clause else ""

        resolutions.append(
            AnaphoraResolution(
                pronoun_node_id=pron.id,
                antecedent_node_id=target_id,
                sentence_id=sent_id,
                pronoun_surface=pron.surface or pron.label,
                pronoun_lemma=pron.lemma or pron.label,
                antecedent_lemma=target_node.lemma or target_node.label,
                matched_features=matched_feats,
                distance_sentences=sent_distance,
            )
        )

    if not resolutions:
        return graph, []

    removed_ids = set(merge_map.keys())
    final_nodes = [n for n in graph.nodes if n.id not in removed_ids]

    seen_edges: set[tuple[str, str, str, str | None]] = set()
    final_edges: list[Edge] = []
    for e in graph.edges:
        src = merge_map.get(e.source, e.source)
        tgt = merge_map.get(e.target, e.target)
        if src == tgt:
            # ребро схлопнулось в петлю после замены — отбрасываем
            continue
        key = (src, tgt, e.relation, e.clause_id)
        if key in seen_edges:
            continue
        seen_edges.add(key)

        if src == e.source and tgt == e.target:
            final_edges.append(e)
            continue

        # ребро было затронуто заменой — обновляем provenance
        replaced_pron_ids = []
        if e.source != src:
            replaced_pron_ids.append(e.source)
        if e.target != tgt:
            replaced_pron_ids.append(e.target)

        prov = e.provenance
        new_inputs = list(prov.inputs)
        for pid in replaced_pron_ids:
            if pid not in new_inputs:
                new_inputs.append(pid)
        notes_parts = [prov.notes] if prov.notes else []
        notes_parts.append(
            f"anaphora_replaced_pronoun_id={','.join(replaced_pron_ids)}"
        )
        new_prov = Provenance(
            rule=_RULE,
            stage=_STAGE,
            inputs=new_inputs,
            document_id=prov.document_id,
            sentence_id=prov.sentence_id,
            clause_id=prov.clause_id,
            config_hash=prov.config_hash,
            notes="; ".join(notes_parts),
        )
        final_edges.append(
            Edge(
                id=e.id,
                source=src,
                target=tgt,
                relation=e.relation,
                kind=e.kind,
                clause_id=e.clause_id,
                provenance=new_prov,
            )
        )

    new_graph = SemanticGraph(
        document_id=graph.document_id,
        nodes=final_nodes,
        edges=final_edges,
    )
    return new_graph, resolutions
