"""Разрешение анафоры (anaphora_resolution_v1).

Rule-based замена местоимений на ближайший согласованный антецедент.
Поддерживаемые типы (управляются `pronoun_types` в `AnaphoraConfig`):

- **personal_3p** — личные 3-го лица (он/она/оно/они и формы
  `PronType=Prs`+`Person=3`). Антецедент ищется в окне предложений
  с salience-скорингом + согласованием по Gender/Number/Animacy.
- **possessive_3p** — притяжательные 3-го лица (его/её/их с
  `Poss=Yes`+`Person=3`+`Reflex≠Yes`). Поиск тот же, что и для
  личных.
- **reflexive** — возвратные (себя/свой с `Reflex=Yes`). Антецедент
  всегда subject текущей клаузы (без поиска по окну).

Семантика замены (отличие от v0): PRON-узел **не удаляется**, его
лексические атрибуты (`label`, `lemma`, `upos`) обновляются на значения
антецедента, а исходные значения сохраняются в `original_lemma`,
`original_upos`. Поле `antecedent_node_id` фиксирует ссылку на
антецедент. Рёбра графа не перенаправляются — связность с антецедентом
возникает естественно через `shared_entity_by_lemma_v0` (общая лемма).

Salience-скоринг (упрощённый Lappin–Leass): сумма весов за роль
кандидата (subject > object > oblique), POS (PROPN-бонус), штраф за
расстояние, бонус за тематическую позицию и за повторное упоминание.
Веса конфигурируются через `SalienceWeights`.

Инвариант CLAUDE.md §9.4 «no silent collapse»: каждый заменённый узел
несёт `provenance.rule = anaphora_resolution_v1` и сохраняет оригинал в
`original_lemma`/`original_upos`; журнал замен — `AnaphoraResolution`.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

from metagraph_nlp.config import SalienceWeights
from metagraph_nlp.domain import (
    AnaphoraResolution,
    Clause,
    IdFactory,
    Node,
    Provenance,
    SemanticGraph,
    Sentence,
)
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

logger = logging.getLogger("metagraph_nlp.anaphora")

_STAGE = "anaphora_resolution"
_RULE = "anaphora_resolution_v1"

_PERSONAL_3P_LEMMAS = {"он", "она", "оно", "они"}

_PRONOUN_TYPE_PERSONAL_3P = "personal_3p"
_PRONOUN_TYPE_POSSESSIVE_3P = "possessive_3p"
_PRONOUN_TYPE_REFLEXIVE = "reflexive"

_STRATEGY_SEARCH = "search"
_STRATEGY_CLAUSE_SUBJECT = "clause_subject"


def _is_reflexive(token: Token) -> bool:
    """Возвратные (себя, свой). Reflex=Yes — это и `себя`, и `свой`."""
    if token.pos not in ("PRON", "DET"):
        return False
    return token.feats.get("Reflex") == "Yes"


def _is_possessive_3p(token: Token) -> bool:
    """Притяжательные 3-го лица (его/её/их с Poss=Yes).

    Natasha кладёт «его-притяжательное» как PRON или DET с Poss=Yes;
    «его-личное» (Gen/Acc от «он») — без Poss=Yes. Reflex=Yes исключаем,
    чтобы «свой» попадал в reflexive-ветку.
    """
    if token.pos not in ("PRON", "DET"):
        return False
    f = token.feats
    if f.get("Reflex") == "Yes":
        return False
    if f.get("Poss") != "Yes":
        return False
    return f.get("Person") == "3"


def _is_personal_3p(token: Token) -> bool:
    """Личные местоимения 3-го лица: он/она/оно/они и их падежные формы.

    Возвратные и притяжательные исключаем — у них отдельные ветки.
    """
    if token.pos != "PRON":
        return False
    f = token.feats
    if f.get("Reflex") == "Yes":
        return False
    if f.get("Poss") == "Yes":
        return False
    if token.lemma in _PERSONAL_3P_LEMMAS:
        return True
    return f.get("PronType") == "Prs" and f.get("Person") == "3"


def _classify_pronoun(token: Token) -> str | None:
    """Определить тип местоимения. Порядок проверки важен."""
    if _is_reflexive(token):
        return _PRONOUN_TYPE_REFLEXIVE
    if _is_possessive_3p(token):
        return _PRONOUN_TYPE_POSSESSIVE_3P
    if _is_personal_3p(token):
        return _PRONOUN_TYPE_PERSONAL_3P
    return None


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


def _salience_score(
    cand_tok: Token,
    cand_pos: tuple[int, int],
    pron_pos: tuple[int, int],
    repeat_count: int,
    weights: SalienceWeights,
) -> int:
    """Численный salience-скоринг кандидата (упрощённый Lappin–Leass)."""
    score = 0

    if cand_tok.deprel in ("nsubj", "nsubj:pass"):
        score += weights.subj
    elif cand_tok.deprel == "obj":
        score += weights.obj
    elif cand_tok.deprel in ("iobj", "obl", "nmod"):
        score += weights.oblique

    if cand_tok.pos == "PROPN":
        score += weights.propn

    sent_distance = pron_pos[0] - cand_pos[0]
    score += weights.recency_per_sent * sent_distance

    if cand_pos[1] <= weights.thematic_token_threshold:
        score += weights.thematic

    score += weights.repeat_mention * repeat_count

    return score


@dataclass
class _Candidate:
    node: Node
    matched_feats: dict[str, str]
    sent_distance: int
    score: int
    cand_pos: tuple[int, int]


def _find_antecedent_by_search(
    pron: Node,
    pron_tok: Token,
    pron_pos: tuple[int, int],
    cand_nodes: list[Node],
    token_by_node: dict[str, Token],
    node_position: callable,
    *,
    search_window_sentences: int,
    require_animacy_match: bool,
    salience_weights: SalienceWeights,
    prior_antecedent_counts: dict[str, int],
) -> _Candidate | None:
    """Найти антецедент в окне предложений с salience-скорингом.

    Hard constraints (Number/Gender/Animacy) — фильтр до скоринга.
    Лучший кандидат = argmax(score); tie-breaker — recency.
    """
    best: _Candidate | None = None
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

        repeat_count = prior_antecedent_counts.get(cand.id, 0)
        score = _salience_score(
            cand_tok, cand_pos, pron_pos, repeat_count, salience_weights
        )

        if best is None or score > best.score or (
            score == best.score and cand_pos > best.cand_pos
        ):
            best = _Candidate(
                node=cand,
                matched_feats=matched,
                sent_distance=sent_distance,
                score=score,
                cand_pos=cand_pos,
            )
    return best


def _find_clause_subject(
    pron: Node,
    nodes_by_clause: dict[str, list[Node]],
    token_by_node: dict[str, Token],
) -> Node | None:
    """Найти subject (nsubj/nsubj:pass) клаузы, к которой принадлежит PRON.

    Используется для возвратных «себя/свой»: антецедент = subject
    текущей клаузы. Если subject не найден — возвращаем None
    (не разрешаем).
    """
    if pron.clause_id is None:
        return None
    candidates = nodes_by_clause.get(pron.clause_id, [])
    for n in candidates:
        if n.id == pron.id:
            continue
        tok = token_by_node.get(n.id)
        if tok is None:
            continue
        if tok.deprel in ("nsubj", "nsubj:pass"):
            return n
    return None


def resolve_anaphora(
    graph: SemanticGraph,
    clauses: list[Clause],
    sentences: list[Sentence],
    parsed_sentences: dict[str, ParsedSentence],
    ids: IdFactory,
    *,
    search_window_sentences: int = 2,
    require_animacy_match: bool = True,
    pronoun_types: list[str] | None = None,
    salience_weights: SalienceWeights | None = None,
) -> tuple[SemanticGraph, list[AnaphoraResolution]]:
    """Заменить местоимения на лексику антецедентов (anaphora_resolution_v1).

    Возвращает новый SemanticGraph и список произведённых замен. Граф на
    входе не мутируется. Узлы PRON остаются в графе с обновлёнными
    `label`/`lemma`/`upos`; исходные значения сохраняются в
    `original_lemma`/`original_upos`. Рёбра не трогаем.

    `ids` принимается для совместимости сигнатур стадий и задела на
    будущее (новые узлы/рёбра при замене сейчас не создаются).
    """
    del ids  # см. docstring

    if pronoun_types is None:
        pronoun_types = [
            _PRONOUN_TYPE_PERSONAL_3P,
            _PRONOUN_TYPE_POSSESSIVE_3P,
            _PRONOUN_TYPE_REFLEXIVE,
        ]
    enabled_types = set(pronoun_types)
    if salience_weights is None:
        salience_weights = SalienceWeights()

    sentence_by_id: dict[str, Sentence] = {s.id: s for s in sentences}
    clause_by_id: dict[str, Clause] = {c.id: c for c in clauses}
    token_by_node = _build_token_index(graph.nodes, clauses, parsed_sentences)

    nodes_by_clause: dict[str, list[Node]] = {}
    for n in graph.nodes:
        if n.clause_id is None:
            continue
        nodes_by_clause.setdefault(n.clause_id, []).append(n)

    def node_position(node: Node) -> tuple[int, int]:
        """Глобальная позиция узла: (sentence.index, token_id_in_sent)."""
        clause = clause_by_id.get(node.clause_id) if node.clause_id else None
        sent = sentence_by_id.get(clause.sentence_id) if clause else None
        sent_idx = sent.index if sent else -1
        return (sent_idx, node.token_id_in_sent or 0)

    pron_nodes: list[tuple[Node, str]] = []  # (node, pronoun_type)
    cand_nodes: list[Node] = []
    for n in graph.nodes:
        tok = token_by_node.get(n.id)
        if tok is None:
            continue
        ptype = _classify_pronoun(tok)
        if ptype is not None:
            if ptype in enabled_types:
                pron_nodes.append((n, ptype))
        elif tok.pos in ("NOUN", "PROPN"):
            cand_nodes.append(n)

    pron_nodes.sort(key=lambda pn: node_position(pn[0]))
    cand_nodes.sort(key=node_position)

    # node_id → (label, lemma, upos, antecedent_id) для применения в финале
    replacement_map: dict[str, tuple[str, str, str | None, str]] = {}
    # антецедент → текущая лемма/upos/label с учётом цепочек замен
    resolutions: list[AnaphoraResolution] = []
    # для repeat_mention в salience: antecedent_id → счётчик
    prior_antecedent_counts: dict[str, int] = {}

    for pron, ptype in pron_nodes:
        pron_tok = token_by_node[pron.id]
        pron_pos = node_position(pron)

        antecedent: Node | None = None
        matched_feats: dict[str, str] = {}
        sent_distance = 0
        score: int | None = None
        strategy: str

        if ptype == _PRONOUN_TYPE_REFLEXIVE:
            strategy = _STRATEGY_CLAUSE_SUBJECT
            cand = _find_clause_subject(pron, nodes_by_clause, token_by_node)
            if cand is not None:
                cand_tok = token_by_node.get(cand.id)
                if cand_tok is not None:
                    # для возвратных hard constraints не критичны: subject
                    # клаузы по определению согласован с собой; matched_feats
                    # просто фиксируем для аудита
                    matched_feats = _agreement_ok(
                        pron_tok, cand_tok, require_animacy_match=False
                    ) or {}
                    antecedent = cand
                    sent_distance = 0
        else:
            strategy = _STRATEGY_SEARCH
            best = _find_antecedent_by_search(
                pron,
                pron_tok,
                pron_pos,
                cand_nodes,
                token_by_node,
                node_position,
                search_window_sentences=search_window_sentences,
                require_animacy_match=require_animacy_match,
                salience_weights=salience_weights,
                prior_antecedent_counts=prior_antecedent_counts,
            )
            if best is not None:
                antecedent = best.node
                matched_feats = best.matched_feats
                sent_distance = best.sent_distance
                score = best.score

        if antecedent is None:
            logger.debug(
                "anaphora unresolved: pron_id=%s surface=%s type=%s",
                pron.id,
                pron.surface,
                ptype,
            )
            continue

        # Цепочка замен: если антецедент сам уже был заменён ранее —
        # берём атрибуты его антецедента (так «он» во второй клаузе
        # цепляется к Ивану через «Он» из первой замены).
        target_id = antecedent.id
        target_label = antecedent.label
        target_lemma = antecedent.lemma
        target_upos = antecedent.upos
        if target_id in replacement_map:
            chained = replacement_map[target_id]
            target_label, target_lemma, target_upos, chained_antecedent_id = chained
            target_id = chained_antecedent_id

        replacement_map[pron.id] = (
            target_label,
            target_lemma or target_label,
            target_upos,
            target_id,
        )
        prior_antecedent_counts[target_id] = prior_antecedent_counts.get(target_id, 0) + 1

        clause = clause_by_id.get(pron.clause_id) if pron.clause_id else None
        sent_id = clause.sentence_id if clause else ""

        resolutions.append(
            AnaphoraResolution(
                pronoun_node_id=pron.id,
                antecedent_node_id=target_id,
                sentence_id=sent_id,
                pronoun_surface=pron.surface or pron.label,
                pronoun_lemma=pron.lemma or pron.label,
                antecedent_lemma=target_lemma or target_label,
                pronoun_type=ptype,
                resolution_strategy=strategy,
                matched_features=matched_feats,
                distance_sentences=sent_distance,
                salience_score=score,
            )
        )

    if not resolutions:
        return graph, []

    # Применяем замены: PRON-узлы остаются, обновляем атрибуты.
    final_nodes: list[Node] = []
    for n in graph.nodes:
        if n.id in replacement_map:
            new_label, new_lemma, new_upos, antecedent_id = replacement_map[n.id]
            prov = n.provenance
            new_inputs = list(prov.inputs)
            if antecedent_id not in new_inputs:
                new_inputs.append(antecedent_id)
            notes_parts = [prov.notes] if prov.notes else []
            notes_parts.append(
                f"anaphora_replaced: original_lemma={n.lemma}; "
                f"antecedent={antecedent_id}"
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
            final_nodes.append(
                n.model_copy(
                    update={
                        "label": new_label,
                        "lemma": new_lemma,
                        "upos": new_upos,
                        "original_lemma": n.lemma,
                        "original_upos": n.upos,
                        "antecedent_node_id": antecedent_id,
                        "provenance": new_prov,
                    }
                )
            )
        else:
            final_nodes.append(n)

    new_graph = SemanticGraph(
        document_id=graph.document_id,
        nodes=final_nodes,
        edges=list(graph.edges),
    )
    return new_graph, resolutions
