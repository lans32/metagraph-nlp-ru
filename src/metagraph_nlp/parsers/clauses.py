"""Выделение клауз.

Две стратегии:

- ``sentence_as_clause_v0`` — раннее MVP-упрощение: одна клауза = одно
  предложение. Используется, если UD-разбор недоступен.
- ``ud_subtree_clauses_v0`` — выделение клауз по поддеревьям UD:
  финитный предикат (root/conj-VERB и подчинённые VERB через ccomp/xcomp/
  advcl/acl[:relcl]) становится головой клаузы; границы клаузы — минимум/
  максимум смещений токенов поддерева с «вырезанными» подчинёнными клаузами.

Правила детерминированы и трассируемы: каждой клаузе приписывается provenance
с именем правила и леммой предиката (см. CLAUDE.md §4.1, §9).
"""

from __future__ import annotations

from metagraph_nlp.domain import Clause, IdFactory, Provenance, Sentence
from metagraph_nlp.domain.text import TextSpan
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

_STAGE = "clauses"
_RULE_SENTENCE_AS_CLAUSE = "sentence_as_clause_v0"
_RULE_UD_SUBTREE = "ud_subtree_clauses_v0"

# Отношения, создающие вложенные/сочинённые клаузы. Вырезаются из поддерева
# родительского предиката, чтобы не перекрывать границы клауз.
_CLAUSE_BOUNDARY_DEPRELS: set[str] = {
    "ccomp",
    "xcomp",
    "advcl",
    "acl",
    "acl:relcl",
    "conj",
}

# Отношения, превращающие подчинённый VERB в отдельную вложенную клаузу
# (без conj — однородные сказуемые обрабатываются отдельно как «верхний
# уровень»).
_SUBORDINATE_CLAUSE_DEPRELS: set[str] = {
    "ccomp",
    "xcomp",
    "advcl",
    "acl",
    "acl:relcl",
}


def extract_clauses(
    sentences: list[Sentence],
    ids: IdFactory,
    parsed_sentences: dict[str, ParsedSentence] | None = None,
    strategy: str = "ud_subtree_clauses_v0",
) -> list[Clause]:
    if strategy == _RULE_SENTENCE_AS_CLAUSE or parsed_sentences is None:
        return [_sentence_as_clause(s, ids) for s in sentences]

    if strategy != _RULE_UD_SUBTREE:
        raise ValueError(f"Unknown clause extraction strategy: {strategy}")

    clauses: list[Clause] = []
    for s in sentences:
        parsed = parsed_sentences.get(s.id)
        if parsed is None or not parsed.tokens:
            clauses.append(_sentence_as_clause(s, ids, note_suffix="no ParsedSentence"))
            continue
        sent_clauses = _extract_ud_clauses(s, parsed, ids)
        if not sent_clauses:
            clauses.append(
                _sentence_as_clause(s, ids, note_suffix="fallback: no finite VERB")
            )
            continue
        clauses.extend(sent_clauses)

    clauses.sort(key=lambda c: (c.span.start, c.span.end))
    return clauses


def _sentence_as_clause(
    sentence: Sentence,
    ids: IdFactory,
    note_suffix: str | None = None,
) -> Clause:
    notes = "MVP stub: clause=sentence; реальные правила будут позже."
    if note_suffix:
        notes = f"{notes} [{note_suffix}]"
    return Clause(
        id=ids.clause(),
        sentence_id=sentence.id,
        document_id=sentence.document_id,
        span=sentence.span,
        head_text=None,
        head_lemma=None,
        provenance=Provenance(
            rule=_RULE_SENTENCE_AS_CLAUSE,
            stage=_STAGE,
            inputs=[sentence.id],
            document_id=sentence.document_id,
            sentence_id=sentence.id,
            notes=notes,
        ),
    )


def _is_finite_verb(token: Token) -> bool:
    if token.pos != "VERB":
        return False
    verb_form = token.feats.get("VerbForm")
    # natasha/pymorphy иногда не ставит VerbForm — считаем такие финитными.
    return verb_form in (None, "Fin")


def _is_participial(token: Token) -> bool:
    return token.pos == "VERB" and token.feats.get("VerbForm") == "Part"


def _collect_predicates(parsed: ParsedSentence) -> list[Token]:
    """Предикаты клауз: финитные VERB + причастия в предикативных позициях.

    Причастие считается предикатом, если:
    - deprel=root (краткое причастие как именное сказуемое);
    - deprel in _SUBORDINATE_CLAUSE_DEPRELS (вложенная клауза);
    - deprel in (acl, acl:relcl) (причастный оборот).
    """
    predicates: list[Token] = []
    seen: set[int] = set()

    for t in parsed.tokens:
        if t.id_in_sent in seen:
            continue

        if _is_finite_verb(t):
            if t.deprel == "root":
                predicates.append(t)
                seen.add(t.id_in_sent)
                continue
            if t.deprel == "conj":
                head = parsed.by_id(t.head)
                if head is not None and head.pos == "VERB":
                    predicates.append(t)
                    seen.add(t.id_in_sent)
                continue
            if t.deprel in _SUBORDINATE_CLAUSE_DEPRELS:
                predicates.append(t)
                seen.add(t.id_in_sent)
            continue

        if _is_participial(t):
            if t.deprel in ("root", "conj") or t.deprel in _SUBORDINATE_CLAUSE_DEPRELS:
                predicates.append(t)
                seen.add(t.id_in_sent)
                continue
            if t.deprel in ("acl", "acl:relcl"):
                predicates.append(t)
                seen.add(t.id_in_sent)

    return predicates


def _determine_clause_type(pred: Token, parsed: ParsedSentence) -> str:
    """Лингвистический тип клаузы по deprel предиката и морфологии."""
    if pred.deprel == "root":
        return "main"
    if pred.deprel == "conj":
        head = parsed.by_id(pred.head)
        if head is not None and head.pos == "VERB":
            return "coord"
        return "other"
    if pred.deprel == "ccomp":
        return "compl"
    if pred.deprel == "xcomp":
        return "xcompl"
    if pred.deprel == "advcl":
        return "adverbial"
    if pred.deprel == "acl:relcl":
        return "relative"
    if pred.deprel == "acl":
        if pred.feats.get("VerbForm") == "Part":
            return "participial"
        return "relative"
    return "other"


def _extract_ud_clauses(
    sentence: Sentence,
    parsed: ParsedSentence,
    ids: IdFactory,
) -> list[Clause]:
    predicates = _collect_predicates(parsed)
    if not predicates:
        return []

    clauses: list[Clause] = []
    for pred in predicates:
        subtree = parsed.subtree_ids(
            pred.id_in_sent, stop_deprels=_CLAUSE_BOUNDARY_DEPRELS
        )
        tokens = [parsed.by_id(tid) for tid in subtree]
        tokens = [t for t in tokens if t is not None]
        if not tokens:
            continue
        # PUNCT не влияет на границы span клаузы — иначе финальная точка
        # расширяет главную клаузу до конца предложения.
        content_tokens = [t for t in tokens if t.pos != "PUNCT"]
        if not content_tokens:
            content_tokens = tokens
        local_start = min(t.start for t in content_tokens)
        local_end = max(t.end for t in content_tokens)

        sent_text = sentence.span.text
        # Границы берём из токенов; текст режем по спану предложения.
        clause_text = sent_text[local_start:local_end]
        global_start = sentence.span.start + local_start
        global_end = sentence.span.start + local_end

        ct = _determine_clause_type(pred, parsed)
        clauses.append(
            Clause(
                id=ids.clause(),
                sentence_id=sentence.id,
                document_id=sentence.document_id,
                span=TextSpan(start=global_start, end=global_end, text=clause_text),
                head_text=pred.text,
                head_lemma=pred.lemma,
                clause_type=ct,
                provenance=Provenance(
                    rule=_RULE_UD_SUBTREE,
                    stage=_STAGE,
                    inputs=[sentence.id],
                    document_id=sentence.document_id,
                    sentence_id=sentence.id,
                    notes=f"predicate={pred.lemma}, type={ct}",
                ),
            )
        )

    clauses.sort(key=lambda c: (c.span.start, c.span.end))
    return clauses
