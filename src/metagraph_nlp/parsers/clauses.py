"""Выделение клауз.

MVP-заглушка: одна клауза = одно предложение. Это явная временная стратегия,
задокументированная в docs/journal/2026-04-15-bootstrap.md. Настоящие правила
(предикативный центр, сложноподчинённые конструкции, однородные сказуемые)
будут добавлены отдельной задачей под skill `rule-based-clause-extractor`.
"""

from __future__ import annotations

from metagraph_nlp.domain import Clause, IdFactory, Provenance, Sentence

_STAGE = "clauses"
_RULE_SENTENCE_AS_CLAUSE = "sentence_as_clause_v0"


def extract_clauses(sentences: list[Sentence], ids: IdFactory) -> list[Clause]:
    clauses: list[Clause] = []
    for s in sentences:
        clauses.append(
            Clause(
                id=ids.clause(),
                sentence_id=s.id,
                document_id=s.document_id,
                span=s.span,
                head_text=None,
                provenance=Provenance(
                    rule=_RULE_SENTENCE_AS_CLAUSE,
                    stage=_STAGE,
                    inputs=[s.id],
                    document_id=s.document_id,
                    sentence_id=s.id,
                    notes="MVP stub: clause=sentence; реальные правила будут позже.",
                ),
            )
        )
    return clauses
