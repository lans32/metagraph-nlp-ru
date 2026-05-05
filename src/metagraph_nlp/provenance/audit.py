"""Журнал событий pipeline (audit log).

Каждая стадия пишет сюда событие: какое правило применилось, что было на
входе, что создано. Сериализуется в JSONL.
"""

from __future__ import annotations

from datetime import datetime, timezone

from pydantic import BaseModel, Field


class AuditEvent(BaseModel):
    stage: str
    rule: str
    inputs: list[str] = Field(default_factory=list)
    outputs: list[str] = Field(default_factory=list)
    params: dict[str, str] = Field(default_factory=dict)
    notes: str | None = None
    timestamp: str | None = Field(
        default=None,
        description="ISO 8601 timestamp в UTC.",
    )


class AuditLog(BaseModel):
    events: list[AuditEvent] = Field(default_factory=list)

    def record(
        self,
        stage: str,
        rule: str,
        inputs: list[str] | None = None,
        outputs: list[str] | None = None,
        params: dict[str, str] | None = None,
        notes: str | None = None,
    ) -> None:
        self.events.append(
            AuditEvent(
                stage=stage,
                rule=rule,
                inputs=inputs or [],
                outputs=outputs or [],
                params=params or {},
                notes=notes,
                timestamp=datetime.now(timezone.utc).isoformat(),
            )
        )

    def to_jsonl(self) -> str:
        return "\n".join(e.model_dump_json(exclude_none=True) for e in self.events)
