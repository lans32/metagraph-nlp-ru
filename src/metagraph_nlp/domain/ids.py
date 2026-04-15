"""Стабильная генерация идентификаторов в рамках одного запуска pipeline.

Инвариант CLAUDE.md §9.5: ID объектов должны быть стабильны внутри одного
запуска. Используем монотонные счётчики по типам сущностей.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class IdFactory:
    """Фабрика ID. Один экземпляр на запуск pipeline."""

    _counters: dict[str, int] = field(default_factory=dict)

    def next_id(self, kind: str) -> str:
        n = self._counters.get(kind, 0) + 1
        self._counters[kind] = n
        return f"{kind}_{n:06d}"

    def doc(self) -> str:
        return self.next_id("doc")

    def sent(self) -> str:
        return self.next_id("sent")

    def clause(self) -> str:
        return self.next_id("clause")

    def node(self) -> str:
        return self.next_id("node")

    def edge(self) -> str:
        return self.next_id("edge")

    def mnode(self) -> str:
        return self.next_id("mnode")

    def medge(self) -> str:
        return self.next_id("medge")
