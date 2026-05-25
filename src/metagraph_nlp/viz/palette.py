"""Детерминированная палитра для раскраски узлов по ключу (clause_id, mnode_id).

Без зависимостей: стабильный hash → индекс в фиксированной палитре.
Один и тот же ключ всегда даёт один и тот же цвет в рамках одного запуска
и между запусками (важно для воспроизводимости артефактов).
"""

from __future__ import annotations

import hashlib

# Достаточно различимая палитра приятных пастельных тонов.
_PALETTE: tuple[str, ...] = (
    "#8ecae6",
    "#ffb703",
    "#fb8500",
    "#90be6d",
    "#f4a261",
    "#bc6c25",
    "#cdb4db",
    "#a8dadc",
    "#e29578",
    "#b5e48c",
    "#f9c74f",
    "#9d4edd",
)


def color_for(key: str | None) -> str:
    """Вернуть стабильный hex-цвет по строковому ключу."""
    if not key:
        return "#cccccc"
    digest = hashlib.md5(key.encode("utf-8")).digest()
    idx = digest[0] % len(_PALETTE)
    return _PALETTE[idx]
