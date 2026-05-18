"""Нормализация сырого текста.

Простые детерминированные операции: Unicode NFC, схлопывание пробелов,
унификация кавычек, дефисов, невидимых символов. Без «умных» эвристик.
"""

from __future__ import annotations

import re
import unicodedata

_WS_RE = re.compile(r"[ \t\f\v]+")
_NL_RE = re.compile(r"\r\n?|\u2028|\u2029")
_MULTI_NL_RE = re.compile(r"\n{3,}")

_QUOTE_MAP = {
    "\u00ab": '"',
    "\u00bb": '"',
    "\u201c": '"',
    "\u201d": '"',
    "\u201e": '"',
    "\u2018": "'",
    "\u2019": "'",
    "\u201a": "'",
}

_DASH_MAP = {
    "\u2010": "-",
    "\u2011": "-",
    "\u2012": "-",
    "\u2013": "-",
    "\u2014": "-",
    "\u2212": "-",
}

_ZW = {"\u200b", "\u200c", "\u200d", "\ufeff"}


def normalize_text(raw: str) -> str:
    """Привести текст к каноническому виду для последующих стадий."""
    if not raw:
        return ""
    text = unicodedata.normalize("NFC", raw)
    text = "".join(ch for ch in text if ch not in _ZW)
    text = text.translate({ord(k): v for k, v in _QUOTE_MAP.items()})
    text = text.translate({ord(k): v for k, v in _DASH_MAP.items()})
    text = _NL_RE.sub("\n", text)
    text = _WS_RE.sub(" ", text)
    text = _MULTI_NL_RE.sub("\n\n", text)
    return text.strip()
