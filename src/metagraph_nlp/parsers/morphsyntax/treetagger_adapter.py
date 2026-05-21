"""TreeTagger как морфо-провайдер (lemma + UPOS + feats).

TreeTagger — статистический HMM/decision-tree теггер (Schmid, 1994). С
параметр-файлом `russian.par` (Sharoff & Nivre, 2011) выдаёт MSD-Russian
теги вида `Ncmsny` и леммы. Этот адаптер — subprocess-обёртка над
`tree-tagger` (Windows `.exe` или Unix shell-скрипт), которая принимает
список токенов и возвращает кортежи (text, lemma, upos, feats) для каждого.

TreeTagger в этом проекте **не** реализует `MorphSyntaxParser` Protocol
напрямую: он не даёт head/deprel. Он используется как `morph_provider`
внутри `TreeTaggerMaltParser`, который и собирает финальный
`ParsedSentence`. Это держит контракт Protocol чистым.

Если TreeTagger вернул `<unknown>` или сохранил surface как лемму, для
русских слов делается fallback на pymorphy3 — по образцу
[natasha_adapter._fallback_lemma](natasha_adapter.py:52).
"""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

from metagraph_nlp.parsers.morphsyntax.treetagger_tagmap import MsdTagMapper

TaggedToken = tuple[str, str, str, dict[str, str]]


class TreeTaggerMorph:
    """Морфо-разметка через TreeTagger: токены → (text, lemma, upos, feats)."""

    _UNKNOWN_LEMMA_MARKERS: frozenset[str] = frozenset({"<unknown>", "@card@"})
    _FALLBACK_SKIP_UPOS: frozenset[str] = frozenset({"PUNCT", "SYM", "X"})

    def __init__(
        self,
        bin_path: Path,
        param_path: Path,
        tagmap: MsdTagMapper,
        abbrev_path: Path | None = None,
        timeout_sec: int = 60,
    ) -> None:
        self._bin = Path(bin_path)
        self._param = Path(param_path)
        self._tagmap = tagmap
        self._abbrev = Path(abbrev_path) if abbrev_path else None
        self._timeout = timeout_sec
        if not self._bin.exists():
            raise FileNotFoundError(f"TreeTagger binary not found: {self._bin}")
        if not self._param.exists():
            raise FileNotFoundError(f"TreeTagger param file not found: {self._param}")

        import pymorphy3

        self._morph = pymorphy3.MorphAnalyzer()

    def tag(self, tokens: list[str]) -> list[TaggedToken]:
        """Разметить список токенов. Возвращает строго len(tokens) элементов."""
        if not tokens:
            return []

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".tt-in.txt", delete=False, encoding="utf-8"
        ) as f_in:
            f_in.write("\n".join(tokens))
            f_in.write("\n")
            in_path = Path(f_in.name)
        out_path = in_path.with_suffix(".tt-out.txt")

        try:
            cmd: list[str] = [
                str(self._bin),
                str(self._param),
                "-token",
                "-lemma",
                "-sgml",
                str(in_path),
                str(out_path),
            ]
            env = dict(os.environ)
            env.setdefault("PYTHONIOENCODING", "utf-8")
            env.setdefault("LC_ALL", "C.UTF-8")
            env.setdefault("LANG", "ru_RU.UTF-8")
            subprocess.run(
                cmd,
                check=True,
                timeout=self._timeout,
                capture_output=True,
                env=env,
            )
            raw_output = out_path.read_text(encoding="utf-8")
        finally:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)

        parsed = self._parse_output(raw_output)
        if len(parsed) != len(tokens):
            parsed = self._align_to_input(parsed, tokens)

        result: list[TaggedToken] = []
        for (text, msd_tag, raw_lemma), input_tok in zip(parsed, tokens):
            upos, feats = self._tagmap.map(msd_tag, lemma=raw_lemma)
            lemma = self._resolve_lemma(input_tok, raw_lemma, upos)
            result.append((input_tok, lemma, upos, feats))
        return result

    @staticmethod
    def _parse_output(raw: str) -> list[tuple[str, str, str]]:
        """Парсинг TreeTagger output: `form\\ttag\\tlemma` на строку."""
        result: list[tuple[str, str, str]] = []
        for line in raw.splitlines():
            if not line.strip():
                continue
            parts = line.split("\t")
            if len(parts) >= 3:
                result.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                result.append((parts[0], parts[1], parts[0]))
        return result

    @staticmethod
    def _align_to_input(
        parsed: list[tuple[str, str, str]],
        input_tokens: list[str],
    ) -> list[tuple[str, str, str]]:
        """Защита от расхождения N_out != N_in.

        Greedy-align по text: для каждого входного токена ищем первое
        совпадение в `parsed`, остальное пропускаем. Если совпадения нет —
        вставляем пустую запись (X, X, surface).
        """
        result: list[tuple[str, str, str]] = []
        cursor = 0
        for tok in input_tokens:
            matched = False
            for i in range(cursor, len(parsed)):
                if parsed[i][0] == tok:
                    result.append(parsed[i])
                    cursor = i + 1
                    matched = True
                    break
            if not matched:
                result.append((tok, "X", tok))
        return result

    def _resolve_lemma(self, surface: str, raw_lemma: str, upos: str) -> str:
        """Если TreeTagger не дал лемму — fallback через pymorphy3."""
        if (
            raw_lemma
            and raw_lemma not in self._UNKNOWN_LEMMA_MARKERS
            and raw_lemma.lower() != surface.lower()
        ):
            return raw_lemma.lower()
        if upos in self._FALLBACK_SKIP_UPOS:
            return surface.lower()
        if not any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in surface):
            return surface.lower()
        parses = self._morph.parse(surface)
        if not parses:
            return surface.lower()
        return parses[0].normal_form
