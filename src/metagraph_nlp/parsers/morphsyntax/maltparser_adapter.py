"""Адаптер MaltParser → ParsedSentence (CoNLL-U через subprocess).

MaltParser — классический transition-based dependency parser (Nivre, 2003).
Вызывается как Java-процесс, принимает CoNLL на stdin, отдаёт CoNLL на stdout.
Токенизация выполняется через razdel (или аналогичный токенизатор).

Для использования:
1. Установить Java ≥8.
2. Скачать maltparser-1.9.2.jar и обученную модель для русского языка.
3. Указать пути в конфиге: morphsyntax.malt_jar, morphsyntax.malt_model.
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token


def _parse_feats(feats_str: str) -> dict[str, str]:
    if feats_str == "_" or not feats_str:
        return {}
    result: dict[str, str] = {}
    for pair in feats_str.split("|"):
        if "=" in pair:
            k, v = pair.split("=", 1)
            result[k] = v
    return result


def parse_conllu(text: str, sentence_text: str) -> list[Token]:
    """Парсинг CoNLL-U формата в список Token с восстановлением офсетов."""
    tokens: list[Token] = []
    search_pos = 0
    for line in text.strip().split("\n"):
        if not line or line.startswith("#"):
            continue
        fields = line.split("\t")
        if len(fields) < 8:
            continue
        tid_str = fields[0]
        if "-" in tid_str or "." in tid_str:
            continue
        form = fields[1]
        start = sentence_text.find(form, search_pos)
        if start == -1:
            start = search_pos
        end = start + len(form)
        search_pos = end

        tokens.append(
            Token(
                id_in_sent=int(tid_str),
                text=form,
                lemma=fields[2] if fields[2] != "_" else form.lower(),
                pos=fields[3] if fields[3] != "_" else "X",
                feats=_parse_feats(fields[5]),
                head=int(fields[6]) if fields[6] != "_" else 0,
                deprel=fields[7] if fields[7] != "_" else "dep",
                start=start,
                end=end,
            )
        )
    return tokens


class MaltParserAdapter:
    """Вызов MaltParser через subprocess (Java); вход/выход в формате CoNLL."""

    def __init__(
        self,
        malt_jar: Path,
        model_path: Path,
        java_bin: str = "java",
    ) -> None:
        self._malt_jar = Path(malt_jar)
        self._model_path = Path(model_path)
        self._java_bin = java_bin
        if not self._malt_jar.exists():
            raise FileNotFoundError(f"MaltParser jar not found: {self._malt_jar}")
        if not self._model_path.exists():
            raise FileNotFoundError(f"MaltParser model not found: {self._model_path}")

    def parse(self, sentence_text: str) -> ParsedSentence:
        from razdel import tokenize as razdel_tokenize

        raw_tokens = list(razdel_tokenize(sentence_text))
        conll_lines: list[str] = []
        for i, tok in enumerate(raw_tokens, start=1):
            conll_lines.append(
                f"{i}\t{tok.text}\t_\t_\t_\t_\t_\t_\t_\t_"
            )
        conll_input = "\n".join(conll_lines) + "\n\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".conll", delete=False, encoding="utf-8"
        ) as f_in:
            f_in.write(conll_input)
            in_path = Path(f_in.name)

        out_path = in_path.with_suffix(".out.conll")
        try:
            cmd = [
                self._java_bin,
                "-jar",
                str(self._malt_jar),
                "-c",
                self._model_path.stem,
                "-i",
                str(in_path),
                "-o",
                str(out_path),
                "-m",
                "parse",
                "-w",
                str(self._model_path.parent),
            ]
            subprocess.run(cmd, check=True, timeout=60, capture_output=True)
            conll_output = out_path.read_text(encoding="utf-8")
        finally:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)

        tokens = parse_conllu(conll_output, sentence_text)
        return ParsedSentence(text=sentence_text, tokens=tokens)
