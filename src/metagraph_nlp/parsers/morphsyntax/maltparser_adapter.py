"""Адаптер MaltParser → ParsedSentence через цепочку razdel → TreeTagger → MaltParser.

MaltParser (Nivre, 2003) — классический transition-based dependency parser,
который работает по уже размеченному входу (lemma + POS + feats). Голый
MaltParser без морфо-разметки даёт мусор, поэтому в этом проекте он
**всегда** идёт в связке с морфо-провайдером (по умолчанию TreeTagger).

Pipeline на одно предложение:
1. `razdel.tokenize` → список токенов с (text, start, end);
2. `TreeTaggerMorph.tag` → lemma + UPOS + feats для каждого токена;
3. Сериализация в CoNLL-U с заполненными колонками 3/4/6;
4. Запуск MaltParser-JVM через subprocess (tempfile);
5. Парсинг CoNLL-U output → list[Token] со spans, восстановленными
   позиционно по razdel-индексу (не через `text.find`, что ломалось бы
   на повторах словоформ).
6. Опциональное применение `deprel_mapping` (для не-UD моделей).
"""

from __future__ import annotations

import subprocess
import tempfile
from pathlib import Path

from metagraph_nlp.parsers.morphsyntax.base import MorphSyntaxParser
from metagraph_nlp.parsers.morphsyntax.treetagger_adapter import TreeTaggerMorph
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


def _format_feats(feats: dict[str, str]) -> str:
    if not feats:
        return "_"
    return "|".join(f"{k}={v}" for k, v in sorted(feats.items()))


def parse_conllu(
    text: str,
    sentence_text: str,
    *,
    spans: list[tuple[int, int]] | None = None,
) -> list[Token]:
    """Парсинг CoNLL-U формата в список Token.

    Если `spans` задан (список (start, end) длины N токенов), start/end
    берутся позиционно по `id_in_sent - 1`. Если `spans=None` — fallback
    на поиск `sentence_text.find(form, search_pos)` (ломкий на повторах).
    """
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
        tid = int(tid_str)
        form = fields[1]
        if spans is not None and 1 <= tid <= len(spans):
            start, end = spans[tid - 1]
        else:
            start = sentence_text.find(form, search_pos)
            if start == -1:
                start = search_pos
            end = start + len(form)
            search_pos = end

        tokens.append(
            Token(
                id_in_sent=tid,
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


class TreeTaggerMaltParser(MorphSyntaxParser):
    """Цепочка razdel → TreeTagger → MaltParser → ParsedSentence."""

    def __init__(
        self,
        malt_jar: Path,
        model_path: Path,
        morph: TreeTaggerMorph,
        java_bin: str = "java",
        deprel_mapping: dict[str, str] | None = None,
        timeout_sec: int = 60,
    ) -> None:
        self._malt_jar = Path(malt_jar)
        self._model_path = Path(model_path)
        self._java_bin = java_bin
        self._morph = morph
        self._deprel_map = deprel_mapping or {}
        self._timeout = timeout_sec
        if not self._malt_jar.exists():
            raise FileNotFoundError(f"MaltParser jar not found: {self._malt_jar}")
        if not self._model_path.exists():
            raise FileNotFoundError(f"MaltParser model not found: {self._model_path}")

    def parse(self, sentence_text: str) -> ParsedSentence:
        if not sentence_text.strip():
            return ParsedSentence(text=sentence_text, tokens=[])

        from razdel import tokenize as razdel_tokenize

        raw_tokens = list(razdel_tokenize(sentence_text))
        if not raw_tokens:
            return ParsedSentence(text=sentence_text, tokens=[])

        spans: list[tuple[int, int]] = [(t.start, t.stop) for t in raw_tokens]
        token_texts: list[str] = [t.text for t in raw_tokens]
        tagged = self._morph.tag(token_texts)

        conll_lines: list[str] = []
        for i, (text, lemma, upos, feats) in enumerate(tagged, start=1):
            conll_lines.append(
                "\t".join(
                    [
                        str(i),
                        text,
                        lemma if lemma else "_",
                        upos if upos else "_",
                        "_",
                        _format_feats(feats),
                        "_",
                        "_",
                        "_",
                        "_",
                    ]
                )
            )
        conll_input = "\n".join(conll_lines) + "\n\n"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".malt-in.conll", delete=False, encoding="utf-8"
        ) as f_in:
            f_in.write(conll_input)
            in_path = Path(f_in.name)
        out_path = in_path.with_suffix(".malt-out.conll")

        try:
            cmd: list[str] = [
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
            subprocess.run(cmd, check=True, timeout=self._timeout, capture_output=True)
            conll_output = out_path.read_text(encoding="utf-8")
        finally:
            in_path.unlink(missing_ok=True)
            out_path.unlink(missing_ok=True)

        tokens = parse_conllu(conll_output, sentence_text, spans=spans)
        if self._deprel_map:
            tokens = [
                t.model_copy(update={"deprel": self._deprel_map.get(t.deprel, t.deprel)})
                for t in tokens
            ]
        return ParsedSentence(text=sentence_text, tokens=tokens)


def build_conll_input(tagged: list[tuple[str, str, str, dict[str, str]]]) -> str:
    """Утилита: собрать CoNLL-U input из размеченных токенов.

    Полезна для тестов и diagnostics: проверить, что lemma/upos/feats
    действительно попадают в нужные колонки (а не подаются как `_`).
    """
    lines: list[str] = []
    for i, (text, lemma, upos, feats) in enumerate(tagged, start=1):
        lines.append(
            "\t".join(
                [
                    str(i),
                    text,
                    lemma if lemma else "_",
                    upos if upos else "_",
                    "_",
                    _format_feats(feats),
                    "_",
                    "_",
                    "_",
                    "_",
                ]
            )
        )
    return "\n".join(lines) + "\n\n"
