"""Адаптер natasha → наш внутренний ParsedSentence.

natasha — pure-Python NLP-пакет, специализированный под русский. Мы
используем его цепочку: Segmenter → NewsMorphTagger (Slovnet+Navec) →
NewsSyntaxParser → MorphVocab (лемматизация). Веса моделей скачиваются
один раз и кэшируются в `~/.natasha/`.

Инициализация тяжёлая (~1–2 секунды, модели ~150 МБ в памяти), поэтому
парсер создаётся как синглтон на процесс через `get_natasha_parser()`.
"""

from __future__ import annotations

from threading import Lock

from metagraph_nlp.parsers.morphsyntax.base import MorphSyntaxParser
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token


class NatashaParser(MorphSyntaxParser):
    """Ленивая обёртка над natasha-цепочкой.

    Лемматизация идёт в два прохода: сначала natasha.MorphVocab, затем —
    если natasha вернула surface без изменений для значимой части речи —
    pymorphy3 как fallback. Это лечит случаи, когда natasha не
    проанализировала словоформу (например, `теорему` → `теорему`).
    """

    def __init__(self) -> None:
        # Импорты лениво внутри __init__, чтобы модуль можно было импортировать
        # в тестах без установленной natasha (для fake-парсеров).
        from natasha import (
            Doc,
            MorphVocab,
            NewsEmbedding,
            NewsMorphTagger,
            NewsSyntaxParser,
            Segmenter,
        )
        import pymorphy3

        self._Doc = Doc
        self._segmenter = Segmenter()
        self._morph_vocab = MorphVocab()
        emb = NewsEmbedding()
        self._morph_tagger = NewsMorphTagger(emb)
        self._syntax_parser = NewsSyntaxParser(emb)
        self._morph = pymorphy3.MorphAnalyzer()

    _FALLBACK_SKIP_POS: frozenset[str] = frozenset({"PUNCT", "SYM"})

    def _fallback_lemma(self, text: str, pos: str) -> str:
        if pos in self._FALLBACK_SKIP_POS:
            return text.lower()
        if not any("а" <= ch.lower() <= "я" or ch.lower() == "ё" for ch in text):
            return text.lower()
        parses = self._morph.parse(text)
        if not parses:
            return text.lower()
        return parses[0].normal_form

    def parse(self, sentence_text: str) -> ParsedSentence:
        if not sentence_text.strip():
            return ParsedSentence(text=sentence_text, tokens=[])

        doc = self._Doc(sentence_text)
        doc.segment(self._segmenter)
        doc.tag_morph(self._morph_tagger)
        doc.parse_syntax(self._syntax_parser)

        for tok in doc.tokens:
            tok.lemmatize(self._morph_vocab)

        if not doc.sents:
            return ParsedSentence(text=sentence_text, tokens=[])

        # На вход нам подают уже одно предложение (razdel), но natasha может
        # внутри разбить его ещё раз — берём первый sent, а остальные токены
        # включаем в хвост под тем же предложением (редкий случай).
        sent = doc.sents[0]

        tokens: list[Token] = []
        # natasha выдаёт id вида "1_1", "1_2". Приводим к 1..N по порядку,
        # сохраняя соответствие head_id через mapping.
        id_map: dict[str, int] = {}
        for i, tok in enumerate(sent.tokens, start=1):
            id_map[tok.id] = i

        for tok in sent.tokens:
            local_id = id_map[tok.id]
            head_raw = getattr(tok, "head_id", None)
            if head_raw in (None, "0", 0):
                head = 0
            else:
                head = id_map.get(head_raw, 0)

            feats = dict(getattr(tok, "feats", None) or {})
            pos = tok.pos or "X"
            raw_lemma = tok.lemma
            if not raw_lemma or raw_lemma == tok.text:
                # natasha не изменила форму — пробуем pymorphy3.
                raw_lemma = self._fallback_lemma(tok.text, pos)
            lemma = raw_lemma.lower()
            # start/stop у natasha — смещения в исходном тексте.
            start = getattr(tok, "start", 0) or 0
            stop = getattr(tok, "stop", start + len(tok.text))
            tokens.append(
                Token(
                    id_in_sent=local_id,
                    text=tok.text,
                    lemma=lemma,
                    pos=pos,
                    feats=feats,
                    head=head,
                    deprel=(tok.rel or "dep"),
                    start=start,
                    end=stop,
                )
            )

        return ParsedSentence(text=sentence_text, tokens=tokens)


_parser: NatashaParser | None = None
_parser_lock = Lock()


def get_natasha_parser() -> NatashaParser:
    """Вернуть синглтон NatashaParser, создавая его при первом вызове."""
    global _parser
    if _parser is None:
        with _parser_lock:
            if _parser is None:
                _parser = NatashaParser()
    return _parser
