"""Нормализация текста, сегментация, выделение клауз."""

from metagraph_nlp.parsers.anaphora import resolve_anaphora
from metagraph_nlp.parsers.clauses import extract_clauses
from metagraph_nlp.parsers.normalize import normalize_text
from metagraph_nlp.parsers.segment import split_sentences

__all__ = ["extract_clauses", "normalize_text", "resolve_anaphora", "split_sentences"]
