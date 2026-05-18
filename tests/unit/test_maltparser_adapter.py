"""Тесты CoNLL-U парсера и MaltParser адаптера."""

from __future__ import annotations

import pytest

from metagraph_nlp.parsers.morphsyntax.maltparser_adapter import (
    MaltParserAdapter,
    parse_conllu,
)


SAMPLE_CONLLU = """\
1\tСтудент\tстудент\tNOUN\t_\tCase=Nom|Number=Sing\t2\tnsubj\t_\t_
2\tчитает\tчитать\tVERB\t_\tVerbForm=Fin\t0\troot\t_\t_
3\tкнигу\tкнига\tNOUN\t_\tCase=Acc|Number=Sing\t2\tobj\t_\t_
4\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_
"""


def test_parse_conllu_basic():
    sent = "Студент читает книгу."
    tokens = parse_conllu(SAMPLE_CONLLU, sent)

    assert len(tokens) == 4
    assert tokens[0].text == "Студент"
    assert tokens[0].lemma == "студент"
    assert tokens[0].pos == "NOUN"
    assert tokens[0].head == 2
    assert tokens[0].deprel == "nsubj"
    assert tokens[0].feats == {"Case": "Nom", "Number": "Sing"}

    assert tokens[1].text == "читает"
    assert tokens[1].pos == "VERB"
    assert tokens[1].head == 0
    assert tokens[1].deprel == "root"
    assert tokens[1].feats == {"VerbForm": "Fin"}


def test_parse_conllu_offsets():
    sent = "Студент читает книгу."
    tokens = parse_conllu(SAMPLE_CONLLU, sent)

    assert tokens[0].start == 0
    assert tokens[0].end == 7
    assert tokens[1].start == 8
    assert tokens[1].end == 14
    assert tokens[2].start == 15
    assert tokens[2].end == 20


def test_parse_conllu_skips_multiword_tokens():
    conllu_with_mwt = """\
1-2\tнебудет\t_\t_\t_\t_\t_\t_\t_\t_
1\tне\tне\tPART\t_\t_\t2\tadvmod\t_\t_
2\tбудет\tбыть\tVERB\t_\t_\t0\troot\t_\t_
"""
    tokens = parse_conllu(conllu_with_mwt, "не будет")
    assert len(tokens) == 2
    assert tokens[0].text == "не"
    assert tokens[1].text == "будет"


def test_parse_conllu_handles_empty_feats():
    conllu = "1\tслово\tслово\tNOUN\t_\t_\t0\troot\t_\t_\n"
    tokens = parse_conllu(conllu, "слово")
    assert tokens[0].feats == {}


def test_maltparser_adapter_rejects_missing_jar(tmp_path):
    with pytest.raises(FileNotFoundError, match="jar not found"):
        MaltParserAdapter(
            malt_jar=tmp_path / "nonexistent.jar",
            model_path=tmp_path / "model.mco",
        )


def test_parser_factory_unknown_parser():
    from metagraph_nlp.config import Config, MorphSyntaxConfig
    from metagraph_nlp.pipeline import get_default_parser

    cfg = Config(morphsyntax=MorphSyntaxConfig(parser="unknown_parser"))
    with pytest.raises(ValueError, match="Unknown parser"):
        get_default_parser(cfg)
