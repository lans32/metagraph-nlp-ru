"""Тесты CoNLL-U парсера и TreeTaggerMaltParser-адаптера."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from metagraph_nlp.parsers.morphsyntax.maltparser_adapter import (
    TreeTaggerMaltParser,
    build_conll_input,
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


def test_parse_conllu_with_explicit_spans():
    """Позиционный mapping через spans закрывает повторы словоформ."""
    sent = "Он сказал, что он пришёл."
    conllu = (
        "1\tОн\tон\tPRON\t_\t_\t2\tnsubj\t_\t_\n"
        "2\tсказал\tсказать\tVERB\t_\t_\t0\troot\t_\t_\n"
        "3\t,\t,\tPUNCT\t_\t_\t2\tpunct\t_\t_\n"
        "4\tчто\tчто\tSCONJ\t_\t_\t6\tmark\t_\t_\n"
        "5\tон\tон\tPRON\t_\t_\t6\tnsubj\t_\t_\n"
        "6\tпришёл\tприйти\tVERB\t_\t_\t2\tccomp\t_\t_\n"
        "7\t.\t.\tPUNCT\t_\t_\t2\tpunct\t_\t_\n"
    )
    spans = [(0, 2), (3, 9), (9, 10), (11, 14), (15, 17), (18, 24), (24, 25)]
    tokens = parse_conllu(conllu, sent, spans=spans)
    assert tokens[0].start == 0 and tokens[0].end == 2
    assert tokens[4].start == 15 and tokens[4].end == 17
    assert sent[tokens[4].start:tokens[4].end] == "он"


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


def test_build_conll_input_fills_lemma_pos_feats():
    """Главный contract: morpho-поля в input MaltParser больше не пустые."""
    tagged = [
        ("Студент", "студент", "NOUN", {"Gender": "Masc", "Case": "Nom"}),
        ("читает", "читать", "VERB", {"Tense": "Pres", "Person": "3"}),
        ("книгу", "книга", "NOUN", {"Gender": "Fem", "Case": "Acc"}),
    ]
    conll = build_conll_input(tagged)
    lines = [line for line in conll.split("\n") if line.strip()]
    assert len(lines) == 3
    cols = lines[0].split("\t")
    assert cols[2] == "студент", "lemma в колонке 3 (CoNLL LEMMA)"
    assert cols[3] == "NOUN", "upos в колонке 4 (CoNLL CPOSTAG)"
    assert "Case=Nom" in cols[5] and "Gender=Masc" in cols[5], \
        "feats в колонке 6 (CoNLL FEATS) — Case и Gender присутствуют"
    assert cols[5] != "_", "feats не должно быть `_` (слепой режим устранён)"


def test_treetagger_malt_parser_rejects_missing_jar(tmp_path):
    morph = MagicMock()
    with pytest.raises(FileNotFoundError, match="jar not found"):
        TreeTaggerMaltParser(
            malt_jar=tmp_path / "nonexistent.jar",
            model_path=tmp_path / "model.mco",
            morph=morph,
        )


def test_treetagger_malt_parser_rejects_missing_model(tmp_path):
    morph = MagicMock()
    jar = tmp_path / "fake.jar"
    jar.write_text("")
    with pytest.raises(FileNotFoundError, match="model not found"):
        TreeTaggerMaltParser(
            malt_jar=jar,
            model_path=tmp_path / "nonexistent.mco",
            morph=morph,
        )


def test_parser_factory_unknown_parser():
    from metagraph_nlp.config import Config, MorphSyntaxConfig
    from metagraph_nlp.pipeline import get_default_parser

    cfg = Config(morphsyntax=MorphSyntaxConfig(parser="unknown_parser"))
    with pytest.raises(ValueError, match="Unknown parser"):
        get_default_parser(cfg)


def test_factory_maltparser_reports_missing_fields():
    """Factory должна указывать конкретно какие поля недостающие."""
    from metagraph_nlp.config import Config, MorphSyntaxConfig
    from metagraph_nlp.pipeline import get_default_parser

    cfg = Config(morphsyntax=MorphSyntaxConfig(parser="maltparser"))
    with pytest.raises(ValueError) as exc:
        get_default_parser(cfg)
    msg = str(exc.value)
    assert "tree_tagger_bin" in msg
    assert "tree_tagger_param" in msg
    assert "malt_jar" in msg
    assert "malt_model" in msg
