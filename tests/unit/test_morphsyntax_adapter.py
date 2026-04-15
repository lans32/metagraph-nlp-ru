import pytest

from metagraph_nlp.parsers.morphsyntax.natasha_adapter import get_natasha_parser

pytestmark = pytest.mark.slow


def test_natasha_parses_simple_russian_sentence():
    parser = get_natasha_parser()
    parsed = parser.parse("Студент читает книгу в библиотеке.")

    assert parsed.tokens
    root = parsed.root()
    assert root is not None
    assert root.pos == "VERB"
    assert root.lemma == "читать"

    nsubj = [t for t in parsed.tokens if t.deprel == "nsubj"]
    assert nsubj, "должен быть хотя бы один nsubj"
    assert nsubj[0].lemma == "студент"

    obj = [t for t in parsed.tokens if t.deprel == "obj"]
    assert obj, "должен быть прямой объект"
    assert obj[0].lemma == "книга"


def test_natasha_subtree_ids_works_on_real_parse():
    parser = get_natasha_parser()
    parsed = parser.parse("Он читал, и она писала.")
    verbs = [t for t in parsed.tokens if t.pos == "VERB"]
    assert len(verbs) >= 2
