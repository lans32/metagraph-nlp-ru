"""Тесты для NP collapse v1: свёртка именных групп через UD-subtree.

В отличие от v0, версия v1 берёт модификаторы напрямую из
`ParsedSentence` (через `Node.token_id_in_sent`), не плодит промежуточные
узлы и не меняет топологию графа. В графе хранится только головной
NOUN/PROPN; модификаторы (amod, det, nummod, appos, flat, nmod:poss)
встраиваются в его лемму в порядке `Token.id_in_sent`.
"""

from __future__ import annotations

from metagraph_nlp.domain import (
    Clause,
    Edge,
    IdFactory,
    Node,
    Provenance,
    SemanticGraph,
)
from metagraph_nlp.domain.text import TextSpan
from metagraph_nlp.graph_builders.np_collapse import collapse_noun_phrases
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

_PROV = Provenance(rule="test", stage="test", inputs=[], document_id="doc-1")


def _make_parsed(tokens: list[Token], text: str) -> ParsedSentence:
    return ParsedSentence(text=text, tokens=tokens)


def _clause(text: str, head_lemma: str) -> Clause:
    return Clause(
        id="cl-1",
        sentence_id="s-1",
        document_id="doc-1",
        span=TextSpan(start=0, end=len(text), text=text),
        head_lemma=head_lemma,
        provenance=_PROV,
    )


def _noun_node(node_id: str, lemma: str, token_id: int, *, upos: str = "NOUN") -> Node:
    return Node(
        id=node_id,
        label=lemma,
        kind="concept",
        lemma=lemma,
        surface=lemma,
        upos=upos,
        clause_id="cl-1",
        token_id_in_sent=token_id,
        provenance=_PROV,
    )


def test_left_amod_collapsed():
    """'молодой исследователь' → 'молодой исследователь' (amod слева от NOUN)."""
    text = "молодой исследователь"
    tokens = [
        Token(id_in_sent=1, text="Молодой", lemma="молодой", pos="ADJ",
              feats={}, head=2, deprel="amod", start=0, end=7),
        Token(id_in_sent=2, text="исследователь", lemma="исследователь",
              pos="NOUN", feats={}, head=0, deprel="root", start=8, end=21),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[_noun_node("n-noun", "исследователь", token_id=2)],
        edges=[],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "исследователь")], IdFactory())

    assert len(result.nodes) == 1
    n = result.nodes[0]
    assert n.lemma == "молодой исследователь"
    assert n.original_lemma == "исследователь"
    assert n.original_upos == "NOUN"
    assert n.provenance.rule == "np_collapse_v1"


def test_right_modifier_via_appos():
    """'исследователь Иван' (appos справа) → лемма в порядке token_id."""
    text = "исследователь Иван"
    tokens = [
        Token(id_in_sent=1, text="исследователь", lemma="исследователь",
              pos="NOUN", feats={}, head=0, deprel="root", start=0, end=13),
        Token(id_in_sent=2, text="Иван", lemma="иван", pos="PROPN",
              feats={}, head=1, deprel="appos", start=14, end=18),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[_noun_node("n-noun", "исследователь", token_id=1)],
        edges=[],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "исследователь")], IdFactory())

    assert len(result.nodes) == 1
    assert result.nodes[0].lemma == "исследователь иван", (
        "appos справа от головы — лемма в порядке token_id (head раньше mod)"
    )


def test_flat_name_chain_collapsed():
    """'Иван Иванович Петров' через flat — один узел."""
    text = "Иван Иванович Петров"
    tokens = [
        Token(id_in_sent=1, text="Иван", lemma="иван", pos="PROPN",
              feats={}, head=0, deprel="root", start=0, end=4),
        Token(id_in_sent=2, text="Иванович", lemma="иванович", pos="PROPN",
              feats={}, head=1, deprel="flat:name", start=5, end=13),
        Token(id_in_sent=3, text="Петров", lemma="петров", pos="PROPN",
              feats={}, head=1, deprel="flat:name", start=14, end=20),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[_noun_node("n-propn", "иван", token_id=1, upos="PROPN")],
        edges=[],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "иван")], IdFactory())

    assert result.nodes[0].lemma == "иван иванович петров"


def test_nummod_collapsed():
    """'два студента' (nummod) → один узел 'два студент'."""
    text = "два студента"
    tokens = [
        Token(id_in_sent=1, text="два", lemma="два", pos="NUM",
              feats={"NumType": "Card"}, head=2, deprel="nummod", start=0, end=3),
        Token(id_in_sent=2, text="студента", lemma="студент", pos="NOUN",
              feats={}, head=0, deprel="root", start=4, end=12),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[_noun_node("n-noun", "студент", token_id=2)],
        edges=[],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "студент")], IdFactory())

    assert result.nodes[0].lemma == "два студент"


def test_nmod_without_poss_is_not_collapsed():
    """'анализ клауз' (nmod без :poss) — НЕ сворачивается."""
    text = "анализ клауз"
    tokens = [
        Token(id_in_sent=1, text="анализ", lemma="анализ", pos="NOUN",
              feats={}, head=0, deprel="root", start=0, end=6),
        Token(id_in_sent=2, text="клауз", lemma="клауза", pos="NOUN",
              feats={"Case": "Gen"}, head=1, deprel="nmod", start=7, end=12),
    ]
    parsed = _make_parsed(tokens, text)
    # В реальном builder обе ноды будут в графе (`_expand_nmod` создаёт
    # узел для `nmod`-токена). Здесь имитируем такую ситуацию.
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            _noun_node("n-head", "анализ", token_id=1),
            _noun_node("n-nmod", "клауза", token_id=2),
        ],
        edges=[
            Edge(id="e-1", source="n-head", target="n-nmod",
                 relation="nmod", clause_id="cl-1", provenance=_PROV),
        ],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "анализ")], IdFactory())

    # Обе ноды на месте, лемма «анализ» не изменилась, ребро тоже на месте.
    assert len(result.nodes) == 2
    assert {n.lemma for n in result.nodes} == {"анализ", "клауза"}
    assert len(result.edges) == 1


def test_nested_modifiers_collected_in_one_pass():
    """Вложенный модификатор-головы 'русские клаузы' внутри NP головы NOUN.

    Сценарий: NP-голова 'факультет' имеет amod-ребёнка 'филологический',
    у которого нет вложенностей. Здесь нет реального вложения; используем
    более сложный кейс: 'умная молодая женщина' — оба amod к 'женщина'.
    """
    text = "умная молодая женщина"
    tokens = [
        Token(id_in_sent=1, text="умная", lemma="умный", pos="ADJ",
              feats={}, head=3, deprel="amod", start=0, end=5),
        Token(id_in_sent=2, text="молодая", lemma="молодой", pos="ADJ",
              feats={}, head=3, deprel="amod", start=6, end=13),
        Token(id_in_sent=3, text="женщина", lemma="женщина", pos="NOUN",
              feats={}, head=0, deprel="root", start=14, end=21),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[_noun_node("n-noun", "женщина", token_id=3)],
        edges=[],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "женщина")], IdFactory())

    assert result.nodes[0].lemma == "умный молодой женщина"


def test_no_modifiers_no_change():
    """Одиночный NOUN без модификаторов не меняется."""
    text = "кот"
    tokens = [
        Token(id_in_sent=1, text="кот", lemma="кот", pos="NOUN",
              feats={}, head=0, deprel="root", start=0, end=3),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[_noun_node("n-1", "кот", token_id=1)],
        edges=[],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "кот")], IdFactory())

    assert len(result.nodes) == 1
    n = result.nodes[0]
    assert n.lemma == "кот"
    assert n.original_lemma is None, "если не было свёртки, original_lemma не выставляется"


def test_verb_node_not_collapsed():
    """VERB-узлы не участвуют в NP collapse даже при наличии детей."""
    text = "бежит кот"
    tokens = [
        Token(id_in_sent=1, text="бежит", lemma="бежать", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=0, end=5),
        Token(id_in_sent=2, text="кот", lemma="кот", pos="NOUN",
              feats={}, head=1, deprel="nsubj", start=6, end=9),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            Node(id="n-v", label="бежать", kind="predicate", lemma="бежать",
                 surface="бежит", upos="VERB", clause_id="cl-1",
                 token_id_in_sent=1, provenance=_PROV),
            _noun_node("n-n", "кот", token_id=2),
        ],
        edges=[
            Edge(id="e-1", source="n-v", target="n-n", relation="nsubj",
                 clause_id="cl-1", provenance=_PROV),
        ],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "бежать")], IdFactory())

    assert len(result.nodes) == 2
    assert len(result.edges) == 1
    verb = next(n for n in result.nodes if n.upos == "VERB")
    assert verb.lemma == "бежать"


def test_topology_unchanged_other_nodes_pass_through():
    """v1 не удаляет узлы и не трогает рёбра — обновляет только NOUN-головы."""
    text = "видит большой кот"
    tokens = [
        Token(id_in_sent=1, text="видит", lemma="видеть", pos="VERB",
              feats={}, head=0, deprel="root", start=0, end=5),
        Token(id_in_sent=2, text="большой", lemma="большой", pos="ADJ",
              feats={}, head=3, deprel="amod", start=6, end=13),
        Token(id_in_sent=3, text="кот", lemma="кот", pos="NOUN",
              feats={}, head=1, deprel="nsubj", start=14, end=17),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            Node(id="n-v", label="видеть", kind="predicate", lemma="видеть",
                 surface="видит", upos="VERB", clause_id="cl-1",
                 token_id_in_sent=1, provenance=_PROV),
            _noun_node("n-noun", "кот", token_id=3),
        ],
        edges=[
            Edge(id="e-1", source="n-v", target="n-noun", relation="видеть",
                 clause_id="cl-1", provenance=_PROV),
        ],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "видеть")], IdFactory())

    assert len(result.nodes) == 2
    # Ребро VERB → NOUN остаётся таким же.
    assert result.edges[0].source == "n-v"
    assert result.edges[0].target == "n-noun"
    # Лемма NOUN-головы обновилась.
    noun = next(n for n in result.nodes if n.upos == "NOUN")
    assert noun.lemma == "большой кот"


def test_repeated_lemma_disambiguated_by_token_id():
    """Повтор леммы в предложении: оба NOUN сворачиваются корректно через token_id."""
    text = "новый дом и старый дом"
    tokens = [
        Token(id_in_sent=1, text="новый", lemma="новый", pos="ADJ",
              feats={}, head=2, deprel="amod", start=0, end=5),
        Token(id_in_sent=2, text="дом", lemma="дом", pos="NOUN",
              feats={}, head=0, deprel="root", start=6, end=9),
        Token(id_in_sent=3, text="и", lemma="и", pos="CCONJ",
              feats={}, head=5, deprel="cc", start=10, end=11),
        Token(id_in_sent=4, text="старый", lemma="старый", pos="ADJ",
              feats={}, head=5, deprel="amod", start=12, end=18),
        Token(id_in_sent=5, text="дом", lemma="дом", pos="NOUN",
              feats={}, head=2, deprel="conj", start=19, end=22),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[
            _noun_node("n-1", "дом", token_id=2),
            _noun_node("n-2", "дом", token_id=5),
        ],
        edges=[],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "дом")], IdFactory())

    lemmas = {n.id: n.lemma for n in result.nodes}
    assert lemmas["n-1"] == "новый дом"
    assert lemmas["n-2"] == "старый дом", (
        "второй 'дом' свёрнут с собственным amod 'старый', а не с чужим 'новый'"
    )


def test_provenance_records_modifier_token_ids():
    """В Provenance.notes лежат token_ids модификаторов."""
    text = "большая книга"
    tokens = [
        Token(id_in_sent=1, text="большая", lemma="большой", pos="ADJ",
              feats={}, head=2, deprel="amod", start=0, end=7),
        Token(id_in_sent=2, text="книга", lemma="книга", pos="NOUN",
              feats={}, head=0, deprel="root", start=8, end=13),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[_noun_node("n-noun", "книга", token_id=2)],
        edges=[],
    )
    result = collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "книга")], IdFactory())

    notes = result.nodes[0].provenance.notes
    assert "modifier_token_ids=[1]" in notes
    assert "collapsed_np=большой книга" in notes


def test_original_graph_not_mutated():
    """collapse_noun_phrases возвращает новый граф, не мутирует исходный."""
    text = "большая книга"
    tokens = [
        Token(id_in_sent=1, text="большая", lemma="большой", pos="ADJ",
              feats={}, head=2, deprel="amod", start=0, end=7),
        Token(id_in_sent=2, text="книга", lemma="книга", pos="NOUN",
              feats={}, head=0, deprel="root", start=8, end=13),
    ]
    parsed = _make_parsed(tokens, text)
    original_node = _noun_node("n-noun", "книга", token_id=2)
    graph = SemanticGraph(document_id="doc-1", nodes=[original_node], edges=[])

    collapse_noun_phrases(graph, {"s-1": parsed}, [_clause(text, "книга")], IdFactory())

    assert graph.nodes[0].lemma == "книга", "исходный граф не мутируется"
    assert graph.nodes[0].original_lemma is None


def test_include_deprels_override_excludes_nummod():
    """Если убрать nummod из include_deprels — числительное не сворачивается."""
    text = "два студента"
    tokens = [
        Token(id_in_sent=1, text="два", lemma="два", pos="NUM",
              feats={}, head=2, deprel="nummod", start=0, end=3),
        Token(id_in_sent=2, text="студента", lemma="студент", pos="NOUN",
              feats={}, head=0, deprel="root", start=4, end=12),
    ]
    parsed = _make_parsed(tokens, text)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[_noun_node("n-noun", "студент", token_id=2)],
        edges=[],
    )
    result = collapse_noun_phrases(
        graph,
        {"s-1": parsed},
        [_clause(text, "студент")],
        IdFactory(),
        include_deprels={"amod", "det", "appos", "flat", "flat:name", "nmod:poss"},
    )

    assert result.nodes[0].lemma == "студент", (
        "nummod не в include_deprels — числительное игнорируется"
    )
