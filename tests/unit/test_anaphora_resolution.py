"""Тесты разрешения анафоры (anaphora_resolution_v0)."""

from __future__ import annotations

from metagraph_nlp.domain import (
    Clause,
    Edge,
    IdFactory,
    Node,
    Provenance,
    SemanticGraph,
    Sentence,
)
from metagraph_nlp.domain.text import TextSpan
from metagraph_nlp.parsers.anaphora import resolve_anaphora
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token

_PROV = Provenance(rule="test", stage="test", inputs=[], document_id="doc-1")


def _sentence(sent_id: str, idx: int, text: str, start: int = 0) -> Sentence:
    return Sentence(
        id=sent_id,
        document_id="doc-1",
        index=idx,
        span=TextSpan(start=start, end=start + len(text), text=text),
        provenance=_PROV,
    )


def _clause(clause_id: str, sent_id: str, text: str) -> Clause:
    return Clause(
        id=clause_id, sentence_id=sent_id, document_id="doc-1",
        span=TextSpan(start=0, end=len(text), text=text),
        provenance=_PROV,
    )


def _node(
    node_id: str, lemma: str, upos: str, clause_id: str, token_id: int,
    surface: str | None = None,
) -> Node:
    return Node(
        id=node_id, label=lemma, lemma=lemma, surface=surface or lemma,
        upos=upos, clause_id=clause_id, token_id_in_sent=token_id,
        provenance=_PROV,
    )


def test_basic_replacement_redirects_edges():
    """«Иван пришёл. Он устал.» → ребро устать→PRON становится устать→Иван."""
    s1 = _sentence("s-1", 0, "Иван пришёл.")
    s2 = _sentence("s-2", 1, "Он устал.")
    c1 = _clause("cl-1", "s-1", "Иван пришёл.")
    c2 = _clause("cl-2", "s-2", "Он устал.")

    parsed_s1 = ParsedSentence(text="Иван пришёл.", tokens=[
        Token(id_in_sent=1, text="Иван", lemma="иван", pos="PROPN",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim"},
              head=2, deprel="nsubj", start=0, end=4),
        Token(id_in_sent=2, text="пришёл", lemma="прийти", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=5, end=11),
    ])
    parsed_s2 = ParsedSentence(text="Он устал.", tokens=[
        Token(id_in_sent=1, text="Он", lemma="он", pos="PRON",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim",
                     "PronType": "Prs", "Person": "3"},
              head=2, deprel="nsubj", start=0, end=2),
        Token(id_in_sent=2, text="устал", lemma="устать", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=3, end=8),
    ])

    n_ivan = _node("n-ivan", "иван", "PROPN", "cl-1", 1, "Иван")
    n_pron = _node("n-pron", "он", "PRON", "cl-2", 1, "Он")
    n_verb_pron = _node("n-verb2", "устать", "VERB", "cl-2", 2)

    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[n_ivan, n_pron, n_verb_pron],
        edges=[
            Edge(id="e-1", source=n_verb_pron.id, target=n_pron.id,
                 relation="устать", clause_id="cl-2", provenance=_PROV),
        ],
    )

    new_graph, resolutions = resolve_anaphora(
        graph, [c1, c2], [s1, s2],
        {"s-1": parsed_s1, "s-2": parsed_s2}, IdFactory(),
    )

    assert len(resolutions) == 1
    r = resolutions[0]
    assert r.pronoun_node_id == "n-pron"
    assert r.antecedent_node_id == "n-ivan"
    assert r.matched_features.get("Gender") == "Masc"
    assert r.matched_features.get("Animacy") == "Anim"

    assert "n-pron" not in {n.id for n in new_graph.nodes}
    assert any(e.target == "n-ivan" and e.source == "n-verb2" for e in new_graph.edges)
    edge = next(e for e in new_graph.edges if e.target == "n-ivan")
    assert edge.provenance.rule == "anaphora_resolution_v0"
    assert "n-pron" in edge.provenance.inputs


def test_gender_agreement_picks_correct_antecedent():
    """«Маша встретила Ивана. Он улыбнулся.» — он → Иван (Masc), не Маша."""
    s1 = _sentence("s-1", 0, "Маша встретила Ивана.")
    s2 = _sentence("s-2", 1, "Он улыбнулся.")
    c1 = _clause("cl-1", "s-1", "Маша встретила Ивана.")
    c2 = _clause("cl-2", "s-2", "Он улыбнулся.")

    parsed_s1 = ParsedSentence(text="Маша встретила Ивана.", tokens=[
        Token(id_in_sent=1, text="Маша", lemma="маша", pos="PROPN",
              feats={"Gender": "Fem", "Number": "Sing", "Animacy": "Anim"},
              head=2, deprel="nsubj", start=0, end=4),
        Token(id_in_sent=2, text="встретила", lemma="встретить", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=5, end=14),
        Token(id_in_sent=3, text="Ивана", lemma="иван", pos="PROPN",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim"},
              head=2, deprel="obj", start=15, end=20),
    ])
    parsed_s2 = ParsedSentence(text="Он улыбнулся.", tokens=[
        Token(id_in_sent=1, text="Он", lemma="он", pos="PRON",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim",
                     "PronType": "Prs", "Person": "3"},
              head=2, deprel="nsubj", start=0, end=2),
        Token(id_in_sent=2, text="улыбнулся", lemma="улыбнуться", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=3, end=12),
    ])

    n_masha = _node("n-masha", "маша", "PROPN", "cl-1", 1, "Маша")
    n_ivan = _node("n-ivan", "иван", "PROPN", "cl-1", 3, "Ивана")
    n_pron = _node("n-pron", "он", "PRON", "cl-2", 1, "Он")
    n_verb = _node("n-verb", "улыбнуться", "VERB", "cl-2", 2)

    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[n_masha, n_ivan, n_pron, n_verb],
        edges=[
            Edge(id="e-1", source=n_verb.id, target=n_pron.id,
                 relation="улыбнуться", clause_id="cl-2", provenance=_PROV),
        ],
    )

    _, resolutions = resolve_anaphora(
        graph, [c1, c2], [s1, s2],
        {"s-1": parsed_s1, "s-2": parsed_s2}, IdFactory(),
    )

    assert len(resolutions) == 1
    assert resolutions[0].antecedent_node_id == "n-ivan"


def test_animacy_filter_blocks_inanimate_for_anim_pron():
    """«Иван увидел шкаф. Он улыбнулся.» — Anim PRON цепляет Иван, не шкаф."""
    s1 = _sentence("s-1", 0, "Иван увидел шкаф.")
    s2 = _sentence("s-2", 1, "Он улыбнулся.")
    c1 = _clause("cl-1", "s-1", "Иван увидел шкаф.")
    c2 = _clause("cl-2", "s-2", "Он улыбнулся.")

    parsed_s1 = ParsedSentence(text="Иван увидел шкаф.", tokens=[
        Token(id_in_sent=1, text="Иван", lemma="иван", pos="PROPN",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim"},
              head=2, deprel="nsubj", start=0, end=4),
        Token(id_in_sent=2, text="увидел", lemma="увидеть", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=5, end=11),
        Token(id_in_sent=3, text="шкаф", lemma="шкаф", pos="NOUN",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Inan"},
              head=2, deprel="obj", start=12, end=16),
    ])
    parsed_s2 = ParsedSentence(text="Он улыбнулся.", tokens=[
        Token(id_in_sent=1, text="Он", lemma="он", pos="PRON",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim",
                     "PronType": "Prs", "Person": "3"},
              head=2, deprel="nsubj", start=0, end=2),
        Token(id_in_sent=2, text="улыбнулся", lemma="улыбнуться", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=3, end=12),
    ])

    n_ivan = _node("n-ivan", "иван", "PROPN", "cl-1", 1, "Иван")
    n_shkaf = _node("n-shkaf", "шкаф", "NOUN", "cl-1", 3, "шкаф")
    n_pron = _node("n-pron", "он", "PRON", "cl-2", 1, "Он")
    n_verb = _node("n-verb", "улыбнуться", "VERB", "cl-2", 2)

    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[n_ivan, n_shkaf, n_pron, n_verb],
        edges=[
            Edge(id="e-1", source=n_verb.id, target=n_pron.id,
                 relation="улыбнуться", clause_id="cl-2", provenance=_PROV),
        ],
    )

    _, resolutions = resolve_anaphora(
        graph, [c1, c2], [s1, s2],
        {"s-1": parsed_s1, "s-2": parsed_s2}, IdFactory(),
    )

    assert len(resolutions) == 1
    assert resolutions[0].antecedent_node_id == "n-ivan"


def test_plural_pronoun_matches_plural_noun():
    """«Дети играли. Они смеялись.» — они → дети."""
    s1 = _sentence("s-1", 0, "Дети играли.")
    s2 = _sentence("s-2", 1, "Они смеялись.")
    c1 = _clause("cl-1", "s-1", "Дети играли.")
    c2 = _clause("cl-2", "s-2", "Они смеялись.")

    parsed_s1 = ParsedSentence(text="Дети играли.", tokens=[
        Token(id_in_sent=1, text="Дети", lemma="ребёнок", pos="NOUN",
              feats={"Number": "Plur", "Animacy": "Anim"},
              head=2, deprel="nsubj", start=0, end=4),
        Token(id_in_sent=2, text="играли", lemma="играть", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=5, end=11),
    ])
    parsed_s2 = ParsedSentence(text="Они смеялись.", tokens=[
        Token(id_in_sent=1, text="Они", lemma="они", pos="PRON",
              feats={"Number": "Plur", "Animacy": "Anim",
                     "PronType": "Prs", "Person": "3"},
              head=2, deprel="nsubj", start=0, end=3),
        Token(id_in_sent=2, text="смеялись", lemma="смеяться", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=4, end=12),
    ])

    n_kids = _node("n-kids", "ребёнок", "NOUN", "cl-1", 1, "Дети")
    n_pron = _node("n-pron", "они", "PRON", "cl-2", 1, "Они")
    n_verb = _node("n-verb", "смеяться", "VERB", "cl-2", 2)

    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[n_kids, n_pron, n_verb],
        edges=[
            Edge(id="e-1", source=n_verb.id, target=n_pron.id,
                 relation="смеяться", clause_id="cl-2", provenance=_PROV),
        ],
    )

    _, resolutions = resolve_anaphora(
        graph, [c1, c2], [s1, s2],
        {"s-1": parsed_s1, "s-2": parsed_s2}, IdFactory(),
    )

    assert len(resolutions) == 1
    assert resolutions[0].antecedent_node_id == "n-kids"
    assert resolutions[0].matched_features.get("Number") == "Plur"


def test_propn_preferred_over_noun_when_pron_animacy_missing():
    """«Иван пришёл в дом. Он устал.» — даже без Animacy у PRON, PROPN-Иван
    должен победить NOUN-дом, потому что (а) PROPN > NOUN, (б) Иван — субъект.
    """
    s1 = _sentence("s-1", 0, "Иван пришёл в дом.")
    s2 = _sentence("s-2", 1, "Он устал.")
    c1 = _clause("cl-1", "s-1", "Иван пришёл в дом.")
    c2 = _clause("cl-2", "s-2", "Он устал.")

    # PRON без Animacy в feats — реалистичный сценарий natasha
    parsed_s1 = ParsedSentence(text="Иван пришёл в дом.", tokens=[
        Token(id_in_sent=1, text="Иван", lemma="иван", pos="PROPN",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim"},
              head=2, deprel="nsubj", start=0, end=4),
        Token(id_in_sent=2, text="пришёл", lemma="прийти", pos="VERB",
              feats={}, head=0, deprel="root", start=5, end=11),
        Token(id_in_sent=3, text="в", lemma="в", pos="ADP",
              feats={}, head=4, deprel="case", start=12, end=13),
        Token(id_in_sent=4, text="дом", lemma="дом", pos="NOUN",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Inan"},
              head=2, deprel="obl", start=14, end=17),
    ])
    parsed_s2 = ParsedSentence(text="Он устал.", tokens=[
        Token(id_in_sent=1, text="Он", lemma="он", pos="PRON",
              feats={"Gender": "Masc", "Number": "Sing",
                     "PronType": "Prs", "Person": "3"},  # Animacy отсутствует
              head=2, deprel="nsubj", start=0, end=2),
        Token(id_in_sent=2, text="устал", lemma="устать", pos="VERB",
              feats={}, head=0, deprel="root", start=3, end=8),
    ])

    n_ivan = _node("n-ivan", "иван", "PROPN", "cl-1", 1, "Иван")
    n_dom = _node("n-dom", "дом", "NOUN", "cl-1", 4, "дом")
    n_pron = _node("n-pron", "он", "PRON", "cl-2", 1, "Он")
    n_verb = _node("n-verb", "устать", "VERB", "cl-2", 2)

    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[n_ivan, n_dom, n_pron, n_verb],
        edges=[
            Edge(id="e-1", source=n_pron.id, target=n_dom.id,
                 relation="устать", clause_id="cl-2", provenance=_PROV),
        ],
    )

    _, resolutions = resolve_anaphora(
        graph, [c1, c2], [s1, s2],
        {"s-1": parsed_s1, "s-2": parsed_s2}, IdFactory(),
    )
    assert len(resolutions) == 1
    assert resolutions[0].antecedent_node_id == "n-ivan"


def test_unresolved_pronoun_stays():
    """Местоимение в начале документа без антецедента остаётся в графе."""
    s1 = _sentence("s-1", 0, "Он пришёл.")
    c1 = _clause("cl-1", "s-1", "Он пришёл.")

    parsed = ParsedSentence(text="Он пришёл.", tokens=[
        Token(id_in_sent=1, text="Он", lemma="он", pos="PRON",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim",
                     "PronType": "Prs", "Person": "3"},
              head=2, deprel="nsubj", start=0, end=2),
        Token(id_in_sent=2, text="пришёл", lemma="прийти", pos="VERB",
              feats={"VerbForm": "Fin"}, head=0, deprel="root", start=3, end=9),
    ])

    n_pron = _node("n-pron", "он", "PRON", "cl-1", 1, "Он")
    n_verb = _node("n-verb", "прийти", "VERB", "cl-1", 2)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[n_pron, n_verb],
        edges=[
            Edge(id="e-1", source=n_verb.id, target=n_pron.id,
                 relation="прийти", clause_id="cl-1", provenance=_PROV),
        ],
    )

    new_graph, resolutions = resolve_anaphora(
        graph, [c1], [s1], {"s-1": parsed}, IdFactory(),
    )
    assert resolutions == []
    assert "n-pron" in {n.id for n in new_graph.nodes}


def test_window_blocks_distant_antecedent():
    """search_window_sentences=1 не даёт зацепить антецедент через 2 предложения."""
    s1 = _sentence("s-1", 0, "Иван пришёл.")
    s2 = _sentence("s-2", 1, "Светило солнце.")
    s3 = _sentence("s-3", 2, "Он устал.")
    c1 = _clause("cl-1", "s-1", "Иван пришёл.")
    c2 = _clause("cl-2", "s-2", "Светило солнце.")
    c3 = _clause("cl-3", "s-3", "Он устал.")

    parsed_s1 = ParsedSentence(text="Иван пришёл.", tokens=[
        Token(id_in_sent=1, text="Иван", lemma="иван", pos="PROPN",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim"},
              head=2, deprel="nsubj", start=0, end=4),
        Token(id_in_sent=2, text="пришёл", lemma="прийти", pos="VERB",
              feats={}, head=0, deprel="root", start=5, end=11),
    ])
    parsed_s2 = ParsedSentence(text="Светило солнце.", tokens=[
        Token(id_in_sent=1, text="Светило", lemma="светить", pos="VERB",
              feats={}, head=0, deprel="root", start=0, end=7),
        Token(id_in_sent=2, text="солнце", lemma="солнце", pos="NOUN",
              feats={"Gender": "Neut", "Number": "Sing", "Animacy": "Inan"},
              head=1, deprel="nsubj", start=8, end=14),
    ])
    parsed_s3 = ParsedSentence(text="Он устал.", tokens=[
        Token(id_in_sent=1, text="Он", lemma="он", pos="PRON",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim",
                     "PronType": "Prs", "Person": "3"},
              head=2, deprel="nsubj", start=0, end=2),
        Token(id_in_sent=2, text="устал", lemma="устать", pos="VERB",
              feats={}, head=0, deprel="root", start=3, end=8),
    ])

    n_ivan = _node("n-ivan", "иван", "PROPN", "cl-1", 1)
    n_sun = _node("n-sun", "солнце", "NOUN", "cl-2", 2)
    n_pron = _node("n-pron", "он", "PRON", "cl-3", 1, "Он")
    n_verb = _node("n-verb", "устать", "VERB", "cl-3", 2)

    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[n_ivan, n_sun, n_pron, n_verb],
        edges=[
            Edge(id="e-1", source=n_verb.id, target=n_pron.id,
                 relation="устать", clause_id="cl-3", provenance=_PROV),
        ],
    )

    _, resolutions = resolve_anaphora(
        graph, [c1, c2, c3], [s1, s2, s3],
        {"s-1": parsed_s1, "s-2": parsed_s2, "s-3": parsed_s3}, IdFactory(),
        search_window_sentences=1,
    )
    # Иван слишком далеко (2 предложения), солнце не подходит по Animacy
    assert resolutions == []


def test_original_graph_not_mutated():
    """resolve_anaphora возвращает новый граф, исходный не меняется."""
    s1 = _sentence("s-1", 0, "Иван пришёл.")
    s2 = _sentence("s-2", 1, "Он устал.")
    c1 = _clause("cl-1", "s-1", "Иван пришёл.")
    c2 = _clause("cl-2", "s-2", "Он устал.")

    parsed_s1 = ParsedSentence(text="Иван пришёл.", tokens=[
        Token(id_in_sent=1, text="Иван", lemma="иван", pos="PROPN",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim"},
              head=2, deprel="nsubj", start=0, end=4),
        Token(id_in_sent=2, text="пришёл", lemma="прийти", pos="VERB",
              feats={}, head=0, deprel="root", start=5, end=11),
    ])
    parsed_s2 = ParsedSentence(text="Он устал.", tokens=[
        Token(id_in_sent=1, text="Он", lemma="он", pos="PRON",
              feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim",
                     "PronType": "Prs", "Person": "3"},
              head=2, deprel="nsubj", start=0, end=2),
        Token(id_in_sent=2, text="устал", lemma="устать", pos="VERB",
              feats={}, head=0, deprel="root", start=3, end=8),
    ])

    n_ivan = _node("n-ivan", "иван", "PROPN", "cl-1", 1)
    n_pron = _node("n-pron", "он", "PRON", "cl-2", 1, "Он")
    n_verb = _node("n-verb", "устать", "VERB", "cl-2", 2)
    graph = SemanticGraph(
        document_id="doc-1",
        nodes=[n_ivan, n_pron, n_verb],
        edges=[
            Edge(id="e-1", source=n_verb.id, target=n_pron.id,
                 relation="устать", clause_id="cl-2", provenance=_PROV),
        ],
    )

    original_node_count = len(graph.nodes)
    resolve_anaphora(
        graph, [c1, c2], [s1, s2],
        {"s-1": parsed_s1, "s-2": parsed_s2}, IdFactory(),
    )
    assert len(graph.nodes) == original_node_count
