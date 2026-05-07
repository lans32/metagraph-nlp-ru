"""Интеграционный тест: разрешение анафоры внутри pipeline.run().

Использует кастомный парсер (без natasha): в нём для двухпредложенного
текста выставлены реальные UD-feats (Gender/Number/Animacy/PronType/Person),
чтобы прогнать стадию `anaphora_resolution` end-to-end.
"""

from __future__ import annotations

from metagraph_nlp.config import AnaphoraConfig, Config
from metagraph_nlp.parsers.morphsyntax.types import ParsedSentence, Token
from metagraph_nlp.pipeline import run


class _AnaphoraParser:
    """Парсер для пары предложений «Иван пришёл домой. Он устал.»"""

    def parse(self, sentence_text: str) -> ParsedSentence:
        s = sentence_text.strip()
        if s.startswith("Иван"):
            return ParsedSentence(text=sentence_text, tokens=[
                Token(id_in_sent=1, text="Иван", lemma="иван", pos="PROPN",
                      feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim"},
                      head=2, deprel="nsubj", start=0, end=4),
                Token(id_in_sent=2, text="пришёл", lemma="прийти", pos="VERB",
                      feats={"VerbForm": "Fin"}, head=0, deprel="root",
                      start=5, end=11),
                Token(id_in_sent=3, text="домой", lemma="дом", pos="NOUN",
                      feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Inan"},
                      head=2, deprel="obl", start=12, end=17),
                Token(id_in_sent=4, text=".", lemma=".", pos="PUNCT",
                      feats={}, head=2, deprel="punct", start=17, end=18),
            ])
        # «Он устал.»
        return ParsedSentence(text=sentence_text, tokens=[
            Token(id_in_sent=1, text="Он", lemma="он", pos="PRON",
                  feats={"Gender": "Masc", "Number": "Sing", "Animacy": "Anim",
                         "PronType": "Prs", "Person": "3"},
                  head=2, deprel="nsubj", start=0, end=2),
            Token(id_in_sent=2, text="устал", lemma="устать", pos="VERB",
                  feats={"VerbForm": "Fin"}, head=0, deprel="root",
                  start=3, end=8),
            Token(id_in_sent=3, text=".", lemma=".", pos="PUNCT",
                  feats={}, head=2, deprel="punct", start=8, end=9),
        ])


def test_pipeline_resolves_anaphora_end_to_end():
    cfg = Config(anaphora=AnaphoraConfig(enabled=True))
    text = "Иван пришёл домой. Он устал."

    result = run(text, config=cfg, parser=_AnaphoraParser())

    # v1: PRON-узел остаётся в графе, но с обновлёнными атрибутами:
    # upos сменился на PROPN (как у антецедента), original_upos=PRON.
    replaced_nodes = [
        n for n in result.graph.nodes if n.antecedent_node_id is not None
    ]
    assert len(replaced_nodes) == 1
    replaced = replaced_nodes[0]
    assert replaced.lemma == "иван"
    assert replaced.original_lemma == "он"
    assert replaced.original_upos == "PRON"
    assert replaced.upos == "PROPN"
    assert replaced.surface == "Он"

    # Зафиксировано ровно одно разрешение, антецедент — Иван.
    assert result.anaphora_resolutions is not None
    assert len(result.anaphora_resolutions) == 1
    r = result.anaphora_resolutions[0]
    assert r.antecedent_lemma == "иван"
    assert r.pronoun_type == "personal_3p"
    assert r.resolution_strategy == "search"
    assert r.matched_features.get("Gender") == "Masc"

    # В audit-логе появилась запись стадии с новой версией правила.
    rules = {e.rule for e in result.audit.events}
    assert "anaphora_resolution_v1" in rules


def test_pipeline_off_by_default_keeps_pron_nodes():
    cfg = Config()  # anaphora.enabled=False по умолчанию
    text = "Иван пришёл домой. Он устал."
    result = run(text, config=cfg, parser=_AnaphoraParser())

    pron_nodes = [n for n in result.graph.nodes if n.upos == "PRON"]
    assert len(pron_nodes) == 1
    assert result.anaphora_resolutions is None
