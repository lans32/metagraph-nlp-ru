"""Тесты MsdTagMapper (MSD-Russian → UD)."""

from __future__ import annotations

from pathlib import Path

import pytest

from metagraph_nlp.parsers.morphsyntax.treetagger_tagmap import (
    MsdTagMapper,
    default_tagset_path,
)


@pytest.fixture(scope="module")
def mapper() -> MsdTagMapper:
    """Реальный msd_ru.yaml из репозитория."""
    return MsdTagMapper(default_tagset_path("msd_ru"))


def test_noun_common_nominative_animate(mapper: MsdTagMapper):
    upos, feats = mapper.map("Ncmsny")
    assert upos == "NOUN"
    assert feats == {
        "Gender": "Masc",
        "Number": "Sing",
        "Case": "Nom",
        "Animacy": "Anim",
    }


def test_noun_common_accusative_inanimate(mapper: MsdTagMapper):
    upos, feats = mapper.map("Ncfsan")
    assert upos == "NOUN"
    assert feats["Gender"] == "Fem"
    assert feats["Case"] == "Acc"
    assert feats["Animacy"] == "Inan"


def test_verb_indicative_present(mapper: MsdTagMapper):
    upos, feats = mapper.map("Vmip3s-a-e")
    assert upos == "VERB"
    assert feats["VerbForm"] == "Fin"
    assert feats["Mood"] == "Ind"
    assert feats["Tense"] == "Pres"
    assert feats["Person"] == "3"
    assert feats["Number"] == "Sing"
    assert feats["Voice"] == "Act"
    assert feats["Aspect"] == "Imp"


def test_verb_past_masculine(mapper: MsdTagMapper):
    upos, feats = mapper.map("Vmis-sma-p")
    assert upos == "VERB"
    assert feats["Mood"] == "Ind"
    assert feats["Tense"] == "Past"
    assert feats["Number"] == "Sing"
    assert feats["Gender"] == "Masc"
    assert feats["Aspect"] == "Perf"


def test_verb_infinitive(mapper: MsdTagMapper):
    upos, feats = mapper.map("Vmn----a-e")
    assert upos == "VERB"
    assert feats["VerbForm"] == "Inf"
    assert feats["Aspect"] == "Imp"


def test_verb_gerund(mapper: MsdTagMapper):
    upos, feats = mapper.map("Vmgp---a-e")
    assert upos == "VERB"
    assert feats["VerbForm"] == "Conv"
    assert feats["Aspect"] == "Imp"


def test_verb_passive_participle(mapper: MsdTagMapper):
    upos, feats = mapper.map("Vmps-smpsp")
    assert upos == "VERB"
    assert feats["VerbForm"] == "Part"
    assert feats["Voice"] == "Pass"
    assert feats["Aspect"] == "Perf"


def test_adjective_full(mapper: MsdTagMapper):
    upos, feats = mapper.map("Afpfsaf")
    assert upos == "ADJ"
    assert feats["Degree"] == "Pos"
    assert feats["Gender"] == "Fem"
    assert feats["Number"] == "Sing"
    assert feats["Case"] == "Acc"


def test_adjective_short(mapper: MsdTagMapper):
    upos, feats = mapper.map("Afpmsns")
    assert upos == "ADJ"
    assert feats["Variant"] == "Short"


def test_pronoun_3rd_person(mapper: MsdTagMapper):
    upos, feats = mapper.map("P-3msnn")
    assert upos == "PRON"
    assert feats["Person"] == "3"
    assert feats["Gender"] == "Masc"
    assert feats["Number"] == "Sing"
    assert feats["Case"] == "Nom"


def test_pronoun_reflexive(mapper: MsdTagMapper):
    upos, feats = mapper.map("P----an")
    assert upos == "PRON"
    assert feats.get("Case") == "Acc"


def test_adposition_with_locative(mapper: MsdTagMapper):
    upos, feats = mapper.map("Sp-l")
    assert upos == "ADP"
    assert feats["Case"] == "Loc"


def test_simple_adverb(mapper: MsdTagMapper):
    upos, feats = mapper.map("R")
    assert upos == "ADV"
    assert feats == {}


def test_conjunction_default_cconj(mapper: MsdTagMapper):
    upos, _ = mapper.map("C", lemma="и")
    assert upos == "CCONJ"


def test_conjunction_subordinating_via_lemma(mapper: MsdTagMapper):
    upos, _ = mapper.map("C", lemma="что")
    assert upos == "SCONJ"
    upos, _ = mapper.map("C", lemma="чтобы")
    assert upos == "SCONJ"


def test_particle(mapper: MsdTagMapper):
    upos, feats = mapper.map("Q")
    assert upos == "PART"


def test_numeral_cardinal(mapper: MsdTagMapper):
    upos, feats = mapper.map("Mc--n")
    assert upos == "NUM"
    assert feats.get("NumType") == "Card"


def test_numeral_ordinal(mapper: MsdTagMapper):
    upos, feats = mapper.map("Momsn")
    assert upos == "NUM"
    assert feats.get("NumType") == "Ord"
    assert feats["Gender"] == "Masc"


def test_punctuation_sent(mapper: MsdTagMapper):
    upos, feats = mapper.map("SENT")
    assert upos == "PUNCT"
    assert feats == {}


def test_punctuation_comma(mapper: MsdTagMapper):
    upos, feats = mapper.map(",")
    assert upos == "PUNCT"


def test_unknown_tag(mapper: MsdTagMapper):
    upos, feats = mapper.map("ZZZ")
    assert upos == "X"
    assert feats == {}


def test_empty_tag(mapper: MsdTagMapper):
    upos, feats = mapper.map("")
    assert upos == "X"


def test_default_tagset_path_resolves_known_name():
    path = default_tagset_path("msd_ru")
    assert path.exists()
    assert path.name == "msd_ru.yaml"


def test_default_tagset_path_passes_through_absolute(tmp_path: Path):
    abs_path = tmp_path / "custom.yaml"
    abs_path.write_text("categories: {}")
    assert default_tagset_path(str(abs_path)) == abs_path
