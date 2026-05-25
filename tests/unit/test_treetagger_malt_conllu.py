"""Contract-тест: TreeTagger → MaltParser передаёт непустые морфо-поля.

Главный смысл рефакторинга — больше не подавать MaltParser-у пустые
поля (`_`) в колонках lemma / POS / feats, потому что transition-based
parser обучен на размеченных входах. Этот тест с замокированным
subprocess проверяет, что в CoNLL-вход MaltParser попадают конкретные
значения lemma / upos / feats, полученные от морфо-провайдера.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from metagraph_nlp.parsers.morphsyntax.maltparser_adapter import (
    TreeTaggerMaltParser,
)


@pytest.fixture
def fake_paths(tmp_path: Path) -> tuple[Path, Path]:
    jar = tmp_path / "fake-maltparser.jar"
    jar.write_text("")
    mco = tmp_path / "fake-model.mco"
    mco.write_text("")
    return jar, mco


def _make_morph_mock(tagged: list[tuple[str, str, str, dict[str, str]]]) -> MagicMock:
    morph = MagicMock()
    morph.tag.return_value = tagged
    return morph


def test_conll_input_contains_morpho_fields(fake_paths, monkeypatch):
    """Перехватываем subprocess.run и проверяем содержимое CoNLL-input."""
    jar, mco = fake_paths
    tagged = [
        ("Студент", "студент", "NOUN", {"Gender": "Masc", "Case": "Nom"}),
        ("читает", "читать", "VERB", {"Tense": "Pres", "Person": "3"}),
        ("книгу", "книга", "NOUN", {"Gender": "Fem", "Case": "Acc"}),
        (".", ".", "PUNCT", {}),
    ]
    morph = _make_morph_mock(tagged)
    parser = TreeTaggerMaltParser(
        malt_jar=jar, model_path=mco, morph=morph, java_bin="java"
    )

    captured_conll: dict[str, str] = {}

    def fake_run(cmd, **kwargs):
        # MaltParser читает вход из -i <path>, пишет в -o <path>.
        # Имитируем: читаем input, формируем минимальный output.
        i_idx = cmd.index("-i") + 1
        o_idx = cmd.index("-o") + 1
        in_path = Path(cmd[i_idx])
        out_path = Path(cmd[o_idx])
        captured_conll["input"] = in_path.read_text(encoding="utf-8")
        # Сгенерируем валидный CoNLL-output с head/deprel.
        n = len([line for line in captured_conll["input"].splitlines() if line.strip()])
        out_lines: list[str] = []
        for i, line in enumerate(captured_conll["input"].splitlines(), start=1):
            if not line.strip():
                continue
            fields = line.split("\t")
            fields[6] = "0" if i == 2 else "2"
            fields[7] = "root" if i == 2 else "nsubj"
            out_lines.append("\t".join(fields))
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        return MagicMock(returncode=0)

    monkeypatch.setattr(
        "metagraph_nlp.parsers.morphsyntax.maltparser_adapter.subprocess.run",
        fake_run,
    )

    parsed = parser.parse("Студент читает книгу.")

    conll = captured_conll["input"]
    assert conll, "subprocess.run должен был получить CoNLL-input"

    lines = [line for line in conll.splitlines() if line.strip()]
    assert len(lines) == 4

    first = lines[0].split("\t")
    assert first[2] == "студент", "lemma в колонке 3"
    assert first[3] == "NOUN", "upos в колонке 4"
    assert "Case=Nom" in first[5] and "Gender=Masc" in first[5], (
        "feats в колонке 6 — Case и Gender присутствуют"
    )
    assert first[5] != "_", "feats не пустые (главный contract задачи)"

    second = lines[1].split("\t")
    assert second[2] == "читать"
    assert second[3] == "VERB"
    assert "Tense=Pres" in second[5]

    assert len(parsed.tokens) == 4
    assert parsed.tokens[0].text == "Студент"
    assert parsed.tokens[0].lemma == "студент"
    assert parsed.tokens[0].pos == "NOUN"
    assert parsed.tokens[0].feats.get("Case") == "Nom"


def test_spans_resolved_positionally_on_repeated_wordforms(fake_paths, monkeypatch):
    """Повтор словоформ («он… он») не должен ломать восстановление spans."""
    jar, mco = fake_paths
    tagged = [
        ("Он", "он", "PRON", {}),
        ("сказал", "сказать", "VERB", {}),
        (",", ",", "PUNCT", {}),
        ("что", "что", "SCONJ", {}),
        ("он", "он", "PRON", {}),
        ("пришёл", "прийти", "VERB", {}),
        (".", ".", "PUNCT", {}),
    ]
    morph = _make_morph_mock(tagged)
    parser = TreeTaggerMaltParser(
        malt_jar=jar, model_path=mco, morph=morph, java_bin="java"
    )

    def fake_run(cmd, **kwargs):
        i_idx = cmd.index("-i") + 1
        o_idx = cmd.index("-o") + 1
        in_path = Path(cmd[i_idx])
        out_path = Path(cmd[o_idx])
        out_lines: list[str] = []
        for i, line in enumerate(in_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split("\t")
            fields[6] = "0" if i == 2 else "2"
            fields[7] = "root"
            out_lines.append("\t".join(fields))
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        return MagicMock(returncode=0)

    monkeypatch.setattr(
        "metagraph_nlp.parsers.morphsyntax.maltparser_adapter.subprocess.run",
        fake_run,
    )

    sent = "Он сказал, что он пришёл."
    parsed = parser.parse(sent)

    assert len(parsed.tokens) == 7
    # razdel дробит "сказал," в "сказал" + "," (нужно проверить).
    second_he = parsed.tokens[4]
    assert second_he.text == "он"
    assert sent[second_he.start:second_he.end] == "он", (
        "позиционный mapping через razdel-spans корректен на повторе"
    )
    assert second_he.start > parsed.tokens[0].end


def test_deprel_mapping_applied(fake_paths, monkeypatch):
    """deprel_mapping переводит legacy-теги в UD."""
    jar, mco = fake_paths
    tagged = [
        ("Студент", "студент", "NOUN", {}),
        ("читает", "читать", "VERB", {}),
    ]
    morph = _make_morph_mock(tagged)
    parser = TreeTaggerMaltParser(
        malt_jar=jar,
        model_path=mco,
        morph=morph,
        deprel_mapping={"SBJ": "nsubj", "PRED": "root"},
    )

    def fake_run(cmd, **kwargs):
        i_idx = cmd.index("-i") + 1
        o_idx = cmd.index("-o") + 1
        in_path = Path(cmd[i_idx])
        out_path = Path(cmd[o_idx])
        out_lines: list[str] = []
        for i, line in enumerate(in_path.read_text(encoding="utf-8").splitlines(), 1):
            if not line.strip():
                continue
            fields = line.split("\t")
            fields[6] = "0" if i == 2 else "2"
            fields[7] = "PRED" if i == 2 else "SBJ"
            out_lines.append("\t".join(fields))
        out_path.write_text("\n".join(out_lines), encoding="utf-8")
        return MagicMock(returncode=0)

    monkeypatch.setattr(
        "metagraph_nlp.parsers.morphsyntax.maltparser_adapter.subprocess.run",
        fake_run,
    )

    parsed = parser.parse("Студент читает")
    deprels = [t.deprel for t in parsed.tokens]
    assert "nsubj" in deprels
    assert "root" in deprels
    assert "SBJ" not in deprels
    assert "PRED" not in deprels
