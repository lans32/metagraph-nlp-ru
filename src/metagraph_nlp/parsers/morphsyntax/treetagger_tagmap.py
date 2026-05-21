"""MSD-Russian → UD UPOS+feats маппер (Sharoff & Nivre, 2011).

TreeTagger с `russian.par` выдаёт теги в схеме MSD-Russian: позиционные коды
вида `Ncmsny` (noun-common-masc-sing-nom-anim) или `Vmip3s-a-e` (verb-main-
indicative-present-3-sing-active-imperfective). Этот модуль переводит их в
UD UPOS + словарь feats, ожидаемый верхними стадиями pipeline.

Словари значений и категорий загружаются из YAML
(`configs/treetagger_tagsets/msd_ru.yaml`) — если схема `.par` сменится на
другую, пользователь подменяет YAML без правки кода. Разбор позиций
(noun/verb/adj/...) живёт в этом файле как набор маленьких чистых функций,
что упрощает unit-тестирование.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


class MsdTagMapper:
    """Маппер MSD-Russian (Sharoff/TreeTagger) → UD UPOS + feats."""

    def __init__(self, yaml_path: Path) -> None:
        with open(yaml_path, encoding="utf-8") as f:
            data: dict[str, Any] = yaml.safe_load(f)
        self._categories: dict[str, dict[str, Any]] = data.get("categories", {})
        self._punct: set[str] = set(data.get("punct_tags", []))
        self._sconj_lemmas: set[str] = set(data.get("sconj_lemmas", []))
        self._values: dict[str, Any] = data.get("values", {})

    def map(self, msd_tag: str, lemma: str | None = None) -> tuple[str, dict[str, str]]:
        """msd_tag → (UD upos, UD feats). Неизвестный код → ('X', {})."""
        if not msd_tag:
            return ("X", {})
        if msd_tag in self._punct:
            return ("PUNCT", {})

        head = msd_tag[0]
        cat = self._categories.get(head)
        if cat is None:
            return ("X", {})

        scheme: str = cat.get("scheme", "simple")
        upos: str = cat.get("upos_default", "X")
        feats: dict[str, str] = {}

        if scheme == "noun":
            upos, feats = self._parse_noun(msd_tag, cat)
        elif scheme == "verb":
            feats = self._parse_verb(msd_tag)
        elif scheme == "adj":
            feats = self._parse_adj(msd_tag)
        elif scheme == "pronoun":
            feats = self._parse_pronoun(msd_tag)
        elif scheme == "numeral":
            upos, feats = self._parse_numeral(msd_tag, cat, upos)
        elif scheme == "adp":
            feats = self._parse_adp(msd_tag)
        elif scheme == "simple":
            pass

        if scheme == "simple" and upos == "CCONJ" and lemma is not None:
            if lemma.lower() in self._sconj_lemmas:
                upos = "SCONJ"

        return (upos, feats)

    def _val(self, group: str, key: str) -> str | None:
        return self._values.get(group, {}).get(key)

    def _parse_noun(
        self, tag: str, cat: dict[str, Any]
    ) -> tuple[str, dict[str, str]]:
        # N<subtype><gender><number><case><animacy>: Ncmsny
        upos = cat.get("upos_default", "NOUN")
        subtype_map: dict[str, str] = cat.get("subtype_upos", {})
        if len(tag) >= 2:
            upos = subtype_map.get(tag[1], upos)
        feats: dict[str, str] = {}
        if len(tag) >= 3:
            v = self._val("gender", tag[2])
            if v:
                feats["Gender"] = v
        if len(tag) >= 4:
            v = self._val("number", tag[3])
            if v:
                feats["Number"] = v
        if len(tag) >= 5:
            v = self._val("case", tag[4])
            if v:
                feats["Case"] = v
        if len(tag) >= 6:
            v = self._val("animacy", tag[5])
            if v:
                feats["Animacy"] = v
        return (upos, feats)

    def _parse_verb(self, tag: str) -> dict[str, str]:
        # V<type><form><tense><person><number><gender><voice>-<aspect>
        # Длина обычно 10: Vmip3s-a-e, Vmn----a-e, Vmps-smpsp, Vmgp---a-e
        feats: dict[str, str] = {}
        if len(tag) >= 3:
            vf = self._values.get("verb_form", {}).get(tag[2])
            if isinstance(vf, dict):
                feats.update(vf)
        if len(tag) >= 4 and tag[3] != "-":
            v = self._val("tense", tag[3])
            if v:
                feats["Tense"] = v
        if len(tag) >= 5 and tag[4] != "-":
            v = self._val("person", tag[4])
            if v:
                feats["Person"] = v
        if len(tag) >= 6 and tag[5] != "-":
            v = self._val("number", tag[5])
            if v:
                feats["Number"] = v
        if len(tag) >= 7 and tag[6] != "-":
            v = self._val("gender", tag[6])
            if v:
                feats["Gender"] = v
        if len(tag) >= 8 and tag[7] != "-":
            v = self._val("voice", tag[7])
            if v:
                feats["Voice"] = v
        if len(tag) >= 9 and tag[8] != "-":
            v = self._values.get("variant", {}).get(tag[8])
            if v:
                feats["Variant"] = v
        if len(tag) >= 10 and tag[9] != "-":
            v = self._val("aspect", tag[9])
            if v:
                feats["Aspect"] = v
        return feats

    def _parse_adj(self, tag: str) -> dict[str, str]:
        # A<type><degree><gender><number><case><form>: Afpfsaf
        feats: dict[str, str] = {}
        if len(tag) >= 3 and tag[2] != "-":
            v = self._val("degree", tag[2])
            if v:
                feats["Degree"] = v
        if len(tag) >= 4 and tag[3] != "-":
            v = self._val("gender", tag[3])
            if v:
                feats["Gender"] = v
        if len(tag) >= 5 and tag[4] != "-":
            v = self._val("number", tag[4])
            if v:
                feats["Number"] = v
        if len(tag) >= 6 and tag[5] != "-":
            v = self._val("case", tag[5])
            if v:
                feats["Case"] = v
        if len(tag) >= 7 and tag[6] != "-":
            v = self._values.get("variant", {}).get(tag[6])
            if v:
                feats["Variant"] = v
        return feats

    def _parse_pronoun(self, tag: str) -> dict[str, str]:
        # P<type><person><gender><number><case><animacy>: P-3msnn, P--msaa, P----an
        feats: dict[str, str] = {}
        if len(tag) >= 3 and tag[2] != "-":
            v = self._val("person", tag[2])
            if v:
                feats["Person"] = v
        if len(tag) >= 4 and tag[3] != "-":
            v = self._val("gender", tag[3])
            if v:
                feats["Gender"] = v
        if len(tag) >= 5 and tag[4] != "-":
            v = self._val("number", tag[4])
            if v:
                feats["Number"] = v
        if len(tag) >= 6 and tag[5] != "-":
            v = self._val("case", tag[5])
            if v:
                feats["Case"] = v
        if len(tag) >= 7 and tag[6] != "-":
            v = self._val("animacy", tag[6])
            if v:
                feats["Animacy"] = v
        return feats

    def _parse_numeral(
        self, tag: str, cat: dict[str, Any], upos: str
    ) -> tuple[str, dict[str, str]]:
        # M<type><gender><number><case>: Mc--n, Momsn, Momsa
        feats: dict[str, str] = {}
        if len(tag) >= 2:
            nt = self._values.get("numeral_type", {}).get(tag[1])
            if isinstance(nt, dict):
                feats.update(nt)
        if len(tag) >= 3 and tag[2] != "-":
            v = self._val("gender", tag[2])
            if v:
                feats["Gender"] = v
        if len(tag) >= 4 and tag[3] != "-":
            v = self._val("number", tag[3])
            if v:
                feats["Number"] = v
        if len(tag) >= 5 and tag[4] != "-":
            v = self._val("case", tag[4])
            if v:
                feats["Case"] = v
        return (upos, feats)

    def _parse_adp(self, tag: str) -> dict[str, str]:
        # Sp-<case>: Sp-l, Sp-d, Sp-g, Sp-a, Sp-i
        feats: dict[str, str] = {}
        if len(tag) >= 4 and tag[3] != "-":
            v = self._val("case", tag[3])
            if v:
                feats["Case"] = v
        return feats


def default_tagset_path(name: str) -> Path:
    """Резолвит имя тегсета в путь.

    Если `name` — абсолютный путь или относительный с расширением `.yaml`,
    возвращает его как есть. Иначе ищет в `configs/treetagger_tagsets/`
    относительно корня проекта.
    """
    p = Path(name)
    if p.is_absolute() or p.suffix == ".yaml":
        return p
    project_root = Path(__file__).resolve().parents[4]
    return project_root / "configs" / "treetagger_tagsets" / f"{name}.yaml"
