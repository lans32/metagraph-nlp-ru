"""Сборка иерархического словаря predicate-классов из RuWordNet.

Скрипт читает якоря из ``configs/predicate_anchors.yaml``, обходит
hyponym-дерево RuWordNet от каждого якоря до ``--max-depth`` и записывает
расширенный словарь YAML version=1 с двумя секциями:

- ``hierarchy``: дерево классов (parent, level, anchor_synset_id, label_ru)
- ``lemmas``: lemma → список путей ``[leaf_slug, ..., root_slug]``

Артефакт коммитится в репозиторий и используется pipeline'ом без runtime
зависимости от ``ruwordnet`` (CLAUDE.md §13, §10 audit trail).

Запуск::

    pip install ruwordnet && ruwordnet download
    python scripts/build_predicate_lexicon.py \\
        --anchors configs/predicate_anchors.yaml \\
        --output configs/predicate_classes_ruwordnet.yaml \\
        --max-depth 3

Принципы:
- Детерминизм: фиксированный порядок ключей (sort), стабильный slug.
- Воспроизводимость: метаданные (timestamp, версия пакета, sha256 anchors).
- Объяснимость: каждый класс хранит anchor_synset_id для трассировки до RuWordNet.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import logging
import re
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import yaml

try:
    from importlib.metadata import version as pkg_version
except ImportError:  # pragma: no cover — Python <3.8 unsupported here
    pkg_version = None  # type: ignore

# Принудительная UTF-8 stdout на Windows (русские леммы в логах).
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", line_buffering=True)

logger = logging.getLogger("build_predicate_lexicon")


_RU_TO_LAT = {
    "а": "a", "б": "b", "в": "v", "г": "g", "д": "d", "е": "e", "ё": "e",
    "ж": "zh", "з": "z", "и": "i", "й": "i", "к": "k", "л": "l", "м": "m",
    "н": "n", "о": "o", "п": "p", "р": "r", "с": "s", "т": "t", "у": "u",
    "ф": "f", "х": "h", "ц": "c", "ч": "ch", "ш": "sh", "щ": "sch",
    "ъ": "", "ы": "y", "ь": "", "э": "e", "ю": "yu", "я": "ya",
}


def transliterate(text: str) -> str:
    """Простая транслитерация кириллицы в латиницу для slug-имён."""
    out = []
    for ch in text.lower():
        if ch in _RU_TO_LAT:
            out.append(_RU_TO_LAT[ch])
        elif ch.isalnum() or ch == "_":
            out.append(ch)
        else:
            out.append(" ")
    return "".join(out)


def slugify_title(title: str, max_words: int = 2, max_len: int = 30) -> str:
    """Превратить ``synset.title`` в короткий ASCII-slug.

    'ИДТИ (РАБОТА УСТРОЙСТВА)' → 'idti_rabota'
    'УДАРИТЬ, НАНЕСТИ УДАР'    → 'udarit_nanesti'
    """
    # Убрать скобочные пояснения: 'ИДТИ (РАБОТА УСТРОЙСТВА)' → 'ИДТИ'
    cleaned = re.sub(r"\([^)]*\)", "", title)
    translit = transliterate(cleaned)
    # Раздробить на токены, фильтровать пустые
    tokens = [t for t in re.split(r"[\s,;:.\-/]+", translit) if t and len(t) >= 2]
    if not tokens:
        # fallback: уникальное число
        tokens = ["x"]
    slug = "_".join(tokens[:max_words])
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


def make_unique_slug(parent_slug: str, title: str, used: set[str]) -> str:
    """Создать уникальный slug ``<parent>_<title-slug>`` с числовым суффиксом при коллизии."""
    base = f"{parent_slug}_{slugify_title(title)}"
    if base not in used:
        return base
    n = 2
    while f"{base}_{n}" in used:
        n += 1
    return f"{base}_{n}"


def is_single_word_lemma(lemma: str) -> bool:
    """Только однословные леммы — natasha лемматизирует каждый токен отдельно.

    Многословные ('ПОДЧИНИТЬ СВОЕЙ ВЛАСТИ') не попадают в граф через
    `predicate.lemma` и поэтому бесполезны для агрегации.
    """
    return " " not in lemma and "-" not in lemma and len(lemma) >= 2


def collect_hierarchy(
    wn,
    anchor_slug: str,
    anchor_synset,
    anchor_label_ru: str,
    max_depth: int,
) -> tuple[dict[str, dict], dict[str, list[list[str]]]]:
    """BFS hyponyms от anchor до max_depth.

    Возвращает (hierarchy, lemmas) где:
    - hierarchy[slug] = {parent, level, anchor_synset_id, label_ru, ru_title}
    - lemmas[lemma] = list of paths [leaf_slug, ..., root_slug]
    """
    hierarchy: dict[str, dict] = {}
    lemmas: dict[str, list[list[str]]] = defaultdict(list)
    used_slugs: set[str] = {anchor_slug}
    visited_synsets: set[str] = set()

    # Queue: (synset, slug, depth, path_to_root)
    queue: list[tuple[object, str, int, list[str]]] = [
        (anchor_synset, anchor_slug, 0, [])
    ]

    while queue:
        synset, slug, depth, parent_path = queue.pop(0)
        if synset.id in visited_synsets:
            continue
        visited_synsets.add(synset.id)

        # Текущий путь = [slug, ...parent_path]
        path_from_self = [slug] + parent_path

        # Уровень: 0 = root, max_depth = leaf, иначе mid
        if depth == 0:
            level = "root"
        elif depth >= max_depth or not (synset.hyponyms or []):
            level = "leaf"
        else:
            level = "mid"

        hierarchy[slug] = {
            "parent": parent_path[0] if parent_path else None,
            "level": level,
            "anchor_synset_id": synset.id,
            "label_ru": anchor_label_ru if depth == 0 else synset.title,
        }

        # Собрать леммы текущего синсета
        for sense in synset.senses:
            lemma = sense.name.lower()
            if not is_single_word_lemma(lemma):
                continue
            # Дедупликация одинаковых путей у одной леммы
            if path_from_self not in lemmas[lemma]:
                lemmas[lemma].append(path_from_self)

        # BFS глубже
        if depth < max_depth:
            for child_synset in (synset.hyponyms or []):
                if child_synset.id in visited_synsets:
                    continue
                child_slug = make_unique_slug(slug, child_synset.title, used_slugs)
                used_slugs.add(child_slug)
                queue.append((child_synset, child_slug, depth + 1, path_from_self))

    return hierarchy, dict(lemmas)


def merge_into_global(
    global_hierarchy: dict[str, dict],
    global_lemmas: dict[str, list[list[str]]],
    local_hierarchy: dict[str, dict],
    local_lemmas: dict[str, list[list[str]]],
) -> None:
    """Слить локальное дерево anchor'а в глобальную структуру."""
    # Hierarchy: разные anchor'ы имеют непересекающиеся slug-namespaces
    # (slug всегда начинается с anchor_slug), поэтому конфликтов быть не должно.
    for slug, meta in local_hierarchy.items():
        if slug in global_hierarchy:
            logger.warning("hierarchy slug collision: %s", slug)
        global_hierarchy[slug] = meta

    # Lemmas: лемма может быть в нескольких anchor-деревьях — добавляем все пути.
    for lemma, paths in local_lemmas.items():
        for path in paths:
            if path not in global_lemmas[lemma]:
                global_lemmas[lemma].append(path)


def resolve_anchor_synset(wn, anchor_name: str, anchor_cfg: dict):
    """Получить synset по resolved_synset_id (если есть) или по seed_lemma."""
    sid = anchor_cfg.get("resolved_synset_id")
    if sid:
        try:
            synset = wn[sid]
        except KeyError as e:
            raise RuntimeError(
                f"Anchor {anchor_name!r}: resolved_synset_id={sid!r} не найден в RuWordNet"
            ) from e
        return synset

    seed = anchor_cfg["seed_lemma"]
    senses = wn.get_senses(seed)
    verb_synsets = {s.synset.id: s.synset for s in senses if s.synset.part_of_speech == "V"}
    if not verb_synsets:
        raise RuntimeError(f"Anchor {anchor_name!r}: seed_lemma={seed!r} не дала verb-синсетов")
    # Если несколько — выбрать тот, у которого больше hyponyms (наиболее общий)
    synset = max(verb_synsets.values(), key=lambda s: len(s.hyponyms or []))
    logger.info(
        "Resolved anchor %s: %s → synset %s (%d hyponyms)",
        anchor_name, seed, synset.id, len(synset.hyponyms or [])
    )
    return synset


def build_lexicon(anchors_path: Path, max_depth: int) -> dict:
    """Главная функция: построить YAML-структуру v1 из anchors + RuWordNet."""
    try:
        from ruwordnet import RuWordNet
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Пакет 'ruwordnet' не установлен. Установите его в dev-окружении:\n"
            "  pip install ruwordnet\n"
            "  ruwordnet download"
        ) from e

    anchors_data = yaml.safe_load(anchors_path.read_text(encoding="utf-8"))
    anchors_cfg = anchors_data["anchors"]

    wn = RuWordNet()
    global_hierarchy: dict[str, dict] = {}
    global_lemmas: dict[str, list[list[str]]] = defaultdict(list)

    for anchor_name in sorted(anchors_cfg.keys()):
        cfg = anchors_cfg[anchor_name]
        synset = resolve_anchor_synset(wn, anchor_name, cfg)
        actual_hyponyms = len(synset.hyponyms or [])
        expected = cfg.get("expected_hyponyms")
        if expected is not None and actual_hyponyms != expected:
            logger.warning(
                "Anchor %s: expected %d hyponyms, found %d (RuWordNet drift?)",
                anchor_name, expected, actual_hyponyms
            )
        local_hierarchy, local_lemmas = collect_hierarchy(
            wn=wn,
            anchor_slug=anchor_name,
            anchor_synset=synset,
            anchor_label_ru=cfg["label_ru"],
            max_depth=max_depth,
        )
        logger.info(
            "Anchor %s: %d classes, %d unique lemmas",
            anchor_name, len(local_hierarchy), len(local_lemmas)
        )
        merge_into_global(global_hierarchy, global_lemmas, local_hierarchy, local_lemmas)

    # Метаданные
    anchors_sha = hashlib.sha256(anchors_path.read_bytes()).hexdigest()[:16]
    pkg_ver = pkg_version("ruwordnet") if pkg_version else "unknown"

    # Детерминированный порядок: hierarchy by slug, lemmas by lemma, paths sorted
    hierarchy_sorted = {
        slug: global_hierarchy[slug] for slug in sorted(global_hierarchy.keys())
    }
    lemmas_sorted: dict[str, list[list[str]]] = {}
    for lemma in sorted(global_lemmas.keys()):
        paths = sorted(global_lemmas[lemma], key=lambda p: (len(p), p))
        lemmas_sorted[lemma] = paths

    return {
        "version": 1,
        "metadata": {
            "source": "ruwordnet-2.0",
            "built_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ruwordnet_package_version": pkg_ver,
            "max_depth": max_depth,
            "anchor_count": len(anchors_cfg),
            "anchors_file_sha256": anchors_sha,
            "class_count": len(hierarchy_sorted),
            "lemma_count": len(lemmas_sorted),
        },
        "hierarchy": hierarchy_sorted,
        "lemmas": lemmas_sorted,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--anchors", type=Path, required=True,
        help="YAML с anchor-классами (например, configs/predicate_anchors.yaml)"
    )
    parser.add_argument(
        "--output", type=Path, required=True,
        help="Выходной YAML словаря version=1"
    )
    parser.add_argument(
        "--max-depth", type=int, default=3,
        help="Максимальная глубина BFS по hyponyms (default: 3)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        help="Уровень логирования (default: INFO)"
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s [%(levelname)s] %(message)s",
    )

    if not args.anchors.exists():
        logger.error("Anchors file not found: %s", args.anchors)
        return 2

    logger.info("Building lexicon from %s (max_depth=%d)", args.anchors, args.max_depth)
    lexicon = build_lexicon(args.anchors, args.max_depth)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        yaml.safe_dump(
            lexicon,
            f,
            allow_unicode=True,
            sort_keys=False,  # порядок уже зафиксирован вручную
            default_flow_style=None,
            width=120,
        )

    logger.info(
        "Wrote %s: %d classes, %d lemmas",
        args.output, lexicon["metadata"]["class_count"], lexicon["metadata"]["lemma_count"]
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
