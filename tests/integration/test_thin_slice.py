"""End-to-end pipeline с реальным natasha-парсером.

Эти тесты нагружают полный UD-стек, поэтому помечены `slow` и не
запускаются в быстром прогоне. Запустить: `pytest -m slow`.
"""

from pathlib import Path

import pytest

from metagraph_nlp.config import Config
from metagraph_nlp.pipeline import run, run_from_file

pytestmark = pytest.mark.slow

SAMPLE = (
    "Студент читает книгу в библиотеке. "
    "Преподаватель объясняет студенту теорему на лекции. "
    "Исследователь анализирует данные эксперимента."
)


def test_thin_slice_in_memory():
    result = run(SAMPLE, config=Config())

    assert result.document.normalized_text
    assert len(result.sentences) == 3
    assert len(result.clauses) >= 3
    assert {c.sentence_id for c in result.clauses} == {s.id for s in result.sentences}

    assert len(result.graph.nodes) >= 3
    assert len(result.graph.edges) >= 1

    # Лемматизация: в графе должна быть начальная форма "студент", не "Студент".
    labels = {n.label for n in result.graph.nodes}
    assert "студент" in labels

    l1_nodes = [mn for mn in result.metagraph.meta_nodes if mn.level == 1]
    l2_nodes = [mn for mn in result.metagraph.meta_nodes if mn.level == 2]
    assert len(l1_nodes) == len(result.clauses)
    for mn in l1_nodes:
        assert mn.type.startswith("clause:")

    # SAMPLE — одно-параграфный, дефолтный конфиг включает paragraph и
    # entity_cluster: ожидаем хотя бы одну paragraph- и одну entity_cluster-
    # L2-метавершину (общая лемма «студент» связывает первые две клаузы).
    para_l2 = [mn for mn in l2_nodes if mn.type == "paragraph"]
    cluster_l2 = [mn for mn in l2_nodes if mn.type == "entity_cluster"]
    assert len(para_l2) == 1
    assert para_l2[0].label.startswith("§0:")
    assert set(para_l2[0].fragment.meta_node_ids) == {mn.id for mn in l1_nodes}
    assert len(cluster_l2) >= 1
    # Кластер содержит как минимум две клаузы с общей сущностью.
    assert all(len(mn.fragment.meta_node_ids) >= 2 for mn in cluster_l2)

    rule_names = {e.rule for e in result.audit.events}
    assert "normalize_text" in rule_names
    assert "razdel.sentenize" in rule_names
    assert "ud_subtree_clauses_v0" in rule_names
    assert "ud_roles_v0" in rule_names
    assert "paragraph_clauses_v0" in rule_names
    assert "entity_cluster_v0" in rule_names
    assert "topic_overlap_v0" in rule_names

    # «студент» встречается в первой и второй клаузах — должно быть
    # хотя бы одно метаребро shared_entity.
    shared = [me for me in result.metagraph.meta_edges if me.type == "shared_entity"]
    assert shared, "ожидается хотя бы одно метаребро shared_entity"

    # При двух L2-стратегиях (paragraph + entity_cluster) их фрагменты
    # пересекаются по L1-клаузам — topic_overlap впервые непустой.
    topic = [me for me in result.metagraph.meta_edges if me.type == "topic_overlap"]
    assert topic, "ожидается хотя бы одно metaedge topic_overlap при двух L2-стратегиях"


def test_thin_slice_writes_artifacts(tmp_path: Path):
    input_path = tmp_path / "input.txt"
    input_path.write_text(SAMPLE, encoding="utf-8")
    out_dir = tmp_path / "run1"

    run_from_file(input_path, out_dir, config=Config())

    expected = [
        "document.json",
        "sentences.jsonl",
        "clauses.jsonl",
        "semantic_graph.json",
        "metagraph.json",
        "audit.jsonl",
        "config.snapshot.yaml",
    ]
    for name in expected:
        p = out_dir / name
        assert p.exists(), f"missing artifact: {name}"
        assert p.stat().st_size > 0


def test_thin_slice_writes_viz_artifacts(tmp_path: Path):
    input_path = tmp_path / "input.txt"
    input_path.write_text(SAMPLE, encoding="utf-8")
    out_dir = tmp_path / "run_viz"

    run_from_file(input_path, out_dir, config=Config(), viz=True)

    viz_files = [
        "semantic_graph.html",
        "metagraph.html",
        "semantic_graph.dot",
        "metagraph.dot",
    ]
    for name in viz_files:
        p = out_dir / name
        assert p.exists(), f"missing viz artifact: {name}"
        assert p.stat().st_size > 0


def test_thin_slice_predicate_hierarchy_multi_level():
    """End-to-end: RuWordNet v1 словарь → L2 predicate-кластеры на нескольких уровнях.

    Текст подобран так, чтобы все 4 глагола (читать/писать/говорить/слышать)
    попали в communication и/или perception ветви RuWordNet, давая
    предикат-кластеры на leaf+mid+root одновременно.
    """
    repo_root = Path(__file__).resolve().parents[2]
    ruwordnet_yaml = repo_root / "configs" / "predicate_classes_ruwordnet.yaml"
    if not ruwordnet_yaml.exists():
        pytest.skip("RuWordNet lexicon not built; run scripts/build_predicate_lexicon.py")

    cfg = Config()
    cfg.aggregation.predicate_classes_path = str(ruwordnet_yaml)
    cfg.aggregation.predicate_class_cluster_levels = ["leaf", "mid", "root"]
    cfg.aggregation.predicate_hierarchy_edges_enabled = True

    sample = (
        "Студент читает книгу. "
        "Преподаватель говорит студенту правду. "
        "Журналист пишет статью. "
        "Слушатель слышит лекцию."
    )

    result = run(sample, config=cfg)

    pred_l2 = [
        mn for mn in result.metagraph.meta_nodes
        if mn.type == "predicate_class"
    ]
    assert pred_l2, "ожидаются L2-кластеры по predicate_class"

    # На multi-level: должны быть label'ы как минимум 2 разных длин
    # (короткие root-классы типа 'communication' + длинные leaf-классы).
    label_lengths = {len(mn.label or "") for mn in pred_l2}
    assert len(label_lengths) >= 2, (
        f"ожидаются predicate-кластеры разных уровней, видим labels: "
        f"{[mn.label for mn in pred_l2]}"
    )

    # Хотя бы один кластер из communication-ветви (4 глагола → много путей).
    communication_clusters = [
        mn for mn in pred_l2
        if mn.label and mn.label.startswith("communication")
    ]
    assert communication_clusters, (
        f"ожидается хотя бы один кластер communication-ветви; "
        f"видим: {[mn.label for mn in pred_l2]}"
    )

    # provenance.notes должны содержать metadata иерархии для v1.
    for mn in communication_clusters:
        assert "level=" in (mn.provenance.notes or "")
        assert "anchor_synset=" in (mn.provenance.notes or "")

    # При predicate_hierarchy_edges_enabled должны быть containment-рёбра.
    containment_edges = [
        me for me in result.metagraph.meta_edges
        if me.type == "containment"
    ]
    if len(pred_l2) >= 2:
        # Если есть хотя бы 2 кластера на разных уровнях одной ветви —
        # должно быть хотя бы одно contains-ребро.
        assert containment_edges, (
            "ожидается хотя бы одно containment-метаребро при включённой "
            "иерархии и наличии нескольких уровней"
        )
        for e in containment_edges:
            assert e.relation == "contains"

    # Аудит: применена иерархия v1.
    pred_events = [
        ev for ev in result.audit.events
        if ev.rule == "predicate_class_cluster_v0"
    ]
    assert pred_events
    params = pred_events[0].params or {}
    assert params.get("hierarchy") == "v1"
