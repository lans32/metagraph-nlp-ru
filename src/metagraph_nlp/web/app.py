"""Streamlit веб-интерфейс для pipeline (п. 3.1.1 ТЗ).

Запуск: streamlit run src/metagraph_nlp/web/app.py

Двухфазная архитектура:
- Phase 1 (анализ текста) кэшируется в st.session_state.
- Phase 2 (агрегация) перезапускается мгновенно при смене пресета/тоглов.
"""

from __future__ import annotations

import hashlib
import sys
import tempfile
from pathlib import Path

_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st
import yaml

from metagraph_nlp.config import Config
from metagraph_nlp.pipeline import PipelineResult, Phase1Result, run_phase1, run_phase2
from metagraph_nlp.web.help import (
    TOOLTIPS_COMMON,
    TOOLTIPS_PHASE1,
    TOOLTIPS_PHASE2,
    render_help_tab,
)
from metagraph_nlp.web.presets import get_presets, get_preset_by_key, PRESET_KEYS


def _load_config(config_file) -> Config:
    if config_file is not None:
        data = yaml.safe_load(config_file.read())
        return Config.model_validate(data or {})
    return Config()


def _compute_cache_key(text: str, config: Config) -> str:
    text_hash = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{text_hash}_{config.phase1_hash()}"


def _render_phase1_config(config: Config) -> Config:
    """Настройки Phase 1: парсинг, клаузы, граф, анафора."""

    st.subheader("Анализ текста")
    st.caption("Изменение этих настроек требует повторного анализа")

    parser_options = ["natasha", "maltparser"]
    parser_labels = {
        "natasha": "Natasha (рекомендуется)",
        "maltparser": "MaltParser (внешний JAR)",
    }
    config.morphsyntax.parser = st.selectbox(
        "Парсер",
        options=parser_options,
        index=parser_options.index(config.morphsyntax.parser)
        if config.morphsyntax.parser in parser_options else 0,
        format_func=lambda x: parser_labels.get(x, x),
        help=TOOLTIPS_PHASE1["parser"],
    )

    clause_options = ["ud_subtree_clauses_v0", "sentence_as_clause_v0"]
    clause_labels = {
        "ud_subtree_clauses_v0": "По поддеревьям UD",
        "sentence_as_clause_v0": "Предложение = клауза",
    }
    config.clauses.strategy = st.selectbox(
        "Стратегия клауз",
        options=clause_options,
        index=clause_options.index(config.clauses.strategy)
        if config.clauses.strategy in clause_options else 0,
        format_func=lambda x: clause_labels.get(x, x),
        help=TOOLTIPS_PHASE1["clause_strategy"],
    )

    config.anaphora.enabled = st.toggle(
        "Разрешение анафоры",
        value=config.anaphora.enabled,
        help=TOOLTIPS_PHASE1["anaphora_enabled"],
    )
    if config.anaphora.enabled:
        _pronoun_type_labels = {
            "personal_3p": "Личные 3-го лица",
            "possessive_3p": "Притяжательные",
            "reflexive": "Возвратные",
        }
        selected_labels = st.multiselect(
            "Типы местоимений",
            options=list(_pronoun_type_labels.values()),
            default=[
                _pronoun_type_labels[t]
                for t in config.anaphora.pronoun_types
                if t in _pronoun_type_labels
            ],
            help=TOOLTIPS_PHASE1["anaphora_pronoun_types"],
        )
        label_to_type = {v: k for k, v in _pronoun_type_labels.items()}
        config.anaphora.pronoun_types = [label_to_type[l] for l in selected_labels]

        config.anaphora.search_window_sentences = st.slider(
            "Окно поиска антецедента",
            min_value=1, max_value=10,
            value=config.anaphora.search_window_sentences,
            help=TOOLTIPS_PHASE1["anaphora_search_window"],
        )
        config.anaphora.require_animacy_match = st.toggle(
            "Требовать совпадения одушевлённости",
            value=config.anaphora.require_animacy_match,
            help=TOOLTIPS_PHASE1["anaphora_animacy_match"],
        )

    config.aggregation.np_collapse_enabled = st.toggle(
        "Свёртка именных групп (NP collapse)",
        value=config.aggregation.np_collapse_enabled,
        help=TOOLTIPS_PHASE1["np_collapse"],
    )

    import os
    cpu_count = os.cpu_count() or 1
    config.morphsyntax.workers = st.slider(
        "Параллельные процессы парсинга",
        min_value=1,
        max_value=max(2, cpu_count),
        value=min(config.morphsyntax.workers, cpu_count),
        help=(
            f"{TOOLTIPS_PHASE1['workers']}\n\n"
            f"Активируется при ≥ {config.morphsyntax.parallel_threshold} предложений; "
            f"на вашем ПК {cpu_count} ядер."
        ),
    )

    return config


def _render_phase2_config(config: Config) -> Config:
    """Настройки Phase 2: агрегация (пресеты + ручные тоглы)."""

    st.subheader("Агрегация")
    st.caption("Изменения применяются мгновенно")

    presets = get_presets()
    preset_names = [p.name for p in presets]
    preset_keys = [p.key for p in presets]

    current_key = st.session_state.get("preset_key", "full")
    current_idx = preset_keys.index(current_key) if current_key in preset_keys else 4

    selected_name = st.selectbox(
        "Пресет",
        options=preset_names,
        index=current_idx,
        help=TOOLTIPS_PHASE2["preset"],
    )
    selected_idx = preset_names.index(selected_name)
    selected_key = preset_keys[selected_idx]
    st.session_state["preset_key"] = selected_key

    preset = presets[selected_idx]
    config = preset.apply(config)

    if selected_key != "custom":
        st.info(f"**{preset.name}**: {preset.description}")
    else:
        st.caption("**Уровень L1**")
        config.aggregation.linguistic_enabled = st.toggle(
            "Лингвистическая агрегация",
            value=config.aggregation.linguistic_enabled,
            help=TOOLTIPS_PHASE2["linguistic"],
        )
        config.aggregation.shared_entity_enabled = st.toggle(
            "Метарёбра shared_entity",
            value=config.aggregation.shared_entity_enabled,
            help=TOOLTIPS_PHASE2["shared_entity"],
        )
        if config.aggregation.shared_entity_enabled:
            config.aggregation.shared_entity_min_lemma_len = st.slider(
                "Мин. длина леммы",
                min_value=1, max_value=10,
                value=config.aggregation.shared_entity_min_lemma_len,
                help=TOOLTIPS_PHASE2["shared_entity_min_lemma_len"],
            )

        st.caption("**Уровень L2**")
        config.aggregation.paragraph_enabled = st.toggle(
            "По параграфам",
            value=config.aggregation.paragraph_enabled,
            help=TOOLTIPS_PHASE2["paragraph"],
        )
        config.aggregation.entity_cluster_enabled = st.toggle(
            "Кластеры (union-find)",
            value=config.aggregation.entity_cluster_enabled,
            help=TOOLTIPS_PHASE2["entity_cluster"],
        )
        if config.aggregation.entity_cluster_enabled:
            config.aggregation.entity_cluster_min_size = st.slider(
                "Мин. размер кластера",
                min_value=2, max_value=10,
                value=config.aggregation.entity_cluster_min_size,
                help=TOOLTIPS_PHASE2["entity_cluster_min_size"],
            )
        config.aggregation.entity_centric_enabled = st.toggle(
            "По сущностям (entity-centric)",
            value=config.aggregation.entity_centric_enabled,
            help=TOOLTIPS_PHASE2["entity_centric"],
        )
        if config.aggregation.entity_centric_enabled:
            config.aggregation.entity_centric_min_freq = st.slider(
                "Мин. частота сущности",
                min_value=1, max_value=10,
                value=config.aggregation.entity_centric_min_freq,
                help=TOOLTIPS_PHASE2["entity_centric_min_freq"],
            )
            config.aggregation.entity_centric_propn_always = st.toggle(
                "Имена собственные всегда",
                value=config.aggregation.entity_centric_propn_always,
                help=TOOLTIPS_PHASE2["entity_centric_propn_always"],
            )
        config.aggregation.predicate_class_cluster_enabled = st.toggle(
            "По классам предикатов",
            value=config.aggregation.predicate_class_cluster_enabled,
            help=TOOLTIPS_PHASE2["predicate_class"],
        )
        if config.aggregation.predicate_class_cluster_enabled:
            config.aggregation.predicate_class_cluster_min_size = st.slider(
                "Мин. размер predicate-класса",
                min_value=2, max_value=10,
                value=config.aggregation.predicate_class_cluster_min_size,
                help=TOOLTIPS_PHASE2["predicate_class_min_size"],
            )
        config.aggregation.topic_overlap_enabled = st.toggle(
            "Метарёбра topic_overlap",
            value=config.aggregation.topic_overlap_enabled,
            help=TOOLTIPS_PHASE2["topic_overlap"],
        )
        if config.aggregation.topic_overlap_enabled:
            config.aggregation.topic_overlap_min_overlap = st.slider(
                "Мин. пересечение",
                min_value=1, max_value=10,
                value=config.aggregation.topic_overlap_min_overlap,
                help=TOOLTIPS_PHASE2["topic_overlap_min_overlap"],
            )

    return config


def _viz_signature(result: PipelineResult) -> str:
    return (
        f"{len(result.graph.nodes)}_{len(result.graph.edges)}_"
        f"{len(result.metagraph.meta_nodes)}_{len(result.metagraph.meta_edges)}"
    )


def _render_results_tail(result: PipelineResult) -> None:
    """Таблицы, профилирование, аудит, экспорт — без визуализации."""
    if result.anaphora_resolutions is not None:
        with st.expander(
            f"Разрешение анафоры ({len(result.anaphora_resolutions)})"
        ):
            if result.anaphora_resolutions:
                anaphora_data = [
                    {
                        "Местоимение": r.pronoun_surface,
                        "Тип": r.pronoun_type,
                        "Антецедент": r.antecedent_lemma,
                        "Стратегия": r.resolution_strategy,
                        "Salience": r.salience_score
                        if r.salience_score is not None
                        else "—",
                        "Признаки": ", ".join(
                            f"{k}={v}" for k, v in r.matched_features.items()
                        ),
                        "PRON id": r.pronoun_node_id,
                        "Антецедент id": r.antecedent_node_id,
                    }
                    for r in result.anaphora_resolutions
                ]
                st.dataframe(anaphora_data, use_container_width=True)
            else:
                st.info("Подходящих местоимений не найдено.")

    with st.expander("Клаузы"):
        clause_data = []
        for c in result.clauses:
            clause_data.append({
                "ID": c.id,
                "Тип": c.clause_type or "-",
                "Предикат": c.head_lemma or "-",
                "Текст": c.span.text,
                "Предложение": c.sentence_id,
            })
        st.dataframe(clause_data, use_container_width=True)

    if result.metrics:
        with st.expander("Профилирование"):
            prof_data = []
            for s in result.metrics.stages:
                prof_data.append({
                    "Стадия": s.stage,
                    "Время (с)": round(s.wall_seconds, 4),
                    "Память (KB)": round(s.peak_memory_kb, 1),
                    "Выход": s.output_count,
                })
            st.dataframe(prof_data, use_container_width=True)
            st.metric("Общее время", f"{result.metrics.total_wall_seconds:.3f} с")

    with st.expander("Audit log"):
        st.code(result.audit.to_jsonl(), language="json")

    st.subheader("Экспорт")
    col_a, col_b, col_c = st.columns(3)
    with col_a:
        st.download_button(
            "Скачать граф (JSON)",
            result.graph.model_dump_json(indent=2),
            "semantic_graph.json",
            "application/json",
        )
    with col_b:
        st.download_button(
            "Скачать метаграф (JSON)",
            result.metagraph.model_dump_json(indent=2),
            "metagraph.json",
            "application/json",
        )
    with col_c:
        st.download_button(
            "Скачать audit log (JSONL)",
            result.audit.to_jsonl(),
            "audit.jsonl",
            "application/jsonl",
        )


def _render_results(result: PipelineResult, *, is_stale: bool = False) -> None:
    """Отображение результатов: статистика, визуализация, таблицы, экспорт."""

    if is_stale:
        st.warning(
            "Настройки анализа или текст изменились. "
            "Нажмите «Анализировать текст» для обновления."
        )

    st.subheader("Статистика")
    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Предложения", len(result.sentences))
    c2.metric("Клаузы", len(result.clauses))
    c3.metric("Узлы графа", len(result.graph.nodes))
    c4.metric("Рёбра графа", len(result.graph.edges))
    c5.metric("Метавершины", len(result.metagraph.meta_nodes))

    st.subheader("Визуализация")

    total_elements = (
        len(result.graph.nodes)
        + len(result.graph.edges)
        + len(result.metagraph.meta_nodes)
        + len(result.metagraph.meta_edges)
    )
    viz_limit = st.session_state.get("viz_limit", 500)
    is_large = total_elements > viz_limit
    sig = _viz_signature(result)
    render_choice = st.session_state.get(f"render_choice_{sig}")

    if is_large and render_choice is None:
        st.warning(
            f"Граф большой ({total_elements} элементов > {viz_limit}). "
            "Авто-рендер отключён, чтобы браузер не завис. "
            "Выберите режим визуализации вручную:"
        )
        c_l2, c_full, c_limit = st.columns([1, 1, 2])
        with c_l2:
            if st.button("Только L2 (быстро)", key=f"render_l2_{sig}"):
                st.session_state[f"render_choice_{sig}"] = "l2"
                st.rerun()
        with c_full:
            if st.button("Полный вид (медленно)", key=f"render_full_{sig}"):
                st.session_state[f"render_choice_{sig}"] = "full"
                st.rerun()
        with c_limit:
            new_limit = st.number_input(
                "Порог авто-отключения",
                min_value=50,
                max_value=10000,
                value=viz_limit,
                step=100,
                key="viz_limit_input",
            )
            if new_limit != viz_limit:
                st.session_state["viz_limit"] = new_limit
                st.rerun()
        st.caption("Таблицы клауз, профилирование, аудит и экспорт доступны ниже без рендера.")
        _render_results_tail(result)
        return

    if is_large:
        c_msg, c_btn = st.columns([4, 1])
        with c_msg:
            if render_choice == "l2":
                st.info(
                    f"Режим «Только L2» ({total_elements} элементов). "
                    "Включите L0/L1/рёбра чекбоксами в Cytoscape по необходимости."
                )
            else:
                st.info(f"Полный вид ({total_elements} элементов).")
        with c_btn:
            if st.button("Скрыть визуализацию", key=f"hide_viz_{sig}"):
                st.session_state.pop(f"render_choice_{sig}", None)
                st.rerun()

    simplify = is_large and render_choice == "l2"
    hidden_levels = [0, 1] if simplify else []
    hidden_etypes = (
        ["base", "shared_entity", "topic_overlap", "contains"] if simplify else []
    )

    with tempfile.TemporaryDirectory() as tmpdir:
        from metagraph_nlp.viz import (
            render_cytoscape_html,
            render_graph_html,
            render_metagraph_html,
        )

        cyto_html_path = Path(tmpdir) / "cytoscape.html"
        render_cytoscape_html(
            result.metagraph,
            result.graph,
            list(result.clauses),
            cyto_html_path,
            hidden_levels=hidden_levels,
            hidden_etypes=hidden_etypes,
        )

        if simplify:
            st.components.v1.html(
                cyto_html_path.read_text(encoding="utf-8"),
                height=700,
                scrolling=True,
            )
        else:
            graph_html_path = Path(tmpdir) / "graph.html"
            meta_html_path = Path(tmpdir) / "metagraph.html"
            render_graph_html(result.graph, list(result.clauses), graph_html_path)
            render_metagraph_html(
                result.metagraph, result.graph, list(result.clauses), meta_html_path
            )
            vtab_cyto, vtab_graph, vtab_meta = st.tabs(
                ["Метаграф (Cytoscape)", "Семантический граф", "Метаграф (pyvis)"]
            )
            with vtab_cyto:
                st.components.v1.html(
                    cyto_html_path.read_text(encoding="utf-8"),
                    height=700,
                    scrolling=True,
                )
            with vtab_graph:
                st.components.v1.html(
                    graph_html_path.read_text(encoding="utf-8"),
                    height=600,
                    scrolling=True,
                )
            with vtab_meta:
                st.components.v1.html(
                    meta_html_path.read_text(encoding="utf-8"),
                    height=600,
                    scrolling=True,
                )

    _render_results_tail(result)


def main() -> None:
    st.set_page_config(
        page_title="Metagraph NLP",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Метаграфовое представление русскоязычного текста")

    if "phase1_result" not in st.session_state:
        st.session_state["phase1_result"] = None
        st.session_state["phase1_cache_key"] = None

    with st.sidebar:
        st.header("Конфигурация pipeline")
        config_file = st.file_uploader(
            "Загрузить конфиг (YAML)",
            type=["yaml", "yml"],
            help=TOOLTIPS_COMMON["config_upload"],
        )
        config = _load_config(config_file)
        config = _render_phase1_config(config)

        st.divider()

        config = _render_phase2_config(config)

        with st.expander("Текущий конфиг (YAML)"):
            st.code(
                yaml.dump(config.model_dump(), allow_unicode=True, default_flow_style=False),
                language="yaml",
            )

    tab_text, tab_file, tab_help = st.tabs(["Ввод текста", "Загрузка файла", "Справка"])

    text_input = ""
    with tab_text:
        text_input = st.text_area(
            "Русскоязычный текст",
            height=200,
            placeholder="Студент читает книгу в библиотеке. Преподаватель объясняет студенту теорему на лекции.",
            help=TOOLTIPS_COMMON["text_area"],
        )

    with tab_file:
        uploaded = st.file_uploader(
            "Текстовый файл (.txt)",
            type=["txt"],
            help=TOOLTIPS_COMMON["file_uploader"],
        )
        if uploaded:
            raw_bytes = uploaded.read()
            text_input = None
            for enc in ("utf-8", "utf-8-sig", "cp1251", "cp866", "koi8-r"):
                try:
                    text_input = raw_bytes.decode(enc)
                    if enc != "utf-8":
                        st.caption(f"Файл прочитан в кодировке {enc}.")
                    break
                except UnicodeDecodeError:
                    continue
            if text_input is None:
                st.error("Не удалось определить кодировку файла. Сохраните файл в UTF-8.")
                text_input = ""
            else:
                st.text_area("Содержимое файла", text_input, height=150, disabled=True)

    with tab_help:
        render_help_tab()

    if st.button(
        "Анализировать текст",
        type="primary",
        help=TOOLTIPS_COMMON["run_button"],
    ):
        if not text_input or not text_input.strip():
            st.warning("Введите текст для обработки.")
        else:
            cache_key = _compute_cache_key(text_input, config)
            with st.spinner("Анализ текста (Phase 1)..."):
                p1 = run_phase1(text_input, config=config)
            st.session_state["phase1_result"] = p1
            st.session_state["phase1_cache_key"] = cache_key

    p1: Phase1Result | None = st.session_state.get("phase1_result")
    if p1 is None:
        st.info("Введите текст и нажмите «Анализировать текст».")
        return

    cache_key = _compute_cache_key(text_input, config) if text_input.strip() else ""
    is_stale = cache_key != st.session_state.get("phase1_cache_key")

    result = run_phase2(p1, config=config)
    _render_results(result, is_stale=is_stale)


if __name__ == "__main__":
    main()
