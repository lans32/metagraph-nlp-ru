"""Streamlit веб-интерфейс для pipeline (п. 3.1.1 ТЗ).

Запуск: streamlit run src/metagraph_nlp/web/app.py
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import streamlit as st
import yaml

from metagraph_nlp.config import Config
from metagraph_nlp.pipeline import run


def _load_config(config_file) -> Config:
    if config_file is not None:
        data = yaml.safe_load(config_file.read())
        return Config.model_validate(data or {})
    return Config()


def main() -> None:
    st.set_page_config(
        page_title="Metagraph NLP",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Метаграфовое представление русскоязычного текста")

    # --- Sidebar: config ---
    with st.sidebar:
        st.header("Настройки")
        config_file = st.file_uploader("Конфиг (YAML)", type=["yaml", "yml"])
        config = _load_config(config_file)
        with st.expander("Текущий конфиг"):
            st.json(config.model_dump())

    # --- Input ---
    tab_text, tab_file = st.tabs(["Ввод текста", "Загрузка файла"])

    text_input = ""
    with tab_text:
        text_input = st.text_area(
            "Русскоязычный текст",
            height=200,
            placeholder="Студент читает книгу в библиотеке. Преподаватель объясняет студенту теорему на лекции.",
        )

    with tab_file:
        uploaded = st.file_uploader("Текстовый файл (.txt)", type=["txt"])
        if uploaded:
            text_input = uploaded.read().decode("utf-8")
            st.text_area("Содержимое файла", text_input, height=150, disabled=True)

    # --- Process ---
    if st.button("Обработать", type="primary"):
        if not text_input or not text_input.strip():
            st.warning("Введите текст для обработки.")
            return

        with st.spinner("Обработка через pipeline..."):
            result = run(text_input, config=config)

        # --- Statistics ---
        st.subheader("Статистика")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Предложения", len(result.sentences))
        c2.metric("Клаузы", len(result.clauses))
        c3.metric("Узлы графа", len(result.graph.nodes))
        c4.metric("Рёбра графа", len(result.graph.edges))
        c5.metric("Метавершины", len(result.metagraph.meta_nodes))

        # --- Visualization ---
        st.subheader("Визуализация")
        with tempfile.TemporaryDirectory() as tmpdir:
            from metagraph_nlp.viz import (
                render_cytoscape_html,
                render_graph_html,
                render_metagraph_html,
            )

            graph_html_path = Path(tmpdir) / "graph.html"
            meta_html_path = Path(tmpdir) / "metagraph.html"
            cyto_html_path = Path(tmpdir) / "cytoscape.html"
            render_graph_html(result.graph, list(result.clauses), graph_html_path)
            render_metagraph_html(
                result.metagraph, result.graph, list(result.clauses), meta_html_path
            )
            render_cytoscape_html(
                result.metagraph, result.graph, list(result.clauses), cyto_html_path
            )

            vtab_graph, vtab_meta, vtab_cyto = st.tabs(
                ["Семантический граф", "Метаграф (pyvis)", "Метаграф (Cytoscape)"]
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
            with vtab_cyto:
                st.components.v1.html(
                    cyto_html_path.read_text(encoding="utf-8"),
                    height=700,
                    scrolling=True,
                )

        # --- Clauses table ---
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

        # --- Profiling ---
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

        # --- Audit log ---
        with st.expander("Audit log"):
            st.code(result.audit.to_jsonl(), language="json")

        # --- Export ---
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


if __name__ == "__main__":
    main()
