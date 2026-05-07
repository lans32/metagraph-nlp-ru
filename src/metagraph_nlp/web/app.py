"""Streamlit веб-интерфейс для pipeline (п. 3.1.1 ТЗ).

Запуск: streamlit run src/metagraph_nlp/web/app.py
"""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

# Bootstrap: путь к src/ может не подхватываться через editable .pth, если
# абсолютный путь содержит не-ASCII символы (cp1251 vs UTF-8 коллизия в site.py
# на русской Windows). Добавляем src/ в sys.path вручную через __file__.
_SRC_DIR = Path(__file__).resolve().parents[2]
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import streamlit as st
import yaml

from metagraph_nlp.config import Config
from metagraph_nlp.pipeline import run


def _load_config(config_file) -> Config:
    if config_file is not None:
        data = yaml.safe_load(config_file.read())
        return Config.model_validate(data or {})
    return Config()


def _render_sidebar_config(config: Config) -> Config:
    """Панель настроек pipeline в sidebar — структурированная, на русском."""

    # ── Морфосинтаксический разбор ──
    st.subheader("Морфосинтаксис")
    parser_options = ["natasha", "maltparser"]
    parser_labels = {
        "natasha": "Natasha (рекомендуется, встроенные модели)",
        "maltparser": "MaltParser (внешний JAR, требует Java)",
    }
    config.morphsyntax.parser = st.selectbox(
        "Парсер",
        options=parser_options,
        index=parser_options.index(config.morphsyntax.parser)
        if config.morphsyntax.parser in parser_options else 0,
        format_func=lambda x: parser_labels.get(x, x),
        help="Инструмент для морфологического и синтаксического анализа. "
             "Natasha — чисто-Python пакет для русского языка, работает из коробки.",
    )

    # ── Клаузы ──
    st.subheader("Выделение клауз")
    clause_options = ["ud_subtree_clauses_v0", "sentence_as_clause_v0"]
    clause_labels = {
        "ud_subtree_clauses_v0": "По поддеревьям UD (основная стратегия)",
        "sentence_as_clause_v0": "Одно предложение = одна клауза (упрощённая)",
    }
    config.clauses.strategy = st.selectbox(
        "Стратегия",
        options=clause_options,
        index=clause_options.index(config.clauses.strategy)
        if config.clauses.strategy in clause_options else 0,
        format_func=lambda x: clause_labels.get(x, x),
        help="UD-стратегия выделяет клаузы по финитным предикатам и их "
             "поддеревьям. Упрощённая — берёт каждое предложение целиком.",
    )

    # ── Построение графа ──
    st.subheader("Семантический граф")
    builder_options = ["ud_roles_v0"]
    builder_labels = {
        "ud_roles_v0": "По UD-ролям (nsubj, obj, obl, nmod)",
    }
    config.graph.builder = st.selectbox(
        "Билдер",
        options=builder_options,
        index=0,
        format_func=lambda x: builder_labels.get(x, x),
        help="Правило построения узлов и рёбер семантического графа из "
             "UD-разбора клаузы.",
    )

    # ── Разрешение анафоры ──
    st.subheader("Разрешение анафоры")
    config.anaphora.enabled = st.toggle(
        "Включить",
        value=config.anaphora.enabled,
        help="Местоимения заменяются на найденный антецедент. PRON-узел "
             "остаётся в графе, его лемма/upos обновляются на значения "
             "антецедента; surface и original_lemma сохраняются для "
             "трассируемости.",
    )
    if config.anaphora.enabled:
        _pronoun_type_labels = {
            "personal_3p": "Личные 3-го лица (он, она, оно, они)",
            "possessive_3p": "Притяжательные (его, её, их)",
            "reflexive": "Возвратные (себя, свой)",
        }
        selected_labels = st.multiselect(
            "Типы местоимений",
            options=list(_pronoun_type_labels.values()),
            default=[
                _pronoun_type_labels[t]
                for t in config.anaphora.pronoun_types
                if t in _pronoun_type_labels
            ],
            help="Какие типы местоимений разрешать. Возвратные используют "
                 "subject текущей клаузы; личные и притяжательные ищут "
                 "антецедент в окне предложений.",
        )
        label_to_type = {v: k for k, v in _pronoun_type_labels.items()}
        config.anaphora.pronoun_types = [label_to_type[l] for l in selected_labels]

        config.anaphora.search_window_sentences = st.slider(
            "Окно поиска антецедента",
            min_value=1,
            max_value=10,
            value=config.anaphora.search_window_sentences,
            help="Максимальное число предложений назад, в которых ищется "
                 "подходящий антецедент для местоимения. Применимо к "
                 "личным и притяжательным; возвратные смотрят в текущую клаузу.",
        )
        config.anaphora.require_animacy_match = st.toggle(
            "Требовать совпадения одушевлённости",
            value=config.anaphora.require_animacy_match,
            help="Если включено, антецедент должен совпадать с местоимением "
                 "по признаку Animacy (Anim/Inan). Если выключено — "
                 "используется только согласование Gender + Number.",
        )

    # ── Агрегация ──
    st.subheader("Агрегация")

    st.caption("**Предобработка графа**")
    config.aggregation.np_collapse_enabled = st.toggle(
        "Свёртка именных групп (NP collapse)",
        value=config.aggregation.np_collapse_enabled,
        help="Сворачивает цепочки NOUN + amod/nmod/det модификаторы в один "
             "узел с составной леммой. Запускается до агрегации.",
    )

    st.caption("**Уровень L1 (клаузы → метавершины)**")
    config.aggregation.linguistic_enabled = st.toggle(
        "Лингвистическая агрегация",
        value=config.aggregation.linguistic_enabled,
        help="Каждая клауза сжимается в метавершину первого уровня "
             "(правило clause_as_metanode_v0).",
    )
    config.aggregation.shared_entity_enabled = st.toggle(
        "Метарёбра shared_entity",
        value=config.aggregation.shared_entity_enabled,
        help="Создаёт метарёбра между L1-метавершинами, если они содержат "
             "общие леммы (правило shared_entity_by_lemma_v0).",
    )
    if config.aggregation.shared_entity_enabled:
        config.aggregation.shared_entity_min_lemma_len = st.slider(
            "Мин. длина леммы для shared_entity",
            min_value=1,
            max_value=10,
            value=config.aggregation.shared_entity_min_lemma_len,
            help="Леммы короче этого порога игнорируются при поиске "
                 "общих сущностей (фильтрация предлогов, частиц и т.п.).",
        )

    st.caption("**Уровень L2 (параграфы, кластеры)**")
    config.aggregation.paragraph_enabled = st.toggle(
        "Метавершины по параграфам",
        value=config.aggregation.paragraph_enabled,
        help="Группирует L1-метавершины одного параграфа в L2-метавершину "
             "(правило paragraph_clauses_v0).",
    )
    config.aggregation.entity_cluster_enabled = st.toggle(
        "Тематические кластеры (entity_cluster)",
        value=config.aggregation.entity_cluster_enabled,
        help="Создаёт L2-метавершины из связных компонент графа shared_entity "
             "(правило entity_cluster_v0). Группирует клаузы по общим "
             "сущностям-леммам, не по кореференции.",
    )
    if config.aggregation.entity_cluster_enabled:
        config.aggregation.entity_cluster_min_size = st.slider(
            "Мин. размер тематического кластера",
            min_value=2,
            max_value=10,
            value=config.aggregation.entity_cluster_min_size,
            help="Минимальное число L1-метавершин в компоненте, "
                 "чтобы он стал L2-метавершиной.",
        )
    config.aggregation.predicate_class_cluster_enabled = st.toggle(
        "Кластеры по классам предикатов",
        value=config.aggregation.predicate_class_cluster_enabled,
        help="Создаёт L2-метавершины, группируя клаузы по классам глаголов "
             "из словаря (motion, communication, cognition, ...). "
             "Правило predicate_class_cluster_v0.",
    )
    if config.aggregation.predicate_class_cluster_enabled:
        config.aggregation.predicate_class_cluster_min_size = st.slider(
            "Мин. размер predicate-класса",
            min_value=2,
            max_value=10,
            value=config.aggregation.predicate_class_cluster_min_size,
            help="Минимальное число клауз с предикатом одного класса, "
                 "чтобы они образовали L2-метавершину.",
        )
    config.aggregation.topic_overlap_enabled = st.toggle(
        "Метарёбра topic_overlap",
        value=config.aggregation.topic_overlap_enabled,
        help="Создаёт L2-метарёбра между L2-метавершинами с "
             "пересекающимися L1-фрагментами (правило topic_overlap_v0).",
    )
    if config.aggregation.topic_overlap_enabled:
        config.aggregation.topic_overlap_min_overlap = st.slider(
            "Мин. пересечение L1-фрагментов",
            min_value=1,
            max_value=10,
            value=config.aggregation.topic_overlap_min_overlap,
            help="Минимальное число общих L1-метавершин между двумя "
                 "L2-метавершинами для создания метаребра.",
        )

    st.caption("**Экспериментальные стратегии**")
    config.aggregation.structural_enabled = st.toggle(
        "Структурная агрегация",
        value=config.aggregation.structural_enabled,
        help="Агрегация по повторяющимся подграфам / изоморфизму "
             "(пока не реализована, зарезервировано).",
        disabled=True,
    )
    config.aggregation.semantic_enabled = st.toggle(
        "Семантическая агрегация",
        value=config.aggregation.semantic_enabled,
        help="Агрегация по косинусной близости в векторном пространстве "
             "(пока не реализована, зарезервировано).",
        disabled=True,
    )

    # ── Экспорт текущего конфига ──
    st.divider()
    with st.expander("Текущий конфиг (YAML)"):
        config_dict = config.model_dump()
        st.code(yaml.dump(config_dict, allow_unicode=True, default_flow_style=False),
                language="yaml")

    return config


def main() -> None:
    st.set_page_config(
        page_title="Metagraph NLP",
        page_icon="🔬",
        layout="wide",
    )
    st.title("Метаграфовое представление русскоязычного текста")

    # --- Sidebar: config ---
    with st.sidebar:
        st.header("Конфигурация pipeline")
        config_file = st.file_uploader(
            "Загрузить конфиг (YAML)",
            type=["yaml", "yml"],
            help="Загрузите свой YAML-файл конфигурации или настройте "
                 "параметры ниже вручную.",
        )
        config = _load_config(config_file)
        config = _render_sidebar_config(config)

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

        # --- Anaphora resolutions ---
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
                            "Δ предложений": r.distance_sentences,
                            "PRON id": r.pronoun_node_id,
                            "Антецедент id": r.antecedent_node_id,
                        }
                        for r in result.anaphora_resolutions
                    ]
                    st.dataframe(anaphora_data, use_container_width=True)
                else:
                    st.info("Подходящих местоимений не найдено.")

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
