"""Визуализация семантического графа и метаграфа.

Модуль строго read-only относительно доменной модели:
никакого нового состояния в `domain/` не вводится. Два формата вывода:

- интерактивный HTML через pyvis (для разработки и оценки результатов);
- статический GraphViz DOT (для вставки в диплом/статью).
"""

from metagraph_nlp.viz.cytoscape_export import (
    metagraph_to_cytoscape_elements,
    render_cytoscape_html,
)
from metagraph_nlp.viz.dot import graph_to_dot, metagraph_to_dot
from metagraph_nlp.viz.html_pyvis import render_graph_html, render_metagraph_html

__all__ = [
    "graph_to_dot",
    "metagraph_to_cytoscape_elements",
    "metagraph_to_dot",
    "render_cytoscape_html",
    "render_graph_html",
    "render_metagraph_html",
]
