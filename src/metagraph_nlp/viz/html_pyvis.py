"""Интерактивный HTML-рендеринг через pyvis (vis.js).

Делает self-contained `.html` файл, который открывается двойным кликом в
браузере. Цель — быстро визуально оценивать качество клауз/графа/агрегации
во время разработки. Тултипы содержат provenance — ключевое свойство
объяснимости (CLAUDE.md §7.4, §10).
"""

from __future__ import annotations

from html import escape as html_escape
from pathlib import Path

from pyvis.network import Network

from metagraph_nlp.domain import Clause, Metagraph, Node, SemanticGraph
from metagraph_nlp.viz.palette import color_for


def _new_network() -> Network:
    net = Network(
        height="800px",
        width="100%",
        directed=True,
        bgcolor="#ffffff",
        font_color="#222222",
        notebook=False,
        cdn_resources="in_line",
    )
    net.toggle_physics(True)
    return net


def _write(net: Network, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # pyvis.Network.write_html на Windows открывает файл без encoding="utf-8"
    # и падает на кириллице. Берём HTML напрямую и пишем сами.
    html = net.generate_html(notebook=False)
    out_path.write_text(html, encoding="utf-8")


def _clause_lookup(clauses: list[Clause]) -> dict[str, str]:
    return {c.id: c.span.text for c in clauses}


def _node_tooltip(node: Node, clause_text: str) -> str:
    lemma_line = f"{node.label}"
    if node.surface and node.surface != node.label:
        lemma_line = f"{node.label} \u2190 {node.surface}"
    upos = node.upos or node.kind
    return html_escape(
        f"{node.id}\n{lemma_line}\nupos: {upos}\nrule: {node.provenance.rule}\n"
        f"clause: {clause_text}"
    )


def render_graph_html(
    graph: SemanticGraph,
    clauses: list[Clause],
    out_path: Path,
) -> None:
    """Семантический граф: узлы окрашены по clause_id, тултипы — provenance."""
    by_clause = _clause_lookup(clauses)
    net = _new_network()

    for n in graph.nodes:
        net.add_node(
            n.id,
            label=n.label,
            color=color_for(n.clause_id),
            shape="ellipse",
            title=_node_tooltip(n, by_clause.get(n.clause_id or "", "")),
        )

    for e in graph.edges:
        net.add_edge(
            e.source,
            e.target,
            label=e.relation,
            title=html_escape(f"{e.id}\nrelation: {e.relation}\nrule: {e.provenance.rule}"),
            arrows="to",
        )

    _write(net, out_path)


def render_metagraph_html(
    metagraph: Metagraph,
    graph: SemanticGraph,
    clauses: list[Clause],
    out_path: Path,
) -> None:
    """Метаграф уровня 1: метавершины + члены + пунктирные contains-рёбра."""
    by_clause = _clause_lookup(clauses)
    node_index = graph.node_index()
    net = _new_network()

    # Слой метавершин. L1 и L2+ рендерятся одинаково, но с разной толщиной
    # рамки — чтобы уровень был виден без тултипа.
    for mn in metagraph.meta_nodes:
        color = color_for(mn.id)
        clause_text = by_clause.get(mn.provenance.clause_id or "", "")
        title = html_escape(
            f"{mn.id}\ntype: {mn.type}\nlevel: {mn.level}\n"
            f"rule: {mn.provenance.rule}\nclause: {clause_text}"
        )
        label = (mn.label or mn.id)[:40]
        net.add_node(
            mn.id,
            label=label,
            color=color,
            shape="box",
            title=title,
            borderWidth=2 + mn.level,
            font={"size": 18, "bold": True},
        )

    # Слой 0: обычные узлы графа, окрашенные цветом своей метавершины.
    node_to_mnode: dict[str, str] = {}
    for mn in metagraph.meta_nodes:
        for nid in mn.fragment.node_ids:
            node_to_mnode[nid] = mn.id

    for n in graph.nodes:
        owner = node_to_mnode.get(n.id)
        color = color_for(owner) if owner else "#dddddd"
        net.add_node(
            n.id,
            label=n.label,
            color=color,
            shape="ellipse",
            title=_node_tooltip(n, by_clause.get(n.clause_id or "", "")),
        )

    # Сплошные предикатные рёбра.
    for e in graph.edges:
        if e.source in node_index and e.target in node_index:
            net.add_edge(
                e.source,
                e.target,
                label=e.relation,
                title=html_escape(f"{e.id}\nrule: {e.provenance.rule}"),
                arrows="to",
            )

    # Пунктирные contains-рёбра — метавершина → её узлы или дочерние метавершины.
    for mn in metagraph.meta_nodes:
        for nid in mn.fragment.node_ids:
            net.add_edge(
                mn.id,
                nid,
                label="contains",
                dashes=True,
                color="#888888",
                arrows="",
                title=html_escape(f"{mn.id} contains {nid}"),
            )
        for child_mid in mn.fragment.meta_node_ids:
            net.add_edge(
                mn.id,
                child_mid,
                label="contains",
                dashes=True,
                color="#555555",
                arrows="",
                title=html_escape(f"{mn.id} contains {child_mid}"),
            )

    # Метарёбра. shared_entity — чёрное; topic_overlap — синее пунктирное.
    for me in metagraph.meta_edges:
        color = "#2a6df4" if me.type == "topic_overlap" else "#222222"
        net.add_edge(
            me.source,
            me.target,
            label=me.relation,
            title=html_escape(
                f"{me.id}\ntype: {me.type}\nlevel: {me.level}\nrule: {me.provenance.rule}"
            ),
            arrows="to",
            width=3,
            color=color,
            dashes=(me.type == "topic_overlap"),
        )

    _write(net, out_path)
