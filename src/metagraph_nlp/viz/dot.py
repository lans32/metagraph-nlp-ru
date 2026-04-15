"""Экспорт в формат GraphViz DOT.

Чистые функции: возвращают строку, ничего не пишут на диск. Не требуют
установленного бинарника `graphviz` — рендеринг в SVG/PDF делает пользователь
отдельной командой `dot -Tsvg ...`.
"""

from __future__ import annotations

from metagraph_nlp.domain import Clause, Metagraph, SemanticGraph
from metagraph_nlp.viz.palette import color_for


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _clause_text(clauses: list[Clause]) -> dict[str, str]:
    return {c.id: c.span.text for c in clauses}


def graph_to_dot(graph: SemanticGraph, clauses: list[Clause]) -> str:
    """DOT для семантического графа: узлы окрашены по clause_id."""
    by_clause = _clause_text(clauses)
    lines: list[str] = [
        f'digraph semantic_graph_{_escape(graph.document_id)} {{',
        '  rankdir=LR;',
        '  node [shape=ellipse, style="filled", fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]

    for n in graph.nodes:
        color = color_for(n.clause_id)
        tooltip = _escape(
            f"{n.id} | kind={n.kind} | rule={n.provenance.rule} | "
            f"clause: {by_clause.get(n.clause_id or '', '')}"
        )
        lines.append(
            f'  "{n.id}" [label="{_escape(n.label)}", '
            f'fillcolor="{color}", tooltip="{tooltip}"];'
        )

    for e in graph.edges:
        tooltip = _escape(f"{e.id} | rule={e.provenance.rule}")
        lines.append(
            f'  "{e.source}" -> "{e.target}" '
            f'[label="{_escape(e.relation)}", tooltip="{tooltip}"];'
        )

    lines.append("}")
    return "\n".join(lines)


def metagraph_to_dot(
    metagraph: Metagraph,
    graph: SemanticGraph,
    clauses: list[Clause],
) -> str:
    """DOT для метаграфа: метавершины + их члены + пунктирные contains-рёбра.

    Двуслойное представление дендрограммы уровня 1 (CLAUDE.md §4.4).
    """
    by_clause = _clause_text(clauses)
    node_index = graph.node_index()

    lines: list[str] = [
        f'digraph metagraph_{_escape(metagraph.document_id)} {{',
        '  rankdir=LR;',
        '  compound=true;',
        '  node [fontname="Helvetica"];',
        '  edge [fontname="Helvetica"];',
    ]

    # Слой 1: метавершины (большие квадраты).
    for mn in metagraph.meta_nodes:
        color = color_for(mn.id)
        clause_text = by_clause.get(mn.provenance.clause_id or "", "")
        tooltip = _escape(
            f"{mn.id} | type={mn.type} | level={mn.level} | "
            f"rule={mn.provenance.rule} | clause: {clause_text}"
        )
        label = _escape((mn.label or mn.id)[:40])
        lines.append(
            f'  "{mn.id}" [shape=box, style="filled,rounded", '
            f'fillcolor="{color}", label="{label}", tooltip="{tooltip}"];'
        )

    # Слой 0: обычные узлы графа, окрашенные цветом «своей» метавершины.
    node_to_mnode: dict[str, str] = {}
    for mn in metagraph.meta_nodes:
        for nid in mn.fragment.node_ids:
            node_to_mnode[nid] = mn.id

    for n in graph.nodes:
        owner = node_to_mnode.get(n.id)
        color = color_for(owner) if owner else "#dddddd"
        tooltip = _escape(
            f"{n.id} | kind={n.kind} | rule={n.provenance.rule} | "
            f"clause: {by_clause.get(n.clause_id or '', '')}"
        )
        lines.append(
            f'  "{n.id}" [shape=ellipse, style="filled", '
            f'fillcolor="{color}", label="{_escape(n.label)}", '
            f'tooltip="{tooltip}"];'
        )

    # Сплошные стрелки — рёбра-предикаты исходного графа.
    for e in graph.edges:
        if e.source in node_index and e.target in node_index:
            tooltip = _escape(f"{e.id} | rule={e.provenance.rule}")
            lines.append(
                f'  "{e.source}" -> "{e.target}" '
                f'[label="{_escape(e.relation)}", tooltip="{tooltip}"];'
            )

    # Пунктирные contains-рёбра от метавершин к их членам (§9.4 no silent collapse).
    for mn in metagraph.meta_nodes:
        for nid in mn.fragment.node_ids:
            lines.append(
                f'  "{mn.id}" -> "{nid}" '
                f'[style="dashed", arrowhead=none, color="#888888", '
                f'label="contains"];'
            )

    # Метарёбра (двойная стрелка). На уровне 1 их пока нет, но рендер готов.
    for me in metagraph.meta_edges:
        tooltip = _escape(f"{me.id} | type={me.type} | rule={me.provenance.rule}")
        lines.append(
            f'  "{me.source}" -> "{me.target}" '
            f'[label="{_escape(me.relation)}", penwidth=2, color="#222222", '
            f'tooltip="{tooltip}"];'
        )

    lines.append("}")
    return "\n".join(lines)
