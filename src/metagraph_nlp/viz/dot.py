"""Экспорт в формат GraphViz DOT.

Чистые функции: возвращают строку, ничего не пишут на диск. Не требуют
установленного бинарника `graphviz` — рендеринг в SVG/PDF делает пользователь
отдельной командой `dot -Tsvg ...`.
"""

from __future__ import annotations

from metagraph_nlp.domain import Clause, Metagraph, Node, SemanticGraph
from metagraph_nlp.viz.palette import color_for


def _escape(s: str) -> str:
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ")


def _clause_text(clauses: list[Clause]) -> dict[str, str]:
    return {c.id: c.span.text for c in clauses}


def _node_display_label(node: Node) -> str:
    """Метка для виз-слоя; для заменённых анафорой — «Иван ←Он»."""
    if node.antecedent_node_id and node.surface:
        return f"{node.label} <-{node.surface}"
    return node.label


def _anaphora_tooltip_extra(node: Node) -> str:
    if not node.antecedent_node_id:
        return ""
    return (
        f" | replaced from pronoun: {node.original_lemma} "
        f"({node.original_upos}) | antecedent={node.antecedent_node_id}"
    )


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
        lemma_part = f"{n.label}"
        if n.surface and n.surface != n.label:
            lemma_part = f"{n.label} <- {n.surface}"
        tooltip = _escape(
            f"{n.id} | {lemma_part} | upos={n.upos or n.kind} | "
            f"rule={n.provenance.rule} | "
            f"clause: {by_clause.get(n.clause_id or '', '')}"
            f"{_anaphora_tooltip_extra(n)}"
        )
        lines.append(
            f'  "{n.id}" [label="{_escape(_node_display_label(n))}", '
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

    Двуслойное представление дендрограммы уровня 1.
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

    # Слой метавершин. L1 — прямоугольники; L2+ — прямоугольники с жирной рамкой
    # (визуально отличаем уровни, не полагаясь только на тултипы).
    for mn in metagraph.meta_nodes:
        color = color_for(mn.id)
        clause_text = by_clause.get(mn.provenance.clause_id or "", "")
        tooltip = _escape(
            f"{mn.id} | type={mn.type} | level={mn.level} | "
            f"rule={mn.provenance.rule} | clause: {clause_text}"
        )
        label = _escape((mn.label or mn.id)[:40])
        penwidth = 1 + mn.level  # level=1 → 2, level=2 → 3, и т.д.
        lines.append(
            f'  "{mn.id}" [shape=box, style="filled,rounded", '
            f'fillcolor="{color}", label="{label}", '
            f'penwidth={penwidth}, tooltip="{tooltip}"];'
        )

    # Слой 0: обычные узлы графа, окрашенные цветом «своей» метавершины.
    node_to_mnode: dict[str, str] = {}
    for mn in metagraph.meta_nodes:
        for nid in mn.fragment.node_ids:
            node_to_mnode[nid] = mn.id

    for n in graph.nodes:
        owner = node_to_mnode.get(n.id)
        color = color_for(owner) if owner else "#dddddd"
        lemma_part = f"{n.label}"
        if n.surface and n.surface != n.label:
            lemma_part = f"{n.label} <- {n.surface}"
        tooltip = _escape(
            f"{n.id} | {lemma_part} | upos={n.upos or n.kind} | "
            f"rule={n.provenance.rule} | "
            f"clause: {by_clause.get(n.clause_id or '', '')}"
            f"{_anaphora_tooltip_extra(n)}"
        )
        lines.append(
            f'  "{n.id}" [shape=ellipse, style="filled", '
            f'fillcolor="{color}", label="{_escape(_node_display_label(n))}", '
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

    # Пунктирные contains-рёбра от метавершин к их членам (no silent collapse).
    # L1: meta → L0 узлы (node_ids); L2+: meta → L1 meta (meta_node_ids).
    for mn in metagraph.meta_nodes:
        for nid in mn.fragment.node_ids:
            lines.append(
                f'  "{mn.id}" -> "{nid}" '
                f'[style="dashed", arrowhead=none, color="#888888", '
                f'label="contains"];'
            )
        for child_mid in mn.fragment.meta_node_ids:
            lines.append(
                f'  "{mn.id}" -> "{child_mid}" '
                f'[style="dashed", arrowhead=none, color="#555555", '
                f'label="contains"];'
            )

    # Метарёбра. shared_entity — серая жирная стрелка; topic_overlap — синий пунктир.
    for me in metagraph.meta_edges:
        tooltip = _escape(
            f"{me.id} | type={me.type} | level={me.level} | rule={me.provenance.rule}"
        )
        if me.type == "topic_overlap":
            style = 'penwidth=2, color="#2a6df4", style="dashed"'
        else:
            style = 'penwidth=2, color="#222222"'
        lines.append(
            f'  "{me.source}" -> "{me.target}" '
            f'[label="{_escape(me.relation)}", {style}, tooltip="{tooltip}"];'
        )

    lines.append("}")
    return "\n".join(lines)
