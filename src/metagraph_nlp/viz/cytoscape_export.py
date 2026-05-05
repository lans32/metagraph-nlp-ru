"""Экспорт метаграфа в формат Cytoscape.js с compound nodes.

Генерирует JSON-элементы и self-contained HTML с Cytoscape.js из CDN.
Compound nodes обеспечивают визуальное вложение: L0-узлы вложены в L1,
L1 — в L2, и т.д. (CLAUDE.md §4.4). Инспектор свойств показывает
provenance каждого элемента (§7.4, §10).
"""

from __future__ import annotations

import json
from pathlib import Path

from metagraph_nlp.domain import Clause, Metagraph, SemanticGraph


def metagraph_to_cytoscape_elements(
    metagraph: Metagraph,
    graph: SemanticGraph,
    clauses: list[Clause],
) -> list[dict]:
    """Преобразовать метаграф + граф в список Cytoscape.js elements."""
    elements: list[dict] = []

    clause_text: dict[str, str] = {c.id: c.span.text for c in clauses}

    node_to_l1: dict[str, str] = {}
    for mn in metagraph.meta_nodes:
        if mn.level == 1:
            for nid in mn.fragment.node_ids:
                node_to_l1[nid] = mn.id

    l1_to_l2: dict[str, str] = {}
    for mn in metagraph.meta_nodes:
        if mn.level >= 2:
            for child_id in mn.fragment.meta_node_ids:
                if child_id not in l1_to_l2:
                    l1_to_l2[child_id] = mn.id

    for mn in metagraph.meta_nodes:
        parent = None
        if mn.level == 1:
            parent = l1_to_l2.get(mn.id)

        data: dict = {
            "id": mn.id,
            "label": (mn.label or mn.id)[:50],
            "kind": "metanode",
            "level": mn.level,
            "type": mn.type,
            "rule": mn.provenance.rule,
            "stage": mn.provenance.stage,
            "clause_text": clause_text.get(mn.provenance.clause_id or "", ""),
        }
        if parent:
            data["parent"] = parent
        elements.append({"data": data, "classes": f"metanode level-{mn.level}"})

    for n in graph.nodes:
        parent = node_to_l1.get(n.id)
        data = {
            "id": n.id,
            "label": n.label,
            "kind": "node",
            "level": 0,
            "lemma": n.lemma or "",
            "surface": n.surface or "",
            "upos": n.upos or n.kind,
            "rule": n.provenance.rule,
            "stage": n.provenance.stage,
            "clause_id": n.clause_id or "",
        }
        if parent:
            data["parent"] = parent
        elements.append({"data": data, "classes": "node level-0"})

    for e in graph.edges:
        elements.append({
            "data": {
                "id": e.id,
                "source": e.source,
                "target": e.target,
                "label": e.relation,
                "kind": "edge",
                "level": 0,
                "rule": e.provenance.rule,
            },
            "classes": "edge level-0",
        })

    for me in metagraph.meta_edges:
        elements.append({
            "data": {
                "id": me.id,
                "source": me.source,
                "target": me.target,
                "label": me.relation,
                "kind": "metaedge",
                "type": me.type,
                "level": me.level,
                "rule": me.provenance.rule,
            },
            "classes": f"metaedge level-{me.level}",
        })

    return elements


_HTML_TEMPLATE = """\
<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<title>Metagraph — Cytoscape.js Viewer</title>
<script src="https://unpkg.com/cytoscape@3.30.4/dist/cytoscape.min.js"></script>
<script src="https://unpkg.com/layout-base@2.0.1/layout-base.js"></script>
<script src="https://unpkg.com/cose-base@2.2.0/cose-base.js"></script>
<script src="https://unpkg.com/cytoscape-fcose@2.2.0/cytoscape-fcose.js"></script>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { font-family: "Segoe UI", Arial, sans-serif; display: flex; height: 100vh; }
  #cy { flex: 1; background: #fafafa; }
  #sidebar {
    width: 320px; background: #fff; border-left: 1px solid #ddd;
    padding: 12px; overflow-y: auto; font-size: 13px;
  }
  #sidebar h3 { margin-bottom: 8px; font-size: 15px; }
  #sidebar table { width: 100%; border-collapse: collapse; }
  #sidebar td { padding: 3px 4px; border-bottom: 1px solid #eee; vertical-align: top; }
  #sidebar td:first-child { font-weight: 600; white-space: nowrap; color: #555; }
  #controls { padding: 8px 12px; background: #f4f4f4; border-bottom: 1px solid #ddd; }
  #controls label { margin-right: 12px; cursor: pointer; user-select: none; }
  #main { display: flex; flex-direction: column; flex: 1; }
</style>
</head>
<body>
<div id="main">
  <div id="controls">
    <strong>Уровни:</strong>
    <label><input type="checkbox" class="level-toggle" data-level="0" checked> L0 (узлы)</label>
    <label><input type="checkbox" class="level-toggle" data-level="1" checked> L1 (клаузы)</label>
    <label><input type="checkbox" class="level-toggle" data-level="2" checked> L2 (параграфы)</label>
  </div>
  <div id="cy"></div>
</div>
<div id="sidebar">
  <h3>Inspector</h3>
  <p id="hint" style="color:#999">Нажмите на элемент графа</p>
  <table id="props" style="display:none"></table>
</div>
<script>
var elements = __ELEMENTS_JSON__;

var cy = cytoscape({
  container: document.getElementById("cy"),
  elements: elements,
  style: [
    {
      selector: "node.node",
      style: {
        "label": "data(label)",
        "font-size": 11,
        "background-color": "#90caf9",
        "text-valign": "center",
        "text-halign": "center",
        "width": 30, "height": 30,
        "shape": "ellipse",
        "border-width": 1,
        "border-color": "#42a5f5",
      }
    },
    {
      selector: "node.metanode.level-1",
      style: {
        "label": "data(label)",
        "font-size": 12,
        "background-color": "#e8f5e9",
        "background-opacity": 0.6,
        "border-width": 2,
        "border-color": "#66bb6a",
        "text-valign": "top",
        "text-halign": "center",
        "shape": "roundrectangle",
        "padding": "12px",
      }
    },
    {
      selector: "node.metanode.level-2",
      style: {
        "label": "data(label)",
        "font-size": 13,
        "background-color": "#fff3e0",
        "background-opacity": 0.4,
        "border-width": 3,
        "border-color": "#ff9800",
        "text-valign": "top",
        "text-halign": "center",
        "shape": "roundrectangle",
        "padding": "18px",
      }
    },
    {
      selector: "edge.edge",
      style: {
        "label": "data(label)",
        "font-size": 9,
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "line-color": "#999",
        "target-arrow-color": "#999",
        "width": 1.5,
      }
    },
    {
      selector: "edge.metaedge",
      style: {
        "label": "data(label)",
        "font-size": 10,
        "curve-style": "bezier",
        "target-arrow-shape": "triangle",
        "line-color": "#222",
        "target-arrow-color": "#222",
        "width": 3,
        "line-style": "solid",
      }
    },
    {
      selector: "edge.metaedge[type='topic_overlap']",
      style: {
        "line-color": "#2a6df4",
        "target-arrow-color": "#2a6df4",
        "line-style": "dashed",
      }
    },
    {
      selector: "edge.metaedge[type='shared_entity']",
      style: {
        "line-color": "#333",
        "target-arrow-color": "#333",
      }
    },
    {
      selector: ":selected",
      style: {
        "border-color": "#e53935",
        "border-width": 3,
        "line-color": "#e53935",
        "target-arrow-color": "#e53935",
      }
    },
  ],
  layout: {
    name: "fcose",
    animate: false,
    nodeDimensionsIncludeLabels: true,
    idealEdgeLength: 120,
    nodeRepulsion: 8000,
    edgeElasticity: 0.2,
    gravity: 0.3,
    gravityRange: 1.5,
    quality: "proof",
  },
});

cy.on("tap", "node, edge", function(evt) {
  var data = evt.target.data();
  var table = document.getElementById("props");
  var hint = document.getElementById("hint");
  hint.style.display = "none";
  table.style.display = "table";
  var html = "";
  var keys = Object.keys(data).sort();
  for (var i = 0; i < keys.length; i++) {
    var k = keys[i];
    if (k === "parent") continue;
    var v = data[k];
    if (typeof v === "object") v = JSON.stringify(v);
    html += "<tr><td>" + k + "</td><td>" + String(v) + "</td></tr>";
  }
  table.innerHTML = html;
});

cy.on("tap", function(evt) {
  if (evt.target === cy) {
    document.getElementById("props").style.display = "none";
    document.getElementById("hint").style.display = "block";
  }
});

document.querySelectorAll(".level-toggle").forEach(function(cb) {
  cb.addEventListener("change", function() {
    var level = this.dataset.level;
    var sel = ".level-" + level;
    if (this.checked) {
      cy.elements(sel).style("display", "element");
    } else {
      cy.elements(sel).style("display", "none");
    }
  });
});
</script>
</body>
</html>
"""


def render_cytoscape_html(
    metagraph: Metagraph,
    graph: SemanticGraph,
    clauses: list[Clause],
    out_path: Path,
) -> None:
    """Записать self-contained Cytoscape.js HTML-файл."""
    elements = metagraph_to_cytoscape_elements(metagraph, graph, clauses)
    elements_json = json.dumps(elements, ensure_ascii=False, indent=None)
    html = _HTML_TEMPLATE.replace("__ELEMENTS_JSON__", elements_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
