"""Экспорт метаграфа в формат Cytoscape.js с compound nodes.

Генерирует JSON-элементы и self-contained HTML с Cytoscape.js из CDN.
Compound nodes обеспечивают визуальное вложение: L0-узлы вложены в L1,
L1 — в L2, и т.д. Инспектор свойств показывает
provenance каждого элемента.
"""

from __future__ import annotations

import json
from collections import defaultdict
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
        # Для узлов, заменённых анафорой, в label показываем «лемма ←surface»
        # (например, «Иван ←Он»), чтобы и подставленная сущность, и место
        # бывшего местоимения были видны в графе.
        display_label = n.label
        if n.antecedent_node_id and n.surface:
            display_label = f"{n.label} ←{n.surface}"
        data = {
            "id": n.id,
            "label": display_label,
            "kind": "node",
            "level": 0,
            "lemma": n.lemma or "",
            "surface": n.surface or "",
            "upos": n.upos or n.kind,
            "rule": n.provenance.rule,
            "stage": n.provenance.stage,
            "clause_id": n.clause_id or "",
            "original_lemma": n.original_lemma or "",
            "original_upos": n.original_upos or "",
            "antecedent_node_id": n.antecedent_node_id or "",
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
            "classes": "edge level-0 etype-base",
        })

    shared_groups: dict[tuple[str, str], list] = defaultdict(list)
    other_metaedges: list = []
    for me in metagraph.meta_edges:
        if me.type == "shared_entity":
            shared_groups[(me.source, me.target)].append(me)
        else:
            other_metaedges.append(me)

    for (src, tgt), group in shared_groups.items():
        lemmas = sorted({me.relation for me in group})
        count = len(lemmas)
        if count == 1:
            label = lemmas[0]
        else:
            label = f"{count} общих: {', '.join(lemmas[:3])}" + (
                "…" if count > 3 else ""
            )
        first = group[0]
        elements.append({
            "data": {
                "id": first.id,
                "source": src,
                "target": tgt,
                "label": label,
                "kind": "metaedge",
                "type": "shared_entity",
                "level": first.level,
                "rule": first.provenance.rule,
                "lemmas": ", ".join(lemmas),
                "bundled_count": count,
                "bundled_ids": ", ".join(me.id for me in group),
            },
            "classes": f"metaedge level-{first.level} etype-shared_entity",
        })

    for me in other_metaedges:
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
            "classes": f"metaedge level-{me.level} etype-{me.type}",
        })

    # Холархия: для каждой L1-метавершины, входящей более чем в один L2,
    # рисуем явные contains-рёбра от каждого «не-основного» L2 к этой L1.
    # Основной parent уже выражен через compound-nodes (data.parent), здесь
    # отражаются дополнительные membership-связи.
    for mn in metagraph.meta_nodes:
        if mn.level < 2:
            continue
        for child_id in mn.fragment.meta_node_ids:
            if l1_to_l2.get(child_id) == mn.id:
                continue  # основной parent — compound-nodes уже его показывают
            elements.append({
                "data": {
                    "id": f"contains_{mn.id}_{child_id}",
                    "source": mn.id,
                    "target": child_id,
                    "label": "contains",
                    "kind": "contains_edge",
                    "level": mn.level,
                    "rule": "holarchy_contains_v0",
                },
                "classes": f"contains-edge level-{mn.level} etype-contains",
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
  #controls { padding: 8px 12px; background: #f4f4f4; border-bottom: 1px solid #ddd; font-size: 13px; }
  #controls label { cursor: pointer; user-select: none; }
  #controls .row { margin-bottom: 4px; display: flex; align-items: center; flex-wrap: wrap; gap: 4px 12px; }
  #controls .row:last-child { margin-bottom: 0; }
  #controls strong { display: inline-block; min-width: 80px; }
  #controls .tb-btn {
    padding: 2px 9px; border: 1px solid #c0c0c0; background: #fff;
    border-radius: 3px; cursor: pointer; font-size: 13px; line-height: 1.4;
  }
  #controls .tb-btn:hover { background: #eaeaea; }
  #controls .tb-btn:active { background: #d8d8d8; }
  #zoom-slider { width: 160px; vertical-align: middle; }
  #zoom-value {
    display: inline-block; min-width: 44px; text-align: right;
    color: #555; font-variant-numeric: tabular-nums;
  }
  #main { display: flex; flex-direction: column; flex: 1; min-width: 0; }
  body.pseudo-fullscreen {
    position: fixed; inset: 0; z-index: 9999;
    width: 100vw; height: 100vh; background: #fff;
  }
  body:fullscreen, body:-webkit-full-screen { background: #fff; }
</style>
</head>
<body>
<div id="main">
  <div id="controls">
    <div class="row">
      <strong>Уровни:</strong>
      <label><input type="checkbox" class="level-toggle" data-level="0" __L0_CHECKED__> L0 (узлы)</label>
      <label><input type="checkbox" class="level-toggle" data-level="1" __L1_CHECKED__> L1 (клаузы)</label>
      <label><input type="checkbox" class="level-toggle" data-level="2" __L2_CHECKED__> L2 (агрегаты)</label>
    </div>
    <div class="row">
      <strong>Рёбра:</strong>
      <label><input type="checkbox" class="etype-toggle" data-etype="base" __E_BASE_CHECKED__> базовые (L0)</label>
      <label><input type="checkbox" class="etype-toggle" data-etype="shared_entity" __E_SHARED_ENTITY_CHECKED__> shared_entity</label>
      <label><input type="checkbox" class="etype-toggle" data-etype="topic_overlap" __E_TOPIC_OVERLAP_CHECKED__> topic_overlap</label>
      <label><input type="checkbox" class="etype-toggle" data-etype="contains" __E_CONTAINS_CHECKED__> contains</label>
    </div>
    <div class="row">
      <strong>Масштаб:</strong>
      <button type="button" class="tb-btn" id="zoom-out" title="Уменьшить">−</button>
      <input type="range" id="zoom-slider" min="0" max="100" step="1" value="67">
      <button type="button" class="tb-btn" id="zoom-in" title="Увеличить">+</button>
      <span id="zoom-value">100%</span>
      <button type="button" class="tb-btn" id="fit-btn" title="Подогнать к экрану">⤢ Подогнать</button>
      <button type="button" class="tb-btn" id="reset-btn" title="Сбросить масштаб (100%)">1:1</button>
      <button type="button" class="tb-btn" id="fullscreen-btn" title="Полноэкранный режим (Esc — выход)">⛶ Полный экран</button>
    </div>
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
  minZoom: 0.1,
  maxZoom: 3,
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
        "width": "mapData(bundled_count, 1, 10, 2, 8)",
      }
    },
    {
      selector: "edge.contains-edge",
      style: {
        "label": "data(label)",
        "font-size": 8,
        "color": "#bbb",
        "curve-style": "bezier",
        "target-arrow-shape": "tee",
        "line-color": "#cfd8dc",
        "target-arrow-color": "#cfd8dc",
        "line-style": "dotted",
        "width": 1.2,
        "opacity": 0.7,
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

var hiddenLevels = __HIDDEN_LEVELS_JSON__;
hiddenLevels.forEach(function(lvl) {
  cy.elements(".level-" + lvl).style("display", "none");
});

var hiddenEtypes = __HIDDEN_ETYPES_JSON__;
hiddenEtypes.forEach(function(et) {
  cy.elements(".etype-" + et).style("display", "none");
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

document.querySelectorAll(".etype-toggle").forEach(function(cb) {
  cb.addEventListener("change", function() {
    var etype = this.dataset.etype;
    var sel = ".etype-" + etype;
    if (this.checked) {
      cy.elements(sel).style("display", "element");
    } else {
      cy.elements(sel).style("display", "none");
    }
  });
});

// Управление масштабом (логарифмический ползунок) и полным экраном.
// Логарифмическая шкала: 1× попадает в ~67% слайдера — визуально центрирует
// «нейтральный» масштаб, что удобнее линейного отображения для диапазона 0.1–3.
(function() {
  var ZOOM_MIN = 0.1, ZOOM_MAX = 3;
  var LOG_LO = Math.log(ZOOM_MIN), LOG_HI = Math.log(ZOOM_MAX);
  var slider = document.getElementById("zoom-slider");
  var zoomValue = document.getElementById("zoom-value");

  function sliderToZoom(v) {
    return Math.exp(LOG_LO + (LOG_HI - LOG_LO) * v / 100);
  }
  function zoomToSliderPos(z) {
    var clamped = Math.min(Math.max(z, ZOOM_MIN), ZOOM_MAX);
    return (Math.log(clamped) - LOG_LO) / (LOG_HI - LOG_LO) * 100;
  }
  function syncSliderFromCy() {
    var z = cy.zoom();
    slider.value = zoomToSliderPos(z);
    zoomValue.textContent = Math.round(z * 100) + "%";
  }
  function applyZoom(z) {
    var w = cy.width(), h = cy.height();
    cy.zoom({ level: z, renderedPosition: { x: w / 2, y: h / 2 } });
  }

  slider.addEventListener("input", function() {
    applyZoom(sliderToZoom(parseFloat(this.value)));
  });
  document.getElementById("zoom-in").addEventListener("click", function() {
    applyZoom(Math.min(ZOOM_MAX, cy.zoom() * 1.25));
  });
  document.getElementById("zoom-out").addEventListener("click", function() {
    applyZoom(Math.max(ZOOM_MIN, cy.zoom() / 1.25));
  });
  document.getElementById("fit-btn").addEventListener("click", function() {
    cy.fit(undefined, 30);
  });
  document.getElementById("reset-btn").addEventListener("click", function() {
    cy.zoom(1); cy.center();
  });

  cy.on("zoom", syncSliderFromCy);
  syncSliderFromCy();

  // Полный экран: пытаемся Fullscreen API; если запрещено (sandbox/iframe
  // без allowfullscreen) — переходим в CSS-«псевдополный» режим, который
  // растягивает body на весь viewport (внутри iframe это всё, что возможно).
  function togglePseudo() {
    document.body.classList.toggle("pseudo-fullscreen");
    setTimeout(function() { cy.resize(); }, 60);
  }
  document.getElementById("fullscreen-btn").addEventListener("click", function() {
    var doc = document;
    var fsEl = doc.fullscreenElement || doc.webkitFullscreenElement || doc.msFullscreenElement;
    if (fsEl) {
      (doc.exitFullscreen || doc.webkitExitFullscreen || doc.msExitFullscreen).call(doc);
      return;
    }
    var el = doc.documentElement;
    var req = el.requestFullscreen || el.webkitRequestFullscreen || el.msRequestFullscreen;
    if (req) {
      try {
        var p = req.call(el);
        if (p && p.catch) p.catch(togglePseudo);
      } catch (e) { togglePseudo(); }
    } else {
      togglePseudo();
    }
  });
  ["fullscreenchange", "webkitfullscreenchange", "msfullscreenchange"].forEach(function(ev) {
    document.addEventListener(ev, function() {
      setTimeout(function() { cy.resize(); }, 60);
    });
  });
})();
</script>
</body>
</html>
"""


_ALL_ETYPES = ("base", "shared_entity", "topic_overlap", "contains")


def render_cytoscape_html(
    metagraph: Metagraph,
    graph: SemanticGraph,
    clauses: list[Clause],
    out_path: Path,
    *,
    hidden_levels: list[int] | None = None,
    hidden_etypes: list[str] | None = None,
) -> None:
    """Записать self-contained Cytoscape.js HTML-файл.

    Args:
        hidden_levels: уровни (0/1/2), чекбоксы которых стартуют
            выключенными. Используется для больших графов: ``[0, 1]``
            скрывает базовый слой и L1, оставляя только L2-агрегаты.
        hidden_etypes: типы рёбер (``base`` / ``shared_entity`` /
            ``topic_overlap`` / ``contains``), стартующие скрытыми.
    """
    elements = metagraph_to_cytoscape_elements(metagraph, graph, clauses)
    elements_json = json.dumps(elements, ensure_ascii=False, indent=None)
    hidden = set(hidden_levels or [])
    hidden_e = set(hidden_etypes or [])
    html = _HTML_TEMPLATE.replace("__ELEMENTS_JSON__", elements_json)
    html = html.replace(
        "__HIDDEN_LEVELS_JSON__",
        json.dumps(sorted(hidden)),
    )
    html = html.replace(
        "__HIDDEN_ETYPES_JSON__",
        json.dumps(sorted(hidden_e)),
    )
    for lvl in (0, 1, 2):
        token = f"__L{lvl}_CHECKED__"
        html = html.replace(token, "" if lvl in hidden else "checked")
    for etype in _ALL_ETYPES:
        token = f"__E_{etype.upper()}_CHECKED__"
        html = html.replace(token, "" if etype in hidden_e else "checked")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(html, encoding="utf-8")
