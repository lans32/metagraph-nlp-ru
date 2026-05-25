"""Интерактивный HTML-рендеринг через pyvis (vis.js).

Делает self-contained `.html` файл, который открывается двойным кликом в
браузере. Цель — быстро визуально оценивать качество клауз/графа/агрегации
во время разработки. Тултипы содержат provenance — ключевое свойство
объяснимости.
"""

from __future__ import annotations

import re
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


# Тулбар с ползунком масштаба и кнопкой полного экрана для pyvis-HTML.
# Инжектируется в верх <body>, чтобы пользователь видел контролы над сетью.
# Логарифмическая шкала слайдера (1× ≈ 67%) согласована с cytoscape_export.
_PYVIS_TOOLBAR = """
<style>
  #mg-toolbar {
    padding: 6px 12px;
    background: #f4f4f4;
    border-bottom: 1px solid #ddd;
    font-family: "Segoe UI", Arial, sans-serif;
    font-size: 13px;
    display: flex;
    align-items: center;
    flex-wrap: wrap;
    gap: 4px 10px;
  }
  #mg-toolbar strong { color: #555; min-width: 80px; }
  #mg-toolbar button {
    padding: 2px 9px; border: 1px solid #c0c0c0; background: #fff;
    border-radius: 3px; cursor: pointer; font-size: 13px; line-height: 1.4;
  }
  #mg-toolbar button:hover { background: #eaeaea; }
  #mg-toolbar button:active { background: #d8d8d8; }
  #mg-zoom-slider { width: 160px; vertical-align: middle; }
  #mg-zoom-value {
    display: inline-block; min-width: 44px; text-align: right;
    color: #555; font-variant-numeric: tabular-nums;
  }
  body.mg-pseudo-fullscreen {
    position: fixed; inset: 0; z-index: 9999;
    width: 100vw; height: 100vh; background: #fff; margin: 0;
    display: flex; flex-direction: column;
  }
  body.mg-pseudo-fullscreen #mynetwork {
    flex: 1 1 auto !important;
    height: auto !important;
    width: 100% !important;
  }
  body:fullscreen, body:-webkit-full-screen {
    display: flex; flex-direction: column; background: #fff;
  }
  body:fullscreen #mynetwork, body:-webkit-full-screen #mynetwork {
    flex: 1 1 auto !important;
    height: auto !important;
    width: 100% !important;
  }
</style>
<div id="mg-toolbar">
  <strong>Масштаб:</strong>
  <button type="button" id="mg-zoom-out" title="Уменьшить">−</button>
  <input type="range" id="mg-zoom-slider" min="0" max="100" step="1" value="67">
  <button type="button" id="mg-zoom-in" title="Увеличить">+</button>
  <span id="mg-zoom-value">100%</span>
  <button type="button" id="mg-fit-btn" title="Подогнать к экрану">⤢ Подогнать</button>
  <button type="button" id="mg-reset-btn" title="Сбросить масштаб (100%)">1:1</button>
  <button type="button" id="mg-fullscreen-btn" title="Полноэкранный режим (Esc — выход)">⛶ Полный экран</button>
</div>
<script>
(function() {
  var ZOOM_MIN = 0.1, ZOOM_MAX = 3;
  var LOG_LO = Math.log(ZOOM_MIN), LOG_HI = Math.log(ZOOM_MAX);
  var slider, zoomValue;

  function sliderToZoom(v) {
    return Math.exp(LOG_LO + (LOG_HI - LOG_LO) * v / 100);
  }
  function zoomToSliderPos(z) {
    var c = Math.min(Math.max(z, ZOOM_MIN), ZOOM_MAX);
    return (Math.log(c) - LOG_LO) / (LOG_HI - LOG_LO) * 100;
  }
  function getNet() {
    return (typeof network !== "undefined" && network) ? network : null;
  }
  function syncFromNet() {
    var net = getNet();
    if (!net || !slider || !zoomValue) return;
    var z = net.getScale();
    slider.value = zoomToSliderPos(z);
    zoomValue.textContent = Math.round(z * 100) + "%";
  }
  function applyZoom(z) {
    var net = getNet();
    if (!net) return;
    var clamped = Math.min(ZOOM_MAX, Math.max(ZOOM_MIN, z));
    net.moveTo({ scale: clamped });
  }

  function init() {
    slider = document.getElementById("mg-zoom-slider");
    zoomValue = document.getElementById("mg-zoom-value");
    if (!slider) return;

    slider.addEventListener("input", function() {
      applyZoom(sliderToZoom(parseFloat(this.value)));
      // moveTo не всегда триггерит событие zoom — обновим UI вручную:
      setTimeout(syncFromNet, 30);
    });
    document.getElementById("mg-zoom-in").addEventListener("click", function() {
      var net = getNet(); if (!net) return;
      applyZoom(net.getScale() * 1.25);
      setTimeout(syncFromNet, 30);
    });
    document.getElementById("mg-zoom-out").addEventListener("click", function() {
      var net = getNet(); if (!net) return;
      applyZoom(net.getScale() / 1.25);
      setTimeout(syncFromNet, 30);
    });
    document.getElementById("mg-fit-btn").addEventListener("click", function() {
      var net = getNet(); if (!net) return;
      net.fit();
      setTimeout(syncFromNet, 50);
    });
    document.getElementById("mg-reset-btn").addEventListener("click", function() {
      var net = getNet(); if (!net) return;
      net.moveTo({ scale: 1.0 });
      setTimeout(syncFromNet, 30);
    });

    function togglePseudo() {
      document.body.classList.toggle("mg-pseudo-fullscreen");
      setTimeout(function() {
        var net = getNet();
        if (net) { net.redraw(); net.fit(); syncFromNet(); }
      }, 80);
    }
    document.getElementById("mg-fullscreen-btn").addEventListener("click", function() {
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
        setTimeout(function() {
          var net = getNet();
          if (net) { net.redraw(); net.fit(); syncFromNet(); }
        }, 80);
      });
    });

    // Сеть появляется после drawGraph(); опрашиваем, пока она не появится,
    // затем подписываемся на изменения масштаба пользователем (колесо мыши).
    var attempts = 0;
    (function hook() {
      var net = getNet();
      if (!net) {
        if (attempts++ < 50) setTimeout(hook, 100);
        return;
      }
      try {
        net.setOptions({ interaction: { zoomMin: ZOOM_MIN, zoomMax: ZOOM_MAX } });
      } catch (e) { /* старый vis.js — пропускаем */ }
      net.on("zoom", syncFromNet);
      net.on("afterDrawing", function() {
        if (Math.abs(parseFloat(slider.value) - zoomToSliderPos(net.getScale())) > 0.5) {
          syncFromNet();
        }
      });
      syncFromNet();
    })();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", init);
  } else {
    init();
  }
})();
</script>
"""

_BODY_OPEN_RE = re.compile(r"(<body[^>]*>)", re.IGNORECASE)


def _inject_toolbar(html: str) -> str:
    """Вставить тулбар в верх <body>; если разметка нестандартная — в начало документа."""
    if _BODY_OPEN_RE.search(html):
        return _BODY_OPEN_RE.sub(
            lambda m: m.group(0) + "\n" + _PYVIS_TOOLBAR, html, count=1
        )
    return _PYVIS_TOOLBAR + html


def _write(net: Network, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # pyvis.Network.write_html на Windows открывает файл без encoding="utf-8"
    # и падает на кириллице. Берём HTML напрямую и пишем сами.
    html = net.generate_html(notebook=False)
    html = _inject_toolbar(html)
    out_path.write_text(html, encoding="utf-8")


def _clause_lookup(clauses: list[Clause]) -> dict[str, str]:
    return {c.id: c.span.text for c in clauses}


def _node_display_label(node: Node) -> str:
    """\u041c\u0435\u0442\u043a\u0430 \u0443\u0437\u043b\u0430 \u0434\u043b\u044f \u0432\u0438\u0437-\u0441\u043b\u043e\u044f.

    \u0414\u043b\u044f \u0443\u0437\u043b\u043e\u0432, \u0437\u0430\u043c\u0435\u043d\u0451\u043d\u043d\u044b\u0445 \u0430\u043d\u0430\u0444\u043e\u0440\u043e\u0439 (antecedent_node_id != None), \u043f\u043e\u043a\u0430\u0437\u044b\u0432\u0430\u0435\u043c
    \u00ab<\u043b\u0435\u043c\u043c\u0430 \u0430\u043d\u0442\u0435\u0446\u0435\u0434\u0435\u043d\u0442\u0430> \u2190<surface \u043c\u0435\u0441\u0442\u043e\u0438\u043c\u0435\u043d\u0438\u044f>\u00bb, \u0447\u0442\u043e\u0431\u044b \u0447\u0438\u0442\u0430\u0442\u0435\u043b\u044c \u0432\u0438\u0434\u0435\u043b
    \u0438 \u043a\u043e\u0433\u043e \u043f\u043e\u0434\u0441\u0442\u0430\u0432\u0438\u043b\u0438, \u0438 \u0433\u0434\u0435 \u0438\u043c\u0435\u043d\u043d\u043e \u0441\u0442\u043e\u044f\u043b\u043e \u043c\u0435\u0441\u0442\u043e\u0438\u043c\u0435\u043d\u0438\u0435.
    """
    if node.antecedent_node_id and node.surface:
        return f"{node.label} \u2190{node.surface}"
    return node.label


def _node_tooltip(node: Node, clause_text: str) -> str:
    lemma_line = f"{node.label}"
    if node.surface and node.surface != node.label:
        lemma_line = f"{node.label} \u2190 {node.surface}"
    upos = node.upos or node.kind
    extra = ""
    if node.antecedent_node_id:
        extra = (
            f"\nreplaced from pronoun: {node.original_lemma} "
            f"({node.original_upos})\nantecedent: {node.antecedent_node_id}"
        )
    return html_escape(
        f"{node.id}\n{lemma_line}\nupos: {upos}\nrule: {node.provenance.rule}\n"
        f"clause: {clause_text}{extra}"
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
            label=_node_display_label(n),
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
            label=_node_display_label(n),
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
