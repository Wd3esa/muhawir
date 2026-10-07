/* Teaching simulations only. No API calls, model requests or analytics. */
"use strict";
const Learning = (() => {
  const $ = id => document.getElementById(id);
  const svgNS = "http://www.w3.org/2000/svg";
  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  let content, view, scenario, selected, index = 0, playing = false;
  let timer = null, frame = null, startTime = 0, packetPath = null;
  const byId = new Map();
  const kind = node => node.kind.toLowerCase().split(" ")[0];
  const element = (tag, className, text) => {
    const el = document.createElement(tag);
    if (className) el.className = className;
    if (text !== undefined) el.textContent = text;
    return el;
  };
  const svgElement = (tag, attrs) => {
    const el = document.createElementNS(svgNS, tag);
    for (const [key, value] of Object.entries(attrs)) el.setAttribute(key, value);
    return el;
  };

  // One-based ranks; RRF combines positions, not incomparable raw scores.
  function fuseRanks(lexical, semantic, k = 60) {
    const scores = new Map();
    for (const list of [lexical, semantic]) list.forEach((id, i) => {
      scores.set(id, (scores.get(id) || 0) + 1 / (k + i + 1));
    });
    return [...scores].map(([id, score]) => ({id, score}))
      .sort((a, b) => b.score - a.score || a.id.localeCompare(b.id));
  }

  function scenariosFor(current) {
    if (current.id === "flow") return content.scenarios;
    if (current.id === "data") return [{id:"build", title:"Build corpus and indexes", route:current.nodes,
      outcome:"Lexical and vector indexes are two branches from the same corpus. The vector branch is optional; indexes are prepared before serving questions."}];
    return [
      {id:"traffic", title:"A visitor reaches Oracle", route:["browser","caddy","uvicorn","fastapi","models","fastapi","operations"], outcome:"The browser uses HTTPS. FastAPI coordinates the answer pipeline; provider calls and checks happen inside it. This is a connection tour, not a live request."},
      {id:"release", title:"Release and backup tour", route:["github","ci","oracle","uvicorn","fastapi","operations","render"], outcome:"Offline CI checks source changes. The owner’s Oracle updater is separate from CI. Render is another deployment and URL; the diagram does not promise automatic failover."},
      {id:"voice", title:"Optional voice interface", route:["browser","speech"], outcome:"Speech recognition and synthesis are separate from evidence retrieval. Browser support and external speech services vary; typed chat remains available."}
    ];
  }

  function stop() {
    playing = false;
    clearTimeout(timer); timer = null;
    cancelAnimationFrame(frame); frame = null;
    $("play").textContent = "Play";
    $("wires").querySelector(".packet")?.remove();
  }

  function chooseView(id, focusTab = false) {
    stop();
    view = content.views.find(item => item.id === id);
    index = 0;
    for (const tab of $("tabs").children) {
      const active = tab.dataset.view === id;
      tab.setAttribute("aria-selected", String(active));
      tab.tabIndex = active ? 0 : -1;
      if (active && focusTab) tab.focus();
    }
    $("map-panel").setAttribute("aria-labelledby", `tab-${id}`);
    $("view-title").textContent = view.title;
    $("view-subtitle").textContent = view.subtitle;
    const options = scenariosFor(view);
    $("scenario").replaceChildren(...options.map(item => {
      const option = element("option", "", item.title); option.value = item.id; return option;
    }));
    scenario = options[0];
    renderNodes();
    selectNode(scenario.route[0]);
    renderStage(false);
    requestAnimationFrame(drawWires);
  }

  function renderNodes() {
    $("nodes").replaceChildren(...view.nodes.map((id, i) => {
      const node = byId.get(id), button = element("button", "node");
      button.type = "button";
      button.dataset.id = id; button.dataset.kind = kind(node);
      button.style.gridRow = view.positions[i][0];
      button.style.gridColumn = view.positions[i][1];
      button.setAttribute("aria-pressed", "false");
      button.setAttribute("aria-controls", "detail");
      if (node.kind.toLowerCase().includes("optional")) button.classList.add("is-optional");
      button.append(element("span", "node-number", String(i + 1).padStart(2, "0")),
        element("span", "kind", node.kind), element("strong", "", node.title), element("small", "", node.summary));
      button.addEventListener("click", () => { selectNode(id); announce(`Selected ${node.title}. Explanation is below or alongside the map.`); });
      return button;
    }));
  }

  function selectNode(id) {
    selected = id;
    const node = byId.get(id);
    for (const button of $("nodes").children) button.setAttribute("aria-pressed", String(button.dataset.id === id));
    const title = element("h3", "", node.title); title.id = "detail-title";
    const pieces = [element("span", "detail-kind", node.kind), title, element("p", "summary", node.summary)];
    for (const [heading, key] of [["How it works","how"],["Why this choice","why"],["Alternative & tradeoff","alternative"],["Try it locally","exercise"]]) {
      pieces.push(element("h4", "", heading), element("p", "", node[key]));
    }
    const links = element("div", "source-links");
    for (const file of node.files) {
      const a = element("a", "", `${file} ↗`);
      a.href = `https://github.com/Wd3esa/muhawir/blob/${content.revision}/${file}`;
      links.append(a);
    }
    pieces.push(element("h4", "", "Read the implementation"), links);
    const connections = element("div", "catalog-list");
    for (const connection of content.connections.filter(c => c.from === id || c.to === id)) {
      const other = connection.from === id ? connection.to : connection.from;
      const button = element("button", "", `${connection.from === id ? "→" : "←"} ${byId.get(other).title}`);
      button.type = "button"; button.title = connection.label;
      button.addEventListener("click", () => {
        if (!view.nodes.includes(other)) chooseView(content.views.find(v => v.nodes.includes(other)).id);
        selectNode(other);
        announce(`Connected component: ${byId.get(other).title}. ${connection.label}.`);
      });
      connections.append(button);
    }
    pieces.push(element("h4", "", "Connected components · follow across maps"), connections);
    $("detail").replaceChildren(...pieces);
    $("detail").scrollTop = 0;
  }

  function stageText(id) {
    const node = byId.get(id);
    if (view.id !== "flow") return node.summary;
    if (id === "cache") return scenario.id === "cached" ? "A valid stand-alone reply matches this version. Jump to display; no new model calls." : "This simulation has no usable cached reply. Continue to the fixed rules.";
    if (id === "rules" && scenario.id === "crisis") return "A fixed crisis response returns human-help guidance. There is no search or language-model call.";
    if (id === "retrieve" && scenario.id === "missing") return index > 0 && scenario.route[index - 1] === id ? "A bounded query-expansion retry still finds insufficient usable evidence. Abstain." : "No usable passages. The real pipeline can try different model-generated search phrases within its retry bounds.";
    if (id === "draft" && scenario.id === "failure") return scenario.route[index - 1] === id ? "Try a configured fallback provider within the retry limits. This tour assumes a draft is recovered so we can show the reader failure." : "A provider fails. A fallback may recover a draft; it is not guaranteed and may spend credit.";
    if (id === "review" && scenario.id === "failure") return "The required support reader remains unavailable after its permitted retries. Do not present the draft as a checked answer.";
    if (id === "answer") return scenario.outcome;
    return node.summary;
  }

  function announce(message) { $("announcement").textContent = message; }

  function renderStage(announceStage = true) {
    const id = scenario.route[index];
    const seen = new Set(scenario.route.slice(0, index));
    for (const node of $("nodes").children) {
      node.classList.toggle("active", node.dataset.id === id);
      node.classList.toggle("visited", seen.has(node.dataset.id));
    }
    $("step-number").textContent = `${index + 1}/${scenario.route.length}`;
    $("step-title").textContent = byId.get(id).title;
    $("step-description").textContent = stageText(id);
    $("outcome").textContent = `Scenario outcome: ${scenario.outcome}`;
    $("step").disabled = index === scenario.route.length - 1;
    if (announceStage) announce(`Step ${index + 1}: ${byId.get(id).title}. ${stageText(id)}`);
    drawWires();
  }

  function advance() {
    if (index < scenario.route.length - 1) {
      index++;
      selectNode(scenario.route[index]);
      renderStage();
    }
  }

  function schedule() {
    if (!playing) return;
    startTime = performance.now();
    drawWires();
    if (!reducedMotion.matches && packetPath) frame = requestAnimationFrame(movePacket);
    timer = setTimeout(() => {
      advance();
      if (index === scenario.route.length - 1) { stop(); drawWires(); announce(`Simulation complete. ${scenario.outcome}`); }
      else schedule();
    }, 2100);
  }

  function movePacket(time) {
    if (!playing || !packetPath || reducedMotion.matches) return;
    const point = packetPath.getPointAtLength(Math.min((time - startTime) / 2000, 1) * packetPath.getTotalLength());
    const packet = $("wires").querySelector(".packet");
    if (packet) { packet.setAttribute("cx", point.x); packet.setAttribute("cy", point.y); }
    frame = requestAnimationFrame(movePacket);
  }

  function drawWires() {
    if (!view) return;
    const svg = $("wires"), box = $("diagram").getBoundingClientRect();
    const rects = new Map([...$("nodes").children].map(node => {
      const r = node.getBoundingClientRect();
      return [node.dataset.id, {x:r.left - box.left,y:r.top - box.top,w:r.width,h:r.height}];
    }));
    svg.setAttribute("viewBox", `0 0 ${box.width} ${box.height}`);
    const defs = svgElement("defs", {}), marker = svgElement("marker", {id:"arrow", viewBox:"0 0 10 10", refX:9, refY:5, markerWidth:5, markerHeight:5, orient:"auto-start-reverse"});
    marker.append(svgElement("path", {d:"M 0 0 L 10 5 L 0 10 z",fill:"#8da9a0"})); defs.append(marker);
    svg.replaceChildren(defs);
    packetPath = null;
    const from = scenario.route[index], to = scenario.route[index + 1];
    const pairs = [...view.edges];
    if (view.id === "flow") for (let i = 0; i < scenario.route.length - 1; i++) {
      const a = scenario.route[i], b = scenario.route[i + 1];
      if (a !== b && !pairs.some(([x,y]) => x === a && y === b)) pairs.push([a,b]);
    }
    for (const [a,b] of pairs) {
      const r = rects.get(a), s = rects.get(b);
      if (!r || !s) continue;
      const highlight = playing && a === from && b === to;
      const shortcut = view.id === "flow" && !view.edges.some(([x,y]) => a === x && b === y);
      let d;
      if (Math.abs(r.y - s.y) < 4 && !shortcut) {
        const direction = s.x > r.x ? 1 : -1;
        const x1 = r.x + (direction > 0 ? r.w : 0), x2 = s.x + (direction > 0 ? 0 : s.w);
        const y1 = r.y + r.h / 2, y2 = s.y + s.h / 2;
        d = `M${x1},${y1} C${(x1+x2)/2},${y1} ${(x1+x2)/2},${y2} ${x2},${y2}`;
      } else if (shortcut) {
        // A shortcut travels outside the card grid instead of crossing evidence cards.
        const side = box.width - 2;
        d = `M${r.x+r.w},${r.y+r.h/2} L${side},${r.y+r.h/2} L${side},${s.y+s.h/2} L${s.x+s.w},${s.y+s.h/2}`;
      } else {
        const down = s.y > r.y;
        const x1 = r.x + r.w / 2, x2 = s.x + s.w / 2;
        const y1 = down ? r.y + r.h : r.y, y2 = down ? s.y : s.y + s.h;
        if (Math.abs(x1 - x2) < 4) d = `M${x1},${y1} L${x2},${y2}`;
        else { const mid = down ? y1 + 16 : y1 - 16; d = `M${x1},${y1} L${x1},${mid} L${x2},${mid} L${x2},${y2}`; }
      }
      const optional = shortcut || [a,b].some(id => byId.get(id).kind.toLowerCase().includes("optional"));
      const path = svgElement("path", {d, class:`wire${optional ? " optional" : ""}${highlight ? " highlight" : ""}`, "marker-end":"url(#arrow)"});
      svg.append(path);
      if (highlight) packetPath = path;
    }
    if (playing && packetPath && !reducedMotion.matches) svg.append(svgElement("circle", {class:"packet",r:5,cx:0,cy:0}));
  }

  function renderCatalog(query = "") {
    const matches = content.nodes.filter(node => Object.values(node).join(" ").toLowerCase().includes(query.trim().toLowerCase()));
    $("catalog-list").replaceChildren(...matches.map(node => {
      const button = element("button", "", node.title);
      button.type = "button";
      button.addEventListener("click", () => {
        chooseView(content.views.find(v => v.nodes.includes(node.id)).id);
        selectNode(node.id);
        const target = $("nodes").querySelector(`[data-id="${node.id}"]`);
        target.focus({preventScroll:true}); target.scrollIntoView({block:"center",behavior:reducedMotion.matches ? "instant" : "smooth"});
        announce(`Selected ${node.title}. Read its explanation alongside or below the map.`);
      });
      return button;
    }));
    if (!matches.length) $("catalog-list").append(element("p", "", "No matching component. Try a broader term."));
  }

  function renderRanks() {
    const lexical = ["A","B","C"], semantic = $("semantic").checked ? ["C","B","D"] : [];
    $("rank-results").replaceChildren(...fuseRanks(lexical, semantic).map(({id,score}) => {
      const tr = element("tr");
      const rank = list => list.includes(id) ? String(list.indexOf(id)+1) : "—";
      for (const value of [id,rank(lexical),rank(semantic),score.toFixed(5)]) tr.append(element("td", "", value));
      return tr;
    }));
  }

  function renderLessons() {
    $("lessons").replaceChildren(...content.lessons.map((lesson, i) => {
      const details = element("details", "lesson"); if (i === 0) details.open = true;
      details.append(element("summary", "", lesson.title));
      const body = element("div", "lesson-body");
      body.append(element("p", "goal", lesson.goal), element("p", "", lesson.explain), element("h4", "", "Why this step"), element("p", "", lesson.why), element("h4", "", "Hands-on task"), element("p", "", lesson.task));
      if (lesson.code) {
        const wrap = element("div", "code-wrap"), pre = element("pre"), code = element("code", "", lesson.code), copy = element("button", "copy", "Copy");
        copy.type = "button"; copy.setAttribute("aria-label", `Copy code for ${lesson.title}`);
        copy.addEventListener("click", async () => {
          try { await navigator.clipboard.writeText(lesson.code); copy.textContent = "Copied"; }
          catch { copy.textContent = "Select code to copy"; }
          setTimeout(() => { copy.textContent = "Copy"; }, 2400);
        });
        pre.append(code); wrap.append(pre, copy); body.append(wrap);
      }
      body.append(element("h4", "", "Explain your result"), element("p", "check", lesson.check));
      details.append(body); return details;
    }));
  }

  async function init() {
    try {
      const response = await fetch("/learning/content.json", {credentials:"omit"});
      if (!response.ok) throw new Error("Lesson content unavailable");
      content = await response.json();
      for (const node of content.nodes) byId.set(node.id, node);
      $("tabs").replaceChildren(...content.views.map(v => {
        const tab = element("button", "", v.title);
        tab.type = "button"; tab.id = `tab-${v.id}`; tab.dataset.view = v.id;
        tab.setAttribute("role", "tab"); tab.setAttribute("aria-controls", "map-panel");
        tab.addEventListener("click", () => chooseView(v.id));
        tab.addEventListener("keydown", event => {
          if (!["ArrowLeft","ArrowRight","Home","End"].includes(event.key)) return;
          event.preventDefault();
          const at = content.views.findIndex(item => item.id === view.id);
          const next = event.key === "Home" ? 0 : event.key === "End" ? content.views.length - 1 : (at + (event.key === "ArrowRight" ? 1 : -1) + content.views.length) % content.views.length;
          chooseView(content.views[next].id, true);
        });
        return tab;
      }));
      $("scenario").addEventListener("change", () => { stop(); scenario = scenariosFor(view).find(s => s.id === $("scenario").value); index = 0; selectNode(scenario.route[0]); renderStage(); });
      $("play").addEventListener("click", () => {
        if (playing) { stop(); drawWires(); return; }
        if (index === scenario.route.length - 1) { index = 0; selectNode(scenario.route[0]); }
        playing = true; $("play").textContent = "Pause"; renderStage(); schedule();
      });
      $("step").addEventListener("click", () => { stop(); advance(); });
      $("restart").addEventListener("click", () => { stop(); index = 0; selectNode(scenario.route[0]); renderStage(); });
      $("search").addEventListener("input", event => renderCatalog(event.target.value));
      $("semantic").addEventListener("change", renderRanks);
      $("component-count").textContent = `${content.nodes.length} components`;
      document.addEventListener("visibilitychange", () => { if (document.hidden) { stop(); drawWires(); } });
      window.addEventListener("pagehide", stop);
      reducedMotion.addEventListener("change", () => { stop(); drawWires(); });
      new ResizeObserver(() => requestAnimationFrame(drawWires)).observe($("diagram"));
      document.fonts.ready.then(drawWires);
      renderCatalog(); renderRanks(); renderLessons(); chooseView("flow");
    } catch (error) {
      $("map-panel").replaceChildren(element("p", "notice", "The interactive lesson could not load. Reload the page, or read the architecture on GitHub using the link below."));
      console.error(error);
    }
  }
  return {init, fuseRanks};
})();
Learning.init();
