// Passive inspector for recorded solver observations. It never advances a brain,
// changes a weight or supplies a controller command. Lines connect actual CSR
// endpoints; their brightness encodes measured contribution, never a travelling
// pulse. Playback time is a viewing aid, not a biological or solver timescale.
const finite = x => typeof x === "number" && Number.isFinite(x);
const fmt = x => !finite(x) ? "—" : x === 0 ? "0" : Math.abs(x) < .001 || Math.abs(x) >= 10000 ? x.toExponential(3) : x.toFixed(4);
const list = x => Array.isArray(x) || ArrayBuffer.isView(x) ? x : [];
const at = (array, index) => list(array)[index];
const clamp = (x, lo, hi) => Math.max(lo, Math.min(hi, x));
const labels = p => Array.isArray(p?.labels) ? p.labels.join(" · ") : String(p?.labels || "unlabelled cell");
const short = (text, n = 26) => String(text).length > n ? String(text).slice(0, n - 1) + "…" : String(text);

function validTrace(t) {
  return t?.schema === "cadence-settlement-trace/v1" && Number.isInteger(t.n) && t.n > 0
    && Array.isArray(t.patches) && t.patches.length > 0 && t.patches.length <= 64
    && Array.isArray(t.edges) && t.edges.length <= 256 && Array.isArray(t.frames) && t.frames.length > 0
    && t.frames.every((f, k) => Number.isInteger(f.iteration) && f.iteration >= 0 && (!k || f.iteration > t.frames[k - 1].iteration)
      && ["potential", "activity", "drive", "bias", "synapticInput", "defect"].every(key => list(f[key]).length === t.patches.length));
}

export function createSettlementView(container, { onContrast } = {}) {
  if (!container?.ownerDocument) throw new TypeError("settlement view requires a DOM container");
  const doc = container.ownerDocument;
  container.classList.add("settlement-view");
  container.innerHTML = `<style>
.settlement-view{color:#dff5ec;font:13px/1.45 system-ui,sans-serif;min-width:0;background:#06100e;padding:12px;border:1px solid #214e3b;border-radius:10px;box-sizing:border-box;container:settlement-inspector / inline-size}
.settlement-view *{box-sizing:border-box}.settlement-view h3{font-size:18px;line-height:1.2;margin:0 0 7px;color:#f2fff9}.settlement-view h4{font-size:13px;margin:13px 0 6px;color:#c1e9d6;font-weight:650}
.settlement-view p{margin:5px 0}.settlement-view .sv-muted{color:#9bb5aa;font-size:12px}.settlement-view .sv-source{overflow-wrap:anywhere;font:11px/1.45 ui-monospace,monospace;color:#9fbdb0}
.settlement-view .sv-status{border-left:3px solid #b5d7c4;padding:5px 8px;background:#10211b;margin:9px 0}.settlement-view .sv-controls{display:flex;align-items:center;gap:6px;flex-wrap:wrap;margin:9px 0}
.settlement-view button,.settlement-view select{font:12px system-ui,sans-serif;border:1px solid #496755;border-radius:5px;background:#10231b;color:#e5fff1;padding:6px 9px;cursor:pointer;max-width:100%}
.settlement-view button:disabled{opacity:.4;cursor:default}.settlement-view button:focus-visible,.settlement-view select:focus-visible{outline:2px solid #ffd166;outline-offset:2px}
.settlement-view button[aria-pressed=true]{border-color:#ffd166;color:#ffe39d}.settlement-view input[type=range]{width:100%;accent-color:#8ce5b5;margin:8px 0}.settlement-view canvas{width:100%;height:auto;max-width:640px;display:block;margin-inline:auto;border:1px solid #234332;border-radius:5px;background:#020a07}
.settlement-view .sv-columns{display:grid;grid-template-columns:minmax(0,1fr);gap:14px;align-items:start}.settlement-view .sv-column{min-width:0}.settlement-view .sv-column>select{width:100%;margin-bottom:6px}
.settlement-view .sv-metrics{display:grid;grid-template-columns:1fr 1fr;gap:6px;margin:8px 0}.settlement-view .sv-metric{background:#0c1b15;padding:7px;border-radius:5px}.settlement-view .sv-metric b{display:block;font:16px/1.3 ui-monospace,monospace;color:#f0fff7}.settlement-view .sv-metric small{font-size:11px;color:#a5c0b2}
.settlement-view .sv-scroll{max-height:200px;overflow:auto}.settlement-view table{width:100%;border-collapse:collapse;font:11px/1.5 ui-monospace,monospace}.settlement-view th,.settlement-view td{padding:4px 5px;border-bottom:1px solid #183b2a;text-align:right;overflow-wrap:anywhere}.settlement-view th:first-child,.settlement-view td:first-child{text-align:left}.settlement-view th{color:#a4c6b5;font-weight:400}
.settlement-view .sv-input{display:grid;grid-template-columns:130px 1fr;gap:10px;align-items:start}.settlement-view .sv-input canvas{height:70px;image-rendering:pixelated}.settlement-view .sv-legend{font-size:11px;color:#a9c7b9;margin-top:5px}.settlement-view .sv-warning{color:#ffd79a}.settlement-view .sv-empty{padding:15px 4px;color:#a9c7b9}.settlement-view .sv-equation{font:11px/1.6 ui-monospace,monospace;color:#bbd7c8;padding:6px;background:#0c1b15;overflow-wrap:anywhere}.settlement-view details{margin:6px 0}.settlement-view summary{cursor:pointer;color:#bad9c8;font-size:12px}.settlement-view .sv-local{font:12px/1.6 ui-monospace,monospace;background:#0c1b15;padding:5px 7px}
@media(max-width:600px){.settlement-view{padding:9px;font-size:12px}.settlement-view .sv-input{grid-template-columns:110px 1fr}}
@container settlement-inspector (min-width:900px){.settlement-view .sv-columns{grid-template-columns:minmax(0,1fr) minmax(0,1fr);gap:20px}.settlement-view .sv-column+.sv-column{border-left:1px solid #234332;padding-left:20px}.settlement-view .sv-input{grid-template-columns:180px minmax(0,1fr);align-items:center}.settlement-view .sv-input canvas{height:108px}.settlement-view .sv-column>.sv-controls:first-child{margin-top:0}.settlement-view .sv-scroll{max-height:180px}}
</style>
<h3>Recorded settlement replay</h3>
<div class="sv-status" data-sv="status" role="status">Waiting for a captured settlement.</div>
<div class="sv-columns"><section class="sv-column" aria-label="Captured input and whole-graph settlement">
<div class="sv-input"><canvas data-sv="retina" width="150" height="90" aria-label="Captured retinal input"></canvas><p class="sv-muted" data-sv="retinainfo">No captured pixel frame is attached.</p></div>
<p class="sv-muted" data-sv="contrastbrief">Pixel effect not isolated yet.</p>
<details><summary>Capture identity and scope</summary><div class="sv-source" data-sv="source">No observation attached.</div><p class="sv-muted">Recorded solver states from one frozen input. Replay is slowed for inspection and has no control authority.</p></details>
<div class="sv-controls"><button data-sv="previous" aria-label="Previous recorded state">‹</button><button data-sv="play">Play replay</button><button data-sv="next" aria-label="Next recorded state">›</button><button data-sv="follow">Follow latest</button><button data-sv="contrast">Pause + compare pixels</button><span class="sv-muted" data-sv="queue"></span></div>
<input data-sv="scrub" type="range" min="0" max="0" value="0" aria-label="Recorded solver state">
<p class="sv-muted" data-sv="frame">No recorded iterations.</p>
<div class="sv-metrics"><div class="sv-metric"><b data-sv="global">—</b><small>Global max defect · all cells</small></div><div class="sv-metric"><b data-sv="local">—</b><small>Max defect · captured cells</small></div></div>
<canvas data-sv="curve" width="520" height="90" aria-label="Recorded global and captured-cell residuals"></canvas>
<p class="sv-legend">Gold: global · green: captured cells · dashed: tolerance. Non-monotone curves are possible.</p>
<div data-sv="comparison" hidden><p data-sv="contraststatus"></p><p class="sv-muted" data-sv="downstream"></p><div class="sv-controls"><button data-sv="actual">Replay actual pixels</button><button data-sv="black">Replay black pixels</button></div><details><summary>Paired readout differences and scope</summary><div class="sv-scroll"><table><thead><tr><th>Readout</th><th>Actual − black</th></tr></thead><tbody data-sv="deltas"></tbody></table></div><p class="sv-muted">Largest absolute readout differences shown, including zero/tiny values. Measured paired response, not proof of biological vision or competent navigation. A visual flash alone is not a causal test.</p></details></div>
</section><section class="sv-column" aria-label="Local patches, overlaps and measured repairs">
<div class="sv-controls"><strong>Local patches and overlaps</strong><button data-sv="difference" disabled>Final pixel difference</button></div>
<select data-sv="patch" aria-label="Inspect a captured neuron patch"></select>
<canvas data-sv="graph" width="520" height="210" aria-label="Actual incoming connections and shared presynaptic cells"></canvas>
<p class="sv-legend" data-sv="graphlegend">Fill: activity · orange ring: defect · violet: shared input. Click a receiving cell.</p>
<div class="sv-local" data-sv="localbrief">No captured local equation.</div>
<p class="sv-muted" data-sv="outputbrief">No selected output activities.</p>
<details><summary>Equation, exact values and omitted inputs</summary><p class="sv-muted" data-sv="graphnote">An excerpt of the full graph, not a replacement network.</p>
<div class="sv-equation" data-sv="equation">rᵢ = Σⱼ wᵢⱼ sⱼ + driveᵢ + biasᵢ − vᵢ</div>
<div class="sv-scroll"><table><tbody data-sv="patchvalues"></tbody></table></div>
<p class="sv-muted">“Repair” means an Euler state update, not a weight update or a proof of confluence/unique equilibrium. Shared inputs are single indexed neural states, not duplicate consensus variables.</p></details>
<details><summary>Selected output cells · replay state</summary><div class="sv-scroll"><table><thead><tr><th>Cell / annotation</th><th>Activity</th><th>Defect</th></tr></thead><tbody data-sv="outputs"></tbody></table></div></details>
</section></div>`;
  const $ = name => container.querySelector(`[data-sv="${name}"]`);
  const ctx = name => $(name).getContext("2d");
  let current = null, pending = null, index = 0, patchIndex = 0, playing = false, held = false, visible = true;
  let lastTick = null, dirty = true, generation = null, externalStatus = "", diagnostic = null, hits = [], dead = false, differenceMode = false;

  function identity(entry = current) { return { ...entry?.trace?.identity, ...entry?.metadata?.identity }; }
  function commit(entry) {
    const previousId = current?.trace.patches[patchIndex]?.id;
    current = entry; index = 0; playing = true; differenceMode = false; lastTick = null; dirty = true;
    const same = entry.trace.patches.findIndex(p => p.id === previousId);
    const receptorIds = new Set(entry.trace.patches.filter(p => /photoreceptor/i.test(labels(p))).map(p => p.id));
    const recipient = entry.trace.patches.findIndex(p => !receptorIds.has(p.id) && entry.trace.edges.some(e => e.post === p.id && receptorIds.has(e.pre)));
    patchIndex = same >= 0 ? same : recipient >= 0 ? recipient : 0;
    $("patch").replaceChildren();
    for (let k = 0; k < entry.trace.patches.length; k++) {
      const p = entry.trace.patches[k], option = doc.createElement("option");
      option.value = String(k); option.textContent = `#${p.id} · ${labels(p)}`; $("patch").append(option);
    }
    $("patch").value = String(patchIndex);
    $("scrub").max = String(entry.trace.frames.length - 1);
  }
  function setTrace(trace, metadata = {}) {
    if (!validTrace(trace)) { setStatus("Capture rejected by viewer: incomplete or invalid trace."); return false; }
    const gen = metadata.identity?.generation ?? trace.identity?.generation;
    if (generation !== null && Number.isInteger(gen) && gen < generation) return false;
    if (Number.isInteger(gen) && gen !== generation) reset(gen);
    const entry = structuredClone({ trace, metadata: { retinal: trace.retinal, ...metadata } });
    externalStatus = "";
    if (!current || (!held && !playing)) commit(entry); else pending = entry;
    dirty = true; return true;
  }
  function setStatus(text) { externalStatus = typeof text === "string" ? text : String(text?.text || ""); dirty = true; }
  function frame() { return current?.trace.frames[index]; }
  function pick(i) { if (!current) return; differenceMode = false; index = clamp(i, 0, current.trace.frames.length - 1); held = true; playing = false; lastTick = null; dirty = true; }
  $("previous").onclick = () => pick(index - 1);
  $("next").onclick = () => pick(index + 1);
  $("scrub").oninput = () => pick(Number($("scrub").value));
  $("play").onclick = () => {
    if (!current) return;
    differenceMode = false; held = true; playing = !playing; if (playing && index === current.trace.frames.length - 1) index = 0;
    lastTick = null; dirty = true;
  };
  $("follow").onclick = () => { held = false; if (pending) { const entry = pending; pending = null; commit(entry); } dirty = true; };
  $("patch").onchange = () => { patchIndex = Number($("patch").value); dirty = true; };
  $("contrast").disabled = typeof onContrast !== "function";
  $("contrast").onclick = () => { if (onContrast) { setStatus("Pixel comparison requested; waiting for the paused diagnostic."); onContrast(); } };
  $("graph").onclick = event => {
    const rect = $("graph").getBoundingClientRect(), x = (event.clientX - rect.left) * 520 / rect.width, y = (event.clientY - rect.top) * 210 / rect.height;
    const hit = hits.find(h => Math.hypot(h.x - x, h.y - y) <= h.radius);
    if (hit && hit.index !== undefined) { patchIndex = hit.index; $("patch").value = String(patchIndex); dirty = true; }
  };
  $("curve").onclick = event => {
    if (!current) return;
    const rect = $("curve").getBoundingClientRect(), proportion = clamp(((event.clientX - rect.left) * 520 / rect.width - 52) / 452, 0, 1);
    const target = proportion * current.trace.frames.at(-1).iteration;
    let best = 0;
    for (let k = 1; k < current.trace.frames.length; k++) if (Math.abs(current.trace.frames[k].iteration - target) < Math.abs(current.trace.frames[best].iteration - target)) best = k;
    pick(best);
  };

  function rows(node, values) {
    node.replaceChildren();
    for (const row of values) {
      const tr = doc.createElement("tr");
      for (const value of row) { const td = doc.createElement("td"); td.textContent = String(value); tr.append(td); }
      node.append(tr);
    }
  }
  function drawCurve() {
    const g = ctx("curve"), W = 520, H = 90;
    g.clearRect(0, 0, W, H); if (!current) return;
    const t = current.trace, fs = t.frames, values = fs.flatMap(f => [f.residual, f.localMaxDefect]).filter(x => finite(x) && x > 0);
    if (finite(t.tolerance) && t.tolerance > 0) values.push(t.tolerance);
    const lo = Math.floor(Math.log10(Math.min(...values, 1e-6))) - 1, hi = Math.max(lo + 1, Math.ceil(Math.log10(Math.max(...values, 1e-6))));
    const x = i => 52 + 452 * i / Math.max(1, fs.at(-1).iteration), y = r => 10 + 56 * (hi - Math.log10(Math.max(10 ** lo, r))) / (hi - lo);
    g.font = "12px monospace"; g.lineWidth = 1;
    for (let k = 0; k <= 3; k++) {
      const power = hi + (lo - hi) * k / 3, yy = 10 + 56 * k / 3;
      g.strokeStyle = "#193a2b"; g.beginPath(); g.moveTo(52, yy); g.lineTo(504, yy); g.stroke();
      g.fillStyle = "#99b6a7"; g.fillText(`1e${power.toFixed(0)}`, 4, yy + 4);
    }
    if (finite(t.tolerance) && t.tolerance > 0) {
      g.strokeStyle = "#738c80"; g.setLineDash([4, 4]); g.beginPath(); g.moveTo(52, y(t.tolerance)); g.lineTo(504, y(t.tolerance)); g.stroke(); g.setLineDash([]);
    }
    for (const [key, color] of [["residual", "#ffd166"], ["localMaxDefect", "#7ce3a7"]]) {
      g.strokeStyle = color; g.lineWidth = 2; g.beginPath(); let started = false;
      for (const f of fs) if (finite(f[key]) && f[key] >= 0) { if (started) g.lineTo(x(f.iteration), y(f[key])); else g.moveTo(x(f.iteration), y(f[key])); started = true; } else started = false;
      g.stroke();
    }
    const f = frame(); g.strokeStyle = "#dff9ed"; g.lineWidth = 1; g.beginPath(); g.moveTo(x(f.iteration), 8); g.lineTo(x(f.iteration), 70); g.stroke();
    g.fillStyle = "#adcaba"; g.fillText("iteration 0", 52, 84); g.textAlign = "right"; g.fillText(String(fs.at(-1).iteration), 504, 84); g.textAlign = "left";
  }

  function neighborhood() {
    const t = current.trace, p = t.patches[patchIndex];
    const incoming = t.edges.map((e, k) => ({ ...e, slot: k })).filter(e => e.post === p.id);
    // A stable, explicitly bounded excerpt; frame activity never rearranges ports.
    const ports = [...new Set(incoming.map(e => e.pre))].slice(0, 8);
    const others = [...new Set(t.edges.filter(e => ports.includes(e.pre) && e.post !== p.id).map(e => e.post))]
      .filter(id => t.patches.some(patch => patch.id === id)).slice(0, 5);
    const shown = t.edges.map((e, k) => ({ ...e, slot: k })).filter(e => ports.includes(e.pre) && (e.post === p.id || others.includes(e.post)));
    return { p, incoming, ports, others, shown };
  }

  function drawGraph() {
    const g = ctx("graph"); g.clearRect(0, 0, 520, 210); hits = []; if (!current) return;
    const t = current.trace, f = frame(), { p, incoming, ports, others, shown } = neighborhood();
    const positions = new Map(), portPositions = new Map();
    positions.set(p.id, { x: 283, y: 100, radius: 24, index: patchIndex });
    ports.forEach((id, k) => portPositions.set(id, { x: 69, y: 33 + (ports.length === 1 ? 67 : k * 142 / Math.max(1, ports.length - 1)), radius: 8 }));
    others.forEach((id, k) => positions.set(id, { x: 444, y: 37 + (others.length === 1 ? 63 : k * 133 / Math.max(1, others.length - 1)), radius: 12, index: t.patches.findIndex(patch => patch.id === id) }));
    g.font = "12px system-ui"; g.fillStyle = "#a1bfae"; g.fillText("Input cells", 30, 15); g.fillText("Receiving patch", 233, 15); g.fillText("Also read by", 410, 15);
    const edgeValue = e => differenceMode ? pixelDifference("contribution", e.id, true) : at(f.contribution, e.slot);
    const maximum = Math.max(1e-30, ...shown.map(e => Math.abs(edgeValue(e) || 0)));
    for (const e of shown) {
      const a = portPositions.get(e.pre), b = positions.get(e.post); if (!a || !b) continue;
      const strength = Math.abs(edgeValue(e) || 0) / maximum;
      g.globalAlpha = .25 + .65 * strength; g.strokeStyle = e.weight < 0 ? "#80baff" : "#81de9d"; g.lineWidth = .7 + 2 * strength;
      g.beginPath(); g.moveTo(a.x + a.radius, a.y); g.lineTo(b.x - b.radius, b.y); g.stroke();
      const dx = b.x - a.x, dy = b.y - a.y, angle = Math.atan2(dy, dx), bx = b.x - Math.cos(angle) * (b.radius + 4), by = b.y - Math.sin(angle) * (b.radius + 4);
      g.beginPath(); g.moveTo(bx, by); g.lineTo(bx - 6 * Math.cos(angle - .4), by - 6 * Math.sin(angle - .4)); g.moveTo(bx, by); g.lineTo(bx - 6 * Math.cos(angle + .4), by - 6 * Math.sin(angle + .4)); g.stroke();
      g.globalAlpha = 1;
      if (e.post === p.id && incoming.length <= 4) { g.fillStyle = "#b4cfbd"; g.font = "12px monospace"; g.fillText(`w ${fmt(e.weight)}`, a.x + 25, (a.y + b.y) / 2 - 3); }
    }
    const node = (id, pos, activity, defect, shared = false) => {
      const a = finite(activity) ? activity : 0;
      const maxDelta = Math.max(1e-30, ...t.patches.map(patch => Math.abs(pixelDifference("activity", patch.id) || 0)));
      const strength = differenceMode ? Math.log1p(1000 * Math.abs(a) / maxDelta) / Math.log(1001) : Math.min(1, Math.abs(a));
      g.fillStyle = a < 0 ? `rgba(74,133,220,${.2 + .8 * strength})` : differenceMode ? `rgba(255,178,92,${.12 + .88 * strength})` : `rgba(110,240,155,${.12 + .88 * strength})`;
      if (!finite(activity)) g.fillStyle = "#39433e";
      g.strokeStyle = shared ? "#c39dff" : "#89b8a0"; g.lineWidth = shared ? 2.5 : 1.2;
      g.beginPath(); g.arc(pos.x, pos.y, pos.radius, 0, 2 * Math.PI); g.fill(); g.stroke();
      if (!differenceMode && finite(defect) && Math.abs(defect) > 0) {
        const ring = clamp(Math.log10(1 + Math.abs(defect) / Math.max(t.tolerance, 1e-12)), 0, 6);
        g.strokeStyle = "#ffb36b"; g.lineWidth = 1.5; g.beginPath(); g.arc(pos.x, pos.y, pos.radius + 3 + ring, 0, 2 * Math.PI); g.stroke();
      }
      g.fillStyle = "#e1f7eb"; g.font = "12px monospace"; g.textAlign = "center";
      if (pos.x < 100) { g.textAlign = "right"; g.fillText(String(id), pos.x - 12, pos.y + 4); g.textAlign = "left"; g.fillText(fmt(activity), pos.x + 13, pos.y + 4); }
      else { g.fillText(`#${id}`, pos.x, pos.y + pos.radius + 14); g.fillStyle = "#adcaba"; g.fillText(fmt(activity), pos.x, pos.y - pos.radius - 6); }
      g.textAlign = "left";
      hits.push({ ...pos, id });
    };
    for (const [id, pos] of portPositions) {
      const e = shown.find(e => e.pre === id), ownIndex = t.patches.findIndex(patch => patch.id === id);
      const shared = (t.sharedInputs || []).some(port => port.id === id && port.posts?.length > 1);
      node(id, { ...pos, index: ownIndex >= 0 ? ownIndex : undefined }, differenceMode && e ? pixelDifference("edgeActivity", e.id, true) : e ? at(f.edgeActivity, e.slot) : null, null, shared);
    }
    for (const [id, pos] of positions) node(id, pos, differenceMode ? pixelDifference("activity", id) : at(f.activity, pos.index), at(f.defect, pos.index));
    g.fillStyle = "#e2f4e8"; g.font = "12px system-ui"; g.textAlign = "center"; g.fillText(short(labels(p), 37), 284, 162);
    g.fillStyle = "#d8c37b"; g.fillText(`external drive ${fmt(at(f.drive, patchIndex))}`, 284, 185); g.textAlign = "left";
    $("graphlegend").textContent = differenceMode ? "Final actual − black activity: orange +, blue −. Log brightness relative to captured max |Δ|; labels are exact measured numbers." : "Fill: activity · orange ring: defect · violet: shared input. Edge brightness: measured |w × activity| (relative).";
    const drawnIncoming = shown.filter(e => e.post === p.id).length;
    $("graphnote").textContent = `Cell #${p.id}: ${drawnIncoming} shown / ${p.sampledIncomingCount ?? incoming.length} captured / ${p.incomingCount ?? "unknown"} total incoming edge classes. Full incoming sum below includes omitted ports. Violet ports are the same indexed cells read by multiple patches, not separate copies.`;
  }

  function retinalPacket(metadata) {
    const r = metadata?.retinal ?? metadata?.retina;
    return r?.frame ?? (r?.rgba ? r : null);
  }
  function receptorChain() {
    if (!current) return null;
    const meta = current.metadata.retinal ?? current.metadata.retina, ids = list(meta?.indices), levels = list(meta?.levels);
    const t = current.trace, f = frame(), selected = t.patches[patchIndex].id;
    if (!ids.length || ids.length !== levels.length) return null;
    const incoming = t.edges.filter(e => e.post === selected && ids.includes(e.pre));
    const id = ids.includes(selected) ? selected : incoming.find(e => t.patches.some(p => p.id === e.pre))?.pre ?? incoming[0]?.pre;
    if (id === undefined) return null;
    const k = ids.indexOf(id), pi = t.patches.findIndex(p => p.id === id), ei = t.edges.findIndex(e => e.pre === id && e.post === selected);
    const chain = { id, selectedPatchId: selected, level: levels[k], drive: pi >= 0 ? at(f.drive, pi) : null,
      activity: pi >= 0 ? at(f.activity, pi) : ei >= 0 ? at(f.edgeActivity, ei) : null, mapping: meta.mapping?.scheme ?? null };
    // Reconstruct only the documented supplied index-grid coordinates. These
    // are display sampling sites, never claims about anatomical receptive fields.
    const rows = meta.mapping?.rows;
    if (chain.mapping === "supplied-retained-index-grid/v1" && Number.isInteger(rows) && rows > 0) {
      const perRow = Math.floor(ids.length / rows), extra = ids.length % rows;
      let left = k;
      for (let row = 0; row < rows; row++) {
        const columns = perRow + (row < extra ? 1 : 0);
        if (left < columns) { chain.u = (left + .5) / columns; chain.v = (row + .5) / rows; break; }
        left -= columns;
      }
    }
    const packet = retinalPacket(current.metadata);
    if (packet && finite(chain.u) && finite(chain.v)) {
      chain.sourceX = chain.u * (packet.width - 1);
      chain.sourceY = (packet.origin === "bottom-left" ? 1 - chain.v : chain.v) * (packet.height - 1);
      const scale = Math.min(150 / packet.width, 90 / packet.height);
      const x = (150 - packet.width * scale) / 2 + (chain.sourceX + .5) * scale;
      const y = (90 - packet.height * scale) / 2 + (chain.v * (packet.height - 1) + .5) * scale;
      chain.previewHighlight = { x, y, bounds: { x0: x - 9, y0: y - 9, x1: x + 9, y1: y + 9 }, annotationOnly: true };
    }
    return chain;
  }
  function drawRetina() {
    const g = ctx("retina"), W = 150, H = 90; g.clearRect(0, 0, W, H);
    const packet = retinalPacket(current?.metadata);
    if (!packet || !Number.isInteger(packet.width) || !Number.isInteger(packet.height)
      || packet.width < 1 || packet.height < 1 || packet.width * packet.height > 65536 || list(packet.rgba).length !== packet.width * packet.height * 4) {
      g.fillStyle = "#9aafa3"; g.font = "11px system-ui"; g.fillText("No captured pixels", 17, 45);
      $("retinainfo").textContent = "No captured pixel frame is attached. This pane never substitutes a fresh camera image."; return;
    }
    const canvas = doc.createElement("canvas"); canvas.width = packet.width; canvas.height = packet.height;
    const c = canvas.getContext("2d"), image = c.createImageData(packet.width, packet.height);
    for (let y = 0; y < packet.height; y++) {
      const sourceY = packet.origin === "bottom-left" ? packet.height - 1 - y : y;
      for (let x = 0; x < packet.width * 4; x++) image.data[y * packet.width * 4 + x] = packet.rgba[sourceY * packet.width * 4 + x];
    }
    c.putImageData(image, 0, 0); g.imageSmoothingEnabled = false;
    const scale = Math.min(W / packet.width, H / packet.height), width = packet.width * scale, height = packet.height * scale;
    g.drawImage(canvas, (W - width) / 2, (H - height) / 2, width, height);
    const meta = current.metadata.retinal ?? current.metadata.retina, encoded = meta?.encoded ?? meta?.summary ?? current.metadata.retinalSummary;
    const count = encoded?.neurons ?? list(meta?.indices).length;
    const chain = receptorChain();
    $("retinainfo").textContent = `${packet.width} × ${packet.height} captured pixels${count ? ` → ${count} receptors` : ""}. `
      + (chain ? `Sample #${chain.id}: encoded luminance ${fmt(chain.level)} → held drive ${fmt(chain.drive)} → activity ${fmt(chain.activity)}. ` : "No captured receptor directly feeds this selected patch. ")
      + "Supplied index-grid, not anatomical retinotopy.";
    if (chain && finite(chain.u) && finite(chain.v)) {
      const x = (W - width) / 2 + (chain.u * (packet.width - 1) + .5) * scale;
      const y = (H - height) / 2 + (chain.v * (packet.height - 1) + .5) * scale;
      g.strokeStyle = "#ffd166"; g.lineWidth = 1.5; g.beginPath(); g.arc(x, y, 4, 0, 2 * Math.PI); g.stroke();
      g.beginPath(); g.moveTo(x - 7, y); g.lineTo(x + 7, y); g.moveTo(x, y - 7); g.lineTo(x, y + 7); g.stroke();
      $("retina").title = `Captured sampling site for receptor #${chain.id}; supplied grid (u=${chain.u.toFixed(5)}, v=${chain.v.toFixed(5)}), bilinear pixels. Yellow mark is a viewer annotation.`;
    } else $("retina").title = "Captured pixels; no sampled receptor coordinate is available.";
  }

  function render() {
    dirty = false;
    $("queue").textContent = pending ? "New capture queued" : held ? "Replay held" : "Following captures";
    $("play").textContent = playing ? "Pause replay" : "Play replay";
    $("play").setAttribute("aria-pressed", String(playing));
    for (const key of ["previous", "next", "play", "scrub", "patch"]) $(key).disabled = !current;
    if (!current) { $("status").textContent = externalStatus || "Waiting for a captured settlement."; return; }
    const t = current.trace, f = frame(), id = identity(), p = t.patches[patchIndex];
    const accepted = current.metadata.policyAccepted;
    const authority = accepted === true ? "Reply admitted by controller" : accepted === false ? "Reply not admitted by controller" : "No actuation admission asserted";
    $("status").textContent = `${externalStatus ? externalStatus + " · " : ""}${t.converged ? "Final residual qualified" : "Final solve unqualified"} · ${authority}`;
    $("source").textContent = `Generation ${id.generation ?? "?"} · request ${id.requestId ?? "?"}${id.epoch !== undefined ? ` · epoch ${id.epoch}` : ""}${finite(id.observationTime) ? ` · observed at ${fmt(id.observationTime)} simulated s` : ""} · ${id.source || "recorded source"} · ${t.phase || "free"} phase\n${t.patches.length} captured cells / ${t.n.toLocaleString()} total · ${t.edges.length} captured / ${(t.edgeCount ?? 0).toLocaleString()} edge classes\n${finite(f.settledCount) ? f.settledCount.toLocaleString() : "?"} cell equations within tolerance ${fmt(t.tolerance)} at this recorded iteration. Playback: 6.25 captured samples/s.`;
    $("scrub").value = String(index);
    $("frame").textContent = `Iteration ${f.iteration} · sample ${index + 1}/${t.frames.length} · ${t.patches.length}/${t.n.toLocaleString()} cells shown`;
    $("global").textContent = fmt(f.residual); $("local").textContent = fmt(f.localMaxDefect);
    const localDelta = pixelDifference("activity", p.id);
    $("localbrief").textContent = `#${p.id} · activity ${fmt(at(f.activity, patchIndex))} · input ${fmt(at(f.synapticInput, patchIndex))}\ndefect ${fmt(at(f.defect, patchIndex))} · actual Δv ${fmt(at(f.deltaV, patchIndex))}${diagnostic ? ` · final pixel Δ ${fmt(localDelta)}` : ""}`;
    $("localbrief").style.whiteSpace = "pre-line";
    const downstream = diagnostic?.nonreceptor;
    const pairId = diagnosticIdentity();
    $("contrastbrief").textContent = downstream
      ? `Recorded pixel pair ${pairId.generation ?? "?"}/${pairId.requestId ?? "?"}: max downstream numerical |Δs| ${fmt(downstream.maxAbsDelta)}. Selected cell's paired final Δs ${fmt(localDelta)}.${diagnostic.comparable !== true ? " Pair unqualified." : ""}`
      : "Pixel effect not isolated yet.";
    $("difference").setAttribute("aria-pressed", String(differenceMode));
    rows($("patchvalues"), [
      ["Potential vᵢ", fmt(at(f.potential, patchIndex))], ["Activity sᵢ", fmt(at(f.activity, patchIndex))],
      ["Full synaptic input Σ w s", fmt(at(f.synapticInput, patchIndex))], ["External drive", fmt(at(f.drive, patchIndex))],
      ["Bias", fmt(at(f.bias, patchIndex))], ["Current remaining defect rᵢ", fmt(at(f.defect, patchIndex))],
      ["Previous-step defect", fmt(at(f.previousDefect, patchIndex))], ["Actual last Euler change Δvᵢ", fmt(at(f.deltaV, patchIndex))],
    ]);
    $("equation").textContent = `rᵢ = Σⱼ wᵢⱼ sⱼ + driveᵢ + biasᵢ − vᵢ\nΔvᵢ = dt × previous-step defect; dt = ${fmt(t.dt)}. ${f.iteration === 0 ? "No previous update at initial capture." : "Δv is one actual solver step, not the gap between replay samples."}`;
    const outputs = t.patches.map((patch, i) => ({ patch, i })).filter(({ patch }) => /mbon|\bdn:|\bmn:|mn9|power:/i.test(labels(patch)));
    $("outputbrief").textContent = outputs.length ? `Selected outputs: ${outputs.slice(0, 2).map(({ patch, i }) => `#${patch.id} activity ${fmt(at(f.activity, i))}`).join(" · ")}` : "No annotated output cells in this capture.";
    rows($("outputs"), outputs.length ? outputs.map(({ patch, i }) => [`#${patch.id} ${short(labels(patch), 49)}`, fmt(at(f.activity, i)), fmt(at(f.defect, i))]) : [["No annotated output cells in this capture", "—", "—"]]);
    drawCurve(); drawGraph(); drawRetina();
  }

  function branchTrace(branch) { return validTrace(branch?.trace) ? branch.trace : validTrace(branch) ? branch : null; }
  function diagnosticIdentity() {
    return diagnostic?.identity ?? branchTrace(diagnostic?.actual)?.identity
      ?? { requestId: diagnostic?.requestId, generation: diagnostic?.generation };
  }
  function pixelDifference(key, id, edge = false) {
    const a = branchTrace(diagnostic?.actual), b = branchTrace(diagnostic?.black);
    if (!a || !b) return null;
    const ai = (edge ? a.edges : a.patches).findIndex(p => p.id === id), bi = (edge ? b.edges : b.patches).findIndex(p => p.id === id);
    if (ai < 0 || bi < 0) return null;
    const av = at(a.frames.at(-1)[key], ai), bv = at(b.frames.at(-1)[key], bi);
    return finite(av) && finite(bv) ? av - bv : null;
  }
  function showBranch(which) {
    const branch = diagnostic?.[which], trace = branchTrace(branch);
    if (!trace) return;
    held = true; differenceMode = false;
    const meta = { identity: diagnostic.identity ?? { requestId: diagnostic.requestId, generation: diagnostic.generation }, retinal: trace.retinal ?? diagnostic.retinal ?? diagnostic.retina,
      policyAccepted: false, diagnosticBranch: which, ...branch?.metadata };
    // A missing black-branch capture is left missing; never fabricate a preview.
    if (which === "black" && !trace.retinal) meta.retinal = null;
    commit({ trace: structuredClone(trace), metadata: structuredClone(meta) });
    $("actual").setAttribute("aria-pressed", String(which === "actual")); $("black").setAttribute("aria-pressed", String(which === "black"));
  }
  function setDiagnostic(result) {
    if (!result || typeof result !== "object") return false;
    const gen = result.identity?.generation ?? result.generation;
    if (generation !== null && Number.isInteger(gen) && gen < generation) return false;
    if (Number.isInteger(gen) && gen !== generation) reset(gen);
    diagnostic = structuredClone(result);
    externalStatus = "";
    $("comparison").hidden = false;
    const a = branchTrace(result.actual), b = branchTrace(result.black);
    const restored = result.stateRestored === true || result.restored === true;
    const complete = result.diagnosticOnly === true && result.comparable === true && !!a?.converged && !!b?.converged && restored;
    $("contraststatus").textContent = complete
      ? "Both paired solves qualified; original state restored. Differences below are measured responses to this pixel intervention."
      : `Incomplete paired diagnostic · actual ${a?.converged ? "qualified" : "unqualified/missing"} · black ${b?.converged ? "qualified" : "unqualified/missing"} · restoration ${restored ? "confirmed" : "not confirmed"}. No successful causal comparison asserted.`;
    $("contraststatus").className = complete ? "" : "sv-warning";
    const downstream = result.nonreceptor;
    $("downstream").textContent = downstream
      ? `Outside photoreceptors: ${downstream.changedCount ?? "?"}/${downstream.count ?? "?"} cells have nonzero numerical differences (including roundoff); max |Δ activity| = ${fmt(downstream.maxAbsDelta)}${downstream.maxIndex != null ? ` at cell #${downstream.maxIndex}` : ""}. Counts do not measure meaningful responses.`
      : "No whole-graph downstream difference summary supplied.";
    const deltas = result.deltaReadouts ?? result.readoutDeltas ?? result.deltas ?? {};
    const entries = Object.entries(deltas).filter(([, value]) => finite(value)).sort((a, b) => Math.abs(b[1]) - Math.abs(a[1])).slice(0, 14);
    rows($("deltas"), entries.length ? entries.map(([name, value]) => [name, fmt(value)]) : [["No numeric readout differences supplied", "—"]]);
    $("actual").disabled = !a; $("black").disabled = !b;
    $("difference").disabled = !complete;
    if (a) showBranch("actual"); dirty = true; return true;
  }
  $("actual").onclick = () => showBranch("actual"); $("black").onclick = () => showBranch("black");
  $("difference").onclick = () => {
    if ($( "difference").disabled) return;
    showBranch("actual"); index = current.trace.frames.length - 1; playing = false; held = true; differenceMode = true; dirty = true;
  };

  function update(now = performance.now()) {
    if (dead || !visible) return;
    if (current && playing) {
      if (lastTick === null) lastTick = now;
      const advance = Math.floor((now - lastTick) / 160);
      if (advance > 0) {
        index = Math.min(current.trace.frames.length - 1, index + advance); lastTick = now; dirty = true;
        if (index === current.trace.frames.length - 1) playing = false;
      }
    } else if (!held && pending) { const next = pending; pending = null; commit(next); }
    if (dirty) render();
  }
  function setVisible(on) { visible = !!on; container.hidden = !visible; lastTick = null; dirty = true; if (visible) update(); }
  function reset(gen = null) {
    current = pending = diagnostic = null; index = patchIndex = 0; playing = held = differenceMode = false; lastTick = null; generation = gen;
    externalStatus = "Waiting for a captured settlement."; $("comparison").hidden = true;
    $("source").textContent = `Generation ${gen ?? "?"} · no captured observation`;
    $("global").textContent = $("local").textContent = "—";
    $("frame").textContent = "No recorded iterations."; $("patch").replaceChildren();
    $("difference").disabled = true; $("contrastbrief").textContent = "Pixel effect not isolated yet.";
    $("localbrief").textContent = "No captured local equation.";
    $("outputbrief").textContent = "No selected output activities.";
    $("retinainfo").textContent = "No captured pixel frame is attached.";
    $("graphnote").textContent = "An excerpt of the full graph, not a replacement network.";
    $("graphlegend").textContent = "Fill: activity · orange ring: defect · violet: shared input. Click a receiving cell.";
    $("equation").textContent = "rᵢ = Σⱼ wᵢⱼ sⱼ + driveᵢ + biasᵢ − vᵢ";
    $("contraststatus").textContent = $("downstream").textContent = "";
    $("scrub").value = $("scrub").max = "0";
    for (const key of ["difference", "actual", "black"]) $(key).setAttribute("aria-pressed", "false");
    rows($("patchvalues"), []); rows($("outputs"), []); rows($("deltas"), []);
    for (const name of ["curve", "graph", "retina"]) ctx(name).clearRect(0, 0, $(name).width, $(name).height);
    dirty = true;
  }
  function snapshot() {
    if (!current) return { visible, hasTrace: false, generation, status: externalStatus };
    const t = current.trace, f = frame(), p = t.patches[patchIndex], packet = retinalPacket(current.metadata), n = neighborhood();
    return structuredClone({ visible, hasTrace: true, identity: identity(), phase: t.phase, diagnosticBranch: current.metadata.diagnosticBranch ?? null,
      replay: true, playing, held, queued: !!pending, graphMode: differenceMode ? "final-pixel-difference" : "replay-activity",
      frameIndex: index, frameCount: t.frames.length, iteration: f.iteration,
      globalResidual: f.residual, localMaxDefect: f.localMaxDefect, settledCount: f.settledCount, n: t.n, tolerance: t.tolerance,
      finalConverged: t.converged, finalReason: t.reason, policyAccepted: current.metadata.policyAccepted ?? null,
      patch: { id: p.id, labels: p.labels, activity: at(f.activity, patchIndex), potential: at(f.potential, patchIndex),
        defect: at(f.defect, patchIndex), deltaV: at(f.deltaV, patchIndex), previousDefect: at(f.previousDefect, patchIndex),
        drive: at(f.drive, patchIndex), bias: at(f.bias, patchIndex), synapticInput: at(f.synapticInput, patchIndex),
        incomingCount: p.incomingCount, capturedIncomingCount: p.sampledIncomingCount,
        displayedIncomingCount: n.shown.filter(e => e.post === p.id).length,
        sharedInputIds: (t.sharedInputs || []).filter(s => s.posts?.includes(p.id)).map(s => s.id), finalPixelActivityDifference: pixelDifference("activity", p.id) },
      retinal: packet ? { width: packet.width, height: packet.height, origin: packet.origin, captured: true,
        firstSourcePixel: Array.from(list(packet.rgba).slice(0, 4)), rgbaFNV1a32: pixelHash(packet.rgba), receptorChain: receptorChain() } : null,
      contrast: diagnostic ? { available: true, identity: diagnosticIdentity(), actualConverged: branchTrace(diagnostic.actual)?.converged ?? false,
        blackConverged: branchTrace(diagnostic.black)?.converged ?? false,
        stateRestored: diagnostic.stateRestored === true || diagnostic.restored === true,
        comparable: diagnostic.comparable === true, nonreceptor: diagnostic.nonreceptor ?? null,
        readoutDeltas: diagnostic.deltaReadouts ?? diagnostic.readoutDeltas ?? diagnostic.deltas ?? {} } : { available: false } });
  }
  function pixelHash(bytes) {
    let hash = 2166136261;
    for (const value of list(bytes)) hash = Math.imul(hash ^ value, 16777619);
    return (hash >>> 0).toString(16).padStart(8, "0");
  }
  render();
  return { setTrace, setDiagnostic, setStatus, setVisible, update, reset, snapshot,
    resize() { dirty = true; }, destroy() { dead = true; current = pending = diagnostic = null; container.replaceChildren(); } };
}
