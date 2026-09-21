// The page: a board to play on, and the brain drawn from what it reports while it thinks.
// Nothing is stored: a reload starts from the brain as it was shipped.
import { Position, PROVEN } from "./brain.js";

// The search plays with the settings the receipt beside the page was measured with, so the
// numbers on the card are the numbers of what plays here. These are used only without one.
const FALLBACK = { reads: 2000, lateStones: 18, lateReads: 30000, visits: 1000000 };
const $ = (id) => document.getElementById(id);
const worker = new Worker("worker.js", { type: "module" });

// ------------------------------------------------------------------ the game
const game = { columns: [], humanFirst: true, over: false, busy: true, score: { you: 0, brain: 0, draw: 0 }, thinkId: 0 };
const boardEl = $("board"), columnsEl = $("columns");
const holes = [];                                       // holes[column][row], row 0 the floor
for (let c = 0; c < 7; c++) {
  const col = document.createElement("div");
  col.className = "col off";
  col.addEventListener("click", () => humanPlays(c));
  holes.push([]);
  for (let r = 5; r >= 0; r--) { const hole = document.createElement("div"); hole.className = "hole"; col.appendChild(hole); holes[c][r] = hole; }
  boardEl.appendChild(col);
  const bar = document.createElement("div");
  bar.className = "bar"; bar.innerHTML = "<i></i><b></b>";
  columnsEl.appendChild(bar);
}
const position = () => { const p = new Position(); for (const c of game.columns) p.play(c); return p; };
const owner = (ply) => ((ply % 2 === 0) === game.humanFirst ? "you" : "brain");

function setState(text, kind) { const el = $("state"); el.textContent = text; el.className = "state " + (kind || ""); }
function enable(on) { for (const col of boardEl.children) col.classList.toggle("off", !on); }
function clearGhosts() { for (const g of boardEl.querySelectorAll(".ghost")) g.remove(); }

function place(column, animate = true) {
  const p = position(), row = p.height[column], ply = game.columns.length;
  for (const s of boardEl.querySelectorAll(".stone.last")) s.classList.remove("last");
  const stone = document.createElement("div");
  stone.className = `stone ${owner(ply)} last`;
  stone.style.setProperty("--fall", String(6 - row));
  if (!animate) stone.style.animation = "none";
  holes[column][row].appendChild(stone);
  game.columns.push(column);
  p.play(column);
  return p;
}

function winningCells(p) {
  const last = game.columns[game.columns.length - 1], row = p.height[last] - 1, side = p.cells[row * 7 + last];
  for (const [dr, dc] of [[0, 1], [1, 0], [1, 1], [1, -1]]) {
    const run = [[row, last]];
    for (const s of [-1, 1]) { let r = row + s * dr, c = last + s * dc; while (r >= 0 && r < 6 && c >= 0 && c < 7 && p.cells[r * 7 + c] === side) { run.push([r, c]); r += s * dr; c += s * dc; } }
    if (run.length >= 4) return run;
  }
  return [];
}

function finish(p) {
  game.over = true; enable(false); clearGhosts();
  if (p.lost) {
    const who = owner(game.columns.length - 1), cells = winningCells(p);
    for (const s of boardEl.querySelectorAll(".stone")) s.classList.add("dim");
    for (const [r, c] of cells) { const s = holes[c][r].firstChild; s.classList.remove("dim"); s.classList.add("win"); }
    game.score[who]++; setState(who === "you" ? "you won" : "the brain won", "over");
  } else { game.score.draw++; setState("drawn", "over"); }
  $("scoreYou").textContent = game.score.you; $("scoreBrain").textContent = game.score.brain; $("scoreDraw").textContent = game.score.draw;
  mind.rest(game.columns);
}

function humanPlays(column) {
  if (game.over || game.busy) return;
  const before = position();
  if (!before.playable(column)) return;
  clearGhosts();
  const p = place(column);
  if (p.lost || p.full) return finish(p);
  brainThinks();
}

function brainThinks() {
  game.busy = true; enable(false);
  setState("the brain is thinking", "busy"); mind.begin();
  worker.postMessage({ type: "think", id: ++game.thinkId, columns: game.columns.slice() });
}

function showColumns(thought) {
  const values = thought ? thought.columns : {};
  [...columnsEl.children].forEach((bar, c) => {
    const v = values[c], fill = bar.firstChild, label = bar.lastChild;
    bar.className = "bar";
    if (v === undefined) { fill.style.height = "0"; label.textContent = ""; return; }
    const proven = Math.abs(v) > PROVEN - 1, shown = Math.max(-1, Math.min(1, v));
    bar.classList.add(shown >= 0 ? "up" : "down", "shown");
    if (proven) bar.classList.add("proven");
    if (c === thought.column) bar.classList.add("best");
    fill.style.height = `${Math.max(3, Math.sqrt(Math.abs(shown)) * 46)}%`;
    label.textContent = proven ? (v > 0 ? "win" : "loss") : (shown >= 0 ? "+" : "−") + Math.abs(shown).toFixed(2).slice(1);
  });
}

function showLine(thought) {
  clearGhosts();
  const heights = Array.from(position().height);
  thought.line.slice(1, 6).forEach((column, k) => {
    const row = heights[column]++; if (row > 5) return;
    const ghost = document.createElement("div");
    ghost.className = `ghost ${owner(game.columns.length + k)}`;
    ghost.style.animationDelay = `${0.12 * k}s`; ghost.textContent = String(k + 1);
    holes[column][row].appendChild(ghost);
  });
}

function brainPlays(thought, ms) {
  const p = place(thought.column);
  showColumns(thought);
  const v = thought.value, sure = Math.abs(v) > PROVEN - 1;
  $("thought").textContent = `${thought.reads.toLocaleString()} boards read · ${thought.depth} plies deep · ${Math.round(ms)} ms` + (sure ? ` · ${v > 0 ? "sees a win" : "sees a loss"} in ${Math.round((PROVEN - Math.abs(v)) * 100)}` : "");
  if (p.lost || p.full) return finish(p);
  showLine(thought);
  game.busy = false; enable(true); setState("your move", "live"); mind.rest(game.columns);
}

function newGame() {
  game.thinkId++; game.columns = []; game.over = false;
  for (const s of boardEl.querySelectorAll(".stone, .ghost")) s.remove();
  showColumns(null); $("thought").textContent = "";
  game.humanFirst = $("side").value === "first";
  document.querySelector(".chip.you i").style.background = "";
  if (game.humanFirst) { game.busy = false; enable(true); setState("your move", "live"); mind.rest([]); } else brainThinks();
}
$("newGame").addEventListener("click", newGame);
$("side").addEventListener("change", newGame);

// ------------------------------------------------------------------ the brain, drawn
const mind = (() => {
  const canvas = $("scan"), ctx = canvas.getContext("2d");
  let W = 0, H = 0, dpr = 1, sizes = { hidden: 256, cells: 4096, active: 32 };
  const retina = new Float32Array(84), seenStone = new Uint8Array(84);
  let context = new Float32Array(256), contextShown = new Float32Array(256), cellGlow = new Float32Array(4096);
  let value = 0, valueShown = 0, slowShown = 0, history = [], queue = [], pending = null, thinking = false, restTrace = null, lastBeat = 0, caption = "at rest", pulses = [], readsShown = 0;
  let geo = null;

  function resize() {
    const box = canvas.parentElement.getBoundingClientRect();
    dpr = Math.min(window.devicePixelRatio || 1, 2); W = Math.max(200, box.width); H = Math.max(200, box.height);
    canvas.width = Math.round(W * dpr); canvas.height = Math.round(H * dpr);
    layout();
  }
  function layout() {
    const side = Math.round(Math.sqrt(sizes.hidden)), pad = 22, head = 34, foot = 44;
    const band = (w, h) => { const cols = Math.ceil(Math.sqrt(sizes.cells * w / h)), rows = Math.ceil(sizes.cells / cols); return { cols, rows }; };
    if (W >= 520) {
      // two rows: the reading and the context above, the record band and the value below
      const rowGap = 46, usable = H - 2 * head - rowGap - foot, top = Math.max(120, usable * 0.54), low = Math.max(90, usable - top);
      const cw = Math.min(top, W * 0.36), rw = Math.min((W - 2 * pad - cw - 70) / 2 - 8, top * 1.12), rh = rw * 6 / 7;
      const total = 2 * rw + 16 + 64 + cw, x0 = (W - total) / 2, y0 = head, gaugeW = 104, kw = W - 2 * pad - gaugeW, ky = head + top + rowGap + 6;
      geo = { stack: false, side, own: { x: x0, y: y0 + (top - rh) / 2, w: rw, h: rh }, other: { x: x0 + rw + 16, y: y0 + (top - rh) / 2, w: rw, h: rh },
        context: { x: x0 + 2 * rw + 16 + 64, y: y0 + (top - cw) / 2, w: cw, h: cw }, records: { x: pad, y: ky, w: kw, h: low - 6, ...band(kw, low - 6) },
        gauge: { x: pad + kw + 84, y: ky, w: 8, h: low - 6 }, spark: { x: pad, y: H - 26, w: W - 2 * pad, h: 20 } };
    } else {
      const rw = Math.min((W - 2 * pad - 14) / 2, 150), rh = rw * 6 / 7, cw = Math.min(W * 0.54, 210), kw = W - 2 * pad - 46;
      const free = H - 3 * head - foot - rh - cw - 30, kh = Math.max(90, Math.min(free, kw * 0.62));
      let y = head;
      geo = { stack: true, side, own: { x: W / 2 - rw - 7, y, w: rw, h: rh }, other: { x: W / 2 + 7, y, w: rw, h: rh } };
      y += rh + head + 6; geo.context = { x: (W - cw) / 2, y, w: cw, h: cw };
      y += cw + head + 6; geo.records = { x: pad, y, w: kw, h: kh, ...band(kw, kh) };
      geo.gauge = { x: pad + kw + 24, y, w: 8, h: kh }; geo.spark = { x: pad, y: H - 26, w: W - 2 * pad, h: 20 };
    }
  }

  function apply(trace, strength) {
    seenStone.fill(0);
    for (let k = 0; k < 84; k++) if (trace.reading[k]) { retina[k] = Math.max(retina[k], strength); seenStone[k] = 1; }
    context = trace.context;
    for (let i = 0; i < trace.cells.length; i++) { const c = trace.cells[i]; cellGlow[c] = Math.max(cellGlow[c], strength * Math.min(1, trace.activity[i] * 4.2)); }
    value = trace.value;
    if (strength > 0.6) { history.push(trace.value); if (history.length > 150) history.shift(); pulses.push({ t: 0 }); if (pulses.length > 14) pulses.shift(); }
  }

  const center = (r) => [r.x + r.w / 2, r.y + r.h / 2];
  function link(a, b, alpha, t, vertical) {
    const [ax, ay] = vertical ? [a.x + a.w / 2, a.y + a.h + 6] : [a.x + a.w + 6, a.y + a.h / 2], [bx, by] = vertical ? [b.x + b.w / 2, b.y - 6] : [b.x - 6, b.y + b.h / 2];
    const spread = (vertical ? Math.min(a.w, b.w) : Math.min(a.h, b.h)) * 0.2;
    ctx.lineWidth = 1;
    for (let k = -2; k <= 2; k++) {
      ctx.strokeStyle = `rgba(120, 190, 220, ${alpha * (1 - Math.abs(k) * 0.22)})`;
      const o = k * spread; ctx.beginPath();
      if (vertical) { ctx.moveTo(ax + o, ay); ctx.bezierCurveTo(ax + o, (ay + by) / 2, bx + o * 2.2, (ay + by) / 2, bx + o * 2.2, by); }
      else { ctx.moveTo(ax, ay + o); ctx.bezierCurveTo((ax + bx) / 2, ay + o, (ax + bx) / 2, by + o, bx, by + o); }
      ctx.stroke();
    }
    if (t !== undefined) {
      const e = t * t * (3 - 2 * t), x = ax + (bx - ax) * e, y = ay + (by - ay) * e;
      const g = ctx.createRadialGradient(x, y, 0, x, y, 11); g.addColorStop(0, "rgba(190, 245, 255, 0.95)"); g.addColorStop(0.4, "rgba(120, 210, 240, 0.35)"); g.addColorStop(1, "rgba(120, 210, 240, 0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, 11, 0, 6.3); ctx.fill();
    }
  }
  function label(text, r, sub, align = "left") {
    const x = align === "right" ? r.x + r.w : r.x;
    ctx.textAlign = align; ctx.fillStyle = "rgba(146, 166, 186, 0.95)"; ctx.font = "600 10px ui-monospace, Menlo, monospace";
    ctx.fillText(text.toUpperCase(), x, r.y - (sub ? 23 : 12));
    if (sub) { ctx.fillStyle = "rgba(99, 120, 141, 0.95)"; ctx.font = "10px ui-monospace, Menlo, monospace"; ctx.fillText(sub, x, r.y - 11); }
    ctx.textAlign = "left";
  }
  function drawRetina(r, offset, rgb) {
    const step = r.w / 7, rad = step * 0.36;
    ctx.fillStyle = "rgba(10, 20, 30, 0.9)"; ctx.strokeStyle = "rgba(34, 54, 74, 0.9)"; ctx.lineWidth = 1;
    roundRect(r.x - 5, r.y - 5, r.w + 10, r.h + 10, 8); ctx.fill(); ctx.stroke();
    for (let row = 0; row < 6; row++) for (let c = 0; c < 7; c++) {
      const v = retina[offset + row * 7 + c], x = r.x + (c + 0.5) * step, y = r.y + r.h - (row + 0.5) * step;
      ctx.fillStyle = "rgba(30, 46, 62, 0.85)"; ctx.beginPath(); ctx.arc(x, y, rad * 0.55, 0, 6.3); ctx.fill();
      if (v > 0.02) {
        const g = ctx.createRadialGradient(x, y, 0, x, y, rad * 2.1); g.addColorStop(0, `rgba(${rgb}, ${0.95 * v})`); g.addColorStop(0.45, `rgba(${rgb}, ${0.5 * v})`); g.addColorStop(1, `rgba(${rgb}, 0)`);
        ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, rad * 2.1, 0, 6.3); ctx.fill();
      }
    }
  }
  function roundRect(x, y, w, h, r) { ctx.beginPath(); ctx.moveTo(x + r, y); ctx.arcTo(x + w, y, x + w, y + h, r); ctx.arcTo(x + w, y + h, x, y + h, r); ctx.arcTo(x, y + h, x, y, r); ctx.arcTo(x, y, x + w, y, r); ctx.closePath(); }

  function frame(now) {
    requestAnimationFrame(frame);
    if (!geo) return;
    // play queued reads at a speed the eye can follow; a long search plays faster
    if (queue.length) {
      const perFrame = Math.max(1, Math.ceil(queue.length / 70));
      for (let k = 0; k < perFrame && queue.length; k++) { const t = queue.shift(); apply(t, 1); readsShown++; caption = `reading a board ${t.depth} ${t.depth === 1 ? "ply" : "plies"} ahead`; }
    } else if (pending) { const done = pending; pending = null; thinking = false; caption = "at rest"; $("mindState").className = "state"; done(); }
    else if (!thinking && restTrace && now - lastBeat > 1500) { lastBeat = now; apply(restTrace, 0.55); }
    $("mindState").textContent = caption;

    for (let k = 0; k < 84; k++) retina[k] *= thinking ? 0.8 : 0.965;
    // at rest the retinas keep showing the board that is really there
    if (!thinking && !queue.length && restTrace) for (let k = 0; k < 84; k++) if (restTrace.reading[k]) retina[k] = Math.max(retina[k], 0.5);
    for (let k = 0; k < contextShown.length; k++) contextShown[k] += (context[k] - contextShown[k]) * (thinking ? 0.5 : 0.12);
    for (let k = 0; k < cellGlow.length; k++) if (cellGlow[k] > 0.003) cellGlow[k] *= thinking ? 0.86 : 0.955; else cellGlow[k] = 0;
    valueShown += (value - valueShown) * 0.25;
    for (const p of pulses) p.t += 0.045;
    pulses = pulses.filter((p) => p.t < 3);

    ctx.setTransform(dpr, 0, 0, dpr, 0, 0); ctx.clearRect(0, 0, W, H);
    const heat = thinking ? 0.2 : 0.08;
    const pulseAt = (k) => { for (const p of pulses) if (p.t >= k && p.t < k + 1) return p.t - k; return undefined; };
    link(geo.stack ? { x: geo.own.x, y: geo.own.y, w: geo.other.x + geo.other.w - geo.own.x, h: geo.own.h } : geo.other, geo.context, heat, pulseAt(0), geo.stack);
    link(geo.context, geo.records, heat, pulseAt(1), true);
    link(geo.records, { x: geo.gauge.x, y: geo.gauge.y, w: 0, h: geo.gauge.h }, heat, pulseAt(2), false);

    label("reading", geo.own, "its stones"); drawRetina(geo.own, 0, "245, 197, 66");
    label("", geo.other, "your stones"); drawRetina(geo.other, 42, "240, 89, 107");

    // the context: one cell per channel, teal where positive and warm where negative
    const C = geo.context, n = geo.side, cs = C.w / n;
    label("context", C, `${sizes.hidden} channels`);
    ctx.fillStyle = "rgba(10, 20, 30, 0.9)"; ctx.strokeStyle = "rgba(34, 54, 74, 0.9)"; ctx.lineWidth = 1; roundRect(C.x - 6, C.y - 6, C.w + 12, C.h + 12, 10); ctx.fill(); ctx.stroke();
    for (let k = 0; k < contextShown.length; k++) {
      const v = contextShown[k], a = Math.min(1, Math.abs(v) * 2.6), x = C.x + (k % n) * cs, y = C.y + Math.floor(k / n) * cs;
      ctx.fillStyle = "rgba(24, 38, 52, 0.9)"; ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
      if (a > 0.03) { ctx.fillStyle = v > 0 ? `rgba(89, 229, 203, ${a})` : `rgba(255, 140, 90, ${a})`; ctx.fillRect(x + 1, y + 1, cs - 2, cs - 2);
        if (a > 0.55) { ctx.shadowColor = v > 0 ? "rgba(89, 229, 203, 0.9)" : "rgba(255, 140, 90, 0.9)"; ctx.shadowBlur = 9; ctx.fillRect(x + 2, y + 2, cs - 4, cs - 4); ctx.shadowBlur = 0; } }
    }

    // the record store: every cell a dot; the cells a read addresses flash
    const R = geo.records, sx = R.w / R.cols, sy = R.h / R.rows, dot = Math.max(1, Math.min(sx, sy) * 0.42);
    label("records", R, geo.stack ? `${sizes.cells.toLocaleString()} cells · none written yet` : `${sizes.cells.toLocaleString()} cells · none written yet · a read addresses ${sizes.active}`);
    ctx.fillStyle = "rgba(10, 20, 30, 0.9)"; ctx.strokeStyle = "rgba(34, 54, 74, 0.9)"; roundRect(R.x - 6, R.y - 6, R.w + 12, R.h + 12, 10); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "rgba(44, 68, 94, 0.8)";
    for (let k = 0; k < sizes.cells; k++) ctx.fillRect(R.x + (k % R.cols + 0.5) * sx - dot / 2, R.y + (Math.floor(k / R.cols) + 0.5) * sy - dot / 2, dot, dot);
    ctx.globalCompositeOperation = "lighter";
    for (let k = 0; k < sizes.cells; k++) {
      const v = cellGlow[k]; if (v <= 0) continue;
      const x = R.x + (k % R.cols + 0.5) * sx, y = R.y + (Math.floor(k / R.cols) + 0.5) * sy, rad = Math.min(sx, sy) * (1.5 + 3.2 * v);
      const g = ctx.createRadialGradient(x, y, 0, x, y, rad); g.addColorStop(0, `rgba(190, 222, 255, ${v})`); g.addColorStop(0.35, `rgba(96, 165, 250, ${0.55 * v})`); g.addColorStop(1, "rgba(96, 165, 250, 0)");
      ctx.fillStyle = g; ctx.beginPath(); ctx.arc(x, y, rad, 0, 6.3); ctx.fill();
    }
    ctx.globalCompositeOperation = "source-over";

    // the value it reads out, for the side that placed the stone
    const G = geo.gauge, zero = G.y + G.h / 2, vy = zero - Math.max(-1, Math.min(1, valueShown)) * G.h / 2;
    if (!geo.stack) label("value", { x: G.x - 40, y: G.y, w: 48 + G.w, h: 0 }, null, "right");
    ctx.fillStyle = "rgba(24, 38, 52, 0.95)"; roundRect(G.x, G.y, G.w, G.h, 4); ctx.fill();
    ctx.fillStyle = valueShown >= 0 ? "rgba(89, 229, 203, 0.95)" : "rgba(255, 140, 90, 0.95)";
    ctx.shadowColor = ctx.fillStyle; ctx.shadowBlur = 10; roundRect(G.x, Math.min(vy, zero), G.w, Math.max(2, Math.abs(vy - zero)), 4); ctx.fill(); ctx.shadowBlur = 0;
    ctx.fillStyle = "#e3ecf4"; ctx.font = "600 14px ui-monospace, Menlo, monospace"; ctx.textAlign = "right";
    const shownValue = (valueShown >= 0 ? "+" : "−") + Math.abs(valueShown).toFixed(2);
    if (geo.stack) ctx.fillText(shownValue, G.x + G.w, G.y - 11); else ctx.fillText(shownValue, G.x - 10, zero + 5);
    ctx.textAlign = "left";

    // the values read in this search, in the order they were read
    const S = geo.spark;
    if (history.length > 2) {
      ctx.strokeStyle = "rgba(34, 54, 74, 0.9)"; ctx.lineWidth = 1; ctx.beginPath(); ctx.moveTo(S.x, S.y); ctx.lineTo(S.x + S.w, S.y); ctx.stroke();
      ctx.strokeStyle = "rgba(143, 200, 230, 0.8)"; ctx.lineWidth = 1.2; ctx.beginPath();
      history.forEach((v, i) => { const x = S.x + (i / 149) * S.w, y = S.y - Math.max(-1, Math.min(1, v)) * S.h / 2; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); }); ctx.stroke();
      ctx.fillStyle = "rgba(99, 120, 141, 0.95)"; ctx.font = "10px ui-monospace, Menlo, monospace"; ctx.fillText("values read in this search", S.x, S.y + S.h / 2 + 12);
    }
  }

  new ResizeObserver(resize).observe(canvas.parentElement);
  requestAnimationFrame(frame);
  return {
    setSizes(s) { sizes = s; contextShown = new Float32Array(s.hidden); context = new Float32Array(s.hidden); cellGlow = new Float32Array(s.cells); layout(); },
    begin() { thinking = true; queue = []; history = []; readsShown = 0; caption = "thinking"; $("mindState").className = "state busy"; },
    read(trace) { queue.push(trace); },
    whenShown(done) { pending = done; },
    rest(columns) { worker.postMessage({ type: "read", columns: columns.slice() }); },
    setRest(trace) { restTrace = trace.empty ? null : trace; lastBeat = 0; },
  };
})();

// ------------------------------------------------------------------ the worker's reports
worker.onmessage = (event) => {
  const m = event.data;
  if (m.type === "ready") {
    mind.setSizes(m.sizes);
    $("facts").textContent = `${m.sizes.hidden} context channels · ${m.sizes.cells.toLocaleString()} record cells\n${m.search.reads.toLocaleString()} boards read a move · up to ${m.search.lateReads.toLocaleString()} from ${m.search.lateStones} stones`;
    $("cardFacts").textContent = `cadence RecordPatchNet · ${(84 * m.sizes.hidden * 2 + m.sizes.hidden * 3 + 1).toLocaleString()} slow parameters · ${m.sizes.cells.toLocaleString()} record cells`;
    newGame();
  } else if (m.type === "read") mind.read(m);
  else if (m.type === "rest") mind.setRest(m);
  else if (m.type === "move" && m.id === game.thinkId) mind.whenShown(() => brainPlays(m.thought, m.ms));
};
const receiptLoaded = fetch("receipt.json").then((r) => r.json()).catch(() => null);
receiptLoaded.then((receipt) => {
  const m = receipt && receipt.search;
  const search = m ? { reads: m.reads, lateStones: m.late_stones, lateReads: m.late_reads, visits: m.visits } : FALLBACK;
  worker.postMessage({ type: "load", url: new URL("brain.json", location.href).href, search });
});

// ------------------------------------------------------------------ the card's numbers come from the receipt beside the page
const NAMES = { one_ply: "one-ply tactician", solver_070: "perfect solver, 70% of its moves", solver_100: "perfect solver" };
receiptLoaded.then((receipt) => {
  if (!receipt) throw new Error("no receipt");
  const rows = Object.entries(receipt.opponents).map(([name, e]) => {
    const pretty = NAMES[name] || (name.startsWith("alphazero=") ? `AlphaZero, ${name.split("@")[1]} simulations` : name);
    const wdl = (a) => a.join("–");
    return `<tr><td>${pretty}</td><td class="n">${wdl(e.win_draw_loss.first)}</td><td class="n">${wdl(e.win_draw_loss.second)}</td><td>${(100 * e.all.blunder_when_not_lost).toFixed(1)}%</td><td>${Math.round(e.latency_ms.mean)} ms</td></tr>`;
  }).join("");
  $("results").innerHTML = `<table><tr><th>against</th><th>moving first</th><th>moving second</th><th>blunders</th><th>a move</th></tr>${rows}<caption>${receipt.games_per_opponent} paired games each, won–drawn–lost. Every move graded by Pascal Pons' solver; a blunder changes the theoretical result, counted where the position was not already lost.</caption></table>`;
}).catch(() => { $("results").textContent = "the receipt is not beside the page"; });
