// The S03 Connect Four page: a board a visitor plays on, with the whole brain beside it.
//
// The visitor drops a stone by clicking a column; the brain answers from the same records the
// Python brain learned, through the same imagination and the same search (connect_four/web/
// brain.js is the port, and connect_four/web/parity.mjs holds it to the Python numbers). Every
// move either side makes is written into the records of the cells that read it, so the brain
// keeps learning from the game being played, and the freeze switch stops it.
//
// Beside the board the whole brain runs through the standard BrainScan renderer: every settling
// step of every phase of the S00 agent is queued by the engine's callbacks and played back
// while the visitor looks at the board, synapses flash when an update moves them, and the two
// record cortices are drawn by the shared records view: the drop cortex shows the cells of the
// chosen column's reading, the line cortex the cells of the windows through the new stone, and
// both flash where the outcome was written. Nothing is scripted: the brain plays only as well
// as the records it has learned let it.

import { BrainScan } from "../../web/brain_scan.js";
import { RecordsView, readsHTML } from "../../web/records_view.js";
import { Brain } from "./brain.js";
import { Board, World, gameConfig } from "./game.js";

const $ = (id) => document.getElementById(id);
const PAGE_OPTIONS = { autoplay: true, ...(window.__CF_PAGE__ || {}) };
const PHASE_TEXT = { free: "free phase", free_post: "free phase after learning", predict: "predict", imagine: "imagine", repair_free: "world repair · free", repair_plus: "world repair · nudged +β", repair_minus: "world repair · nudged −β", bootstrap: "bootstrap value (time limit)", score_plus: "actor score · +β", score_minus: "actor score · −β" };
const REGION_NOTE = { sensory: "the board, cell by cell", goal: "whose move it is", efference: "copy of the executed column", workspace: "association", dynamics: "reads the efference copy", context: "trace of the workspace", motor: "column readout", prediction: "predicted consequences", perception: "perception", recall: "recall", intention: "intention" };
const RED = [240, 89, 107], YELLOW = [245, 197, 66], RECORDS = [96, 165, 250], ACCENT = [89, 229, 203];
const DROP_MS = 260, GAP_MS = 190;  // one stone falls in DROP_MS, the next starts GAP_MS later
const shade = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(`cannot load ${url}: ${response.status}`);
  return response.json();
}

/** The checkpoint the page carries and the fixed cells of its life (inlined, or fetched in development). */
async function loadPage() {
  const checkpoint = window.__CHECKPOINT__ || (await fetchJSON(document.body.dataset.checkpoint));
  let cortices = window.__CORTICES__ || null;
  if (!cortices && document.body.dataset.cortices) cortices = await fetchJSON(document.body.dataset.cortices);
  return { checkpoint, cortices };
}

function main({ checkpoint: FIRST, cortices: CORTICES }) {
  const MANIFEST = window.__MANIFEST__ || null;
  const scan = new BrainScan($("scan"), FIRST.agent.atlas, { labels: $("labels"), strip: $("strip"), labelTop: 12, montageRows: 5 });
  const queue = [];       // settle frames, records events and markers, in the order the brain produced them
  let queuedSteps = 0;

  // -- the life, rebuilt whenever another checkpoint is loaded
  let checkpoint = { id: MANIFEST ? MANIFEST.inlined : "inlined", label: FIRST.label || "this page's brain" };
  let brain = null, world = null, config = null, views = null, fields = null;
  let restState = null, lastS = null, lastV = null, previousWeights = null;

  // -- the game
  let human = "red", started = false, over = false, thinking = false, display = null, payloadNow = null;
  let lastSearch = null, lastCells = null, winningLine = null, lastCell = -1;
  let hover = null, boardFrame = { x: 0, y: 0, cell: 1, radius: 1 };
  const falling = [];
  let nextFall = 0;
  const score = { you: 0, brain: 0, draws: 0 };
  const stats = freshStats();
  let running = true, currentPhase = null, ledgerPhase = null, phaseText = "—", phaseClass = "", lastStepText = "";
  let lastFrame = performance.now();
  let flashText = "", flashUntil = 0;
  const collected = new Map();
  const checkpoints = MANIFEST ? MANIFEST.checkpoints.slice() : [];
  const cache = new Map();
  const boardCanvas = $("board");
  scan.onhover = (hit) => { if (hit) $("inspector").textContent = `${hit.region} · neuron ${hit.neuron}\nactivation ${hit.activation.toFixed(4)} · change ${hit.change.toExponential(2)}` + (hit.potential === null ? "" : ` · potential ${hit.potential.toFixed(4)}`); };

  function freshStats() {
    return { events: 0, decisions: 0, games: 0, moves: 0, decide_ms: 0, move_ms: 0, worst_decide_ms: 0, worst_move_ms: 0, dopamine: 0, td_error: 0, learning: {}, controller: "—", last_column: null };
  }

  // ---------------------------------------------------------------- the records cortices

  /** What each cortex collected during one step of the brain, flushed into the queue after it. */
  const report = {
    bucket(cortex) {
      let entry = collected.get(cortex);
      if (!entry) { entry = { code: null, reading: null, imagine: [], index: [], values: [], fields: [], written: 0 }; collected.set(cortex, entry); }
      return entry;
    },
    code(cortex, code, reading) { const b = this.bucket(cortex); b.code = code; b.reading = Float64Array.from(reading); },
    lineCodes(cortex, codes, reading) {
      if (!codes.length) return;
      const b = this.bucket(cortex);
      b.code = codes[0]; b.reading = Float64Array.from(reading); b.imagine = codes.slice(1, 16);
    },
    write(cortex, code, out) {
      const b = this.bucket(cortex);
      if (b.index.length < 600) for (let k = 0; k < code.index.length; k++) { b.index.push(code.index[k]); b.values.push(code.values[k]); }
      b.written += out.written;
      for (const f of out.report) {
        const seen = b.fields.find((x) => x.name === f.name);
        if (seen) seen.error = Math.max(seen.error, f.error);
        else b.fields.push({ ...f });
      }
    },
  };

  /** The read of each field after the writes, as the records view renders it. */
  function readsOf(which, code) {
    if (!code) return null;
    const records = which === "drop" ? brain.drop.records : brain.lines.records, out = {};
    for (const f of records.fields) {
      const read = records.read(code, f.name);
      if (f.name === "value") { out.value = Float64Array.from([0.5 + read[0]]); continue; }
      let total = 0;
      const positive = Float64Array.from(read, (v) => (v > 0 ? v : 0));
      for (const v of positive) total += v;
      out[f.name] = total > 0 ? Float64Array.from(positive, (v) => v / total) : new Float64Array(f.width).fill(1 / f.width);
    }
    return out;
  }

  function flushRecords() {
    for (const [cortex, b] of collected) {
      const which = cortex === brain.drop ? "drop" : "lines";
      queue.push({
        kind: "records", which, code: b.code, reading: b.reading, imagine: b.imagine, reads: readsOf(which, b.code),
        write: b.index.length ? { index: Int32Array.from(b.index), values: Float64Array.from(b.values), fields: b.fields, written: b.written } : null,
      });
    }
    collected.clear();
  }

  // ---------------------------------------------------------------- the life

  function installLife(payload, label, id) {
    queue.length = 0; queuedSteps = 0; currentPhase = null; ledgerPhase = null; collected.clear();
    checkpoint = { id, label };
    brain = new Brain(payload, {
      report,
      cortices: CORTICES,
      engine: {
        onSettleStep: (phase, s, v, meta) => { queue.push({ kind: "step", phase, s: Float32Array.from(s), v: Float32Array.from(v), batch: meta.batch, step: meta.step, moved: meta.moved }); queuedSteps += 1; },
        onPhase: (info) => queue.push({ kind: "phase", phase: info.phase, steps: info.steps, residual: info.residual, converged: info.converged, batch: info.batch }),
        onLearn: (info) => queue.push({ kind: "learn", phase: info.phase, changed: info.changed.length, rejected: info.rejected, dopamine: info.dopamine, td_error: info.td_error, weights: info.changed.length ? brain.agent.effectiveWeights() : null }),
      },
    });
    brain.setLearning($("learning").value === "on");
    config = gameConfig(payload.brain.game);
    fields = {
      drop: [{ name: "landing", kind: "categorical", width: config.rows + 1 }],
      lines: [{ name: "complete", kind: "categorical", width: 2 }, { name: "value", kind: "continuous", width: 1, lo: 0.0, hi: 1.0 }],
    };
    const agent = brain.agent;
    scan.setAtlas(payload.agent.atlas);
    restState = Float32Array.from(agent.warm ? agent.warm.v : new Float64Array(agent.n), (v) => agent._act(v));
    lastV = Float32Array.from(agent.warm ? agent.warm.v : new Float64Array(agent.n));
    lastS = restState;
    scan.set(restState, { potential: lastV, draw: false });
    previousWeights = agent.effectiveWeights();
    scan.setWeights(previousWeights);
    scan.fit();
    installRecords(payload);
    buildLegend(payload);
    world = new World(config, { lifeId: "web" });
    brain.newWorld();
    Object.assign(stats, freshStats());
    updateFacts(payload);
    newGame({ quiet: true });
  }

  function installRecords(payload) {
    const make = (canvas, block, cells) => new RecordsView(canvas, {
      granules: block.cells, active: block.active, inputs: block.inputs, groups: block.groups || [], color: RECORDS,
      rate: block.config.rate, rewardRate: block.config.valued_rate, habituation: block.config.habituation,
    });
    views = { drop: make($("dropRecords"), payload.brain.cortices.drop), lines: make($("lineRecords"), payload.brain.cortices.lines) };
    const drop = payload.brain.cortices.drop, lines = payload.brain.cortices.lines;
    $("dropInfo").textContent = `${drop.cells.toLocaleString()} cells · ${drop.active} active · ${brain.drop.records.parameters().toLocaleString()} records`;
    $("lineInfo").textContent = `${lines.cells.toLocaleString()} cells · ${lines.active} active · ${brain.lines.records.parameters().toLocaleString()} records`;
    $("dropReads").innerHTML = readsHTML(null, fields.drop);
    $("lineReads").innerHTML = readsHTML(null, fields.lines);
  }

  function buildLegend(payload) {
    payloadNow = payload;
    const items = payload.agent.atlas.regions.map((r) => `<span title="${r.name}"><i style="background:rgb(${r.color.join(",")})"></i><b>${r.label ?? r.name}</b><em>${r.count}</em>${REGION_NOTE[r.name] ? ` ${REGION_NOTE[r.name]}` : ""}</span>`);
    items.push(`<span title="the drop records"><i style="background:rgb(${RECORDS.join(",")})"></i><b>drop records</b><em>${payload.brain.cortices.drop.cells.toLocaleString()}</em> where a stone lands</span>`);
    items.push(`<span title="the line records"><i style="background:rgb(${RECORDS.join(",")})"></i><b>line records</b><em>${payload.brain.cortices.lines.cells.toLocaleString()}</em> lines and outcomes</span>`);
    $("legend").innerHTML = items.join("");
    $("boardKey").innerHTML = [
      `<span><i class="dot" style="background:${humanColour()}"></i>you</span>`,
      `<span><i class="dot" style="background:${brainColour()}"></i>the brain</span>`,
      `<span><i class="dot" style="background:rgba(143,163,184,0.35)"></i>where your stone would land</span>`,
      `<span><i style="background:rgb(${ACCENT.join(",")})"></i>the column the search chose</span>`,
    ].join("");
  }

  function updateFacts(payload) {
    const b = payload.brain, agent = brain.agent;
    const records = brain.drop.records.parameters() + brain.lines.records.parameters();
    $("counts").textContent = `${checkpoint.label} · ${b.counts.transitions.toLocaleString()} moves watched · ${(b.counts.drop_writes + b.counts.line_writes + b.counts.value_writes).toLocaleString()} records written\n`
      + `${agent.n.toLocaleString()} neurons · ${agent.E.toLocaleString()} synapses · ${records.toLocaleString()} records in two cortices`;
  }

  // ---------------------------------------------------------------- the game

  const humanColour = () => (human === "red" ? `rgb(${RED.join(",")})` : `rgb(${YELLOW.join(",")})`);
  const brainColour = () => (human === "red" ? `rgb(${YELLOW.join(",")})` : `rgb(${RED.join(",")})`);
  const colourOf = (value) => (value === 1 ? RED : YELLOW);  // player 0 is red and opens

  /** A new game: the held decision is closed, the board cleared, and the brain opens when the visitor plays yellow. */
  function newGame({ quiet = false } = {}) {
    if (world && !world.closed) { brain.step(world.truncate()); flushRecords(); }
    started = false; over = false; winningLine = null; lastCell = -1; lastSearch = null;
    falling.length = 0; nextFall = 0;
    display = new Board(config);  // the empty board the page shows until the first stone falls
    $("imaginedColumns").innerHTML = "";
    $("imaginedNote").textContent = "it has not moved yet";
    $("hint").hidden = false;
    $("hint").innerHTML = human === "red" ? "<b>Click a column</b><span>you are red and you move first</span>" : "<b>The brain opens</b><span>you are yellow and answer its move</span>";
    if (human === "yellow") {
      const moment = world.reset({ first: "candidate" });
      started = true;
      runBrain(moment);
    }
    if (!quiet) flash(human === "red" ? "a new game: you are red and you open" : "a new game: the brain opens");
    updateBoardState();
  }

  /** The board the page shows: the world's while a game runs, an empty one before the first stone. */
  function boardNow() { return started ? world.board : display; }

  function legalNow() {
    if (over) return [];
    if (!started) return Array.from({ length: config.cols }, (_, c) => c);
    return world.closed ? [] : world.board.legalColumns();
  }

  /** The visitor's move, and the brain's answer. */
  function drop(column) {
    column = column | 0;
    if (over || thinking || !legalNow().includes(column)) return false;
    $("hint").hidden = true;
    const t0 = performance.now();
    const moment = started ? world.reply(column) : world.reset({ first: "opponent", opening: column });
    started = true;
    animateStone(lastCellOf());
    runBrain(moment, t0);
    return true;
  }

  function lastCellOf() {
    const board = world.board, column = board.moves[board.moves.length - 1];
    return (board.heights[column] - 1) * config.cols + column;
  }

  /** One exchange: the brain reads the moment, searches, moves, and reads its own move. */
  function runBrain(moment, since = null) {
    thinking = true;
    updateBoardState();
    const t0 = performance.now();
    const decision = brain.step(moment);
    const decide = performance.now() - t0;
    stats.events += 1;
    stats.decide_ms = decide;
    stats.worst_decide_ms = Math.max(stats.worst_decide_ms, decide);
    readLearning();
    flushRecords();
    if (decision === null) { thinking = false; finishGame(since === null ? decide : performance.now() - since); return; }
    stats.decisions += 1;
    stats.controller = decision.controller;
    stats.last_column = decision.action;
    lastSearch = brain.lastSearch;
    showImagined(lastSearch);
    const mid = world.act(decision.decision_id, decision.action);
    animateStone(lastCellOf());
    brain.step(mid);
    stats.events += 1;
    readLearning();
    flushRecords();
    thinking = false;
    const move = since === null ? performance.now() - t0 : performance.now() - since;
    stats.move_ms = move;
    stats.worst_move_ms = Math.max(stats.worst_move_ms, move);
    stats.moves += 1;
    if (mid.terminated) finishGame(move);
    else updateBoardState();
  }

  function readLearning() {
    const last = brain.agent.lastReport;
    if (!last) return;
    stats.learning = last.learning;
    if (last.learning.dopamine !== undefined) { stats.dopamine = last.learning.dopamine; stats.td_error = last.learning.td_error; }
  }

  function finishGame(seconds = 0) {
    over = true;
    stats.games += 1;
    const winner = world.winner();
    if (winner === "candidate") score.brain += 1;
    else if (winner === "opponent") score.you += 1;
    else score.draws += 1;
    const board = world.board;
    if (board.moves.length) {
      const column = board.moves[board.moves.length - 1], row = board.heights[column] - 1;
      lastCell = row * config.cols + column;
      winningLine = board.winner === null ? null : board.lineThrough(row, column);
    }
    $("youWins").textContent = String(score.you);
    $("brainWins").textContent = String(score.brain);
    $("draws").textContent = String(score.draws);
    $("hint").hidden = false;
    $("hint").innerHTML = winner === "candidate" ? "<b>The brain won</b><span>press New game</span>" : winner === "opponent" ? "<b>You won</b><span>press New game</span>" : "<b>A draw</b><span>press New game</span>";
    updateBoardState();
  }

  function updateBoardState() {
    const el = $("state");
    if (thinking) { el.textContent = "the brain is searching"; el.className = "state busy"; }
    else if (over) { const w = world.winner(); el.textContent = w === "candidate" ? "the brain won" : w === "opponent" ? "you won" : "a draw"; el.className = "state over"; }
    else { el.textContent = flashUntil > performance.now() ? flashText : "your move"; el.className = "state live"; }
    $("score").className = `score${human === "yellow" ? " swapped" : ""}`;
    for (const chip of document.querySelectorAll(".score .chip")) chip.classList.toggle("swapped", human === "yellow");
    $("counters").innerHTML = rows([
      ["game", `${stats.games + 1} · move ${boardNow().moves.length}`],
      ["you / brain", `${score.you} / ${score.brain}${score.draws ? ` · ${score.draws} drawn` : ""}`],
      ["last move", stats.move_ms ? `${stats.move_ms.toFixed(0)} ms · search ${stats.decide_ms.toFixed(0)} ms` : "—"],
      ["written", `${brain.counts.drop_writes.toLocaleString()} + ${brain.counts.line_writes.toLocaleString()}`],
    ]);
  }

  function flash(text, ms = 2600) { flashText = text; flashUntil = performance.now() + ms; updateBoardState(); }

  function showImagined(info) {
    if (!info) return;
    const chosen = info.column, legal = new Set(info.columns);
    const cells = [];
    for (let c = 0; c < config.cols; c++) {
      const at = info.columns.indexOf(c);
      const read = at >= 0 ? info.read[at] : null, searched = at >= 0 ? info.searched[at] : null;
      const height = read === null ? 0 : Math.max(2, Math.round(read * 30));
      const mark = searched === null ? "" : `<u style="bottom:${(Math.max(0, Math.min(1, searched)) * 100).toFixed(1)}%"></u>`;
      cells.push(`<div class="col${c === chosen ? " chosen" : ""}${legal.has(c) ? "" : " illegal"}">`
        + `<div class="meter" title="${legal.has(c) ? `the records read ${read.toFixed(3)} for this column; the search made it ${searched === null ? "—" : searched.toFixed(3)}` : "this column is full"}"><i style="height:${height}px"></i>${mark}</div>`
        + `<span class="value">${read === null ? "—" : read.toFixed(2)}</span><span class="name">${c + 1}</span></div>`);
    }
    $("imaginedColumns").innerHTML = cells.join("");
    const invalid = info.invalid ? ` · ${info.invalid} imagined boards rejected` : "";
    $("imaginedNote").textContent = `${info.expansions.toLocaleString()} imagined transitions · ${info.depth} ply · value ${info.value.toFixed(3)}${invalid}`;
  }

  // ---------------------------------------------------------------- the board

  function animateStone(cell) {
    const now = performance.now();
    nextFall = Math.max(now, nextFall);
    falling.push({ cell, start: nextFall, duration: DROP_MS });
    nextFall += GAP_MS;
  }

  function drawBoard(now = performance.now()) {
    const rect = boardCanvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    if (!rect.width || !rect.height) return;
    const w = Math.max(1, Math.round(rect.width * dpr)), h = Math.max(1, Math.round(rect.height * dpr));
    if (boardCanvas.width !== w || boardCanvas.height !== h) { boardCanvas.width = w; boardCanvas.height = h; }
    const ctx = boardCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = rect.width, H = rect.height;
    ctx.clearRect(0, 0, W, H);
    const strip = Math.min(48, H * 0.13);              // the band under the board the messages live in
    const pad = Math.max(6, Math.min(W, H) * 0.026);
    const cell = Math.min((W - 2 * pad) / config.cols, (H - strip - 2 * pad) / config.rows);
    const boardW = cell * config.cols, boardH = cell * config.rows;
    const x0 = (W - boardW) / 2, y0 = Math.max(pad, (H - strip - boardH) / 2);
    const radius = cell * 0.39, corner = Math.min(14, cell * 0.3);
    boardFrame = { x: x0, y: y0, cell, radius };
    const board = boardNow(), legal = legalNow();
    const centre = (row, col) => [x0 + (col + 0.5) * cell, y0 + (config.rows - 1 - row + 0.5) * cell];
    const plate = (context) => { context.beginPath(); context.roundRect(x0 - pad * 0.5, y0 - pad * 0.5, boardW + pad, boardH + pad, corner); };
    // the stones, drawn behind the front plate so a falling one shows through the holes it passes
    ctx.save();
    ctx.beginPath();
    ctx.rect(x0 - pad * 0.5, y0 - pad * 0.5 - cell * 1.4, boardW + pad, boardH + pad + cell * 1.4);
    ctx.clip();
    ctx.fillStyle = "#05090e";
    plate(ctx);
    ctx.fill();
    for (let row = 0; row < config.rows; row++) {
      for (let col = 0; col < config.cols; col++) {
        const index = row * config.cols + col, value = board.cells[index];
        if (!value) continue;
        const wait = falling.find((f) => f.cell === index);
        let dy = 0;
        if (wait) {
          const t = (now - wait.start) / wait.duration;
          if (t < 0) continue;                          // this stone has not been released yet
          if (t < 1) { const eased = 1 - (1 - t) * (1 - t); dy = -(config.rows - row + 0.7) * cell * (1 - eased); }
        }
        const [cx, cy] = centre(row, col);
        stone(ctx, cx, cy + dy, radius, colourOf(value), 1.0);
        if (index === lastCell && dy === 0) { ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2); ctx.strokeStyle = "rgba(255,255,255,0.55)"; ctx.lineWidth = 1.6; ctx.stroke(); }
      }
    }
    ctx.restore();
    // the front plate: the board with its holes cut out
    ctx.save();
    ctx.beginPath();
    ctx.roundRect(x0 - pad * 0.5, y0 - pad * 0.5, boardW + pad, boardH + pad, corner);
    for (let row = 0; row < config.rows; row++) {
      for (let col = 0; col < config.cols; col++) { const [cx, cy] = centre(row, col); ctx.moveTo(cx + radius, cy); ctx.arc(cx, cy, radius, 0, Math.PI * 2); }
    }
    const ground = ctx.createLinearGradient(0, y0, 0, y0 + boardH);
    ground.addColorStop(0, "#12293b"); ground.addColorStop(1, "#0a1826");
    ctx.fillStyle = ground;
    ctx.fill("evenodd");
    ctx.restore();
    plate(ctx);
    ctx.strokeStyle = "rgba(96,165,250,0.22)"; ctx.lineWidth = 1; ctx.stroke();
    for (let row = 0; row < config.rows; row++) {
      for (let col = 0; col < config.cols; col++) {
        const [cx, cy] = centre(row, col);
        ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2);
        ctx.strokeStyle = "rgba(4,9,14,0.9)"; ctx.lineWidth = 1.2; ctx.stroke();
      }
    }
    // the column under the pointer: the plate lit, and the stone that would land there
    if (hover !== null && !over && !thinking && legal.includes(hover)) {
      ctx.fillStyle = "rgba(143,163,184,0.06)";
      ctx.fillRect(x0 + hover * cell, y0 - pad * 0.5, cell, boardH + pad);
      const row = board.heights[hover];
      if (row < config.rows) {
        const [cx, cy] = centre(row, hover);
        stone(ctx, cx, cy, radius * 0.92, human === "red" ? RED : YELLOW, 0.26);
        ctx.beginPath(); ctx.arc(cx, cy, radius * 0.92, 0, Math.PI * 2);
        ctx.setLineDash([3, 4]); ctx.lineWidth = 1.2; ctx.strokeStyle = shade(human === "red" ? RED : YELLOW, 0.5); ctx.stroke(); ctx.setLineDash([]);
      }
    }
    // the line that ended the game
    if (winningLine && winningLine.length) {
      const first = centre(Math.floor(winningLine[0] / config.cols), winningLine[0] % config.cols);
      const last = centre(Math.floor(winningLine[winningLine.length - 1] / config.cols), winningLine[winningLine.length - 1] % config.cols);
      ctx.beginPath(); ctx.moveTo(first[0], first[1]); ctx.lineTo(last[0], last[1]);
      ctx.strokeStyle = "rgba(255,255,255,0.9)"; ctx.lineWidth = Math.max(3, cell * 0.07); ctx.lineCap = "round"; ctx.stroke();
    }
    // the column the search chose, marked over the board
    if (lastSearch && !over) {
      const x = x0 + (lastSearch.column + 0.5) * cell, y = y0 - pad * 0.5 - 3;
      ctx.beginPath(); ctx.moveTo(x - cell * 0.13, y - 7); ctx.lineTo(x + cell * 0.13, y - 7); ctx.lineTo(x, y); ctx.closePath();
      ctx.fillStyle = `rgb(${ACCENT.join(",")})`; ctx.fill();
    }
    while (falling.length && now - falling[0].start > falling[0].duration) falling.shift();
  }

  /** One stone, lit from the upper left. */
  function stone(ctx, cx, cy, radius, colour, alpha) {
    const gradient = ctx.createRadialGradient(cx - radius * 0.35, cy - radius * 0.4, radius * 0.1, cx, cy, radius);
    gradient.addColorStop(0, `rgba(${Math.min(255, colour[0] + 30)},${Math.min(255, colour[1] + 30)},${Math.min(255, colour[2] + 30)},${alpha})`);
    gradient.addColorStop(1, `rgba(${(colour[0] * 0.66) | 0},${(colour[1] * 0.66) | 0},${(colour[2] * 0.66) | 0},${alpha})`);
    ctx.beginPath(); ctx.arc(cx, cy, radius, 0, Math.PI * 2);
    ctx.fillStyle = gradient; ctx.fill();
  }

  boardCanvas.addEventListener("pointermove", (event) => {
    const rect = boardCanvas.getBoundingClientRect();
    const col = Math.floor((event.clientX - rect.left - boardFrame.x) / boardFrame.cell);
    hover = col >= 0 && col < config.cols ? col : null;
  });
  boardCanvas.addEventListener("pointerleave", () => { hover = null; });
  boardCanvas.addEventListener("pointerdown", (event) => {
    const rect = boardCanvas.getBoundingClientRect();
    const col = Math.floor((event.clientX - rect.left - boardFrame.x) / boardFrame.cell);
    if (col < 0 || col >= config.cols) return;
    if (over) { flash("that game is over: press New game"); return; }
    if (!drop(col)) flash("that column is full");
  });

  // ---------------------------------------------------------------- playback of the brain

  function applyRecords(item) {
    const view = views[item.which];
    if (!view) return;
    if (item.code) { view.code({ index: item.code.index, values: item.code.values }, "decide"); view.setReading(item.reading); }
    for (const code of item.imagine) view.code({ index: code.index, values: code.values, batch: 1 }, "imagine");
    if (item.write) {
      view.write(item.write);
      setPhase(`${item.which === "drop" ? "drop" : "line"} records: ${item.write.written} written into ${item.write.index.length} cells`, "learn");
      currentPhase = null;
    }
    $(item.which === "drop" ? "dropReads" : "lineReads").innerHTML = readsHTML(item.reads, fields[item.which], item.write);
  }

  function flashWeights(item) {
    if (item.weights) {
      const agent = brain.agent, heat = new Float32Array(scan.n);
      let top = 1e-12;
      for (let e = 0; e < agent.E; e++) { const d = Math.abs(item.weights[e] - previousWeights[e]); heat[agent.pre[e]] += d; heat[agent.post[e]] += d; }
      for (let i = 0; i < scan.n; i++) if (heat[i] > top) top = heat[i];
      for (let i = 0; i < scan.n; i++) heat[i] = Math.sqrt(heat[i] / top);
      previousWeights = item.weights;
      scan.setWeights(item.weights);
      const scale = scan.scale;
      scan.show(lastS, heat, { potential: lastV, draw: false });
      scan.scale = scale;
      currentPhase = null;
    }
    setPhase(item.rejected ? `${item.phase} update rejected (phases did not converge)` : item.phase === "world" ? `world repair moved ${item.changed.toLocaleString()} synapses` : item.phase === "credit" ? (item.changed ? `credit moved ${item.changed.toLocaleString()} synapses` : `critic · dopamine ${item.dopamine.toFixed(4)}`) : item.phase, "learn");
  }

  function drain(limit) {
    let steps = 0;
    while (queue.length && (limit === 0 || steps < limit)) {
      const item = queue.shift();
      if (item.kind === "step") {
        queuedSteps -= 1;
        if (item.phase !== currentPhase) { currentPhase = item.phase; scan.reset(); setPhase((PHASE_TEXT[item.phase] || item.phase) + (item.batch > 1 ? ` · batch ${item.batch}` : ""), ""); }
        scan.step(item.s, { potential: item.v, draw: false });
        lastS = item.s; lastV = item.v; steps += 1;
        lastStepText = `${item.phase} step ${item.step} · largest movement ${item.moved.toExponential(2)} · ${queue.length} frames queued`;
      } else if (item.kind === "phase") ledgerPhase = item;
      else if (item.kind === "learn") flashWeights(item);
      else if (item.kind === "records") applyRecords(item);
    }
    if (steps) updateReadouts();
    return steps;
  }

  function updateReadouts() {
    const agent = brain.agent, d = stats.dopamine;
    const half = Math.min(50, (Math.abs(d) / 0.25) * 50);
    const fill = $("dopamineFill");
    fill.className = "fill" + (d < 0 ? " negative" : "");
    fill.style.left = d < 0 ? `${50 - half}%` : "50%";
    fill.style.width = `${half}%`;
    $("critic").innerHTML = rows([["clipped TD", d.toFixed(4)], ["critic bias", agent.bCritic.toFixed(4)], ["controller", stats.controller]]);
    const lp = ledgerPhase || agent.lastPhase, L = agent.ledger.toDict();
    const unconverged = Object.values(L.unconverged_phases).reduce((a, b) => a + b, 0);
    $("ledger").innerHTML = rows([
      ["phase", lp ? `${lp.phase} · ${lp.steps} ${lp.converged ? "✓" : "✗"}` : "—"],
      ["rejected", `${L.rejected_updates} · ${unconverged} unconverged`],
      ["transitions", `${brain.counts.transitions.toLocaleString()}`],
      ["imagined", `${brain.planner.nodes.toLocaleString()} boards`],
    ]);
  }

  const rows = (pairs) => pairs.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");

  function setPhase(text, cls) { phaseText = text; phaseClass = cls; }

  function showPhase() {
    const el = $("phase");
    if (el.textContent !== phaseText) el.textContent = phaseText;
    if (el.className !== `phase ${phaseClass}`.trim()) el.className = `phase ${phaseClass}`.trim();
    if ($("status").textContent !== lastStepText) $("status").textContent = lastStepText;
  }

  // ---------------------------------------------------------------- the receipt

  function showReceipt() {
    if (!MANIFEST || !MANIFEST.receipt) { $("receipt").hidden = true; return; }
    const r = MANIFEST.receipt, pct = (x) => `${(100 * x).toFixed(1)}%`;
    const facts = [
      ["Tactical suite", pct(r.tactical.success), `${r.tactical.cases.toLocaleString()} held-out positions with one right column: win now, block now, do not give the win away`],
      ["Against one-ply", r.paired.one_ply.toFixed(3), `paired games against an opponent that takes a win and blocks one; 0.5 is even, 1 is every game won`],
      ["Against minimax 256", r.paired.minimax_256.toFixed(3), `the same schedule against a search of 256 nodes with the true rules`],
      ["Planning gain", `+${r.planning_gain.toFixed(3)}`, `the same records without search score this much less against one-ply`],
      ["Decision latency", `${r.latency_ms.p95.toFixed(0)} ms`, `p95 over the life in Python (p50 ${r.latency_ms.p50.toFixed(0)} ms); the page measures its own above`],
      ["Next board exact", pct(r.validity), `held-out single moves whose next board the drop records compose exactly`],
    ];
    const source = `${r.run_id} · seed ${r.seed} · ${r.games.total.toLocaleString()} games lived · ${r.decisions.toLocaleString()} decisions · ${r.transitions.toLocaleString()} moves watched · ${r.imagined.toLocaleString()} boards imagined · cadence ${String(r.cadence).slice(0, 12)} · every gate passed${r.pilot ? " (pilot budgets)" : ""}`;
    $("receipt").innerHTML = facts.map(([name, value, note]) => `<div class="fact"><h4>${name}</h4><b>${value}</b><span>${note}</span></div>`).join("") + `<div class="source">${source}</div>`;
  }

  // ---------------------------------------------------------------- checkpoints

  function buildCheckpoints() {
    if (!checkpoints.length) return;
    const select = $("checkpoint");
    select.innerHTML = checkpoints.map((c) => `<option value="${c.id}">${c.label}${c.id === MANIFEST.inlined ? " · on this page" : ""}</option>`).join("");
    select.value = checkpoint.id;
    select.onchange = () => selectCheckpoint(select.value);
    $("checkpointBox").hidden = false;
  }

  async function selectCheckpoint(id) {
    const select = $("checkpoint");
    if (id === checkpoint.id) return;
    const entry = checkpoints.find((c) => c.id === id);
    if (!entry) return;
    if (id === (MANIFEST ? MANIFEST.inlined : "inlined")) { installLife(FIRST, FIRST.label, id); flash("this page's brain"); buildCheckpoints(); return; }
    select.disabled = true;
    flash(`loading ${entry.label}…`, 60000);
    try {
      let payload = cache.get(id);
      if (!payload) {
        const response = await fetch(new URL(entry.file, document.baseURI));
        if (!response.ok) throw Error(`${entry.file}: ${response.status}`);
        payload = await response.json();
        cache.set(id, payload);
      }
      installLife(payload, entry.label, id);
      flash(`${entry.label}: a new game against this brain`);
    } catch (error) {
      select.value = checkpoint.id;
      flash(`could not load that brain: ${error.message}`, 6000);
      console.error(error);
    } finally { select.disabled = false; buildCheckpoints(); }
  }

  // ---------------------------------------------------------------- controls and the loop

  $("newGame").onclick = () => newGame();
  $("side").onchange = () => { human = $("side").value; buildLegend(payloadNow); newGame(); };
  $("learning").onchange = () => { brain.setLearning($("learning").value === "on"); $("note").textContent = brain.learning ? "every move is written into the records" : "the records are frozen: the brain plays and writes nothing"; flash(brain.learning ? "learning on" : "learning frozen"); };
  $("play").onclick = () => setRunning(!running);
  $("step").onclick = () => { setRunning(false); drain(1); };
  $("fit").onclick = () => scan.fit();
  $("view").onchange = () => { scan.options.mode = $("view").value; scan.draw(); };

  function setRunning(value) { running = value; $("play").textContent = running ? "Pause" : "Play"; }

  function frame(now) {
    const dt = Math.min(0.5, Math.max(1 / 240, (now - lastFrame) / 1000));
    lastFrame = now;
    if (queue.length) {
      // the brain view plays back what the last moves queued, over about a second of wall time,
      // however fast this machine draws
      if (queue.length > 4000) drain(0);
      else if (running) drain(Math.max(1, Math.ceil(queue.length / Math.max(2, 1.2 / dt))));
    }
    if (flashUntil && now > flashUntil && !thinking && !over) { flashUntil = 0; updateBoardState(); }
    showPhase();
    drawBoard(now);
    scan.draw(now);
    if (views) { views.drop.draw(); views.lines.draw(); }
    requestAnimationFrame(frame);
  }

  addEventListener("resize", () => { scan.draw(); drawBoard(); if (views) { views.drop.draw(); views.lines.draw(); } });

  // ---------------------------------------------------------------- start

  installLife(FIRST, checkpoint.label, checkpoint.id);
  buildCheckpoints();
  showReceipt();
  setRunning(PAGE_OPTIONS.autoplay !== false);
  updateReadouts();
  requestAnimationFrame(frame);

  window.__page = {
    scan,
    get brain() { return brain; },
    get world() { return world; },
    get views() { return views; },
    get stats() {
      return {
        ...stats, checkpoint: checkpoint.id, label: checkpoint.label, learning: brain.learning, human, over, started,
        queued: queue.length, score: { ...score }, transitions: brain.counts.transitions, extended: brain.extended,
        records: { drop: brain.drop.records.writes, lines: brain.lines.records.writes }, imagined: brain.planner.nodes, expansion: brain.expansion,
        experience: brain.block.counts.transitions, renderer: scan.snapshot(),
      };
    },
    /** The board as it stands: the cells, whose move it is and what the legal columns are. */
    get board() {
      return { cells: Array.from(boardNow().cells), legal: legalNow(), moves: [...boardNow().moves], winner: over ? world.winner() : null, over, started, human, candidate: world.candidate };
    },
    /** What the last search saw: the columns, the value the records read for each and the value the search gave it. */
    get imagined() { return lastSearch === null ? null : { column: lastSearch.column, columns: [...lastSearch.columns], read: [...lastSearch.read], searched: [...lastSearch.searched], depth: lastSearch.depth, expansions: lastSearch.expansions, value: lastSearch.value, invalid: lastSearch.invalid, landings: lastSearch.landings.map((l) => [...l]) }; },
    checkpoints: checkpoints.map((c) => c.id),
    drop(column) { return drop(column); },
    /** One legal column, drawn without a generator of its own (the check plays this way). */
    randomColumn() { const legal = legalNow(); return legal.length ? legal[Math.floor(Math.random() * legal.length)] : null; },
    newGame(options = {}) { if (options.human) { human = options.human; $("side").value = human; buildLegend(payloadNow); } newGame(); return this.board; },
    selectCheckpoint,
    setLearning(on) { $("learning").value = on ? "on" : "off"; brain.setLearning(on); },
    drainAll() { drain(0); return queue.length; },
    pause() { setRunning(false); }, play() { setRunning(true); },
    /** Where a column sits on the screen, for pointer events. */
    toClient(column) { const rect = boardCanvas.getBoundingClientRect(); return [rect.left + boardFrame.x + (column + 0.5) * boardFrame.cell, rect.top + boardFrame.y + boardFrame.cell * 0.5]; },
    ready: true,
  };
  window.__brainScan = scan;
}

loadPage().then(main).catch((error) => { $("counts").textContent = `error: ${error.message}`; $("state").textContent = "failed to load"; console.error(error); });
