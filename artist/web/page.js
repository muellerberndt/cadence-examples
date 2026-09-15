// The S04 artist page: the canvas the artist draws on, with the whole brain beside it. A visitor
// draws a figure on the canvas or picks one of the target families; the artist starts from an
// empty canvas, chooses a stroke by imagining what its pen would leave, follows that stroke with
// torques chosen through the same records, lifts the pen between strokes and looks at its own
// canvas again every second decision. Every choice comes from ExperienceAgent.step through the
// two-level search of artist.js, and the outcome of every decision is written into the records
// the reading touched, so the brain keeps learning while it draws. Beside the canvas the whole
// brain runs through the standard BrainScan renderer and the records-cortex view: every settling
// step of every phase is queued by the engine's callbacks and played back in step with the body.
// The brain selector loads another point of this artist's life from the checkpoint manifest
// beside the page. Nothing is scripted: the drawing is only as good as the model it has learned.

import { BrainScan } from "../../web/brain_scan.js";
import { ExperienceAgent } from "../../web/engine.js";
import { RecordsView, pathwayNames, readingGroups } from "../../web/records_view.js";
import { ArtistBrain, CanvasWorld, PageRandom, canvasBody, canvasConfig, drawnTarget, familyDrawing, foregroundF1, handOfPixel, pixelOf, splitChoice, startAngles } from "./artist.js";

const $ = (id) => document.getElementById(id);
const PAGE_OPTIONS = { autoplay: true, ...(window.__ARTIST_PAGE__ || {}) };
const CHECKPOINT_FORMAT = "cadence-artist-checkpoints/1";
const PHASE_TEXT = { free: "free phase", free_post: "free phase after learning", predict: "predict (the chosen action's consequence)", imagine: "imagine (eighteen choices at once)", repair_free: "world repair · free", repair_plus: "world repair · nudged +β", repair_minus: "world repair · nudged −β", bootstrap: "bootstrap value (time limit)", score_plus: "actor score · +β", score_minus: "actor score · −β" };
const REGION_NOTE = { sensory: "the body, the pen and the two windows", goal: "the committed stroke", efference: "copy of the executed choice", workspace: "association", dynamics: "reads the efference copy", context: "trace of the workspace", motor: "torque and pen readout", prediction: "predicted consequences", perception: "perception", recall: "recall" };
const INK_COLOR = [143, 225, 157], TARGET_COLOR = [219, 230, 240], PLAN_COLOR = [96, 165, 250], MARK_COLOR = [237, 129, 182], ARM_COLOR = [110, 148, 184];
const VIEW = { cx: 0.0, cy: 0.48, half: 0.54 }; // the framed part of the body's plane, in arm lengths
const DEMO_FAMILIES = ["segment", "curve", "two_segments", "polygon"];
const COMMIT_DELAY = 900; // the window for adding another stroke before the artist is given the figure
const MIN_SKETCH = 6; // canvas pixels of stroke: shorter figures are not a target
const shade = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

async function loadSnapshot() {
  if (window.__SNAPSHOT__) return window.__SNAPSHOT__;
  const url = document.body.dataset.snapshot;
  const response = await fetch(url);
  if (!response.ok) throw Error(`cannot load ${url}: ${response.status}`);
  return response.json();
}

/**
 * The per-field record reads for this stage. `readsHTML` of web/records_view.js draws one track
 * per value, which is right for a two-value body field and unreadable for the 64 values of the
 * ink window, so the window is drawn here as the 8 by 8 patch it is; the body fields keep the
 * shared markup. Copied into the page because the shared module is not edited from a stage.
 */
function artistReadsHTML(prediction, fields, lastWrite, color) {
  if (!prediction) return `<div class="read muted">no read yet</div>`;
  const errors = new Map((lastWrite ? lastWrite.fields : []).map((f) => [f.name, f]));
  const c = color.join(",");
  return fields.map((f) => {
    const p = prediction[f.name];
    if (!p) return "";
    const e = errors.get(f.name);
    const err = e ? `<em title="the largest error of the last write at rate ${e.rate}">err ${e.error.toFixed(2)}</em>` : "";
    if (f.width > 8) {
      const side = Math.round(Math.sqrt(f.width));
      let top = 1e-9;
      for (const v of p) if (v > top) top = v;
      const cells = Array.from(p, (v, k) => `<i title="cell ${Math.floor(k / side)},${k % side}: ${v.toFixed(3)}" style="background:rgba(${c},${(0.06 + 0.94 * Math.min(1, Math.max(0, v / top))).toFixed(3)})"></i>`).join("");
      return `<div class="read"><span class="name" title="${f.name}: the ink the next decision leaves in the window around the pen">${f.name}</span><span class="patch" style="grid-template-columns:repeat(${side},6px)">${cells}</span><b title="the strongest cell of the window">${top.toFixed(2)}</b>${err}</div>`;
    }
    const parts = Array.from(p, (v) => { const t = Math.min(1, Math.max(0, (v - f.lo) / (f.hi - f.lo))); return `<span class="track" title="${f.lo} to ${f.hi}"><i style="left:${(t * 100).toFixed(1)}%;background:rgb(${c})"></i></span><b>${v.toFixed(3)}</b>`; }).join("");
    return `<div class="read"><span class="name" title="${f.name}">${f.name}</span>${parts}${err}</div>`;
  }).join("");
}

function main(FIRST) {
  const scan = new BrainScan($("scan"), FIRST.atlas, { labels: $("labels"), strip: $("strip"), labelTop: 12, montageRows: 5 });
  const queue = []; // settle frames and markers in the order the engine produced them
  let queuedSteps = 0;

  // -- the life, rebuilt whenever another checkpoint is loaded
  let snapshot = null, checkpoint = { id: "inlined", label: "this page's brain" };
  let agent = null, brain = null, records = null, world = null, geometry = null, stageConfig = null;
  let moment = null, decisionSeconds = 0.1;
  let predictionFields = [], lastPrediction = null, lastWrite = null;
  let restState = null, lastS = null, lastV = null, previousWeights = null;
  let stats = freshStats();

  // -- the page's own state
  let drawing = null, nextDrawing = null, switchRequested = false, drawingCount = 0, lastResult = null;
  let sketch = null, commitTimer = null, commitAt = 0, held = false;
  let demo = false, demoAt = 0, demoTimer = null;
  let running = PAGE_OPTIONS.autoplay !== false, pace = 1, due = 0, fps = 60, lastFrame = performance.now();
  let stageView = null, shownAt = 0, currentPhase = null, ledgerPhase = null, lastStepText = "", flashText = "", flashUntil = 0;
  let phaseText = "—", phaseClass = "", showPlan = true;
  const results = []; // every finished drawing: {n, family, chamfer, f1, decisions, brain}
  const rng = new PageRandom(0xa11ce);
  const checkpoints = [], snapshotCache = new Map();
  const stageCanvas = $("canvas");
  let frame2 = { cx: 0, cy: 0, scale: 1 };
  scan.onhover = (hit) => { if (hit) $("inspector").textContent = `${hit.region} · neuron ${hit.neuron}\nactivation ${hit.activation.toFixed(4)} · change ${hit.change.toExponential(2)}` + (hit.potential === null ? "" : ` · potential ${hit.potential.toFixed(4)}`); };

  function freshStats() {
    return { events: 0, decisions: 0, drawings: 0, finished: 0, reward: 0, chamfer: 1, f1: 0, prediction_mse: null, controller: "—", dopamine: 0, td_error: 0, compute_ms: 0, rejected: 0, learning: {}, last_decision: null };
  }

  // ---------------------------------------------------------------- the life

  /** A checkpoint exported without the page's extras borrows them from the inlined snapshot. */
  function fillExtras(snap) {
    snap.extra = snap.extra || {};
    for (const key of ["canvas", "intention", "intention_interval", "intention_gain", "planner", "body", "receipt"]) {
      if (snap.extra[key] === undefined || snap.extra[key] === null) snap.extra[key] = (FIRST.extra || {})[key];
    }
    return snap;
  }

  function installLife(snap, label, id) {
    fillExtras(snap);
    snapshot = snap;
    checkpoint = { id, label };
    queue.length = 0; queuedSteps = 0; currentPhase = null; ledgerPhase = null;
    agent = new ExperienceAgent(snap, {
      onSettleStep: (phase, s, v, meta) => { queue.push({ kind: "step", phase, s: Float32Array.from(s), v: Float32Array.from(v), batch: meta.batch, step: meta.step, moved: meta.moved }); queuedSteps += 1; },
      onPhase: (info) => queue.push({ kind: "phase", phase: info.phase, steps: info.steps, residual: info.residual, converged: info.converged, batch: info.batch }),
      onLearn: (info) => queue.push({ kind: "learn", phase: info.phase, changed: info.changed.length, rejected: info.rejected, dopamine: info.dopamine, td_error: info.td_error, weights: info.changed.length ? agent.effectiveWeights() : null }),
      onRecords: (info) => queue.push({ ...info, kind: "records", event: info.kind }),
    });
    agent.config.actor.epsilon = Number($("epsilon").value) || 0;
    predictionFields = agent.config.graph.prediction;
    lastPrediction = null; lastWrite = null;
    const extra = snap.extra;
    stageConfig = { canvas: extra.canvas, intention: extra.intention, planner: extra.planner, intention_interval: extra.intention_interval, intention_gain: extra.intention_gain };
    brain = new ArtistBrain(agent, stageConfig);
    geometry = brain.geometry;
    world = new CanvasWorld(canvasConfig(extra.canvas), canvasBody(extra.body, { gain: (extra.body && extra.body.gain) || 1.0 }), { seed: 0x5eed, lifeId: "web" });
    decisionSeconds = world.body.config.dt * world.body.config.substeps;
    // the brain in the scan and the records cortex beside it
    scan.setAtlas(snap.atlas);
    restState = Float32Array.from(agent.warm ? agent.warm.v : new Float64Array(agent.n), (v) => agent._act(v));
    lastV = Float32Array.from(agent.warm ? agent.warm.v : new Float64Array(agent.n));
    lastS = restState;
    scan.set(restState, { potential: lastV, draw: false });
    previousWeights = agent.effectiveWeights();
    scan.setWeights(previousWeights);
    scan.fit();
    installRecords(snap);
    buildLegend();
    stats = freshStats();
    moment = null; drawing = null; switchRequested = false;
    $("finishValue").textContent = geometry.finish.toFixed(3);
    $("capValue").textContent = String(geometry.max_decisions);
    showReceipt();
    updateFacts();
    updateKey();
    if (nextDrawing === null && lastResult !== null && lastResult.spec) nextDrawing = lastResult.spec; // the same figure, drawn by the brain just loaded
    if (nextDrawing !== null) { switchRequested = true; moment = beginDrawing(); }
    stageView = restingView();
    shownAt = -1e9;
    applyEvent(stageView);
    updateHud();
  }

  function installRecords(snap) {
    records = null;
    if (!agent.records) { $("recordsPanel").hidden = true; return; }
    const cfg = agent.config.records, rc = snap.records || {};
    records = new RecordsView($("records"), { granules: agent.records.granules, active: cfg.active, inputs: agent.reading.length, groups: readingGroups(snap), pathways: pathwayNames(snap), color: rc.color || undefined, rate: cfg.rate, rewardRate: cfg.reward_rate, habituation: cfg.habituation });
    $("recordsPanel").hidden = false;
    $("recordsInfo").textContent = `the world model · ${agent.records.granules.toLocaleString()} granules, ${cfg.active} active per reading · ${agent.records.parameters().toLocaleString()} records in ${predictionFields.length} fields · the executed reading's code in blue, writes in orange, the search's reads in grey`;
    $("recordReads").innerHTML = artistReadsHTML(null, predictionFields, null, records.spec.color);
  }

  function buildLegend() {
    const items = snapshot.atlas.regions.map((r) => `<span title="${r.name}"><i style="background:rgb(${r.color.join(",")})"></i><b>${r.label ?? r.name}</b><em>${r.count}</em>${REGION_NOTE[r.name] ? ` ${REGION_NOTE[r.name]}` : ""}</span>`);
    if (records) items.push(`<span title="the world model"><i style="background:rgb(${records.spec.color.join(",")})"></i><b>records cortex</b><em>${agent.records.granules.toLocaleString()}</em> the world model</span>`);
    $("legend").innerHTML = items.join("");
  }

  function updateFacts() {
    const lived = (snapshot.ledger && snapshot.ledger.real_transitions) || 0;
    $("counts").textContent = `${checkpoint.label} · ${lived.toLocaleString()} decisions lived before this page\n${agent.n.toLocaleString()} neurons · ${agent.E.toLocaleString()} synapses${agent.records ? ` · ${agent.records.granules.toLocaleString()} granules, ${agent.records.parameters().toLocaleString()} records` : ""}`;
  }

  /** The numbers of the run the page's brains come from, as the receipt records them. */
  function showReceipt() {
    const r = (snapshot.extra && snapshot.extra.receipt) || null;
    if (!r) { $("receipt").innerHTML = ""; $("receiptNote").textContent = "this page was built from one exported snapshot, so it carries no run receipt."; return; }
    const pct = (x) => (x === null || x === undefined ? "—" : x.toFixed(2));
    const chips = [
      ["held-out F1", pct(r.f1), true], ["Chamfer", r.chamfer === null || r.chamfer === undefined ? "—" : r.chamfer.toFixed(4), true],
      ["frozen", pct(r.born_frozen)], ["permuted", pct(r.corrupted_model)], ["MLP", pct(r.online_mlp)],
      ["p95", r.latency_p95_ms === null || r.latency_p95_ms === undefined ? "—" : `${r.latency_p95_ms.toFixed(0)} ms`],
    ];
    $("receipt").innerHTML = chips.map(([k, v, lead]) => `<span class="${lead ? "lead" : ""}"><span>${k}</span><b>${v}</b></span>`).join("");
    $("receiptNote").textContent = `${r.run_id} on ${r.split} seed ${r.seed}, ${(r.decisions || 0).toLocaleString()} decisions lived. On ${r.drawings || 0} held-out drawings the artist reaches F1 ${pct(r.f1)} and Chamfer ${r.chamfer === null || r.chamfer === undefined ? "—" : r.chamfer.toFixed(4)}, against ${pct(r.born_frozen)} for the same architecture frozen from birth, ${pct(r.corrupted_model)} with its records permuted across cells, ${pct(r.single_intention)} with replanning removed, ${pct(r.online_mlp)} for an online MLP with a replay ring behind the same search, ${pct(r.oracle)} for the supplied path oracle and ${pct(r.random)} for random scribbling. A decision takes ${r.latency_p95_ms === null || r.latency_p95_ms === undefined ? "—" : r.latency_p95_ms.toFixed(0)} ms at the p95 in Python, inside the 100 ms a decision has.`;
  }

  // ---------------------------------------------------------------- drawings

  function familyTarget(family) {
    const spec = familyDrawing(family, rng, geometry);
    spec.source = "family";
    return spec;
  }

  /** Draw this figure: the running drawing is abandoned and the next one starts at the next event. */
  function requestDrawing(spec, { keepDemo = false } = {}) {
    if (!keepDemo) stopDemo();
    nextDrawing = spec;
    switchRequested = true;
    $("again").disabled = false;
    if (moment === null) moment = beginDrawing();
    updateHud();
  }

  function beginDrawing() {
    const spec = nextDrawing;
    nextDrawing = null; switchRequested = false;
    if (spec === null) { drawing = null; return null; }
    brain.detach(); // between drawings the life drops a decision without its outcome and starts a fresh stream
    const m = world.reset(spec);
    brain.begin(world.view());
    drawing = { spec, n: ++drawingCount, brain: checkpoint.label, decisions: 0, start: world.discrepancy, cancelled: false };
    stats.drawings += 1;
    return m;
  }

  function finishDrawing(d, ended) {
    if (d === null || d.decisions === 0 || ended === "cancelled") return; // an abandoned drawing is not a result
    const measure = world.measure();
    lastResult = { n: d.n, family: d.spec.family, name: d.spec.name, brain: d.brain, decisions: d.decisions, chamfer: measure.chamfer, f1: measure.f1, precision: measure.precision, recall: measure.recall, ink: measure.ink, strokes: measure.strokes, ended, spec: d.spec };
    results.push(lastResult);
    if (results.length > 24) results.shift();
    stats.finished += 1;
  }

  function drawingInfo() {
    if (drawing === null) return null;
    const measure = { chamfer: world.discrepancy, f1: stats.f1 };
    return { n: drawing.n, family: drawing.spec.family, name: drawing.spec.name, brain: drawing.brain, decisions: drawing.decisions, chamfer: measure.chamfer, f1: measure.f1, progress: Math.min(1, drawing.decisions / geometry.max_decisions), strokes: world.strokes };
  }

  function startDemo() { demo = true; demoAt = 0; requestDrawing(familyTarget(DEMO_FAMILIES[0]), { keepDemo: true }); }
  function stopDemo() { demo = false; if (demoTimer !== null) { clearTimeout(demoTimer); demoTimer = null; } }

  function afterDrawing() {
    if (!demo) return;
    demoTimer = setTimeout(() => {
      demoTimer = null;
      if (!demo) return;
      demoAt = (demoAt + 1) % DEMO_FAMILIES.length;
      requestDrawing(familyTarget(DEMO_FAMILIES[demoAt]), { keepDemo: true });
    }, 2200);
  }

  // ---------------------------------------------------------------- one event

  function wantsEvents() { return !held && (moment !== null || nextDrawing !== null) && (drawing !== null || nextDrawing !== null); }

  function computeEvent() {
    if (switchRequested && nextDrawing !== null) { finishDrawing(drawing, "cancelled"); moment = beginDrawing(); }
    if (moment === null) return false;
    const t0 = performance.now();
    const before = world.canvas;
    const decision = brain.step(moment, world.view());
    const report = agent.lastReport;
    stats.events += 1;
    const scores = report.scores[0];
    if (scores) { const values = Object.values(scores).filter(Number.isFinite); stats.prediction_mse = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null; }
    stats.learning = report.learning;
    if (report.learning.dopamine !== undefined) { stats.dopamine = report.learning.dopamine; stats.td_error = report.learning.td_error; }
    stats.rejected = agent.ledger.rejected_updates;
    stats.last_decision = decision === null ? null : { decision_id: decision.decision_id, action: decision.action, controller: decision.controller, event_id: decision.event_id, budget: { ...decision.budget } };
    let trace = null, penDown = 0, stroke = null, ended = null;
    const intention = { ...brain.intention, endpoint: Array.from(brain.intention.endpoint), direction: Array.from(brain.intention.direction), replanned: brain.replanned, from: Array.from(brain.penPixel) };
    const imagined = decision !== null && brain.planner.lastImagined ? brain.planner.lastImagined : null;
    if (decision === null) {
      ended = moment.terminated ? "finished" : "stopped";
      finishDrawing(drawing, ended);
      drawing = null;
      moment = beginDrawing();
      stats.reward = 0; stats.controller = "—";
      if (moment === null) afterDrawing();
    } else {
      stats.decisions += 1;
      stats.controller = decision.controller;
      penDown = splitChoice(decision.action)[1];
      moment = world.act(decision.decision_id, decision.action);
      stroke = world.lastStroke;
      trace = poseTrace(world.body.trace);
      stats.reward = moment.reward;
      drawing.decisions += 1;
    }
    const measure = world.measure();
    stats.chamfer = measure.chamfer; stats.f1 = measure.f1;
    stats.compute_ms = performance.now() - t0;
    queue.push({
      kind: "event", canvas: world.canvas, before, target: world.target, stroke, trace, penDown, penState: world.penState,
      pose: { elbow: Array.from(world.body.elbow), hand: Array.from(world.body.hand) }, intention, imagined: showPlan ? imagined : null,
      episode: world.episode, tick: world.tick, ended, measure, sketchless: true,
      stats: { ...stats, learning: { ...stats.learning } }, ledger: agent.ledger.toDict(), parameter_version: agent.parameterVersion,
      epsilon: agent.config.actor.epsilon, drawing: drawingInfo(), spec: drawing ? drawing.spec : lastResult ? lastResult.spec : null,
    });
    return true;
  }

  function poseTrace(thetas) {
    if (!thetas) return null;
    const [l1, l2] = world.body.config.lengths, out = [];
    for (let k = 0; k < thetas.length; k += 2) {
      const a = thetas[k], b = thetas[k + 1], ex = l1 * Math.cos(a), ey = l1 * Math.sin(a);
      out.push([ex, ey, ex + l2 * Math.cos(a + b), ey + l2 * Math.sin(a + b)]);
    }
    return out;
  }

  function restingView() {
    return {
      kind: "event", canvas: world.canvas, before: world.canvas, target: world.target, stroke: null, trace: null, penDown: 0, penState: world.penState,
      pose: { elbow: Array.from(world.body.elbow), hand: Array.from(world.body.hand) }, intention: null, imagined: null,
      episode: world.episode, tick: world.tick, ended: null, measure: world.measure(), stats: { ...stats },
      ledger: agent.ledger.toDict(), parameter_version: agent.parameterVersion, epsilon: agent.config.actor.epsilon,
      drawing: drawingInfo(), spec: drawing ? drawing.spec : null,
    };
  }

  // ---------------------------------------------------------------- playback

  function applyRecords(item) {
    if (!records) return;
    if (item.event === "code") {
      records.code(item, "decide");
      records.setReading(item.reading, item.blockNorm);
      lastPrediction = item.prediction;
      $("recordReads").innerHTML = artistReadsHTML(lastPrediction, predictionFields, lastWrite, records.spec.color);
    } else if (item.event === "imagine") records.code(item, "imagine");
    else if (item.event === "write") {
      records.write(item);
      lastWrite = item;
      $("recordReads").innerHTML = artistReadsHTML(lastPrediction, predictionFields, lastWrite, records.spec.color);
      setPhase(`records: ${item.written} field${item.written === 1 ? "" : "s"} written into ${item.index.length} cells`, "learn");
      currentPhase = null;
    }
  }

  function flashWeights(item) {
    if (item.weights) {
      const heat = new Float32Array(scan.n);
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
    setPhase(item.rejected ? `${item.phase} update rejected (phases did not converge)` : item.phase === "world" ? `world repair moved ${item.changed.toLocaleString()} synapses` : item.phase === "credit" ? (item.changed ? `motor credit moved ${item.changed.toLocaleString()} synapses` : `critic · dopamine ${item.dopamine.toFixed(4)}`) : item.phase, "learn");
  }

  function applyEvent(item) {
    stageView = item;
    shownAt = performance.now();
    const s = item.stats, d = s.dopamine;
    const half = Math.min(50, (Math.abs(d) / 0.25) * 50);
    const fill = $("dopamineFill");
    fill.className = "fill" + (d < 0 ? " negative" : "");
    fill.style.left = d < 0 ? `${50 - half}%` : "50%";
    fill.style.width = `${half}%`;
    const worldRow = agent.records ? ["written now", s.learning.records_written === undefined ? "—" : `${s.learning.records_written} fields`] : ["world step", s.learning.world_scale_step === undefined ? "—" : s.learning.world_scale_step.toExponential(2)];
    $("critic").innerHTML = rows([["clipped TD", d.toFixed(4)], ["critic bias", agent.bCritic.toFixed(4)], worldRow]);
    const lp = ledgerPhase || agent.lastPhase, L = item.ledger;
    const unconverged = Object.values(L.unconverged_phases).reduce((a, b) => a + b, 0);
    $("ledger").innerHTML = rows([
      ["phase", lp ? `${lp.phase} · ${lp.steps} ${lp.converged ? "✓" : "✗"}` : "—"],
      ["rejected", `${L.rejected_updates} · ${unconverged} unconverged`],
      agent.records ? ["records", `${L.record_writes.toLocaleString()} writes`] : ["ring", "—"],
      ["imagined", `${L.imagined.toLocaleString()} reads`],
    ]);
    const info = item.drawing;
    const b = (s.last_decision && s.last_decision.budget) || null;
    $("counters").innerHTML = rows([
      ["drawing", info ? `${info.n} · ${info.name}` : "—"],
      ["decisions", `${item.tick}/${geometry.max_decisions} · ${world.strokes} strokes`],
      ["choice", `${s.controller} · ε ${item.epsilon.toFixed(2)} · pen ${item.penState ? "down" : "up"}`],
      ["imagined", b ? `${b.expansions || 0} choices · ${b.intentions || 0} strokes` : "—"],
      ["compute", `${s.compute_ms.toFixed(0)} ms per decision`],
    ]);
    updateHud();
  }

  const rows = (pairs) => pairs.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");

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
      else if (item.kind === "event") applyEvent(item);
    }
    return steps;
  }

  function setPhase(text, cls) { phaseText = text; phaseClass = cls; }

  function showPhase() {
    const el = $("phase");
    if (el.textContent !== phaseText) el.textContent = phaseText;
    if (el.className !== `phase ${phaseClass}`.trim()) el.className = `phase ${phaseClass}`.trim();
    if ($("status").textContent !== lastStepText) $("status").textContent = lastStepText;
  }

  // ---------------------------------------------------------------- the canvas view

  function drawStage(now = performance.now()) {
    const rect = stageCanvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    if (!rect.width || !rect.height) return;
    const w = Math.max(1, Math.round(rect.width * dpr)), h = Math.max(1, Math.round(rect.height * dpr));
    if (stageCanvas.width !== w || stageCanvas.height !== h) { stageCanvas.width = w; stageCanvas.height = h; }
    const ctx = stageCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = rect.width, H = rect.height, view = stageView || restingView();
    const scale = Math.min(W, H) / (2 * VIEW.half);
    frame2 = { cx: W / 2, cy: H / 2, scale };
    const X = (x) => frame2.cx + (x - VIEW.cx) * scale, Y = (y) => frame2.cy - (y - VIEW.cy) * scale;
    const px = Math.max(0.7, Math.min(1.8, Math.min(W, H) / 420));
    const cell = geometry.pixel * scale;
    const seconds = decisionSeconds / Math.max(0.01, pace);
    const u = Math.max(0, Math.min(1, (now - shownAt) / (seconds * 1000)));
    const at = (row, column) => { const p = handOfPixel([row, column], geometry); return [X(p[0]), Y(p[1])]; };
    const box = (row, column) => { const [cx, cy] = at(row, column); return [cx - cell / 2, cy - cell / 2]; };
    // the ground
    const ground = ctx.createRadialGradient(frame2.cx, frame2.cy, 0, frame2.cx, frame2.cy, Math.hypot(W, H) / 2);
    ground.addColorStop(0, "#0b1621"); ground.addColorStop(1, "#04080c");
    ctx.fillStyle = ground; ctx.fillRect(0, 0, W, H);
    // the canvas: its border and a grid every eight pixels
    const size = geometry.size;
    const [left, top] = box(-0.5, -0.5), [right, bottom] = box(size - 0.5, size - 0.5);
    ctx.fillStyle = "#08121a"; ctx.fillRect(left, top, right - left, bottom - top);
    ctx.strokeStyle = "rgba(143,163,184,0.10)"; ctx.lineWidth = 1;
    for (let k = 8; k < size; k += 8) {
      const [gx, gy] = box(k - 0.5, k - 0.5);
      ctx.beginPath(); ctx.moveTo(left, gy); ctx.lineTo(right, gy); ctx.moveTo(gx, top); ctx.lineTo(gx, bottom); ctx.stroke();
    }
    ctx.strokeStyle = "rgba(143,163,184,0.28)"; ctx.strokeRect(left, top, right - left, bottom - top);
    // the target
    const target = view.target;
    ctx.fillStyle = shade(TARGET_COLOR, 0.30);
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) if (target[r * size + c]) { const [bx, by] = box(r, c); ctx.fillRect(bx, by, cell, cell); }
    // what the fast level expects its chosen action to leave
    if (view.imagined && view.imagined.candidates) {
      const chosen = view.imagined.candidates[view.imagined.chosen];
      if (chosen) for (const [r, c, weight] of chosen.points) { const [bx, by] = box(r, c); ctx.fillStyle = shade(PLAN_COLOR, 0.12 + 0.5 * Math.min(1, weight)); ctx.fillRect(bx + cell * 0.2, by + cell * 0.2, cell * 0.6, cell * 0.6); }
    }
    // the ink, with the pixels of the decision being shown coming up over it
    const canvas = view.canvas, fresh = new Set();
    if (view.stroke) for (let k = 0; k < view.stroke.length; k += 2) fresh.add(view.stroke[k] * size + view.stroke[k + 1]);
    for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) {
      const index = r * size + c;
      if (!canvas[index]) continue;
      const [bx, by] = box(r, c);
      const rising = fresh.has(index) ? u : 1;
      if (rising <= 0) continue;
      ctx.fillStyle = shade(INK_COLOR, 0.55 + 0.45 * rising);
      ctx.fillRect(bx, by, cell, cell);
      if (rising < 1) { ctx.strokeStyle = shade(INK_COLOR, 0.8); ctx.lineWidth = px; ctx.strokeRect(bx, by, cell, cell); }
    }
    // the committed stroke: from the pen to the endpoint, dashed when the pen travels up
    const intention = view.intention;
    if (showPlan && intention && intention.valid) {
      const [fx, fy] = at(intention.from[0], intention.from[1]);
      const [tx, ty] = at(intention.endpoint[0], intention.endpoint[1]);
      ctx.strokeStyle = shade(MARK_COLOR, intention.pen ? 0.85 : 0.4);
      ctx.lineWidth = 1.6 * px;
      ctx.setLineDash(intention.pen ? [] : [3, 4]);
      ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.setLineDash([]);
      ctx.fillStyle = shade(MARK_COLOR, 0.9);
      ctx.beginPath(); ctx.arc(tx, ty, 2.6 * px, 0, Math.PI * 2); ctx.fill();
    }
    // the visitor's strokes, over everything the artist has done
    if (sketch) {
      ctx.lineCap = "round"; ctx.lineJoin = "round";
      ctx.strokeStyle = shade(TARGET_COLOR, 0.9); ctx.lineWidth = Math.max(2, cell * 0.8);
      for (const stroke of sketch.strokes) {
        if (stroke.length < 2) { const [sx, sy] = at(stroke[0][0], stroke[0][1]); ctx.beginPath(); ctx.arc(sx, sy, cell * 0.45, 0, Math.PI * 2); ctx.fillStyle = shade(TARGET_COLOR, 0.9); ctx.fill(); continue; }
        ctx.beginPath();
        stroke.forEach((p, i) => { const [sx, sy] = at(p[0], p[1]); return i ? ctx.lineTo(sx, sy) : ctx.moveTo(sx, sy); });
        ctx.stroke();
      }
    }
    // the arm, interpolated over the body steps of the decision being shown
    const pose = poseAt(view, u);
    const elbow = [X(pose.elbow[0]), Y(pose.elbow[1])], hand = [X(pose.hand[0]), Y(pose.hand[1])], base = [X(0), Y(0)];
    drawLink(ctx, base, elbow, 7 * px, 5.5 * px);
    drawLink(ctx, elbow, hand, 5.5 * px, 4 * px);
    ctx.fillStyle = "#0b1622"; ctx.strokeStyle = "#31506e"; ctx.lineWidth = 1.4 * px;
    ctx.beginPath(); ctx.arc(base[0], base[1], 8 * px, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#cedae6"; ctx.strokeStyle = "#0a131c";
    ctx.beginPath(); ctx.arc(elbow[0], elbow[1], 4 * px, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    // the pen: filled while it draws, an open ring while it travels
    const down = view.penState;
    ctx.fillStyle = shade(INK_COLOR, down ? 0.22 : 0.10);
    ctx.beginPath(); ctx.arc(hand[0], hand[1], 9 * px, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = shade(INK_COLOR, 0.9); ctx.lineWidth = 1.6 * px;
    ctx.beginPath(); ctx.arc(hand[0], hand[1], 4.6 * px, 0, Math.PI * 2); ctx.stroke();
    if (down) { ctx.fillStyle = `rgb(${INK_COLOR.join(",")})`; ctx.beginPath(); ctx.arc(hand[0], hand[1], 2.6 * px, 0, Math.PI * 2); ctx.fill(); }
    // the two aligned views the brain reads, in the corner
    thumbnails(ctx, W, view, px);
    scaleBar(ctx, W, H, scale, px);
  }

  /** The target and the canvas as the brain sees them, side by side in the top right corner. */
  function thumbnails(ctx, W, view, px) {
    const size = geometry.size, side = Math.min(62, W * 0.17), cell = side / size;
    const y = 10, gap = 8;
    const panes = [["target", view.target, TARGET_COLOR], ["canvas", view.canvas, INK_COLOR]];
    panes.forEach(([label, image, color], k) => {
      const x = W - 10 - (panes.length - k) * side - (panes.length - 1 - k) * gap;
      ctx.fillStyle = "rgba(5,10,15,0.8)"; ctx.fillRect(x, y, side, side);
      ctx.strokeStyle = "rgba(143,163,184,0.25)"; ctx.lineWidth = 1; ctx.strokeRect(x + 0.5, y + 0.5, side - 1, side - 1);
      ctx.fillStyle = shade(color, 0.92);
      for (let r = 0; r < size; r++) for (let c = 0; c < size; c++) if (image[r * size + c]) ctx.fillRect(x + c * cell, y + r * cell, Math.max(1, cell), Math.max(1, cell));
      ctx.fillStyle = "rgba(146,166,186,0.8)";
      ctx.font = `${9 * Math.max(1, px * 0.9)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
      ctx.textAlign = "center";
      ctx.fillText(label, x + side / 2, y + side + 11);
    });
    ctx.textAlign = "left";
  }

  function poseAt(view, u) {
    if (!view.trace || !view.trace.length) return { elbow: view.pose.elbow, hand: view.pose.hand };
    const k = Math.min(view.trace.length - 1, Math.floor(u * view.trace.length));
    const step = view.trace[k];
    return { elbow: [step[0], step[1]], hand: [step[2], step[3]] };
  }

  function drawLink(ctx, a, b, wa, wb) {
    const dx = b[0] - a[0], dy = b[1] - a[1], len = Math.hypot(dx, dy) || 1, angle = Math.atan2(dy, dx);
    const nx = -dy / len, ny = dx / len;
    ctx.beginPath();
    ctx.moveTo(a[0] + nx * wa, a[1] + ny * wa);
    ctx.lineTo(b[0] + nx * wb, b[1] + ny * wb);
    ctx.arc(b[0], b[1], wb, angle + Math.PI / 2, angle - Math.PI / 2, true);
    ctx.lineTo(a[0] - nx * wa, a[1] - ny * wa);
    ctx.arc(a[0], a[1], wa, angle - Math.PI / 2, angle - 1.5 * Math.PI, true);
    ctx.closePath();
    const grad = ctx.createLinearGradient(a[0] + nx * wa, a[1] + ny * wa, a[0] - nx * wa, a[1] - ny * wa);
    grad.addColorStop(0, `rgb(${ARM_COLOR.join(",")})`); grad.addColorStop(1, "#1e3247");
    ctx.fillStyle = grad; ctx.fill();
    ctx.strokeStyle = "rgba(11,20,29,0.9)"; ctx.lineWidth = 1; ctx.stroke();
  }

  function scaleBar(ctx, W, H, scale, px) {
    const length = geometry.extent * scale, x0 = 14, x1 = x0 + length, y = H - 13;
    ctx.strokeStyle = "rgba(146,166,186,0.5)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.moveTo(x0, y - 3); ctx.lineTo(x0, y + 3); ctx.moveTo(x1, y - 3); ctx.lineTo(x1, y + 3); ctx.stroke();
    ctx.fillStyle = "rgba(146,166,186,0.75)"; ctx.font = `${10 * Math.max(1, px * 0.9)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textAlign = "left"; ctx.fillText(`${geometry.extent} arm lengths`, x0, y - 6);
  }

  // ---------------------------------------------------------------- the readouts

  function updateHud() {
    const view = stageView, info = view ? view.drawing : null;
    const shown = info || lastResult;
    const drawingNow = sketch !== null;
    let state = "", cls = running ? "live" : "";
    $("chamferValue").textContent = shown ? shown.chamfer.toFixed(4) : "—";
    $("f1Value").textContent = shown ? `F1 ${shown.f1.toFixed(3)} · ${view && view.measure ? view.measure.ink : 0} pixels inked` : "";
    $("progressFill").style.width = `${Math.round((shown ? Math.max(0, Math.min(1, shown.f1)) : 0) * 100)}%`;
    if (drawingNow) { state = sketch.pointer !== null ? "drawing the target" : `the artist starts in ${Math.max(0, (commitAt - performance.now()) / 1000).toFixed(1)} s`; cls = "busy"; }
    else if (nextDrawing) { state = "starting"; cls = "busy"; }
    else if (info) state = `drawing ${info.n} · ${info.decisions} decisions · ${info.strokes} strokes`;
    else if (lastResult) state = `drawing ${lastResult.n} · ${lastResult.ended === "finished" ? "finished" : "stopped"}`;
    else state = "draw a figure";
    if (!running && !drawingNow) state = `paused · ${state}`;
    if (performance.now() < flashUntil) { state = flashText; cls = "busy"; }
    $("drawState").textContent = state;
    $("drawState").className = `state ${cls}`.trim();
    const best = results.length ? Math.max(...results.map((r) => r.f1)) : null;
    $("history").innerHTML = results.slice(-6).map((r) => `<span class="${r.f1 === best ? "best" : ""}" title="drawing ${r.n}: ${r.name} · ${r.brain} · ${r.decisions} decisions, Chamfer ${r.chamfer.toFixed(4)}">${r.n} · ${r.f1.toFixed(2)}</span>`).join("");
    const hint = $("hint");
    const idle = !drawingNow && !info && !nextDrawing;
    if (idle && !lastResult) { hint.hidden = false; hint.className = "hint"; hint.firstElementChild.textContent = "Draw a figure on the canvas"; }
    else if (!drawingNow && (demo || lastResult)) { hint.hidden = false; hint.className = "hint small"; hint.firstElementChild.textContent = "Draw your own figure on the canvas"; }
    else hint.hidden = true;
  }

  function flash(text, ms = 2600) { flashText = text; flashUntil = performance.now() + ms; updateHud(); }

  function updateKey() {
    const items = [
      [TARGET_COLOR, 0.45, "target", "box"], [INK_COLOR, 1, "ink", "box"],
      [PLAN_COLOR, 0.7, "imagined marks", "box"], [MARK_COLOR, 0.9, "the stroke", ""],
      [INK_COLOR, 0.9, "the pen", "dot"],
    ];
    $("canvasKey").innerHTML = items.map(([c, a, text, shape]) => `<span><i class="${shape}" style="background:${shade(c, a)}"></i>${text}</span>`).join("");
  }

  // ---------------------------------------------------------------- drawing the target

  const toPixel = (clientX, clientY) => {
    const rect = stageCanvas.getBoundingClientRect();
    const x = (clientX - rect.left - frame2.cx) / frame2.scale + VIEW.cx;
    const y = -(clientY - rect.top - frame2.cy) / frame2.scale + VIEW.cy;
    const p = pixelOf([x, y], geometry);
    const lo = geometry.margin, hi = geometry.size - 1 - geometry.margin;
    return [Math.min(hi, Math.max(lo, p[0])), Math.min(hi, Math.max(lo, p[1]))];
  };

  stageCanvas.addEventListener("pointerdown", (event) => {
    if (event.button > 0) return;
    event.preventDefault();
    stopDemo();
    if (sketch === null) { sketch = { strokes: [], pointer: null }; nextDrawing = null; lastResult = null; }
    if (commitTimer !== null) { clearTimeout(commitTimer); commitTimer = null; }
    stageCanvas.setPointerCapture(event.pointerId);
    sketch.pointer = event.pointerId;
    sketch.strokes.push([toPixel(event.clientX, event.clientY)]);
    held = true;
    updateHud();
  });

  stageCanvas.addEventListener("pointermove", (event) => {
    if (!sketch || sketch.pointer !== event.pointerId) return;
    const stroke = sketch.strokes[sketch.strokes.length - 1];
    const moves = typeof event.getCoalescedEvents === "function" ? event.getCoalescedEvents() : [];
    for (const move of moves.length ? moves : [event]) {
      const p = toPixel(move.clientX, move.clientY), last = stroke[stroke.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= 0.35) stroke.push(p);
    }
  });

  const releasePointer = (event) => {
    if (!sketch || sketch.pointer !== event.pointerId) return;
    sketch.pointer = null;
    commitAt = performance.now() + COMMIT_DELAY;
    if (commitTimer !== null) clearTimeout(commitTimer);
    commitTimer = setTimeout(commitSketch, COMMIT_DELAY);
    updateHud();
  };
  stageCanvas.addEventListener("pointerup", releasePointer);
  stageCanvas.addEventListener("pointercancel", releasePointer);

  function commitSketch() {
    commitTimer = null;
    const drawn = sketch;
    sketch = null; held = false;
    if (!drawn) return;
    let length = 0;
    for (const stroke of drawn.strokes) for (let k = 1; k < stroke.length; k++) length += Math.hypot(stroke[k][0] - stroke[k - 1][0], stroke[k][1] - stroke[k - 1][1]);
    if (length < MIN_SKETCH) { flash("that figure is too short: draw a longer stroke"); updateHud(); return; }
    const spec = drawnTarget(drawn.strokes, geometry, startAngles(rng, geometry)); // a fresh pose over the canvas, as a family target gets
    let pixels = 0;
    for (const v of spec.target) pixels += v;
    if (!pixels) { flash("that figure left no pixel on the canvas"); updateHud(); return; }
    requestDrawing(spec);
  }

  // ---------------------------------------------------------------- the other controls

  for (const button of document.querySelectorAll(".chip[data-family]")) button.onclick = () => requestDrawing(familyTarget(button.dataset.family));
  $("again").onclick = () => { const spec = (drawing && drawing.spec) || (lastResult && lastResult.spec); if (spec) requestDrawing(spec); };
  $("clear").onclick = () => {
    stopDemo();
    if (commitTimer !== null) { clearTimeout(commitTimer); commitTimer = null; }
    sketch = null; held = false; nextDrawing = null; lastResult = null;
    drawing = null; moment = null; switchRequested = false;
    world.reset({ target: new Uint8Array(geometry.size * geometry.size), theta: world.body.theta });
    $("again").disabled = true;
    stageView = restingView();
    applyEvent(stageView);
  };
  $("pace").onchange = () => { pace = Number($("pace").value); };
  $("showPlan").onchange = () => { showPlan = $("showPlan").checked; };
  $("play").onclick = () => setRunning(!running);
  $("step").onclick = () => { setRunning(false); if (wantsEvents()) { computeEvent(); drain(0); showAtOnce(); } else flash("draw a figure or pick a family first"); };
  $("fit").onclick = () => scan.fit();
  $("view").onchange = () => { scan.options.mode = $("view").value; scan.draw(); };
  $("epsilon").oninput = () => { agent.config.actor.epsilon = Number($("epsilon").value); $("epsilonValue").textContent = agent.config.actor.epsilon.toFixed(2); };

  function setRunning(value) {
    running = value;
    $("play").textContent = running ? "Pause" : "Play";
    updateHud();
  }

  // ---------------------------------------------------------------- checkpoints

  async function loadCheckpoints() {
    let body = null;
    if (location.protocol === "file:") return; // a file page cannot fetch its neighbours
    try {
      const response = await fetch(new URL("checkpoints.json", document.baseURI));
      if (!response.ok) return;
      body = await response.json();
    } catch (error) { return; }
    if (!body || body.format !== CHECKPOINT_FORMAT) return;
    for (const entry of body.checkpoints || []) if (entry && entry.file && entry.id) checkpoints.push(entry);
    if (!checkpoints.length) return;
    const select = $("checkpoint");
    select.innerHTML = [`<option value="inlined">${checkpoint.label}</option>`, ...checkpoints.map((c) => `<option value="${c.id}">${c.label}${c.bytes ? ` (${(c.bytes / 1e6).toFixed(1)} MB)` : ""}</option>`)].join("");
    select.value = "inlined";
    select.onchange = () => selectCheckpoint(select.value);
    $("checkpointBox").hidden = false;
  }

  async function selectCheckpoint(id) {
    const select = $("checkpoint");
    if (id === checkpoint.id) return;
    if (id === "inlined") { installLife(FIRST, "this page's brain", "inlined"); flash("this page's brain"); return; }
    const entry = checkpoints.find((c) => c.id === id);
    if (!entry) return;
    select.disabled = true;
    flash(`loading ${entry.label}…`, 60000);
    try {
      let snap = snapshotCache.get(id);
      if (!snap) {
        const response = await fetch(new URL(entry.file, document.baseURI));
        if (!response.ok) throw Error(`${entry.file}: ${response.status}`);
        snap = await response.json();
        if (snap.format !== "cadence-experience-web/1") throw Error(`${entry.file} is not a cadence-experience-web/1 snapshot`);
        snapshotCache.set(id, snap);
      }
      installLife(snap, entry.label, id);
      flash(`${entry.label} · ${nextDrawing || drawing ? "drawing the same figure" : "draw a figure"}`);
    } catch (error) {
      select.value = checkpoint.id;
      flash(`could not load that brain: ${error.message}`, 6000);
      console.error(error);
    } finally { select.disabled = false; }
  }

  // ---------------------------------------------------------------- the loop

  function frame(now) {
    const dt = Math.min(0.5, Math.max(1 / 240, (now - lastFrame) / 1000));
    lastFrame = now;
    fps = fps * 0.9 + 0.1 / dt;
    if (running && wantsEvents()) {
      due = Math.min(1.5, due + (dt * pace) / decisionSeconds);
      if (due >= 1 && queuedSteps < 260) { computeEvent(); due -= 1; }
    } else due = 0;
    if (queue.length) {
      const framesPerEvent = running ? Math.max(1, (decisionSeconds / Math.max(0.01, pace)) * fps) : 24;
      drain(queuedSteps ? Math.max(1, Math.ceil(queuedSteps / framesPerEvent)) : 0);
    }
    if (sketch || commitTimer !== null || performance.now() < flashUntil + 60) updateHud();
    showPhase();
    if (running || queue.length) { scan.draw(now); if (records) records.draw(); }
    drawStage(now);
    requestAnimationFrame(frame);
  }

  function showAtOnce() {
    shownAt = -1e9;
    showPhase();
    scan.draw();
    drawStage();
    if (records) records.draw();
  }

  addEventListener("resize", () => { scan.draw(); if (records) records.draw(); drawStage(); });

  // ---------------------------------------------------------------- start

  installLife(FIRST, "this page's brain", "inlined");
  $("epsilonValue").textContent = agent.config.actor.epsilon.toFixed(2);
  $("again").disabled = true;
  setRunning(running);
  setPhase("a new drawing on this canvas", "");
  loadCheckpoints();
  if (PAGE_OPTIONS.autoplay !== false) startDemo();
  requestAnimationFrame(frame);

  window.__page = {
    scan, get agent() { return agent; }, get brain() { return brain; }, get world() { return world; }, get records() { return records; }, queue,
    get stats() {
      return {
        ...stats, parameter_version: agent.parameterVersion, record_writes: agent.ledger.record_writes, imagined: agent.ledger.imagined,
        records: records ? { ...records.counts, active: records.activeIndex.length } : null, episode: world.episode, tick: world.tick, queued: queue.length,
        epsilon: agent.config.actor.epsilon, checkpoint: checkpoint.id, experience: (snapshot.ledger && snapshot.ledger.real_transitions) || 0,
        renderer: scan.snapshot(), drawing: this.drawing,
      };
    },
    /** What the canvas is doing: the running drawing, the last finished one and every result so far. */
    get drawing() {
      const info = drawingInfo();
      return {
        state: sketch ? (sketch.pointer !== null ? "drawing" : "waiting") : nextDrawing ? "queued" : info ? "drawing" : lastResult ? "done" : "idle",
        family: drawing ? drawing.spec.family : nextDrawing ? nextDrawing.family : lastResult ? lastResult.family : null,
        strokes: world.strokes, ink: world.measure().ink, pen: world.penState,
        running: info, last: lastResult ? { ...lastResult, spec: undefined } : null, results: results.map((r) => ({ ...r, spec: undefined })),
        checkpoints: checkpoints.map((c) => c.id), brain: checkpoint.label,
      };
    },
    /** The canvas, the target and the measures, for an independent measurement. */
    log() { const m = world.measure(); return { canvas: Array.from(world.canvas), target: Array.from(world.target), ...m, f1_check: foregroundF1(world.canvas, world.target, geometry.size) }; },
    /** One decision, computed and shown at once (the headless check drives the page with these). */
    step() { if (wantsEvents()) computeEvent(); drain(0); showAtOnce(); return this.stats; },
    run(count = 1) { let ran = 0; while (ran < count && wantsEvents()) { computeEvent(); drain(0); ran += 1; } showAtOnce(); return { ran, ...this.stats }; },
    /** Draw one of the target families, or the figure already on the canvas. */
    family(name) { requestDrawing(familyTarget(name)); return this.drawing; },
    pause() { setRunning(false); }, play() { setRunning(true); },
    selectCheckpoint,
    /** Where a canvas pixel sits on the screen, for pointer events. */
    toClient(row, column) {
      const rect = stageCanvas.getBoundingClientRect(), p = handOfPixel([row, column], geometry);
      return [rect.left + frame2.cx + (p[0] - VIEW.cx) * frame2.scale, rect.top + frame2.cy - (p[1] - VIEW.cy) * frame2.scale];
    },
    ready: true,
  };
  window.__brainScan = scan;
}

loadSnapshot().then(main).catch((error) => { $("counts").textContent = `error: ${error.message}`; $("drawState").textContent = "failed to load"; console.error(error); });
