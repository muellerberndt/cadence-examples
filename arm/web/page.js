// The S01 arm page: a drawing surface over the arm's workspace, with the whole brain beside
// it. The visitor's strokes become a moving target (resampled to even spacing, clipped to the
// reachable ring, one point per decision, the way movingPath produces a moving target); the
// arm copies them with its hand and leaves ink, and a tracking error reads out live. Every
// torque comes from ExperienceAgent.step through the planner, which imagines each of the nine
// torque pairs through the records cortex; the outcome of every decision is written into the
// records, so the brain learns while it copies. Beside the drawing the whole brain runs
// through the standard BrainScan renderer and the records-cortex view: every settling step of
// every phase is queued by the engine's callbacks and played back in step with the body,
// synapses flash when an update moves them, and the ledger and the dopamine bar follow the
// events in the same order. The reaching task of the earlier page (static and moving targets,
// the body change and its return) is the second tab. The brain selector loads another point
// of this arm's life from the checkpoint manifest beside the page. Nothing is scripted: the
// arm copies only as well as the model it has learned lets it.

import { BrainScan } from "../../web/brain_scan.js";
import { ExperienceAgent } from "../../web/engine.js";
import { RecordsView, pathwayNames, readingGroups, readsHTML } from "../../web/records_view.js";
import { ModelPlanner } from "./planner.js";
import { Arm, ArmRandom, CopyPath, FIGURES, TORQUES, clipToRing, distanceToDrawing, exampleFigure, figurePath, movingPath, workspaceOf } from "./arm.js";

const $ = (id) => document.getElementById(id);
const PAGE_OPTIONS = { autoplay: true, ...(window.__ARM_PAGE__ || {}) };
const CHECKPOINT_FORMAT = "cadence-arm-checkpoints/1";
const PHASE_TEXT = { free: "free phase", free_post: "free phase after learning", predict: "predict (the chosen torque's consequence)", imagine: "imagine (nine torques at once)", repair_free: "world repair · free", repair_plus: "world repair · nudged +β", repair_minus: "world repair · nudged −β", bootstrap: "bootstrap value (time limit)", score_plus: "actor score · +β", score_minus: "actor score · −β" };
const REGION_NOTE = { sensory: "the arm's own senses", goal: "the target marker", efference: "copy of the executed torque", workspace: "association", dynamics: "reads the efference copy", context: "trace of the workspace", motor: "torque readout", prediction: "predicted consequences", perception: "perception", recall: "recall", intention: "intention" };
const INK_COLOR = [143, 225, 157], MARK_COLOR = [237, 129, 182], PLAN_COLOR = [96, 165, 250], DRAW_COLOR = [219, 230, 240], TORQUE_COLOR = [255, 191, 112];
const DEMO_FIGURES = ["letter", "star", "spiral", "circle"];
const IMAGINED_SCALE = 3; // the planner's imagined hand paths, drawn at three times their size
const COPY_SETTINGS = { approach: 60, startRadius: 0.06, tail: 10, commitDelay: 900, minLength: 0.08 }; // commitDelay: the window for adding another stroke before the copy starts
const shade = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

async function loadSnapshot() {
  if (window.__SNAPSHOT__) return window.__SNAPSHOT__;
  const url = document.body.dataset.snapshot;
  const response = await fetch(url);
  if (!response.ok) throw Error(`cannot load ${url}: ${response.status}`);
  return response.json();
}

function main(FIRST) {
  const scan = new BrainScan($("scan"), FIRST.atlas, { labels: $("labels"), strip: $("strip"), labelTop: 12, montageRows: 5 });
  const queue = []; // settle frames and markers in the order the engine produced them
  let queuedSteps = 0;

  // -- the life, rebuilt whenever another checkpoint is loaded
  let snapshot = null, checkpoint = { id: "inlined", label: "this page's brain" };
  let agent = null, planner = null, records = null, arm = null;
  let moment = null, continued = false, decisionSeconds = 0.1, baseHorizon = 200, baseLengths = [0.5, 0.5];
  let predictionFields = [], lastPrediction = null, lastWrite = null;
  let restState = null, lastS = null, lastV = null, previousWeights = null;
  let stats = freshStats();

  // -- the page's own state
  let task = "draw", reachMode = "static", pendingReach = null, placedTarget = null;
  let figure = null, copy = null, nextCopy = null, displayCopy = null, copyCount = 0, lastResult = null;
  let sketch = null, commitTimer = null, commitAt = 0, held = false;
  let demo = false, demoAt = 0, demoTimer = null;
  let running = PAGE_OPTIONS.autoplay !== false, pace = 1, due = 0, fps = 60, lastFrame = performance.now();
  let armView = null, armShownAt = 0, currentPhase = null, ledgerPhase = null, lastStepText = "", flashText = "", flashUntil = 0, lastBrainDraw = 0;
  let phaseText = "—", phaseClass = "";
  const results = []; // every finished copy: {n, figure, brain, mean, ink, max, count}
  const trail = [], targetTrail = [];
  const pathRng = new ArmRandom(0xa11ce);
  const checkpoints = [], snapshotCache = new Map();
  const armCanvas = $("arm");
  let viewFrame = { cx: 0, cy: 0, scale: 1 };
  let predicted = null, actual = null;
  scan.onhover = (hit) => { if (hit) $("inspector").textContent = `${hit.region} · neuron ${hit.neuron}\nactivation ${hit.activation.toFixed(4)} · change ${hit.change.toExponential(2)}` + (hit.potential === null ? "" : ` · potential ${hit.potential.toFixed(4)}`); };

  function freshStats() {
    return { events: 0, decisions: 0, episodes: 0, successes: 0, reward: 0, distance: 0, prediction_mse: null, controller: "—", dopamine: 0, td_error: 0, compute_ms: 0, hold: 0, rejected: 0, learning: {}, last_decision: null, continued: false };
  }

  // ---------------------------------------------------------------- the life

  /** A checkpoint exported without the page's extras borrows them from the inlined snapshot. */
  function fillExtras(snap) {
    snap.extra = snap.extra || {};
    if (!snap.extra.planner && FIRST.extra) snap.extra.planner = FIRST.extra.planner;
    if (!snap.extra.arm && FIRST.extra) snap.extra.arm = FIRST.extra.arm;
    return snap;
  }

  function installLife(snap, label, id) {
    fillExtras(snap);
    snapshot = snap;
    checkpoint = { id, label };
    queue.length = 0; queuedSteps = 0; currentPhase = null; ledgerPhase = null;
    planner = new ModelPlanner(snap.extra.planner || {});
    agent = new ExperienceAgent(snap, {
      planner: planner.forAgent(),
      onSettleStep: (phase, s, v, meta) => { queue.push({ kind: "step", phase, s: Float32Array.from(s), v: Float32Array.from(v), batch: meta.batch, step: meta.step, moved: meta.moved }); queuedSteps += 1; },
      onPhase: (info) => queue.push({ kind: "phase", phase: info.phase, steps: info.steps, residual: info.residual, converged: info.converged, batch: info.batch }),
      onLearn: (info) => queue.push({ kind: "learn", phase: info.phase, changed: info.changed.length, rejected: info.rejected, dopamine: info.dopamine, td_error: info.td_error, weights: info.changed.length ? agent.effectiveWeights() : null }),
      onRecords: (info) => queue.push({ ...info, kind: "records", event: info.kind }),
    });
    agent.config.actor.epsilon = Number($("epsilon").value) || 0;
    predictionFields = agent.config.graph.prediction;
    lastPrediction = null; lastWrite = null;
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
    // the body: the life continues where the snapshot left it when the export carries the
    // decision awaiting its outcome and the moment that answers it; otherwise that decision is
    // abandoned (Agent.abandon: an evaluation copy moving to another body) and an episode begins
    const armConfig = snap.extra.arm || {};
    baseLengths = [...(armConfig.lengths || [0.5, 0.5])];
    arm = new Arm(armConfig, { seed: 0x5eed, life_id: "web", event: agent.lastEvent[0] + 1, episode: agent.episode[0] });
    baseHorizon = arm.config.horizon;
    decisionSeconds = arm.config.dt * arm.config.substeps;
    const next = snap.extra.next_moment || null, pending = agent.pending[0];
    if (pending !== null && next && next.feedback_for === pending.decision.decision_id && next.episode_id === agent.episode[0] && next.event_id > agent.lastEvent[0]) { arm.restore(next); moment = next; continued = true; }
    else { if (pending !== null) agent.abandon(0); moment = arm.reset(); continued = false; }
    stats = freshStats();
    stats.continued = continued;
    stats.distance = arm.distance();
    copy = null; nextCopy = null; displayCopy = null; predicted = null; actual = null; trail.length = 0; targetTrail.length = 0;
    armView = restingView();
    armShownAt = -1e9;
    applyEvent(armView);
    updateFacts();
    updateKey();
    updateHud();
    if (figure) requestCopy(figure, { keepDemo: true }); // the same drawing, copied by the brain just loaded
  }

  function installRecords(snap) {
    records = null;
    if (!agent.records) { $("recordsPanel").hidden = true; return; }
    const cfg = agent.config.records, rc = snap.records || {};
    records = new RecordsView($("records"), { granules: agent.records.granules, active: cfg.active, inputs: agent.reading.length, groups: readingGroups(snap), pathways: pathwayNames(snap), color: rc.color || undefined, rate: cfg.rate, rewardRate: cfg.reward_rate, habituation: cfg.habituation });
    $("recordsPanel").hidden = false;
    $("recordsInfo").textContent = `the world model · ${agent.records.granules.toLocaleString()} granules, ${cfg.active} active per reading · ${agent.records.parameters().toLocaleString()} records in ${predictionFields.length} fields · the executed reading's code in blue, writes in orange, imagined reads in grey`;
    $("recordReads").innerHTML = readsHTML(null, predictionFields);
  }

  function buildLegend() {
    const items = snapshot.atlas.regions.map((r) => `<span title="${r.name}"><i style="background:rgb(${r.color.join(",")})"></i><b>${r.label ?? r.name}</b><em>${r.count}</em>${REGION_NOTE[r.name] ? ` ${REGION_NOTE[r.name]}` : ""}</span>`);
    if (records) items.push(`<span title="the world model"><i style="background:rgb(${records.spec.color.join(",")})"></i><b>records cortex</b><em>${agent.records.granules.toLocaleString()}</em> the world model</span>`);
    $("legend").innerHTML = items.join("");
  }

  function updateFacts() {
    const lived = (snapshot.ledger && snapshot.ledger.real_transitions) || 0;
    $("counts").textContent = `${checkpoint.label} · ${lived.toLocaleString()} decisions lived before this page\n${agent.n.toLocaleString()} neurons · ${agent.E.toLocaleString()} synapses · ${snapshot.atlas.regions.length} regions${agent.records ? ` · ${agent.records.granules.toLocaleString()} granules` : ""}`;
  }

  // ---------------------------------------------------------------- figures and copies

  function buildCopy(fig) {
    const ws = workspaceOf(arm.config), speed = Number($("targetSpeed").value) || 0.12;
    const path = figurePath(fig.strokes, { spacing: speed * decisionSeconds, inner: ws.inner, outer: ws.outer });
    const c = new CopyPath(path, COPY_SETTINGS);
    c.figure = fig; c.speed = speed; c.brain = checkpoint.label;
    c.sum = 0; c.count = 0; c.inkSum = 0; c.max = 0; c.cancelled = false; c.n = 0;
    c.ink = []; c.log = [];
    return c;
  }

  /** Copy this figure: the running episode ends at the next event and the copy starts on the one after. */
  function requestCopy(fig, { keepDemo = false } = {}) {
    if (!keepDemo) stopDemo();
    if (task !== "draw") setTask("draw", { quiet: true });
    figure = fig;
    if (copy) copy.cancelled = true;
    displayCopy = null;
    nextCopy = buildCopy(fig);
    if (moment === null) moment = beginEpisode();
    else arm.requestTruncation();
    $("again").disabled = false;
    updateHud();
  }

  function beginEpisode() {
    if (task === "draw") {
      if (nextCopy === null) { copy = null; return null; }
      copy = nextCopy; nextCopy = null;
      copy.n = ++copyCount;
      displayCopy = copy; // its ink and its path stay on the surface after the copy is over
      const c = copy;
      arm.config.horizon = c.horizon;
      arm.movingTarget = (tick) => c.targetAt(tick, arm.hand);
      trail.length = 0; targetTrail.length = 0;
      return arm.reset({ theta: arm.theta }); // the body keeps its pose; the episode is the copy
    }
    copy = null;
    arm.config.horizon = baseHorizon;
    if (pendingReach !== null) { reachMode = pendingReach; pendingReach = null; }
    trail.length = 0; targetTrail.length = 0;
    if (placedTarget !== null) { const t = placedTarget; placedTarget = null; arm.movingTarget = null; return arm.reset({ theta: arm.theta, target: t }); }
    arm.movingTarget = reachMode === "static" ? null : movingPath(reachMode, pathRng);
    return arm.reset();
  }

  function finishCopy(c) {
    if (c.cancelled || c.count === 0) return;
    lastResult = { n: c.n, figure: c.figure.name, brain: checkpoint.label, speed: c.speed, points: c.points.length, mean: c.sum / c.count, ink: c.inkSum / c.count, max: c.max, count: c.count, log: c.log };
    results.push(lastResult);
    if (results.length > 24) results.shift();
  }

  function copyInfo(c, tick) {
    const index = c.indexAt(tick);
    const phase = c.start === null ? "approach" : c.finishedAt(tick) ? "finished" : index >= c.points.length - 1 ? "finishing" : "copying";
    return { n: c.n, figure: c.figure.name, brain: c.brain, phase, progress: c.progressAt(tick), mean: c.count ? c.sum / c.count : null, ink: c.count ? c.inkSum / c.count : null, count: c.count, max: c.max, points: c.points.length, speed: c.speed };
  }

  function startDemo() {
    demo = true;
    demoAt = 0;
    requestCopy(exampleOf(DEMO_FIGURES[0]), { keepDemo: true });
  }

  function stopDemo() { demo = false; if (demoTimer !== null) { clearTimeout(demoTimer); demoTimer = null; } }

  function exampleOf(name) { return { name: FIGURES[name] || name, strokes: exampleFigure(name), example: name }; }

  function afterCopy() {
    if (!demo) return;
    demoTimer = setTimeout(() => {
      demoTimer = null;
      if (!demo) return;
      demoAt = (demoAt + 1) % DEMO_FIGURES.length;
      requestCopy(exampleOf(DEMO_FIGURES[demoAt]), { keepDemo: true });
    }, 2400);
  }

  // ---------------------------------------------------------------- one event

  function wantsEvents() { return moment !== null && !held && (task === "reach" || copy !== null || nextCopy !== null); }

  function computeEvent() {
    if (moment === null) return false;
    const t0 = performance.now();
    const decision = agent.step(moment);
    const report = agent.lastReport;
    stats.events += 1;
    const scores = report.scores[0];
    if (scores) { const values = Object.values(scores).filter(Number.isFinite); stats.prediction_mse = values.length ? values.reduce((a, b) => a + b, 0) / values.length : null; }
    stats.learning = report.learning;
    if (report.learning.dopamine !== undefined) { stats.dopamine = report.learning.dopamine; stats.td_error = report.learning.td_error; }
    stats.rejected = agent.ledger.rejected_updates;
    stats.last_decision = decision === null ? null : { decision_id: decision.decision_id, action: decision.action, controller: decision.controller, event_id: decision.event_id };
    let imagined = null, trace = null, torque = null, inkFrom = 0, inkTo = 0, ended = null;
    let previousTarget = arm.target;
    if (decision === null) {
      stats.episodes += 1;
      if (moment.terminated) stats.successes += 1;
      ended = copy;
      if (copy) finishCopy(copy);
      moment = beginEpisode();
      // the episode's last event shows the finished copy whole: all of its ink, nothing new
      inkFrom = inkTo = displayCopy && !displayCopy.cancelled ? displayCopy.ink.length : 0;
      previousTarget = arm.target;
      stats.reward = 0; stats.controller = "—";
      predicted = null; actual = null;
      if (moment === null && ended && !ended.cancelled) afterCopy();
    } else {
      stats.decisions += 1;
      stats.controller = decision.controller;
      if (decision.controller === "planner" && planner.lastImagined) imagined = planner.lastImagined;
      torque = TORQUES[decision.action];
      const observation = moment.observation, prediction = decision.prediction;
      predicted = prediction && prediction.dd_hand ? [observation.d_hand[0] + prediction.dd_hand[0], observation.d_hand[1] + prediction.dd_hand[1]] : null;
      moment = arm.act(decision.decision_id, decision.action);
      actual = [moment.observation.d_hand[0], moment.observation.d_hand[1]];
      stats.reward = moment.reward;
      trace = handTrace(arm.trace);
      const hand = arm.hand;
      trail.push([hand[0], hand[1]]);
      if (trail.length > 240) trail.shift();
      if (copy) {
        const tick = arm.tick, index = copy.indexAt(tick), pen = index >= 1 ? copy.pen[index] : 0;
        inkFrom = copy.ink.length;
        for (const step of trace) copy.ink.push([step[2], step[3], pen]);
        inkTo = copy.ink.length;
        if (copy.movingAt(tick)) {
          const target = arm.target, error = Math.hypot(hand[0] - target[0], hand[1] - target[1]);
          copy.sum += error; copy.count += 1; copy.max = Math.max(copy.max, error);
          copy.inkSum += distanceToDrawing(copy.path.drawn, hand);
          copy.log.push([hand[0], hand[1], target[0], target[1]]);
        }
        if (copy.finishedAt(tick)) arm.requestTruncation();
      }
    }
    stats.distance = arm.distance();
    stats.hold = arm.hold;
    stats.compute_ms = performance.now() - t0;
    queue.push({
      kind: "event", state: arm.state, previousTarget, trace, imagined, torque, episode: arm.episode, tick: arm.tick,
      trail: trail.map((p) => [p[0], p[1]]), predicted, actual, config: { ...arm.config, lengths: [...arm.config.lengths] },
      stats: { ...stats, learning: { ...stats.learning } }, ledger: agent.ledger.toDict(), parameter_version: agent.parameterVersion,
      ring: agent.ring.length, task, mode: reachMode, epsilon: agent.config.actor.epsilon,
      copy: displayCopy && !displayCopy.cancelled ? displayCopy : null, copyInfo: copy && !copy.cancelled ? copyInfo(copy, arm.tick) : null, inkFrom, inkTo,
    });
    return true;
  }

  function handTrace(thetas) {
    if (!thetas) return null;
    const [l1, l2] = arm.config.lengths, out = [];
    for (let k = 0; k < thetas.length; k += 2) {
      const a = thetas[k], b = thetas[k + 1], ex = l1 * Math.cos(a), ey = l1 * Math.sin(a);
      out.push([ex, ey, ex + l2 * Math.cos(a + b), ey + l2 * Math.sin(a + b)]);
    }
    return out;
  }

  function restingView() {
    return {
      kind: "event", state: arm.state, previousTarget: arm.target, trace: null, imagined: null, torque: null, episode: arm.episode, tick: arm.tick,
      trail: [], predicted: null, actual: null, config: { ...arm.config, lengths: [...arm.config.lengths] }, stats: { ...stats },
      ledger: agent.ledger.toDict(), parameter_version: agent.parameterVersion, ring: agent.ring.length, task, mode: reachMode,
      epsilon: agent.config.actor.epsilon, copy: null, copyInfo: null, inkFrom: 0, inkTo: 0,
    };
  }

  // ---------------------------------------------------------------- playback

  function applyRecords(item) {
    if (!records) return;
    if (item.event === "code") {
      records.code(item, "decide");
      records.setReading(item.reading, item.blockNorm);
      lastPrediction = item.prediction;
      $("recordReads").innerHTML = readsHTML(lastPrediction, predictionFields, lastWrite);
    } else if (item.event === "imagine") records.code(item, "imagine");
    else if (item.event === "write") {
      records.write(item);
      lastWrite = item;
      $("recordReads").innerHTML = readsHTML(lastPrediction, predictionFields, lastWrite);
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
    armView = item;
    armShownAt = performance.now();
    if (item.state && item.copy) targetTrail.push([item.state.target[0], item.state.target[1]]);
    if (targetTrail.length > 26) targetTrail.shift();
    const s = item.stats, c = item.config, d = s.dopamine;
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
      agent.records ? ["records", `${L.record_writes.toLocaleString()} writes`] : ["ring", `${item.ring}`],
      ["imagined", `${L.imagined.toLocaleString()} reads`],
    ]);
    $("counters").innerHTML = rows([
      ["episode", `${item.episode} · tick ${item.tick}/${c.horizon}`], ["decisions", `${s.decisions}`],
      ["controller", `${s.controller} · ε ${item.epsilon.toFixed(2)}`], ["links", `${c.lengths[0].toFixed(2)} · ${c.lengths[1].toFixed(2)} · ${s.compute_ms.toFixed(0)} ms`],
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

  // ---------------------------------------------------------------- the arm view

  function drawArm(now = performance.now()) {
    const rect = armCanvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    if (!rect.width || !rect.height) return;
    const w = Math.max(1, Math.round(rect.width * dpr)), h = Math.max(1, Math.round(rect.height * dpr));
    if (armCanvas.width !== w || armCanvas.height !== h) { armCanvas.width = w; armCanvas.height = h; }
    const ctx = armCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const W = rect.width, H = rect.height, view = armView || restingView(), cfg = view.config;
    const reach = cfg.lengths[0] + cfg.lengths[1];
    const scale = Math.min(W, H) / 2 / (Math.max(reach, workspaceOf(cfg).outer) * 1.03);
    viewFrame = { cx: W / 2, cy: H / 2, scale };
    const X = (x) => viewFrame.cx + x * scale, Y = (y) => viewFrame.cy - y * scale;
    const px = Math.max(0.7, Math.min(1.8, Math.min(W, H) / 420));
    const seconds = decisionSeconds / Math.max(0.01, pace);
    const u = Math.max(0, Math.min(1, (now - armShownAt) / (seconds * 1000)));
    // the ground and the reachable ring
    const ground = ctx.createRadialGradient(viewFrame.cx, viewFrame.cy * 1.05, 0, viewFrame.cx, viewFrame.cy, Math.hypot(W, H) / 2);
    ground.addColorStop(0, "#0b1621"); ground.addColorStop(1, "#04080c");
    ctx.fillStyle = ground; ctx.fillRect(0, 0, W, H);
    const ws = workspaceOf(cfg);
    ctx.beginPath();
    ctx.arc(viewFrame.cx, viewFrame.cy, ws.outer * scale, 0, Math.PI * 2);
    ctx.arc(viewFrame.cx, viewFrame.cy, ws.inner * scale, 0, Math.PI * 2, true);
    ctx.fillStyle = "rgba(89,229,203,0.055)"; ctx.fill();
    ctx.lineWidth = 1; ctx.setLineDash([3, 6]); ctx.strokeStyle = "rgba(143,163,184,0.3)";
    for (const r of [ws.inner, ws.outer]) { ctx.beginPath(); ctx.arc(viewFrame.cx, viewFrame.cy, r * scale, 0, Math.PI * 2); ctx.stroke(); }
    ctx.setLineDash([]);
    ctx.strokeStyle = "rgba(143,163,184,0.10)";
    ctx.beginPath(); ctx.moveTo(X(-reach), Y(0)); ctx.lineTo(X(reach), Y(0)); ctx.moveTo(X(0), Y(-reach)); ctx.lineTo(X(0), Y(reach)); ctx.stroke();
    // the drawing: the visitor's live strokes, or the path the target follows
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    if (sketch) {
      ctx.strokeStyle = shade(DRAW_COLOR, 0.85); ctx.lineWidth = 2.4 * px;
      for (const stroke of sketch.strokes) {
        if (stroke.length < 2) { ctx.beginPath(); ctx.arc(X(stroke[0][0]), Y(stroke[0][1]), 2 * px, 0, Math.PI * 2); ctx.fillStyle = shade(DRAW_COLOR, 0.85); ctx.fill(); continue; }
        ctx.beginPath(); ctx.moveTo(X(stroke[0][0]), Y(stroke[0][1]));
        for (let k = 1; k < stroke.length; k++) ctx.lineTo(X(stroke[k][0]), Y(stroke[k][1]));
        ctx.stroke();
      }
    } else {
      const shown = nextCopy || (view.copy && !view.copy.cancelled ? view.copy : null);
      if (shown) {
        for (const stroke of shown.path.drawn) {
          if (stroke.length < 2) continue;
          ctx.beginPath(); ctx.moveTo(X(stroke[0][0]), Y(stroke[0][1]));
          for (let k = 1; k < stroke.length; k++) ctx.lineTo(X(stroke[k][0]), Y(stroke[k][1]));
          ctx.strokeStyle = shade(DRAW_COLOR, 0.16); ctx.lineWidth = 8 * px; ctx.stroke();
          ctx.strokeStyle = shade(DRAW_COLOR, 0.42); ctx.lineWidth = 1.2 * px; ctx.stroke();
        }
      }
    }
    // the ink: the hand's own path at the body's clock, up to the moment being shown
    const inked = view.copy && !view.copy.cancelled ? view.copy : null;
    if (inked && inked.ink.length) {
      const upTo = Math.max(0, Math.min(inked.ink.length, Math.round(view.inkFrom + (view.inkTo - view.inkFrom) * u)));
      const down = new Path2D(), up = new Path2D();
      for (let i = 1; i < upTo; i++) {
        const a = inked.ink[i - 1], b = inked.ink[i];
        const path = b[2] ? down : up;
        path.moveTo(X(a[0]), Y(a[1])); path.lineTo(X(b[0]), Y(b[1]));
      }
      ctx.strokeStyle = shade(INK_COLOR, 0.12); ctx.lineWidth = 7 * px; ctx.stroke(down);
      ctx.strokeStyle = shade(INK_COLOR, 0.35); ctx.lineWidth = 4 * px; ctx.stroke(down);
      ctx.strokeStyle = shade(INK_COLOR, 0.98); ctx.lineWidth = 1.8 * px; ctx.stroke(down);
      ctx.setLineDash([1.5, 4]); ctx.strokeStyle = shade(INK_COLOR, 0.3); ctx.lineWidth = 1 * px; ctx.stroke(up); ctx.setLineDash([]);
    }
    // the hand trail of the reaching task
    if (view.task === "reach" && view.trail.length > 1) {
      for (let k = 1; k < view.trail.length; k++) {
        ctx.strokeStyle = shade(INK_COLOR, 0.06 + 0.5 * (k / view.trail.length));
        ctx.lineWidth = 1.4 * px;
        ctx.beginPath(); ctx.moveTo(X(view.trail[k - 1][0]), Y(view.trail[k - 1][1])); ctx.lineTo(X(view.trail[k][0]), Y(view.trail[k][1])); ctx.stroke();
      }
    }
    // the body, interpolated over the body steps of the decision being shown
    const pose = poseAt(view, u);
    const elbow = [X(pose.elbow[0]), Y(pose.elbow[1])], hand = [X(pose.hand[0]), Y(pose.hand[1])], base = [X(0), Y(0)];
    // what the planner imagined for each torque, from the hand it imagined from
    if (view.imagined) {
      const best = view.imagined.candidates[0], k = IMAGINED_SCALE; // imagined motion is a few hundredths of an arm length
      for (const candidate of view.imagined.candidates) {
        const strong = candidate === best, from = candidate.hands[0];
        const at = (p) => [X(from[0] + (p[0] - from[0]) * k), Y(from[1] + (p[1] - from[1]) * k)];
        ctx.strokeStyle = shade(PLAN_COLOR, strong ? 0.95 : 0.34);
        ctx.lineWidth = (strong ? 2.2 : 1.2) * px;
        ctx.beginPath();
        candidate.hands.forEach((p, i) => { const [x, y] = at(p); return i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
        ctx.stroke();
        const [lx, ly] = at(candidate.hands[candidate.hands.length - 1]);
        ctx.fillStyle = shade(PLAN_COLOR, strong ? 0.95 : 0.4);
        ctx.beginPath(); ctx.arc(lx, ly, (strong ? 2.6 : 1.7) * px, 0, Math.PI * 2); ctx.fill();
      }
    }
    drawLink(ctx, base, elbow, 9 * px, 7 * px, ["#4a6c8c", "#1e3247"]);
    drawLink(ctx, elbow, hand, 7 * px, 5 * px, ["#567c9c", "#243b52"]);
    if (view.torque) drawTorques(ctx, base, elbow, hand, view.torque, px);
    ctx.fillStyle = "#0b1622"; ctx.strokeStyle = "#31506e"; ctx.lineWidth = 1.6 * px;
    ctx.beginPath(); ctx.arc(base[0], base[1], 11 * px, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = "#9db2c7"; ctx.beginPath(); ctx.arc(base[0], base[1], 4 * px, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = "#cedae6"; ctx.strokeStyle = "#0a131c"; ctx.lineWidth = 1.4 * px;
    ctx.beginPath(); ctx.arc(elbow[0], elbow[1], 5 * px, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
    ctx.fillStyle = shade(INK_COLOR, 0.16); ctx.beginPath(); ctx.arc(hand[0], hand[1], 11 * px, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = shade(INK_COLOR, 0.3); ctx.beginPath(); ctx.arc(hand[0], hand[1], 7 * px, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgb(${INK_COLOR.join(",")})`; ctx.beginPath(); ctx.arc(hand[0], hand[1], 4.4 * px, 0, Math.PI * 2); ctx.fill();
    // the target: a bright marker with the trail of its last positions
    const target = targetAt(view, u);
    for (let k = 0; k < targetTrail.length - 1; k++) {
      const p = targetTrail[k];
      ctx.fillStyle = shade(MARK_COLOR, 0.05 + 0.2 * (k / targetTrail.length));
      ctx.beginPath(); ctx.arc(X(p[0]), Y(p[1]), 1.6 * px, 0, Math.PI * 2); ctx.fill();
    }
    if (view.task === "reach") {
      ctx.strokeStyle = shade(MARK_COLOR, 0.4); ctx.lineWidth = 1.2 * px;
      ctx.beginPath(); ctx.arc(X(target[0]), Y(target[1]), cfg.success_radius * scale, 0, Math.PI * 2); ctx.stroke();
    }
    ctx.fillStyle = shade(MARK_COLOR, 0.2); ctx.beginPath(); ctx.arc(X(target[0]), Y(target[1]), 8 * px, 0, Math.PI * 2); ctx.fill();
    ctx.fillStyle = `rgb(${MARK_COLOR.join(",")})`; ctx.beginPath(); ctx.arc(X(target[0]), Y(target[1]), 3.2 * px, 0, Math.PI * 2); ctx.fill();
    // the reaching task also shows what the model predicted against what happened
    if (view.task === "reach") {
      const from = view.trail.length > 1 ? view.trail[view.trail.length - 2] : pose.hand;
      arrow(ctx, X, Y, from, view.predicted, "#59e5cb", px);
      arrow(ctx, X, Y, from, view.actual, "#ffbf70", px);
    }
    drawScaleBar(ctx, W, H, scale, px);
  }

  function poseAt(view, u) {
    if (!view.trace || !view.trace.length) return { elbow: view.state.elbow, hand: view.state.hand };
    const k = Math.min(view.trace.length - 1, Math.floor(u * view.trace.length));
    const step = view.trace[k];
    return { elbow: [step[0], step[1]], hand: [step[2], step[3]] };
  }

  function targetAt(view, u) {
    const t = view.state.target, p = view.previousTarget || t;
    return [p[0] + (t[0] - p[0]) * u, p[1] + (t[1] - p[1]) * u];
  }

  function drawLink(ctx, a, b, wa, wb, colors) {
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
    grad.addColorStop(0, colors[0]); grad.addColorStop(1, colors[1]);
    ctx.fillStyle = grad; ctx.fill();
    ctx.strokeStyle = "rgba(11,20,29,0.9)"; ctx.lineWidth = 1; ctx.stroke();
  }

  /** The executed torque at each joint: an arc on the free side of the joint, turning the way the torque pushes. */
  function drawTorques(ctx, base, elbow, hand, torque, px) {
    const joints = [[base, elbow, torque[0], 15 * px], [elbow, hand, torque[1], 11 * px]];
    for (const [centre, next, value, radius] of joints) {
      if (!value) continue;
      const angle = Math.atan2(next[1] - centre[1], next[0] - centre[0]) + Math.PI;
      const from = angle - 0.85, to = angle + 0.85;
      ctx.strokeStyle = shade(TORQUE_COLOR, 0.85); ctx.lineWidth = 1.7 * px;
      ctx.beginPath(); ctx.arc(centre[0], centre[1], radius, from, to); ctx.stroke();
      const tip = value > 0 ? from : to, direction = value > 0 ? -1 : 1; // the y axis points up in the world
      const tx = centre[0] + radius * Math.cos(tip), ty = centre[1] + radius * Math.sin(tip);
      const ax = -Math.sin(tip) * direction, ay = Math.cos(tip) * direction;
      ctx.fillStyle = shade(TORQUE_COLOR, 0.9);
      ctx.beginPath();
      ctx.moveTo(tx + ax * 4 * px, ty + ay * 4 * px);
      ctx.lineTo(tx - ay * 2.6 * px - ax * 1.5 * px, ty + ax * 2.6 * px - ay * 1.5 * px);
      ctx.lineTo(tx + ay * 2.6 * px - ax * 1.5 * px, ty - ax * 2.6 * px - ay * 1.5 * px);
      ctx.closePath(); ctx.fill();
    }
  }

  function arrow(ctx, X, Y, from, delta, color, px) {
    if (!delta) return;
    const k = 5.0; // displacements are a few hundredths of an arm length: drawn at five times their size
    const tx = X(from[0] + delta[0] * k), ty = Y(from[1] + delta[1] * k), fx = X(from[0]), fy = Y(from[1]);
    const len = Math.hypot(tx - fx, ty - fy);
    if (len < 1) return;
    const ux = (tx - fx) / len, uy = (ty - fy) / len;
    ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2 * px;
    ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty); ctx.stroke();
    ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(tx - 7 * px * ux + 3.5 * px * uy, ty - 7 * px * uy - 3.5 * px * ux); ctx.lineTo(tx - 7 * px * ux - 3.5 * px * uy, ty - 7 * px * uy + 3.5 * px * ux); ctx.closePath(); ctx.fill();
  }

  function drawScaleBar(ctx, W, H, scale, px) {
    const length = 0.1 * scale, x0 = 14, x1 = x0 + length, y = H - 13;
    ctx.strokeStyle = "rgba(146,166,186,0.5)"; ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(x0, y); ctx.lineTo(x1, y); ctx.moveTo(x0, y - 3); ctx.lineTo(x0, y + 3); ctx.moveTo(x1, y - 3); ctx.lineTo(x1, y + 3); ctx.stroke();
    ctx.fillStyle = "rgba(146,166,186,0.75)"; ctx.font = `${10 * Math.max(1, px * 0.9)}px ui-monospace, SFMono-Regular, Menlo, monospace`;
    ctx.textAlign = "left"; ctx.fillText("0.1 arm length", x0, y - 6);
  }

  // ---------------------------------------------------------------- the readouts around the canvas

  function updateHud() {
    const view = armView, info = view ? view.copyInfo : null;
    const drawing = sketch !== null;
    let label = "Tracking error", value = "—", unit = "arm lengths, mean hand to target", ink = "", progress = 0;
    let state = "", cls = running ? "live" : "";
    if (task === "reach") {
      label = "Distance to target";
      value = view ? view.stats.distance.toFixed(3) : "—";
      unit = "arm lengths, hand to target";
      progress = view ? Math.min(1, view.tick / Math.max(1, view.config.horizon)) : 0;
      state = view ? `episode ${view.episode} · tick ${view.tick}` : "reaching";
    } else {
      const shown = info && info.mean !== null ? info : lastResult;
      if (shown) {
        value = shown.mean.toFixed(3);
        if (shown.ink !== null && shown.ink !== undefined) ink = `ink ${shown.ink.toFixed(3)} from your drawing`;
      }
      progress = info ? info.progress : lastResult ? 1 : 0;
      if (drawing) { state = sketch.pointer !== null ? "drawing" : `copy in ${Math.max(0, (commitAt - performance.now()) / 1000).toFixed(1)} s`; cls = "busy"; }
      else if (nextCopy) { state = "starting"; cls = "busy"; }
      else if (info) state = info.phase === "approach" ? `copy ${info.n} · to the start` : info.phase === "copying" ? `copy ${info.n} · ${Math.round(info.progress * 100)}%` : `copy ${info.n} · finishing`;
      else if (lastResult) state = `copy ${lastResult.n} · done`;
      else state = "draw a figure";
    }
    if (!running && task !== "reach" && !drawing) state = `paused · ${state}`;
    if (performance.now() < flashUntil) { state = flashText; cls = "busy"; }
    $("hudLabel").textContent = label;
    $("errorValue").textContent = value;
    $("errorUnit").textContent = unit;
    $("errorInk").textContent = ink;
    $("progressFill").style.width = `${Math.round(progress * 100)}%`;
    $("copyState").textContent = state;
    $("copyState").className = `state ${cls}`.trim();
    const best = results.length ? Math.min(...results.map((r) => r.mean)) : null;
    $("history").innerHTML = task === "reach" ? "" : results.slice(-6).map((r) => `<span class="${r.mean === best ? "best" : ""}" title="copy ${r.n}: ${r.figure} · ${r.brain} · ${r.count} decisions">${r.n} · ${r.mean.toFixed(3)}</span>`).join("");
    const hint = $("hint");
    const idle = !drawing && !info && !nextCopy;
    if (task !== "draw") hint.hidden = true;
    else if (idle && !lastResult) { hint.hidden = false; hint.className = "hint"; hint.firstElementChild.textContent = "Draw a figure inside the ring"; }
    else if (!drawing && (demo || lastResult)) { hint.hidden = false; hint.className = "hint small"; hint.firstElementChild.textContent = "Draw your own figure anywhere in the ring"; }
    else hint.hidden = true;
  }

  function flash(text, ms = 2600) { flashText = text; flashUntil = performance.now() + ms; updateHud(); }

  function updateKey() {
    const items = task === "draw"
      ? [[DRAW_COLOR, 0.35, "your drawing"], [INK_COLOR, 1, "ink: the hand's path"], [MARK_COLOR, 1, "target", true], [PLAN_COLOR, 0.9, `the nine imagined torques ×${IMAGINED_SCALE}`], [TORQUE_COLOR, 0.9, "the executed torque"]]
      : [[MARK_COLOR, 1, "target and success radius", true], [INK_COLOR, 1, "hand trail"], [[89, 229, 203], 1, "predicted displacement ×5"], [[255, 191, 112], 1, "actual displacement ×5"], [PLAN_COLOR, 0.9, `the nine imagined torques ×${IMAGINED_SCALE}`]];
    $("armKey").innerHTML = items.map(([c, a, text, dot]) => `<span><i class="${dot ? "dot" : ""}" style="background:${shade(c, a)}"></i>${text}</span>`).join("");
  }

  // ---------------------------------------------------------------- drawing and the other controls

  const toWorld = (clientX, clientY) => {
    const rect = armCanvas.getBoundingClientRect();
    return [(clientX - rect.left - viewFrame.cx) / viewFrame.scale, -(clientY - rect.top - viewFrame.cy) / viewFrame.scale];
  };

  armCanvas.addEventListener("pointerdown", (event) => {
    if (event.button > 0) return;
    event.preventDefault();
    stopDemo();
    if (task === "reach") { placeTarget(toWorld(event.clientX, event.clientY)); return; }
    if (sketch === null) { // a new figure: the running copy ends and its ink goes with it
      sketch = { strokes: [], pointer: null };
      if (copy) { copy.cancelled = true; if (moment !== null) arm.requestTruncation(); }
      nextCopy = null; displayCopy = null; figure = null; lastResult = null;
    }
    if (commitTimer !== null) { clearTimeout(commitTimer); commitTimer = null; }
    armCanvas.setPointerCapture(event.pointerId);
    sketch.pointer = event.pointerId;
    sketch.strokes.push([toWorld(event.clientX, event.clientY)]);
    held = true;
    updateHud();
  });

  armCanvas.addEventListener("pointermove", (event) => {
    if (!sketch || sketch.pointer !== event.pointerId) return;
    const stroke = sketch.strokes[sketch.strokes.length - 1];
    const moves = typeof event.getCoalescedEvents === "function" ? event.getCoalescedEvents() : [];
    for (const move of moves.length ? moves : [event]) {
      const p = toWorld(move.clientX, move.clientY), last = stroke[stroke.length - 1];
      if (Math.hypot(p[0] - last[0], p[1] - last[1]) >= 0.004) stroke.push(p);
    }
  });

  const releasePointer = (event) => {
    if (!sketch || sketch.pointer !== event.pointerId) return;
    sketch.pointer = null;
    commitAt = performance.now() + COPY_SETTINGS.commitDelay;
    if (commitTimer !== null) clearTimeout(commitTimer);
    commitTimer = setTimeout(commitSketch, COPY_SETTINGS.commitDelay);
    updateHud();
  };
  armCanvas.addEventListener("pointerup", releasePointer);
  armCanvas.addEventListener("pointercancel", releasePointer);

  function commitSketch() {
    commitTimer = null;
    const drawn = sketch;
    sketch = null; held = false;
    if (!drawn) return;
    let length = 0;
    for (const stroke of drawn.strokes) for (let k = 1; k < stroke.length; k++) length += Math.hypot(stroke[k][0] - stroke[k - 1][0], stroke[k][1] - stroke[k - 1][1]);
    if (length < COPY_SETTINGS.minLength) { flash("that figure is too short: draw a longer stroke"); return; }
    requestCopy({ name: "your drawing", strokes: drawn.strokes });
  }

  function placeTarget(point) {
    const ws = workspaceOf(arm.config), target = clipToRing(point, ws.inner, ws.outer);
    placedTarget = target;
    pendingReach = "static";
    $("mode").value = "static";
    if (moment === null) moment = beginEpisode(); else arm.requestTruncation();
    flash(`target at (${target[0].toFixed(2)}, ${target[1].toFixed(2)}) from the next episode`);
  }

  function setTask(name, { quiet = false } = {}) {
    if (name === task) return;
    task = name;
    $("modeDraw").setAttribute("aria-selected", String(name === "draw"));
    $("modeReach").setAttribute("aria-selected", String(name === "reach"));
    $("drawTools").hidden = name !== "draw";
    $("reachTools").hidden = name === "draw";
    $("speedControl").hidden = name !== "draw";
    if (!quiet) stopDemo();
    if (sketch) { sketch = null; held = false; if (commitTimer !== null) { clearTimeout(commitTimer); commitTimer = null; } }
    if (copy) copy.cancelled = true;
    if (name === "reach") displayCopy = null;
    nextCopy = null;
    if (moment === null) { if (name === "reach") moment = beginEpisode(); }
    else arm.requestTruncation();
    updateKey();
    updateHud();
  }

  $("modeDraw").onclick = () => { setTask("draw"); if (figure) requestCopy(figure); };
  $("modeReach").onclick = () => setTask("reach");
  $("again").onclick = () => { if (figure) requestCopy(figure); };
  $("clear").onclick = () => {
    stopDemo();
    if (commitTimer !== null) { clearTimeout(commitTimer); commitTimer = null; }
    sketch = null; held = false; figure = null; nextCopy = null; displayCopy = null; lastResult = null;
    if (copy) { copy.cancelled = true; if (moment !== null) arm.requestTruncation(); }
    $("again").disabled = true;
    updateHud();
  };
  for (const button of document.querySelectorAll(".chip[data-figure]")) button.onclick = () => requestCopy(exampleOf(button.dataset.figure));
  $("targetSpeed").onchange = () => flash(`the next copy moves at ${$("targetSpeed").value} arm lengths per second`);
  $("pace").onchange = () => { pace = Number($("pace").value); };
  $("target").onclick = () => { arm.requestTruncation(); flash("a new target at the next episode"); };
  $("mode").onchange = () => { pendingReach = $("mode").value; if (moment === null) moment = beginEpisode(); else arm.requestTruncation(); };
  $("lengthen").onclick = () => { arm.config.lengths[0] *= 1.15; flash(`upper link now ${arm.config.lengths[0].toFixed(3)}: the model must adapt`); };
  $("resetBody").onclick = () => { arm.config.lengths = [...baseLengths]; flash("body restored"); };
  $("play").onclick = () => setRunning(!running);
  $("step").onclick = () => { setRunning(false); if (wantsEvents()) computeEvent(); else flash("draw a figure or pick an example first"); };
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
    } catch (error) { return; } // no manifest beside the page (a file:// page carries only its own brain)
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
      flash(`${entry.label} · ${figure ? "copying the same figure" : "draw a figure"}`);
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
    // A paused page holds its picture still: the brain is drawn while it plays, and once more
    // whenever the window changes. The drawing surface follows the pointer every frame.
    if (running || queue.length) { scan.draw(now); if (records) records.draw(); lastBrainDraw = now; }
    drawArm(now);
    requestAnimationFrame(frame);
  }

  function showAtOnce() {
    armShownAt = -1e9;
    showPhase();
    scan.draw();
    drawArm();
    if (records) records.draw();
    lastBrainDraw = performance.now();
  }

  addEventListener("resize", () => { scan.draw(); if (records) records.draw(); drawArm(); });

  // ---------------------------------------------------------------- start

  installLife(FIRST, "this page's brain", "inlined");
  $("epsilonValue").textContent = agent.config.actor.epsilon.toFixed(2);
  $("again").disabled = true;
  setRunning(running);
  setPhase(continued ? `the life continues at event ${moment.event_id}` : "a new episode on this body", "");
  loadCheckpoints();
  if (PAGE_OPTIONS.autoplay !== false) startDemo();
  requestAnimationFrame(frame);

  window.__page = {
    scan, get agent() { return agent; }, get arm() { return arm; }, get planner() { return planner; }, get records() { return records; }, queue,
    get stats() {
      return {
        ...stats, parameter_version: agent.parameterVersion, ring: agent.ring.length, record_writes: agent.ledger.record_writes, imagined: agent.ledger.imagined,
        records: records ? { ...records.counts, active: records.activeIndex.length } : null, episode: arm.episode, tick: arm.tick, queued: queue.length,
        task, mode: reachMode, epsilon: agent.config.actor.epsilon, lengths: [...arm.config.lengths], checkpoint: checkpoint.id,
        experience: (snapshot.ledger && snapshot.ledger.real_transitions) || 0, renderer: scan.snapshot(), copy: this.copy,
      };
    },
    /** What the drawing surface is doing: the running copy, the last finished one and every result so far. */
    get copy() {
      const info = copy && !copy.cancelled ? copyInfo(copy, arm.tick) : null;
      return {
        state: sketch ? (sketch.pointer !== null ? "drawing" : "waiting") : nextCopy ? "queued" : info ? info.phase : lastResult ? "done" : "idle",
        figure: figure ? figure.name : null, strokes: figure ? figure.strokes.length : 0,
        points: info ? info.points : nextCopy ? nextCopy.points.length : lastResult ? lastResult.points : 0,
        penUp: copy && !copy.cancelled ? Array.from(copy.pen).filter((p) => !p).length : nextCopy ? Array.from(nextCopy.pen).filter((p) => !p).length : 0,
        running: info, last: lastResult ? { ...lastResult, log: undefined } : null, results: results.map((r) => ({ ...r, log: undefined })),
        checkpoints: checkpoints.map((c) => c.id), brain: checkpoint.label,
      };
    },
    /** The hand and target positions of every decision the target moved on, for an independent measurement. */
    log(which = "last") { const source = which === "running" ? copy : lastResult; return source && source.log ? source.log.map((row) => [...row]) : []; },
    /** One event, computed and shown at once (the headless check drives the page with these). */
    step() { if (wantsEvents()) computeEvent(); drain(0); showAtOnce(); return this.stats; },
    run(count = 1) { let ran = 0; while (ran < count && wantsEvents()) { computeEvent(); drain(0); ran += 1; } showAtOnce(); return { ran, ...this.stats }; },
    pause() { setRunning(false); }, play() { setRunning(true); },
    selectCheckpoint,
    /** Where a point of the workspace sits on the screen, for pointer events. */
    toClient(x, y) { const rect = armCanvas.getBoundingClientRect(); return [rect.left + viewFrame.cx + x * viewFrame.scale, rect.top + viewFrame.cy - y * viewFrame.scale]; },
    ready: true,
  };
  window.__brainScan = scan;
}

loadSnapshot().then(main).catch((error) => { $("counts").textContent = `error: ${error.message}`; $("copyState").textContent = "failed to load"; console.error(error); });
