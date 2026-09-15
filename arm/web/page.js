// The S01 arm page: the body on the left, the whole brain on the right through the standard
// BrainScan renderer, the same engine that passes the parity check driving both. Every
// settling step of every phase is queued by the engine's callbacks and played back at the
// chosen rate; learning flashes the synapses that moved; the ledger and the dopamine bar
// follow the events in the same order. With a records world head the records cortex beside
// the scan shows every code, imagined read and write in the same queue (RecordsView).
// Learning never stops while it plays and nothing is scripted: the arm reaches only as well
// as the model it has learned lets it.

import { BrainScan } from "../../web/brain_scan.js";
import { ExperienceAgent } from "../../web/engine.js";
import { RecordsView, pathwayNames, readingGroups, readsHTML } from "../../web/records_view.js";
import { ModelPlanner } from "./planner.js";
import { Arm, ArmRandom, movingPath } from "./arm.js";

const $ = (id) => document.getElementById(id);
const PHASE_LABEL = { free: "free phase", free_post: "free phase after learning", predict: "predict (the chosen action's consequence)", imagine: "imagine (planner: nine torques at once)", repair_free: "world repair · free", repair_plus: "world repair · nudged +β", repair_minus: "world repair · nudged −β", bootstrap: "bootstrap value (time limit)", score_plus: "actor score · +β", score_minus: "actor score · −β" };

async function loadSnapshot() {
  if (window.__SNAPSHOT__) return window.__SNAPSHOT__;
  const url = document.body.dataset.snapshot;
  const response = await fetch(url);
  if (!response.ok) throw Error(`cannot load ${url}: ${response.status}`);
  return response.json();
}

function main(SNAPSHOT) {
  const planner = new ModelPlanner(SNAPSHOT.extra.planner || {});
  const queue = []; // settle frames and markers in the order the engine produced them
  let agent = null;
  agent = new ExperienceAgent(SNAPSHOT, {
    planner: planner.forAgent(),
    onSettleStep: (phase, s, v, meta) => queue.push({ kind: "step", phase, s: Float32Array.from(s), v: Float32Array.from(v), batch: meta.batch, step: meta.step, moved: meta.moved }),
    onPhase: (info) => queue.push({ kind: "phase", phase: info.phase, steps: info.steps, residual: info.residual, converged: info.converged, batch: info.batch }),
    onLearn: (info) => queue.push({ kind: "learn", phase: info.phase, changed: info.changed.length, rejected: info.rejected, dopamine: info.dopamine, td_error: info.td_error, weights: info.changed.length ? agent.effectiveWeights() : null }),
    onRecords: (info) => queue.push({ ...info, kind: "records", event: info.kind }), // event: "code" | "imagine" | "write"
  });
  const scan = new BrainScan($("scan"), SNAPSHOT.atlas, { labels: $("labels"), strip: $("strip") });
  $("strip").style.height = `${Math.min(340, 30 + 19 * scan.montage().length)}px`;
  $("legend").innerHTML = SNAPSHOT.atlas.regions.map((r) => `<span><i style="background:rgb(${r.color.join(",")})"></i>${r.name} · ${r.count}</span>`).join("");
  // -- the records cortex (the world head): its own region beside the scan
  let records = null, lastPrediction = null, lastWrite = null;
  const predictionFields = agent.config.graph.prediction;
  if (agent.records) {
    const cfg = agent.config.records, rc = SNAPSHOT.records || {};
    records = new RecordsView($("records"), { granules: agent.records.granules, active: cfg.active, inputs: agent.reading.length, groups: readingGroups(SNAPSHOT), pathways: pathwayNames(SNAPSHOT), color: rc.color || undefined, rate: cfg.rate, rewardRate: cfg.reward_rate, habituation: cfg.habituation });
    $("recordsPanel").hidden = false;
    $("recordsInfo").textContent = `cerebellum-like / dentate-like · ${agent.records.granules.toLocaleString()} granules, ${cfg.active} active per reading · ${agent.reading.length} reading neurons · ${predictionFields.length} record fields, ${agent.records.parameters().toLocaleString()} records · rate ${cfg.rate}, reward ${cfg.reward_rate}${cfg.habituation > 0 ? `, habituation ${cfg.habituation}` : ""} · plain code blue, valued code pink, writes orange (reward pink), imagined reads grey`;
    $("legend").innerHTML += `<span><i style="background:rgb(${records.spec.color.join(",")})"></i>records cortex (cerebellum-like / dentate-like) · ${agent.records.granules.toLocaleString()} granules</span>`;
    $("recordReads").innerHTML = readsHTML(null, predictionFields);
  }
  function applyRecords(item) {
    if (!records) return;
    if (item.event === "code") { records.code(item, "decide"); records.setReading(item.reading, item.blockNorm); lastPrediction = item.prediction; $("recordReads").innerHTML = readsHTML(lastPrediction, predictionFields, lastWrite); }
    else if (item.event === "imagine") records.code(item, "imagine");
    else if (item.event === "write") {
      records.write(item); lastWrite = item;
      $("recordReads").innerHTML = readsHTML(lastPrediction, predictionFields, lastWrite);
      $("phase").textContent = `records: ${item.written} field${item.written === 1 ? "" : "s"} written into ${item.index.length} cells${item.fields.some((f) => f.reward) ? " (reward)" : ""}`;
      $("phase").className = "learn";
      currentPhase = null;
    }
  }
  scan.onhover = (hit) => { if (hit) $("inspector").textContent = `${hit.region} · neuron ${hit.neuron}\nactivation ${hit.activation.toFixed(4)} · change ${hit.change.toExponential(2)}` + (hit.potential === null ? "" : ` · potential ${hit.potential.toFixed(4)}`); };
  const restState = Float32Array.from(agent.warm ? agent.warm.v : new Float64Array(agent.n), (v) => agent._act(v));
  scan.set(restState, { potential: Float32Array.from(agent.warm ? agent.warm.v : new Float64Array(agent.n)), draw: false });
  let lastS = restState, lastV = Float32Array.from(agent.warm ? agent.warm.v : new Float64Array(agent.n));
  let previousWeights = agent.effectiveWeights();
  scan.setWeights(previousWeights);

  // -- the body
  const armConfig = SNAPSHOT.extra.arm || {};
  const baseLengths = [...(armConfig.lengths || [0.5, 0.5])];
  const arm = new Arm(armConfig, { seed: 0x5eed, life_id: "web", event: agent.lastEvent[0] + 1, episode: agent.episode[0] });
  const pathRng = new ArmRandom(0xa11ce);
  let mode = "static", pendingMode = null;
  const stats = { events: 0, decisions: 0, episodes: 0, successes: 0, reward: 0, distance: 0, prediction_mse: null, controller: "—", dopamine: 0, td_error: 0, compute_ms: 0, hold: 0, rejected: 0, learning: {}, last_decision: null, continued: false };
  // The life continues where the snapshot left it when the export carries the decision awaiting
  // its outcome and the moment that answers it: the body is restored from that moment's sensors
  // and the moment is fed back first. Otherwise the pending decision is abandoned (Agent.abandon:
  // an evaluation copy moving to another body) and a new episode begins on this body.
  const nextMoment = SNAPSHOT.extra && SNAPSHOT.extra.next_moment ? SNAPSHOT.extra.next_moment : null;
  const pendingNow = agent.pending[0];
  let moment;
  if (pendingNow !== null && nextMoment && nextMoment.feedback_for === pendingNow.decision.decision_id && nextMoment.episode_id === agent.episode[0] && nextMoment.event_id > agent.lastEvent[0]) {
    arm.restore(nextMoment);
    moment = nextMoment;
    stats.continued = true;
  } else {
    if (pendingNow !== null) agent.abandon(0);
    moment = arm.reset();
  }
  const trail = [];
  let predicted = null, actual = null;
  let armView = { state: arm.state, trail: [], predicted: null, actual: null, config: { ...arm.config } };
  let ledgerPhase = null, lastLearn = null;
  let currentPhase = null;
  stats.distance = arm.distance();

  function computeEvent() {
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
    if (decision === null) {
      stats.episodes += 1;
      if (moment.terminated) stats.successes += 1;
      if (pendingMode !== null) { mode = pendingMode; pendingMode = null; }
      arm.movingTarget = mode === "static" ? null : movingPath(mode, pathRng);
      moment = arm.reset();
      trail.length = 0; predicted = null; actual = null; stats.reward = 0; stats.controller = "—";
    } else {
      stats.decisions += 1; stats.controller = decision.controller;
      const obs = moment.observation, pred = decision.prediction;
      predicted = pred.dd_hand ? [obs.d_hand[0] + pred.dd_hand[0], obs.d_hand[1] + pred.dd_hand[1]] : null;
      moment = arm.act(decision.decision_id, decision.action);
      actual = [moment.observation.d_hand[0], moment.observation.d_hand[1]];
      stats.reward = moment.reward;
      const hand = arm.hand; trail.push([hand[0], hand[1]]); if (trail.length > 300) trail.shift();
    }
    stats.distance = arm.distance(); stats.hold = arm.hold;
    stats.compute_ms = performance.now() - t0;
    queue.push({ kind: "event", state: arm.state, episode: arm.episode, tick: arm.tick, trail: trail.map((p) => [p[0], p[1]]), predicted, actual, config: { ...arm.config, lengths: [...arm.config.lengths] }, stats: { ...stats, learning: { ...stats.learning } }, ledger: agent.ledger.toDict(), parameter_version: agent.parameterVersion, ring: agent.ring.length, mode, epsilon: agent.config.actor.epsilon });
  }

  // -- playback of the queued frames
  function flash(item) {
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
    lastLearn = item;
    const label = item.rejected ? `${item.phase} update rejected (phases did not converge)` : item.phase === "world" ? `world repair moved ${item.changed.toLocaleString()} synapses` : item.phase === "credit" ? (item.changed ? `motor credit moved ${item.changed.toLocaleString()} synapses` : `critic · dopamine ${item.dopamine.toFixed(4)}`) : item.phase;
    $("phase").textContent = label;
    $("phase").className = "learn";
  }

  function applyEvent(item) {
    armView = item;
    const s = item.stats, c = item.config;
    $("counters").innerHTML = [
      ["episode", `${item.episode}`], ["tick", `${item.tick} / ${c.horizon}`], ["events", `${s.events}`], ["decisions", `${s.decisions}`],
      ["episodes ended", `${s.episodes}`], ["successes", `${s.successes}`], ["distance", `${s.distance.toFixed(3)}`], ["in radius", `${s.hold} / ${c.success_hold}`],
      ["last reward", `${s.reward.toFixed(4)}`], ["controller", s.controller], ["prediction mse", s.prediction_mse === null ? "—" : s.prediction_mse.toExponential(2)], ["target", item.mode],
      ["links", `${c.lengths[0].toFixed(3)} · ${c.lengths[1].toFixed(3)}`], ["compute", `${s.compute_ms.toFixed(0)} ms`],
    ].map(([k, v]) => `<div><span>${k}</span><b>${v}</b></div>`).join("");
    const d = s.dopamine, half = Math.min(50, (Math.abs(d) / 0.25) * 50);
    const fill = $("dopamineFill");
    fill.className = "fill" + (d < 0 ? " negative" : "");
    fill.style.left = d < 0 ? `${50 - half}%` : "50%";
    fill.style.width = `${half}%`;
    const worldRow = agent.records ? ["records written", s.learning.records_written === undefined ? "—" : `${s.learning.records_written} fields`] : ["world step", s.learning.world_scale_step === undefined ? "—" : s.learning.world_scale_step.toExponential(2)];
    $("critic").innerHTML = [["dopamine (clipped TD)", d.toFixed(5)], ["|TD error|", s.td_error.toFixed(5)], ["critic bias", agent.bCritic.toFixed(5)], worldRow, ["replayed", s.learning.replayed === undefined ? "—" : `${s.learning.replayed}`]].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
    const lp = ledgerPhase || agent.lastPhase, L = item.ledger;
    const unconverged = Object.values(L.unconverged_phases).reduce((a, b) => a + b, 0);
    $("ledger").innerHTML = [
      ["last phase", lp ? `${lp.phase} · ${lp.steps} steps · residual ${lp.residual.toExponential(2)} ${lp.converged ? "✓" : "✗"}` : "—"],
      ["rejected updates", `${L.rejected_updates}`], ["unconverged phases", `${unconverged}`], ["parameter version", `${item.parameter_version}`],
      agent.records ? ["records", `${L.record_writes} writes · no ring`] : ["replay ring", `${item.ring} transitions`],
      ["real transitions", `${L.real_transitions}`], [agent.records ? "imagined reads" : "imagined", `${L.imagined}`], ["phases", Object.entries(L.phases).map(([k, v]) => `${k} ${v}`).join(" · ")], ["exploration ε", item.epsilon.toFixed(2)],
    ].map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
    $("counts").textContent = `${scan.n.toLocaleString()} neurons · ${scan.edges.toLocaleString()} synapses · ${scan.atlas.regions.length} regions${agent.records ? ` · ${agent.records.granules.toLocaleString()} granules` : ""} · parameter version ${item.parameter_version}`;
  }

  function drain(count) {
    let steps = 0;
    while (queue.length && (count === 0 || steps < count)) {
      const item = queue.shift();
      if (item.kind === "step") {
        if (item.phase !== currentPhase) { currentPhase = item.phase; scan.reset(); $("phase").textContent = (PHASE_LABEL[item.phase] || item.phase) + (item.batch > 1 ? ` · batch ${item.batch}, row 1 shown` : ""); $("phase").className = ""; }
        scan.step(item.s, { potential: item.v, draw: false });
        lastS = item.s; lastV = item.v; steps += 1;
        $("status").textContent = `${item.phase} step ${item.step} · largest movement ${item.moved.toExponential(2)} · ${queue.length} frames queued`;
      } else if (item.kind === "phase") ledgerPhase = item;
      else if (item.kind === "learn") flash(item);
      else if (item.kind === "records") applyRecords(item);
      else if (item.kind === "event") applyEvent(item);
    }
    return steps;
  }

  // -- the arm canvas
  const armCanvas = $("arm");
  function drawArm() {
    const r = armCanvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    if (!r.width) return;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (armCanvas.width !== w || armCanvas.height !== h) { armCanvas.width = w; armCanvas.height = h; }
    const ctx = armCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#05090e"; ctx.fillRect(0, 0, r.width, r.height);
    const cx = r.width / 2, cy = r.height / 2, reach = (armView.config.lengths[0] + armView.config.lengths[1]) || 1;
    const scale = (Math.min(r.width, r.height) / 2) * 0.9 / Math.max(1.0, reach);
    const X = (x) => cx + x * scale, Y = (y) => cy - y * scale;
    ctx.strokeStyle = "#162230"; ctx.lineWidth = 1; ctx.setLineDash([4, 6]);
    ctx.beginPath(); ctx.arc(cx, cy, reach * scale, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]);
    ctx.strokeStyle = "#101a26"; ctx.beginPath(); ctx.moveTo(cx - reach * scale, cy); ctx.lineTo(cx + reach * scale, cy); ctx.moveTo(cx, cy - reach * scale); ctx.lineTo(cx, cy + reach * scale); ctx.stroke();
    const st = armView.state, target = st.target;
    ctx.strokeStyle = "rgba(237,129,182,0.45)"; ctx.lineWidth = 1.5;
    ctx.beginPath(); ctx.arc(X(target[0]), Y(target[1]), armView.config.success_radius * scale, 0, Math.PI * 2); ctx.stroke();
    ctx.fillStyle = "#ed81b6"; ctx.beginPath(); ctx.arc(X(target[0]), Y(target[1]), 5, 0, Math.PI * 2); ctx.fill();
    const trailPoints = armView.trail;
    if (trailPoints.length > 1) {
      for (let k = 1; k < trailPoints.length; k++) {
        ctx.strokeStyle = `rgba(143,225,157,${0.08 + 0.6 * (k / trailPoints.length)})`; ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(X(trailPoints[k - 1][0]), Y(trailPoints[k - 1][1])); ctx.lineTo(X(trailPoints[k][0]), Y(trailPoints[k][1])); ctx.stroke();
      }
    }
    const elbow = st.elbow, hand = st.hand;
    ctx.lineCap = "round"; ctx.lineJoin = "round";
    ctx.strokeStyle = "#2b425a"; ctx.lineWidth = 12; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(X(elbow[0]), Y(elbow[1])); ctx.lineTo(X(hand[0]), Y(hand[1])); ctx.stroke();
    ctx.strokeStyle = "#8fa3b8"; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(cx, cy); ctx.lineTo(X(elbow[0]), Y(elbow[1])); ctx.lineTo(X(hand[0]), Y(hand[1])); ctx.stroke();
    ctx.fillStyle = "#dbe6f0"; for (const [px, py] of [[0, 0], [elbow[0], elbow[1]]]) { ctx.beginPath(); ctx.arc(X(px), Y(py), 5, 0, Math.PI * 2); ctx.fill(); }
    ctx.fillStyle = "#8fe19d"; ctx.beginPath(); ctx.arc(X(hand[0]), Y(hand[1]), 7, 0, Math.PI * 2); ctx.fill();
    const arrow = (from, delta, color) => {
      if (!delta) return;
      const k = 5.0; // displacements are a few hundredths of an arm length: drawn at 5x
      const tx = X(from[0] + delta[0] * k), ty = Y(from[1] + delta[1] * k), fx = X(from[0]), fy = Y(from[1]);
      const len = Math.hypot(tx - fx, ty - fy); if (len < 1) return;
      const ux = (tx - fx) / len, uy = (ty - fy) / len;
      ctx.strokeStyle = color; ctx.fillStyle = color; ctx.lineWidth = 2.5;
      ctx.beginPath(); ctx.moveTo(fx, fy); ctx.lineTo(tx, ty); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(tx, ty); ctx.lineTo(tx - 8 * ux + 4 * uy, ty - 8 * uy - 4 * ux); ctx.lineTo(tx - 8 * ux - 4 * uy, ty - 8 * uy + 4 * ux); ctx.closePath(); ctx.fill();
    };
    const origin = armView.actual && trailPoints.length > 1 ? trailPoints[trailPoints.length - 2] : [hand[0], hand[1]];
    arrow(origin, armView.predicted, "#59e5cb");
    arrow(origin, armView.actual, "#ffbf70");
    ctx.fillStyle = "#6f8397"; ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "left";
    ctx.fillText(`hand (${hand[0].toFixed(2)}, ${hand[1].toFixed(2)}) · target (${target[0].toFixed(2)}, ${target[1].toFixed(2)}) · arrows ×5`, 8, r.height - 8);
  }

  // -- controls
  let playing = false, stepping = false;
  const speed = () => Number($("speed").value);
  function tick(now) {
    if (playing) {
      const rate = speed();
      if (rate === 0) { computeEvent(); drain(0); }
      else { if (queue.length < Math.max(64, rate * 2)) computeEvent(); drain(rate); }
    } else if (stepping) {
      const rate = speed();
      drain(rate === 0 ? 0 : rate);
      if (!queue.length) stepping = false;
    }
    scan.draw(now);
    drawArm();
    if (records) records.draw();
    requestAnimationFrame(tick);
  }
  $("play").onclick = () => { playing = !playing; stepping = false; $("play").textContent = playing ? "Pause" : "Play"; };
  $("step").onclick = () => { playing = false; $("play").textContent = "Play"; computeEvent(); stepping = true; };
  $("fit").onclick = () => scan.fit();
  $("view").onchange = () => { scan.options.mode = $("view").value; scan.draw(); };
  $("epsilon").oninput = () => { agent.config.actor.epsilon = Number($("epsilon").value); $("epsilonValue").textContent = agent.config.actor.epsilon.toFixed(2); };
  $("epsilon").value = String(agent.config.actor.epsilon); $("epsilonValue").textContent = agent.config.actor.epsilon.toFixed(2);
  $("target").onclick = () => { arm.requestTruncation(); $("status").textContent = "a new target at the next event (the current decision is fed back first)"; };
  $("mode").onchange = () => { pendingMode = $("mode").value; arm.requestTruncation(); };
  $("lengthen").onclick = () => { arm.config.lengths[0] *= 1.15; $("status").textContent = `upper link now ${arm.config.lengths[0].toFixed(3)}: the model must adapt`; };
  $("resetBody").onclick = () => { arm.config.lengths = [...baseLengths]; $("status").textContent = "body restored"; };

  applyEvent({ state: arm.state, episode: arm.episode, tick: arm.tick, trail: [], predicted: null, actual: null, config: { ...arm.config, lengths: [...arm.config.lengths] }, stats: { ...stats }, ledger: agent.ledger.toDict(), parameter_version: agent.parameterVersion, ring: agent.ring.length, mode, epsilon: agent.config.actor.epsilon });
  $("phase").textContent = stats.continued ? `resting · the life continues at event ${moment.event_id} (decision ${pendingNow.decision.decision_id} awaits its outcome) · press Play` : "resting · a new episode on this body · press Play";
  scan.fit();
  requestAnimationFrame(tick);

  window.__page = {
    scan, agent, arm, queue, records,
    get stats() { return { ...stats, parameter_version: agent.parameterVersion, ring: agent.ring.length, record_writes: agent.ledger.record_writes, imagined: agent.ledger.imagined, records: records ? { ...records.counts, active: records.activeIndex.length } : null, episode: arm.episode, tick: arm.tick, queued: queue.length, mode, epsilon: agent.config.actor.epsilon, lengths: [...arm.config.lengths], renderer: scan.snapshot() }; },
    /** One event, computed and shown at once (the headless check drives the page with this). */
    step() { computeEvent(); drain(0); scan.draw(); drawArm(); if (records) records.draw(); return window.__page.stats; },
    ready: true,
  };
  window.__brainScan = scan;
}

loadSnapshot().then(main).catch((err) => { $("counts").textContent = `error: ${err.message}`; $("phase").textContent = "failed to load"; console.error(err); });
