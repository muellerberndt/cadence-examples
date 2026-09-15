// The S02 page: the remembered world on the left, the three stores under it, the whole brain
// on the right through the standard BrainScan renderer, the same engine and the same Brain
// wrapper that pass the parity check driving all of it. The curriculum runs as run.py runs
// it (explore; remembered requests at delays 8, 16 and 32; visible and invisible moves; the
// cue task at delays 8 and 32; the door change; words): every wander ends with a truncated
// closing moment and every request, name request and cue decision is a new episode of the
// same life. Every settling step of every phase is queued by the engine's callbacks and
// played back at the chosen rate; learning flashes the synapses that moved; with a records
// world head the records cortex beside the scan shows every code, imagined read and write
// in the same queue (RecordsView). Nothing is scripted: the agent reaches only what its
// stores and its learned model let it reach.

import { BrainScan } from "../../web/brain_scan.js";
import { ExperienceAgent } from "../../web/engine.js";
import { RecordsView, pathwayNames, readingGroups, readsHTML } from "../../web/records_view.js";
import { ACTIONS, CUE_WORDS, DIRECTIONS, DIRECTION_NAMES, GOAL_CUE, GOAL_EXPLORE, GOAL_OBJECT, GOAL_WORD, OBJECTS, OUTCOMES, ROOMS, SIZE, WORDS, World, Curriculum, argmaxOf, sameCell, cellKey } from "./world.js";
import { Brain } from "./planner.js";

const $ = (id) => document.getElementById(id);
const PHASE_LABEL = { free: "free phase", free_post: "free phase after learning", predict: "predict (the chosen action's consequence)", imagine: "imagine (the search's batch of consequences)", repair_free: "world repair · free", repair_plus: "world repair · nudged +β", repair_minus: "world repair · nudged −β", bootstrap: "bootstrap value (time limit)", score_plus: "actor score · +β", score_minus: "actor score · −β" };
const OBJECT_COLORS = ["#ff6b6b", "#59e5cb", "#ffd166", "#c792ea"]; // the world's four colours, by colour index
const OBJECT_NAMES = ["A", "B", "C", "D"];
const PAGE_BUDGET = { explore_steps: 100, remember_episodes: 5, correction_episodes: 3, cue_episodes: 6, door_trials: 4, word_episodes: 4, teach_steps: 24 };
const INSPECTED_OUTCOME = OUTCOMES.indexOf("inspected");
const GOAL_NAMES = ["explore", "reach A", "reach B", "reach C", "reach D", "reach the named object", "the cued room", "fetch"];

async function loadSnapshot() {
  if (window.__SNAPSHOT__) return window.__SNAPSHOT__;
  const url = document.body.dataset.snapshot;
  const response = await fetch(url);
  if (!response.ok) throw Error(`cannot load ${url}: ${response.status}`);
  return response.json();
}

function main(SNAPSHOT) {
  const extra = SNAPSHOT.extra || {}, stage = extra.config || {};
  const queue = []; // settle frames and markers in the order the engine produced them
  let imagineEvery = 4; // imagined settles are long batches: one frame in four unless asked for every step
  let agent = null;
  agent = new ExperienceAgent(SNAPSHOT, {
    onSettleStep: (phase, s, v, meta) => { if (phase === "imagine" && imagineEvery > 1 && meta.step % imagineEvery !== 1) return; queue.push({ kind: "step", phase, s: Float32Array.from(s), v: Float32Array.from(v), batch: meta.batch, step: meta.step, moved: meta.moved }); },
    onPhase: (info) => queue.push({ kind: "phase", phase: info.phase, steps: info.steps, residual: info.residual, converged: info.converged, batch: info.batch }),
    onLearn: (info) => queue.push({ kind: "learn", phase: info.phase, changed: info.changed.length, rejected: info.rejected, dopamine: info.dopamine, td_error: info.td_error, weights: info.changed.length ? agent.effectiveWeights() : null }),
    onRecords: (info) => queue.push({ ...info, kind: "records", event: info.kind }), // event: "code" | "imagine" | "write"
  });
  const brain = new Brain(agent, { ...extra, planner: { ...(extra.planner || {}), ...(stage.planner || {}) } }); // the stage's search budget, the fixture's generator state
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

  // -- the world: the fixture's map continues, the page's own random stream from here on
  const worldConfig = stage.world_config || {};
  const world = extra.world ? World.fromState(extra.world, { seed: 0x5eed, life_id: "web" }) : new World(worldConfig, { seed: 0x5eed, life_id: "web" });
  if (worldConfig.horizon) world.config.horizon = worldConfig.horizon | 0;
  const cur = new Curriculum(world, 0xa11ce, { cueRule: extra.curriculum && extra.curriculum.cue_rule !== undefined ? extra.curriculum.cue_rule : null });
  const decisionEpsilon = stage.actor && stage.actor.epsilon !== undefined ? Number(stage.actor.epsilon) : 0.1; // run.py: the cue decision explores at the life's own rate
  if (agent.pending[0] !== null) agent.abandon(0); // the page starts fresh episodes on this world
  const stats = { events: 0, decisions: 0, episodes: 0, reward: 0, controller: "—", expansions: 0, correct: null, dopamine: 0, td_error: 0, compute_ms: 0, rejected: 0, learning: {}, last_decision: null, successes: {}, phase: "curriculum", stage: "", episode: 0, episodes: 0 };
  const trail = [];
  let lastLearn = null, ledgerPhase = null, currentPhase = null;
  let worldView = null, storeView = null;

  // -- one event: the brain steps a moment, the world executes the decision, one frame is queued
  function stepBrain(m) {
    const d = brain.step(m);
    const report = agent.lastReport;
    stats.events += 1;
    const scores = report.scores[0];
    if (scores) { const values = Object.entries(scores).filter(([k]) => k.endsWith("/correct")).map(([, v]) => v); stats.correct = values.length ? [values.reduce((a, b) => a + b, 0), values.length] : null; }
    stats.learning = report.learning;
    if (report.learning.dopamine !== undefined) { stats.dopamine = report.learning.dopamine; stats.td_error = report.learning.td_error; }
    stats.rejected = agent.ledger.rejected_updates;
    stats.last_decision = d === null ? null : { decision_id: d.decision_id, action: d.action, controller: d.controller, event_id: d.event_id, expansions: d.budget.expansions ?? null };
    if (d === null) { stats.episodes += 1; stats.controller = "—"; trail.length = 0; }
    else { stats.decisions += 1; stats.controller = d.controller; stats.expansions = d.budget.expansions ?? 0; }
    return d;
  }

  function visibleNeighbours(m) {
    if (!m || argmaxOf(m.observation.outcome) !== INSPECTED_OUTCOME) return [];
    const cell = [argmaxOf(m.observation.x), argmaxOf(m.observation.y)], out = [];
    for (const d of DIRECTION_NAMES) if (m.observed[`${d}_object`] && m.observed[`${d}_object`].every((x) => x === 1)) out.push([cell[0] + DIRECTIONS[d][0], cell[1] + DIRECTIONS[d][1]]);
    return out;
  }

  function pushFrame(info, m, d, next) {
    const planned = d !== null && d.controller === "planner";
    const shown = next || m;
    const goal = m === null ? GOAL_EXPLORE : argmaxOf(m.goal), closing = m !== null && !!(m.terminated || m.truncated);
    const view = {
      agent: world.agent, objects: world.objects(), colors: Array.from({ length: OBJECTS }, (_, k) => world.colorOf(k)), edges: world.state().edges, doors: world.state().doors,
      carrying: world.carrying, tick: world.tick, horizon: world.config.horizon, deadline: world.deadline, episode: world.episode, word: world.word, doorRule: world.doorRule,
      visible: visibleNeighbours(shown), decisionAction: d === null ? null : d.action, controller: d === null ? null : d.controller, goal,
      target: planned ? brain.lastTarget : null, search: planned ? brain.lastSearch : null, requestTarget: info.target || null, rewarded: info.rewarded || null,
      cueHeld: brain.cue.word, trail: trail.map((c) => [c[0], c[1]]), closing, info: { ...info },
    };
    const stores = {
      places: brain.places.records().map((r) => ({ object: r.object, where: r.where, known: r.known })), truth: world.objects(),
      cue: brain.cue.word, cueValue: Array.from(brain.cue.value), words: brain.words.table().map((row) => Array.from(row)), referents: Array.from({ length: WORDS }, (_, w) => brain.words.referent(w + 1)),
      placeWrites: brain.places.writes, wordWrites: brain.words.writes, mapWrites: brain.map.writes, mapKnown: brain.map.records().length, recall: Float32Array.from(agent.lastRecall),
    };
    queue.push({ kind: "event", world: view, stores, stats: { ...stats, successes: JSON.parse(JSON.stringify(stats.successes)), learning: { ...stats.learning } }, ledger: agent.ledger.toDict(), parameter_version: agent.parameterVersion, ring: agent.ring.length, epsilon: brain.epsilon });
  }

  function* transition(m, info, beforeAct = null) {
    const t0 = performance.now();
    const d = stepBrain(m);
    let next = null;
    if (d !== null) {
      if (beforeAct) beforeAct();
      next = world.act(d.decision_id, d.action);
      stats.reward = next.reward;
      const a = world.agent; if (!trail.length || !sameCell(trail[trail.length - 1], a)) trail.push(a); if (trail.length > 24) trail.shift();
    }
    stats.compute_ms = performance.now() - t0;
    pushFrame(info, m, d, next);
    yield;
    return { d, next };
  }

  function* closeEpisode(m, info) {
    const closed = { ...m, truncated: true, final_observation: m.observation }; // run.py close_episode: the wander ends as a time limit would
    const t0 = performance.now();
    const d = stepBrain(closed);
    if (d !== null) throw Error("a truncated moment must close the episode");
    stats.compute_ms = performance.now() - t0;
    pushFrame({ ...info, stage: "close" }, closed, null, null);
    yield;
  }

  function* finishEpisode(m, info) {
    let steps = 0, reward = 0.0;
    for (;;) {
      const r = yield* transition(m, { ...info, t: steps + 1 });
      if (r.d === null) return { steps, reward, terminated: m.terminated };
      steps += 1; reward += r.next.reward; m = r.next;
    }
  }

  function tally(phase, ok) { const s = stats.successes[phase] || (stats.successes[phase] = { done: 0, ok: 0 }); s.done += 1; s.ok += ok; }
  let requestedPhase = null;
  const switchRequested = () => requestedPhase !== null;

  // -- run.py's phases
  function* runExplore(steps) {
    let done = 0;
    brain.setEpsilon(1.0);
    while (done < steps && !switchRequested()) {
      cur.explore();
      const m = world.reset();
      const outcome = yield* finishEpisode(m, { phase: "explore", stage: "explore", done, steps });
      done += outcome.steps;
      tally("explore", 1);
    }
    brain.setEpsilon(0.0);
  }

  function* runRemember(delay, episodes, move, key = null) {
    const phase = key || `remember-${delay}${move ? "-" + move : ""}`;
    for (let e = 0; e < episodes; e++) {
      if (switchRequested()) return;
      const episode = cur.remember(delay, { move });
      let m = world.reset({ agent: episode.target_cell });
      brain.setEpsilon(1.0);
      const info = { phase, stage: "wander", episode: e + 1, episodes, delay, object: episode.object_id };
      let closed = false, moved = false;
      for (let t = 0; t < delay; t++) {
        const r = yield* transition(m, { ...info, t: t + 1 });
        if (r.d === null) { closed = true; break; }
        m = r.next;
        if (move === "visible" && !moved && t >= Math.floor(delay / 2) && world.objectAt(world.agent) === null) { moved = true; cur.moveObject(episode, { visible: true }); const seen = world.observation(); m = { ...m, observation: seen.observation, observed: seen.observed }; } // onto the agent's own empty cell, in view before its next decision (run.py run_remember)
        if (move === "invisible" && t === Math.floor(delay / 2)) cur.moveObject(episode, { visible: false });
      }
      brain.setEpsilon(0.0);
      if (closed) { tally(phase, 0); continue; }
      yield* closeEpisode(m, info);
      m = cur.request(episode);
      const outcome = yield* finishEpisode(m, { ...info, stage: "request", target: episode.target_cell });
      tally(phase, outcome.terminated ? 1 : 0);
    }
  }

  function* runCue(delay, episodes) {
    const phase = `cue-${delay}`;
    for (let e = 0; e < episodes; e++) {
      if (switchRequested()) return;
      const episode = cur.cue(delay);
      let m = world.reset({ agent: world.config.junction });
      brain.setEpsilon(1.0);
      const info = { phase, stage: "wander", episode: e + 1, episodes, delay, cue: CUE_WORDS[episode.cue] };
      let closed = false;
      for (let t = 0; t < delay; t++) {
        const r = yield* transition(m, { ...info, t: t + 1 }, t === 0 ? () => cur.hideCue() : null);
        if (r.d === null) { closed = true; break; }
        m = r.next;
      }
      brain.setEpsilon(decisionEpsilon);
      if (closed) { tally(phase, 0); continue; }
      yield* closeEpisode(m, info);
      m = cur.armCue(episode);
      const outcome = yield* finishEpisode(m, { ...info, stage: "decision", rewarded: episode.rewarded_room });
      tally(phase, outcome.reward >= 1.0 ? 1 : 0);
    }
  }

  function* runWords(episodes) {
    for (let e = 0; e < episodes; e++) {
      if (switchRequested()) return;
      const episode = cur.word();
      let m = world.reset({ agent: episode.old_cell });
      brain.setEpsilon(1.0);
      const info = { phase: "words", stage: "teach", episode: e + 1, episodes, word: episode.cue, object: episode.object_id, delay: PAGE_BUDGET.teach_steps };
      let closed = false;
      for (let t = 0; t < PAGE_BUDGET.teach_steps; t++) {
        const r = yield* transition(m, { ...info, t: t + 1 });
        if (r.d === null) { closed = true; break; }
        m = r.next;
      }
      brain.setEpsilon(0.0);
      if (closed) { tally("words", 0); continue; }
      yield* closeEpisode(m, info);
      m = cur.nameRequest(episode);
      const outcome = yield* finishEpisode(m, { ...info, stage: "name request", target: episode.target_cell });
      tally("words", outcome.terminated ? 1 : 0);
    }
  }

  const B = PAGE_BUDGET;
  const RUNNERS = {
    curriculum: function* () {
      world.setDoorRule("toggle"); // a new cycle of the curriculum: the doors toggle again until the door phase locks them
      yield* runExplore(B.explore_steps);
      for (const delay of [8, 16, 32]) yield* runRemember(delay, B.remember_episodes, null);
      yield* runRemember(16, B.correction_episodes, "visible");
      yield* runRemember(16, B.correction_episodes, "invisible");
      for (const delay of [8, 32]) yield* runCue(delay, B.cue_episodes);
      yield* runRemember(8, B.door_trials, null, "door-before");
      cur.lockDoors();
      yield* runRemember(8, B.door_trials, null, "door-locked");
      yield* runWords(B.word_episodes);
    },
    explore: function* () { yield* runExplore(B.explore_steps); },
    "remember-8": function* () { yield* runRemember(8, B.remember_episodes, null); },
    "remember-16": function* () { yield* runRemember(16, B.remember_episodes, null); },
    "remember-32": function* () { yield* runRemember(32, B.remember_episodes, null); },
    "remember-16-visible": function* () { yield* runRemember(16, B.correction_episodes, "visible"); },
    "remember-16-invisible": function* () { yield* runRemember(16, B.correction_episodes, "invisible"); },
    "cue-8": function* () { yield* runCue(8, B.cue_episodes); },
    "cue-32": function* () { yield* runCue(32, B.cue_episodes); },
    "door-locked": function* () { cur.lockDoors(); yield* runRemember(8, B.door_trials, null, "door-locked"); },
    words: function* () { yield* runWords(B.word_episodes); },
  };
  function* driver() {
    for (;;) {
      if (requestedPhase !== null) { stats.phase = requestedPhase; requestedPhase = null; }
      yield* RUNNERS[stats.phase]();
    }
  }
  const life = driver();
  function computeEvent() { life.next(); }

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

  const cellText = (c) => (c ? `(${c[0]},${c[1]})` : "—");
  function stageText(v) {
    const i = v.info;
    if (i.stage === "rest") return `resting · the life continues on the fixture's map (${extra.steps_before ?? "its"} explore steps so far) · press Play`;
    const head = i.phase === "explore" ? `explore · ${i.done} / ${i.steps} steps` : `${i.phase} · episode ${i.episode} / ${i.episodes}`;
    if (i.stage === "explore") return `${head} · tick ${v.tick} / ${v.horizon}`;
    if (i.stage === "wander" || i.stage === "teach") return `${head} · ${i.stage} ${i.t} / ${i.delay}` + (i.cue ? ` · cue word ${i.cue}` : "") + (i.word ? ` · word ${i.word} names ${OBJECT_NAMES[i.object]}` : "");
    if (i.stage === "close") return `${head} · the wander closes (truncated)`;
    if (i.stage === "request" || i.stage === "name request") return `${head} · ${i.stage} · target ${cellText(i.target)} · ${v.deadline === null ? "" : `${Math.max(0, v.deadline - v.tick)} decisions left`}`;
    if (i.stage === "decision") return `${head} · the two-choice decision at the junction · paid room ${cellText(i.rewarded)}`;
    return head;
  }

  function applyEvent(item) {
    worldView = item.world; storeView = item.stores;
    const s = item.stats, v = item.world;
    $("phaseLabel").textContent = stageText(v);
    $("successes").innerHTML = Object.entries(s.successes).map(([k, t]) => `<span>${k}<b>${k === "explore" ? `${t.done} episodes` : `${t.ok} / ${t.done}`}</b></span>`).join("") || "<span>no episode finished yet</span>";
    $("counters").innerHTML = [
      ["episode", `${v.episode}`], ["tick", `${v.tick} / ${v.horizon}`], ["events", `${s.events}`], ["decisions", `${s.decisions}`],
      ["episodes ended", `${s.episodes}`], ["goal", GOAL_NAMES[v.goal] || `${v.goal}`], ["action", v.decisionAction === null ? "—" : ACTIONS[v.decisionAction]], ["controller", s.controller],
      ["expansions", s.controller === "planner" ? `${s.expansions}` : "—"], ["plan", v.search ? `${v.search.kind} · ${v.search.path.length} actions` : "—"], ["last reward", `${s.reward.toFixed(3)}`], ["consequences", s.correct ? `${s.correct[0]}/${s.correct[1]}` : "—"],
      ["deadline", v.deadline === null ? "—" : `${Math.max(0, v.deadline - v.tick)} left`], ["word shown", v.word ? `${v.word}` : "none"], ["doors", v.doorRule], ["exploration ε", item.epsilon.toFixed(2)], ["compute", `${s.compute_ms.toFixed(0)} ms`],
    ].map(([k, val]) => `<div><span>${k}</span><b>${val}</b></div>`).join("");
    const st = item.stores;
    $("places").innerHTML = st.places.map((r) => { const truth = st.truth[r.object]; const ok = r.where && truth && sameCell(r.where, truth); return `<div class="row"><i style="background:${OBJECT_COLORS[v.colors[r.object]]}"></i><span>object ${OBJECT_NAMES[r.object]}</span><b>${r.where ? `remembered at ${cellText(r.where)}` : "unknown"}</b><span>known ${r.known.toFixed(2)}</span><span class="${truth ? (ok ? "ok" : r.where ? "bad" : "") : ""}">truth ${truth ? cellText(truth) : "carried"}${truth && r.where ? (ok ? " ✓" : " ✗") : ""}</span></div>`; }).join("") + `<div class="row muted"><span>place writes ${st.placeWrites}</span><span>map: ${st.mapKnown} cells known · ${st.mapWrites} writes</span></div>`;
    $("cue").innerHTML = `<div class="row"><span>held word</span><b>${st.cue ? `${st.cue}${CUE_WORDS.includes(st.cue) ? " (a cue word)" : ""}` : "none"}</b><span>shown now: ${v.word || "none"}</span></div><div class="bars">${st.cueValue.map((x, k) => `<i title="word ${k}" style="height:${Math.round(4 + 14 * x)}px;background:${x > 0 ? "#c792ea" : "#1c2a3a"}"></i>`).join("")}</div>`;
    $("wordsTable").innerHTML = `<table><tr><th>word</th>${OBJECT_NAMES.map((n, k) => `<th><i style="background:${OBJECT_COLORS[v.colors[k]]}"></i>${n}</th>`).join("")}<th>referent</th></tr>` + st.words.map((row, w) => { const ref = st.referents[w]; return `<tr><td>${w + 1}</td>${row.map((x) => `<td style="background:rgba(89,229,203,${Math.min(1, Math.max(0, x)).toFixed(2)})">${x.toFixed(2)}</td>`).join("")}<td>${ref.object === null ? "—" : OBJECT_NAMES[ref.object]}</td></tr>`; }).join("") + `</table><div class="row muted"><span>scene writes ${st.wordWrites}</span></div>`;
    drawRecall(st.recall);
    const d = s.dopamine, half = Math.min(50, (Math.abs(d) / 0.25) * 50);
    const fill = $("dopamineFill");
    fill.className = "fill" + (d < 0 ? " negative" : "");
    fill.style.left = d < 0 ? `${50 - half}%` : "50%";
    fill.style.width = `${half}%`;
    const worldRow = agent.records ? ["records written", s.learning.records_written === undefined ? "—" : `${s.learning.records_written} fields`] : ["world step", s.learning.world_scale_step === undefined ? "—" : s.learning.world_scale_step.toExponential(2)];
    $("critic").innerHTML = [["dopamine (clipped TD)", d.toFixed(5)], ["|TD error|", s.td_error.toFixed(5)], ["critic bias", agent.bCritic.toFixed(5)], worldRow, ["replayed", s.learning.replayed === undefined ? "—" : `${s.learning.replayed}`]].map(([k, val]) => `<span>${k}</span><b>${val}</b>`).join("");
    const lp = ledgerPhase || agent.lastPhase, L = item.ledger;
    const unconverged = Object.values(L.unconverged_phases).reduce((a, b) => a + b, 0);
    $("ledger").innerHTML = [
      ["last phase", lp ? `${lp.phase} · ${lp.steps} steps · residual ${lp.residual.toExponential(2)} ${lp.converged ? "✓" : "✗"}` : "—"],
      ["rejected updates", `${L.rejected_updates}`], ["unconverged phases", `${unconverged}`], ["parameter version", `${item.parameter_version}`],
      agent.records ? ["records", `${L.record_writes} writes · no ring`] : ["replay ring", `${item.ring} transitions`],
      ["real transitions", `${L.real_transitions}`], [agent.records ? "imagined reads" : "imagined", `${L.imagined}`], ["phases", Object.entries(L.phases).map(([k, val]) => `${k} ${val}`).join(" · ")], ["searches", `${brain.searches} · reused ${brain.reused} · fallbacks ${brain.planner.fallbacks}`],
    ].map(([k, val]) => `<span>${k}</span><b>${val}</b>`).join("");
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

  // -- the recall port's drive from the stores: the place read, the cue, the word read
  const recallCanvas = $("recall");
  function drawRecall(values) {
    const r = recallCanvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    if (!r.width) return;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (recallCanvas.width !== w || recallCanvas.height !== h) { recallCanvas.width = w; recallCanvas.height = h; }
    const ctx = recallCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#05090e"; ctx.fillRect(0, 0, r.width, r.height);
    const n = values.length, bw = r.width / n, gain = brain.recallGain || 1;
    const groups = [[0, SIZE * SIZE + 1, "#8fe19d", "place read"], [SIZE * SIZE + 1, SIZE * SIZE + 1 + WORDS + 1, "#c792ea", "cue"], [SIZE * SIZE + 1 + WORDS + 1, n, "#59e5cb", "word read"]];
    for (const [a, b, color, label] of groups) {
      ctx.fillStyle = color;
      for (let k = a; k < b; k++) { const x = Math.min(1, Math.max(0, values[k] / gain)); ctx.globalAlpha = 0.25 + 0.75 * x; ctx.fillRect(k * bw + 0.5, r.height - 14 - x * (r.height - 18), Math.max(1, bw - 1), x * (r.height - 18) + 1); }
      ctx.globalAlpha = 1; ctx.font = "10px ui-monospace, monospace"; ctx.textAlign = "left"; ctx.fillText(label, a * bw + 2, r.height - 3);
      ctx.strokeStyle = "#1c2a3a"; ctx.beginPath(); ctx.moveTo(b * bw, 0); ctx.lineTo(b * bw, r.height); ctx.stroke();
    }
  }

  // -- the world canvas
  const worldCanvas = $("world");
  function drawWorld() {
    const r = worldCanvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    if (!r.width) return;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (worldCanvas.width !== w || worldCanvas.height !== h) { worldCanvas.width = w; worldCanvas.height = h; }
    const ctx = worldCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#05090e"; ctx.fillRect(0, 0, r.width, r.height);
    const margin = 22, cs = (Math.min(r.width, r.height) - 2 * margin - 18) / SIZE;
    const X = (x) => margin + x * cs, Y = (y) => margin + y * cs, CX = (c) => X(c[0]) + cs / 2, CY = (c) => Y(c[1]) + cs / 2;
    ctx.font = "10px ui-monospace, monospace"; ctx.fillStyle = "#4f6479"; ctx.textAlign = "center";
    for (let k = 0; k < SIZE; k++) { ctx.fillText(`${k}`, X(k) + cs / 2, margin - 8); ctx.fillText(`${k}`, margin - 10, Y(k) + cs / 2 + 3); }
    if (!worldView) { ctx.fillStyle = "#6f8397"; ctx.fillText("press Play", r.width / 2, r.height / 2); return; }
    const v = worldView;
    // cells: visited trail, the neighbourhood the last inspect revealed, the rooms of the cue task
    for (const c of ROOMS) { ctx.fillStyle = "rgba(237,129,182,0.08)"; ctx.fillRect(X(c[0]) + 1, Y(c[1]) + 1, cs - 2, cs - 2); }
    for (const c of v.visible) { ctx.fillStyle = "rgba(143,225,157,0.16)"; ctx.fillRect(X(c[0]) + 1, Y(c[1]) + 1, cs - 2, cs - 2); }
    ctx.strokeStyle = "#101a26"; ctx.lineWidth = 1;
    for (let k = 0; k <= SIZE; k++) { ctx.beginPath(); ctx.moveTo(X(k), Y(0)); ctx.lineTo(X(k), Y(SIZE)); ctx.moveTo(X(0), Y(k)); ctx.lineTo(X(SIZE), Y(k)); ctx.stroke(); }
    if (v.trail.length > 1) { ctx.strokeStyle = "rgba(143,225,157,0.35)"; ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(CX(v.trail[0]), CY(v.trail[0])); for (const c of v.trail.slice(1)) ctx.lineTo(CX(c), CY(c)); ctx.stroke(); }
    // walls and doors on the edges
    const doorKeys = new Set(v.doors.map(([a, b]) => `${a}|${b}`));
    for (const [a, b, kind] of v.edges) {
      const vertical = a[1] === b[1]; // (x, y)-(x+1, y): a vertical boundary at X(x+1)
      const x0 = vertical ? X(a[0] + 1) : X(a[0]), y0 = vertical ? Y(a[1]) : Y(a[1] + 1), x1 = vertical ? X(a[0] + 1) : X(a[0] + 1), y1 = vertical ? Y(a[1] + 1) : Y(a[1] + 1);
      const isDoor = doorKeys.has(`${a}|${b}`);
      if (kind === "wall") { ctx.strokeStyle = "#8fa3b8"; ctx.lineWidth = 4; ctx.setLineDash([]); }
      else if (kind === "door_closed") { ctx.strokeStyle = "#ff8c5a"; ctx.lineWidth = 4; ctx.setLineDash([]); }
      else if (isDoor) { ctx.strokeStyle = "#8fe19d"; ctx.lineWidth = 2; ctx.setLineDash([4, 4]); }
      else continue;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.strokeStyle = "#8fa3b8"; ctx.lineWidth = 4; ctx.strokeRect(X(0), Y(0), SIZE * cs, SIZE * cs);
    // the junction of the cue task
    ctx.fillStyle = "#6f8397"; ctx.font = "9px ui-monospace, monospace"; ctx.fillText("junction", CX([1, 1]), Y(1) + 10); ctx.fillText("room", CX(ROOMS[0]), Y(ROOMS[0][1]) + 10); ctx.fillText("room", CX(ROOMS[1]), Y(ROOMS[1][1]) + 10);
    // remembered places (the PlaceStore) as diamonds, the truth as discs
    if (storeView) for (const rec of storeView.places) if (rec.where) { const c = rec.where; ctx.strokeStyle = OBJECT_COLORS[v.colors[rec.object]]; ctx.lineWidth = 1.5; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.moveTo(CX(c), CY(c) - cs * 0.42); ctx.lineTo(CX(c) + cs * 0.42, CY(c)); ctx.lineTo(CX(c), CY(c) + cs * 0.42); ctx.lineTo(CX(c) - cs * 0.42, CY(c)); ctx.closePath(); ctx.stroke(); ctx.setLineDash([]); }
    for (const [k, c] of Object.entries(v.objects)) if (c) { ctx.fillStyle = OBJECT_COLORS[v.colors[k]]; ctx.beginPath(); ctx.arc(CX(c) - cs * 0.22, CY(c) + cs * 0.2, cs * 0.16, 0, Math.PI * 2); ctx.fill(); ctx.fillStyle = "#05090e"; ctx.font = `bold ${Math.round(cs * 0.2)}px system-ui, sans-serif`; ctx.fillText(OBJECT_NAMES[k], CX(c) - cs * 0.22, CY(c) + cs * 0.2 + cs * 0.07); }
    // targets: the request's true target (privileged, for the viewer) and the target the search aimed at
    if (v.requestTarget) { ctx.strokeStyle = "#ed81b6"; ctx.lineWidth = 2; ctx.beginPath(); ctx.arc(CX(v.requestTarget), CY(v.requestTarget), cs * 0.44, 0, Math.PI * 2); ctx.stroke(); ctx.fillStyle = "#ed81b6"; ctx.font = "9px ui-monospace, monospace"; ctx.fillText("target", CX(v.requestTarget), Y(v.requestTarget[1] + 1) - 4); }
    if (v.rewarded) { ctx.fillStyle = "#ed81b6"; ctx.font = "9px ui-monospace, monospace"; ctx.fillText("pays", CX(v.rewarded), Y(v.rewarded[1] + 1) - 4); }
    if (v.target && !(v.requestTarget && sameCell(v.target, v.requestTarget))) { ctx.strokeStyle = "#59e5cb"; ctx.lineWidth = 1.5; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(CX(v.target), CY(v.target), cs * 0.36, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = "#59e5cb"; ctx.font = "9px ui-monospace, monospace"; ctx.fillText(v.goal === GOAL_EXPLORE || v.goal === GOAL_CUE ? "explore" : "believed", CX(v.target), Y(v.target[1] + 1) - 4); }
    // the imagined path of the last search
    if (v.search && v.search.cells.length) {
      const from = v.trail.length > 1 && v.controller === "planner" ? v.trail[v.trail.length - 2] : v.agent;
      ctx.strokeStyle = v.search.kind === "reuse" ? "#ffbf70" : "#59e5cb"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(CX(from), CY(from));
      for (const c of v.search.cells) ctx.lineTo(CX(c), CY(c));
      ctx.stroke();
      const last = v.search.cells[v.search.cells.length - 1];
      ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(CX(last), CY(last), 4, 0, Math.PI * 2); ctx.fill();
    }
    // the agent and what it carries
    const a = v.agent;
    ctx.fillStyle = "#dbe6f0"; ctx.beginPath(); ctx.arc(CX(a), CY(a), cs * 0.2, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#05090e"; ctx.lineWidth = 2; ctx.stroke();
    if (v.carrying !== null) { ctx.fillStyle = OBJECT_COLORS[v.colors[v.carrying]]; ctx.beginPath(); ctx.arc(CX(a) + cs * 0.22, CY(a) - cs * 0.22, cs * 0.12, 0, Math.PI * 2); ctx.fill(); }
    if (v.decisionAction !== null) {
      const name = ACTIONS[v.decisionAction];
      if (name in DIRECTIONS) { const [dx, dy] = DIRECTIONS[name]; ctx.strokeStyle = "#ffbf70"; ctx.lineWidth = 2.5; ctx.beginPath(); ctx.moveTo(CX(a), CY(a)); ctx.lineTo(CX(a) + dx * cs * 0.4, CY(a) + dy * cs * 0.4); ctx.stroke(); }
      else { ctx.fillStyle = "#ffbf70"; ctx.font = "9px ui-monospace, monospace"; ctx.fillText(name, CX(a), CY(a) + cs * 0.4); }
    }
    ctx.fillStyle = "#6f8397"; ctx.font = "11px system-ui, sans-serif"; ctx.textAlign = "left";
    ctx.fillText(`tick ${v.tick}/${v.horizon}${v.deadline === null ? "" : ` · ${Math.max(0, v.deadline - v.tick)} to deadline`} · word ${v.word || "–"} · cue ${v.cueHeld || "–"} · doors ${v.doorRule}${v.closing ? " · closed" : ""}`, 8, r.height - 6);
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
    drawWorld();
    if (records) records.draw();
    requestAnimationFrame(tick);
  }
  $("play").onclick = () => { playing = !playing; stepping = false; $("play").textContent = playing ? "Pause" : "Play"; };
  $("step").onclick = () => { playing = false; $("play").textContent = "Play"; computeEvent(); stepping = true; };
  $("fit").onclick = () => scan.fit();
  $("view").onchange = () => { scan.options.mode = $("view").value; scan.draw(); };
  $("imagine").onchange = () => { imagineEvery = Number($("imagine").value); };
  $("phaseSelect").onchange = () => { requestedPhase = $("phaseSelect").value; $("status").textContent = `${requestedPhase} from the next episode on (the current episode finishes first)`; };

  pushFrame({ phase: "curriculum", stage: "rest" }, null, null, null); // the map as loaded, before the first event
  drain(0);
  $("counts").textContent = `${scan.n.toLocaleString()} neurons · ${scan.edges.toLocaleString()} synapses · ${scan.atlas.regions.length} regions${agent.records ? ` · ${agent.records.granules.toLocaleString()} granules` : ""} · parameter version ${agent.parameterVersion}`;
  $("successes").innerHTML = "<span>no episode finished yet</span>";
  $("phase").textContent = `resting · ${extra.steps_before ?? "the snapshot's"} explore steps before this page · press Play`;
  scan.fit();
  requestAnimationFrame(tick);

  window.__page = {
    scan, agent, brain, world, cur, queue, records,
    get stats() { return { ...stats, successes: JSON.parse(JSON.stringify(stats.successes)), parameter_version: agent.parameterVersion, ring: agent.ring.length, record_writes: agent.ledger.record_writes, imagined: agent.ledger.imagined, records: records ? { ...records.counts, active: records.activeIndex.length } : null, episode: world.episode, tick: world.tick, queued: queue.length, epsilon: brain.epsilon, searches: brain.searches, reused: brain.reused, place_writes: brain.places.writes, map_writes: brain.map.writes, word_writes: brain.words.writes, renderer: scan.snapshot() }; },
    /** One event, computed and shown at once (the headless check drives the page with this). */
    step() { computeEvent(); drain(0); scan.draw(); drawWorld(); if (records) records.draw(); return window.__page.stats; },
    ready: true,
  };
  window.__brainScan = scan;
}

loadSnapshot().then(main).catch((err) => { $("counts").textContent = `error: ${err.message}`; $("phase").textContent = "failed to load"; console.error(err); });
