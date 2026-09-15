// The S02 world page: the world on the left, the whole brain beside it. The visitor asks for an
// object, drags one to another cell, teaches a word and asks by it, runs the cue task and locks
// the doors; the agent answers with its own decisions. Every action comes from
// ExperienceAgent.step through the same Brain wrapper that passes the parity check: the stores
// read into the recall port, the planner searches over cells through the consequences the
// records cortex imagines, and the outcome of every decision is written into the records that
// were read. Beside the world the whole brain runs through the standard BrainScan renderer and
// the records-cortex view: every settling step of every phase is queued by the engine's
// callbacks and played back, synapses flash when an update moves them, and the ledger and the
// dopamine bar follow the events in the same order. The run.py curriculum (explore, remembered
// requests at delays 8, 16 and 32, the seen and unseen corrections, the cue task at delays 8
// and 32, the locked doors, the words) is the second control, in the order run.py runs it. The
// brain selector loads another point of this life from the checkpoint manifest beside the page.
// Nothing is scripted: the agent reaches only what its stores and its learned model let it reach.

import { BrainScan } from "../../web/brain_scan.js";
import { ExperienceAgent } from "../../web/engine.js";
import { RecordsView, pathwayNames, readingGroups, readsHTML } from "../../web/records_view.js";
import { ACTIONS, CUE_WORDS, DIRECTIONS, DIRECTION_NAMES, GOAL_CUE, GOAL_EXPLORE, OBJECTS, OUTCOMES, ROOMS, SIZE, WORDS, World, Curriculum, argmaxOf, cellKey, episodeRecord, goalVector, sameCell } from "./world.js";
import { Brain } from "./planner.js";

const $ = (id) => document.getElementById(id);
const PAGE_OPTIONS = { autoplay: true, ...(window.__WORLD_PAGE__ || {}) };
const CHECKPOINT_FORMAT = "cadence-world-checkpoints/1";
const PHASE_TEXT = { free: "free phase", free_post: "free phase after learning", predict: "predict (the chosen action's consequence)", imagine: "imagine (the search's batch of consequences)", repair_free: "world repair · free", repair_plus: "world repair · nudged +β", repair_minus: "world repair · nudged −β", bootstrap: "bootstrap value (time limit)", score_plus: "actor score · +β", score_minus: "actor score · −β" };
const REGION_NOTE = { sensory: "the cell, the passages, what is carried", goal: "what is asked for", efference: "copy of the executed action", workspace: "association", dynamics: "reads the efference copy", context: "trace of the workspace", motor: "action readout", prediction: "predicted consequences", recall: "what the stores read", perception: "perception", intention: "intention" };
const OBJECT_COLORS = ["#ff6b6b", "#59e5cb", "#ffd166", "#c792ea"]; // the world's four colours, by colour index
const OBJECT_NAMES = ["A", "B", "C", "D"];
const GOAL_NAMES = ["explore", "reach A", "reach B", "reach C", "reach D", "reach the named object", "the cued room", "fetch"];
const INSPECTED_OUTCOME = OUTCOMES.indexOf("inspected");
const PAGE_BUDGET = { explore_steps: 100, remember_episodes: 5, correction_episodes: 3, cue_episodes: 6, door_trials: 4, word_episodes: 4, teach_steps: 24 };
const DEMO_AFTER = 70; // wandering events before the page asks for an object by itself
const PLAN_COLOR = "#59e5cb", REUSE_COLOR = "#ffbf70", TARGET_COLOR = "#ed81b6", AGENT_COLOR = "#dbe6f0";

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

  // -- the life, rebuilt whenever another checkpoint is loaded
  let snapshot = null, checkpoint = { id: "inlined", label: "this page's brain" };
  let agent = null, brain = null, records = null, world = null, cur = null, life = null;
  let predictionFields = [], lastPrediction = null, lastWrite = null;
  let lastS = null, lastV = null, previousWeights = null;
  let stats = freshStats();

  // -- the page's own state
  let current = null, requested = null, demo = false, demoIdle = 0;
  let phaseEpsilon = 1.0, pageEpsilon = 0, decisionEpsilon = 0.1;
  let running = PAGE_OPTIONS.autoplay !== false, imagineEvery = 4;
  let worldView = null, storeView = null, currentPhase = null, ledgerPhase = null;
  let phaseText = "—", phaseClass = "", lastStepText = "", flashText = "", flashUntil = 0;
  let dragging = null, hover = null, selected = null; // the object under the pointer and the one being moved
  let viewFrame = { x0: 22, y0: 56, cs: 1 };
  const results = []; // every finished task: {kind, label, ok, decisions, brain}
  const trail = [];
  const checkpoints = [], snapshotCache = new Map();
  const worldCanvas = $("world"), recallCanvas = $("recall");
  scan.onhover = (hit) => { if (hit) $("inspector").textContent = `${hit.region} · neuron ${hit.neuron}\nactivation ${hit.activation.toFixed(4)} · change ${hit.change.toExponential(2)}` + (hit.potential === null ? "" : ` · potential ${hit.potential.toFixed(4)}`); };

  function freshStats() {
    return { events: 0, decisions: 0, episodes: 0, reward: 0, controller: "—", expansions: 0, correct: null, dopamine: 0, td_error: 0, compute_ms: 0, rejected: 0, learning: {}, last_decision: null, successes: {}, task: "wander" };
  }

  // ---------------------------------------------------------------- the life

  /** A checkpoint exported without the page's extras borrows them from the inlined snapshot. */
  function fillExtras(snap) {
    snap.extra = snap.extra || {};
    for (const key of ["config", "planner", "world", "curriculum", "recall"]) if (snap.extra[key] === undefined && FIRST.extra) snap.extra[key] = FIRST.extra[key];
    return snap;
  }

  function installLife(snap, label, id) {
    fillExtras(snap);
    snapshot = snap;
    checkpoint = { id, label };
    const extra = snap.extra, stage = extra.config || {};
    queue.length = 0; currentPhase = null; ledgerPhase = null;
    agent = new ExperienceAgent(snap, {
      onSettleStep: (phase, s, v, meta) => { if (phase === "imagine" && imagineEvery > 1 && meta.step % imagineEvery !== 1) return; queue.push({ kind: "step", phase, s: Float32Array.from(s), v: Float32Array.from(v), batch: meta.batch, step: meta.step, moved: meta.moved }); },
      onPhase: (info) => queue.push({ kind: "phase", phase: info.phase, steps: info.steps, residual: info.residual, converged: info.converged, batch: info.batch }),
      onLearn: (info) => queue.push({ kind: "learn", phase: info.phase, changed: info.changed.length, rejected: info.rejected, dopamine: info.dopamine, td_error: info.td_error, weights: info.changed.length ? agent.effectiveWeights() : null }),
      onRecords: (info) => queue.push({ ...info, kind: "records", event: info.kind }), // event: "code" | "imagine" | "write"
    });
    brain = new Brain(agent, { ...extra, planner: { ...(extra.planner || {}), ...(stage.planner || {}) } }); // the stage's search budget, the snapshot's generator state
    predictionFields = agent.config.graph.prediction;
    lastPrediction = null; lastWrite = null;
    decisionEpsilon = stage.actor && stage.actor.epsilon !== undefined ? Number(stage.actor.epsilon) : 0.1; // run.py: the cue decision explores at the life's own rate
    // the brain in the scan and the records cortex beside it
    scan.setAtlas(snap.atlas);
    const rest = Float32Array.from(agent.warm ? agent.warm.v : new Float64Array(agent.n), (v) => agent._act(v));
    lastV = Float32Array.from(agent.warm ? agent.warm.v : new Float64Array(agent.n));
    lastS = rest;
    scan.set(rest, { potential: lastV, draw: false });
    previousWeights = agent.effectiveWeights();
    scan.setWeights(previousWeights);
    scan.fit();
    installRecords(snap);
    buildLegend();
    // the world: kept as the visitor left it when this brain lived in the same map, rebuilt from
    // the snapshot's dump otherwise (another life, another permutation of walls, doors and colours)
    const dump = extra.world, worldConfig = stage.world_config || {};
    if (world === null || !sameMap(world, dump)) {
      world = dump ? World.fromState(dump, { seed: 0x5eed, life_id: "web" }) : new World(worldConfig, { seed: 0x5eed, life_id: "web" });
      if (worldConfig.horizon) world.config.horizon = worldConfig.horizon | 0;
      cur = new Curriculum(world, 0xa11ce, { cueRule: extra.curriculum && extra.curriculum.cue_rule !== undefined ? extra.curriculum.cue_rule : null });
      trail.length = 0;
    }
    if (agent.pending[0] !== null) agent.abandon(0); // the page starts fresh episodes on this world
    world.continueFrom({ event: agent.lastEvent[0] + 1, episode: agent.episode[0] }); // the next moment comes after everything this brain has seen
    stats = freshStats();
    life = driver();
    current = null; requested = null;
    setPhaseEpsilon(1.0);
    worldView = null; storeView = null;
    pushFrame({ task: "wander", stage: "rest" }, null, null, null); // the world as loaded, before the first event
    drain(0);
    updateFacts();
    updateHud();
  }

  /** Two snapshots share a map when their walls, doors and colours are the same permutation. */
  function sameMap(w, dump) {
    if (!dump) return true;
    const state = w.state();
    return JSON.stringify(state.edges) === JSON.stringify(dump.edges) && JSON.stringify(state.colors) === JSON.stringify(dump.colors) && JSON.stringify(state.words) === JSON.stringify(dump.words);
  }

  function installRecords(snap) {
    records = null;
    if (!agent.records) { $("recordsPanel").hidden = true; return; }
    const cfg = agent.config.records, rc = snap.records || {};
    records = new RecordsView($("records"), { granules: agent.records.granules, active: cfg.active, inputs: agent.reading.length, groups: readingGroups(snap), pathways: pathwayNames(snap), color: rc.color || undefined, rate: cfg.rate, rewardRate: cfg.reward_rate, habituation: cfg.habituation });
    $("recordsPanel").hidden = false;
    $("recordsInfo").textContent = `the world model · ${agent.records.granules.toLocaleString()} cells, ${cfg.active} active · ${agent.records.parameters().toLocaleString()} records in ${predictionFields.length} fields`;
    $("recordReads").innerHTML = readsHTML(null, predictionFields);
  }

  function buildLegend() {
    const items = snapshot.atlas.regions.map((r) => `<span title="${r.name}"><i style="background:rgb(${r.color.join(",")})"></i><b>${r.label ?? r.name}</b><em>${r.count}</em>${REGION_NOTE[r.name] ? ` ${REGION_NOTE[r.name]}` : ""}</span>`);
    if (records) items.push(`<span title="the world model"><i style="background:rgb(${records.spec.color.join(",")})"></i><b>records cortex</b><em>${agent.records.granules.toLocaleString()}</em> the world model</span>`);
    $("legend").innerHTML = items.join("");
  }

  function updateFacts() {
    const lived = (snapshot.ledger && snapshot.ledger.real_transitions) || 0;
    $("counts").textContent = `${checkpoint.label} · ${lived.toLocaleString()} decisions lived before this page\n${agent.n.toLocaleString()} neurons · ${agent.E.toLocaleString()} synapses · ${snapshot.atlas.regions.length} regions${agent.records ? ` · ${agent.records.granules.toLocaleString()} cells` : ""}`;
    const receipt = (snapshot.extra && snapshot.extra.receipt) || null;
    if (receipt) {
      const m = receipt.metrics || {};
      const value = (name, digits = 2) => (m[name] === undefined || m[name] === null ? "—" : Number(m[name]).toFixed(digits));
      $("receiptNote").innerHTML = `<b>The receipt.</b> This brain is seed ${receipt.seed} of ${receipt.run_id} (${receipt.split} split, ${receipt.steps.toLocaleString()} real steps). Its held-out rates on that run: requests at delay 32 ${value("request_success_32")} · the correction after a move in view ${value("correction_visible")} · the cue at delay 8 ${value("cue_delay8")} · a word as a request ${value("grounding")} · navigation after the doors locked ${value("door_recovery")} (last quarter) · the five visible consequences ${value("consequence_accuracy", 3)}. With the place records erased the same brain reaches ${value("erased_places", 2)}; frozen from birth it reaches ${value("born_frozen", 2)}.`;
    }
  }

  // ---------------------------------------------------------------- one event

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

  function tally(name, ok) { const s = stats.successes[name] || (stats.successes[name] = { done: 0, ok: 0 }); s.done += 1; s.ok += ok; }

  function record(kind, label, ok, decisions, text = null) {
    results.push({ kind, label, ok, decisions, text: text || label, brain: checkpoint.label });
    if (results.length > 24) results.shift();
    updateHud();
  }

  const switchRequested = () => requested !== null;

  function setPhaseEpsilon(value) {
    phaseEpsilon = value;
    brain.setEpsilon(value >= 1 ? 1.0 : Math.max(value, pageEpsilon));
  }

  // ---------------------------------------------------------------- the tasks the visitor starts

  /** Every object is somewhere: the cue task clears the world, and a page the visitor acts on keeps
   * its four objects. Only objects that are nowhere are placed, in free cells. */
  function scatterMissing() {
    const objects = world.objects(), taken = new Set([cellKey(world.agent)]);
    for (const c of Object.values(objects)) if (c) taken.add(cellKey(c));
    for (let k = 0; k < OBJECTS; k++) {
      if (objects[k] || world.carrying === k) continue;
      let cell = world.randomCell();
      for (let guard = 0; guard < 200 && taken.has(cellKey(cell)); guard++) cell = world.randomCell();
      taken.add(cellKey(cell));
      world.placeObject(k, cell);
    }
  }

  /** The objects the page held before a task cleared them, back in their cells when those are free. */
  function restoreObjects(saved) {
    const here = world.objects(), taken = new Set([cellKey(world.agent)]);
    for (const c of Object.values(here)) if (c) taken.add(cellKey(c));
    for (let k = 0; k < OBJECTS; k++) {
      const cell = saved[k];
      if (!cell || here[k] || world.carrying === k || taken.has(cellKey(cell))) continue;
      taken.add(cellKey(cell));
      world.placeObject(k, cell);
    }
  }

  /** The wander's setting: no goal, no reward, every action legal, and the objects stay where they are
   * (run.py's explore scatters them again; on this page the world is the visitor's). */
  function wanderSetup() {
    scatterMissing();
    world.setMask(null);
    world.setWord(0);
    world.setGoal(goalVector(GOAL_EXPLORE));
    world.rewardRule = null;
  }

  /** Free wandering, exploration rate one: what fills the stores. It ends when the visitor asks for something. */
  function* runWander() {
    setPhaseEpsilon(1.0);
    while (!switchRequested()) {
      wanderSetup();
      let m = world.reset();
      const info = { task: "wander", stage: "wander" };
      for (;;) {
        demoIdle += 1;
        if (demo && demoIdle > DEMO_AFTER) demoAsk();
        const r = yield* transition(m, info);
        if (r.d === null) break;
        m = r.next;
        if (switchRequested()) { yield* closeEpisode(m, info); break; }
      }
    }
  }

  /** "Ask for A": a new episode of the same life from a distant cell, the goal naming the object. */
  function* runAsk(task) {
    const k = task.object;
    scatterMissing(); // the cue task empties the world; an object asked for is in it
    const here = world.objects()[k];
    const episode = episodeRecord("remember", { object_id: k, old_cell: here || world.agent });
    setPhaseEpsilon(0.0);
    const m = cur.request(episode);
    const info = { task: "ask", stage: "request", object: k, target: episode.target_cell };
    const outcome = yield* finishEpisode(m, info);
    tally("ask", outcome.terminated ? 1 : 0);
    record("ask", OBJECT_NAMES[k], outcome.terminated, outcome.steps, `asked for ${OBJECT_NAMES[k]}`);
  }

  /** The cue task: the word shown at the junction, a wander of `delay` steps with it hidden, then the two-choice decision. */
  function* runCue(delay, episodes) {
    for (let e = 0; e < episodes; e++) {
      if (e && switchRequested()) return;
      const episode = cur.cue(delay);
      let m = world.reset({ agent: world.config.junction });
      setPhaseEpsilon(1.0);
      const info = { task: "cue", stage: "wander", episode: e + 1, episodes, delay, cue: CUE_WORDS[episode.cue] };
      let closed = false;
      for (let t = 0; t < delay; t++) {
        const r = yield* transition(m, { ...info, t: t + 1 }, t === 0 ? () => cur.hideCue() : null);
        if (r.d === null) { closed = true; break; }
        m = r.next;
      }
      setPhaseEpsilon(decisionEpsilon);
      if (closed) { tally("cue", 0); record("cue", `c${delay}`, false, 0, `the cue task at delay ${delay}`); continue; }
      yield* closeEpisode(m, info);
      m = cur.armCue(episode);
      const outcome = yield* finishEpisode(m, { ...info, stage: "decision", rewarded: episode.rewarded_room });
      tally("cue", outcome.reward >= 1.0 ? 1 : 0);
      record("cue", `c${delay}`, outcome.reward >= 1.0, outcome.steps, `the cue task at delay ${delay}`);
    }
  }

  /** Words: the word shown while its object is in view (joint attention), then the word alone as a request. */
  function* runWords(episodes) {
    for (let e = 0; e < episodes; e++) {
      if (e && switchRequested()) return;
      const episode = cur.word();
      let m = world.reset({ agent: episode.old_cell });
      setPhaseEpsilon(1.0);
      const info = { task: "word", stage: "teach", episode: e + 1, episodes, word: episode.cue, object: episode.object_id, delay: PAGE_BUDGET.teach_steps };
      let closed = false;
      for (let t = 0; t < PAGE_BUDGET.teach_steps; t++) {
        const r = yield* transition(m, { ...info, t: t + 1 });
        if (r.d === null) { closed = true; break; }
        m = r.next;
      }
      setPhaseEpsilon(0.0);
      if (closed) { tally("word", 0); record("word", `w${episode.cue}`, false, 0, `word ${episode.cue} for ${OBJECT_NAMES[episode.object_id]}`); continue; }
      yield* closeEpisode(m, info);
      m = cur.nameRequest(episode);
      const outcome = yield* finishEpisode(m, { ...info, stage: "name request", target: episode.target_cell });
      tally("word", outcome.terminated ? 1 : 0);
      record("word", `w${episode.cue}`, outcome.terminated, outcome.steps, `word ${episode.cue} for ${OBJECT_NAMES[episode.object_id]}`);
    }
  }

  // ---------------------------------------------------------------- run.py's own phases

  function* runExplore(steps) {
    let done = 0;
    setPhaseEpsilon(1.0);
    while (done < steps && !switchRequested()) {
      cur.explore();
      const m = world.reset();
      const outcome = yield* finishEpisode(m, { task: "explore", stage: "explore", done, steps });
      done += outcome.steps;
      tally("explore", 1);
    }
  }

  function* runRemember(delay, episodes, move, key = null) {
    const phase = key || `remember-${delay}${move ? "-" + move : ""}`;
    for (let e = 0; e < episodes; e++) {
      if (switchRequested()) return;
      const episode = cur.remember(delay, { move });
      let m = world.reset({ agent: episode.target_cell });
      setPhaseEpsilon(1.0);
      const info = { task: phase, stage: "wander", episode: e + 1, episodes, delay, object: episode.object_id };
      let closed = false, moved = false;
      for (let t = 0; t < delay; t++) {
        const r = yield* transition(m, { ...info, t: t + 1 });
        if (r.d === null) { closed = true; break; }
        m = r.next;
        if (move === "visible" && !moved && t >= Math.floor(delay / 2) && world.objectAt(world.agent) === null) { moved = true; cur.moveObject(episode, { visible: true }); const seen = world.observation(); m = { ...m, observation: seen.observation, observed: seen.observed }; } // onto the agent's own empty cell, in view before its next decision (run.py run_remember)
        if (move === "invisible" && t === Math.floor(delay / 2)) cur.moveObject(episode, { visible: false });
      }
      setPhaseEpsilon(0.0);
      if (closed) { tally(phase, 0); continue; }
      yield* closeEpisode(m, info);
      m = cur.request(episode);
      const outcome = yield* finishEpisode(m, { ...info, stage: "request", target: episode.target_cell });
      tally(phase, outcome.terminated ? 1 : 0);
      record("ask", OBJECT_NAMES[episode.object_id], outcome.terminated, outcome.steps, `asked for ${OBJECT_NAMES[episode.object_id]}`);
    }
  }

  const B = PAGE_BUDGET;
  const RUNNERS = {
    wander: runWander,
    ask: runAsk,
    word: function* () { yield* runWords(1); },
    cue: function* (task) { const saved = world.objects(); yield* runCue(task.delay, 1); restoreObjects(saved); }, // the cue task clears the world; the visitor's objects go back
    curriculum: function* () {
      world.setDoorRule("toggle"); // a new cycle of the curriculum: the doors toggle again until the door phase locks them
      updateDoors();
      yield* runExplore(B.explore_steps);
      for (const delay of [8, 16, 32]) yield* runRemember(delay, B.remember_episodes, null);
      yield* runRemember(16, B.correction_episodes, "visible");
      yield* runRemember(16, B.correction_episodes, "invisible");
      for (const delay of [8, 32]) yield* runCue(delay, B.cue_episodes);
      yield* runRemember(8, B.door_trials, null, "door-before");
      cur.lockDoors(); updateDoors();
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
    "door-locked": function* () { cur.lockDoors(); updateDoors(); yield* runRemember(8, B.door_trials, null, "door-locked"); },
    words: function* () { yield* runWords(B.word_episodes); },
  };

  function* driver() {
    for (;;) {
      if (requested !== null) { current = requested; requested = null; }
      else if (current === null || !current.repeat) current = { kind: "wander", repeat: true };
      stats.task = current.kind;
      yield* RUNNERS[current.kind](current);
    }
  }

  function computeEvent() { life.next(); }

  /** The visitor's task starts at the next event; a wander closes its episode first. */
  function requestTask(task, { keepDemo = false } = {}) {
    requested = task;
    demoIdle = 0;
    if (!keepDemo) stopDemo();
    updateHud();
    return task;
  }

  function stopDemo() { demo = false; }

  function demoAsk() {
    const known = brain.places.records().filter((r) => r.where && r.known >= 0.5);
    demoIdle = 0;
    if (!known.length) return;
    requestTask({ kind: "ask", object: known[Math.floor(Math.random() * known.length)].object }, { keepDemo: true });
  }

  // ---------------------------------------------------------------- playback

  function applyRecords(item) {
    if (!records) return;
    if (item.event === "code") { records.code(item, "decide"); records.setReading(item.reading, item.blockNorm); lastPrediction = item.prediction; $("recordReads").innerHTML = readsHTML(lastPrediction, predictionFields, lastWrite); }
    else if (item.event === "imagine") records.code(item, "imagine");
    else if (item.event === "write") {
      records.write(item); lastWrite = item;
      $("recordReads").innerHTML = readsHTML(lastPrediction, predictionFields, lastWrite);
      setPhase(`records: ${item.written} field${item.written === 1 ? "" : "s"} written into ${item.index.length} cells${item.fields.some((f) => f.reward) ? " (reward)" : ""}`, "learn");
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

  const rows = (pairs) => pairs.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
  const cellText = (c) => (c ? `(${c[0]},${c[1]})` : "—");

  function applyEvent(item) {
    worldView = item.world; storeView = item.stores;
    const s = item.stats, v = item.world, st = item.stores;
    const d = s.dopamine, half = Math.min(50, (Math.abs(d) / 0.25) * 50);
    const fill = $("dopamineFill");
    fill.className = "fill" + (d < 0 ? " negative" : "");
    fill.style.left = d < 0 ? `${50 - half}%` : "50%";
    fill.style.width = `${half}%`;
    const worldRow = agent.records ? ["written now", s.learning.records_written === undefined ? "—" : `${s.learning.records_written} fields`] : ["world step", s.learning.world_scale_step === undefined ? "—" : s.learning.world_scale_step.toExponential(2)];
    $("critic").innerHTML = rows([["clipped TD", d.toFixed(4)], ["critic bias", agent.bCritic.toFixed(4)], worldRow, ["consequences", s.correct ? `${s.correct[0]}/${s.correct[1]} right` : "—"]]);
    const lp = ledgerPhase || agent.lastPhase, L = item.ledger;
    const unconverged = Object.values(L.unconverged_phases).reduce((a, b) => a + b, 0);
    $("ledger").innerHTML = rows([
      ["phase", lp ? `${lp.phase} · ${lp.steps} ${lp.converged ? "✓" : "✗"}` : "—"],
      ["rejected", `${L.rejected_updates} · ${unconverged} unconverged`],
      agent.records ? ["records", `${L.record_writes.toLocaleString()} writes`] : ["ring", `${item.ring}`],
      ["imagined", `${L.imagined.toLocaleString()} reads`],
      ["searches", `${brain.searches} · ${brain.reused} reused`],
    ]);
    $("counters").innerHTML = rows([
      ["episode", `${v.episode} · tick ${v.tick}/${v.horizon}`], ["goal", GOAL_NAMES[v.goal] || `${v.goal}`],
      ["action", `${v.decisionAction === null ? "—" : ACTIONS[v.decisionAction]} · ${s.controller}`],
      ["plan", v.search ? `${v.search.kind} · ${v.search.path.length} actions · ${s.expansions} expansions` : "—"],
      ["doors", `${v.doorRule} · ε ${item.epsilon.toFixed(2)} · ${s.compute_ms.toFixed(0)} ms`],
    ]);
    // the stores, under the world
    $("places").innerHTML = `<span class="tag">places</span>` + st.places.map((r) => {
      const truth = st.truth[r.object], ok = r.where && truth && sameCell(r.where, truth);
      const cls = !r.where ? "unknown" : ok ? "" : "stale";
      return `<span class="item ${cls}" title="${OBJECT_NAMES[r.object]}: remembered ${cellText(r.where)}, truth ${truth ? cellText(truth) : "carried"}, known ${r.known.toFixed(2)}"><i style="background:${OBJECT_COLORS[v.colors[r.object]]}"></i>${OBJECT_NAMES[r.object]} <b>${r.where ? cellText(r.where) : "not seen"}</b>${r.where ? (ok ? " ✓" : " ✗") : ""}</span>`;
    }).join("") + `<span class="item unknown">map <b>${st.mapKnown}/36 cells</b></span>`;
    const referents = st.referents.map((ref, w) => (ref.object === null ? null : `<span class="item" title="word ${w + 1} names ${OBJECT_NAMES[ref.object]} at strength ${ref.strength === undefined ? "" : ref.strength.toFixed(2)}"><i style="background:${OBJECT_COLORS[v.colors[ref.object]]}"></i>word ${w + 1} <b>${OBJECT_NAMES[ref.object]}</b></span>`)).filter(Boolean);
    $("wordsRow").innerHTML = `<span class="tag">words</span>` + (referents.length ? referents.join("") : `<span class="item unknown">nothing grounded yet</span>`)
      + (st.cue ? `<span class="item held">cue held <b>${st.cue}</b></span>` : "") + (v.word ? `<span class="item held">word shown <b>${v.word}</b></span>` : "");
    updateHud();
  }

  function drain(limit) {
    let steps = 0;
    while (queue.length && (limit === 0 || steps < limit)) {
      const item = queue.shift();
      if (item.kind === "step") {
        if (item.phase !== currentPhase) { currentPhase = item.phase; scan.reset(); setPhase((PHASE_TEXT[item.phase] || item.phase) + (item.batch > 1 ? ` · batch ${item.batch}, row 1 shown` : ""), ""); }
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

  // ---------------------------------------------------------------- the readout over the world

  function taskText(v) {
    const i = v.info;
    if (i.stage === "rest") return `resting · ${((snapshot.extra && snapshot.extra.steps_before) || 0).toLocaleString()} steps lived`;
    if (i.task === "ask" || i.stage === "request" || i.stage === "name request") return `asked for ${i.object !== undefined ? OBJECT_NAMES[i.object] : "the named object"} · target ${cellText(i.target)}`;
    if (i.stage === "close") return "the wander closes (truncated)";
    if (i.task === "cue") return i.stage === "decision" ? `the two-choice decision at the junction · ${cellText(i.rewarded)} pays` : `cue word ${i.cue} · wander ${i.t}/${i.delay}`;
    if (i.task === "word") return `word ${i.word} names ${OBJECT_NAMES[i.object]} · teaching ${i.t}/${i.delay}`;
    if (i.task === "wander") return `wandering · tick ${v.tick}/${v.horizon}`;
    if (i.task === "explore") return `explore · ${i.done}/${i.steps} steps · tick ${v.tick}/${v.horizon}`;
    return `${i.task} · ${i.stage}${i.episode ? ` · episode ${i.episode}/${i.episodes}` : ""}${i.t ? ` · ${i.t}/${i.delay}` : ""}`;
  }

  function updateHud() {
    const v = worldView, i = v ? v.info : { task: "wander", stage: "rest" };
    const known = storeView ? storeView.places.filter((r) => r.where).length : 0;
    let label = "Wander", value = `${known}/4`, unit = "places it remembers", progress = 0;
    if (i.stage === "request" || i.stage === "name request") {
      const left = v.deadline === null ? null : Math.max(0, v.deadline - v.tick);
      label = i.stage === "name request" ? `Asked by word ${i.word}` : `Asked for ${OBJECT_NAMES[i.object] ?? "?"}`;
      value = left === null ? "—" : `${left}`;
      unit = `decisions left of the deadline · target ${cellText(i.target)}`;
      progress = v.deadline ? Math.min(1, v.tick / v.deadline) : 0;
    } else if (i.task === "cue") {
      label = "Cue task";
      value = i.stage === "decision" ? "north or west" : `${i.t ?? 0}/${i.delay ?? 0}`;
      unit = i.stage === "decision" ? `the word is hidden; the choice comes from the reward the model predicts` : `wander steps with the cue word held`;
      progress = i.delay ? (i.t ?? 0) / i.delay : 0;
    } else if (i.task === "word" && i.stage === "teach") {
      label = `Teaching word ${i.word}`;
      value = `${i.t ?? 0}/${i.delay ?? 0}`;
      unit = `the word is shown while ${OBJECT_NAMES[i.object]} is in view`;
      progress = i.delay ? (i.t ?? 0) / i.delay : 0;
    } else if (v) {
      progress = v.horizon ? v.tick / v.horizon : 0;
    }
    $("hudLabel").textContent = label;
    $("hudValue").textContent = value;
    $("hudUnit").textContent = unit;
    $("progressFill").style.width = `${Math.round(Math.max(0, Math.min(1, progress)) * 100)}%`;
    $("history").innerHTML = results.slice(-5).map((r) => `<span class="${r.ok ? "ok" : "bad"}" title="${r.text} · ${r.brain} · ${r.decisions} decisions">${r.label}${r.ok ? "✓" : "✗"}</span>`).join("");
    // the state chip and the hint
    let state = v ? taskText(v) : "loading", cls = running ? "live" : "";
    if (requested) state = `${requested.kind === "ask" ? `ask for ${OBJECT_NAMES[requested.object]}` : requested.kind} at the next event`;
    if (!running) state = `paused · ${state}`;
    if (performance.now() < flashUntil) { state = flashText; cls = "busy"; }
    $("taskState").textContent = state;
    $("taskState").className = `state ${cls}`.trim();
    const hint = $("hint"), idle = i.task === "wander" || i.stage === "rest";
    if (dragging || !idle) hint.hidden = true;
    else if (!results.length && stats.events <= 6) { hint.hidden = false; hint.className = "hint"; }
    else { hint.hidden = false; hint.className = "hint small"; hint.firstElementChild.textContent = "Ask for an object, or drag one to another cell"; }
  }

  function flash(text, ms = 2600) { flashText = text; flashUntil = performance.now() + ms; updateHud(); }

  function updateDoors() {
    const locked = world.doorRule === "locked";
    $("doors").textContent = locked ? "Unlock doors" : "Lock doors";
    $("doors").className = locked ? "on" : "";
  }

  function buildKey() {
    $("worldKey").innerHTML = [
      ["#8fa3b8", "wall", false], ["#ff8c5a", "door shut", false], ["#8fe19d", "door open", false],
      [PLAN_COLOR, "imagined path", false], [REUSE_COLOR, "the action", false],
      [TARGET_COLOR, "asked for", true], ["#c792ea", "remembered (diamond)", true], [AGENT_COLOR, "the agent", true],
    ].map(([color, text, dot]) => `<span><i class="${dot ? "dot" : ""}" style="background:${color}"></i>${text}</span>`).join("");
  }

  // ---------------------------------------------------------------- the world canvas

  function drawWorld() {
    const r = worldCanvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    if (!r.width) return;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (worldCanvas.width !== w || worldCanvas.height !== h) { worldCanvas.width = w; worldCanvas.height = h; }
    const ctx = worldCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const ground = ctx.createRadialGradient(r.width / 2, r.height * 0.52, 0, r.width / 2, r.height / 2, Math.hypot(r.width, r.height) / 2);
    ground.addColorStop(0, "#0b1621"); ground.addColorStop(1, "#04080c");
    ctx.fillStyle = ground; ctx.fillRect(0, 0, r.width, r.height);
    // the readout band stays clear at the top; the grid sits below it with room for the cell numbers
    const band = 42, left = 16, top = band + 14, edge = 8;
    const cs = Math.min((r.width - left - edge) / SIZE, (r.height - top - edge) / SIZE);
    const x0 = left + ((r.width - left - edge) - SIZE * cs) / 2, y0 = top + ((r.height - top - edge) - SIZE * cs) / 2;
    viewFrame = { x0, y0, cs, width: r.width, height: r.height };
    const X = (x) => x0 + x * cs, Y = (y) => y0 + y * cs, CX = (c) => X(c[0]) + cs / 2, CY = (c) => Y(c[1]) + cs / 2;
    const px = Math.max(0.8, Math.min(2, Math.min(r.width, r.height) / 420));
    ctx.font = "10px ui-monospace, monospace"; ctx.fillStyle = "#4f6479"; ctx.textAlign = "center";
    for (let k = 0; k < SIZE; k++) { ctx.fillText(`${k}`, X(k) + cs / 2, y0 - 5); ctx.fillText(`${k}`, x0 - 8, Y(k) + cs / 2 + 3); }
    if (!worldView) { ctx.fillStyle = "#6f8397"; ctx.fillText("press Play", r.width / 2, r.height / 2); return; }
    const v = worldView;
    // the rooms of the cue task, the neighbourhood the last inspect revealed, the cell under the pointer
    for (const c of ROOMS) { ctx.fillStyle = "rgba(237,129,182,0.08)"; ctx.fillRect(X(c[0]) + 1, Y(c[1]) + 1, cs - 2, cs - 2); }
    for (const c of v.visible) { ctx.fillStyle = "rgba(143,225,157,0.16)"; ctx.fillRect(X(c[0]) + 1, Y(c[1]) + 1, cs - 2, cs - 2); }
    if (dragging && dragging.cell) { ctx.fillStyle = "rgba(89,229,203,0.14)"; ctx.fillRect(X(dragging.cell[0]) + 1, Y(dragging.cell[1]) + 1, cs - 2, cs - 2); }
    ctx.strokeStyle = "#101a26"; ctx.lineWidth = 1;
    for (let k = 0; k <= SIZE; k++) { ctx.beginPath(); ctx.moveTo(X(k), Y(0)); ctx.lineTo(X(k), Y(SIZE)); ctx.moveTo(X(0), Y(k)); ctx.lineTo(X(SIZE), Y(k)); ctx.stroke(); }
    if (v.trail.length > 1) { ctx.strokeStyle = "rgba(143,225,157,0.3)"; ctx.lineWidth = 2 * px; ctx.beginPath(); ctx.moveTo(CX(v.trail[0]), CY(v.trail[0])); for (const c of v.trail.slice(1)) ctx.lineTo(CX(c), CY(c)); ctx.stroke(); }
    // walls and doors on the edges
    const doorKeys = new Set(v.doors.map(([a, b]) => `${a}|${b}`));
    for (const [a, b, kind] of v.edges) {
      const vertical = a[1] === b[1]; // (x, y)-(x+1, y): a vertical boundary at X(x+1)
      const x0 = vertical ? X(a[0] + 1) : X(a[0]), y0 = vertical ? Y(a[1]) : Y(a[1] + 1), x1 = vertical ? X(a[0] + 1) : X(a[0] + 1), y1 = vertical ? Y(a[1] + 1) : Y(a[1] + 1);
      const isDoor = doorKeys.has(`${a}|${b}`);
      if (kind === "wall") { ctx.strokeStyle = "#8fa3b8"; ctx.lineWidth = 3.4 * px; ctx.setLineDash([]); }
      else if (kind === "door_closed") { ctx.strokeStyle = "#ff8c5a"; ctx.lineWidth = 3.4 * px; ctx.setLineDash([]); }
      else if (isDoor) { ctx.strokeStyle = "#8fe19d"; ctx.lineWidth = 1.8 * px; ctx.setLineDash([4, 4]); }
      else continue;
      ctx.beginPath(); ctx.moveTo(x0, y0); ctx.lineTo(x1, y1); ctx.stroke(); ctx.setLineDash([]);
    }
    ctx.strokeStyle = "#8fa3b8"; ctx.lineWidth = 3.4 * px; ctx.strokeRect(X(0), Y(0), SIZE * cs, SIZE * cs);
    ctx.fillStyle = "#6f8397"; ctx.font = `${Math.max(9, Math.round(cs * 0.16))}px ui-monospace, monospace`;
    ctx.fillText("junction", CX(world.config.junction), Y(world.config.junction[1]) + 11);
    for (const c of ROOMS) ctx.fillText("room", CX(c), Y(c[1]) + 11);
    // remembered places as diamonds, the objects as discs
    if (storeView) for (const rec of storeView.places) if (rec.where) { const c = rec.where; ctx.strokeStyle = OBJECT_COLORS[v.colors[rec.object]]; ctx.lineWidth = 1.5 * px; ctx.setLineDash([2, 2]); ctx.beginPath(); ctx.moveTo(CX(c), CY(c) - cs * 0.42); ctx.lineTo(CX(c) + cs * 0.42, CY(c)); ctx.lineTo(CX(c), CY(c) + cs * 0.42); ctx.lineTo(CX(c) - cs * 0.42, CY(c)); ctx.closePath(); ctx.stroke(); ctx.setLineDash([]); }
    for (const [k, c] of Object.entries(v.objects)) {
      if (!c || (dragging && dragging.object === Number(k))) continue;
      drawObject(ctx, Number(k), CX(c) - cs * 0.2, CY(c) + cs * 0.2, cs, v, hover === Number(k));
    }
    // targets: what was asked for (privileged, for the viewer) and the cell the search aimed at
    if (v.requestTarget) { ctx.strokeStyle = TARGET_COLOR; ctx.lineWidth = 2 * px; ctx.beginPath(); ctx.arc(CX(v.requestTarget), CY(v.requestTarget), cs * 0.44, 0, Math.PI * 2); ctx.stroke(); ctx.fillStyle = TARGET_COLOR; ctx.font = "9px ui-monospace, monospace"; ctx.fillText("asked for", CX(v.requestTarget), Y(v.requestTarget[1] + 1) - 4); }
    if (v.rewarded) { ctx.fillStyle = TARGET_COLOR; ctx.font = "9px ui-monospace, monospace"; ctx.fillText("pays", CX(v.rewarded), Y(v.rewarded[1] + 1) - 4); }
    if (v.target && !(v.requestTarget && sameCell(v.target, v.requestTarget))) { ctx.strokeStyle = PLAN_COLOR; ctx.lineWidth = 1.5 * px; ctx.setLineDash([3, 3]); ctx.beginPath(); ctx.arc(CX(v.target), CY(v.target), cs * 0.36, 0, Math.PI * 2); ctx.stroke(); ctx.setLineDash([]); ctx.fillStyle = PLAN_COLOR; ctx.font = "9px ui-monospace, monospace"; ctx.fillText(v.goal === GOAL_EXPLORE || v.goal === GOAL_CUE ? "explore" : "believed", CX(v.target), Y(v.target[1] + 1) - 4); }
    // the imagined path of the last search
    if (v.search && v.search.cells.length) {
      const from = v.trail.length > 1 && v.controller === "planner" ? v.trail[v.trail.length - 2] : v.agent;
      ctx.strokeStyle = v.search.kind === "reuse" ? REUSE_COLOR : PLAN_COLOR; ctx.lineWidth = 2.5 * px; ctx.beginPath(); ctx.moveTo(CX(from), CY(from));
      for (const c of v.search.cells) ctx.lineTo(CX(c), CY(c));
      ctx.stroke();
      const last = v.search.cells[v.search.cells.length - 1];
      ctx.fillStyle = ctx.strokeStyle; ctx.beginPath(); ctx.arc(CX(last), CY(last), 3.5 * px, 0, Math.PI * 2); ctx.fill();
    }
    // the agent and what it carries
    const a = v.agent;
    ctx.fillStyle = AGENT_COLOR; ctx.beginPath(); ctx.arc(CX(a), CY(a), cs * 0.19, 0, Math.PI * 2); ctx.fill();
    ctx.strokeStyle = "#05090e"; ctx.lineWidth = 2; ctx.stroke();
    if (v.carrying !== null) { ctx.fillStyle = OBJECT_COLORS[v.colors[v.carrying]]; ctx.beginPath(); ctx.arc(CX(a) + cs * 0.22, CY(a) - cs * 0.22, cs * 0.11, 0, Math.PI * 2); ctx.fill(); }
    if (v.decisionAction !== null) {
      const name = ACTIONS[v.decisionAction];
      if (name in DIRECTIONS) { const [dx, dy] = DIRECTIONS[name]; ctx.strokeStyle = REUSE_COLOR; ctx.lineWidth = 2.5 * px; ctx.beginPath(); ctx.moveTo(CX(a), CY(a)); ctx.lineTo(CX(a) + dx * cs * 0.4, CY(a) + dy * cs * 0.4); ctx.stroke(); }
      else { ctx.fillStyle = REUSE_COLOR; ctx.font = "9px ui-monospace, monospace"; ctx.textAlign = "center"; ctx.fillText(name, CX(a), CY(a) + cs * 0.4); }
    }
    // the object being dragged follows the pointer
    if (dragging && dragging.at) drawObject(ctx, dragging.object, dragging.at[0], dragging.at[1], cs, v, true);
  }

  function drawObject(ctx, k, x, y, cs, v, lit) {
    ctx.fillStyle = OBJECT_COLORS[v.colors[k]];
    if (lit) { ctx.shadowColor = OBJECT_COLORS[v.colors[k]]; ctx.shadowBlur = 12; }
    ctx.beginPath(); ctx.arc(x, y, cs * (lit ? 0.19 : 0.16), 0, Math.PI * 2); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.fillStyle = "#05090e"; ctx.textAlign = "center"; ctx.font = `bold ${Math.round(cs * 0.2)}px system-ui, sans-serif`;
    ctx.fillText(OBJECT_NAMES[k], x, y + cs * 0.07);
  }

  /** The recall port's drive from the stores: the place read, the cue, the word read. */
  function drawRecall(values) {
    const r = recallCanvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    if (!r.width || !values) return;
    const w = Math.max(1, Math.round(r.width * dpr)), h = Math.max(1, Math.round(r.height * dpr));
    if (recallCanvas.width !== w || recallCanvas.height !== h) { recallCanvas.width = w; recallCanvas.height = h; }
    const ctx = recallCanvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#050a0f"; ctx.fillRect(0, 0, r.width, r.height);
    const n = values.length, bw = r.width / n, gain = brain.recallGain || 1;
    const groups = [[0, SIZE * SIZE + 1, "#8fe19d", "place read"], [SIZE * SIZE + 1, SIZE * SIZE + 1 + WORDS + 1, "#c792ea", "cue"], [SIZE * SIZE + 1 + WORDS + 1, n, "#59e5cb", "word read"]];
    for (const [a, b, color, label] of groups) {
      ctx.fillStyle = color;
      for (let k = a; k < b; k++) { const x = Math.min(1, Math.max(0, values[k] / gain)); ctx.globalAlpha = 0.22 + 0.78 * x; ctx.fillRect(k * bw + 0.5, r.height - 1 - x * (r.height - 2), Math.max(1, bw - 1), x * (r.height - 2) + 1); }
      ctx.globalAlpha = 1; ctx.font = "9px ui-monospace, monospace"; ctx.textAlign = "left"; ctx.fillStyle = "#63788d"; ctx.fillText(label, a * bw + 3, 9);
      ctx.strokeStyle = "#1c2a3a"; ctx.beginPath(); ctx.moveTo(b * bw, 0); ctx.lineTo(b * bw, r.height); ctx.stroke();
    }
  }

  // ---------------------------------------------------------------- dragging an object

  function cellAt(clientX, clientY) {
    const r = worldCanvas.getBoundingClientRect();
    const x = Math.floor((clientX - r.left - viewFrame.x0) / viewFrame.cs), y = Math.floor((clientY - r.top - viewFrame.y0) / viewFrame.cs);
    return x >= 0 && x < SIZE && y >= 0 && y < SIZE ? [x, y] : null;
  }

  function objectAtCell(cell) {
    if (!cell || !worldView) return null;
    for (const [k, c] of Object.entries(worldView.objects)) if (c && sameCell(c, cell)) return Number(k);
    return null;
  }

  function pointerPosition(event) { const r = worldCanvas.getBoundingClientRect(); return [event.clientX - r.left, event.clientY - r.top]; }

  worldCanvas.addEventListener("pointerdown", (event) => {
    if (event.button > 0) return;
    const cell = cellAt(event.clientX, event.clientY), k = objectAtCell(cell);
    if (selected !== null && cell) { placeObject(selected, cell); selected = null; return; } // a second tap moves the selected object
    if (k === null) return;
    event.preventDefault();
    worldCanvas.setPointerCapture(event.pointerId);
    dragging = { object: k, pointer: event.pointerId, at: pointerPosition(event), cell, from: cell, moved: false };
    worldCanvas.className = "grabbing";
    stopDemo();
  });

  worldCanvas.addEventListener("pointermove", (event) => {
    const cell = cellAt(event.clientX, event.clientY);
    if (dragging && dragging.pointer === event.pointerId) {
      dragging.at = pointerPosition(event);
      if (cell && !sameCell(cell, dragging.from)) dragging.moved = true;
      dragging.cell = cell;
      return;
    }
    const k = objectAtCell(cell);
    hover = k;
    worldCanvas.className = k === null ? "" : "grab";
  });

  const releasePointer = (event) => {
    if (!dragging || dragging.pointer !== event.pointerId) return;
    const cell = cellAt(event.clientX, event.clientY), k = dragging.object, moved = dragging.moved;
    dragging = null;
    worldCanvas.className = "";
    if (!moved) { selected = k; flash(`${OBJECT_NAMES[k]} picked up: tap a cell to put it there`); return; } // a tap selects; the next tap places
    selected = null;
    if (cell) placeObject(k, cell);
  };
  worldCanvas.addEventListener("pointerup", releasePointer);
  worldCanvas.addEventListener("pointercancel", releasePointer);

  /** The visitor moves an object: the world changes, the agent's record of it does not. */
  function placeObject(k, cell) {
    if (!worldView) return false;
    if (world.carrying === k) { flash(`the agent is carrying ${OBJECT_NAMES[k]}`); return false; }
    const other = objectAtCell(cell);
    if (other !== null && other !== k) { flash(`${OBJECT_NAMES[other]} is already in ${cellText(cell)}`); return false; }
    if (sameCell(world.objects()[k], cell)) return false;
    world.placeObject(k, cell);
    const remembered = brain.places.recall(k);
    const seen = sameCell(world.agent, cell);
    flash(seen ? `${OBJECT_NAMES[k]} moved to ${cellText(cell)} in full view` : remembered.where ? `${OBJECT_NAMES[k]} moved to ${cellText(cell)}; the agent still remembers ${cellText(remembered.where)}` : `${OBJECT_NAMES[k]} moved to ${cellText(cell)}`);
    if (worldView) { worldView.objects = world.objects(); }
    return true;
  }

  // ---------------------------------------------------------------- controls

  for (const button of document.querySelectorAll(".chip[data-object]")) {
    const k = Number(button.dataset.object);
    button.innerHTML = `<i></i>${OBJECT_NAMES[k]}`;
    button.onclick = () => { requestTask({ kind: "ask", object: k }); flash(`asking for ${OBJECT_NAMES[k]} at the next event`); };
  }
  $("wander").onclick = () => { requestTask({ kind: "wander", repeat: true }); flash("wandering: exploration rate 1, the sightings are its own"); };
  $("word").onclick = () => { requestTask({ kind: "word" }); flash("a word shown while its object is in view, then the word alone as a request"); };
  $("cue").onclick = () => { requestTask({ kind: "cue", delay: Number($("delay").value) || 8 }); flash(`the cue task at delay ${$("delay").value}`); };
  $("delay").onchange = () => flash(`the next cue task wanders ${$("delay").value} steps`);
  $("doors").onclick = () => {
    const locked = world.doorRule === "locked";
    world.setDoorRule(locked ? "toggle" : "locked");
    updateDoors();
    flash(locked ? "interact opens doors again" : "interact no longer opens a door: the model must learn the change");
  };
  $("phaseSelect").onchange = () => {
    const value = $("phaseSelect").value;
    if (value === "free") { requestTask({ kind: "wander", repeat: true }); flash("this page's tasks: ask, move, teach, cue, doors"); return; }
    requestTask({ kind: value, repeat: true });
    flash(`${value} · run.py's own phase, from the next event`);
  };
  $("speed").onchange = () => { flash(`${$("speed").value === "0" ? "a whole event" : $("speed").value + " settling steps"} per frame`); };
  $("imagine").onchange = () => { imagineEvery = Number($("imagine").value); };
  $("play").onclick = () => setRunning(!running);
  $("step").onclick = () => { setRunning(false); computeEvent(); drain(0); showAtOnce(); };
  $("fit").onclick = () => scan.fit();
  $("view").onchange = () => { scan.options.mode = $("view").value; scan.draw(); };
  $("epsilon").oninput = () => { pageEpsilon = Number($("epsilon").value); $("epsilonValue").textContent = pageEpsilon.toFixed(2); setPhaseEpsilon(phaseEpsilon); };

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
      flash(`${entry.label}: its stores are empty, so let it wander before asking`);
    } catch (error) {
      select.value = checkpoint.id;
      flash(`could not load that brain: ${error.message}`, 6000);
      console.error(error);
    } finally { select.disabled = false; }
  }

  // ---------------------------------------------------------------- the loop

  const rate = () => Number($("speed").value);

  function frame(now) {
    if (running) {
      const steps = rate();
      if (steps === 0) { computeEvent(); drain(0); }
      else { if (queue.length < Math.max(64, steps * 2)) computeEvent(); drain(steps); }
    }
    if (performance.now() < flashUntil + 80) updateHud();
    showPhase();
    if (running || queue.length) { scan.draw(now); if (records) records.draw(); }
    drawWorld();
    if (storeView) drawRecall(storeView.recall);
    requestAnimationFrame(frame);
  }

  function showAtOnce() {
    showPhase();
    scan.draw();
    drawWorld();
    if (storeView) drawRecall(storeView.recall);
    if (records) records.draw();
  }

  addEventListener("resize", () => { scan.draw(); if (records) records.draw(); drawWorld(); if (storeView) drawRecall(storeView.recall); });

  // ---------------------------------------------------------------- start

  buildKey();
  installLife(FIRST, "this page's brain", "inlined");
  updateDoors();
  $("epsilonValue").textContent = pageEpsilon.toFixed(2);
  setRunning(running);
  setPhase(`resting · ${((FIRST.extra && FIRST.extra.steps_before) || 0).toLocaleString()} steps lived before this page`, "");
  loadCheckpoints();
  demo = PAGE_OPTIONS.autoplay !== false;
  requestAnimationFrame(frame);

  window.__page = {
    scan, get agent() { return agent; }, get brain() { return brain; }, get world() { return world; }, get cur() { return cur; }, get records() { return records; }, queue,
    get stats() {
      return {
        ...stats, parameter_version: agent.parameterVersion, ring: agent.ring.length, record_writes: agent.ledger.record_writes, imagined: agent.ledger.imagined,
        records: records ? { ...records.counts, active: records.activeIndex.length } : null, episode: world.episode, tick: world.tick, queued: queue.length,
        epsilon: brain.epsilon, searches: brain.searches, reused: brain.reused, place_writes: brain.places.writes, map_writes: brain.map.writes, word_writes: brain.words.writes,
        checkpoint: checkpoint.id, experience: (snapshot.ledger && snapshot.ledger.real_transitions) || 0, renderer: scan.snapshot(),
      };
    },
    /** What the page is doing: the running task, its stage, what the stores hold and every finished task. */
    get task() {
      const v = worldView, i = v ? v.info : {};
      return {
        kind: (current && current.kind) || "wander", stage: i.stage || "rest", requested: requested ? requested.kind : null,
        doors: world.doorRule, objects: world.objects(), agent: world.agent, deadline: world.deadline, tick: world.tick,
        places: brain.places.records().map((r) => ({ object: r.object, where: r.where, known: r.known })),
        words: Array.from({ length: WORDS }, (_, w) => brain.words.referent(w + 1).object),
        results: results.map((r) => ({ ...r })), successes: JSON.parse(JSON.stringify(stats.successes)),
        checkpoints: checkpoints.map((c) => c.id), brain: checkpoint.label,
      };
    },
    /** One event, computed and shown at once (the headless check drives the page with these). */
    step() { computeEvent(); drain(0); showAtOnce(); return this.stats; },
    run(count = 1) { let ran = 0; const t0 = performance.now(); while (ran < count) { computeEvent(); drain(0); ran += 1; } showAtOnce(); return { ran, seconds: (performance.now() - t0) / 1000, ...this.stats }; },
    /** Events until the running task ends, at most `count` (the check measures a whole request this way). */
    runTask(count = 400) {
      const t0 = performance.now(); let ran = 0;
      const started = results.length;
      while (ran < count && results.length === started) { computeEvent(); drain(0); ran += 1; }
      showAtOnce();
      return { ran, seconds: (performance.now() - t0) / 1000, finished: results.length > started, last: results.length ? { ...results[results.length - 1] } : null };
    },
    ask(k) { requestTask({ kind: "ask", object: k | 0 }); return this.task; },
    move(k, cell) { return placeObject(k | 0, cell); },
    pause() { setRunning(false); }, play() { setRunning(true); },
    selectCheckpoint,
    /** Where a cell sits on the screen, for pointer events. */
    toClient(cell) { const r = worldCanvas.getBoundingClientRect(); return [r.left + viewFrame.x0 + (cell[0] + 0.5) * viewFrame.cs, r.top + viewFrame.y0 + (cell[1] + 0.5) * viewFrame.cs]; },
    cellAt(x, y) { return cellAt(x, y); },
    ready: true,
  };
  window.__brainScan = scan;
}

loadSnapshot().then(main).catch((error) => { $("counts").textContent = `error: ${error.message}`; $("taskState").textContent = "failed to load"; console.error(error); });
