// The patch net receives rendered eye pixels, settles sensory responses and
// chooses an odor goal and approach/avoid; matching executed outcomes update
// KC->MBON synapses. The body controller executes those goals.
// The retina, full connectome, repair replay and console share this one demo.
import * as THREE from "three";
import { Flight, cos_, sin_ } from "./body.js";
import { createRoom, createRenderer, createComposer, CameraRig, toThree, fromThree } from "./room.js";
import { createFly } from "./fly.js";
import { createFlyEye } from "./eye.js";
import { GoalLife } from "./goal-life.js";
import { PathHistory, projectCommand } from "./path-history.js";
import { createPathView } from "./path-view.js";
import { createSettlementView } from "./settlement-view.js";
import { BrainStateTelemetry } from "./neural-state.js";
import { PhysicsClock } from "./physics-clock.js";
import { createBrainConsole, parseConsoleCommand, formatBrainStatus } from "./brain-console.js";
import { BrainScan, decodeAtlas } from "./brain_scan.js";
import { replayScanFrame } from "./neural-replay.js";
import { progressScanFrame } from "./settlement-progress.js";
import { chooseWorkerJob, qualifiedObservationId } from "./worker-scheduling.js";
import { GlitchPass } from "three/addons/postprocessing/GlitchPass.js";

const $ = (id) => document.getElementById(id);
const params = new URLSearchParams(location.search);
// One demo controller: the brain chooses goals; its body executes them.
const makeLife = (body, seed, generation) => new GoalLife(body, room, seed, { generation });
// A short failed attempt gives way to the next actual sensory snapshot.
// Admission still requires the all-neuron residual; unfinished solves never
// produce goals. Offline diagnostics retain the engine's larger 1024-step cap.
const CONTROL_TOLERANCE = 1e-6, OBSERVATION_CAP = 256;
const SCAN_INTERVAL_MS = 1000 / 15;
const TRACE_OPTIONS = Object.freeze({ maxPatches: 48, maxEdges: 128, maxFrames: 24 });
const REPLAY_OPTIONS = Object.freeze({ maxFrames: 7 });
const REPLAY_FRAME_MS = 150;
// The lessons: the constants of tools/learn_odour.py; data/lessons_full.json (written with the receipt) overrides them.
// eta 1.0 and etaCritic 0.05: the T-maze's constants (tools/tmaze.py, receipts/g4_tmaze_reversal.json), one decision per search at the fruit.
// Measured on the way there: eta 10 put an output cell on its rail after one lesson, where the nudge has no slope and nothing moves again, and a
// critic at 0.5 learned the outcome in four trials and cancelled the dopamine before the actor had moved. The operating points the lessons need
// (the local neurons' gain, the seam started naive, the two readouts calibrated) travel in the payload: fruitfly/lessons.py.
let LESSONS = { outputs: ["mbon:MBON11:right", "mbon:MBON05:left"], actions: [0, 1], plastic: { pre: ["kc"], post: ["mbon"] }, critic: "kc", beta: 0.1, temperature: 0.3, nudgedSteps: 10, tolerance: 1e-3, gamma: 0.95, lam: 0.9, eta: 1.0, etaBias: 0, etaCritic: 0.05, cap: 3, dopamineCap: 1, tonic: {} };

// ---- the room ------------------------------------------------------------------------------
const canvas = $("world");
const renderer = createRenderer(canvas);
// the pixel ratio decides the cost of the bloom pass and of every extra view: 1.25 keeps a Retina laptop at a high frame rate (?dpr=2 for the full density)
renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, Number(params.get("dpr") || 1.25)));
const scene = new THREE.Scene(); scene.background = new THREE.Color(0x000000);
const world = new THREE.Group(); world.rotation.x = -Math.PI / 2; scene.add(world);
const room = createRoom(world);
const fly = createFly(); world.add(fly.group, fly.shadow, fly.locator);
const pathHistory = new PathHistory();
const pathView = createPathView(world, $("path-panel"));
const rig = new CameraRig(innerWidth / innerHeight);
rig.zoom = 0.035; // Frame the physical fly closely in the main view.
const { composer, bloom } = createComposer(renderer, scene, rig.camera);
// Neural commands are shown to the viewer; annotations never become retinal
// stimuli. Both the raw capture and the displayed eye exclude this group.
const eye = createFlyEye(renderer, scene, { hide: [fly.group, fly.shadow, fly.locator, pathView.group] });
// a rare glitch in the Matrix: the screen tears for a moment every so often, and the room's own elements jitter
const glitch = new GlitchPass(); glitch.goWild = false; glitch.enabled = false; composer.addPass(glitch);
// debug switches for the frame budget: ?nobloom=1 ?noscan=1 ?noviews=1 ?lines=60000 ?particles=40000 ?scanrate=2 (draw the brain every n-th frame)
const DEBUG = { nobloom: !!params.get("nobloom"), noscan: !!params.get("noscan"), noviews: !!params.get("noviews"), scanrate: Number(params.get("scanrate") || 1) };
if (DEBUG.nobloom) bloom.enabled = false;
let nextGlitch = 12 + 30 * Math.random(), glitchLeft = 0;
function glitches(dt) {
  nextGlitch -= dt;
  if (nextGlitch <= 0) { glitchLeft = 0.12 + 0.25 * Math.random(); nextGlitch = 15 + 40 * Math.random(); glitch.enabled = true; document.body.classList.add("glitch"); if (room.glitch) room.glitch(); }
  if (glitchLeft > 0) { glitchLeft -= dt; if (glitchLeft <= 0) { glitch.enabled = false; document.body.classList.remove("glitch"); } }
}

// ---- the fly and its life --------------------------------------------------------------------
const S = { speed: 1, pilot: "assisted", paused: false, ready: false, brainReady: false, atlasReady: false, learnReady: false, simTime: 0, brainDue: 0, pending: false, brainSteps: 0, brainMs: 0, active: 0, frames: 0, fps: 0, fpsClock: 0, lastScan: 0, lastScanDraw: -Infinity, lastHud: -Infinity, controlReplies: 0, controlAccepted: 0, controlLate: 0, controlFailed: 0, solveMs: 0, seed: Number(params.get("seed") || 1), lessons: [], lessonStats: null, lastLesson: null, decisions: 0, generation: 0, requestId: 0, nextControl: 0, pendingId: null, controlReply: null };
Object.assign(S, { workerBusy: false, workerJobId: null, workerStarted: 0, progressFrames: 0, settlementProgress: null });
let flight = new Flight([1.2, 1.0, 1.2], 0.3);
function attachSurfaces(body) {
  body.surfaceHeight = (x, y) => {
    const t = room.table, heights = [room.surfaceZ(x, y)];
    if (t && x >= t.x0 && x <= t.x1 && y >= t.y0 && y <= t.y1) heights.push(t.z);
    return heights;
  };
}
attachSurfaces(flight);
let life = makeLife(flight, S.seed, S.generation);
$("demo-mode").textContent = "Brain chooses goals · body executes";
{
  $("path-legend").innerHTML = 'Grey: measured path through the body controller.<br><span id="path-duration">Brain-selected goals; supplied movement mechanics.</span>';
  $("path-note").textContent = "The trail records actual movement. The neural fruit/action choice is shown separately from its body-controller route.";
}
const brainStates = new BrainStateTelemetry({ generation: S.generation, maxAgeSeconds: 1 });
let pendingObservationTime = null;
if (room.setSugar) room.setSugar(life.sugar);

// Actual solver traces are a separate observation surface. They never replace
// the checked motor reply, and replay time never advances the neural equations.
let brainView = ["brain", "scan", "settling"].includes(params.get("style")) ? params.get("style") : "scan";
let detailPanel = "brain";
let eyeVisible = true;
const terminal = createBrainConsole($("brain-console"), { execute: executeCommand, now: () => life.clock });
terminal.write("link", "Connecting the full patch net. /help lists commands; /status reads the brain.");
terminal.write("model", "The brain chooses the fruit and approach/avoid, and learns from executed choices. The body controller navigates, lands, feeds and grooms. /about explains.");
// Observe existing life events without inserting any policy, reward or input.
function watchLifeEvents() {
  const original = life.log.bind(life);
  life.log = (message) => { original(message); terminal.write("event", message, { generation: S.generation }); };
}
watchLifeEvents();
let lastCourseId = null, lastCourseActive = false, lastCoursePhase = null, lastBodyMode = life.mode, lastExecutedChoice = null, lastStartupPhase = null;
let inspectionPending = null, inspectionTimeout = null, lastDiagnostic = null;
let liveProgressPacket = null, displayedProgress = null;
function clearInspection() {
  if (inspectionTimeout !== null) clearTimeout(inspectionTimeout);
  inspectionTimeout = null; inspectionPending = null;
}
const settlementInputs = new Map();
const settlementView = createSettlementView($("settlement-view"), { onContrast: compareRetinalPixels });
function rememberSettlementInput(request, retina = null) {
  if (!request.trace) return;
  settlementInputs.set(request.requestId, { generation: S.generation, requestId: request.requestId,
    observationTime: life.clock, retinal: retina ? structuredClone(retina) : null });
  while (settlementInputs.size > 4) settlementInputs.delete(settlementInputs.keys().next().value);
}
function showSettlementReply(message, policyAccepted) {
  const input = settlementInputs.get(message.requestId); settlementInputs.delete(message.requestId);
  if (!message.trace || !input || input.generation !== S.generation || message.generation !== S.generation) return;
  if (message.trace.identity?.requestId !== message.requestId || message.trace.identity?.generation !== message.generation) {
    settlementView.setStatus("Trace identity mismatch; recording discarded."); return;
  }
  settlementView.setTrace(message.trace, {
    identity: { requestId: message.requestId, generation: message.generation, observationTime: input.observationTime },
    retinal: { frame: input.retinal, ...message.trace.retinal },
    readouts: message.readouts || {}, policyAccepted,
    status: policyAccepted ? "Recorded observation admitted for control" : "Recorded observation not admitted for control",
  });
}
function compareRetinalPixels() {
  if (inspectionPending) return;
  if (!S.brainReady) {
    settlementView.setStatus("Wait for the brain to finish loading before comparing eye pixels."); return;
  }
  setDetailPanel("brain"); setBrainView("settling");
  // Inspection holds the body and withdraws actuation. The worker restores its
  // exact pre-probe state; neither comparison can earn action/learning credit.
  if (!S.paused) togglePause();
  const requestId = ++S.requestId;
  try {
    const retina = eye.capture(flight);
    // Encoders update display/contact caches. Run them on a shallow observer
    // copy so the diagnostic gets fresh current-body senses without changing
    // live controller caches or reusing another generation's observation.
    const sensorProbe = Object.assign(Object.create(Object.getPrototypeOf(life)), life);
    const senses = sensorProbe.senses(null, 0);
    inspectionPending = { requestId, generation: S.generation, retinal: structuredClone(retina), observationTime: life.clock };
    settlementView.setStatus("Body paused. Comparing these pixels with black input from the same neural state…");
    const inspectedWorker = worker, generation = S.generation;
    inspectionTimeout = setTimeout(() => {
      if (worker !== inspectedWorker || inspectionPending?.requestId !== requestId || S.generation !== generation) return;
      inspectedWorker.terminate(); workerFailed("pixel comparison timed out; reset to reload the brain");
    }, 30000);
    worker.postMessage({ type: "inspect:retina", requestId, generation: S.generation,
      senses, retina, steps: OBSERVATION_CAP, tolerance: CONTROL_TOLERANCE, trace: TRACE_OPTIONS });
  } catch (error) {
    clearInspection(); settlementView.setStatus(`Pixel comparison unavailable: ${error.message}`);
  }
}

// Passive observation only: these records never feed the brain or the pilot.
let pathDecision = null, appliedPathObservation = null;
let displayedPaths = null, lastPathDraw = -Infinity, lastPathSignature = "";
function recordPath(afterPhysics = false) {
  const nav = life.authority.navigation;
  const observation = life._routeObservation;
  if (afterPhysics) appliedPathObservation = observation?.requestId ?? null;
  const active = !!(nav?.active && observation && appliedPathObservation === observation.requestId
    && life.clock < observation.deadline && !S.paused && S.brainReady
    && life.neuralEnabled && !life.wingsOff);
  const frame = { generation: S.generation, time: life.clock, position: flight.p,
    heading: Number.isFinite(life.heading) ? life.heading : 0,
    command: active ? nav.command : { yawRate: 0, forwardSpeed: 0, verticalSpeed: 0 },
    active, attention: active ? observation.attention : null,
    action: active ? life.search?.neuralChoice?.action ?? null : null,
    observationId: active ? observation.requestId : null, deadline: active ? observation.deadline : null };
  pathHistory.record(frame);
  return frame;
}
function pathState() {
  const history = pathHistory.snapshot(), choice = life.search?.neuralChoice;
  return { history, projection: projectCommand(history), enabled: pathView.enabled,
    decision: pathDecision ? structuredClone(pathDecision) : null,
    decisionCurrent: !!(pathDecision && choice && !choice.revoked
      && pathDecision.generation === S.generation && pathDecision.requestId === choice.requestId
      && pathDecision.searchToken === life.search.neuralToken),
    executed: !!choice?.executed, mode: S.pilot, paused: S.paused, grounded: !!life.wingsOff,
    motionSupport: structuredClone(life.authority.motionSupport), goal: structuredClone(life.authority.goal ?? null), hybrid: true };
}
function refreshPaths(force = true, now = performance.now()) {
  const current = recordPath();
  if (!pathView.enabled) return;
  const signature = `${S.generation}:${current.active}:${S.paused}:${S.pilot}:${pathDecision?.requestId}`;
  // Physics samples stay at their own cadence. The passive viewer draws at
  // most 15 Hz, with immediate withdrawal/reset, and does no hidden rendering.
  if (!force && signature === lastPathSignature && (S.paused || now - lastPathDraw < SCAN_INTERVAL_MS)) return;
  displayedPaths = { state: pathState(), wall: now };
  pathView.update(displayedPaths.state); lastPathDraw = now; lastPathSignature = signature;
}
function setPaths(on) {
  pathView.setEnabled(on);
  if (on) refreshPaths();
}
setPaths(params.get("paths") === "1" && params.get("card") !== "1");
recordPath();

// ---- the brain in its worker ---------------------------------------------------------------
let worker = null;
let members = null, whole = 0, activation = null, payloadInfo = null;
function updateActivation(m) {
  if (activation && members && m.s) { for (let i = 0; i < members.length; i++) activation[members[i]] = m.s[i]; S.newState = true; }
}
function consumeControlReply() {
  const m = S.controlReply;
  if (!m) return;
  S.controlReply = null;
  if (m.generation !== S.generation || m.requestId !== S.pendingId) return;
  S.pending = false; S.pendingId = null; S.pendingSince = 0;
  S.brainSteps += m.iterations || 0; S.solveMs = m.ms || 0; S.brainMs = S.solveMs / Math.max(1, m.iterations || 0); S.active = m.active || 0;
  S.controlReplies++;
  if (m.type === "assisted_decision") {
    const accepted = life.applyAssistedDecision(m);
    worker.postMessage({ type: accepted ? "assist:accept" : "assist:cancel", requestId: m.requestId, generation: m.generation, searchToken: m.searchToken });
    if (accepted) {
      // Preserve the exact sampled evidence before execution telemetry changes.
      // No activation arrays or live controller objects enter the viewer.
      pathDecision = structuredClone({ requestId: m.requestId, generation: m.generation,
        searchToken: m.searchToken, targetSelection: m.targetSelection, fruit: life.search.fruit, decision: m.decision });
      S.decisions++; S.controlAccepted++;
      const targets = m.targetSelection.p;
      terminal.write("attention", `Choice #${m.requestId}: ${life.search.fruit} · ${m.decision.action === 0 ? "approach" : "avoid"}. Target odds: banana ${(100 * targets[0]).toFixed(1)}% / bread ${(100 * targets[1]).toFixed(1)}%; approach ${(100 * m.decision.p[0]).toFixed(1)}%. Body controller will execute this neural choice.`,
        { generation: m.generation, requestId: m.requestId, fruit: life.search.fruit, action: m.decision.action, targetSelection: "neural", targetProbabilities: targets.slice() });
    }
    else { S.controlFailed++; S.assistedDecision = null; }
  } else {
    const accepted = life.applyObservation(m);
    if (accepted) S.controlAccepted++; else S.controlFailed++;
    if (pendingObservationTime !== null) brainStates.observe(m, { generation: S.generation, requestId: m.requestId, observationTime: pendingObservationTime, neuronCount: payloadInfo?.n });
    showSettlementReply(m, accepted);
    acceptScanReplay(m);
    terminal.write(accepted ? "settle" : "hold", accepted
      ? `Solve #${m.requestId}: ${m.iterations} local-update sweeps · residual ${m.residual.toExponential(2)} · checked neural observation admitted.`
      : `Solve #${m.requestId}: input not admitted · ${life.authority.lastDiscard?.reason || m.reason || "expired or unqualified reply"}.`,
      { generation: m.generation, requestId: m.requestId, accepted, iterations: m.iterations, residual: m.residual });
  }
  updateActivation(m);
}
function workerFailed(reason) {
  S.workerBusy = false; S.workerJobId = null; liveProgressPacket = null;
  S.pending = false; S.pendingId = null; S.pendingSince = 0; S.controlReply = null;
  clearInspection(); settlementView.setStatus(`Worker unavailable: ${reason}. Any displayed trace is recorded history.`);
  brainStates.invalidate(reason, S.requestId); pendingObservationTime = null;
  life.revokeNeural(reason); S.brainReady = false; S.learnReady = false;
  life.log(reason); finishLoading();
  refreshPaths();
}
function startWorker() {
  if (worker) worker.terminate();
  const current = new Worker("./worker.js", { type: "module" }); worker = current;
  current.onmessage = (e) => {
    if (worker !== current) return;
    const m = e.data;
    if (m.type === 'settlement_progress') {
      if (S.paused || m.generation !== S.generation || m.requestId !== S.workerJobId || m.authoritative !== false) return;
      // A withdrawn request may finish in the serial worker. Its real iterates
      // remain diagnostic; request ownership still bars actuation and learning.
      liveProgressPacket = m;
      S.progressFrames++;
      S.settlementProgress = { requestId: m.requestId, generation: m.generation, phase: m.phase, iteration: m.iteration,
        residual: m.residual, tolerance: m.tolerance, receivedAt: performance.now(), completed: false,
        withdrawn: m.requestId !== S.pendingId, authoritative: false };
      return;
    }
    if (m.type === "retina_diagnostic" || (m.type === "error" && inspectionPending?.requestId === m.requestId)) {
      if (!inspectionPending || m.generation !== S.generation || m.requestId !== inspectionPending.requestId) return;
      const input = inspectionPending; clearInspection();
      if (m.type === "error" || !m.actual?.trace || !m.black?.trace) {
        settlementView.setStatus(`Pixel comparison failed: ${m.reason || "no valid trace"}. Body remains paused.`); return;
      }
      lastDiagnostic = structuredClone({ ...m, retinal: { frame: input.retinal, ...m.actual.trace.retinal },
        identity: { requestId: m.requestId, generation: m.generation, observationTime: input.observationTime } });
      settlementView.setDiagnostic(lastDiagnostic);
      return; // Diagnostic replies have no route to actuation or reward credit.
    }
    if (m.type === "ready") {
      if (m.generation !== S.generation) return;
      members = m.members; whole = m.whole.neurons; payloadInfo = m;
      S.brainReady = true; loadingStep("the brain is loaded", 0.9);
      current.postMessage({ type: "learn:init", config: LESSONS, seed: S.seed });
      finishLoading();
      fillCard(); return;
    }
    if (["control", "assisted_decision"].includes(m.type)) {
      if (m.requestId === S.workerJobId) { S.workerBusy = false; S.workerJobId = null; }
      if (S.settlementProgress?.requestId === m.requestId) S.settlementProgress.completed = true;
      if (S.paused || m.generation !== S.generation || m.requestId !== S.pendingId) {
        if (m.type === "assisted_decision") current.postMessage({ type: "assist:cancel", requestId: m.requestId, generation: m.generation, searchToken: m.searchToken });
        return;
      }
      S.controlReply = m; return;
    }
    if (m.type === "error") { workerFailed(m.reason || "worker error"); return; }
    if (m.type === "learn:ready") {
      S.learnReady = true; S.lessonStats = m;
      terminal.write("link", `${payloadInfo.n.toLocaleString()} neurons connected · learning ready · time ${S.speed}×.`, { generation: S.generation }); return;
    }
    if (m.type === "lesson") {
      if (m.generation !== S.generation) return;
      const previousUpdates = S.lessonStats?.updates || 0;
      S.lessonStats = { ...S.lessonStats, ...m };
      if (m.delta !== undefined) { S.lastLesson = m; S.lessons.push(m.delta); if (S.lessons.length > 60) S.lessons.shift(); life.log(`dopamine ${m.delta.toFixed(2)}: ${m.why || "a lesson"}`); }
      else if (m.lost) life.log("no executed decision to credit");
      if (m.accepted === true && m.updates > previousUpdates) terminal.write("learn", `Update ${m.updates}: ${m.moved ?? 0} connections updated · reward error ${Number(m.delta).toFixed(3)}.`,
        { generation: m.generation, requestId: m.requestId, updates: m.updates, moved: m.moved, changed: m.changed, delta: m.delta });
      return;
    }
  };
  current.onerror = (e) => { if (worker === current) workerFailed(`worker failed: ${e.message || e}`); };
  current.onmessageerror = () => { if (worker === current) workerFailed("invalid worker message"); };
  current.postMessage({ type: "init", url: "./data/brain_full.json", generation: S.generation });
}
fetch("./data/lessons_full.json").then((r) => (r.ok ? r.json() : null)).then((j) => { if (j && j.config) LESSONS = { ...LESSONS, ...j.config }; }).catch(() => {}).finally(() => { if (!worker) startWorker(); });

// ---- the whole nervous system ---------------------------------------------------------------
let scan = null;
let neuralReplay = null, replayFrameIndex = -1, lastReplayFrameAt = -Infinity, displayedReplay = null;
function resetScanReplay() {
  neuralReplay = null; replayFrameIndex = -1; displayedReplay = null; lastReplayFrameAt = -Infinity;
  liveProgressPacket = null; displayedProgress = null; S.settlementProgress = null;
  $("scan-playback").textContent = "Waiting for a recorded sensory solve";
}
function acceptScanReplay(message) {
  const replay = message.replay;
  if (!replay || replay.identity?.generation !== S.generation || replay.identity?.requestId !== message.requestId
    || replay.identity?.source !== "control") return;
  // Retain one bounded recording and show its actual frames in order. Faster
  // incoming recordings can be skipped by the viewer, never by the controller.
  if (neuralReplay && replayFrameIndex < neuralReplay.frames.length - 1) return;
  neuralReplay = replay; replayFrameIndex = -1; lastReplayFrameAt = -Infinity;
}
function advanceScanReplay(now) {
  if (S.workerBusy && S.settlementProgress && !S.settlementProgress.completed) return;
  if (!neuralReplay || !scan || !members || replayFrameIndex >= neuralReplay.frames.length - 1
    || (S.paused && replayFrameIndex >= 0) || now - lastReplayFrameAt < REPLAY_FRAME_MS) return;
  try {
    const index = replayFrameIndex + 1;
    const measured = replayScanFrame(neuralReplay, index, members, activation.length);
    scan.show(measured.activity, measured.heat, { draw: false });
    replayFrameIndex = index; lastReplayFrameAt = now;
    const frame = neuralReplay.frames[index];
    const indices = [...new Set([0, Math.floor(neuralReplay.n / 4), Math.floor(neuralReplay.n / 2), Math.floor(3 * neuralReplay.n / 4), neuralReplay.n - 1])];
    displayedReplay = { identity: { ...neuralReplay.identity }, frameIndex: index,
      iteration: measured.iteration, residual: measured.residual, frameCount: neuralReplay.frames.length,
      iterations: neuralReplay.iterations, neuronCount: neuralReplay.n, peakDelta: measured.maxAbsDeltaV,
      samples: indices.map(i => ({ neuronIndex: i, atlasIndex: members[i], activity: measured.activity[members[i]],
        deltaV: frame.deltaV[i], heat: measured.heat[members[i]] })) };
    $("scan-playback").textContent = `Recorded solve #${messageId(neuralReplay)} · step ${measured.iteration}/${neuralReplay.iterations}`;
    $("brain-encoding").textContent = `${measured.mappedCount.toLocaleString()} neurons · measured activity and local |Δv| · fixed repair scale`;
  } catch (error) {
    resetScanReplay(); $("scan-playback").textContent = `Recording unavailable: ${error.message}`;
  }
}
function showLiveProgress() {
  const m = liveProgressPacket;
  if (!m || !scan || !members || !activation || S.paused) return;
  liveProgressPacket = null;
  try {
    const frame = progressScanFrame(m, members, activation.length);
    scan.show(frame.activity, frame.heat, { draw: false });
    displayedProgress = { ...S.settlementProgress, framesReceived: S.progressFrames, neuronCount: m.n };
    const withdrawn = S.settlementProgress?.requestId === m.requestId && S.settlementProgress.withdrawn;
    $('scan-playback').textContent = `Live #${m.requestId} · ${m.phase} · step ${m.iteration} · ${m.residual.toExponential(2)}${withdrawn ? ' · withdrawn input' : ' · checking'}`;
    $('brain-encoding').textContent = `${m.n.toLocaleString()} neurons · actual local repairs · fixed scale · intermediate state`;
  } catch (error) { $('scan-playback').textContent = `Progress unavailable: ${error.message}`; }
}
function messageId(replay) { return replay.identity.requestId; }
function showDefaultScan() {
  $("brain").classList.remove("big"); $("brain-big").textContent = "expand";
  setBrainView("scan"); setDetailPanel("brain");
}
function setBrainView(which) {
  if (!["settling", "brain", "scan"].includes(which)) throw new RangeError("unknown brain view");
  brainView = which;
  $("brain").classList.toggle("settling", which === "settling");
  document.body.classList.toggle("brain-inspector", detailPanel === "brain" && which === "settling");
  $("brain-heading").textContent = which === "settling" ? "Brain mechanics" : "Nervous system";
  settlementView.setVisible(detailPanel === "brain" && which === "settling");
  if (scan && which !== "settling") { scan.setStyle(which); scan.resize?.(); scan.fit(); }
  styleButtons(which); settlementView.resize?.();
}
$("style-settling").onclick = () => setBrainView("settling");
$("style-brain").onclick = () => setBrainView("brain");
$("style-scan").onclick = () => setBrainView("scan");
$("brain-fit").onclick = () => { if (brainView === "settling") settlementView.resize?.(); else scan?.fit(); };
$("brain-big").onclick = () => {
  const big = $("brain").classList.toggle("big"); $("brain-big").textContent = big ? "shrink" : "expand";
  if (scan) { scan.options.labelTop = big ? 12 : 4; scan.options.labelCount = big ? 0 : 5; scan.options.exposure = big ? 0.85 : 0.62; }
  setTimeout(() => { if (brainView !== "settling") { scan?.resize?.(); scan?.fit(); } settlementView.resize?.(); }, 300);
};
setBrainView(brainView);
fetch("./data/atlas.json").then((r) => r.json()).then((raw) => {
  const atlas = decodeAtlas(raw);
  // the Matrix palette: every region its own shade of green (hue 105 to 160, the mushroom body the brightest)
  atlas.regions.forEach((r, k) => {
    const name = (r.name || "").toLowerCase();
    const hue = name.includes("mushroom") ? 130 : 105 + (k * 37) % 56, light = name.includes("mushroom") ? 0.72 : 0.42 + ((k * 13) % 5) * 0.06;
    const c = new THREE.Color().setHSL(hue / 360, 0.95, light);
    r.color = [Math.round(255 * c.r), Math.round(255 * c.g), Math.round(255 * c.b)];
  });
  activation = new Float32Array(atlas.n);
  if (DEBUG.noscan) { S.atlasReady = true; loadingStep("the nervous system is drawn", 0.6); return; }
  scan = new BrainScan($("scan"), atlas, { style: brainView === "brain" ? "brain" : "scan", particles: false, labels: $("labels"), strip: $("strip"), shell: false, spin: true, spinRate: 0.06, montageRows: 9, restAlpha: 0.06, lineBudget: Number(params.get("lines") || 12000), particleBudget: Number(params.get("particles") || 8000), background: [0, 0.012, 0.004], hot: [0.85, 1.0, 0.88], cool: [0.2, 0.75, 0.45], exposure: 0.62, glow: 2.4, bloom: 0.9, labelTop: 4, labelCount: 5, heatDecay: 0.93, dpr: Number(params.get("dpr") || 1.25) });
  scan.set(activation);
  $("legend").innerHTML = atlas.regions.map((r) => `<span><i style="background:rgb(${r.color.join(",")})"></i>${r.name}</span>`).join("");
  S.atlasReady = true; loadingStep("the nervous system is drawn", 0.6);
  setTimeout(() => { if (brainView !== "settling") { scan.resize?.(); scan.fit(); } }, 50);
  addEventListener("resize", () => { if (brainView !== "settling") { scan.resize?.(); scan.fit(); } settlementView.resize?.(); });
  setBrainView(brainView);
});
function styleButtons(which) { for (const style of ["brain", "scan", "settling"]) $("style-" + style).classList.toggle("on", which === style); }
S.speed = 1;

// ---- loading -------------------------------------------------------------------------------
function loadingStep(text, frac) { $("loading-text").textContent = text; $("loading-bar").style.width = (100 * frac).toFixed(0) + "%"; }
function finishLoading() { if (S.ready) return; S.ready = true; S.clockReset = true; loadingStep("ready", 1); setTimeout(() => $("loading").classList.add("gone"), 400); }
loadingStep("loading the nervous system", 0.1);

// ---- the pointer: pick a fruit up, put it down on the table, strike the fly on a fruit ---------
const mouse = new THREE.Vector2(), ray = new THREE.Raycaster();
let carrying = null, hover = null;
function mainCamera() { return rig.camera; }
function fruitUnder(e) {
  if (!room.fruits) return null;
  mouse.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, mainCamera());
  let best = null, bestD = Infinity;
  for (const [name, f] of Object.entries(room.fruits)) {
    if (!f.group) continue;
    const hits = ray.intersectObject(f.group, true);
    if (hits.length && hits[0].distance < bestD) { best = name; bestD = hits[0].distance; }
  }
  if (best) return best;
  // a generous pick: the ray passing within 6 cm of a fruit's position
  for (const [name, f] of Object.entries(room.fruits)) { const p = toThree(...f.pos); const d = ray.ray.distanceToPoint(p); if (d < 0.06 && d < bestD) { best = name; bestD = d; } }
  return best;
}
function tableUnder(e) {
  mouse.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, mainCamera());
  if (room.tableMesh) { const hits = ray.intersectObject(room.tableMesh, true); if (hits.length) return fromThree(hits[0].point); }
  // the table top as a plane, when no mesh is exposed
  const t = room.table, plane = new THREE.Plane().setFromNormalAndCoplanarPoint(new THREE.Vector3(0, 1, 0), toThree(0, 0, t.z)), hit = new THREE.Vector3();
  if (ray.ray.intersectPlane(plane, hit)) { const p = fromThree(hit); if (p[0] > t.x0 && p[0] < t.x1 && p[1] > t.y0 && p[1] < t.y1) return p; }
  return null;
}
function flyUnder(e) {
  mouse.set(e.clientX / innerWidth * 2 - 1, -(e.clientY / innerHeight) * 2 + 1);
  ray.setFromCamera(mouse, mainCamera());
  return ray.ray.distanceToPoint(toThree(...flight.p)) < (life.mode === "flying" ? 0.03 : 0.05);
}
canvas.addEventListener("pointermove", (e) => { const f = fruitUnder(e); if (f !== hover) { hover = f; if (room.highlightFruit) room.highlightFruit(carrying || hover); canvas.style.cursor = f || carrying ? "pointer" : "default"; } });
canvas.addEventListener("pointerdown", (e) => {
  if (carrying) { // put the fruit down where the table was clicked
    const p = tableUnder(e);
    if (p && room.placeFruit) { const t = room.table, x = Math.min(t.x1 - 0.08, Math.max(t.x0 + 0.08, p[0])), y = Math.min(t.y1 - 0.08, Math.max(t.y0 + 0.08, p[1])), from = room.fruits[carrying].pos.slice(); room.placeFruit(carrying, [x, y]); invalidateObservation("fruit moved"); if (life.fruitMoved(carrying, from)) queueReward(); if (!S.paused) life.resumeBodySupport(); }
    carrying = null; if (room.highlightFruit) room.highlightFruit(null); return;
  }
  if (flyUnder(e) && life.mode !== "flying") { strike(); return; }
  const f = fruitUnder(e);
  if (f) { carrying = f; if (room.highlightFruit) room.highlightFruit(f); life.log(`the ${f} is picked up`); }
});
canvas.addEventListener("wheel", (e) => { rig.wheel(e.deltaY); e.preventDefault(); }, { passive: false });
function strike() {
  if (S.paused || !S.ready) return;
  // Credit any already executed choice before withdrawing its observation.
  // The physical impulse changes the next real sensory solve; the display
  // never invents a neural response or a reward for an unexecuted choice.
  const punished = life.punished();
  life.log(punished ? "struck" : "startled");
  if (punished) queueReward();
  flight.swat([0.3 * (Math.random() - 0.5), 0.3 * (Math.random() - 0.5), 0.6], [0, 0, 2], [20 * (Math.random() - 0.5), 20 * (Math.random() - 0.5), 10]);
  invalidateObservation("physical nudge changed observation");
  life.respondToNudge();
}
function queueReward() {
  const r = life.pendingReward; if (!r) return;
  life.pendingReward = null;
  if (!S.learnReady) return;
  if (!r.neuralCredit || r.fresh) {
    cancelAssistedDecision(); life.log("outcome without an executed neural choice; no update"); return;
  }
  worker.postMessage({ type: "assist:reward", ...r, senses: lastSenses, steps: OBSERVATION_CAP, tolerance: CONTROL_TOLERANCE });
  S.assistedDecision = null;
}

// ---- console actions -------------------------------------------------------------------------
function setCam(mode) { rig.setMode(mode); setEyeMain(false); }
function setSpeed(speed) { S.speed = speed; S.clockReset = true; }
function invalidateObservation(reason, retainState = false) {
  cancelAssistedDecision(); life.revokeNeural(reason);
  if (!retainState) brainStates.invalidate(reason, S.requestId);
  pendingObservationTime = null;
  S.pending = false; S.pendingId = null; S.pendingSince = 0; S.controlReply = null;
  if (S.settlementProgress && !S.settlementProgress.completed) S.settlementProgress.withdrawn = true;
  // Do not queue a second solve behind an obsolete synchronous job. Its final
  // reply releases workerBusy even though it can no longer grant authority.
  refreshPaths();
}
function cancelAssistedDecision() {
  if (S.assistedDecision) worker?.postMessage({ type: "assist:cancel", ...S.assistedDecision });
  S.assistedDecision = null;
}
function resetBrain() {
  life.revokeNeural("reset");
  const sugar = life.sugar;
  S.ready = false; S.generation++; S.pending = false; S.pendingId = null; S.pendingSince = 0; S.controlReply = null;
  S.workerBusy = false; S.workerJobId = null; S.workerStarted = 0; S.progressFrames = 0;
  S.brainReady = false; S.learnReady = false; S.lessonStats = null; S.lastLesson = null;
  S.simTime = 0; S.brainDue = 0; S.brainSteps = 0;
  S.controlReplies = 0; S.controlAccepted = 0; S.controlLate = 0; S.controlFailed = 0; S.solveMs = 0;
  S.clockReset = true; S.lessons = []; S.decisions = 0; S.assistedDecision = null;
  clearInspection(); lastDiagnostic = null; settlementInputs.clear(); settlementView.reset(S.generation); lastSenses = {};
  brainStates.reset(S.generation); pendingObservationTime = null;
  flight = new Flight([1.2, 1.0, 1.2], 0.3); attachSurfaces(flight);
  life = makeLife(flight, S.seed, S.generation);
  resetScanReplay();
  watchLifeEvents(); lastCourseId = null; lastCourseActive = false; lastCoursePhase = null; lastBodyMode = life.mode; lastExecutedChoice = null; lastStartupPhase = null; life.setSugar(sugar);
  pathDecision = null; appliedPathObservation = null; refreshPaths();
  life.log("new brain and body"); startWorker(); hud();
  if (payloadInfo) fillCard();
}
function setSugar(name) {
  invalidateObservation("sugar moved");
  life.setSugar(name); if (room.setSugar) room.setSugar(name);
  if (!S.paused && S.ready) life.resumeBodySupport();
}
function togglePause() {
  if (S.paused && inspectionPending) { settlementView.setStatus("Pixel comparison is running; the body remains paused until it finishes."); return; }
  invalidateObservation("paused or resumed", true); S.clockReset = true; S.paused = !S.paused;
  if (!S.paused) life.resumeBodySupport();
  document.body.classList.toggle("paused", S.paused);
  life.log(S.paused ? "paused" : "resumed"); refreshPaths(); hud();
}
// One detail panel at a time. Hiding a viewer never removes retinal input.
function setDetailPanel(which) {
  detailPanel = which;
  for (const name of ["brain", "paths", "instruments", "about"]) {
    document.body.classList.toggle(`detail-${name}`, which === name);
  }
  document.body.classList.toggle("instruments", which === "instruments");
  $("card").classList.toggle("open", which === "about");
  setPaths(which === "paths");
  setBrainView(brainView);
  document.body.classList.toggle("details-open", which !== null);
}
for (const b of document.querySelectorAll("[data-close-detail]")) b.onclick = showDefaultScan;
setDetailPanel(params.get("paths") === "1" ? "paths" : detailPanel);
$("eye").addEventListener("click", () => setEyeMain(!S.eyeMain));
$("eye").addEventListener("keydown", e => {
  if (e.key === "Enter" || e.key === " ") { e.preventDefault(); e.stopPropagation(); setEyeMain(!S.eyeMain); }
});
function gust() {
  if (S.paused || !S.ready) return;
  const [w, x, y, z] = flight.q, c = cos_(-0.175), sn = sin_(-0.175);
  flight.q = [w * c - y * sn, x * c + z * sn, y * c + w * sn, z * c - x * sn];
  flight.kick(0, -25, 0); invalidateObservation("gust changed observation"); life.log("a gust");
  life.respondToNudge();
}
function executeCommand(raw) {
  const parsed = parseConsoleCommand(raw);
  if (!parsed) return;
  const { command, args } = parsed;
  const noArgs = () => { if (args.length) throw new Error(`/${command} takes no arguments.`); };
  const option = (choices, fallback) => {
    if (args.length > 1 || (args.length && !choices.includes(args[0]))) throw new Error(`Use /${command} ${choices.join(" | ")}.`);
    return args[0] ?? fallback;
  };
  switch (command) {
    case "help": noArgs(); return [
      "/status — current brain, body and learning readings",
      "/pause /resume · /gust /nudge — interact with the fly",
      "/brain [settling|3d|scan|off] · /paths [on|off] · /senses [on|off]",
      "/eye [on|off|full] · /camera [follow|room] · scroll the scene to zoom",
      "/speed [0.25|0.5|1] · /sugar [banana|bread|none]",
      "/about /close · /reset — new brain and body · /clear — clear this log",
      "Enter runs a command · ↑/↓ history · Tab completes · / focuses · Escape leaves input, then closes views",
    ];
    case "status": noArgs(); return formatBrainStatus(stateSnapshot(), {
      mode: life.mode, speedScale: S.speed, navigation: null,
      hybrid: true,
      goal: structuredClone(life.authority.goal ?? null),
      startup: structuredClone(life.authority.startup ?? null),
      targetSelection: structuredClone(life.authority.targetSelection.last ?? null),
      waitingSupport: structuredClone(life.authority.waitingSupport ?? null),
      progress: structuredClone(S.settlementProgress),
      motionSupport: structuredClone(life.authority.motionSupport),
    });
    case "pause": case "resume": {
      noArgs(); const pause = command === "pause";
      if (S.paused !== pause) togglePause();
      return S.paused ? "Paused. /status shows recorded readings; /resume continues." : "Running.";
    }
    case "gust": case "nudge":
      noArgs();
      if (S.paused || !S.ready) return "Resume after the brain has loaded to apply a disturbance.";
      if (command === "gust") gust(); else strike();
      return "Physical disturbance applied; waiting for a fresh sensory solve.";
    case "brain": {
      const mode = option(["settling", "3d", "scan", "off"], "scan");
      if (mode === "off") { if (detailPanel === "brain") setDetailPanel(null); return "Brain view closed; neural computation continues."; }
      setBrainView(mode === "3d" ? "brain" : mode); setDetailPanel("brain"); return "Brain inspector open. /close returns to flight.";
    }
    case "paths": case "senses": {
      const panel = command === "paths" ? "paths" : "instruments";
      const on = option(["on", "off"], "on") === "on";
      if (on) setDetailPanel(panel); else if (detailPanel === panel) showDefaultScan();
      return `${command === "paths" ? "Flight path" : "Sensory and motor readings"} ${on ? "open" : "closed"}.`;
    }
    case "eye": {
      const view = option(["on", "off", "full"], "on");
      if (view === "full") setEyeMain(true); else setEyeVisible(view === "on");
      return view === "full" ? "Fly's eye fills the scene. Escape or /eye on restores the inset." : `Fly's-eye inset ${view}; retinal input continues.`;
    }
    case "camera": { const mode = option(["follow", "room"], "follow"); setCam(mode); return `${mode} camera. Scroll over the scene to zoom.`; }
    case "speed": { const speed = option(["0.25", "0.5", "1"], String(S.speed)); setSpeed(Number(speed)); return `Simulation time ${speed}×.`; }
    case "sugar": {
      const fruit = option(["banana", "bread", "none"], life.sugar || "none");
      setSugar(fruit === "none" ? null : fruit); return `Sugar: ${fruit}.`;
    }
    case "reset": noArgs(); resetBrain(); return "Reloading the brain and resetting its body and learning.";
    case "about": noArgs(); setDetailPanel("about"); return "Model details open.";
    case "close": noArgs(); showDefaultScan(); setEyeMain(false); return "Back to flight and the nervous-system scan.";
    case "clear": noArgs(); terminal.clear(); return;
    default: throw new Error(`Unknown command /${command}. Type /help.`);
  }
}
addEventListener("keydown", (e) => {
  if (e.target instanceof HTMLElement && (e.target.matches("input,select,textarea,a") || e.target.isContentEditable)) return;
  if (e.key === "Escape") { showDefaultScan(); setEyeMain(false); }
  if (e.key === "/") { e.preventDefault(); terminal.focus(); $("console-input").value = "/"; }
  if (e.key === "1") setCam("follow"); if (e.key === "2") setCam("room"); if (e.key === "3") setCam("eye");
  if (e.key === " ") { e.preventDefault(); togglePause(); }
  if (e.key === "i") setDetailPanel(detailPanel === "instruments" ? null : "instruments"); if (e.key === "e") setEyeMain(!S.eyeMain);
  if (e.key.toLowerCase() === "g") gust(); if (e.key.toLowerCase() === "w") strike();
});

function resize() { const w = innerWidth, h = innerHeight; renderer.setSize(w, h); composer.setSize(w, h); bloom.resolution.set(w / 2, h / 2); rig.camera.aspect = w / h; rig.camera.updateProjectionMatrix(); }
addEventListener("resize", resize); resize();

// ---- the HUD --------------------------------------------------------------------------------
const SENSE_ROWS = [["haltere tone", (s) => s["haltere:left"]], ["ocellus L", (s) => s["ocelli:left"]], ["ocellus R", (s) => s["ocelli:right"]], ["banana odour L", (s) => s["orn:decaying_fruit:left"]], ["banana odour R", (s) => s["orn:decaying_fruit:right"]], ["bread odour L", (s) => s["orn:yeasty:left"]], ["bread odour R", (s) => s["orn:yeasty:right"]], ["sugar", (s) => s["grn:sugar:labellum"]], ["leg touch", (s) => s["leg_touch"]], ["wind", (s) => s["jo:C:left"]]];
const MOTOR_ROWS = [["MBON11 approach", "mbon:MBON11:right"], ["MBON05 avoid", "mbon:MBON05:left"], ["PAM dopamine", "dan:pam"], ["PPL1 dopamine", "dan:ppl1"], ["Kenyon cells", "kc"], ["DNa02 left", "dn:DNa02:left"], ["DNa02 right", "dn:DNa02:right"], ["DNp09 forward", "dn:DNp09"], ["landing DNs", "dn:landing"], ["giant fibre", "gf"], ["MN9 feeding", "mn9"], ["power L", "power:left"], ["power R", "power:right"]];
function row(parent, label, cls) { const d = document.createElement("div"); d.className = "row"; d.innerHTML = `<span>${label}</span><div class="bar ${cls}"><i></i></div><span class="num">0.00</span>`; parent.appendChild(d); return { fill: d.querySelector("i"), num: d.querySelector(".num"), label: d.querySelector("span") }; }
const senseBars = SENSE_ROWS.map(([label]) => row($("senses"), label, "")), motorBars = MOTOR_ROWS.map(([label]) => row($("motors"), label, "motor"));
let lastSenses = {};
// Numeric telemetry is separate from the controller. Only current, matching
// sensory solves supply activity/iteration/mismatch cards; diagnostics cannot.
function stateSnapshot() {
  const neural = brainStates.read({ time: life.clock, generation: S.generation, enabled: S.brainReady, paused: S.paused });
  const sample = neural.available ? neural.sample : null;
  const neurons = payloadInfo?.n ?? null;
  return { generation: S.generation, time: life.clock, paused: S.paused, neural,
    hunger: life.hunger, neurons,
    activityFraction: sample?.converged && sample.active !== null && neurons ? sample.active / neurons : null,
    settlingSteps: sample?.iterations ?? null,
    learningUpdates: S.learnReady ? S.lessonStats?.updates ?? 0 : null,
    changedConnections: S.learnReady ? S.lessonStats?.changed ?? 0 : null,
    attention: life.search?.neuralChoice && !life.search.neuralChoice.revoked ? life.search.fruit : null,
    speed: flight.speed(), altitude: flight.p[2], wingHz: life.controls.f };
}
function hud() {
  if (document.body.classList.contains("instruments")) {
    SENSE_ROWS.forEach(([_, f], k) => { const v = Math.max(0, Math.min(1, f(lastSenses) || 0)); senseBars[k].fill.style.width = (100 * v).toFixed(1) + "%"; senseBars[k].num.textContent = v.toFixed(2); });
    MOTOR_ROWS.forEach(([_, name], k) => { const v = Math.max(0, Math.min(1, life.readouts[name] || 0)); motorBars[k].fill.style.width = (100 * v).toFixed(1) + "%"; motorBars[k].num.textContent = v.toFixed(2); });
  }
  $("stats").textContent = "";
  const info = payloadInfo ? `<b>${payloadInfo.n.toLocaleString()}</b> neurons · ${(payloadInfo.edges / 1e6).toFixed(2)} M connection classes · last solve ${S.solveMs.toFixed(0)} ms` : "loading the brain";
  $("brainfoot").innerHTML = info + (scan ? ` · ${S.fps.toFixed(0)} fps` : "");
}

// ---- the model card -------------------------------------------------------------------------
function fillCard() {
  const m = payloadInfo;
  $("eye-input").textContent = "live sensory view";
  $("card-brain").innerHTML = `<b>A connected brain.</b> All ${m.n.toLocaleString()} neurons in the retained BANC fixture participate in a recurrent patch net, connected by ${m.edges.toLocaleString()} directed connection classes.`;
  $("card-body").innerHTML = "<b>Brain chooses goals · body executes.</b> Settled neural odor comparisons select banana or bread; mushroom-body outputs choose approach or avoid. Matching executed choices can learn from outcomes. Fresh checked motor outputs can add wing steering trim. The body controller supplies routes, room clearance, flight balance, landing/perch hold, takeoff, feeding and grooming. These movement routines are supplied mechanics, not learned motor competence.";
  $("card-senses").innerHTML = "<b>Sense, settle, respond.</b> Head-camera pixels enter 1,831 photoreceptors. Odor, taste, touch, rotation and airspeed enter other sensory populations. Use a gust or nudge to change the input, then inspect the measured settlement.";
  $("card-claim").innerHTML = "<b>Read /status.</b> Hunger is the body's appetite variable. Neural activity counts cells at activity ≥0.5. Settling steps count actual solver updates, and learning updates count completed outcome-linked learning calls. Mismatch is the largest neural equation defect before and after a sensory solve. Paused readings are marked recorded; expired readings clear.";
  $("card-sources").innerHTML = '<li>BANC release 888, Lee lab and BANC community (CC BY).</li><li><a href="https://github.com/muellerberndt/cadence-examples/blob/main/fly-matrix/docs/FLY_MATRIX_BRAIN">How the model works: equations, senses, learning and body control.</a></li>';

}

// ---- the fly's view ----------------------------------------------------------------------------------
function drawEye(now) { // the rect in CSS pixels from the bottom-left: three.js applies the pixel ratio itself
  if (S.eyeMain) { eye.render(flight, 0, 0, innerWidth, innerHeight, now); return; }  // the compound eye as the whole screen (key e)
  const box = $("eye"); if (!box) return;
  const r = box.getBoundingClientRect();
  if (r.width < 8 || r.height < 8) return;
  eye.render(flight, Math.round(r.left), Math.round(innerHeight - r.bottom), Math.round(r.width), Math.round(r.height), now);
}
function setEyeMain(on) {
  S.eyeMain = !!on;
  if (on) eyeVisible = true;
  $("eye").hidden = !eyeVisible;
  $("eye").style.visibility = on ? "hidden" : "";
  document.body.classList.toggle("eye-full", S.eyeMain);
}
function setEyeVisible(on) { eyeVisible = !!on; setEyeMain(false); $("eye").hidden = !eyeVisible; }

// ---- the loop --------------------------------------------------------------------------------
// Keep ordinary rendering stalls on the 1× wall clock. Hidden tabs and explicit
// pause/reset still discard their backlog; a single severe stall is bounded.
const physicsClock = new PhysicsClock({ maxWallStep: 1 });
const physicalStep = { step(dt) {
  const before = life.authority.motionSupport.appliedFrames;
  const decides = life.step(dt); recordPath(true);
  const support = life.authority.motionSupport, observation = support.source;
  const enacted = support.appliedFrames > before;
  // The log observes the same physical step as the path recorder. Admission
  // alone is not execution, and recording a command never actuates the fly.
  if (enacted && observation && observation.requestId !== lastCourseId) {
    const c = support.originalCommand, heading = life.heading + (c.forwardSpeed < 0 ? Math.PI : 0);
    terminal.write("course", `Input #${observation.requestId}: neural heading ${((heading * 180 / Math.PI % 360 + 360) % 360).toFixed(1)}° · turn ${(c.yawRate * 180 / Math.PI).toFixed(2)}°/s · travel target ${(Math.hypot(...support.desiredVelocityWorld.slice(0, 2)) * 100).toFixed(0)} cm/s · ${support.phase === "fresh" ? "fresh input" : "recorded input"}.`,
      { generation: S.generation, requestId: observation.requestId, framesApplied: support.appliedFrames,
        freshFramesApplied: life.authority.navigation.framesApplied, phase: support.phase, command: { ...c }, heading });
    lastCourseId = observation.requestId;
  }
  if (enacted && lastCoursePhase === "fresh" && support.phase !== "fresh") terminal.write("carry", "Continuing the recorded neural course while the next input settles.",
    { generation: S.generation, requestId: observation?.requestId ?? lastCourseId, phase: support.phase });
  if (lastCourseActive && !enacted) terminal.write("hold", "Course execution stopped; waiting for a checked input.", { generation: S.generation, requestId: lastCourseId });
  lastCoursePhase = support.phase;
  lastCourseActive = enacted;
  const choice = life.search?.neuralChoice;
  if (choice?.executed && choice.requestId !== lastExecutedChoice) {
    terminal.write("choice", `Choice #${choice.requestId} executed: ${choice.action === 0 ? "approach" : "avoid"} ${choice.fruit} through the body controller.`,
      { generation: S.generation, requestId: choice.requestId, searchToken: choice.searchToken, action: choice.action, fruit: choice.fruit, hybrid: true });
    lastExecutedChoice = choice.requestId;
  }
  const startup = life.authority.startup;
  if (startup && startup.phase !== lastStartupPhase) {
    terminal.write("body", startup.active
      ? "Launch along the initial body heading while the brain settles its first choice; no food selected."
      : `Initial body launch ${startup.phase}${startup.reason ? `: ${startup.reason}` : ""}.`,
      { generation: S.generation, startup: structuredClone(startup), neuralCredit: false });
    lastStartupPhase = startup.phase;
  }
  if (life.mode !== lastBodyMode) { terminal.write("body", life.mode, { generation: S.generation }); lastBodyMode = life.mode; }
  if (decides) { life.decide(); if (life.odourTick() && S.learnReady) life.wantDecision = true; }
} };
document.addEventListener("visibilitychange", () => { S.clockReset = true; });
let last = performance.now();
function frame(now) {
  requestAnimationFrame(frame);
  const elapsedWall = Math.max(0, (now - last) / 1000); last = now;
  const dtWall = Math.min(0.05, elapsedWall);
  if (S.clockReset) physicsClock.reset();
  const physicsElapsed = S.clockReset || document.hidden ? 0 : elapsedWall; S.clockReset = false;
  if (S.ready && !S.paused) {
    // Keep fixed 0.5 ms physical steps and fractional time between frames.
    // Rendering no longer silently converts a slow frame into slow motion.
    // The clock bounds long stalls; resets/hidden tabs never accumulate a jump.
    const advanced = physicsClock.advance(physicalStep, physicsElapsed, S.speed);
    S.simTime = life.clock; S.brainDue += advanced;
    lastSenses = life.senses(null, advanced);
    if (life.pendingReward) queueReward();
    if (S.assistedDecision && life.search?.neuralToken !== S.assistedDecision.searchToken) cancelAssistedDecision();
    consumeControlReply();
    if (S.workerBusy && now - S.workerStarted > 30000) { worker.terminate(); workerFailed("brain timeout; reset to reconnect"); }
    if (!S.pending && !S.workerBusy && S.brainReady && S.learnReady) {
      let request = null;
      const observationDue = S.brainDue >= .1;
      const observe = () => {
        const r = life.expectObservation(++S.requestId, S.generation, { steps: OBSERVATION_CAP, tolerance: CONTROL_TOLERANCE });
        if (r) r.type = "assist:observe";
        return r;
      };
      // Goal requests carry a fresh sensory snapshot and independently certify
      // both attended candidates and the selected free/positive/negative phases.
      // A failed unattended observation must not prevent those solves. This
      // scheduling permission grants no direct motor or learning authority.
      const job = chooseWorkerJob({ observationDue, wantDecision: life.wantDecision,
        qualifiedObservationId: qualifiedObservationId(life, S.generation), lastDecisionId: life._lastDecisionId,
        decisionCertifiesInput: true });
      if (job === "decision") {
        life.wantDecision = false;
        request = life.openDecision(++S.requestId, S.generation, { steps: OBSERVATION_CAP, tolerance: CONTROL_TOLERANCE, temperature: LESSONS.temperature });
        if (request) { request.type = "assist:decide"; S.assistedDecision = { requestId: request.requestId, generation: request.generation, searchToken: request.searchToken }; }
      }
      if (!request && observationDue) request = observe();
      if (request) {
        S.pending = true; S.pendingId = request.requestId; S.pendingSince = now; S.brainDue = 0;
        S.workerBusy = true; S.workerJobId = request.requestId; S.workerStarted = now;
        pendingObservationTime = request.type === "assist:observe" ? life.clock : null;
        // Capture actual head-camera pixels for every neural observation, even
        // when the eye inset is hidden. A failed capture grants no neural command.
        try {
          const retina = eye.capture(flight);
          if (request.type === "assist:observe" && detailPanel === "brain" && brainView === "settling") request.trace = TRACE_OPTIONS;
          if (request.type === "assist:observe" && scan && detailPanel === "brain" && brainView !== "settling") request.replay = REPLAY_OPTIONS;
          rememberSettlementInput(request, retina);
          worker.postMessage({ ...request, senses: lastSenses, retina, progress: true });
        }
        catch (error) { workerFailed(`retinal capture failed: ${error.message}`); }
      }
    }
  }
  const cam = mainCamera();
  refreshPaths(false, now);
  fly.update(flight, life.controls, dtWall, room.surfaceZ, cam === rig.camera && rig.mode === "room", life.pose());
  rig.update(flight, S.paused ? 0 : dtWall, null, { orbit: false, trackHeading: false });
  if (cam.aspect !== innerWidth / innerHeight) { cam.aspect = innerWidth / innerHeight; cam.updateProjectionMatrix(); }
  glitches(dtWall);
  composer.render();
  if (!DEBUG.noviews && (eyeVisible || S.eyeMain)) drawEye(now);
  if (scan) {
    if (detailPanel === "brain" && brainView !== "settling") { showLiveProgress(); advanceScanReplay(now); }
    if (S.newState && !neuralReplay && !S.workerBusy && now - S.lastScan > 33) { scan.set(activation, { draw: false }); S.newState = false; S.lastScan = now; }
    if (detailPanel === "brain" && brainView !== "settling" && now - S.lastScanDraw >= SCAN_INTERVAL_MS && (DEBUG.scanrate <= 1 || (S.frames % DEBUG.scanrate) === 0)) { scan.draw(now); S.lastScanDraw = now; }
  }
  if (detailPanel === "brain" && brainView === "settling") settlementView.update(now);
  S.frames++; S.fpsClock += elapsedWall;
  if (S.fpsClock >= 1) { S.fps = S.frames / S.fpsClock; S.frames = 0; S.fpsClock = 0; window.__fps = S.fps; }
  if (now - S.lastHud >= 200) { hud(); S.lastHud = now; }
  window.__frames = (window.__frames || 0) + 1;
}
requestAnimationFrame(frame);
// Presentation-only close-up (index.html?card=1). The brain and physics keep
// running normally; historical frozen/sitting pose query parameters are ignored.
if (params.get("card") === "1") {
  for (const id of ["panel", "brain-console", "brain", "eye", "title", "credit", "card"]) { const el = $(id); if (el) el.style.display = "none"; }
  document.body.classList.add("card");
  rig.zoom = Number(params.get("d") || 0.0065);

}
window.__app = { S, life: () => life, flight: () => flight, scan: () => scan, brainInfo: () => payloadInfo && ({ n: payloadInfo.n, edges: payloadInfo.edges, synapses: payloadInfo.synapses, whole: payloadInfo.whole }), setCam, setSugar, setEyeMain, strike, paths: pathState, pathDisplay: () => displayedPaths && structuredClone(displayedPaths), setPaths, pathAnnotations: pathView.group, get worker() { return worker; }, eye, renderer, rig, room };
window.__app.setBrainView = setBrainView;
window.__app.console = { snapshot: () => terminal.snapshot() };
window.__app.scanReplay = () => displayedReplay ? { ...structuredClone(displayedReplay), displayed: detailPanel === "brain" && brainView !== "settling" } : null;
window.__app.settlementProgress = () => displayedProgress ? structuredClone(displayedProgress) : null;
window.__app.details = () => ({ panel: detailPanel, eyeVisible, eyeMain: !!S.eyeMain, speed: S.speed, droppedWallSeconds: physicsClock.droppedWallSeconds });
window.__app.compareRetinalPixels = compareRetinalPixels;
window.__app.settlement = () => ({ mode: brainView, generation: S.generation,
  pendingDiagnostic: inspectionPending ? { requestId: inspectionPending.requestId, generation: inspectionPending.generation } : null,
  diagnostic: lastDiagnostic ? structuredClone(lastDiagnostic) : null, view: settlementView.snapshot() });

window.__app.states = () => structuredClone(stateSnapshot());
