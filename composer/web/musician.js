import { BrainScan, layoutAtlas } from "./brain_scan.js";
const $ = (id) => document.getElementById(id);
const DELTAS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
const DURATIONS = [1, 2, 3, 4, 6, 8, 12, 16, 24, 32, 48, 64];
const FAMILY_COLORS = ["#f4c86a", "#89e6bf", "#f6a061", "#baa7fb", "#6ba6ff", "#f691ad", "#c9d6d3", "#ff7b7b"];
const LABELS = {
  ear: "Ear · 16 heard events",
  sense: "Sense · chroma, sounding, clock",
  mood: "Mood · 7 classes",
  interval: "Interval sense · relative pitch",
  belt: "Auditory belt · shared embedding",
  melody: "Melody cortex",
  harmony: "Harmony cortex",
  rhythm: "Rhythm cortex",
  timbre: "Timbre cortex",
  form: "Form cortex · coherence",
  phrase: "Phrase cortex",
  prefrontal: "Prefrontal · working memory",
  recall: "Recall · form record",
  piece: "Piece · slow trace",
  intention: "Intention · next event",
};
// What each region is to the musician: the scan colours regions by role and reads the
// rest from the names (ear, sense, the cortices, prefrontal, recall).
const ROLES = { mood: "value", interval: "sensory", belt: "sensory", form: "association", piece: "memory", intention: "motor" };
const SHORT = Object.fromEntries(Object.entries(LABELS).map(([name, label]) => [name, label.split(" ·")[0]]));

let info = null; // /api/state brain + topology
let composition = null; // the loaded composition record
let liveRoll = { events: [], starts: [], total: 0, candidates: [], phraseStart: 0, valences: [], surprise: [], plan: null };
let mode = "idle"; // idle | composing | playing | scrub
const frameCache = new Map(); // "live/3" -> {data: Float32Array, peaks}
const pending = new Map();
let current = { key: null, frame: null, iteration: 0, event: null, phase: "" };
const liveQueue = []; // frame keys still to animate during composing
let liveIteration = 0;
let eventCursor = 0;
let silentClock = null;
let lastTick = 0;
let hover = null;
let shown = {}, // the frame the scan shows: {key, frame, iteration, signal}
  glow = null; // the heat of the iteration shown, for the afterglow of the next
const EMPTY_ATLAS = { n: 0, synapses: 0, regions: [], region: new Uint16Array(0), positions: new Float32Array(0), pre: new Uint32Array(0), post: new Uint32Array(0), weight: new Float32Array(0), palette: {} };
const network = new BrainScan($("brain-network"), EMPTY_ATLAS, { labels: $("brain-labels"), interaction: $("brain") });
$("fit-brain").onclick = () => network.fit();
$("expand-brain").onclick = () =>
  document.fullscreenElement ? document.exitFullscreen() : $("brain").closest(".brain-panel").requestFullscreen?.();
$("brain").addEventListener("pointermove", (e) => {
  const r = $("brain").getBoundingClientRect();
  hover = [e.clientX - r.left, e.clientY - r.top, e.clientX, e.clientY];
});
$("brain").addEventListener("pointerleave", () => (hover = null));
document.querySelectorAll("[data-mood]").forEach((b) => (b.onclick = () => ($("mood").value = b.dataset.mood)));
$("scrub").oninput = () => {
  $("follow").checked = false;
  current.iteration = +$("scrub").value;
};

const fmt = (n) => Number(n).toLocaleString();
function fit(canvas) {
  const r = canvas.getBoundingClientRect(),
    dpr = Math.min(devicePixelRatio, 2);
  if (canvas.width !== Math.round(r.width * dpr) || canvas.height !== Math.round(r.height * dpr)) {
    canvas.width = Math.round(r.width * dpr);
    canvas.height = Math.round(r.height * dpr);
  }
  const c = canvas.getContext("2d");
  c.setTransform(dpr, 0, 0, dpr, 0, 0);
  c.clearRect(0, 0, r.width, r.height);
  return [c, r.width, r.height];
}
function text(parent, tag, content, cls) {
  const el = document.createElement(tag);
  el.textContent = content;
  if (cls) el.className = cls;
  parent.append(el);
  return el;
}

// ---- brain topology and layout --------------------------------------------------------
async function loadBrain() {
  const state = await (await fetch("/api/state")).json();
  info = state;
  const g = state.topology;
  $("neurons").textContent = g.synapses_total && g.synapses_total > g.synapses
    ? `${fmt(g.neurons)} neurons · ${fmt(g.synapses)} strongest of ${fmt(g.synapses_total)} synapses drawn`
    : `${fmt(g.neurons)} neurons · ${fmt(g.synapses)} synapses`;
  $("brain-line").textContent = `${state.brain.design.cortex}-wide cortices, ${fmt(state.brain.trainable_parameters)} trainable parameters · ${state.backend} backend`;
  const data = new Float32Array(await (await fetch(g.url)).arrayBuffer());
  mount(g, data);
  const table = document.createElement("table");
  for (const [name, r] of Object.entries(g.regions)) {
    const tr = document.createElement("tr");
    text(tr, "td", LABELS[name] || name);
    text(tr, "td", fmt(r.neurons), "n");
    table.append(tr);
  }
  $("regions").replaceChildren(table);
  const t = state.training || {};
  $("training").textContent = t.best_validation_nll
    ? `Checkpoint ${state.checkpoint}: best held-out surprise ${t.best_validation_nll.toFixed(4)} after ${fmt(t.updates)} updates (${(t.seconds / 3600).toFixed(1)} h). Imitation of whole pieces as streams; no backpropagation.`
    : `Checkpoint ${state.checkpoint}.`;
  if (state.result) useComposition(state.result);
}
// Every synapse of the served topology as typed arrays: source, target, weight.
function split(data) {
  const count = data.length / 3,
    pre = new Uint32Array(count),
    post = new Uint32Array(count),
    weight = new Float32Array(count);
  for (let e = 0; e < count; e++) {
    pre[e] = data[e * 3];
    post[e] = data[e * 3 + 1];
    weight[e] = data[e * 3 + 2];
  }
  return { pre, post, weight };
}
// The whole brain laid out for the scan: regions placed by their connections, neurons by
// their synapses.
function mount(g, data) {
  const { pre, post, weight } = split(data);
  const groups = new Array(g.neurons).fill("other");
  for (const [name, r] of Object.entries(g.regions)) groups.fill(name, r.start, r.start + r.neurons);
  network.setAtlas(layoutAtlas({ n: g.neurons, pre, post, weight, groups, labels: SHORT, roles: ROLES, seed: 0 }));
  network.fit();
  shown = {};
}
function regionColorOf(name) {
  return network.atlas.regions.find((r) => r.name === name)?.color ?? [143, 163, 184];
}

// ---- frames ------------------------------------------------------------------------------
function frameUrl(key) {
  return `/frames/${composition ? composition.id : liveRoll.id}/${key}.bin`;
}
function decode(buffer) {
  const n = info.topology.neurons,
    values = new Float32Array(buffer),
    iterations = values.length / (3 * n);
  // the change scale of each iteration: the largest repair so far, easing by a tenth per
  // iteration as the settling calms so later, smaller repairs stay visible; the mismatch alike
  const scale = new Float32Array(iterations),
    mismatchScale = new Float32Array(iterations);
  let s = 1e-6,
    m = 1e-6;
  for (let f = 0; f < iterations; f++) {
    let peak = 0,
      worst = 0;
    const rep = (f * 3 + 1) * n,
      mis = (f * 3 + 2) * n;
    for (let i = 0; i < n; i++) {
      const r = Math.abs(values[rep + i]),
        e = Math.abs(values[mis + i]);
      if (r > peak) peak = r;
      if (e > worst) worst = e;
    }
    s = Math.max(peak, s * 0.9);
    m = Math.max(worst, m * 0.9);
    scale[f] = s;
    mismatchScale[f] = m;
  }
  const regionSeries = {};
  for (const [name, r] of Object.entries(info.topology.regions)) {
    const mean = [],
      mismatch = [],
      repair = [];
    for (let f = 0; f < iterations; f++) {
      let s = 0,
        m = 0,
        p = 0;
      for (let i = r.start; i < r.start + r.neurons; i++) {
        s += values[(f * 3 + 0) * n + i];
        const e = values[(f * 3 + 2) * n + i];
        m += e * e;
        const q = values[(f * 3 + 1) * n + i];
        p += q * q;
      }
      mean.push(s / r.neurons);
      mismatch.push(Math.sqrt(m / r.neurons));
      repair.push(Math.sqrt(p / r.neurons));
    }
    regionSeries[name] = { mean, mismatch, repair };
  }
  const whole = [];
  for (let f = 0; f < iterations; f++) {
    let worst = 0;
    const base = (f * 3 + 2) * n;
    for (let i = 0; i < n; i++) worst = Math.max(worst, Math.abs(values[base + i]));
    whole.push(worst);
  }
  return { values, iterations, n, scale, mismatchScale, regionSeries, whole };
}
// One recorded iteration into the scan: the activation travels as messages, the repair is
// the heat (with afterglow while the replay advances), the chosen signal is the brightness.
function present(frame, i, signal, advancing) {
  const n = frame.n,
    activation = channelOf(frame, i, 0),
    repair = channelOf(frame, i, 1),
    mismatch = channelOf(frame, i, 2);
  const level = new Float32Array(n),
    heat = new Float32Array(n);
  const s = Math.max(1e-9, frame.scale[i]),
    m = Math.max(1e-9, frame.mismatchScale[i]);
  const prior = advancing && glow && glow.length === n ? glow : null;
  for (let k = 0; k < n; k++) {
    const change = Math.min(1, Math.abs(repair[k]) / s);
    heat[k] = Math.max(change, prior ? prior[k] * 0.86 : 0);
    level[k] = signal === "repair" ? repair[k] / s : signal === "mismatch" ? mismatch[k] / m : activation[k];
  }
  glow = heat;
  network.show(activation, heat, { level });
}
function getFrame(key) {
  if (frameCache.has(key)) return frameCache.get(key);
  if (!pending.has(key) && key) {
    pending.set(
      key,
      fetch(frameUrl(key))
        .then((r) => (r.ok ? r.arrayBuffer() : Promise.reject(r.status)))
        .then((buffer) => {
          frameCache.set(key, decode(buffer));
          if (frameCache.size > 24) frameCache.delete(frameCache.keys().next().value);
        })
        .catch(() => {})
        .finally(() => pending.delete(key)),
    );
  }
  return null;
}
function channelOf(frame, iteration, k) {
  const n = frame.n;
  return frame.values.subarray((iteration * 3 + k) * n, (iteration * 3 + k + 1) * n);
}

// ---- composing live ----------------------------------------------------------------------
async function compose() {
  const r = await fetch("/api/compose", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({
      mood: $("mood").value,
      bars: +$("bars").value,
      seed: +$("seed").value,
      futures: +$("futures").value,
      edits: +$("edits").value,
      tempo: +$("tempo").value,
      temperature: +$("temperature").value,
      render: true,
    }),
  });
  const j = await r.json();
  if (!r.ok) $("phase").textContent = j.error;
}
$("compose").onclick = (e) => {
  e.preventDefault();
  compose();
};
function handle(event) {
  const s = event.stage;
  if (s === "priming") {
    mode = "composing";
    liveRoll = { events: [], starts: [], total: event.bars * 16, candidates: [], phraseStart: 0, valences: [], surprise: [], id: null, bpm: event.tempo, plan: null, profiles: null };
    liveQueue.length = 0;
    $("thought").textContent = "Hearing the opening";
    $("thought-detail").textContent = `mood ${Object.entries(event.brief).map(([k, v]) => `${k} ${v}`).join(", ")}`;
    $("alternatives").replaceChildren();
  } else if (s === "planning") {
    $("thought").textContent = `Planning ${event.bars} bars before the first note`;
    $("thought-detail").textContent = "the plan head imagines the profile of every bar; the listener keeps the best-formed plan";
  } else if (s === "planned") {
    liveRoll.plan = event.plan;
    const best = event.plan.candidates[event.plan.winner];
    $("thought").textContent = `Planned ${event.plan.plan.length} bars: chose plan ${event.plan.winner + 1} of ${event.plan.candidates.length}`;
    $("thought-detail").textContent = `contrast ${best.contrast.toFixed(2)} · climax ${best.climax.toFixed(1)} · ending ${best.ending.toFixed(2)} · return ${best.return.toFixed(0)} · plan surprise ${best.mean_plan_surprise.toFixed(2)}`;
    const el = text($("alternatives"), "div", "The plan of the whole piece", "candidate");
    event.plan.candidates.forEach((c, j) => {
      const m = document.createElement("meter");
      m.min = -3;
      m.max = 5;
      m.value = c.score;
      m.title = `Plan ${j + 1}: ${c.score.toFixed(3)}${j === event.plan.winner ? " · chosen" : ""} · contrast ${c.contrast.toFixed(2)} climax ${c.climax} ending ${c.ending.toFixed(2)} return ${c.return}`;
      el.append(m);
    });
  } else if (s === "imagined") {
    const p = event.phrase;
    liveRoll.candidates = p.candidates.map((c, j) => ({ tokens: c.tokens, score: c.score, winner: j === p.winner }));
    liveRoll.phraseStart = p.from_step;
    liveRoll.valences.push({ step: p.from_step, valence: p.valence });
    $("thought").textContent = `Imagined ${p.candidates.length} futures from step ${p.from_step}`;
    $("thought-detail").textContent = `chose future ${p.winner + 1} · valence ${p.valence >= 0 ? "+" : ""}${p.valence.toFixed(3)} · surprise ${p.candidates[p.winner].mean_surprise.toFixed(2)} · mood fit ${p.candidates[p.winner].mood_fit.toFixed(2)}`;
    const el = text($("alternatives"), "div", `Phrase ${p.index + 1} from bar ${Math.floor(p.from_step / 16) + 1}`, "candidate");
    p.candidates.forEach((c, j) => {
      const m = document.createElement("meter");
      m.min = -3;
      m.max = 3;
      m.value = c.score;
      m.title = `Future ${j + 1}: ${c.score.toFixed(3)}${j === p.winner ? " · chosen" : ""} · ${c.issues.join(", ") || "no issues"}`;
      el.append(m);
    });
  } else if (s === "committed") {
    liveRoll.events.push(...event.events);
    liveRoll.starts.push(...event.starts);
    if (event.live) for (let j = event.live[0]; j < event.live[1]; j++) liveQueue.push({ key: `live/${j}`, event: liveRoll.events.length - (event.live[1] - j), phase: "composing" });
    liveRoll.candidates = [];
    $("phase").textContent = `Composing · step ${event.step}/${event.total}`;
  } else if (s === "listening") {
    $("thought").textContent = "Listening to the whole draft";
    $("thought-detail").textContent = `${event.events} events, one stream from the start`;
  } else if (s === "editing") {
    const width = event.width ?? 4;
    $("thought").textContent = `Re-imagining bars ${event.bar + 1}–${event.bar + width}`;
    $("thought-detail").textContent = event.per_bar ? "the bars that disagree most with the plan and with the brain's own expectation" : "the weakest passage by its own surprise, mood fit and health";
  } else if (s === "edited") {
    const width = event.width ?? 4;
    $("thought").textContent = event.accepted ? `Edit of bars ${event.bar + 1}–${event.bar + width} kept` : `Edit of bars ${event.bar + 1}–${event.bar + width} rejected`;
    $("thought-detail").textContent = `whole-piece score ${event.before.toFixed(3)} → ${event.after.toFixed(3)}`;
    if (event.accepted && event.events) {
      liveRoll.events = event.events;
      liveRoll.starts = event.starts;
      if (event.live) for (let j = event.live[0]; j < event.live[1]; j++) liveQueue.push({ key: `live/${j}`, event: event.first_event + (j - event.live[0]), phase: "editing" });
    }
  } else if (s === "listening back") {
    $("thought").textContent = "Listening back to the finished piece";
    $("thought-detail").textContent = "recording every iteration for playback";
  } else if (s === "ready") {
    $("phase").textContent = "Ready";
  } else if (s === "failed") {
    mode = "idle";
    $("phase").textContent = "Failed: " + event.error;
  }
}
async function pollEvents() {
  try {
    const data = await (await fetch(`/api/events?after=${eventCursor}`)).json();
    for (const e of data.events) {
      eventCursor = e.sequence;
      handle(e);
    }
  } catch (err) {
    /* the server may be restarting */
  }
}
let lastResult = null;
async function poll() {
  try {
    const s = await (await fetch(`/api/state?result=${lastResult ?? ""}`)).json();
    if (s.result) useComposition(s.result);
    if (s.busy) {
      $("compose").disabled = true;
      if (mode !== "composing") mode = "composing";
    } else {
      $("compose").disabled = false;
      if (mode === "composing" && !liveQueue.length) mode = composition ? "scrub" : "idle";
    }
    if (s.error) $("phase").textContent = s.error;
  } catch (err) {
    /* keep polling */
  }
}

// ---- a finished composition --------------------------------------------------------------
function useComposition(r) {
  composition = r;
  lastResult = r.id;
  liveRoll.id = r.id;
  frameCache.clear();
  $("midi").href = r.files.midi;
  $("midi").download = "musician.mid";
  if (r.files.audio) {
    $("wav").href = r.files.audio;
    $("wav").download = "musician.wav";
    $("audio").src = r.files.audio;
    $("wav").style.display = "";
  } else {
    $("wav").style.display = "none";
    $("audio").removeAttribute("src");
  }
  $("mood").value = r.mood_text;
  $("seed").value = r.seed;
  $("bars").value = r.bars;
  const kept = r.edits.filter((e) => e.accepted).length;
  $("review").textContent = `${r.events.length} events over ${r.bars} bars at ${r.tempo_bpm} BPM · ${r.phrases.length} phrases of ${r.phrases[0]?.candidates.length ?? 0} imagined futures · ${kept}/${r.edits.length} edits kept · whole-piece score ${r.draft_score.score.toFixed(3)} → ${r.final_score.score.toFixed(3)} (surprise ${r.final_score.mean_surprise.toFixed(2)} vs target ${r.target_surprise.toFixed(2)}, mood fit ${r.final_score.mood_fit.toFixed(2)}) · ${r.seconds.toFixed(1)} s`;
  if (r.final_score && r.final_score.form !== undefined) {
    const f = r.final_score;
    $("review").textContent += ` · form: plan fit ${(f.plan_fit ?? 0).toFixed(2)}, contrast ${f.contrast.toFixed(2)}, climax ${f.climax.toFixed(1)}, ending ${f.ending.toFixed(2)}, return ${f.return.toFixed(0)} (fidelity ${f.return_fidelity.toFixed(2)})`;
  }
  if (mode !== "composing") mode = "scrub";
}
function secondsToStep(t) {
  const bpm = composition?.tempo_bpm ?? 100;
  return (t * bpm * 4) / 60;
}
function eventAt(step) {
  // the events of the final piece at this step: the last onset at or before it, and the
  // share of its slot (events with the same onset are settled one after another)
  const starts = composition.starts;
  let k = -1;
  for (let i = 0; i < starts.length; i++) if (starts[i] <= step) k = i;
  if (k < 0) return null;
  let first = k;
  while (first > 0 && starts[first - 1] === starts[k]) first--;
  let last = k;
  while (last + 1 < starts.length && starts[last + 1] === starts[k]) last++;
  const next = last + 1 < starts.length ? starts[last + 1] : composition.bars * 16;
  const span = Math.max(1e-6, next - starts[k]),
    fraction = Math.min(0.999, (step - starts[k]) / span),
    count = last - first + 1,
    which = Math.min(count - 1, Math.floor(fraction * count)),
    inner = fraction * count - which;
  return { event: first + which, fraction: inner, next: last + 1 };
}
function playbackTime() {
  if (silentClock) return (performance.now() - silentClock) / 1000;
  const a = $("audio");
  return a.src && !a.paused ? a.currentTime : null;
}
function startPlayback() {
  liveQueue.length = 0;
  if (mode === "composing" && !$("compose").disabled) mode = "scrub";
}
$("audio").addEventListener("play", startPlayback);
$("play-silent").onclick = () => {
  startPlayback();
  if (silentClock) {
    silentClock = null;
    $("play-silent").textContent = "▶ Silent";
  } else {
    if ($("audio").src) $("audio").pause();
    silentClock = performance.now();
    $("play-silent").textContent = "■ Stop";
  }
};

// ---- drawing ---------------------------------------------------------------------------------
function drawRoll() {
  const [c, w, h] = fit($("roll"));
  const events = mode === "composing" ? liveRoll.events : composition?.events ?? [];
  const starts = mode === "composing" ? liveRoll.starts : composition?.starts ?? [];
  const total = mode === "composing" ? liveRoll.total : (composition?.bars ?? 16) * 16;
  const x = (step) => (step / total) * w,
    y = (pitch) => h - ((pitch + 4) / 80) * h;
  for (let bar = 0; bar * 16 < total; bar++) {
    c.fillStyle = bar % 4 === 0 ? "#1c2f38" : "#14252c";
    c.fillRect(x(bar * 16), 0, 1, h);
  }
  const t = playbackTime();
  const step = t !== null && composition ? secondsToStep(t) : null;
  const here = mode === "composing" ? liveRoll.events.length - 1 : step !== null ? eventAt(step)?.event : current.event;
  events.forEach((e, k) => {
    const [pitch, duration, , fam] = e;
    c.fillStyle = FAMILY_COLORS[fam] + (k === here ? "ff" : "b0");
    c.fillRect(x(starts[k]), y(pitch) - 2, Math.max(2, x(DURATIONS[duration]) - 1), 3);
  });
  if (mode === "composing" && liveRoll.candidates.length) {
    for (const cand of liveRoll.candidates) {
      let s = liveRoll.phraseStart;
      c.fillStyle = cand.winner ? "#f4c86acc" : "#7f949766";
      for (const [pitch, duration, delta] of cand.tokens) {
        s += DELTAS[delta];
        c.fillRect(x(s), y(pitch) - 1, Math.max(2, x(DURATIONS[duration]) - 1), 2);
      }
    }
  }
  if (step !== null) {
    c.fillStyle = "#e4efed";
    c.fillRect(x(step), 0, 1.5, h);
  }
  const p = mode === "composing" ? Math.max(0, ...liveRoll.starts, liveRoll.phraseStart) : null;
  if (p !== null) {
    c.fillStyle = "#f4c86a";
    c.fillRect(x(p), 0, 1.5, h);
  }
}
const LAG_BARS = [0, 1, 2, 3, 4, 6, 8, 12, 16, 32];
// The plan of the piece bar by bar: a block per bar whose height is the planned density and
// whose brightness is the planned loudness, an arc from every bar that returns to the bar it
// returns to, and, once the piece is heard back, a mark on every bar whose realised density,
// loudness, register or texture differs from the plan.
function drawForm() {
  const [c, w, h] = fit($("form"));
  const plan = mode === "composing" ? liveRoll.plan : composition?.plan;
  const profiles = mode === "composing" ? null : composition?.profiles;
  const bars = mode === "composing" ? liveRoll.total / 16 : composition?.bars ?? 16;
  c.fillStyle = "#657f80";
  c.font = "9px ui-monospace, monospace";
  if (!plan) {
    c.fillText(composition || mode === "composing" ? "no plan: this brain writes bar by bar" : "the plan of the piece appears here", 6, 11);
    return;
  }
  const rows = plan.plan,
    bw = w / bars;
  rows.forEach((row, b) => {
    const density = row[0] / 5,
      loud = 0.35 + 0.65 * (row[3] / 3);
    c.fillStyle = `rgba(107,166,255,${loud.toFixed(2)})`;
    const bh = 6 + density * (h - 22);
    c.fillRect(b * bw + 1, h - bh, Math.max(1, bw - 2), bh);
    if (profiles && profiles[b]) {
      const real = profiles[b];
      const off = [0, 3, 1, 4].filter((k) => real[k] !== row[k]).length;
      if (off) {
        c.fillStyle = off >= 3 ? "#ff7b7b" : "#ffb17a";
        c.fillRect(b * bw + 1, h - 3, Math.max(1, bw - 2), 3);
      }
    }
    const lag = LAG_BARS[row[7]];
    if (lag && b - lag >= 0) {
      const x0 = (b - lag + 0.5) * bw,
        x1 = (b + 0.5) * bw;
      c.strokeStyle = "#89e6bfaa";
      c.lineWidth = 1;
      c.beginPath();
      c.moveTo(x0, 14);
      c.quadraticCurveTo((x0 + x1) / 2, 14 - Math.min(12, lag * 3), x1, 14);
      c.stroke();
    }
  });
  c.fillStyle = "#657f80";
  c.fillText("plan · density, loudness, returns", 6, 11);
}
function drawTimeline() {
  const [c, w, h] = fit($("timeline"));
  const total = mode === "composing" ? liveRoll.total : (composition?.bars ?? 16) * 16;
  const x = (step) => (step / total) * w;
  const surprise = mode === "composing" ? liveRoll.surprise : composition?.listen_surprise ?? [];
  const starts = mode === "composing" ? liveRoll.starts : composition?.starts ?? [];
  const peak = Math.max(1, ...surprise);
  c.fillStyle = "#89e6bf99";
  surprise.forEach((s, k) => c.fillRect(x(starts[k]), h - (s / peak) * h * 0.8, 2, (s / peak) * h * 0.8));
  const valences = mode === "composing" ? liveRoll.valences : composition?.phrases.map((p) => ({ step: p.from_step, valence: p.valence })) ?? [];
  const vpeak = Math.max(0.1, ...valences.map((v) => Math.abs(v.valence)));
  c.strokeStyle = "#f691ad";
  c.lineWidth = 1.5;
  c.beginPath();
  valences.forEach((v, i) => {
    const px = x(v.step),
      py = h / 2 - (v.valence / vpeak) * h * 0.4;
    i ? c.lineTo(px, py) : c.moveTo(px, py);
  });
  c.stroke();
  c.fillStyle = "#657f80";
  c.font = "9px ui-monospace, monospace";
  c.fillText("surprise per event · valence per phrase", 6, 11);
}
function currentFrameKey() {
  if (mode === "composing") {
    if (!liveQueue.length) return null;
    return liveQueue[0];
  }
  if (!composition) return null;
  if (!$("follow").checked && current.key) return { key: current.key, event: current.event, phase: current.phase };
  const t = playbackTime();
  if (t === null) return current.key ? { key: current.key, event: current.event, phase: current.phase } : { key: "listen/0", event: 0, phase: "listening back", fraction: 0 };
  const at = eventAt(secondsToStep(t));
  if (!at) return { key: "listen/0", event: 0, phase: "listening back", fraction: 0 };
  for (let k = at.event + 1; k < Math.min(composition.events.length, at.event + 4); k++) getFrame(`listen/${k}`);
  return { key: `listen/${at.event}`, event: at.event, phase: "listening back", fraction: at.fraction };
}
function drawBrain(dt, now) {
  const [c, w, h] = fit($("brain"));
  if (!info) {
    network.draw(now);
    return;
  }
  const want = currentFrameKey();
  let frame = want ? getFrame(want.key) : null;
  if (want && frame) {
    if (mode === "composing") {
      liveIteration += dt * 40;
      if (liveIteration >= frame.iterations) {
        liveIteration = 0;
        liveQueue.shift();
        if (liveQueue.length) getFrame(liveQueue[0].key);
      }
      current = { key: want.key, frame, iteration: Math.min(frame.iterations - 1, Math.floor(liveIteration)), event: want.event, phase: want.phase };
      if (liveQueue.length > 6) liveQueue.splice(0, liveQueue.length - 6); // never fall behind the composer
    } else if ($("follow").checked && want.fraction !== undefined) {
      current = { key: want.key, frame, iteration: Math.min(frame.iterations - 1, Math.floor(want.fraction * frame.iterations)), event: want.event, phase: want.phase };
      $("scrub").max = frame.iterations - 1;
      $("scrub").value = current.iteration;
    } else {
      current = { ...current, key: want.key, frame, event: want.event, phase: want.phase };
      $("scrub").max = frame.iterations - 1;
    }
  }
  frame = current.frame;
  const signal = $("signal").value,
    mapped = network.n === info.topology.neurons;
  if (frame && mapped) {
    const i = Math.min(frame.iterations - 1, current.iteration);
    if (shown.frame !== frame || shown.iteration !== i || shown.signal !== signal) {
      present(frame, i, signal, shown.frame === frame && i === shown.iteration + 1);
      shown = { key: current.key, frame, iteration: i, signal };
    } else network.draw(now);
  } else network.draw(now);
  const hit = hover && frame && mapped ? network.inspect(hover[2], hover[3]) : null;
  if (hit) {
    const best = hit.neuron,
      name = hit.region;
    const lines = [
      `${LABELS[name] || name} · neuron ${best}`,
      `activation ${channelOf(frame, current.iteration, 0)[best].toFixed(4)} · repair ${channelOf(frame, current.iteration, 1)[best].toExponential(2)}`,
      `equation mismatch ${channelOf(frame, current.iteration, 2)[best].toExponential(2)}`,
    ];
    c.font = "10px system-ui";
    const tw = Math.min(w - 12, Math.max(...lines.map((t) => c.measureText(t).width)) + 16),
      x = Math.min(w - tw - 6, Math.max(6, hover[0] + 10)),
      y = Math.min(h - 56, Math.max(4, hover[1] + 12));
    c.fillStyle = "#081419ee";
    c.fillRect(x, y, tw, 52);
    c.fillStyle = "#e4efed";
    lines.forEach((line, i) => c.fillText(line, x + 8, y + 14 + i * 15, tw - 16));
  }
  if (frame) {
    const surprise = mode === "composing" ? null : composition?.listen_surprise?.[current.event];
    const phrase = composition?.phrases.find((p) => current.event >= p.first_event && current.event < p.first_event + (p.surprise?.length ?? 0));
    $("equilibrium").textContent = `${current.phase} · event ${current.event + 1} · iteration ${current.iteration + 1}/${frame.iterations} · whole-brain mismatch ${frame.whole[current.iteration].toExponential(2)}${surprise !== undefined && surprise !== null ? ` · surprise ${surprise.toFixed(2)}` : ""}${phrase ? ` · phrase valence ${phrase.valence >= 0 ? "+" : ""}${phrase.valence.toFixed(3)}` : ""}`;
  }
  drawWaves(frame);
}
function drawWaves(frame) {
  const [c, w, h] = fit($("waves"));
  if (!frame) return;
  const names = Object.keys(frame.regionSeries),
    cols = 4,
    rows = Math.ceil(names.length / cols),
    cw = w / cols,
    ch = h / rows;
  names.forEach((name, k) => {
    const s = frame.regionSeries[name],
      x0 = (k % cols) * cw,
      y0 = Math.floor(k / cols) * ch;
    c.fillStyle = `rgba(${regionColorOf(name).join(",")},.9)`;
    c.font = "9px system-ui";
    c.fillText(LABELS[name]?.split(" ·")[0] || name, x0 + 4, y0 + 10);
    const plot = (series, color, peak) => {
      c.strokeStyle = color;
      c.lineWidth = 1;
      c.beginPath();
      series.forEach((v, i) => {
        const px = x0 + 4 + (i / (series.length - 1)) * (cw - 8),
          py = y0 + ch - 4 - (Math.abs(v) / peak) * (ch - 16);
        i ? c.lineTo(px, py) : c.moveTo(px, py);
      });
      c.stroke();
    };
    plot(s.mean, "#89e6bf", Math.max(1e-6, ...s.mean.map(Math.abs)));
    plot(s.mismatch, "#baa7fb", Math.max(1e-9, ...s.mismatch));
    const i = current.iteration;
    c.fillStyle = "#e4efed88";
    c.fillRect(x0 + 4 + (i / (frame.iterations - 1)) * (cw - 8), y0 + 12, 1, ch - 16);
  });
}
function animate(t) {
  const dt = Math.min(0.05, (t - lastTick) / 1000 || 0.016);
  lastTick = t;
  drawForm();
  drawRoll();
  drawTimeline();
  drawBrain(dt, t);
  requestAnimationFrame(animate);
}
window.__musicianRenderer = network;
window.__musicianDebug = () => ({
  graph: network.snapshot(),
  mode,
  current: { key: current.key, iteration: current.iteration, event: current.event },
  cached: [...frameCache.keys()],
  queue: liveQueue.length,
  composition: composition?.id,
  glError: network.gl?.getError(),
});
loadBrain();
poll();
pollEvents();
setInterval(poll, 600);
setInterval(pollEvents, 150);
requestAnimationFrame(animate);
