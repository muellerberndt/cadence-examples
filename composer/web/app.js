import { BrainScan, layoutAtlas } from "./brain_scan.js";
const $ = (id) => document.getElementById(id),
  labels = {
    auditory_history: "Auditory history",
    harmony: "Harmonic context",
    rhythm: "Rhythmic context",
    phrase_memory: "Phrase context",
    note_intention: "Note intention",
    motif_cue: "Motif position cue",
    motif_recall: "Learned motif recall",
    expectation_cue: "Musical context cue",
    harmonic_expectation: "Harmonic expectation",
    rhythmic_expectation: "Rhythmic expectation",
  };
let report = null,
  trace = null,
  frame = 0,
  last = 0,
  credit = 0,
  replaying = true,
  currentId = null,
  polling = false;
// What each population is to the composing circuit: the scan colours regions by role.
const roles = {
  auditory_history: "sensory",
  motif_cue: "sensory",
  expectation_cue: "sensory",
  harmony: "association",
  rhythm: "association",
  phrase_memory: "memory",
  motif_recall: "memory",
  harmonic_expectation: "memory",
  rhythmic_expectation: "memory",
  note_intention: "motor",
};
let displayScales = null,
  hover = null;
$("brain").addEventListener("pointermove", (e) => {
  const r = $("brain").getBoundingClientRect();
  hover = [e.clientX - r.left, e.clientY - r.top, e.clientX, e.clientY];
});
$("brain").addEventListener("pointerleave", () => (hover = null));
const EMPTY_ATLAS = {
  n: 0,
  synapses: 0,
  regions: [],
  region: new Uint16Array(0),
  positions: new Float32Array(0),
  pre: new Uint32Array(0),
  post: new Uint32Array(0),
  weight: new Float32Array(0),
  palette: {},
};
const network = new BrainScan($("brain-network"), EMPTY_ATLAS, {
  labels: $("brain-labels"),
  interaction: $("brain"),
});
const graphCache = new Map();
let canvasMap = null,
  loaded = null, // the graph the scan shows: {id, n, pre, post}
  shown = {}, // the frame the scan shows: {trace, frame, signal}
  glow = null; // the heat of the frame shown, for the afterglow of the next
let graphLoading = null,
  pendingTrace = null,
  live = { busy: false, events: [], trials: [] },
  eventCursor = 0;
let eventPolling = false,
  playMode = "final",
  acknowledged = 0; // when the server last accepted a composition: earlier polls are stale
const pitchName = (p) =>
  ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"][p % 12] +
  (Math.floor(p / 12) - 1);
$("fit-brain").onclick = () => network.fit();
$("expand-brain").onclick = () =>
  document.fullscreenElement
    ? document.exitFullscreen()
    : $("brain").closest(".brain-panel").requestFullscreen?.();
function unpack(t) {
  // Compositions recorded before the neuron/synapse keys carry owner/seam keys.
  t.neuron_ids ??= t.owner_ids;
  if (t.topology) {
    t.topology.neurons ??= t.topology.owners;
    t.topology.synapses ??= t.topology.seams;
    t.topology.neuron_ids ??= t.topology.owner_ids;
  }
  for (const r of Object.values(t.regions || {})) r.neurons ??= r.owners;
  for (const r of Object.values(t.topology?.regions || {})) r.neurons ??= r.owners;
  if (t.encoded_frames && !t.frames) {
    const bytes = Uint8Array.from(atob(t.encoded_frames.data), (c) =>
      c.charCodeAt(0),
    );
    const values = new Float32Array(bytes.buffer),
      [frames, , neurons] = t.encoded_frames.shape;
    t.frames = Array.from({ length: frames }, (_, f) =>
      Object.fromEntries(
        ["activation", "repair", "mismatch"].map((name, k) => [
          name,
          values.subarray((f * 3 + k) * neurons, (f * 3 + k + 1) * neurons),
        ]),
      ),
    );
  }
  return t;
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
function sameWiring(a, b) {
  if (!a || a.pre.length !== b.pre.length) return false;
  for (let e = 0; e < a.pre.length; e++)
    if (a.pre[e] !== b.pre[e] || a.post[e] !== b.post[e]) return false;
  return true;
}
// The whole brain laid out for the scan: regions placed by their connections, neurons by
// their synapses. A graph with the same wiring and new weights keeps its layout.
function mount(g, data) {
  const { pre, post, weight } = split(data);
  if (loaded && loaded.n === g.neurons && sameWiring(loaded, { pre, post })) {
    network.setWeights(weight);
  } else {
    const groups = new Array(g.neurons).fill("other");
    for (const [name, r] of Object.entries(g.regions))
      groups.fill(name, r.start, r.start + r.neurons);
    network.setAtlas(
      layoutAtlas({ n: g.neurons, pre, post, weight, groups, labels, roles, seed: 0 }),
    );
    network.fit();
  }
  loaded = { id: g.id, n: g.neurons, pre, post };
  canvasMap = null;
  shown = {};
}
async function loadGraph(t) {
  const g = t.topology;
  if (!g || (loaded?.id === g.id && !t.edge_changes_url) || graphLoading === g.id)
    return;
  graphLoading = g.id;
  try {
    let data = graphCache.get(g.id);
    if (!data) {
      const response = await fetch(g.url);
      if (!response.ok) throw Error("Topology unavailable");
      data = new Float32Array(await response.arrayBuffer());
      if (data.length !== g.synapses * 3) throw Error("Incomplete topology");
      graphCache.set(g.id, data);
      if (graphCache.size > 3)
        graphCache.delete(graphCache.keys().next().value);
    }
    if (trace?.topology?.id === g.id) {
      if (loaded?.id !== g.id) mount(g, data);
      if (t.edge_changes_url) {
        const delta = new Float32Array(
          await (await fetch(t.edge_changes_url)).arrayBuffer(),
        );
        if (trace === t) {
          // learned change per neuron: the weight changes of its synapses, scaled to the largest
          const moved = new Float32Array(g.neurons);
          let peak = 1e-9;
          for (let e = 0; e < delta.length; e++) {
            const a = data[e * 3],
              b = data[e * 3 + 1],
              d = Math.abs(delta[e]);
            moved[a] += d;
            moved[b] += d;
          }
          for (const x of moved) if (x > peak) peak = x;
          t.plasticity = moved.map((x) => x / peak);
          shown = {};
        }
      }
    }
  } catch (err) {
    $("equilibrium").textContent = err.message;
  } finally {
    if (graphLoading === g.id) graphLoading = null;
  }
}
// The display scale of each recorded frame: the largest repair so far, easing by a tenth per
// frame as the settling calms so later, smaller repairs stay visible; the mismatch and the
// activation alike (the scan expects activations within the unit range).
function measure(frames) {
  const scale = new Float32Array(frames.length),
    mismatch = new Float32Array(frames.length),
    activation = new Float32Array(frames.length);
  let s = 1e-6,
    m = 1e-6,
    a = 1e-6;
  frames.forEach((f, t) => {
    let peak = 0,
      worst = 0,
      loud = 0;
    for (let i = 0; i < f.repair.length; i++) {
      const r = Math.abs(f.repair[i]),
        e = Math.abs(f.mismatch[i]),
        v = Math.abs(f.activation[i]);
      if (r > peak) peak = r;
      if (e > worst) worst = e;
      if (v > loud) loud = v;
    }
    s = Math.max(peak, s * 0.9);
    m = Math.max(worst, m * 0.9);
    a = Math.max(loud, a * 0.9);
    scale[t] = s;
    mismatch[t] = m;
    activation[t] = a;
  });
  return { scale, mismatch, activation };
}
// One recorded frame into the scan: the activation, scaled to the largest so far, travels as
// messages along the synapses; the repair is the heat (with afterglow while the replay
// advances); the chosen signal is the brightness.
function present(f, t, kind, plastic, advancing) {
  const n = f.activation.length,
    message = new Float32Array(n),
    level = new Float32Array(n),
    heat = new Float32Array(n);
  const s = Math.max(1e-9, displayScales.scale[t]),
    m = Math.max(1e-9, displayScales.mismatch[t]),
    a = Math.max(1e-9, displayScales.activation[t]);
  const prior = advancing && glow && glow.length === n ? glow : null,
    learned = trace.plasticity;
  for (let i = 0; i < n; i++) {
    const change = Math.min(1, Math.abs(f.repair[i]) / s);
    message[i] = f.activation[i] / a;
    heat[i] = plastic
      ? (learned?.[i] ?? 0)
      : Math.max(change, prior ? prior[i] * 0.86 : 0);
    level[i] = plastic
      ? (learned?.[i] ?? 0)
      : kind === "repair"
        ? f.repair[i] / s
        : kind === "mismatch"
          ? f.mismatch[i] / m
          : message[i];
  }
  glow = plastic ? null : heat;
  network.show(message, heat, { level });
}
const fmt = (n) => Number(n).toLocaleString();
function fit(canvas) {
  const r = canvas.getBoundingClientRect(),
    d = Math.min(devicePixelRatio, 2);
  if (
    canvas.width !== Math.round(r.width * d) ||
    canvas.height !== Math.round(r.height * d)
  ) {
    canvas.width = Math.round(r.width * d);
    canvas.height = Math.round(r.height * d);
  }
  const c = canvas.getContext("2d");
  c.setTransform(d, 0, 0, d, 0, 0);
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
function useTrace(t) {
  if (!t) return;
  trace = unpack(t);
  loadGraph(trace);
  displayScales = measure(trace.frames);
  glow = null;
  frame = credit = 0;
  replaying = true;
  $("modulators").replaceChildren();
  for (const [name, m] of Object.entries(t.modulators)) {
    const el = text($("modulators"), "span", name);
    text(el, "b", Number(m.value).toFixed(3));
    el.title = m.effect;
  }
}
function useReport(r) {
  if (!currentId) {
    $("prompt").value = r.brief.prompt;
    $("seed").value = r.seed;
  }
  report = r;
  currentId = r.id;
  if (r.traces[0]?.topology) useTrace(r.traces[0]);
  $("notes").textContent = `${r.events.length} notes · ${r.brief.bpm} BPM`;
  playVersion("final", false);
  $("midi").href = r.files.midi;
  $("midi").download = "cadence.mid";
  $("wav").href = r.files.audio;
  $("wav").download = "cadence.wav";
  $("review").textContent =
    `${r.revision.accepted ? "Revised" : "Kept"} phrase at bar ${r.revision.bar + 1}. Explicit score ${r.before.score.toFixed(2)} → ${r.after.score.toFixed(2)}. ${r.phrases.length} phrases, ${r.phrases[0].candidates.length} imagined alternatives each. ${r.seconds.toFixed(1)} s computation.`;
  $("alternatives").replaceChildren();
  r.phrases.forEach((p, i) => {
    const el = text(
      $("alternatives"),
      "div",
      `Bars ${p.bar + 1}–${Math.min(p.bar + 4, r.brief.bars)}`,
      "candidate",
    );
    const button = text(el, "button", "Inspect");
    button.onclick = () => {
      const actual = r.rehearsals?.find((t) => t.bar === p.bar);
      useTrace(actual?.trace || r.traces[i]);
    };
    p.candidates.forEach((c, j) => {
      const m = document.createElement("meter");
      m.min = -2;
      m.max = 6;
      m.value = c.critique.score;
      m.title = `Candidate ${j + 1}: ${c.critique.score.toFixed(3)}${j === p.winner ? " · selected" : ""}`;
      el.append(m);
    });
  });
}
function playVersion(which, start = true) {
  if (!report) return;
  playMode = which;
  $("audio").src =
    which === "draft" ? report.files.draft_audio : report.files.audio;
  $("draft").classList.toggle("selected", which === "draft");
  $("final").classList.toggle("selected", which === "final");
  if (start)
    $("audio")
      .play()
      .catch(() => {});
}
$("draft").onclick = () => playVersion("draft");
$("final").onclick = () => playVersion("final");
$("replay").onclick = () => {
  pendingTrace = null;
  frame = credit = 0;
  replaying = true;
};
$("release").onclick = async () => {
  try {
    $("audio").pause();
    pendingTrace = null;
    const released = trace?.release || (await api("/api/release", {})).trace;
    useTrace(released);
    $("thought").textContent = "Input released in an isolated state";
  } catch (err) {
    $("thought").textContent = err.message;
  }
};
for (const b of document.querySelectorAll("[data-prompt]"))
  b.onclick = () => {
    $("prompt").value = b.dataset.prompt;
  };
async function api(path, data) {
  const r = await fetch(path, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(data),
  });
  const value = await r.json();
  if (!r.ok) throw Error(value.error || r.statusText);
  return value;
}
$("brief").onsubmit = async (e) => {
  e.preventDefault();
  $("compose").disabled = true;
  $("audio").pause();
  try {
    await api("/api/compose", {
      prompt: $("prompt").value,
      seed: Number($("seed").value),
    });
    acknowledged = performance.now();
    await poll();
  } catch (err) {
    $("phase").textContent = err.message;
    $("compose").disabled = false;
  }
};
async function feedback(rating) {
  $("like").disabled = $("dislike").disabled = true;
  $("feedback-status").textContent =
    "Updating musical relationships and checking retained knowledge…";
  try {
    const r = await api("/api/feedback", { rating });
    pendingTrace = null;
    useTrace(r.trace);
    $("signal").value = "plasticity";
    $("feedback-status").textContent =
      `${r.retained ? "Preference retained" : "Update rolled back"} · valence ${r.valence.toFixed(1)} · validation loss ${r.before.mean_nll.toFixed(3)} → ${r.after.mean_nll.toFixed(3)}`;
  } catch (err) {
    $("feedback-status").textContent = err.message;
  } finally {
    $("like").disabled = $("dislike").disabled = false;
  }
}
$("like").onclick = () => feedback(1);
$("dislike").onclick = () => feedback(-1);
let lastProgress = null;
async function poll() {
  if (polling) return;
  polling = true;
  const started = performance.now();
  try {
    const r = await (
      await fetch(
        `/api/state?result=${currentId ?? ""}&trace=${lastProgress ?? ""}`,
      )
    ).json();
    live.busy = r.busy;
    // a poll sent before the server accepted a composition cannot re-enable the button
    $("compose").disabled = r.busy || started < acknowledged;
    $("phase").textContent = r.error || r.progress.stage;
    $("neurons").textContent =
      `${fmt(r.brain.neurons)} neurons\n${fmt(r.brain.directed_synapses)} synapses`;
    $("regions").replaceChildren();
    for (const [name, count] of Object.entries(r.brain.regions)) {
      const el = text($("regions"), "div", labels[name] || name, "region");
      text(el, "b", fmt(count));
    }
    if (r.training)
      $("training").textContent =
        `Checkpoint trained with ${fmt(r.training.updates * r.training.batch)} sampled events from the classical MIDI split. ${fmt(r.brain.trainable_parameters)} trainable parameters. Held-out validation loss: ${r.training.best_validation_nll.toFixed(3)} nats. The receipt records data and source hashes.`;
    if (r.result && r.result.id !== currentId) useReport(r.result);
    if (r.trace && r.trace_revision !== lastProgress) {
      lastProgress = r.trace_revision;
      if (trace && replaying) pendingTrace = r.trace;
      else useTrace(r.trace);
    }
  } catch (err) {
    $("phase").textContent = "Connection lost · retrying";
  } finally {
    polling = false;
  }
}
async function pollEvents() {
  if (eventPolling) return;
  eventPolling = true;
  try {
    const data = await (await fetch(`/api/events?after=${eventCursor}`)).json();
    for (const e of data.events) {
      eventCursor = e.sequence;
      if (e.brief) {
        live.brief = e.brief;
        live.trials = [];
        live.revision = null;
      }
      if (e.committed) live.events = e.committed;
      if (e.trials) live.trials = e.trials;
      if (e.stage === "Phrase committed" || e.stage === "Ready")
        live.trials = [];
      if (e.revision) live.revision = e.revision;
      if (e.candidate) live.candidate = e.candidate;
      $("thought").textContent = e.stage;
      $("trial-info").textContent = e.candidate
        ? `Candidate ${e.candidate}/${e.candidates} · bar ${e.bar + 1}`
        : e.winner
          ? `Selected candidate ${e.winner}`
          : e.bar !== undefined
            ? `Bar ${e.bar + 1}`
            : "";
      if (e.note_options) {
        $("note-votes").replaceChildren();
        for (const n of e.note_options) {
          const chip = text($("note-votes"), "span", pitchName(n.pitch));
          chip.title = `Neural logit ${n.activation.toFixed(3)} before musical priors`;
        }
      }
    }
  } catch (err) {
    $("thought").textContent = "Reconnecting to live decisions…";
  } finally {
    eventPolling = false;
  }
}
function drawScore() {
  const [c, w, h] = fit($("score"));
  const brief = live.busy && live.brief ? live.brief : report?.brief;
  if (!brief) {
    c.fillStyle = "#8ea7a6";
    c.font = "12px system-ui";
    c.fillText("Your composition will appear here.", 20, h / 2);
    return;
  }
  const end = brief.bars * 16;
  for (let bar = 0; bar < brief.bars; bar++) {
    const x = (bar / brief.bars) * w;
    c.strokeStyle = bar % 4 === 0 ? "#345052" : "#1b3038";
    c.beginPath();
    c.moveTo(x, 0);
    c.lineTo(x, h);
    c.stroke();
    if (bar % 4 === 0) {
      c.fillStyle = "#73928f";
      c.font = "9px ui-monospace, monospace";
      c.fillText(String(bar + 1), x + 3, 12);
    }
  }
  const notes = (events, color, alpha = 1) => {
    c.globalAlpha = alpha;
    c.fillStyle = color;
    for (const e of events) {
      const x = (e.step / end) * w,
        y = 22 + ((90 - e.pitch) / 46) * (h - 32);
      c.fillRect(x, y, Math.max(2, (e.duration / end) * w - 1), 3);
    }
    c.globalAlpha = 1;
  };
  if (live.busy) {
    notes(live.events, "#b1f0cd");
    for (let i = 0; i < live.trials.length; i++)
      notes(
        live.trials[i].notes,
        i === live.candidate - 1 ? "#ffc875" : "#ba9df8",
        i === live.candidate - 1 ? 0.85 : 0.17,
      );
  } else if (report) {
    if (report.revision.accepted) {
      c.fillStyle = "#b7a1fb14";
      c.fillRect(
        (report.revision.bar / brief.bars) * w,
        0,
        (4 / brief.bars) * w,
        h,
      );
    }
    notes(playMode === "draft" ? report.draft : report.events, "#b1f0cd");
  }
  if (!$("audio").paused) {
    c.strokeStyle = "#f2e1bb";
    const x = ($("audio").currentTime / 60) * w;
    c.beginPath();
    c.moveTo(x, 0);
    c.lineTo(x, h);
    c.stroke();
  }
}
function drawCanvasMap(c, w, h) {
  // Without WebGL2 the scan draws the neurons; every synapse is rasterised here once per view.
  if (!loaded || network.n !== loaded.n) return;
  const d = Math.min(devicePixelRatio, 2);
  const view = `${loaded.id}:${w}:${h}:${network.camera.x}:${network.camera.y}:${network.camera.zoom}:${d}`;
  if (canvasMap?.view !== view) {
    const bitmap = canvasMap?.canvas ?? document.createElement("canvas");
    bitmap.width = Math.round(w * d);
    bitmap.height = Math.round(h * d);
    const base = bitmap.getContext("2d");
    base.setTransform(d, 0, 0, d, 0, 0);
    const at = network.screenAll(),
      paths = new Map(),
      { pre, post } = loaded;
    for (let e = 0; e < pre.length; e++) {
      const k = network.atlas.region[pre[e]];
      if (!paths.has(k)) paths.set(k, new Path2D());
      const path = paths.get(k);
      path.moveTo(at[pre[e] * 2], at[pre[e] * 2 + 1]);
      path.lineTo(at[post[e] * 2], at[post[e] * 2 + 1]);
    }
    base.lineWidth = 0.4;
    for (const [k, path] of paths) {
      base.strokeStyle = `rgba(${network.atlas.regions[k].color.join(",")},.13)`;
      base.stroke(path);
    }
    canvasMap = { view, canvas: bitmap, synapses: pre.length };
  }
  c.drawImage(canvasMap.canvas, 0, 0, w, h);
}
function drawBrain(dt, now) {
  if (pendingTrace && !replaying) {
    const next = pendingTrace;
    pendingTrace = null;
    useTrace(next);
  }
  const [c, w, h] = fit($("brain"));
  if (!trace) {
    c.fillStyle = "#8ea7a6";
    c.font = "12px system-ui";
    c.fillText("Compose to inspect a recorded settling.", 20, h / 2);
    fit($("waves"));
    network.draw(now);
    return;
  }
  if (replaying) {
    credit += dt * (live.busy || !$("audio").paused ? 45 : 12);
    frame = Math.min(trace.frames.length - 1, Math.floor(credit));
    if (frame === trace.frames.length - 1) replaying = false;
  }
  const f = trace.frames[frame],
    signal = $("signal").value,
    plastic = signal === "plasticity",
    kind = plastic ? "activation" : signal;
  const mapped = network.n === f.activation.length;
  if (mapped && (shown.trace !== trace || shown.frame !== frame || shown.signal !== signal)) {
    present(f, frame, kind, plastic, shown.trace === trace && frame === shown.frame + 1);
    shown = { trace, frame, signal };
  } else network.draw(now);
  if (mapped && !network.enabled) drawCanvasMap(c, w, h);
  const hit = hover && mapped ? network.inspect(hover[2], hover[3]) : null;
  if (hit) {
    const best = hit.neuron,
      name = hit.region,
      offset = best - (trace.regions[name]?.start ?? 0);
    const note =
      (name === "note_intention" || name === "motif_recall") && offset < 61
        ? " · " + pitchName(36 + offset)
        : "";
    const lines = [
      `${labels[name] || name} · neuron ${trace.neuron_ids[best]}${note}`,
      `Activity ${f.activation[best].toFixed(4)} · repair ${f.repair[best].toExponential(2)}`,
      `Equation mismatch ${f.mismatch[best].toExponential(2)}`,
    ];
    c.font = "10px system-ui";
    const tw = Math.min(
        w - 12,
        Math.max(...lines.map((t) => c.measureText(t).width)) + 16,
      ),
      x = Math.min(w - tw - 6, Math.max(6, hover[0] + 10)),
      y = Math.min(h - 56, Math.max(4, hover[1] + 12));
    c.fillStyle = "#081419ee";
    c.fillRect(x, y, tw, 52);
    c.fillStyle = "#e4efed";
    lines.forEach((line, i) =>
      c.fillText(line, x + 8, y + 14 + i * 15, tw - 16),
    );
  }
  $("equilibrium").textContent =
    `${trace.released ? "Input-release probe" : (trace.origin || "Recorded settling") + (replaying ? " · replaying" : trace.equation_error.at(-1) <= 1e-5 ? " · settled" : " · budget reached")} · step ${trace.iterations?.[frame] ?? frame}/${trace.steps} · whole-brain equation error ${trace.equation_error[frame].toExponential(2)}`;
  drawWaves();
}
function drawWaves() {
  const [c, w, h] = fit($("waves"));
  const names = Object.keys(trace.regions);
  names.forEach((name, k) => {
    const y = 12 + (k * (h - 15)) / names.length,
      rh = (h - 15) / names.length,
      series = trace.populations.map((p) => p[name]);
    c.fillStyle = "#8da6a3";
    c.font = "9px ui-monospace, monospace";
    c.fillText(labels[name] || name, 0, y + 4, 105);
    const peak = Math.max(
      1e-9,
      ...series.flatMap((p) => [Math.abs(p.mean), p.mismatch]),
    );
    for (const [key, color] of [
      ["mean", "#89e6bf"],
      ["mismatch", "#baa7fb"],
    ]) {
      c.strokeStyle = color;
      c.lineWidth = 1;
      c.beginPath();
      series.slice(0, frame + 1).forEach((p, i) => {
        const x = 115 + ((w - 122) * i) / Math.max(1, series.length - 1),
          v = y + rh * 0.35 - (p[key] / peak) * rh * 0.4;
        i ? c.lineTo(x, v) : c.moveTo(x, v);
      });
      c.stroke();
    }
  });
}
function animate(t) {
  const dt = Math.min(0.05, (t - last) / 1000 || 0.016);
  last = t;
  drawScore();
  drawBrain(dt, t);
  requestAnimationFrame(animate);
}
let hearing = false;
async function hearPlayback() {
  if (hearing || live.busy || !report || $("audio").paused) return;
  hearing = true;
  try {
    const heard = await api("/api/hear", {
      seconds: $("audio").currentTime,
      version: playMode,
      id: report.id,
    });
    if (!$("audio").paused && !live.busy) {
      if (replaying) pendingTrace = heard.trace;
      else useTrace(heard.trace);
      $("thought").textContent = "Reading the playing score";
    }
  } catch (err) {
    /* A composition or preference update can supersede playback readback. */
  } finally {
    hearing = false;
  }
}
setInterval(hearPlayback, 1000);
window.__composerRenderer = network;
window.__composerDebug = () => ({
  graph: {
    ...network.snapshot(),
    allEdgesSubmitted:
      network.enabled || canvasMap?.synapses === network.edges,
  },
  canvasSynapses: canvasMap?.synapses ?? 0,
  traceNeurons: trace?.neuron_ids.length,
  frames: trace?.frames.length,
  origin: trace?.origin,
  frame,
  live,
  pending: !!pendingTrace,
  glError: network.gl?.getError(),
});
poll();
pollEvents();
setInterval(poll, 500);
setInterval(pollEvents, 160);
requestAnimationFrame(animate);
