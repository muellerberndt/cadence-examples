import { CircuitMap, regionColor } from "./circuit_map.js";
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
let displayPeaks = {},
  hover = null;
$("brain").addEventListener("pointermove", (e) => {
  const r = $("brain").getBoundingClientRect();
  hover = [e.clientX - r.left, e.clientY - r.top];
});
$("brain").addEventListener("pointerleave", () => (hover = null));
const network = new CircuitMap($("brain-network"), $("brain"));
const graphCache = new Map();
let canvasMap = null;
let graphLoading = null,
  pendingTrace = null,
  live = { busy: false, events: [], trials: [] },
  eventCursor = 0;
let eventPolling = false,
  playMode = "final",
  viewTime = 0;
const pitchName = (p) =>
  ["C", "C♯", "D", "E♭", "E", "F", "F♯", "G", "A♭", "A", "B♭", "B"][p % 12] +
  (Math.floor(p / 12) - 1);
$("fit-brain").onclick = () => network.fit();
$("expand-brain").onclick = () =>
  document.fullscreenElement
    ? document.exitFullscreen()
    : $("brain").closest(".brain-panel").requestFullscreen?.();
function unpack(t) {
  if (t.encoded_frames && !t.frames) {
    const bytes = Uint8Array.from(atob(t.encoded_frames.data), (c) =>
      c.charCodeAt(0),
    );
    const values = new Float32Array(bytes.buffer),
      [frames, , owners] = t.encoded_frames.shape;
    t.frames = Array.from({ length: frames }, (_, f) =>
      Object.fromEntries(
        ["activation", "repair", "mismatch"].map((name, k) => [
          name,
          values.subarray((f * 3 + k) * owners, (f * 3 + k + 1) * owners),
        ]),
      ),
    );
  }
  return t;
}
async function loadGraph(t) {
  const g = t.topology;
  if (!g) {
    network.graph(t.owner_ids.length, t.edges);
    return;
  }
  if ((network.id === g.id && !t.edge_changes_url) || graphLoading === g.id)
    return;
  graphLoading = g.id;
  try {
    let data = graphCache.get(g.id);
    if (!data) {
      const response = await fetch(g.url);
      if (!response.ok) throw Error("Topology unavailable");
      data = new Float32Array(await response.arrayBuffer());
      if (data.length !== g.seams * 3) throw Error("Incomplete topology");
      graphCache.set(g.id, data);
      if (graphCache.size > 3)
        graphCache.delete(graphCache.keys().next().value);
    }
    if (trace?.topology?.id === g.id) {
      network.graph(g.owners, data, g.id);
      if (t.edge_changes_url) {
        const delta = new Float32Array(
          await (await fetch(t.edge_changes_url)).arrayBuffer(),
        );
        if (trace === t) {
          network.plasticity(delta);
          t.plasticity = new Float32Array(g.owners);
          for (let e = 0; e < delta.length; e++) {
            const a = data[e * 3],
              b = data[e * 3 + 1];
            t.plasticity[a] += Math.abs(delta[e]);
            t.plasticity[b] += Math.abs(delta[e]);
          }
          const peak = Math.max(1e-9, ...t.plasticity);
          t.plasticity = t.plasticity.map((x) => x / peak);
        }
      }
    }
  } catch (err) {
    $("equilibrium").textContent = err.message;
  } finally {
    if (graphLoading === g.id) graphLoading = null;
  }
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
  displayPeaks = {};
  for (const kind of ["activation", "repair", "mismatch"]) {
    const peaks = Array(t.frames[0][kind].length).fill(0.02);
    for (const r of Object.values(t.regions)) {
      let peak = 0.02;
      for (const f of t.frames)
        for (let i = r.start; i < r.start + r.shown; i++)
          peak = Math.max(peak, Math.abs(f[kind][i]));
      for (let i = r.start; i < r.start + r.shown; i++) peaks[i] = peak;
    }
    displayPeaks[kind] = peaks;
  }
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
  try {
    const r = await (
      await fetch(
        `/api/state?result=${currentId ?? ""}&trace=${lastProgress ?? ""}`,
      )
    ).json();
    live.busy = r.busy;
    $("compose").disabled = r.busy;
    $("phase").textContent = r.error || r.progress.stage;
    $("owners").textContent =
      `${fmt(r.brain.owners)} owners\n${fmt(r.brain.directed_seams)} seams`;
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
function drawCanvasMap(c, w, h, positions, groups) {
  if (network.id !== trace.topology?.id || !network.topology) return;
  const key = `${network.id}:${w}:${h}`;
  if (canvasMap?.key !== key) {
    const paths = new Map();
    for (let e = 0; e < network.edges; e++) {
      const a = network.topology[e * 3],
        b = network.topology[e * 3 + 1];
      const role = groups[a];
      if (!paths.has(role)) paths.set(role, new Path2D());
      const path = paths.get(role);
      path.moveTo(...positions[a]);
      path.lineTo(...positions[b]);
    }
    canvasMap = {
      key,
      paths,
      canvas: document.createElement("canvas"),
      seams: 0,
    };
  }
  const d = Math.min(devicePixelRatio, 2);
  const view = `${network.camera.x}:${network.camera.y}:${network.camera.zoom}:${d}`;
  if (canvasMap.view !== view) {
    const bitmap = canvasMap.canvas;
    bitmap.width = Math.round(w * d);
    bitmap.height = Math.round(h * d);
    const base = bitmap.getContext("2d");
    base.setTransform(d, 0, 0, d, 0, 0);
    network.transformContext(base, w, h);
    base.lineWidth = 0.4 / network.camera.zoom;
    for (const [role, path] of canvasMap.paths) {
      base.strokeStyle = `rgba(${regionColor(role).join(",")},.13)`;
      base.stroke(path);
    }
    canvasMap.view = view;
    canvasMap.seams = network.edges;
  }
  c.save();
  c.setTransform(d, 0, 0, d, 0, 0);
  c.drawImage(canvasMap.canvas, 0, 0, w, h);
  c.restore();
}
function drawBrain(dt) {
  viewTime += dt;
  if (pendingTrace && !replaying) {
    const next = pendingTrace;
    pendingTrace = null;
    useTrace(next);
  }
  const [c, w, h] = fit($("brain"));
  if (!trace) {
    c.fillStyle = "#8ea7a6";
    c.font = "12px system-ui";
    c.fillText("Compose to inspect a recorded settlement.", 20, h / 2);
    fit($("waves"));
    return;
  }
  if (replaying) {
    credit += dt * (live.busy || !$("audio").paused ? 45 : 12);
    frame = Math.min(trace.frames.length - 1, Math.floor(credit));
    if (frame === trace.frames.length - 1) replaying = false;
  }
  const f = trace.frames[frame],
    plastic = $("signal").value === "plasticity",
    kind = plastic ? "activation" : $("signal").value,
    values = f[kind],
    positions = [],
    groups = [];
  c.save();
  network.transformContext(c, w, h);
  const names = Object.keys(trace.regions);
  const locations = {
    auditory_history: [0.15, 0.18],
    harmony: [0.48, 0.17],
    rhythm: [0.48, 0.81],
    phrase_memory: [0.48, 0.48],
    note_intention: [0.82, 0.61],
    motif_cue: [0.15, 0.52],
    motif_recall: [0.15, 0.82],
    expectation_cue: [0.82, 0.13],
    harmonic_expectation: [0.82, 0.37],
    rhythmic_expectation: [0.82, 0.85],
  };
  names.forEach((name) => {
    const r = trace.regions[name],
      [cx, cy] = locations[name] || [0.5, 0.5],
      x = cx * w,
      y = cy * h;
    const radius = Math.min(w * 0.115, h * (cx > 0.7 ? 0.075 : 0.13));
    const glow = c.createRadialGradient(x, y, 0, x, y, radius * 1.4);
    const role = regionColor(name);
    glow.addColorStop(0, `rgba(${role.join(",")},.09)`);
    glow.addColorStop(1, "#10202600");
    c.fillStyle = glow;
    c.fillRect(x - radius * 1.4, y - radius * 1.4, radius * 2.8, radius * 2.8);
    for (let k = 0; k < r.shown; k++) {
      const a = k * 2.39996,
        rad = radius * Math.sqrt((k + 0.5) / r.shown);
      groups[r.start + k] = name;
      positions[r.start + k] = [
        x + Math.cos(a) * rad,
        y + Math.sin(a) * rad * 0.7,
      ];
    }
    c.fillStyle = "#b4ccc7";
    c.textAlign = "center";
    c.font = "10px system-ui";
    c.fillText(
      labels[name] || name,
      x,
      y - radius * 0.78,
      Math.max(90, w * 0.3),
    );
    c.fillStyle = "#657f80";
    c.font = "9px ui-monospace, monospace";
    c.fillText(`${r.owners} owners`, x, y + radius * 0.85);
  });
  c.textAlign = "left";
  if (network.id === trace.topology?.id)
    network.draw({
      positions,
      groups,
      activation: f.activation,
      repair: f.repair,
      mismatch: f.mismatch,
      plasticity: trace.plasticity,
      scales: [
        displayPeaks.activation,
        displayPeaks.repair,
        displayPeaks.mismatch,
      ],
      channel: plastic
        ? 3
        : kind === "repair"
          ? 1
          : kind === "mismatch"
            ? 2
            : 0,
      time: viewTime,
      moving: replaying,
    });
  if (!network.enabled) drawCanvasMap(c, w, h, positions, groups);
  if (!network.enabled)
    values.forEach((v, i) => {
      const [x, y] = positions[i];
      const a = plastic
          ? (trace.plasticity?.[i] ?? 0)
          : Math.min(1, Math.abs(v) / displayPeaks[kind][i]),
        repair = Math.min(1, Math.abs(f.repair[i]) * 8);
      const color = regionColor(groups[i]).map((x) =>
        Math.round(x * (1 - a * 0.6) + 245 * a * 0.6),
      );
      c.fillStyle = `rgba(${color.join(",")},${0.22 + a * 0.78})`;
      c.shadowColor = v < 0 ? "#baa7fb" : "#89e6bf";
      c.shadowBlur = repair * 13;
      c.beginPath();
      c.arc(x, y, 1.2 + a * 2.6, 0, Math.PI * 2);
      c.fill();
      c.shadowBlur = 0;
    });
  c.restore();
  if (hover) {
    const [hx, hy] = network.worldPoint(...hover, w, h);
    let best = -1,
      distance = 12 / network.camera.zoom;
    positions.forEach(([x, y], i) => {
      const d = Math.hypot(x - hx, y - hy);
      if (d < distance) {
        distance = d;
        best = i;
      }
    });
    if (best >= 0) {
      const name = groups[best],
        offset = best - trace.regions[name].start;
      const note =
        (name === "note_intention" || name === "motif_recall") && offset < 61
          ? " · " + pitchName(36 + offset)
          : "";
      const lines = [
        `${labels[name] || name} · owner ${trace.owner_ids[best]}${note}`,
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
  }
  $("equilibrium").textContent =
    `${trace.released ? "Input-release probe" : (trace.origin || "Recorded settlement") + (replaying ? " · replaying" : trace.equation_error.at(-1) <= 1e-5 ? " · settled" : " · budget reached")} · step ${trace.iterations?.[frame] ?? frame}/${trace.steps} · whole-brain equation error ${trace.equation_error[frame].toExponential(2)}`;
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
  drawBrain(dt);
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
    allEdgesSubmitted: network.enabled || canvasMap?.seams === network.edges,
  },
  canvasSeams: canvasMap?.seams ?? 0,
  traceOwners: trace?.owner_ids.length,
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
