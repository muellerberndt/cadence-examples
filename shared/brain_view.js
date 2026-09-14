import { settlingTrace } from "./telemetry.js";
import { BrainScan, layoutAtlas, PALETTE, roleOf } from "./brain_scan.js";
import { thoughtSnapshot } from "./thought_trace.js";

const labels = {
  sensory: "Sensory input",
  interneuron: "Interneurons",
  motor: "Motor correction",
  visual: "Visual error",
  place: "Spatial field",
  key: "Task / cue input",
  record: "Associative memory",
};
// Region roles the standard palette cannot read from these demos' region keys.
const ROLES = {
  ink: "vision",
  missing: "vision",
  readback: "sensory",
  error: "value",
  features: "sensory",
  futures: "association",
  self_input: "sensory",
  self_state: "value",
  premotor: "motor",
  patch: "association",
};
// Sheets: (rows, columns) of the regions that are images or fields.
const SHAPES = { retina: [24, 24], place: [12, 19] };
export const regionColor = (name) => PALETTE[roleOf(name, ROLES)] ?? PALETTE.other;
const copy = (source) => ({
  ...source,
  state: source.state.slice(),
  edges: source.edges.map((e) => e.slice()),
  drive: source.drive?.slice(),
  input: (source.input ?? source.drive)?.slice(),
  mask: source.mask?.slice(),
  weights: (source.weights ?? source.edges.map((e) => e[2])).slice(),
});
const different = (a, b) =>
  !a || a.length !== b.length || b.some((v, i) => Math.abs(v - a[i]) > 1e-12);

export class BrainView {
  constructor() {
    this.canvas = document.getElementById("brain-scene");
    this.ctx = this.canvas.getContext("2d");
    this.networkCanvas = document.createElement("canvas");
    this.networkCanvas.className = "brain-network";
    this.networkCanvas.setAttribute("aria-hidden", "true");
    this.canvas.before(this.networkCanvas);
    // The whole brain, one integrated scan: the standard Cadence component draws every
    // neuron and synapse; this overlay adds the demo's labels, readouts and plasticity.
    this.scan = null;
    this.topology = null;
    this.atlasGroups = null;
    this.atlasGroupsKey = "";
    this.activeCount = null;
    this.waveCanvas = document.getElementById("brain-waves");
    this.waveCtx = this.waveCanvas.getContext("2d");
    this.$ = (id) => document.getElementById(id);
    this.heat = true;
    this.follow = !matchMedia("(prefers-reduced-motion: reduce)").matches;
    this.reset();
    this.$("brain-heat").onclick = () => {
      this.heat = !this.heat;
      this.$("brain-heat").setAttribute("aria-pressed", String(this.heat));
      this.$("brain-heat").textContent = this.heat
        ? "Heatmap on"
        : "Heatmap off";
    };
    this.$("brain-live").onclick = () => {
      this.trace = null;
      this.auto = null;
      this.pending = null;
      this.frozen = false;
      this.$("brain-freeze").textContent = "Pause view";
    };
    this.$("brain-follow").onclick = () => {
      this.follow = !this.follow;
      this.trace = null;
      this.auto = null;
      this.pending = null;
      this.followLabel();
      if (this.follow && this.source?.recurrent) this.startCascade(this.source);
    };
    this.$("brain-replay").onclick = () => this.replay(false);
    this.$("brain-release").onclick = () => this.replay(true);
    this.$("brain-freeze").onclick = () => {
      this.frozen = !this.frozen;
      this.$("brain-freeze").textContent = this.frozen
        ? "Resume view"
        : "Pause view";
    };
    this.$("brain-signal").onchange = () =>
      (this.signal = this.$("brain-signal").value);
    this.$("brain-speed").onchange = () =>
      (this.rate = +this.$("brain-speed").value);
    this.$("brain-step-back").onclick = () => this.seek(-1);
    this.$("brain-step-next").onclick = () => this.seek(1);
    this.$("brain-timeline").oninput = () => this.seek(0, +this.$("brain-timeline").value);
    this.$("brain-future").oninput = () => {
      const thought = this.source?.thought;
      const index = +this.$("brain-future").value;
      if (!thought?.evaluations[index]) return;
      const captured = thoughtSnapshot(this.source, thought.evaluations[index], index, thought.evaluations.length);
      this.source = captured;
      this.auto = this.pending = null;
      this.trace = this.prepareTrace(captured, false);
      this.frame = this.credit = 0;
      this.frozen = true;
      this.$("brain-freeze").textContent = "Resume view";
    };
    this.$("brain-fit").onclick = () => this.scan?.fit();
    this.$("brain-expand").onclick = () => {
      const panel = this.canvas.closest(".brain-panel");
      if (document.fullscreenElement) document.exitFullscreen();
      else panel.requestFullscreen?.();
    };
    this.canvas.onpointermove = this.canvas.onpointerdown = (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.hover = [e.clientX - r.left, e.clientY - r.top];
    };
    this.canvas.onpointerleave = () => (this.hover = null);
  }
  followLabel() {
    this.$("brain-follow").setAttribute("aria-pressed", String(this.follow));
    this.$("brain-follow").textContent = this.follow
      ? "Following settling"
      : "Follow settling";
  }
  reset() {
    this.scan?.fit();
    this.source = this.old = this.trace = this.auto = this.pending = null;
    this.diagnostic = null;
    this.frozen = false;
    this.signal = "repair";
    this.rate = 40;
    this.age = this.frame = this.credit = this.writes = this.cascades = 0;
    this.received = this.coalesced = 0;
    this.lastPublication = null;
    this.thoughtEpoch = null;
    this.thoughtCursor = -1;
    this.thoughtClock = 0;
    this.waveHistory = {};
    this.waveStamp = null;
    this.changes = [];
    this.repairs = [];
    this.history = [];
    this.trail = [];
    this.hover = null;
    this.trailStamp = null;
    this.displayed = null;
    this.$("brain-signal").value = "repair";
    this.$("brain-speed").value = "40";
    this.$("brain-freeze").textContent = "Pause view";
    this.followLabel();
  }
  prepareTrace(source, release) {
    const trace = settlingTrace(source, release);
    trace.peaks = trace.frames.map((a) =>
      Math.max(
        0,
        ...a
          .filter(
            (_, i) =>
              source.recurrentMask?.[i] ??
              i < (source.recurrentCount ?? a.length),
          )
          .map(Math.abs),
      ),
    );
    // Fixed scales over the captured trajectory make amplitude and fading comparable.
    trace.scales = { activity: {}, repair: {}, input: {}, mismatch: {} };
    const add = (kind, values) =>
      values.forEach((v, i) => {
        const g = source.groups?.[i] ?? "patch";
        trace.scales[kind][g] = Math.max(
          trace.scales[kind][g] ?? 0.01,
          Math.abs(v),
        );
      });
    trace.frames.forEach((a, t) => {
      add("activity", a);
      add("mismatch", trace.mismatches?.[t] ?? a.map(() => 0));
      add(
        "repair",
        a.map((v, i) => (t ? v - trace.frames[t - 1][i] : 0)),
      );
    });
    add("input", source.input ?? source.drive ?? []);
    return trace;
  }
  startCascade(source) {
    const captured = copy(source),
      trace = this.prepareTrace(captured, false);
    if (trace.residuals.every((v) => v === 0)) {
      this.auto = this.pending = null;
      return;
    }
    this.auto = {
      source: captured,
      trace,
      elapsed: 0,
      frame: 0,
    };
    this.pending = null;
    this.cascades++;
  }
  capture(source) {
    this.trace = this.auto = this.pending = null;
    this.update(source);
    if (this.follow && source?.recurrent) this.startCascade(source);
  }
  seek(delta, absolute = null) {
    if (!this.source) return;
    if (!this.trace) {
      const captured = this.auto?.source ?? this.source;
      const frame = this.auto?.frame ?? 0;
      this.source = captured;
      this.trace = this.auto?.trace ?? this.prepareTrace(captured, false);
      this.auto = this.pending = null;
      this.frame = frame;
    }
    this.frame = Math.max(0, Math.min(this.trace.frames.length - 1,
      absolute ?? this.frame + delta));
    this.credit = this.frame;
    this.frozen = true;
    this.$("brain-freeze").textContent = "Resume view";
  }
  replay(release) {
    if (!this.source?.recurrent) return;
    if (release && this.source.phase) {
      const { phase, activeSynapses, recordedTrace, ...whole } = this.source;
      this.source = whole;
    }
    this.auto = this.pending = null;
    this.trailStamp = null;
    this.trail = [];
    this.trace = this.prepareTrace(this.source, release);
    this.signal = release ? "activity" : "repair";
    this.$("brain-signal").value = this.signal;
    this.frame = this.credit = 0;
    this.frozen = false;
    this.$("brain-freeze").textContent = "Pause view";
  }
  update(source) {
    if (!source || this.frozen || this.trace) return;
    const newPublication = source !== this.lastPublication;
    this.lastPublication = source;
    if (newPublication) this.received++;
    const weights = source.weights ?? source.edges.map((e) => e[2]);
    const learned = Object.fromEntries(
      (source.learned ?? []).map(([, id, value]) => [id, value]),
    );
    const consolidated = Object.fromEntries(
      (source.consolidated ?? []).map(([, id, value]) => [id, value]),
    );
    const slowDelta = Array(weights.length).fill(0);
    (source.consolidated ?? []).forEach(([index, id, value]) =>
      slowDelta[index] = value - (this.old?.consolidated?.[id] ?? value));
    if (slowDelta.some(v => Math.abs(v) > 1e-12)) {
      this.slowChanges = slowDelta;
      this.slowFlashAt = this.age;
    }
    if (this.slowChanges?.length !== weights.length)
      this.slowChanges = Array(weights.length).fill(0);
    const delta = Array(weights.length).fill(0);
    (source.learned ?? []).forEach(
      ([index, id, value]) =>
        (delta[index] = value - (this.old?.learned?.[id] ?? value)),
    );
    if ([...delta, ...slowDelta].some((v) => Math.abs(v) > 1e-12)) {
      this.writes++;
      this.changes = delta;
      this.flashAt = this.age;
    }
    if (this.changes.length !== weights.length)
      this.changes = Array(weights.length).fill(0);
    this.repairs = source.state.map((v, i) => v - (this.old?.state[i] ?? v));
    const changed =
      different(this.source?.state, source.state) ||
      different(this.source?.weights, weights) ||
      different(this.source?.drive ?? [], source.drive ?? []) ||
      different(this.source?.mask ?? [], source.mask ?? []);
    this.old = { state: source.state.slice(), learned, consolidated };
    this.source = copy(source);
    if (this.auto && this.auto.source.state.length !== source.state.length)
      this.startCascade(source);
    if (changed || !this.diagnostic)
      this.diagnostic = this.prepareTrace(this.source, false);
    if (changed && this.follow && source.recurrent) {
      if (!this.auto || this.auto.frame >= this.auto.trace.frames.length - 1)
        this.startCascade(source);
      else {
        // Preserve the largest detuning until it has been displayed. A quiet
        // motor tick must not erase a just-published task/map change.
        const peak = Math.max(0, ...(this.diagnostic?.mismatches?.[0] ?? []).map(Math.abs));
        if (!this.pending || peak >= this.pendingPeak) {
          this.pending = this.source;
          this.pendingPeak = peak;
        }
        this.coalesced++;
      }
    }
    this.syncTopology(this.auto?.source ?? this.source);
    this.readout(source);
  }
  regionLabel(source, g) {
    return source.regionLabels?.[g] ?? labels[g] ?? g;
  }
  /** Lay the whole connectome out once per structure; follow weight changes in place. */
  syncTopology(source) {
    const n = source.state.length, edges = source.edges, count = edges.length;
    const groups = source.groups ?? Array(n).fill("patch");
    const groupsKey = groups === this.atlasGroups ? this.atlasGroupsKey : groups.join("");
    const same =
      this.topology &&
      this.scan &&
      this.scan.n === n &&
      this.topology.length === count * 3 &&
      groupsKey === this.atlasGroupsKey &&
      !edges.some((e, i) => e[0] !== this.topology[i * 3] || e[1] !== this.topology[i * 3 + 1]);
    if (!same) {
      const pre = new Uint32Array(count), post = new Uint32Array(count), weight = new Float32Array(count);
      edges.forEach(([a, b, w], i) => { pre[i] = a; post[i] = b; weight[i] = w; });
      const positions = {};
      for (const [g, samples] of Object.entries(source.visualSamples ?? {}))
        positions[g] = samples.map(([sx, sy]) => [sx, -sy]);
      const names = [...new Set(groups)];
      const atlas = layoutAtlas({
        n, pre, post, weight, groups,
        shapes: SHAPES, roles: ROLES, positions,
        labels: Object.fromEntries(names.map((g) => [g, this.regionLabel(source, g)])),
      });
      if (!this.scan) this.scan = new BrainScan(this.networkCanvas, atlas, { interaction: this.canvas, montageRows: 0 });
      else this.scan.setAtlas(atlas);
      this.topology = new Float32Array(count * 3);
      edges.forEach(([a, b, w], i) => this.topology.set([a, b, w], i * 3));
      this.atlasGroups = groups;
      this.atlasGroupsKey = groupsKey;
      this.activeCount = null;
    }
    const active = source.activeSynapses ?? count;
    if (active !== this.activeCount || edges.some((e, i) => Math.fround(e[2]) !== this.topology[i * 3 + 2])) {
      edges.forEach((e, i) => { this.topology[i * 3 + 2] = e[2]; });
      this.scan.setWeights(edges.map((e, i) => (i < active ? e[2] : 0)));
      this.activeCount = active;
    }
  }
  readout(source) {
    this.history.push(
      Math.max(
        0,
        ...source.state
          .slice(0, source.recurrentCount ?? source.state.length)
          .map(Math.abs),
      ),
    );
    if (this.history.length > 150) this.history.shift();
    this.$("brain-replay").disabled = this.$("brain-release").disabled =
      !source.recurrent;
    this.$("brain-follow").disabled = !source.recurrent;
    this.$("brain-follow").textContent = !source.recurrent
      ? "Direct read / write"
      : this.follow
        ? "Following settling"
        : "Follow settling";
    this.$("brain-memory").textContent = source.memory;
    this.$("brain-signal").querySelector('[value="consolidation"]').disabled = !source.consolidated?.length;
    this.$("brain-count").textContent =
      `${source.state.length} neurons · ${source.edges.length} synapses`;
    const behavior = source.behavior ?? { label: "Reading", tone: "neutral" };
    this.$("brain-behavior").textContent = behavior.label;
    this.$("brain-behavior").dataset.tone = behavior.tone;
    this.$("brain-adapters").textContent = source.adapters ?? "";
  }
  draw(dt) {
    if (!this.source) return;
    const { width: w, height: h } = this.canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio, 2),
      c = this.ctx;
    if (
      this.canvas.width !== Math.round(w * dpr) ||
      this.canvas.height !== Math.round(h * dpr)
    ) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    if (!this.frozen) {
      this.age += dt;
      const thought = this.source.thought;
      this.$("brain-futures").hidden = !thought;
      if (thought) {
        if (thought.epoch !== this.thoughtEpoch) {
          this.thoughtEpoch = thought.epoch;
          this.thoughtCursor = -1;
          this.thoughtClock = 0;
        }
        this.$("brain-future").max = Math.max(0, thought.evaluations.length - 1);
        this.$("brain-future-count").textContent = `${thought.evaluated.toLocaleString()} actual value evaluations recorded`;
        this.thoughtClock += dt;
        if (!this.trace && this.follow && this.thoughtClock >= 0.3
            && this.thoughtCursor < thought.evaluations.length - 1) {
          // Live playback samples the complete retained evaluation history.
          // The separate future slider can inspect every recorded evaluation.
          this.thoughtCursor = Math.min(thought.evaluations.length - 1,
            this.thoughtCursor + Math.max(1, Math.ceil(thought.evaluations.length / 24)));
          this.$("brain-future").value = this.thoughtCursor;
          this.startCascade(thoughtSnapshot(this.source,
            thought.evaluations[this.thoughtCursor], this.thoughtCursor, thought.evaluations.length));
          this.thoughtClock = 0;
        }
      }
      if (this.trace) {
        this.credit += dt * this.rate;
        this.frame = Math.min(
          this.trace.frames.length - 1,
          Math.floor(this.credit),
        );
      } else if (this.auto) {
        this.auto.elapsed += dt;
        const end = this.auto.trace.frames.length - 1;
        this.auto.frame = Math.min(
          end,
          Math.floor(this.auto.source.phase ? this.auto.elapsed * 8
            : end > 120
              ? Math.expm1(Math.min(1, this.auto.elapsed / 3) * Math.log1p(end))
              : this.auto.elapsed * this.rate),
        );
        if (this.auto.frame === end && this.pending)
          this.startCascade(this.pending);
      }
    }
    const source = this.auto?.source ?? this.source;
    this.syncTopology(source);
    const scan = this.scan;
    const trace = this.trace ?? this.auto?.trace;
    const frame = this.trace ? this.frame : (this.auto?.frame ?? 0);
    const s = trace ? trace.frames[frame] : source.state;
    const diff = trace
      ? s.map((v, i) => v - trace.frames[Math.max(0, frame - 1)][i])
      : this.repairs;
    const input = (source.input ?? s.map(() => 0)).map((v, i) =>
      trace?.release &&
      (source.recurrentMask?.[i] ?? i < (source.recurrentCount ?? s.length))
        ? 0
        : v,
    );
    const values =
      this.signal === "input"
        ? input
        : this.signal === "repair"
          ? diff
          : this.signal === "mismatch"
            ? (trace?.mismatches?.[frame] ??
              this.diagnostic?.mismatches?.at(-1) ??
              s.map(() => 0))
            : s;
    const groups = source.groups ?? s.map(() => "patch");
    const groupNames = [...new Set(groups)].sort((a, b) => {
      const order = { sensory: 0, interneuron: 1, motor: 2 };
      return (order[a] ?? 3) - (order[b] ?? 3);
    });
    const regionLabel = (g) => this.regionLabel(source, g);
    this.$("brain-regions").textContent = groupNames
      .map(regionLabel)
      .join(" · ");
    const maxima = {},
      repairMax = {};
    values.forEach((v, i) => {
      const g = groups[i];
      maxima[g] = Math.max(maxima[g] ?? 0.01, Math.abs(v));
      repairMax[g] = Math.max(repairMax[g] ?? 0.01, Math.abs(diff[i]));
    });
    const scales = trace?.scales[this.signal] ?? maxima;
    const logarithmic = this.signal === "repair" || this.signal === "mismatch";
    const intensity = (v, scale) => Math.abs(v) <= 1e-8 ? 0
      : Math.min(1, Math.log1p(Math.abs(v) / 1e-8) / Math.log1p(scale / 1e-8));
    const norm = values.map((v, i) => logarithmic
      ? Math.sign(v) * intensity(v, scales[groups[i]] ?? .01)
      : Math.max(-1, Math.min(1, v / (scales[groups[i]] ?? .01))));
    const repairs = diff.map((v, i) => intensity(v,
      trace?.scales.repair[groups[i]] ?? repairMax[groups[i]]));
    // Only actual displayed changes replenish the trail; holding a replay frame must not keep it alive.
    const stamp = trace
      ? `${this.cascades}:${!!this.trace}:${trace.release}:${frame}`
      : this.old;
    if (!this.frozen) {
      const fresh = this.trailStamp !== stamp;
      this.trail = s.map((_, i) =>
        Math.max(
          (this.trail[i] ?? 0) * Math.exp(-dt * 4),
          fresh ? repairs[i] : 0,
        ),
      );
      // The scan's glow follows the change itself (linear on the region scale), so a wave
      // of repairs reads as a wave; the logarithmic trail above feeds the readouts.
      this.glow = s.map((_, i) =>
        Math.max(
          (this.glow?.[i] ?? 0) * Math.exp(-dt * 4),
          fresh ? Math.min(1, Math.abs(diff[i]) / (trace?.scales.repair[groups[i]] ?? repairMax[groups[i]] ?? 0.01)) : 0,
        ),
      );
      this.trailStamp = stamp;
    }
    if (this.glow?.length !== s.length) this.glow = s.map(() => 0);
    const lasting = this.signal === "consolidation",
      plastic = this.signal === "plasticity" || lasting,
      max = Math.max(1e-12, ...s.map(Math.abs));
    // The scan: messages are the actual activations, the glow is the change that just
    // happened, and the brightness is the selected signal on its region scale.
    scan.options.field = this.heat && !plastic;
    scan.options.particles = !plastic;
    scan.setVisible(source.mask ?? null);
    scan.show(s.map((v) => v / max), this.glow, { level: norm });
    const flat = scan.screenAll();
    const positions = Array.from({ length: s.length }, (_, i) => [flat[2 * i], flat[2 * i + 1]]);
    const slowWeights = Object.fromEntries((source.consolidated ?? []).map(([i, , v]) => [i, v]));
    if (!scan.enabled || plastic)
      source.edges.forEach(([a, b, weight], i) => {
        if (source.mask?.[a] === 0 || source.mask?.[b] === 0) return;
        if (lasting) weight = slowWeights[i] ?? 0;
        const delta =
          ((lasting ? this.slowChanges : this.changes)[i] ?? 0) *
          Math.max(0, 1 - (this.age - ((lasting ? this.slowFlashAt : this.flashAt) ?? -10)) / 1.5);
        const strength = Math.min(1, Math.abs(s[a] * weight) / max);
        const pulse = i < (source.activeSynapses ?? source.edges.length)
          ? repairs[a] * Math.min(1, Math.abs(weight)) : 0;
        c.strokeStyle = plastic
          ? `rgba(${delta < 0 ? "121,184,255" : "255,196,113"},${0.04 + Math.min(1, Math.abs(weight)) * 0.3 + Math.min(1, Math.abs(delta)) * 0.6})`
          : `rgba(${pulse > 0.08 ? "191,168,255" : "130,182,200"},${0.035 + strength * 0.18 + pulse * 0.55})`;
        c.lineWidth = plastic
          ? 0.6 + Math.min(2, Math.abs(weight)) + Math.min(2, Math.abs(delta))
          : 0.5 + pulse * 1.3;
        c.beginPath();
        c.moveTo(...positions[a]);
        const [ax, ay] = positions[a],
          [bx, by] = positions[b];
        const bend = groups[a] !== groups[b] ? (bx - ax) * 0.18 : 0;
        c.quadraticCurveTo((ax + bx) / 2 - bend, (ay + by) / 2 + bend, bx, by);
        c.stroke();
        // A packet represents a changed outgoing message at this captured iteration.
        // No change means no moving packet, including at a fixed point.
        const message = i < (source.activeSynapses ?? source.edges.length) ? diff[a] * weight : 0;
        if (
          !plastic &&
          trace &&
          frame < trace.frames.length - 1 &&
          Math.abs(message) > 1e-15 &&
          pulse > 0.03
        ) {
          const progress = (this.age * 2.2) % 1,
            inv = 1 - progress;
          const px =
            inv * inv * ax +
            2 * inv * progress * ((ax + bx) / 2 - bend) +
            progress * progress * bx;
          const py =
            inv * inv * ay +
            2 * inv * progress * ((ay + by) / 2 + bend) +
            progress * progress * by;
          c.fillStyle = message < 0 ? "#79b8ff" : "#ffc471";
          c.shadowColor = c.fillStyle;
          c.shadowBlur = 8;
          c.beginPath();
          c.arc(px, py, 1.2 + Math.min(2, pulse * 2), 0, Math.PI * 2);
          c.fill();
          c.shadowBlur = 0;
        }
      });
    let hovered = -1,
      nearest = 18;
    positions.forEach(([x, y], i) => {
      if (source.mask?.[i] === 0) {
        // A silenced or absent neuron: a dim mark where it would be.
        c.fillStyle = "rgba(101,54,71,0.6)";
        c.beginPath();
        c.arc(x, y, 1.6, 0, Math.PI * 2);
        c.fill();
      }
      if (s.length <= 4) {
        c.fillStyle = "#a3b4b9";
        c.font = "10px ui-monospace, monospace";
        c.textAlign = "center";
        c.fillText(source.names?.[i] ?? `Neuron ${i}`, x, y + 22, 120);
        c.fillText(values[i].toExponential(2), x, y + 35);
      }
      if (this.hover) {
        const d = Math.hypot(x - this.hover[0], y - this.hover[1]);
        if (d < nearest) {
          nearest = d;
          hovered = i;
        }
      }
    });
    // Region labels ride on the scan, above each region: its name and its live neurons.
    c.textAlign = "center";
    scan.atlas.regions.forEach((region, k) => {
      const members = scan.atlas.memberIndices[k];
      const [x, above] = scan.toScreen(region.center[0], region.center[1] + region.extent[1]);
      const [, middle] = scan.toScreen(region.center[0], region.center[1]);
      if (x < -60 || x > w + 60 || middle < -20 || middle > h + 20) return;
      const top = Math.max(34, Math.min(h - 4, above)); // below the caption line
      const enabled = members.filter((i) => source.mask?.[i] !== 0).length;
      c.font = "11px system-ui, sans-serif";
      c.fillStyle = `rgba(${region.color.join(",")},0.92)`;
      c.shadowColor = "#000";
      c.shadowBlur = 4;
      c.fillText(`${region.label ?? region.name} · ${enabled === members.length ? enabled : `${enabled}/${members.length}`}`, x, top - 6);
      c.shadowBlur = 0;
    });
    c.textAlign = "left";
    c.font = "10px ui-monospace, monospace";
    c.fillStyle = "#91a6b1";
    c.fillText(
      trace
        ? trace.release
          ? "INPUT RELEASE / ISOLATED COPY"
          : this.trace
            ? "CAPTURED REPAIR REPLAY"
            : "SAMPLED REPAIR / TIME EXPANDED"
        : "LIVE CIRCUIT",
      8,
      15,
    );
    if (hovered >= 0)
      c.fillText(
        `${source.names?.[hovered] ?? `Neuron ${hovered}`} · ${regionLabel(groups[hovered])} · ${values[hovered].toExponential(3)}`,
        8,
        h - 22,
        w - 16,
      );
    const series = trace ? trace.peaks.slice(0, frame + 1) : this.history;
    const peak = Math.max(1e-12, ...(trace?.peaks ?? series));
    c.strokeStyle = "#91d0d7";
    c.lineWidth = 1;
    c.beginPath();
    series.forEach((v, i) => {
      const x =
          8 +
          (i * (w - 16)) /
            Math.max(149, (trace?.peaks.length ?? series.length) - 1),
        y = h - 3 - (12 * v) / peak;
      i ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.stroke();
    this.$("brain-status").textContent = trace
      ? `${source.phase ?? (trace.release ? "Input released in an isolated copy" : this.trace ? "Settling replay" : "Sampled settling")} · iteration ${frame}/${trace.frames.length - 1}${frame === trace.frames.length - 1 ? " · replay complete" : ""}`
      : `Live ${source.recurrent ? "settled state" : "associative read/write"} · ${this.writes} observed weight changes`;
    const equilibrium = source.equilibrium;
    const mismatch = trace?.mismatches?.[frame] ?? this.diagnostic?.mismatches?.at(-1) ?? [];
    const error = Math.max(0, ...mismatch.map(Math.abs));
    const changing = diff.filter((v, i) => source.mask?.[i] !== 0 && Math.abs(v) > 1e-8).length;
    const messages = source.edges.filter(([a, b, weight], i) => i < (source.activeSynapses ?? source.edges.length) && source.mask?.[a] !== 0
      && source.mask?.[b] !== 0 && Math.abs(diff[a] * weight) > 1e-8).length;
    this.$("brain-repair-count").textContent = `${changing} / ${s.length}`;
    this.$("brain-message-count").textContent = `${messages} / ${source.edges.length}`;
    this.$("brain-error-now").textContent = error.toExponential(1);
    this.$("brain-timeline").max = (trace ?? this.diagnostic)?.frames.length - 1 || 0;
    this.$("brain-timeline").value = trace ? frame : this.$("brain-timeline").max;
    this.$("brain-step-label").textContent = `${trace ? frame : this.$("brain-timeline").max} / ${this.$("brain-timeline").max}`;
    this.$("brain-equilibrium").textContent = equilibrium
      ? `${source.phase ? "Isolated future value cortex" : error <= equilibrium.tolerance ? "At equilibrium" : "Repairing the displayed state"} · error now ${error.toExponential(1)} → final ${equilibrium.residual.toExponential(1)}`
      : "Reference circuit · independently inspected";
    this.$("brain-equilibrium").title = equilibrium
      ? Object.entries(equilibrium.regions)
          .map(([g, e]) => `${regionLabel(g)}: ${e.toExponential(2)}`)
          .join(" · ")
      : "";
    this.$("brain-region-errors").textContent = equilibrium
      ? Object.entries(equilibrium.regions)
          .map(([g, e]) => `${regionLabel(g)}: ${e.toExponential(2)}`)
          .join(" · ")
      : "";
    this.$("brain-legend").hidden = !this.heat && !plastic;
    this.$("brain-legend-label").textContent =
      `${plastic ? (lasting ? "Persistent synaptic strength · gold strengthens, blue weakens" : "Total synaptic strength · gold strengthens, blue weakens") : this.signal === "input" ? "Input drive" : this.signal === "repair" ? "Activity change" : this.signal === "mismatch" ? "Equation mismatch" : "Activity"}${plastic ? "" : logarithmic ? " · logarithmic magnitude" : " · relative region scale"}${trace ? " · fixed during replay" : ""}`;
    this.$("brain-scale").textContent = plastic
      ? `Max |synaptic strength| ${Math.max(0, ...(lasting ? source.consolidated ?? [] : source.edges).map(([, , v]) => Math.abs(v))).toExponential(2)} · connections show ${lasting ? "persistent" : "total"} weights; flashes show their signed changes. Neuron colors still show activity.`
      : `Max |value| ${Math.max(0, ...values.map(Math.abs)).toExponential(2)} · dimensionless model units. ${trace ? "Captured trajectory scales" : "Current region scales"}; hover or tap for signed values. No neurotransmitter concentrations are modeled.`;
    const waveTrace = trace ?? this.diagnostic;
    const waveFrame = trace ? frame : (this.diagnostic?.frames.length ?? 1) - 1;
    if (!this.frozen && stamp !== this.waveStamp) {
      for (const [group, series] of Object.entries(waveTrace?.populations ?? {})) {
        const history = this.waveHistory[group] ??= [];
        history.push(series[waveFrame]);
        if (history.length > 180) history.shift();
      }
      this.waveStamp = stamp;
    }
    this.drawWaves(
      source,
      this.trace ? waveTrace : { populations: this.waveHistory },
      this.trace ? waveFrame : 179,
      regionLabel,
    );
    this.$("brain-modulators").textContent = source.modulators
      ? Object.entries(source.modulators)
          .map(
            ([name, m]) =>
              `${name}: ${Number(m.value).toFixed(2)} · ${m.effect}`,
          )
          .join(" | ")
      : "Modulators: no neurotransmitter model · activity and mismatch are measured model signals";
    this.displayed = {
      state: s.slice(),
      repairs: diff.slice(),
      mismatch: (
        trace?.mismatches?.[frame] ??
        this.diagnostic?.mismatches?.at(-1) ??
        []
      ).slice(),
      frame,
      regions: groupNames.map(regionLabel),
      trail: this.trail.slice(),
      changing, messages, error,
      phase: source.phase ?? "Joint equilibrium",
    };
  }
  drawWaves(source, trace, frame, label) {
    const canvas = this.waveCanvas,
      c = this.waveCtx,
      { width: w, height: h } = canvas.getBoundingClientRect();
    const dpr = Math.min(devicePixelRatio, 2);
    if (
      canvas.width !== Math.round(w * dpr) ||
      canvas.height !== Math.round(h * dpr)
    ) {
      canvas.width = Math.round(w * dpr);
      canvas.height = Math.round(h * dpr);
    }
    c.setTransform(dpr, 0, 0, dpr, 0, 0);
    c.clearRect(0, 0, w, h);
    c.fillStyle = "#96aaa9";
    c.font = "9px ui-monospace, monospace";
    c.fillText(
      this.trace ? "CAPTURED ITERATIONS · activity / mismatch" : "POPULATION HISTORY · displayed activity / mismatch",
      4,
      10,
      w - 8,
    );
    const entries = Object.entries(trace.populations ?? {}),
      rows = Math.max(1, entries.length),
      rh = (h - 17) / rows;
    entries.forEach(([g, series], r) => {
      const y = 18 + r * rh;
      const color = regionColor(g);
      c.fillStyle = `rgba(${color.join(",")},0.9)`;
      c.fillText(label(g), 4, y + rh * 0.65, 120);
      const peak = Math.max(
        0.01,
        ...series.flatMap((v) => [Math.abs(v.mean), v.mismatch]),
      );
      for (const [key, stroke] of [
        ["mean", `rgba(${color.join(",")},0.95)`],
        ["mismatch", "#bfa8ff"],
      ]) {
        c.strokeStyle = stroke;
        c.lineWidth = 1;
        c.beginPath();
        series.slice(0, frame + 1).forEach((v, i) => {
          const x = 132 + ((w - 138) * i) / Math.max(this.trace ? 1 : 179, series.length - 1),
            py = y + rh * 0.55 - ((rh - 2) * 0.45 * v[key]) / peak;
          i ? c.lineTo(x, py) : c.moveTo(x, py);
        });
        c.stroke();
      }
    });
  }
  snapshot() {
    return {
      neurons: this.source?.state.length ?? 0,
      synapses: this.source?.edges.length ?? 0,
      recurrent: this.source?.recurrent ?? false,
      equilibrium: this.source?.equilibrium,
      writes: this.writes,
      signal: this.signal,
      heatmap: this.heat,
      input: this.source?.input,
      replay: !!this.trace,
      release: this.trace?.release ?? false,
      frame: this.frame,
      last: this.trace?.frames.at(-1),
      state: this.source?.state,
      weights: this.source?.weights,
      follow: this.follow,
      cascades: this.cascades,
      received: this.received,
      coalesced: this.coalesced,
      automatic: !!this.auto,
      behavior: this.source?.behavior,
      displayed: this.displayed,
      topology: this.scan
        ? this.scan.snapshot()
        : { renderer: "none", neurons: 0, synapses: 0, zoom: 1, allEdgesSubmitted: false },
    };
  }
}
