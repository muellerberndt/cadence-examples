import { repairTrace } from "./telemetry.js";

const labels = {
  sensory: "Sensory input",
  interneuron: "Interneurons",
  motor: "Motor correction",
  visual: "Visual error",
  place: "Spatial field",
  key: "Task / cue input",
  record: "Associative memory",
};
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
    this.canvas.onpointermove = this.canvas.onpointerdown = (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.hover = [e.clientX - r.left, e.clientY - r.top];
    };
    this.canvas.onpointerleave = () => (this.hover = null);
  }
  followLabel() {
    this.$("brain-follow").setAttribute("aria-pressed", String(this.follow));
    this.$("brain-follow").textContent = this.follow
      ? "Following repairs"
      : "Follow repairs";
  }
  reset() {
    this.source = this.old = this.trace = this.auto = this.pending = null;
    this.frozen = false;
    this.signal = "activity";
    this.rate = 40;
    this.age = this.frame = this.credit = this.writes = this.cascades = 0;
    this.changes = [];
    this.repairs = [];
    this.history = [];
    this.trail = [];
    this.hover = null;
    this.trailStamp = null;
    this.displayed = null;
    this.$("brain-signal").value = "activity";
    this.$("brain-speed").value = "40";
    this.$("brain-freeze").textContent = "Pause view";
    this.followLabel();
  }
  prepareTrace(source, release) {
    const trace = repairTrace(source, release);
    trace.peaks = trace.frames.map((a) =>
      Math.max(
        0,
        ...a.slice(0, source.recurrentCount ?? a.length).map(Math.abs),
      ),
    );
    // Fixed scales over the captured trajectory make amplitude and fading comparable.
    trace.scales = { activity: {}, repair: {}, input: {} };
    const add = (kind, values) =>
      values.forEach((v, i) => {
        const g = source.groups?.[i] ?? "patch";
        trace.scales[kind][g] = Math.max(
          trace.scales[kind][g] ?? 1e-12,
          Math.abs(v),
        );
      });
    trace.frames.forEach((a, t) => {
      add("activity", a);
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
  replay(release) {
    if (!this.source?.recurrent) return;
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
    const weights = source.weights ?? source.edges.map((e) => e[2]);
    const learned = Object.fromEntries(
      (source.learned ?? []).map(([, id, value]) => [id, value]),
    );
    const delta = Array(weights.length).fill(0);
    (source.learned ?? []).forEach(
      ([index, id, value]) =>
        (delta[index] = value - (this.old?.learned?.[id] ?? value)),
    );
    if (delta.some((v) => Math.abs(v) > 1e-12)) {
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
    this.old = { state: source.state.slice(), learned };
    this.source = copy(source);
    if (changed && this.follow && source.recurrent) {
      if (!this.auto || this.auto.elapsed >= 1.6) this.startCascade(source);
      else this.pending = this.source;
    }
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
        ? "Following repairs"
        : "Follow repairs";
    this.$("brain-memory").textContent = source.memory;
    this.$("brain-count").textContent =
      `${source.state.length} owners · ${source.edges.length} seams`;
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
      if (this.trace) {
        this.credit += dt * this.rate;
        this.frame = Math.min(
          this.trace.frames.length - 1,
          Math.floor(this.credit),
        );
      } else if (this.auto) {
        this.auto.elapsed += dt;
        // Time expansion of early iterations, compression of the long convergence tail.
        const end = this.auto.trace.frames.length - 1;
        this.auto.frame = Math.min(
          end,
          Math.floor(
            Math.expm1(Math.min(1, this.auto.elapsed / 1.6) * Math.log1p(end)),
          ),
        );
        if (this.auto.elapsed >= 1.6 && this.pending)
          this.startCascade(this.pending);
      }
    }
    const source = this.auto?.source ?? this.source;
    const trace = this.trace ?? this.auto?.trace;
    const frame = this.trace ? this.frame : (this.auto?.frame ?? 0);
    const s = trace ? trace.frames[frame] : source.state;
    const diff = trace
      ? s.map((v, i) => v - trace.frames[Math.max(0, frame - 1)][i])
      : this.repairs;
    const input = (source.input ?? s.map(() => 0)).map((v, i) =>
      trace?.release && i < (source.recurrentCount ?? s.length) ? 0 : v,
    );
    const values =
      this.signal === "input" ? input : this.signal === "repair" ? diff : s;
    const groups = source.groups ?? s.map(() => "patch");
    const groupNames = [...new Set(groups)].sort((a, b) => {
      const order = { sensory: 0, interneuron: 1, motor: 2 };
      return (order[a] ?? 0) - (order[b] ?? 0);
    });
    const regionLabel = (g) => source.regionLabels?.[g] ?? labels[g] ?? g;
    this.$("brain-regions").textContent = groupNames
      .map(regionLabel)
      .join(" · ");
    const maxima = {},
      repairMax = {};
    values.forEach((v, i) => {
      const g = groups[i];
      maxima[g] = Math.max(maxima[g] ?? 1e-12, Math.abs(v));
      repairMax[g] = Math.max(repairMax[g] ?? 1e-12, Math.abs(diff[i]));
    });
    const scales = trace?.scales[this.signal] ?? maxima;
    const norm = values.map((v, i) =>
      Math.max(-1, Math.min(1, v / (scales[groups[i]] ?? 1e-12))),
    );
    const repairs = diff.map((v, i) =>
      Math.min(
        1,
        Math.abs(v) / (trace?.scales.repair[groups[i]] ?? repairMax[groups[i]]),
      ),
    );
    // Only actual displayed changes replenish the trail; holding a replay frame must not keep it alive.
    const stamp = trace
      ? `${this.cascades}:${!!this.trace}:${trace.release}:${frame}`
      : this.old;
    if (!this.frozen) {
      this.trail = s.map((_, i) =>
        Math.max(
          (this.trail[i] ?? 0) * Math.exp(-dt * 4),
          this.trailStamp !== stamp ? repairs[i] : 0,
        ),
      );
      this.trailStamp = stamp;
    }
    const regions = {},
      positions = Array(s.length);
    const top = 30,
      bottom = h - 38,
      gap = 12,
      usable = bottom - top;
    groupNames.forEach((g, gi) => {
      let x = 8,
        y = top,
        rw = w - 16,
        rh = usable;
      if (groupNames.includes("place")) {
        if (g === "place") rw = (w - 28) * 0.64;
        else {
          x = 20 + (w - 28) * 0.64;
          rw = w - x - 8;
          rh = (usable - gap) / 2;
          y += (gi - 1) * (rh + gap);
        }
      } else if (groupNames.length === 2) {
        rw = (w - 16 - gap) / 2;
        x += gi * (rw + gap);
      } else {
        rh = (usable - gap * (groupNames.length - 1)) / groupNames.length;
        y += gi * (rh + gap);
      }
      const members = groups.flatMap((v, i) => (v === g ? [i] : []));
      regions[g] = { x, y, rw, rh, members };
      c.fillStyle = "#122129";
      c.strokeStyle = "#2b424b";
      c.lineWidth = 1;
      c.beginPath();
      c.roundRect(x, y, rw, rh, 8);
      c.fill();
      c.stroke();
      c.fillStyle = "#d8e6e9";
      c.font = "11px system-ui, sans-serif";
      c.textAlign = "left";
      c.fillText(regionLabel(g), x + 9, y + 17, rw - 18);
      members.forEach((i, k) => {
        if (g === "place") {
          positions[i] = [
            x + 14 + ((i % 19) / 18) * (rw - 28),
            y + 36 + (Math.floor(i / 19) / 12) * (rh - 54),
          ];
        } else {
          const cols =
            members.length <= 2
              ? 1
              : Math.max(
                  2,
                  Math.ceil(
                    Math.sqrt((members.length * rw) / Math.max(35, rh - 40)),
                  ),
                );
          const rows = Math.ceil(members.length / cols);
          positions[i] = [
            x + 14 + (((k % cols) + 0.5) / cols) * (rw - 28),
            y + 29 + ((Math.floor(k / cols) + 0.5) / rows) * (rh - 42),
          ];
        }
      });
    });
    const plastic = this.signal === "plasticity",
      max = Math.max(1e-12, ...s.map(Math.abs));
    source.edges.forEach(([a, b, weight], i) => {
      if (source.mask?.[a] === 0 || source.mask?.[b] === 0) return;
      const delta =
        (this.changes[i] ?? 0) *
        Math.max(0, 1 - (this.age - (this.flashAt ?? -10)) / 1.5);
      const strength = Math.min(1, Math.abs(s[a] * weight) / max);
      const pulse = repairs[a] * Math.min(1, Math.abs(weight));
      c.strokeStyle = plastic
        ? `rgba(${delta < 0 ? "121,184,255" : "255,196,113"},${0.04 + Math.min(1, Math.abs(weight)) * 0.3 + Math.min(1, Math.abs(delta)) * 0.6})`
        : `rgba(${pulse > 0.08 ? "191,168,255" : "130,182,200"},${0.035 + strength * 0.18 + pulse * 0.55})`;
      c.lineWidth = plastic
        ? 0.6 + Math.min(2, Math.abs(weight)) + Math.min(2, Math.abs(delta))
        : 0.5 + pulse * 1.3;
      c.beginPath();
      c.moveTo(...positions[a]);
      c.lineTo(...positions[b]);
      c.stroke();
    });
    let hovered = -1,
      nearest = 18;
    positions.forEach(([x, y], i) => {
      const value = Math.abs(norm[i]),
        muted = source.mask?.[i] === 0;
      const radius = s.length <= 12 ? 4 + value * 4 : 1.4 + value * 1.7;
      const rgb = norm[i] < 0 ? "121,184,255" : "255,196,113";
      if (this.heat && !plastic && !muted && value > 0.001) {
        const size = s.length <= 12 ? 25 : 10;
        const glow = c.createRadialGradient(x, y, 0, x, y, size);
        glow.addColorStop(0, `rgba(${rgb},${value * 0.45})`);
        glow.addColorStop(1, `rgba(${rgb},0)`);
        c.fillStyle = glow;
        c.fillRect(x - size, y - size, size * 2, size * 2);
      }
      c.fillStyle = muted
        ? "#653647"
        : (this.heat && !plastic) ||
            this.signal === "repair" ||
            this.signal === "input"
          ? `rgb(${rgb})`
          : "#96f0cc";
      c.globalAlpha = muted ? 0.4 : 0.22 + 0.78 * value;
      c.beginPath();
      c.arc(x, y, radius, 0, Math.PI * 2);
      c.fill();
      c.globalAlpha = 1;
      if (!muted && this.trail[i] > 0.015) {
        c.strokeStyle = `rgba(191,168,255,${this.trail[i]})`;
        c.lineWidth = 1.5;
        c.beginPath();
        c.arc(x, y, radius + 3 + this.trail[i] * 2, 0, Math.PI * 2);
        c.stroke();
      }
      if (s.length <= 4) {
        c.fillStyle = "#a3b4b9";
        c.font = "10px ui-monospace, monospace";
        c.textAlign = "center";
        c.fillText(
          source.names?.[i] ?? `Owner ${i}`,
          x,
          y + 22,
          regions[groups[i]].rw - 16,
        );
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
    Object.values(regions).forEach(({ x, y, rw, rh, members }) => {
      const enabled = members.filter((i) => source.mask?.[i] !== 0).length;
      c.fillText(
        `${enabled}/${members.length} enabled`,
        x + 9,
        y + rh - 5,
        rw - 18,
      );
    });
    if (hovered >= 0)
      c.fillText(
        `${source.names?.[hovered] ?? `Owner ${hovered}`} · ${values[hovered].toExponential(3)}`,
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
      ? `${trace.release ? "Input released in an isolated copy" : this.trace ? "Repair replay" : "Sampled settlement"} · iteration ${frame}/${trace.frames.length - 1}${frame === trace.frames.length - 1 ? " · settled" : ""}`
      : `Live ${source.recurrent ? "settled state" : "associative read/write"} · ${this.writes} observed weight changes`;
    this.$("brain-legend").hidden = !this.heat || plastic;
    this.$("brain-legend-label").textContent =
      `${this.signal === "input" ? "Input drive" : this.signal === "repair" ? "State change" : "Activity"} · relative region scale${trace ? " · fixed during replay" : ""}`;
    this.$("brain-scale").textContent =
      `Max |value| ${Math.max(0, ...values.map(Math.abs)).toExponential(2)} · dimensionless model units. ${trace ? "Captured trajectory scales" : "Current region scales"}; hover or tap for signed values. No neurotransmitter concentrations are modeled.`;
    this.displayed = {
      state: s.slice(),
      repairs: diff.slice(),
      frame,
      regions: groupNames.map(regionLabel),
      trail: this.trail.slice(),
    };
  }
  snapshot() {
    return {
      owners: this.source?.state.length ?? 0,
      seams: this.source?.edges.length ?? 0,
      recurrent: this.source?.recurrent ?? false,
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
      automatic: !!this.auto,
      behavior: this.source?.behavior,
      displayed: this.displayed,
    };
  }
}
