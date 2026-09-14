import { repairTrace } from "./telemetry.js";

export class BrainView {
  constructor() {
    this.canvas = document.getElementById("brain-scene");
    this.ctx = this.canvas.getContext("2d");
    this.$ = (id) => document.getElementById(id);
    this.heat = true;
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
      this.frozen = false;
      this.$("brain-freeze").textContent = "Pause view";
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
    this.canvas.onpointermove = (e) => {
      const r = this.canvas.getBoundingClientRect();
      this.hover = [e.clientX - r.left, e.clientY - r.top];
    };
    this.canvas.onpointerleave = () => (this.hover = null);
  }
  reset() {
    this.source = null;
    this.old = null;
    this.trace = null;
    this.frozen = false;
    this.signal = "activity";
    this.rate = 40;
    this.age = 0;
    this.frame = 0;
    this.changes = [];
    this.repairs = [];
    this.history = [];
    this.credit = 0;
    this.writes = 0;
    this.signature = "";
    if (this.$) {
      this.$("brain-signal").value = "activity";
      this.$("brain-freeze").textContent = "Pause view";
    }
  }
  replay(release) {
    if (!this.source?.recurrent) return;
    this.trace = repairTrace(this.source, release);
    this.trace.peaks = this.trace.frames.map((a) =>
      Math.max(
        0,
        ...a.slice(0, this.source.recurrentCount ?? a.length).map(Math.abs),
      ),
    );
    this.signal = release ? "activity" : "repair";
    this.$("brain-signal").value = this.signal;
    this.frame = 0;
    this.credit = 0;
    this.frozen = false;
    this.$("brain-freeze").textContent = "Pause view";
  }
  update(source) {
    if (!source || this.frozen || this.trace) return;
    const weights = source.weights ?? source.edges.map((e) => e[2]);
    const learned = Object.fromEntries(
      (source.learned ?? []).map(([index, id, value]) => [id, value]),
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
    this.old = {
      weights: weights.slice(),
      state: source.state.slice(),
      learned,
    };
    this.source = {
      ...source,
      state: source.state.slice(),
      edges: source.edges.map((e) => e.slice()),
      drive: source.drive?.slice(),
      input: (source.input ?? source.drive)?.slice(),
      mask: source.mask?.slice(),
      weights: weights.slice(),
    };
    this.history.push(
      Math.max(
        0,
        ...source.state
          .slice(0, source.recurrentCount ?? source.state.length)
          .map(Math.abs),
      ),
    );
    if (this.history.length > 150) this.history.shift();
    this.$("brain-replay").disabled = !source.recurrent;
    this.$("brain-release").disabled = !source.recurrent;
    this.$("brain-memory").textContent = source.memory;
    this.$("brain-count").textContent =
      `${source.state.length} owners · ${source.edges.length} seams`;
  }
  draw(dt) {
    if (!this.source) return;
    const r = this.canvas.getBoundingClientRect(),
      w = r.width,
      h = r.height,
      dpr = Math.min(devicePixelRatio, 2);
    if (
      this.canvas.width !== Math.round(w * dpr) ||
      this.canvas.height !== Math.round(h * dpr)
    ) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    const c = this.ctx;
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
      }
    }
    const s = this.trace ? this.trace.frames[this.frame] : this.source.state;
    const diff = this.trace
      ? s.map((v, i) => v - this.trace.frames[Math.max(0, this.frame - 1)][i])
      : this.repairs;
    const input = (this.source.input ?? s.map(() => 0)).map((v, i) =>
      this.trace?.release && i < (this.source.recurrentCount ?? s.length)
        ? 0
        : v,
    );
    const values =
      this.signal === "input" ? input : this.signal === "repair" ? diff : s;
    const scaleValues =
      this.signal === "activity" && this.trace?.release
        ? this.source.state
        : values;
    const max = Math.max(1e-12, ...s.map(Math.abs)),
      dm = Math.max(1e-12, ...diff.map(Math.abs));
    const cx = w * 0.5,
      cy = h * 0.48,
      rx = Math.min(w * 0.35, h * 0.48),
      ry = h * 0.35;
    // Anatomical visual metaphor only: all plotted nodes and edges are actual software owners/seams.
    c.save();
    c.translate(cx, cy);
    for (const side of [-1, 1]) {
      c.save();
      c.scale(side, 1);
      const gradient = c.createRadialGradient(rx * 0.4, 0, 0, rx * 0.4, 0, rx);
      gradient.addColorStop(0, "#39454c55");
      gradient.addColorStop(1, "#a5b4bc08");
      c.fillStyle = gradient;
      c.strokeStyle = "#9daeb04d";
      c.lineWidth = 1.1;
      c.beginPath();
      c.moveTo(rx * 0.05, -ry * 0.8);
      c.bezierCurveTo(
        rx * 0.22,
        -ry * 1.18,
        rx * 0.86,
        -ry * 1.1,
        rx,
        -ry * 0.35,
      );
      c.bezierCurveTo(
        rx * 1.18,
        ry * 0.3,
        rx * 0.56,
        ry * 1.13,
        rx * 0.14,
        ry * 0.9,
      );
      c.bezierCurveTo(
        -rx * 0.03,
        ry * 0.65,
        rx * 0.07,
        -ry * 0.4,
        rx * 0.05,
        -ry * 0.8,
      );
      c.fill();
      c.stroke();
      c.strokeStyle = "#a9b4b51e";
      for (let k = 0; k < 7; k++) {
        const y = -ry * 0.7 + k * ry * 0.22;
        c.beginPath();
        c.moveTo(rx * 0.15, y);
        c.bezierCurveTo(
          rx * 0.52,
          y - ry * 0.17,
          rx * 0.22,
          y + ry * 0.25,
          rx * (0.84 - 0.08 * Math.abs(k - 3)),
          y + ry * 0.05,
        );
        c.stroke();
      }
      c.restore();
    }
    c.restore();
    const groupNames = [...new Set(this.source.groups ?? ["patch"])],
      counts = {},
      totals = {};
    (this.source.groups ?? s.map(() => "patch")).forEach(
      (g) => (totals[g] = (totals[g] ?? 0) + 1),
    );
    const positions = s.map((v, i) => {
      const g = this.source.groups?.[i] ?? "patch",
        gi = groupNames.indexOf(g),
        k = counts[g] ?? 0;
      counts[g] = k + 1;
      const z =
        groupNames.length === 1
          ? 0
          : (gi / (groupNames.length - 1) - 0.5) * 1.15;
      const a = k * 2.399963,
        radius = Math.sqrt((k + 0.5) / totals[g]),
        side = k % 2 ? 1 : -1;
      const y = z + 0.3 * radius * Math.sin(a);
      return [
        cx +
          side *
            rx *
            (0.5 +
              0.38 * radius * Math.cos(a) * Math.sqrt(Math.max(0, 1 - y * y))),
        cy + ry * y,
      ];
    });
    const plastic = this.signal === "plasticity";
    this.source.edges.forEach(([a, b, weight], i) => {
      if (this.source.mask?.[a] === 0 || this.source.mask?.[b] === 0) return;
      const delta =
          (this.changes[i] ?? 0) *
          Math.max(0, 1 - (this.age - (this.flashAt ?? -10)) / 1.5),
        signal = plastic
          ? Math.min(1, Math.abs(delta))
          : Math.min(1, (Math.abs(s[a]) / max) * Math.abs(weight));
      c.strokeStyle = plastic
        ? delta < 0
          ? `rgba(125,176,255,${0.035 + Math.min(1, Math.abs(weight)) * 0.2 + signal * 0.75})`
          : `rgba(255,190,98,${0.035 + Math.min(1, Math.abs(weight)) * 0.2 + signal * 0.75})`
        : `rgba(160,202,225,${0.018 + signal * 0.17})`;
      c.lineWidth = plastic
        ? 0.6 + Math.min(1, Math.abs(weight)) + Math.abs(delta)
        : 0.6;
      c.beginPath();
      c.moveTo(...positions[a]);
      c.lineTo(...positions[b]);
      c.stroke();
    });
    const groupMax = {};
    scaleValues.forEach((v, i) => {
      const g = this.source.groups?.[i] ?? "patch";
      groupMax[g] = Math.max(groupMax[g] ?? 1e-12, Math.abs(v));
    });
    const heatColor = (v) => (v < 0 ? "#79b8ff" : "#ffc471");
    const normalized = values.map((v, i) =>
      Math.max(
        -1,
        Math.min(1, v / groupMax[this.source.groups?.[i] ?? "patch"]),
      ),
    );
    if (this.heat && !plastic) {
      positions.forEach(([x, y], i) => {
        if (this.source.mask?.[i] === 0 || Math.abs(normalized[i]) < 1e-6)
          return;
        const radius = rx * (s.length < 20 ? 0.22 : 0.08);
        const intensity = Math.abs(normalized[i]);
        const rgb = normalized[i] < 0 ? "121,184,255" : "255,196,113";
        const glow = c.createRadialGradient(x, y, 0, x, y, radius);
        glow.addColorStop(0, `rgba(${rgb},${intensity * 0.42})`);
        glow.addColorStop(1, `rgba(${rgb},0)`);
        c.fillStyle = glow;
        c.fillRect(x - radius, y - radius, radius * 2, radius * 2);
      });
    }
    let hovered = -1,
      nearest = 16;
    positions.forEach(([x, y], i) => {
      const value = Math.abs(normalized[i]);
      const muted = this.source.mask?.[i] === 0;
      c.fillStyle = muted
        ? "#653647"
        : (this.heat && !plastic) ||
            this.signal === "repair" ||
            this.signal === "input"
          ? heatColor(values[i])
          : "#96f0cc";
      c.globalAlpha = muted ? 0.35 : 0.2 + 0.8 * value;
      c.shadowColor = c.fillStyle;
      c.shadowBlur = value > 0.2 ? 12 * value : 0;
      c.beginPath();
      c.arc(
        x,
        y,
        s.length < 20 ? 4 + value * 7 : 1.3 + value * 2.2,
        0,
        Math.PI * 2,
      );
      c.fill();
      if (this.hover) {
        const d = Math.hypot(x - this.hover[0], y - this.hover[1]);
        if (d < nearest) {
          nearest = d;
          hovered = i;
        }
      }
    });
    c.globalAlpha = 1;
    c.shadowBlur = 0;
    c.font = "10px ui-monospace, monospace";
    c.fillStyle = "#91a6b1";
    c.textAlign = "left";
    c.fillText("SCHEMATIC / MRI-INSPIRED", 14, 22);
    c.textAlign = "right";
    c.fillText(this.trace ? "DIAGNOSTIC REPLAY" : "LIVE STATE", w - 14, 22);
    c.textAlign = "left";
    if (hovered >= 0)
      c.fillText(
        `${this.source.names?.[hovered] ?? `Owner ${hovered}`} · ${values[hovered].toExponential(3)} ${this.signal === "repair" ? "change" : this.signal === "input" ? "drive" : "state"}`,
        14,
        h - 38,
      );
    const series = this.trace
      ? this.trace.peaks.slice(0, this.frame + 1)
      : this.history;
    const peak = Math.max(1e-12, ...series);
    c.strokeStyle = "#91d0d7";
    c.lineWidth = 1;
    c.beginPath();
    series.forEach((v, i) => {
      const x = 14 + (i * (w - 28)) / Math.max(149, series.length - 1),
        y = h - 10 - (20 * v) / peak;
      i ? c.lineTo(x, y) : c.moveTo(x, y);
    });
    c.stroke();
    this.$("brain-status").textContent = this.trace
      ? `${this.trace.release ? "Input released in an isolated copy" : "Repair replay"} · iteration ${this.frame}/${this.trace.frames.length - 1} · ${this.rate} iterations/s${this.frame === this.trace.frames.length - 1 ? " · complete" : ""}`
      : `Live activity · ${this.source.recurrent ? "recurrent settlement" : "direct associative read/write"} · ${this.writes} observed weight changes`;
    const legend = this.$("brain-legend");
    legend.hidden = !this.heat || plastic;
    this.$("brain-legend-label").textContent =
      `${this.signal === "input" ? "Input drive" : this.signal === "repair" ? "State change" : "Activity"} · relative region scale${this.trace?.release && this.signal === "activity" ? " · fixed during decay" : ""}`;
    const quantity = {
      activity: "State",
      repair: "State change",
      input: "Input drive",
      plasticity: "State (edges show learned weights)",
    }[this.signal];
    this.$("brain-scale").textContent =
      `${quantity} max |value| ${Math.max(0, ...values.map(Math.abs)).toExponential(2)} · dimensionless model units. Colors use ${this.trace?.release && this.signal === "activity" ? "captured region scales (fixed during decay)" : "current region scales"}; hover for signed values. No neurotransmitter concentrations are modeled.`;
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
    };
  }
}
