import { mountGame } from "../connect-four/view.js";
import { Forager } from "../fly/brain.js";
import { MotorGate } from "./motor_gate.js";
import { BrainView } from "./brain_view.js";
import { mountWorm } from "../worm/view.js";
import { showGuide } from "./guides.js";
import { mountArm } from "../eye-arm/view.js";
import { mountEmbodied } from "../mouse/view.js";
import {
  Worm,
  MLP,
  SynapticMemory,
  flowers,
  zeros,
} from "./engine.js";
const $ = (id) => document.getElementById(id),
  colors = [
    "#87ecc2",
    "#ffb17a",
    "#bfa8ff",
    "#8acfff",
    "#ff88af",
    "#fff1a7",
    "#92caa0",
    "#deb6aa",
  ];
const pct = (x) => (100 * x).toFixed(1) + "%",
  sci = (x) => x.toExponential(1),
  mean = (a) => a.reduce((x, y) => x + y, 0) / a.length;
let composite,
  bodyView,
  wormView = "habitat";
let data,
  evidence,
  memoryRuntime,
  strategyEvidence,
  worm,
  net,
  mode = "worm",
  frame = 0,
  paused = false,
  updates = 1;
let field,
  agents,
  mask,
  drive,
  result,
  mlpState,
  points = [],
  drag = -1;
const brainView = new BrainView();
const motorGate = new MotorGate();
function act(make) {
  if (motorGate.prepare(() => {
    const commit = make();
    brainView.update(brainSource());
    return commit;
  }, () => brainSource().steps ?? 24)) {
    brainView.trace = null;
    brainView.update(brainSource());
    brainView.replay(false);
  }
}
function primaryControls(ids) {
  const bar = document.createElement("div");
  bar.className = "buttons primary-controls";
  bar.setAttribute("aria-label", "Quick demo controls");
  for (const id of ids) {
    const control = $(id);
    if (control) bar.append(control);
  }
  canvas.after(bar);
}
$("brain-think").onclick = () => {
  motorGate.enabled = !motorGate.enabled;
  motorGate.cancel();
  brainView.trace = null;
  $("brain-think").setAttribute("aria-pressed", String(motorGate.enabled));
  $("brain-think").textContent = motorGate.enabled
    ? "Inspect before moving"
    : "Slow thought";
};
// An edited world invalidates its pending action.
for (const event of ["pointerdown", "click", "change", "keydown"]) {
  document.addEventListener(
    event,
    (e) => {
      if (!e.target.closest(".brain-panel")) {
        motorGate.cancel();
        brainView.trace = null;
      }
    },
    true,
  );
}

const canvas = $("scene"),
  ctx = canvas.getContext("2d");
let width = 900,
  height = 430;
function fit() {
  const box = canvas.getBoundingClientRect();
  width = box.width;
  height = box.height;
  const dpr = Math.min(devicePixelRatio, 2);
  canvas.width = width * dpr;
  canvas.height = height * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  layout();
}
function layout() {
  if (!data) return;
  let counters = { sensory: 0, interneuron: 0, motor: 0 };
  let totals = { sensory: 0, interneuron: 0, motor: 0 };
  data.groups.forEach((g) => totals[g]++);
  points = data.groups.map((g, i) => {
    const k = counters[g]++,
      a = k * 2.39996;
    const r = Math.sqrt((k + 0.5) / totals[g]);
    const cx = { sensory: 0.18, interneuron: 0.5, motor: 0.82 }[g];
    return [
      width * (cx + Math.cos(a) * r * 0.14),
      height * (0.46 + Math.sin(a) * r * 0.33),
    ];
  });
}
function metrics(rows) {
  return (
    '<div class="metrics">' +
    rows
      .map(
        ([label, v, id]) =>
          `<div class="metric"><span>${label}</span><strong${id ? ` id="${id}"` : ""}>${v}</strong></div>`,
      )
      .join("") +
    "</div>"
  );
}
function explain(items) {
  $("explain").innerHTML = items
    .map(
      ([title, copy], i) =>
        `<div><span class="number">0${i + 1}</span><h3>${title}</h3><p>${copy}</p></div>`,
    )
    .join("");
}
function table(head, rows) {
  return (
    "<table><thead><tr>" +
    head.map((t) => `<th scope="col">${t}</th>`).join("") +
    "</tr></thead><tbody>" +
    rows
      .map(
        (r) =>
          "<tr>" +
          r
            .map((v, i) => `<td${i === 1 ? ' class="good"' : ""}>${v}</td>`)
            .join("") +
          "</tr>",
      )
      .join("") +
    "</tbody></table>"
  );
}
function resetFly() {
  motorGate.cancel();
  field = flowers();
  agents = [
    new Forager(new SynapticMemory(), -0.02),
    new Forager(new MLP(evidence.browser.online_mlp), 0.02, updates),
  ];
  paused = false;
  if ($("pause")) $("pause").textContent = "Pause";
}
function solve() {
  const t = performance.now();
  result = worm.settle(drive, mask);
  result.ms = performance.now() - t;
  mlpState = worm.surrogate(net, drive, mask);
  const norm = Math.sqrt(result.state.reduce((s, v) => s + v * v, 0));
  const error =
    Math.sqrt(result.state.reduce((s, v, i) => s + (v - mlpState[i]) ** 2, 0)) /
    Math.max(norm, 1e-12);
  if ($("residual")) {
    $("residual").textContent = sci(result.residual);
    $("mlp-error").textContent = pct(error);
    $("lesions").textContent = mask.filter((v) => !v).length;
    $("steps").textContent = result.steps;
  }
  $("stage-readout").textContent =
    `${result.ms.toFixed(1)} ms · measured in this browser`;
}
function renderEvidence() {
  if (mode === "worm") {
    const conditions = ["intact", "new_mixtures", "many_lesions"];
    $("evidence-title").textContent = "Reuse the mechanism. Skip the refit.";
    $("evidence-note").textContent =
      "Three seeds, 128 held-out queries per condition. Mean relative whole-state error against an independent converged solver. The MLP receives current drives, lesion masks and an exact first propagation.";
    $("evidence-table").innerHTML = table(
      ["New situation", "Cadence", "Trained MLP", "32-step recurrence"],
      conditions.map((c, i) => {
        const r = evidence.worm.flatMap((s) =>
          s.rows.filter((x) => x.condition === c),
        );
        return [
          [
            "Intact circuit",
            "New sensory mixtures",
            "Up to 64 removed neurons",
          ][i],
          ...["cadence", "mlp", "unrolled32"].map((a) =>
            pct(mean(r.map((x) => x.arms[a].relative_rmse))),
          ),
        ];
      }),
    );
    $("boundary").textContent =
      "The chemical topology comes from public C. elegans data. Positive normalized weights and a contractive tanh rule are supplied. This is a numerical circuit model, not validated worm physiology or locomotion. A conventional tied recurrence also solves it. The trained surrogate is faster per query but approximate; full runtimes and training costs are in the evidence. No model is retrained when you remove a neuron.";
  } else {
    $("evidence-title").textContent =
      mode === "fly"
        ? "The memory inside the behavior"
        : "One write can replace an old answer";
    $("evidence-note").textContent =
      "Same revealed samples: 128 writes, eight keys, four values, three seeds. Query every seen key after each write. Online MLP: 8 → 32 → 4, SGD; no test-based tuning. This is a controlled memory comparison, separate from the live foraging trajectories.";
    const timing = memoryRuntime.rows,
      local = timing.find((r) => r.kind === "cadence"),
      one = timing.find((r) => r.kind === "1"),
      hundred = timing.find((r) => r.kind === "100");
    $("evidence-note").textContent +=
      ` Separate local runtime trial: ${local.median_ms.toFixed(3)} ms per distinct-key stream; ${(hundred.median_ms / local.median_ms).toFixed(1)}× lower time than MLP/100 updates and ${(one.median_ms / local.median_ms).toFixed(1)}× than MLP/1 update. Recorded on ${memoryRuntime.cpu}, Node ${memoryRuntime.runtime}; 11 repetitions, one stream. See Comparison contracts for the receipt and limits. The live consolidating kernel is measured separately at ${timing.find(r => r.kind === "consolidating").median_ms.toFixed(3)} ms; both kernel timings exclude graded readout, bodies and visualization.`;
    $("evidence-table").innerHTML = table(
      [
        "Key similarity",
        "Cadence",
        "MLP · 1 update",
        "MLP · 10",
        "MLP · 100",
        "Exact lookup",
      ],
      [0, 0.5, 0.9].map((c) => {
        const rows = evidence.memory.filter((r) => r.correlation === c);
        return [
          c === 0 ? "Distinct keys" : `Cosine ${c}`,
          ...["cadence", "mlp1", "mlp10", "mlp100", "dictionary"].map((a) =>
            pct(mean(rows.map((r) => r.arms[a].accuracy))),
          ),
        ];
      }),
    );
    $("boundary").textContent =
      "The reference kernel uses 32 fast-memory values and a residual write. The mouse and fly add 32 persistent weights, for 64 stored values across the same 32 memory contacts. The MLP uses 420 parameters and 1, 10 or 100 SGD updates on the identical new sample. More MLP work is not a guarantee of retaining old associations. Explicit keys also admit exact dictionary storage, which succeeds. Correlated keys interfere with residual memory. The fly is a simplified 2D body with supplied visual ports, steering and exploration; it is not a FlyWire reconstruction. Each live agent collects its own encounters, so live nectar totals are illustrative, not a matched benchmark.";
  }
}
function setMode(next) {
  mode = next;
  brainView.reset();
  motorGate.cancel();
  $("brain-think").disabled = mode === "game";
  showGuide($("demo-guide"), mode);
  $("worm-views").hidden = mode !== "worm";
  document
    .querySelectorAll("[data-worm-view]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.wormView === wormView)),
    );
  document.title = `Cadence · ${{ mouse: "Teachable mouse", arm: "Eye & arm", fly: "Embodied forager", worm: "C. elegans", game: "Connect Four reasoner" }[mode]}`;
  bodyView = null;
  document.querySelector(".task-commands")?.remove();
  document.querySelector(".primary-controls")?.remove();
  paused = false;
  if (!document.body.dataset.demo) location.hash = next;
  document
    .querySelectorAll("[data-tab]")
    .forEach((b) =>
      b.setAttribute("aria-pressed", String(b.dataset.tab === next)),
    );
  if (mode === "worm" && wormView === "habitat") {
    bodyView = mountWorm({ data, $, ctx, metrics, explain, act });
    primaryControls(["worm-tools", "worm-pause"]);
    document.querySelector(".evidence details a").href =
      "evidence/evidence.json";
    renderEvidence();
    fit();
    return;
  }
  if (mode === "game") {
    bodyView = mountGame({
      $,
      ctx,
      metrics,
      explain,
      table,
      evidence: strategyEvidence,
    });
    primaryControls(["game-columns", "watch-thought", "new-game"]);
    fit();
    return;
  }
  if (mode === "mouse" || mode === "arm") {
    bodyView = (mode === "arm" ? (_, api) => mountArm(api) : mountEmbodied)(
      mode,
      {
        $,
        metrics,
        explain,
        table,
        pct,
        mean,
        ctx,
        circle,
        line,
        text,
        act,
        observe: source => brainView.capture(source),
        resume: () => {
          motorGate.cancel();
          motorGate.enabled = false;
          $("brain-think").setAttribute("aria-pressed", "false");
          $("brain-think").textContent = "Slow thought";
          brainView.trace = brainView.auto = brainView.pending = null;
          brainView.frozen = false;
          $("brain-freeze").textContent = "Pause view";
        },
        evidence: composite,
      },
    );
    if (mode === "arm") primaryControls(["clear-pad", "restart-arm", "pause"]);
    fit();
    return;
  }
  document.querySelector(".evidence details a").href = "evidence/evidence.json";
  if (mode === "worm") {
    $("headline").innerHTML = "Change the circuit.<br>Keep the mechanism.";
    $("intro").textContent =
      "Stimulate a real chemical connectome. Remove neurons. Watch local feedback produce a new answer, without another training run.";
    $("stage-label").textContent =
      "LIVE CIRCUIT · C. ELEGANS CHEMICAL TOPOLOGY";
    $("stage-detail").textContent =
      `${data.names.length} neurons · ${data.edges.length.toLocaleString()} synapses`;
    $("stage-hint").textContent =
      "Click a neuron to remove it. Use the selector for keyboard access.";
    canvas.setAttribute(
      "aria-label",
      "Worm chemical network. Green sensory neurons, orange interneurons, pale motor neurons. Activity brightness changes with stimulation.",
    );
    $("controls").innerHTML =
      `<div><h2>Ask the circuit a new question.</h2><p>Local activity settles through measured connections and a supplied rule.</p></div><div><label>Stimulate</label><div class="buttons">${Object.keys(
        data.stimuli,
      )
        .filter((k) => data.stimuli[k].length)
        .map(
          (k, i) =>
            `<button data-stim="${k}" aria-pressed="${i === 0}">${{ anterior: "Front touch", posterior: "Tail touch", nose: "Nose", odor: "Odor" }[k]}</button>`,
        )
        .join(
          "",
        )}</div></div><div><label for="amplitude">Input strength</label><input id="amplitude" type="range" min="0" max="2" step=".05" value="1"></div><div><label for="lesion-node">Remove / restore a neuron</label><select id="lesion-node">${data.names.map((n, i) => `<option value="${i}">${n}</option>`).join("")}</select><div class="buttons" style="margin-top:8px"><button id="toggle-node">Toggle neuron</button><button id="cut-many">Remove 32</button><button id="restore">Restore all</button></div></div>${metrics(
        [
          ["Removed neurons", "0", "lesions"],
          ["Settling steps", "0", "steps"],
          ["Remaining discrepancy", "0", "residual"],
          ["MLP approximation error", "0", "mlp-error"],
        ],
      )}`;
    mask = Array(data.names.length).fill(1);
    drive = zeros(data.names.length);
    data.stimuli.anterior.forEach((i) => (drive[i] = 1));
    const refresh = () => {
      drive.fill(0);
      document
        .querySelectorAll("[data-stim][aria-pressed=true]")
        .forEach((b) =>
          data.stimuli[b.dataset.stim].forEach(
            (i) => (drive[i] = +$("amplitude").value),
          ),
        );
      solve();
    };
    document.querySelectorAll("[data-stim]").forEach(
      (b) =>
        (b.onclick = () => {
          b.setAttribute(
            "aria-pressed",
            b.getAttribute("aria-pressed") !== "true",
          );
          refresh();
        }),
    );
    $("amplitude").oninput = refresh;
    $("toggle-node").onclick = () => {
      const i = +$("lesion-node").value;
      mask[i] = 1 - mask[i];
      solve();
    };
    $("cut-many").onclick = () => {
      for (let k = 0; k < 32; k++) mask[(k * 47 + 13) % mask.length] = 0;
      solve();
    };
    $("restore").onclick = () => {
      mask.fill(1);
      solve();
    };
    explain([
      [
        "A measured topology",
        "Public worm connections define who can influence whom. The transfer rule and effective weights are explicit modeling choices.",
      ],
      [
        "A local settling process",
        "Neurons combine synaptic input and update their own state. Remove a relay and the surviving network settles again.",
      ],
      [
        "An inspectable answer",
        "Compare the fixed-point discrepancy and the trained surrogate. The conventional recurrent control stays in the evidence table.",
      ],
    ]);
    solve();
  } else if (mode === "fly") {
    $("headline").innerHTML = "An encounter becomes<br>a better next choice.";
    $("intro").textContent =
      "A body moves. Sensors report. A small memory changes. Follow a fly-inspired agent as it discovers which flowers are worth returning to.";
    $("stage-label").textContent = "LIVE EMBODIMENT · FLY-INSPIRED 2D FORAGER";
    $("stage-detail").textContent = "Vision → memory → steering → encounter";
    $("stage-hint").textContent =
      "Drag a flower. Change the nectar. The next encounter teaches the agent.";
    canvas.setAttribute(
      "aria-label",
      "Two fly-inspired agents move between flowers: green uses Cadence consolidating synapses, orange uses an online MLP.",
    );
    $("controls").innerHTML =
      `<div><h2>Learning stays inside the loop.</h2><p>Flower type and position are visual ports. Nectar is revealed only on contact. Body and steering are supplied.</p></div><div><label for="update-budget">Orange MLP: updates per encounter</label><select id="update-budget"><option value="1">1 SGD update</option><option value="10">10 SGD updates</option><option value="100">100 SGD updates</option></select></div><div class="buttons"><button id="change-nectar">Change nectar</button><button id="pause">Pause</button><button id="reset-fly">Reset</button></div>${metrics(
        [
          ["Cadence · nectar", "0", "fast-nectar"],
          ["MLP · nectar", "0", "slow-nectar"],
          ["Cadence · encounters", "0", "encounters"],
        ],
      )}<div><label>Latest experience</label><p id="last-event" aria-live="polite">Waiting for the first encounter…</p><p class="note">Live totals reflect different trajectories. The matched memory experiment is below.</p></div>`;
    $("update-budget").value = String(updates);
    primaryControls(["change-nectar", "pause", "reset-fly"]);
    $("update-budget").onchange = () => {
      updates = +$("update-budget").value;
      resetFly();
    };
    $("change-nectar").onclick = () => {
      field.forEach((f) => (f.value = 3 - f.value));
      $("last-event").textContent =
        "Nectar changed. Agents must discover it through encounters.";
    };
    $("pause").onclick = () => {
      paused = !paused;
      if (paused) motorGate.cancel();
      $("pause").textContent = paused ? "Resume" : "Pause";
    };
    $("reset-fly").onclick = resetFly;
    resetFly();
    explain([
      [
        "See and approach",
        "The visual adapter supplies flower types and positions. A declared motor controller steers the body toward the selected target.",
      ],
      [
        "Remember the encounter",
        "Contact reveals nectar. Fast synapses compare prediction with observation and write only the residual. No offline fitting is required.",
      ],
      [
        "Close the loop",
        "Movement changes the next observation. Changed nectar can revise the next preference. As observations age, a supplied exploration rule revisits neglected flowers—even ones that used to have little nectar.",
      ],
    ]);
  }
  renderEvidence();
  fit();
}
function line(a, b, color, weight = 1) {
  ctx.strokeStyle = color;
  ctx.lineWidth = weight;
  ctx.beginPath();
  ctx.moveTo(...a);
  ctx.lineTo(...b);
  ctx.stroke();
}
function circle(x, y, r, color) {
  ctx.fillStyle = color;
  ctx.beginPath();
  ctx.arc(x, y, r, 0, Math.PI * 2);
  ctx.fill();
}
function text(t, x, y, color = "#a3b4b9", size = 11, align = "left") {
  ctx.font = `${size}px -apple-system, sans-serif`;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.fillText(t, x, y);
}
function drawWorm() {
  const s = result.state;
  for (let e = 0; e < data.edges.length; e += 2) {
    const [a, b, w] = data.edges[e];
    const alpha = Math.min(0.38, 0.025 + s[a] * 0.75);
    line(
      points[a],
      points[b],
      `rgba(135,236,194,${alpha})`,
      s[a] > 0.2 ? 1 : 0.6,
    );
  }
  points.forEach(([x, y], i) => {
    const active = s[i],
      c = !mask[i]
        ? "#ff7398"
        : { sensory: colors[0], interneuron: colors[1], motor: "#e3eee6" }[
            data.groups[i]
          ];
    if (active > 0.08)
      circle(x, y, 5 + active * 8, `rgba(135,236,194,${active * 0.17})`);
    ctx.globalAlpha = mask[i] ? 0.23 + Math.min(0.77, active * 2) : 0.95;
    circle(x, y, !mask[i] ? 3.5 : 2 + active * 3, c);
    ctx.globalAlpha = 1;
  });
  text("SENSORY", width * 0.18, height * 0.9, colors[0], 10, "center");
  text("INTERNEURONS", width * 0.5, height * 0.9, colors[1], 10, "center");
  text("MOTOR READOUT", width * 0.82, height * 0.9, "#e3eee6", 10, "center");
  text(
    "Node positions are a schematic layout; brightness is modeled activity.",
    width / 2,
    height * 0.97,
    "#81979b",
    10,
    "center",
  );
}
function drawFly() {
  const pad = 35,
    scaleX = width - 2 * pad,
    scaleY = height - 2 * pad,
    xy = (x, y) => [pad + x * scaleX, pad + y * scaleY];
  for (let x = pad; x < width; x += 35)
    for (let y = pad; y < height; y += 35) circle(x, y, 0.7, "#254039");
  field.forEach((f, i) => {
    const [x, y] = xy(f.x, f.y);
    for (let p = 0; p < 5; p++)
      circle(
        x + Math.cos(p * 1.256) * 7,
        y + Math.sin(p * 1.256) * 7,
        6,
        colors[i] + "80",
      );
    circle(x, y, 5, colors[i]);
    text(String(i + 1), x, y + 26, "#acbdb7", 10, "center");
  });
  agents.forEach((a, i) => {
    const c = i ? colors[1] : colors[0];
    for (let k = 1; k < a.path.length; k++) {
      ctx.globalAlpha = (k / a.path.length) * 0.3;
      line(xy(...a.path[k - 1]), xy(...a.path[k]), c, 1.3);
    }
    ctx.globalAlpha = 1;
    const [x, y] = xy(a.x, a.y);
    if (a.target >= 0) {
      ctx.setLineDash([3, 6]);
      line([x, y], xy(field[a.target].x, field[a.target].y), c + "60");
      ctx.setLineDash([]);
    }
    ctx.save();
    ctx.translate(x, y);
    ctx.rotate(a.angle);
    ctx.fillStyle = c;
    ctx.shadowColor = c;
    ctx.shadowBlur = 12;
    ctx.beginPath();
    ctx.ellipse(0, 0, 10, 4, 0, 0, Math.PI * 2);
    ctx.fill();
    ctx.shadowBlur = 0;
    const wing = 7 + Math.sin(frame * 1.8 + i) * 3;
    ctx.fillStyle = c + "45";
    for (const side of [-1, 1]) {
      ctx.beginPath();
      ctx.ellipse(-2, side * wing, 11, 4, side * 0.5, 0, Math.PI * 2);
      ctx.fill();
      for (let k = 0; k < 3; k++)
        line([-5 + k * 5, side * 3], [-9 + k * 7, side * 9], c, 1);
    }
    circle(8, 0, 4, c);
    line([10, -2], [18, -7], c);
    line([10, 2], [18, 7], c);
    ctx.restore();
    text(i ? "MLP" : "CADENCE", x, y - 22, c, 10, "center");
  });
  $("fast-nectar").textContent = agents[0].nectar;
  $("slow-nectar").textContent = agents[1].nectar;
  $("encounters").textContent = agents[0].encounters;
  const last = agents[0].last;
  if (last && agents[0].time - last.time < 0.1)
    $("last-event").textContent =
      `Flower ${last.kind + 1}: expected ${last.predicted}, observed ${last.value}. Memory updated.`;
  $("stage-readout").textContent =
    `${agents[0].time.toFixed(1)} simulated seconds · ${paused ? "paused" : "closed loop"}`;
}
function brainSource() {
  if (bodyView) return bodyView.brain?.();
  if (mode === "worm")
    return {
      adapters: "Supplied: stimulus ports · normalized chemical weights",
      regionLabels: {
        sensory: "Sensory input",
        interneuron: "Interneurons",
        motor: "Motor output",
      },
      behavior: { label: "Circuit probe", tone: "neutral" },
      state: result.state,
      drive,
      mask,
      edges: data.edges,
      names: data.names,
      groups: data.groups,
      recurrent: true,
      steps: result.steps,
      memory:
        "Supplied chemical circuit: no trained plasticity. Release input shows transient recurrent decay, not durable task memory.",
    };
  if (mode === "fly") {
    const agent = agents[0];
    const circuit = agent.brain(field);
    const recent = agent.last && agent.time - agent.last.time < 1.5;
    circuit.behavior = paused
      ? { label: "Paused", tone: "neutral" }
      : recent
        ? {
            label: agent.last.value > 0 ? "Nectar received" : "Empty flower",
            tone: agent.last.value > 0 ? "positive" : "correcting",
          }
        : { label: "Seeking nectar", tone: "seeking" };
    return circuit;
  }
  return null;
}
let previous = 0;
function animate(t) {
  const dt = Math.min(0.05, (t - previous) / 1000 || 0.016);
  previous = t;
  frame++;
  if (!paused && !brainView.frozen && motorGate.advance(dt * brainView.rate)) {
    brainView.trace = null;
    brainView.auto = null;
  }
  ctx.clearRect(0, 0, width, height);
  if (bodyView) bodyView.draw(width, height, dt);
  else if (mode === "worm") drawWorm();
  else if (mode === "fly") {
    if (!paused) {
      if (!motorGate.busy)
        act(() => {
          const moves = agents.map((a) => a.step(field, dt, true));
          return () => moves.forEach((move) => move?.());
        });
    }
    drawFly();
  }
  if (frame % 6 === 0) brainView.update(brainSource());
  brainView.draw(dt);
  if (motorGate.busy) {
    $("brain-behavior").textContent = "Thinking · motor output held";
    $("brain-status").textContent =
      `Time-expanded settling · ${Math.ceil(motorGate.remaining)} iterations before movement`;
  }
  requestAnimationFrame(animate);
}
canvas.addEventListener("pointerdown", (e) => {
  const r = canvas.getBoundingClientRect(),
    x = e.clientX - r.left,
    y = e.clientY - r.top;
  if (bodyView) {
    canvas.setPointerCapture(e.pointerId);
    bodyView.pointer(x, y, width, height);
    return;
  }
  if (mode === "worm") {
    let best = -1,
      d = 14;
    points.forEach(([a, b], i) => {
      const q = Math.hypot(x - a, y - b);
      if (q < d) {
        d = q;
        best = i;
      }
    });
    if (best >= 0) {
      mask[best] = 1 - mask[best];
      $("lesion-node").value = best;
      solve();
    }
  } else if (mode === "fly") {
    drag = field.findIndex(
      (f) =>
        Math.hypot(
          x - (35 + f.x * (width - 70)),
          y - (35 + f.y * (height - 70)),
        ) < 22,
    );
    if (drag >= 0) canvas.setPointerCapture(e.pointerId);
  }
});
canvas.addEventListener("pointermove", (e) => {
  if (bodyView?.pointerMove) {
    const r = canvas.getBoundingClientRect();
    bodyView.pointerMove(e.clientX - r.left, e.clientY - r.top, width, height);
    return;
  }
  if (drag < 0 || mode !== "fly") return;
  const r = canvas.getBoundingClientRect();
  field[drag].x = Math.max(
    0.05,
    Math.min(0.95, (e.clientX - r.left - 35) / (width - 70)),
  );
  field[drag].y = Math.max(
    0.05,
    Math.min(0.95, (e.clientY - r.top - 35) / (height - 70)),
  );
});
canvas.addEventListener("pointerup", () => {
  drag = -1;
  bodyView?.pointerEnd?.();
});
canvas.addEventListener("pointercancel", () => {
  drag = -1;
  bodyView?.pointerEnd?.();
});
canvas.addEventListener("keydown", (e) => bodyView?.key?.(e));
document.querySelectorAll("[data-worm-view]").forEach(
  (b) =>
    (b.onclick = () => {
      wormView = b.dataset.wormView;
      setMode("worm");
    }),
);
window.addEventListener("resize", fit);
document.querySelectorAll("[data-tab]").forEach((b) => {
  if (b.tagName === "BUTTON") b.onclick = () => setMode(b.dataset.tab);
});
window.addEventListener("hashchange", () => {
  const next = location.hash.slice(1);
  if (["worm", "fly", "mouse", "arm", "game"].includes(next) && mode !== next)
    setMode(next);
});
try {
  [
    data,
    evidence,
    composite,
    memoryRuntime,
    strategyEvidence,
    ] = await Promise.all(
    [
      "worm/worm.json",
      "evidence/evidence.json",
      "evidence/composite_evidence.json",
      "benchmarks/memory/evidence.json",
      "connect-four/evidence.json",
    ].map(async (url) => {
      const r = await fetch(url);
      if (!r.ok) throw Error(url);
      return r.json();
    }),
  );
  worm = new Worm(data);
  net = new MLP(evidence.browser.worm_mlp);
  setMode(
    document.body.dataset.demo ||
      (["worm", "fly", "mouse", "arm", "game"].includes(
        location.hash.slice(1),
      )
        ? location.hash.slice(1)
        : "mouse"),
  );
  requestAnimationFrame(animate);
  // Read-only snapshots support reproducible engine/UI checks.
  window.showcase = {
    snapshot: () => ({
      mode,
      paused,
      brain: brainView.snapshot(),
      body: bodyView?.snapshot(),
      mask: mask?.slice(),
      drive: drive?.slice(),
      result,
      mlpState,
      agents: agents?.map((a) => ({
        x: a.x,
        y: a.y,
        encounters: a.encounters,
        nectar: a.nectar,
      })),
    }),
  };
} catch (error) {
  $("controls").innerHTML =
    '<div class="error">Could not load the model. Run <code>python serve.py</code> from the repository and reload this page.</div>';
  console.error(error);
}
