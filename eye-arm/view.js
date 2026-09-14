import { DrawingArm, RETINA } from "./brain.js";
import { PAPER_SIZE } from "./paper.js";
import { forward } from "../shared/embodied.js";

export function mountArm(api) {
  const { $, ctx, metrics, explain, table, pct } = api;
  const pad = document.createElement("canvas");
  pad.width = pad.height = 192;
  const pen = pad.getContext("2d", { willReadFrequently: true });
  const copy = document.createElement("canvas");
  copy.width = copy.height = PAPER_SIZE;
  const copyContext = copy.getContext("2d"),
    copyImage = copyContext.createImageData(PAPER_SIZE, PAPER_SIZE);
  let paperRevision = -1;
  let arm,
    paused = false,
    painting = false,
    last = null,
    credit = 0,
    eyeOn = true,
    revision = 0;
  const layout = (w, h) => {
    const size = Math.min(w * 0.4 - 18, h - 88);
    return {
      x: 18,
      y: 58,
      size,
      bodySize: Math.min(w * 0.57 - 22, h - 35),
      bodyX: w * 0.43,
      bodyY: 30,
    };
  };
  const blank = () => {
    pen.fillStyle = "#fff";
    pen.fillRect(0, 0, 192, 192);
  };
  const observe = () => {
    const small = document.createElement("canvas");
    small.width = small.height = RETINA;
    const c = small.getContext("2d");
    c.drawImage(pad, 0, 0, RETINA, RETINA);
    const p = c.getImageData(0, 0, RETINA, RETINA).data;
    const darkness = Array.from({ length: RETINA ** 2 }, (_, i) =>
      eyeOn ? 1 - (p[i * 4] + p[i * 4 + 1] + p[i * 4 + 2]) / 765 : 0,
    );
    arm = new DrawingArm(darkness);
    paperRevision = -1;
    $("stage-detail").textContent =
      `${arm.targets.length * 3 + 17} neurons · two joints + pencil lift`;
    revision++;
    $("image-status").textContent = arm.targets.length
      ? `${arm.targets.length} visible retinal targets. Following connected strokes and checking the ink.`
      : "Draw on the left pad to give the eye something to see.";
    $("feedback").textContent = "Feedback on";
    $("feedback").setAttribute("aria-pressed", "true");
    $("joint-motors").setAttribute("aria-pressed", "true");
    $("pencil-motors").setAttribute("aria-pressed", "true");
  };
  const example = (kind) => {
    blank();
    pen.strokeStyle = "#17252b";
    pen.lineWidth = 4;
    pen.lineCap = "round";
    pen.lineJoin = "round";
    pen.beginPath();
    for (let i = 0; i <= 180; i++) {
      const t = (i / 180) * Math.PI * 2,
        r =
          kind === "spiral"
            ? 7 + (58 * i) / 180
            : kind === "leaf"
              ? 62
              : 50 + 17 * Math.cos(5 * t);
      const a = kind === "spiral" ? t * 2.7 : t;
      const x = 96 + r * Math.cos(a) * (kind === "leaf" ? 0.65 : 1),
        y = 96 + r * Math.sin(a);
      i ? pen.lineTo(x, y) : pen.moveTo(x, y);
    }
    pen.stroke();
    observe();
  };
  $("headline").innerHTML = "Draw something.<br>Let the eye and arm copy it.";
  $("intro").textContent =
    "Draw on the left. The eye compares your marks with the actual ink. Motor neurons move the shoulder, elbow and pencil, following connected strokes and repairing missing marks.";
  $("stage-label").textContent = "LIVE SENSOR → BRAIN → MOTOR → BODY";
  $("stage-detail").textContent = "Two joints + pencil lift";
  $("stage-hint").textContent =
    "Draw with a mouse or finger on the left pad. Lift your finger to let the arm copy it.";
  $("scene").setAttribute(
    "aria-label",
    "Reference drawing pad on the left, an eye following its visual target, and a motor-driven arm drawing on the right. Use the outline selector or upload as keyboard alternatives.",
  );
  $("controls").innerHTML =
    `<div><h2>Your drawing is the instruction.</h2><p>Motor activity drives both joints and raises or lowers the pencil. Ink appears only when the pencil touches the paper.</p></div>
  <div><label for="target-image">Try an outline</label><select id="target-image"><option value="flower">Flower</option><option value="leaf">Leaf</option><option value="spiral">Spiral</option></select><div class="buttons" style="margin-top:8px"><button id="clear-pad">Clear pad</button><button id="restart-arm">Copy again</button><button id="erase-copy">Erase a patch</button></div></div>
  <div><label for="image-upload">Or show an image</label><input id="image-upload" type="file" accept="image/png,image/jpeg,image/webp"><p>Use a dark line drawing on a light background. The eye samples 48 × 48 pixels.</p></div>
  <div class="buttons"><button id="disturb">Disturb a joint</button><button id="pause">Pause</button><button id="feedback" aria-pressed="true">Feedback on</button></div>
  <div class="buttons"><button id="joint-motors" aria-pressed="true">Joint motors</button><button id="pencil-motors" aria-pressed="true">Pencil motors</button><button id="eye-on" aria-pressed="true">Eye on</button></div>
  ${metrics([
    ["Marks inked", "0%", "coverage"],
    ["Pencil height", "Raised", "pencil-height"],
    ["Ink samples", "0", "ink-count"],
  ])}<p id="image-status" aria-live="polite"></p>`;
  $("target-image").onchange = () => example($("target-image").value);
  $("clear-pad").onclick = () => {
    blank();
    observe();
  };
  $("restart-arm").onclick = observe;
  $("erase-copy").onclick = () => arm.erasePatch();
  $("disturb").onclick = () => arm.disturb();
  $("pause").onclick = () => {
    paused = !paused;
    $("pause").textContent = paused ? "Resume" : "Pause";
  };
  $("feedback").onclick = () => {
    arm.closed = !arm.closed;
    $("feedback").setAttribute("aria-pressed", String(arm.closed));
    $("feedback").textContent = arm.closed ? "Feedback on" : "Feedback off";
  };
  for (const [id, indices] of [
    ["joint-motors", [11, 12, 13, 14]],
    ["pencil-motors", [15, 16]],
  ])
    $(id).onclick = () => {
      const enabled = !arm.brain.motor.mask[indices[0]];
      indices.forEach((i) => (arm.brain.motor.mask[i] = +enabled));
      $(id).setAttribute("aria-pressed", String(enabled));
    };
  $("eye-on").onclick = () => {
    eyeOn = !eyeOn;
    $("eye-on").setAttribute("aria-pressed", String(eyeOn));
    $("eye-on").textContent = eyeOn ? "Eye on" : "Eye off";
    observe();
  };
  $("image-upload").onchange = async (e) => {
    const file = e.target.files[0];
    if (!file) return;
    if (file.size > 5e6) {
      $("image-status").textContent = "Choose an image smaller than 5 MB.";
      return;
    }
    try {
      const bitmap = await createImageBitmap(file);
      blank();
      const s = Math.min(184 / bitmap.width, 184 / bitmap.height);
      pen.drawImage(
        bitmap,
        (192 - bitmap.width * s) / 2,
        (192 - bitmap.height * s) / 2,
        bitmap.width * s,
        bitmap.height * s,
      );
      bitmap.close();
      observe();
      $("image-status").textContent =
        `Image received: ${arm.targets.length} retinal targets. Copying visible marks.`;
    } catch {
      $("image-status").textContent =
        "Could not read that image. Try PNG, JPEG or WebP.";
    }
  };
  example("flower");
  explain([
    [
      "Retina and attention",
      "Reference and actual-ink neurons feed missing-mark neurons. A supplied attention rule follows connected dark pixels toward missing ink. Erase a patch to see the arm return and repair it. No stroke coordinates enter the controller.",
    ],
    [
      "Coupled visual and motor regions",
      "Target, pen position and height feed error neurons. Reciprocal visual/premotor synapses coordinate the shoulder and elbow; six antagonistic motor units drive joints and pencil lift.",
    ],
    [
      "Physical readback",
      "The displayed ink raster, joint angles and pencil height return from the body. The pencil stays down on connected strokes and lifts before crossing blank space. Disturb a joint or silence motor populations to test the loop. Geometry and circuit weights are supplied; this is not trained visual recognition.",
    ],
  ]);
  $("evidence-title").textContent =
    "Test the nervous system by interrupting it";
  $("evidence-note").textContent =
    "Tests check connected strokes, separate marks, erased-ink repair, disturbances and motor ablations. Marks inked measures actual ink at reference samples; it does not certify an exact image match. Sampling can lose fine detail, and the pencil cannot erase unwanted ink.";
  $("evidence-table").innerHTML = table(
    ["Intervention", "Expected causal effect"],
    [
      [
        "Joint motor neurons off",
        "Joint drive vanishes; residual body velocity damps",
      ],
      ["Pencil motor neurons off while raised", "No contact and no ink"],
      ["Eye off before copying", "No visual targets"],
      [
        "Erase a patch of the copy",
        "Missing-mark activity returns; the arm repairs the gap",
      ],
      [
        "Readback on after a disturbance",
        "New motor corrections follow the actual pose",
      ],
    ],
  );
  $("boundary").textContent =
    "An engineered computational nervous system: pixel sampling, attention selection, arm geometry and weights are supplied. The displayed motor units actually actuate the body. No biological neuron simulation, learned anatomy or broad ML superiority is claimed.";
  document.querySelector(".evidence details a").href = "eye-arm/evidence.json";
  const mark = (x, y, w, h) => {
    const a = layout(w, h);
    if (x < a.x || x > a.x + a.size || y < a.y || y > a.y + a.size) return null;
    return [((x - a.x) / a.size) * 192, ((y - a.y) / a.size) * 192];
  };
  return {
    draw(w, h, dt) {
      credit += dt * 100;
      while (credit >= 1) {
        if (!paused && !painting) api.act(() => arm.step(true));
        credit--;
      }
      const a = layout(w, h),
        xy = (p) => [a.bodyX + p[0] * a.bodySize, a.bodyY + p[1] * a.bodySize];
      ctx.fillStyle = "#edf3ec";
      ctx.fillRect(a.x, a.y, a.size, a.size);
      ctx.drawImage(pad, a.x, a.y, a.size, a.size);
      ctx.fillStyle = "#91a6b1";
      ctx.font = "11px system-ui";
      ctx.textAlign = "left";
      ctx.fillText("YOUR DRAWING", a.x, 24);
      ctx.fillText("MOTOR-DRIVEN COPY", a.bodyX + 16, 24);
      // Gaze points to the selected retinal sample, never to hidden stroke metadata.
      const gaze = arm.targets[arm.target],
        ex = a.x + a.size * 0.5,
        ey = 42;
      ctx.strokeStyle = "#b8d5d1";
      ctx.lineWidth = 1.5;
      ctx.beginPath();
      ctx.ellipse(ex, ey, 15, 8, 0, 0, Math.PI * 2);
      ctx.stroke();
      const gx = gaze ? (gaze[0] - 0.26) / 0.48 : 0.5,
        gy = gaze ? (gaze[1] - 0.2) / 0.48 : 0.5;
      ctx.fillStyle = eyeOn ? "#87ecc2" : "#667379";
      ctx.beginPath();
      ctx.arc(ex + (gx - 0.5) * 10, ey + 2, 4, 0, Math.PI * 2);
      ctx.fill();
      if (gaze && eyeOn) {
        ctx.setLineDash([3, 5]);
        ctx.strokeStyle = "#619e9055";
        ctx.beginPath();
        ctx.moveTo(ex, ey + 8);
        ctx.lineTo(a.x + gx * a.size, a.y + gy * a.size);
        ctx.stroke();
        ctx.setLineDash([]);
      }
      const paper = xy([0.23, 0.17]);
      ctx.fillStyle = "#e5eadd";
      ctx.fillRect(...paper, a.bodySize * 0.54, a.bodySize * 0.54);
      if (paperRevision !== arm.paper.revision) {
        arm.paper.pixels.forEach((v, i) => {
          copyImage.data.set([36, 78, 67, v ? 255 : 0], i * 4);
        });
        copyContext.putImageData(copyImage, 0, 0);
        paperRevision = arm.paper.revision;
      }
      ctx.drawImage(
        copy,
        ...xy([0.26, 0.2]),
        a.bodySize * 0.48,
        a.bodySize * 0.48,
      );
      const base = [0.5, 0.94],
        elbow = [
          0.5 + 0.43 * Math.cos(arm.q[0]),
          0.94 + 0.43 * Math.sin(arm.q[0]),
        ],
        tip = forward(arm.q);
      for (const [p, q, thickness] of [
        [base, elbow, 13],
        [elbow, tip, 10],
      ]) {
        ctx.strokeStyle = "#54737b";
        ctx.lineWidth = thickness;
        ctx.lineCap = "round";
        ctx.beginPath();
        ctx.moveTo(...xy(p));
        ctx.lineTo(...xy(q));
        ctx.stroke();
      }
      for (const p of [base, elbow]) {
        ctx.fillStyle = "#aacac3";
        ctx.beginPath();
        ctx.arc(...xy(p), 6, 0, Math.PI * 2);
        ctx.fill();
      }
      const [px, py] = xy(tip),
        lift = arm.z * 22;
      ctx.fillStyle = "#15362c44";
      ctx.beginPath();
      ctx.ellipse(px, py, 5 + arm.z * 3, 2, 0, 0, Math.PI * 2);
      ctx.fill();
      ctx.strokeStyle = "#e6b865";
      ctx.lineWidth = 5;
      ctx.beginPath();
      ctx.moveTo(px, py - lift);
      ctx.lineTo(px + 9, py - lift - 26);
      ctx.stroke();
      ctx.fillStyle = arm.lifted ? "#e6b865" : "#244e43";
      ctx.beginPath();
      ctx.arc(px, py - lift, 2, 0, Math.PI * 2);
      ctx.fill();
      ctx.fillStyle = "#a3b4b9";
      ctx.font = "11px system-ui";
      ctx.fillText(
        arm.lifted ? "Pencil raised" : "Pencil touching paper",
        a.bodyX + 14,
        h - 7,
      );
      $("coverage").textContent = pct(arm.coverage);
      $("pencil-height").textContent = arm.z.toFixed(2);
      $("ink-count").textContent = arm.ink.length;
      $("stage-readout").textContent = painting
        ? "Eye waiting for your stroke"
        : `${arm.targets.length} retinal targets · ${arm.lifted ? "pencil up" : "pencil down"}`;
    },
    pointer(x, y, w, h) {
      const p = mark(x, y, w, h);
      if (!p) return;
      painting = true;
      last = p;
      pen.fillStyle = "#17252b";
      pen.beginPath();
      pen.arc(...p, 2, 0, Math.PI * 2);
      pen.fill();
    },
    pointerMove(x, y, w, h) {
      if (!painting) return;
      const p = mark(x, y, w, h);
      if (!p) {
        last = null;
        return;
      }
      pen.strokeStyle = "#17252b";
      pen.lineWidth = 4;
      pen.lineCap = "round";
      pen.beginPath();
      pen.moveTo(...(last ?? p));
      pen.lineTo(...p);
      pen.stroke();
      last = p;
    },
    pointerEnd() {
      if (painting) {
        painting = false;
        last = null;
        observe();
      }
    },
    brain() {
      return {
        ...arm.brain.snapshot(),
        behavior: paused
          ? { label: "Paused", tone: "neutral" }
          : painting
            ? { label: "Watching your drawing", tone: "seeking" }
            : arm.target < 0
              ? {
                  label: arm.targets.length
                    ? "Copy complete"
                    : "Waiting for drawing",
                  tone: arm.targets.length ? "positive" : "neutral",
                }
              : {
                  label: arm.finishedOnce
                    ? "Repairing missing ink"
                    : arm.lifted
                      ? "Reaching / pencil up"
                      : "Following a stroke",
                  tone: "seeking",
                },
      };
    },
    snapshot() {
      return {
        coverage: arm.coverage,
        phase: arm.phase,
        strokes: arm.strokes,
        missing: arm.brain.missing.state.filter((v) => v > 0.1).length,
        ink: arm.ink.length,
        q: arm.q.slice(),
        z: arm.z,
        feedback: arm.closed,
        target: arm.target,
        targets: arm.targets.length,
        revision,
        motor: arm.brain.motor.state.slice(),
        motorMask: arm.brain.motor.mask.slice(),
        retina: arm.brain.eye.state.slice(),
      };
    },
  };
}
