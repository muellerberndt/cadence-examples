import {
  Mouse,
  Arm,
  drawingTargets,
  forward,
  TaskLessons,
  stations,
} from "./embodied.js";
export function mountEmbodied(mode, api) {
  const {
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
    evidence,
  } = api;
  let lessons,
    cue = 0;
  let mouse,
    arm,
    seed = 13,
    paused = false,
    heat = true,
    image = "flower",
    tickCredit = 0;
  const title =
    mode === "mouse"
      ? "Teach a new task.<br>Watch it find a way."
      : "An eye sees the error.<br>An arm makes the correction.";
  $("headline").innerHTML = title;
  $("intro").textContent =
    mode === "mouse"
      ? "Demonstrate a destination. Task memory retains it, a spatial network finds a route, and the moving body closes the loop. Teach a different task without erasing the old ones."
      : "Present an outline. Watch visual and motor regions settle together, move the joints, and compare the drawing with the image again.";
  const worldXY = (w, h) => {
    const cell = Math.min((w - 42) / 19, (h - 35) / 13);
    return { cell, ox: (w - cell * 19) / 2, oy: (h - cell * 13) / 2 };
  };
  const armXY = (w, h) => {
    const size = Math.min(w - 30, h - 10);
    return { size, ox: (w - size) / 2, oy: 0 };
  };
  if (mode === "mouse") {
    try {
      lessons = new TaskLessons(
        JSON.parse(
          localStorage.getItem("cadence.mouse.lessons.v1") || "null",
        ) ?? undefined,
      );
    } catch {
      lessons = new TaskLessons();
    }
    mouse = new Mouse(seed);
    $("stage-label").textContent =
      "LIVE EMBODIMENT · VISUAL MAP + SPATIAL FIELD + MOTOR";
    $("stage-detail").textContent = "Simplified mouse · supplied maze rules";
    $("stage-hint").textContent =
      "Click a wall or corridor to change it. Green shows the settled goal field.";
    $("scene").setAttribute(
      "aria-label",
      "A mouse navigates a labyrinth toward cheese. Brightness shows the spatial goal field. Controls can alter the maze and goal.",
    );
    $("controls").innerHTML =
      `<div><h2>Several parts. One closed loop.</h2><p>Vision updates the map. Neighboring place patches settle. Motor output moves the body; position readback selects the next action.</p></div><div><label for="task-cue">Task cue</label><select id="task-cue"><option value="0">Find cheese</option><option value="1">Go home</option><option value="2">Get water</option><option value="3">New task · untaught</option></select><label for="teach-goal" style="margin-top:9px">Demonstrate a destination</label><select id="teach-goal"><option value="0">Cheese</option><option value="1">Home</option><option value="2">Water</option><option value="3">Flag</option></select><div class="buttons" style="margin-top:8px"><button id="teach-task">Teach task</button><button id="perform-task">Perform task</button></div><p id="lesson-status" aria-live="polite">Three initial demonstrations. Task 4 is yours to teach.</p></div><div class="buttons"><button id="new-maze">New maze</button><button id="move-goal">Move goal</button><button id="pause">Pause</button></div><details><summary>Edit the maze with the keyboard</summary><label for="edit-cell">Edit corridor / wall</label><select id="edit-cell"></select><button id="toggle-wall" style="margin-top:8px">Toggle selected cell</button><p>Unreachable goals stop the mouse; it does not walk through walls.</p></details><div class="buttons"><button id="heat" aria-pressed="true">Show goal field</button></div>${metrics(
        [
          ["Body moves", "0", "mouse-moves"],
          ["Spatial residual", "0", "spatial-residual"],
          ["Goal", "Navigating", "mouse-state"],
        ],
      )}`;
    function choices() {
      $("edit-cell").innerHTML = mouse.world.grid
        .map((v, i) => {
          const x = i % 19,
            y = Math.floor(i / 19);
          return x > 0 && x < 18 && y > 0 && y < 12
            ? `<option value="${i}">Column ${x}, row ${y} · ${v ? "wall" : "corridor"}</option>`
            : "";
        })
        .join("");
    }
    function perform() {
      const destination = lessons.recall(cue);
      if (destination === null) {
        $("lesson-status").textContent =
          "This task is unfamiliar. Demonstrate its destination first.";
        return;
      }
      mouse.world.goal = stations(mouse.world)[destination];
      mouse.repair();
      $("lesson-status").textContent =
        `Task ${cue + 1}: recalling ${["cheese", "home", "water", "flag"][destination]}. Lessons carry into new mazes.`;
    }
    $("task-cue").onchange = () => {
      cue = +$("task-cue").value;
    };
    $("perform-task").onclick = perform;
    $("teach-task").onclick = () => {
      lessons.teach(cue, +$("teach-goal").value);
      try {
        localStorage.setItem(
          "cadence.mouse.lessons.v1",
          JSON.stringify(lessons.records),
        );
      } catch {}
      $("lesson-status").textContent =
        `Task ${cue + 1} learned in one residual write. Earlier tasks retained. Saved in this browser.`;
      if (cue === 3)
        $("task-cue").options[3].textContent = "New task · learned";
    };
    $("new-maze").onclick = () => {
      seed += 10;
      mouse = new Mouse(seed);
      perform();
      choices();
    };
    $("move-goal").onclick = () => {
      const floors = mouse.world.grid
        .map((v, i) => (v ? -1 : i))
        .filter((i) => i >= 0 && i !== mouse.cell);
      mouse.world.goal = floors[(seed * 17 + mouse.moves * 3) % floors.length];
      mouse.repair();
    };
    $("toggle-wall").onclick = () => {
      mouse.toggle(+$("edit-cell").value);
    };
    $("pause").onclick = () => {
      paused = !paused;
      $("pause").textContent = paused ? "Resume" : "Pause";
    };
    $("heat").onclick = () => {
      heat = !heat;
      $("heat").setAttribute("aria-pressed", String(heat));
    };
    choices();
    perform();
    if (lessons.known.has(3)) {
      $("task-cue").options[3].textContent = "New task · learned";
      $("lesson-status").textContent =
        "Saved lessons restored from this browser.";
    }
    explain([
      [
        "Task memory + visual map",
        "Three initial demonstrations teach named task cues. Teach or revise a fourth in one residual write. A supplied visual map locates the recalled destination in each new maze.",
      ],
      [
        "Spatial equilibrium",
        "A goal drive spreads through a contractive network of place patches. Each patch reads neighboring values; a local gradient guides the next step.",
      ],
      [
        "Motor / body feedback",
        "A moving body reports its actual cell. Editing the map or moving the cheese causes a new spatial settlement and new motor choices.",
      ],
    ]);
    $("evidence-title").textContent = "A changed world gets a fresh route";
    $("evidence-note").textContent =
      "Twelve generated mazes, three conditions each: initial navigation, a changed corridor where an alternate path exists, and a moved goal. Every body position is checked against walls.";
    $("evidence-table").innerHTML = table(
      [
        "Condition",
        "Cadence field + body",
        "Frozen original route still valid",
        "BFS finds a route",
      ],
      ["new_maze", "changed_corridor", "moved_goal"].map((c) => {
        const rows = evidence.mouse.filter((r) => r.condition === c);
        return [
          {
            new_maze: "New maze",
            changed_corridor: "Changed corridor",
            moved_goal: "Moved goal",
          }[c],
          `${rows.filter((r) => r.success).length}/${rows.length}`,
          `${rows.filter((r) => r.frozen_route_valid).length}/${rows.length}`,
          `${rows.filter((r) => r.bfs_replanner_success).length}/${rows.length}`,
        ];
      }),
    );
    $("boundary").textContent =
      "Task cues are explicit symbolic keys, not natural-language instructions. Lessons persist only in this browser; a dictionary can also retain these mappings. The graph and full occupancy map are supplied; the mouse is a simplified planar body. Cadence solves a tanh spatial field, and a local readout chooses an ascending neighbor. BFS is a strong conventional control and also succeeds. The frozen-route control demonstrates why stale plans need feedback; it is not an MLP trained to navigate. Changing a corridor may be impossible without disconnecting a perfect maze; each scheduled edit is recorded in the evidence.";
  } else {
    arm = new Arm();
    $("stage-label").textContent =
      "LIVE COMPOSITE · VISUAL ERROR ↔ MOTOR CORRECTION";
    $("stage-detail").textContent =
      "Two joints · four coupled error/correction owners";
    $("stage-hint").textContent =
      "White: target outline. Green: deposited ink. Disturb a joint to test correction.";
    $("scene").setAttribute(
      "aria-label",
      "A two-joint arm sketches a target outline. A small circuit shows reciprocal visual-error and motor-correction activity.",
    );
    $("controls").innerHTML =
      `<div><h2>Drawing is repeated correction.</h2><p>Visual error and motor correction settle jointly. The body moves, and its new pose becomes the next observation.</p></div><div><label for="target-image">Target outline</label><select id="target-image"><option value="flower">Flower</option><option value="leaf">Leaf</option><option value="spiral">Spiral</option></select></div><div><label for="image-upload">Or present your own image</label><input id="image-upload" type="file" accept="image/png,image/jpeg,image/webp" style="width:100%;font-size:11px"><p>Local only. The visual adapter extracts up to 320 edge points; the arm sketches the outline.</p></div><div class="buttons"><button id="disturb">Disturb a joint</button><button id="feedback" aria-pressed="true">Feedback on</button><button id="restart-arm">Start again</button><button id="pause">Pause</button></div>${metrics(
        [
          ["Target coverage", "0%", "coverage"],
          ["Visual error", "—", "visual-error"],
          ["Deposited ink samples", "0", "ink-count"],
        ],
      )}<p id="image-status" aria-live="polite">Supplied arm geometry and visual adapter; no pretrained image model.</p>`;
    $("target-image").onchange = () => {
      image = $("target-image").value;
      arm = new Arm(drawingTargets(image));
      $("feedback").textContent = "Feedback on";
      $("feedback").setAttribute("aria-pressed", "true");
    };
    $("disturb").onclick = () => arm.disturb();
    $("feedback").onclick = () => {
      arm.closed = !arm.closed;
      $("feedback").textContent = arm.closed ? "Feedback on" : "Feedback off";
      $("feedback").setAttribute("aria-pressed", String(arm.closed));
    };
    $("restart-arm").onclick = () => {
      arm = new Arm(arm.targets);
      $("feedback").textContent = "Feedback on";
      $("feedback").setAttribute("aria-pressed", "true");
    };
    $("pause").onclick = () => {
      paused = !paused;
      $("pause").textContent = paused ? "Resume" : "Pause";
    };
    $("image-upload").onchange = async (e) => {
      const file = e.target.files[0];
      if (!file) return;
      if (file.size > 5e6) {
        $("image-status").textContent =
          "Please choose an image smaller than 5 MB.";
        return;
      }
      try {
        const bitmap = await createImageBitmap(file),
          c = document.createElement("canvas");
        c.width = c.height = 48;
        const cx = c.getContext("2d");
        cx.fillStyle = "white";
        cx.fillRect(0, 0, 48, 48);
        const scale = Math.min(46 / bitmap.width, 46 / bitmap.height);
        cx.drawImage(
          bitmap,
          (48 - bitmap.width * scale) / 2,
          (48 - bitmap.height * scale) / 2,
          bitmap.width * scale,
          bitmap.height * scale,
        );
        bitmap.close();
        const pixels = cx.getImageData(0, 0, 48, 48).data,
          gray = (i) =>
            (pixels[i * 4] + pixels[i * 4 + 1] + pixels[i * 4 + 2]) / 765,
          targets = [];
        for (let y = 1; y < 47; y++)
          for (let x = 1; x < 47; x++) {
            const i = y * 48 + x;
            if (
              Math.abs(gray(i + 1) - gray(i - 1)) +
                Math.abs(gray(i + 48) - gray(i - 48)) >
              0.28
            )
              targets.push([0.29 + (x / 48) * 0.42, 0.23 + (y / 48) * 0.42]);
          }
        if (targets.length < 4) {
          $("image-status").textContent =
            "Few edges detected. Try a high-contrast drawing or photograph.";
          return;
        }
        const selected = targets.filter(
          (_, i) => i % Math.max(1, Math.ceil(targets.length / 320)) === 0,
        );
        arm = new Arm(selected);
        $("feedback").textContent = "Feedback on";
        $("feedback").setAttribute("aria-pressed", "true");
        $("image-status").textContent =
          `Image received: ${selected.length} edge targets. Drawing from visual readback.`;
      } catch (error) {
        $("image-status").textContent =
          "Could not decode that image. Try PNG, JPEG or WebP.";
      }
    };
    explain([
      [
        "Eye / visual error region",
        "The image adapter supplies edge targets. The system compares a target with observed pen position and deposited ink. It draws an outline rather than generating an illustration.",
      ],
      [
        "Joint equilibrium with motor regions",
        "Visual-error owners receive negative feedback from predicted joint movement. Motor owners receive the visual residual through the arm Jacobian. Four owners settle together.",
      ],
      [
        "Body / proprioceptive return",
        "The joints move, then their actual pose is read back. A disturbance tests whether the next correction follows the real arm or a stale internal estimate.",
      ],
    ]);
    $("evidence-title").textContent = "Feedback repairs a disturbed drawing";
    $("evidence-note").textContent =
      "Three outlines × three disturbance times. Same controller and joint disturbance; only visual/proprioceptive readback changes. Coverage counts target points close to deposited ink after 4,000 control steps.";
    $("evidence-table").innerHTML = table(
      ["Target", "Coupled feedback", "Feedback disabled"],
      ["flower", "leaf", "spiral"].map((name) => [
        name,
        ...[true, false].map((mode) =>
          pct(
            mean(
              evidence.arm
                .filter((r) => r.image === name && r.feedback === mode)
                .map((r) => r.coverage),
            ),
          ),
        ),
      ]),
    );
    $("boundary").textContent =
      "The visual edge adapter, arm geometry and Jacobian are supplied. The example demonstrates coupled error correction and embodied readback, not learned vision, learned anatomy, or a general advantage over backpropagation. Coverage uses a 0.022-distance tolerance in normalized workspace units; it does not score artistic quality. The same four-owner numerical update agrees with the Python Cadence engine.";
  }
  const evidenceLink = document.querySelector(".evidence details a");
  evidenceLink.href = "showcase/composite_evidence.json";
  return {
    draw(w, h, dt) {
      if (mode === "mouse") {
        if (!paused) mouse.step(dt);
        const { cell, ox, oy } = worldXY(w, h),
          world = mouse.world,
          state = mouse.circuit.state,
          max = Math.max(...state),
          xy = (i) => [
            ox + ((i % 19) + 0.5) * cell,
            oy + (Math.floor(i / 19) + 0.5) * cell,
          ];
        world.grid.forEach((wall, i) => {
          const x = ox + (i % 19) * cell,
            y = oy + Math.floor(i / 19) * cell;
          ctx.fillStyle = wall
            ? "#304740"
            : heat
              ? `rgba(135,236,194,${0.025 + 0.25 * Math.sqrt(state[i] / Math.max(max, 1e-12))})`
              : "#10221e";
          ctx.fillRect(x + 0.5, y + 0.5, cell - 1, cell - 1);
        });
        for (let i = 1; i < mouse.path.length; i++)
          line(xy(mouse.path[i - 1]), xy(mouse.path[i]), "#87ecc255", 2);
        stations(world).forEach((i, k) => {
          const [sx, sy] = xy(i);
          if (i !== world.goal) {
            circle(sx, sy, 5, ["#ffd580", "#bfacfa", "#8acfff", "#ffb17a"][k]);
            text(["C", "H", "W", "F"][k], sx, sy + 3, "#0a1715", 8, "center");
          }
        });
        const [gx, gy] = xy(world.goal);
        const goalKind = stations(world).indexOf(world.goal);
        if (goalKind === 3) {
          line([gx - 5, gy - 10], [gx - 5, gy + 10], "#ffb17a", 2);
          ctx.fillStyle = "#ffb17a";
          ctx.beginPath();
          ctx.moveTo(gx - 4, gy - 10);
          ctx.lineTo(gx + 9, gy - 5);
          ctx.lineTo(gx - 4, gy);
          ctx.fill();
        } else if (goalKind === 1) {
          ctx.fillStyle = "#bfacfa";
          ctx.fillRect(gx - 7, gy - 2, 14, 11);
          ctx.beginPath();
          ctx.moveTo(gx - 10, gy - 2);
          ctx.lineTo(gx, gy - 11);
          ctx.lineTo(gx + 10, gy - 2);
          ctx.fill();
        } else if (goalKind === 2) {
          circle(gx, gy, 9, "#8acfff");
          text("W", gx, gy + 4, "#092028", 11, "center");
        } else {
          ctx.fillStyle = "#ffd580";
          ctx.beginPath();
          ctx.moveTo(gx - 8, gy + 7);
          ctx.lineTo(gx + 9, gy + 7);
          ctx.lineTo(gx + 5, gy - 8);
          ctx.closePath();
          ctx.fill();
          circle(gx, gy + 3, 2, "#ba8f46");
          circle(gx + 4, gy - 2, 1.5, "#ba8f46");
        }
        const x = ox + mouse.x * cell,
          y = oy + mouse.y * cell;
        ctx.save();
        ctx.translate(x, y);
        ctx.rotate(mouse.heading);
        ctx.strokeStyle = "#e5c3b0";
        ctx.lineWidth = 1.5;
        ctx.beginPath();
        ctx.moveTo(-8, 0);
        ctx.quadraticCurveTo(-17, 8, -21, 3);
        ctx.stroke();
        ctx.fillStyle = "#d5c8b8";
        ctx.beginPath();
        ctx.ellipse(0, 0, 10, 6, 0, 0, Math.PI * 2);
        ctx.fill();
        circle(6, -5, 3, "#edcfc4");
        circle(6, 5, 3, "#edcfc4");
        circle(10, 0, 2, "#f3bfb4");
        circle(7, -2, 1, "#17221f");
        line([9, -2], [15, -5], "#e7ddd0", 0.7);
        line([9, 2], [15, 5], "#e7ddd0", 0.7);
        ctx.restore();
        $("mouse-moves").textContent = mouse.moves;
        $("spatial-residual").textContent =
          mouse.circuit.residual.toExponential(1);
        $("mouse-state").textContent =
          mouse.cell === world.goal
            ? "Reached"
            : mouse.next() < 0
              ? "No route"
              : "Navigating";
        $("stage-readout").textContent =
          `${world.grid.filter((v) => !v).length} place owners · ${mouse.circuit.steps} settling steps`;
      } else {
        tickCredit += dt * 65;
        while (tickCredit >= 1) {
          if (!paused) arm.step();
          tickCredit--;
        }
        const { size, ox, oy } = armXY(w, h),
          xy = (p) => [ox + p[0] * size, oy + p[1] * size];
        // Camera inset shows the actual supplied edge image, not a stock illustration.
        ctx.fillStyle = "#182d2a";
        ctx.fillRect(20, 24, 100, 100);
        arm.targets.forEach((p) =>
          circle(
            28 + (p[0] - 0.25) * 170,
            30 + (p[1] - 0.2) * 170,
            1,
            "#e2ece3",
          ),
        );
        text("VISUAL TARGET", 70, 142, "#a3b4b9", 9, "center");
        arm.targets.forEach((p) => circle(...xy(p), 1.5, "#eff7ef55"));
        for (let i = 0; i < arm.ink.length; i++) {
          const p = arm.ink[i];
          if (
            i &&
            Math.hypot(p[0] - arm.ink[i - 1][0], p[1] - arm.ink[i - 1][1]) <
              0.04
          )
            line(xy(arm.ink[i - 1]), xy(p), "#87ecc2", 2);
          else circle(...xy(p), 1.5, "#87ecc2");
        }
        const base = [0.5, 0.94],
          elbow = [
            0.5 + 0.43 * Math.cos(arm.q[0]),
            0.94 + 0.43 * Math.sin(arm.q[0]),
          ],
          tip = forward(arm.q);
        line(xy(base), xy(elbow), "#30434d", 17);
        line(xy(elbow), xy(tip), "#415963", 13);
        line(xy(base), xy(elbow), "#6c8790", 2);
        line(xy(elbow), xy(tip), "#8aa7ac", 2);
        for (const p of [base, elbow]) {
          circle(...xy(p), 11, "#1b2e33");
          circle(...xy(p), 5, "#aac3bb");
        }
        circle(...xy(tip), 4, arm.lifted ? "#ffb17a" : "#87ecc2");
        if (arm.target >= 0) {
          const goal = arm.targets[arm.target];
          ctx.setLineDash([3, 5]);
          line(xy(tip), xy(goal), "#ffb17a80");
          ctx.setLineDash([]);
          circle(...xy(goal), 4, "#ffb17a");
        }
        // The four displayed nodes are the actual coupled visual/motor settlement.
        const nodes = [
            [35, h - 105],
            [35, h - 60],
            [105, h - 105],
            [105, h - 60],
          ],
          s = arm.settlement?.state ?? [0, 0, 0, 0];
        for (let i = 0; i < 2; i++)
          for (let j = 2; j < 4; j++) line(nodes[i], nodes[j], "#3e675b");
        nodes.forEach((p, i) => {
          circle(...p, 5 + Math.abs(s[i]) * 35, i < 2 ? "#ffb17a" : "#87ecc2");
        });
        text("VISUAL", 35, h - 27, "#ffb17a", 9, "center");
        text("MOTOR", 105, h - 27, "#87ecc2", 9, "center");
        $("coverage").textContent = pct(arm.coverage);
        $("ink-count").textContent = arm.ink.length;
        $("visual-error").textContent =
          arm.target < 0
            ? "Done"
            : Math.hypot(
                ...(arm.settlement?.drive ?? [0, 0]).slice(0, 2),
              ).toFixed(3);
        $("stage-readout").textContent = arm.closed
          ? "Visual + proprioceptive readback active"
          : "Readback disabled · internal pose estimate";
      }
    },
    pointer(x, y, w, h) {
      if (mode !== "mouse") return;
      const { cell, ox, oy } = worldXY(w, h),
        a = Math.floor((x - ox) / cell),
        b = Math.floor((y - oy) / cell);
      if (a >= 0 && a < 19 && b >= 0 && b < 13) mouse.toggle(b * 19 + a);
    },
    snapshot() {
      return mode === "mouse"
        ? {
            cell: mouse.cell,
            goal: mouse.world.goal,
            lessons: lessons.records,
            moves: mouse.moves,
            residual: mouse.circuit.residual,
          }
        : {
            coverage: arm.coverage,
            ink: arm.ink.length,
            feedback: arm.closed,
            q: arm.q.slice(),
            target: arm.target,
          };
    },
  };
}
