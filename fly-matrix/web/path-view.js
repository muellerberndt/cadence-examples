// The patch net's applied local commands and measured body motion are visible
// here. The gold guide holds the current command briefly; it is not a route
// planner or a body simulation. Assistance still supplies flight stabilization.
// This viewer has no controller/worker writes. Exclude its group from the eye.
import * as THREE from "three";

const CYAN = 0x69e8ff, GRAY = 0x80998d, GOLD = 0xffd166;
const percent = x => `${(100 * x).toFixed(1)}%`;
const drawText = x => Number.isFinite(x) ? x.toFixed(4) : "unavailable";

export function createPathView(world, panel) {
  const group = new THREE.Group(); group.name = "observer-path-annotations"; world.add(group);
  const trail = new THREE.LineSegments(new THREE.BufferGeometry(), new THREE.LineBasicMaterial({ vertexColors: true, depthTest: false, transparent: true, opacity: .9 }));
  const guide = new THREE.Line(new THREE.BufferGeometry(), new THREE.LineDashedMaterial({ color: GOLD, dashSize: .003, gapSize: .002, depthTest: false, transparent: true, opacity: .95 }));
  trail.renderOrder = guide.renderOrder = 20;
  trail.frustumCulled = guide.frustumCulled = false;
  group.add(trail, guide);
  const $ = id => panel.querySelector(`#${id}`);
  const map = $("path-map"), g = map.getContext("2d");
  const activeColor = new THREE.Color(CYAN), otherColor = new THREE.Color(GRAY);
  let enabled = true;

  function setEnabled(on) {
    enabled = !!on; group.visible = enabled; panel.hidden = !enabled;
    document.body.classList.toggle("paths-visible", enabled);
  }

  function update(state) {
    if (!enabled) return;
    const { history, projection, decision, decisionCurrent } = state;
    const current = history.current;
    if (!current) return;
    const points = [...history.points];
    if (points.at(-1)?.time !== current.time) points.push(current);
    // Actual samples are never filled in with commanded positions. A grey
    // segment means a neural command was not active throughout that interval.
    const vertices = [], colors = [];
    for (let k = 1; k < points.length; k++) {
      const a = points[k - 1], b = points[k];
      const color = b.intervalActive ? activeColor : otherColor;
      vertices.push(...a.position, ...b.position); colors.push(color.r, color.g, color.b, color.r, color.g, color.b);
    }
    trail.geometry.dispose(); trail.geometry = new THREE.BufferGeometry();
    trail.geometry.setAttribute("position", new THREE.Float32BufferAttribute(vertices, 3));
    trail.geometry.setAttribute("color", new THREE.Float32BufferAttribute(colors, 3));
    guide.geometry.dispose(); guide.geometry = new THREE.BufferGeometry().setFromPoints(projection.map(p => new THREE.Vector3(...p.position)));
    guide.visible = projection.length > 1;
    if (guide.visible) guide.computeLineDistances();

    const status = state.paused ? "Paused · command withdrawn"
      : state.grounded ? "Ground contact · wings inactive"
      : current.active ? `Neural command active · observation #${current.observationId}`
      : state.hybrid ? (state.goal?.executed ? "Body controller executing neural goal" : "Body holding · awaiting neural goal") : "Waiting for an applied neural command";
    $("path-status").textContent = status;
    $("path-status").dataset.active = String(current.active);
    const c = current.command;
    const turn = Math.abs(c.yawRate * 180 / Math.PI);
    const signed = (v, plus, minus, scale) => `${Math.abs(v * scale).toFixed(2)} ${v < 0 ? minus : plus}`;
    $("path-command").textContent = current.active
      ? `Turn ${turn.toFixed(2)}°/s ${c.yawRate < 0 ? "right" : "left"} · ${signed(c.forwardSpeed, "cm/s forward", "cm/s backward", 100)} · ${signed(c.verticalSpeed, "cm/s climb", "cm/s descend", 100)}`
      : state.motionSupport?.active && state.motionSupport.source
        ? `Executing recorded course #${state.motionSupport.source.requestId} · ${(100 * Math.hypot(...state.motionSupport.velocityWorld)).toFixed(1)} cm/s smoothed target · awaiting fresh neural input.`
        : state.hybrid ? "The brain selects the target and approach/avoid; the body controller supplies the route and landing mechanics." : "No active neural velocity request.";
    const duration = projection.length ? projection.at(-1).time - current.time : 0;
    $("path-duration").textContent = state.hybrid ? "Grey: actual movement through the body controller; no neural velocity projection."
      : duration ? `Gold: raw neural request before cruise scaling, held for ${duration.toFixed(2)} s` : "Gold projection waits for a fresh neural movement command";
    $("path-time").textContent = `${current.time.toFixed(2)} s simulated · ${Math.max(0, current.time - (points[0]?.time ?? current.time)).toFixed(1)} s of trail`;

    if (decision) {
      const target = decision.targetSelection, action = decision.decision;
      const tied = target && Math.max(...target.p) - Math.min(...target.p) < .02;
      $("path-choice").textContent = `${decisionCurrent ? "Current odor attention" : "Last odor choice (ended)"}: ${target?.fruit ?? decision.fruit}. ${target ? (tied ? "Near tie — no clear preference." : "Sampled from the neural scores.") : "Supplied target."}`;
      $("path-candidates").textContent = target ? target.candidates.map((c, k) => `${c.fruit}: ${percent(target.p[k])} · score ${c.score.toExponential(3)}`).join("\n") : "This comparison controller supplies the target.";
      $("path-sampling").textContent = `${target ? `Odor draw ${drawText(target.draw)} · ` : ""}approach ${percent(action.p[0])} / avoid ${percent(action.p[1])}\nAction draw ${drawText(action.draw)} → ${action.action === 0 ? "approach" : "avoid"} intent${decisionCurrent ? (state.executed ? " · enacted" : " · awaiting actuation") : " · historical"}`;
    } else {
      $("path-choice").textContent = "Odor attention: waiting for a checked decision.";
      $("path-candidates").textContent = "Banana and bread are compared through the brain.";
      $("path-sampling").textContent = "The sampler uses their probabilities. A near tie can choose either odor.";
    }
    drawMap(points, projection, current);
  }

  function drawMap(points, projection, current) {
    const W = map.width, H = map.height, pad = 22;
    g.clearRect(0, 0, W, H); g.fillStyle = "#020c09"; g.fillRect(0, 0, W, H);
    const positions = [...points, ...projection].map(p => p.position);
    const xs = positions.map(p => p[0]), ys = positions.map(p => p[1]);
    const lowX = Math.min(...xs), highX = Math.max(...xs), lowY = Math.min(...ys), highY = Math.max(...ys);
    const scale = Math.min((W - 2 * pad) / Math.max(.10, highX - lowX), (H - 2 * pad) / Math.max(.10, highY - lowY));
    const midX = (lowX + highX) / 2, midY = (lowY + highY) / 2;
    const xy = p => [W / 2 + (p[0] - midX) * scale, H / 2 - (p[1] - midY) * scale];
    g.lineWidth = 2.5;
    for (let k = 1; k < points.length; k++) {
      g.strokeStyle = points[k].intervalActive ? "#69e8ff" : "#80998d";
      g.beginPath(); g.moveTo(...xy(points[k - 1].position)); g.lineTo(...xy(points[k].position)); g.stroke();
    }
    if (projection.length) {
      g.strokeStyle = "#ffd166"; g.setLineDash([7, 5]); g.beginPath();
      projection.forEach((p, k) => k ? g.lineTo(...xy(p.position)) : g.moveTo(...xy(p.position))); g.stroke(); g.setLineDash([]);
      g.fillStyle = "#ffd166"; g.beginPath(); g.arc(...xy(projection.at(-1).position), 3, 0, 2 * Math.PI); g.fill();
    }
    const [x, y] = xy(current.position);
    g.fillStyle = "#e3fbff"; g.beginPath(); g.arc(x, y, 4, 0, 2 * Math.PI); g.fill();
    g.font = "13px monospace"; g.fillStyle = "#96b4a4"; g.fillText("TOP VIEW · +y ↑", 12, 18);
    // A real metric scale keeps auto-fitting from exaggerating tiny motion.
    const metres = 10 ** Math.floor(Math.log10(80 / scale)), pixels = metres * scale;
    g.strokeStyle = "#96b4a4"; g.lineWidth = 1; g.beginPath(); g.moveTo(12, H - 15); g.lineTo(12 + pixels, H - 15); g.stroke();
    g.fillText(`${(metres * 100).toFixed(metres < .01 ? 1 : 0)} cm`, 18 + pixels, H - 11);
    g.fillText("+x →", W - 57, H - 11);
  }

  return { group, update, setEnabled, get enabled() { return enabled; } };
}
