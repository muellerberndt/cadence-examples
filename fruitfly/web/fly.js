// The fly in the Matrix: an articulated wireframe body about 3 mm long, in body coordinates
// (x forward, y left, z up, metres). Head, thorax and abdomen as dim fills with glowing edges;
// compound eyes brighter; antennae with aristae, a jointed proboscis, six legs of three
// segments each (solved from foot targets every frame) and two halteres as glowing lines with
// points at the joints; two wings as bright outlines with veins over a faint fill, drawn while
// flying at the instantaneous stroke angle beside a translucent sweep wedge whose angle follows
// the stroke amplitude, and folded flat over the abdomen when the fly sits. Below the fly a
// faint disc of light rests on the surface beneath it.
//
// update(flight, controls, dt, surfaceZ, showLocator, pose) takes the pose of the fly's life:
//   { mode: "flying" | "landed" | "grooming" | "feeding" | "takeoff",
//     t: seconds of the life clock, groom: "head" | "wings" | "legs" | null,
//     feed: 0..1 (proboscis extension), hunger: 0..1 }
// A call without a pose is a flying fly (a landed one when the physics says landed). When the fly
// stands, the body centre stays where the physics puts it (RADIUS above the surface) and the
// tarsi reach down to the surface.
import * as THREE from "three";
import { HINGE_Y, HINGE_Z } from "./body.js";
import { radialTexture, dotTexture, wire, lineMaterial, MATRIX } from "./room.js";

const MM = 1e-3;
const TWO_PI = Math.PI * 2;
const SWEEP_HOVER = 1.0472;     // rad, half the stroke angle at hover amplitude (60 degrees)
const SWEEP_MAX = 1.396;        // rad, 80 degrees
const WING_LENGTH = 2.3 * MM;
const FAN_SEGMENTS = 18;
const FOLD_S = 0.15;            // s, the wings fold or unfold, the legs go from tucked to standing
const TAKEOFF_S = 0.06;         // s, the legs extend at takeoff before they tuck

const COLOURS = { edge: MATRIX.edge, eye: MATRIX.bright, fill: MATRIX.fill, eyeFill: 0x0b3a1c, wing: MATRIX.edge, joint: MATRIX.bright };

// -- pieces -----------------------------------------------------------------------------------

const _a = new THREE.Vector3(), _b = new THREE.Vector3(), _c = new THREE.Vector3(), _d = new THREE.Vector3(), _e = new THREE.Vector3();

function wingShape() {
  const s = new THREE.Shape();
  s.moveTo(0, 0);
  s.bezierCurveTo(0.55 * MM, 0.5 * MM, 0.5 * MM, 1.9 * MM, 0.05 * MM, 2.25 * MM);
  s.bezierCurveTo(-0.3 * MM, 2.1 * MM, -0.42 * MM, 1.0 * MM, 0, 0);
  return s;
}

// The wing's outline and veins as line segments in the wing's plane: five longitudinal veins
// fanning from the hinge, two cross veins, a thickened leading edge.
function wingLines(shape) {
  const out = [];
  const pts = shape.getPoints(28);
  for (let i = 0; i < pts.length - 1; i++) out.push(pts[i].x, pts[i].y, 0, pts[i + 1].x, pts[i + 1].y, 0);
  const X = (u) => (-0.42 + 0.97 * u) * MM, Y = (v) => v * 2.25 * MM;
  const curves = [[0.5, 0.0, 0.34, 0.5, 0.2, 0.94], [0.52, 0.0, 0.5, 0.5, 0.42, 0.97], [0.55, 0.0, 0.66, 0.5, 0.64, 0.95], [0.58, 0.02, 0.8, 0.45, 0.8, 0.88], [0.62, 0.05, 0.9, 0.35, 0.92, 0.68]];
  for (const [x0, y0, x1, y1, x2, y2] of curves) {
    const c = new THREE.QuadraticBezierCurve(new THREE.Vector2(X(x0), Y(y0)), new THREE.Vector2(X(x1), Y(y1)), new THREE.Vector2(X(x2), Y(y2)));
    const p = c.getPoints(10);
    for (let i = 0; i < p.length - 1; i++) out.push(p[i].x, p[i].y, 0, p[i + 1].x, p[i + 1].y, 0);
  }
  for (const [x0, y0, x1, y1] of [[0.45, 0.42, 0.62, 0.44], [0.62, 0.62, 0.8, 0.6]]) out.push(X(x0), Y(y0), 0, X(x1), Y(y1), 0);
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(out, 3));
  return g;
}

function fanGeometry() {
  const g = new THREE.BufferGeometry();
  const pos = new Float32Array(3 * (FAN_SEGMENTS + 2));
  g.setAttribute("position", new THREE.BufferAttribute(pos, 3));
  const idx = [];
  for (let k = 0; k < FAN_SEGMENTS; k++) idx.push(0, k + 1, k + 2);
  g.setIndex(idx);
  return g;
}

function buildWing(side, shape, linesGeo) {
  const pivot = new THREE.Group();
  pivot.position.set(0, side * HINGE_Y, HINGE_Z);
  const tilt = new THREE.Group();
  pivot.add(tilt);
  const fan = new THREE.Mesh(fanGeometry(), new THREE.MeshBasicMaterial({ color: COLOURS.wing, transparent: true, opacity: 0.1, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
  fan.renderOrder = 1; tilt.add(fan);
  const blade = new THREE.Group();
  const fill = new THREE.Mesh(new THREE.ShapeGeometry(shape, 12), new THREE.MeshBasicMaterial({ color: COLOURS.wing, transparent: true, opacity: 0.12, side: THREE.DoubleSide, depthWrite: false, blending: THREE.AdditiveBlending }));
  fill.renderOrder = 1;
  const lines = new THREE.LineSegments(linesGeo, new THREE.LineBasicMaterial({ color: COLOURS.wing, transparent: true, opacity: 0.85, blending: THREE.AdditiveBlending, depthWrite: false }));
  lines.renderOrder = 2;
  blade.add(fill, lines);
  if (side < 0) blade.scale.y = -1;
  tilt.add(blade);
  return { pivot, tilt, fan, blade, fill, lines, side };
}

// The six legs: hip on the thorax, the three segment lengths, the standing foot, the tucked foot.
// Right legs mirror y. Lengths in mm.
const LEGS = [
  { name: "front", hip: [0.5, 0.24, -0.2], Lf: 0.6, Lt: 0.72, Lta: 0.42, stand: [1.0, 0.8], tuck: [0.15, 0.4, -0.6], tuckDir: [-1, 0, -0.15], tuckBulge: [1, 0.3, 0.35] },
  { name: "mid", hip: [0.2, 0.3, -0.22], Lf: 0.7, Lt: 0.85, Lta: 0.48, stand: [0.3, 1.3], tuck: [-1.55, 0.45, -0.6], tuckDir: [-1, 0.1, -0.1], tuckBulge: [0, 1, -0.5] },
  { name: "hind", hip: [-0.15, 0.28, -0.22], Lf: 0.8, Lt: 0.95, Lta: 0.55, stand: [-1.0, 1.1], tuck: [-2.15, 0.3, -0.5], tuckDir: [-1, 0, -0.05], tuckBulge: [0, 1, -0.5] },
];

export function createFly() {
  const group = new THREE.Group();           // at the flight position and attitude
  const body = new THREE.Group();            // pitched forward a little when feeding
  group.add(body);

  // the thorax: a wire globe with its poles along x; the abdomen: a banded lathe, five tergites
  const thorax = wire(new THREE.SphereGeometry(0.46 * MM, 10, 7), COLOURS.edge, 0.7, COLOURS.fill, 0.75);
  thorax.rotation.z = Math.PI / 2; thorax.scale.set(0.85, 1.0, 0.8); thorax.position.set(0.15 * MM, 0, 0); body.add(thorax);
  const abdPts = [];
  for (let i = 0; i <= 12; i++) {
    const u = i / 12, ell = Math.sqrt(Math.max(0, 1 - ((u - 0.4) / 0.6) ** 2));
    const r = 0.36 * MM * (0.04 + 0.96 * ell) * (1 + 0.06 * Math.sin(TWO_PI * 5 * u - 0.8) * (u > 0.05 && u < 0.92 ? 1 : 0));
    abdPts.push(new THREE.Vector2(Math.max(0.004 * MM, r), u * 1.75 * MM));
  }
  const abdomen = wire(new THREE.LatheGeometry(abdPts, 10), COLOURS.edge, 0.7, COLOURS.fill, 0.75);
  abdomen.rotation.z = Math.PI / 2; abdomen.position.set(-0.2 * MM, 0, -0.03 * MM); abdomen.scale.set(1, 1, 0.82); body.add(abdomen);

  // the head: its own group, so it can turn; the eyes brighter
  const head = new THREE.Group(); head.position.set(0.92 * MM, 0, 0.06 * MM); body.add(head);
  const skull = wire(new THREE.SphereGeometry(0.3 * MM, 8, 6), COLOURS.edge, 0.7, COLOURS.fill, 0.75);
  skull.scale.set(0.8, 1.0, 0.92); head.add(skull);
  for (const s of [1, -1]) {
    const eye = wire(new THREE.SphereGeometry(0.2 * MM, 8, 6), COLOURS.eye, 0.95, COLOURS.eyeFill, 0.8);
    eye.scale.set(0.85, 0.62, 1.05); eye.position.set(0.05 * MM, s * 0.2 * MM, 0.02 * MM); eye.rotation.z = s * 0.15; head.add(eye);
  }

  // the thin parts as one set of glowing line segments in the body frame, rewritten every frame:
  // 18 leg segments, 2 proboscis segments, 6 antenna segments, 2 haltere stalks, 4 bristles
  const N_SEG = 18 + 2 + 6 + 2 + 4;
  const linePos = new Float32Array(6 * N_SEG);
  const lineGeo = new THREE.BufferGeometry(); lineGeo.setAttribute("position", new THREE.BufferAttribute(linePos, 3));
  const lines = new THREE.LineSegments(lineGeo, lineMaterial(COLOURS.edge, 0.85)); lines.frustumCulled = false; lines.renderOrder = 1; body.add(lines);
  // and the joints as glowing points: knees, ankles, feet, labellum, two funiculi, two haltere knobs, three ocelli
  const N_PTS = 18 + 1 + 2 + 2 + 3;
  const ptPos = new Float32Array(3 * N_PTS);
  const ptGeo = new THREE.BufferGeometry(); ptGeo.setAttribute("position", new THREE.BufferAttribute(ptPos, 3));
  const points = new THREE.Points(ptGeo, new THREE.PointsMaterial({ color: COLOURS.joint, size: 3.5, sizeAttenuation: false, map: dotTexture(), transparent: true, opacity: 0.95, blending: THREE.AdditiveBlending, depthWrite: false }));
  points.frustumCulled = false; points.renderOrder = 2; body.add(points);
  let segI = 0, ptI = 0;
  const putSeg = (a, b) => { linePos[segI++] = a.x; linePos[segI++] = a.y; linePos[segI++] = a.z; linePos[segI++] = b.x; linePos[segI++] = b.y; linePos[segI++] = b.z; };
  const putPt = (p) => { ptPos[ptI++] = p.x; ptPos[ptI++] = p.y; ptPos[ptI++] = p.z; };

  const legs = [];
  for (const s of [1, -1]) for (const L of LEGS) legs.push({ ...L, side: s, hipV: new THREE.Vector3(L.hip[0] * MM, s * L.hip[1] * MM, L.hip[2] * MM), foot: new THREE.Vector3(), ankle: new THREE.Vector3(), kneeV: new THREE.Vector3(), dir: new THREE.Vector3(), bulge: new THREE.Vector3() });
  const antennae = [1, -1].map((s) => ({ side: s, twitch: 0, next: 0.4 + 0.5 * (s + 1) }));
  const halteres = [1, -1].map((s) => ({ side: s, base: new THREE.Vector3(-0.5 * MM, s * 0.38 * MM, 0.08 * MM), dir: new THREE.Vector3(-0.45, s * 0.85, 0.25).normalize() }));
  const bristles = [[-0.3, 0.12, 0.33], [-0.1, -0.2, 0.34], [0.15, 0.22, 0.35], [0.35, -0.1, 0.33]].map(([x, y, z]) => ({ a: new THREE.Vector3(x * MM, y * MM, z * MM), b: new THREE.Vector3((x - 0.16) * MM, y * 1.3 * MM, (z + 0.12) * MM) }));

  const shape = wingShape(), wingGeo = wingLines(shape);
  const wings = [buildWing(1, shape, wingGeo), buildWing(-1, shape, wingGeo)];
  for (const w of wings) body.add(w.pivot);

  // the light on the surface beneath, and a locator for the room view
  const shadow = new THREE.Mesh(new THREE.CircleGeometry(1, 28), new THREE.MeshBasicMaterial({ map: radialTexture("rgba(57,255,106,.6)", "rgba(57,255,106,0)"), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending }));
  shadow.renderOrder = 1;
  const locator = new THREE.Sprite(new THREE.SpriteMaterial({ map: radialTexture("rgba(157,255,176,.85)", "rgba(157,255,176,0)"), transparent: true, depthWrite: false, depthTest: false, sizeAttenuation: false, opacity: 0.7 }));
  locator.scale.set(0.022, 0.022 * 1.6, 1); locator.visible = false;

  const fan = new Float32Array(3 * (FAN_SEGMENTS + 2));
  const smooth = (u) => u * u * (3 - 2 * u);
  const clamp01 = (u) => Math.min(1, Math.max(0, u));
  const S = { mode: "flying", stand: 0, takeoff: -1, headYaw: 0, headPitch: 0, wantYaw: 0, wantPitch: 0, headTau: 0.4, headNext: 0.5, feed: 0, groomSide: 1, groomT: 0 };
  const n = new THREE.Vector3(), nb = new THREE.Vector3(), tb = new THREE.Vector3(), qb = new THREE.Quaternion(), qi = new THREE.Quaternion();
  const P0 = new THREE.Vector3(), P1 = new THREE.Vector3(), P2 = new THREE.Vector3(), ext1 = new THREE.Vector3(), ext2 = new THREE.Vector3(), fold1 = new THREE.Vector3(), fold2 = new THREE.Vector3();
  const hp = new THREE.Vector3(), hq = new THREE.Vector3(), hr = new THREE.Vector3(), hs = new THREE.Vector3();

  // The surface under the fly as a plane in the body group's frame: nb . p = dist.
  let dist = 0;
  const surfaceAt = (x, y) => (Math.abs(nb.z) < 0.2 ? -1.2 * MM : (dist - nb.x * x - nb.y * y) / nb.z);

  function solveLeg(L, footTarget, tarsusDir, bulge) {
    // the ankle sits one tarsus back from the foot along tarsusDir; the femur and tibia meet at a
    // knee pushed toward bulge; a reach beyond the leg's length moves the foot in
    L.dir.copy(tarsusDir).normalize();
    L.ankle.copy(footTarget).addScaledVector(L.dir, -L.Lta * MM);
    _a.subVectors(L.ankle, L.hipV); let d = _a.length();
    const reach = (L.Lf + L.Lt) * MM * 0.995;
    if (d > reach) { _a.multiplyScalar(reach / d); d = reach; L.ankle.copy(L.hipV).add(_a); }
    if (d < 1e-7) { _a.set(0, L.side, -1); d = 1e-7; }
    L.foot.copy(L.ankle).addScaledVector(L.dir, L.Lta * MM);
    const Lf = L.Lf * MM, Lt = L.Lt * MM;
    const a = (Lf * Lf - Lt * Lt + d * d) / (2 * d), hk = Math.sqrt(Math.max(0, Lf * Lf - a * a));
    _b.copy(_a).multiplyScalar(1 / d);                                   // hip to ankle
    _c.copy(bulge).addScaledVector(_b, -bulge.dot(_b));                  // the bulge, made perpendicular
    if (_c.lengthSq() < 1e-12) _c.set(0, L.side, 0).addScaledVector(_b, -_b.y * L.side);
    _c.normalize();
    L.kneeV.copy(L.hipV).addScaledVector(_b, a).addScaledVector(_c, hk);
    putSeg(L.hipV, L.kneeV); putSeg(L.kneeV, L.ankle); putSeg(L.ankle, L.foot);
    putPt(L.kneeV); putPt(L.ankle); putPt(L.foot);
  }

  function update(flight, c, dt, surfaceZ, showLocator, pose) {
    const [x, y, z] = flight.p, q = flight.q;
    group.position.set(x, y, z);
    group.quaternion.set(q[1], q[2], q[3], q[0]);
    if (!pose) pose = { mode: flight.landed ? "landed" : "flying", t: flight.t, groom: null, feed: 0, hunger: 0 };
    const t = pose.t || 0, mode = pose.mode || "flying";
    const sitting = mode === "landed" || mode === "grooming" || mode === "feeding";

    // mode changes: a takeoff starts the jump; a landing starts the fold
    if (mode !== S.mode) {
      if (mode === "takeoff" || (mode === "flying" && S.mode !== "flying")) S.takeoff = 0;
      if (mode === "grooming") { S.groomSide = Math.sin(t * 977.7) > 0 ? 1 : -1; S.groomT = t; }
      S.mode = mode;
    }
    if (S.takeoff >= 0) { S.takeoff += dt; if (S.takeoff > 1) S.takeoff = -1; }
    const jumping = S.takeoff >= 0 && S.takeoff < TAKEOFF_S;
    const standWant = sitting || jumping ? 1 : 0;
    S.stand += Math.sign(standWant - S.stand) * Math.min(Math.abs(standWant - S.stand), dt / FOLD_S);
    const stand = smooth(S.stand), flying = 1 - stand;
    S.feed += ((mode === "feeding" ? clamp01(pose.feed ?? 1) : 0) - S.feed) * (1 - Math.exp(-dt / 0.25));

    // the surface plane in the body group's frame; the body itself stays at the physics centre
    const zs = surfaceZ(x, y), h = Math.max(0, z - zs);
    n.set(0, 0, 1).applyQuaternion(qi.copy(group.quaternion).invert());   // world up in the fly's frame
    body.position.set(0.04 * MM * S.feed, 0, 0);
    body.rotation.set(0, 0.08 * S.feed, 0);
    qb.setFromEuler(body.rotation); nb.copy(n).applyQuaternion(qi.copy(qb).invert());
    tb.copy(body.position); dist = -h - n.dot(tb);

    // the head: idle wiggles when sitting, steady in flight, lowered when grooming or feeding
    S.headNext -= dt;
    if (S.headNext <= 0) {
      S.headNext = 0.7 + 1.8 * Math.random();
      S.wantYaw = (Math.random() - 0.5) * 0.7; S.wantPitch = (Math.random() - 0.5) * 0.35;
      S.headTau = Math.random() < 0.3 ? 0.05 : 0.35;
    }
    const groom = mode === "grooming" ? (pose.groom || "head") : null;
    let yawT = sitting ? S.wantYaw : 0, pitchT = sitting ? S.wantPitch : 0;
    if (groom === "head") { pitchT = 0.32 + 0.06 * Math.sin(t * 9); yawT = 0.1 * Math.sin(t * 4.4); }
    if (groom === "legs") pitchT = 0.18;
    if (mode === "feeding") pitchT = 0.3 + 0.05 * S.feed * Math.sin(t * 19);
    const kh = 1 - Math.exp(-dt / (sitting ? S.headTau : 0.15));
    S.headYaw += (yawT - S.headYaw) * kh; S.headPitch += (pitchT - S.headPitch) * kh;
    head.rotation.set(0, S.headPitch, S.headYaw);
    head.updateMatrix();

    // the abdomen breathes when sitting
    const breath = 1 + 0.03 * stand * Math.sin(TWO_PI * 1.1 * t);
    abdomen.scale.set(breath, 1, 0.82 * breath);

    segI = 0; ptI = 0;

    // the legs
    const jump = S.takeoff >= 0 ? Math.sin(Math.PI * Math.min(1, S.takeoff / TAKEOFF_S)) * Math.max(0, 1 - Math.max(0, S.takeoff - TAKEOFF_S) / 0.1) : 0;
    const gs = S.groomSide, gt = t - S.groomT;
    for (const L of legs) {
      const s = L.side;
      // the standing foot, on the surface, pushed down by the jump
      const sx = L.stand[0] * MM, sy = s * L.stand[1] * MM;
      _e.set(sx, sy, surfaceAt(sx, sy) - 0.45 * MM * jump);
      L.dir.set(sx - L.hipV.x, sy - L.hipV.y, 0).normalize(); L.bulge.set(0, s, 0.7);
      let tarsusDir = _d.copy(L.dir);
      if (groom === "head" && L.name === "front") {
        const ph = 0.5 - 0.5 * Math.cos(TWO_PI * 2.2 * gt + (s > 0 ? 0 : Math.PI));
        _a.set(0.72 * MM, s * 0.42 * MM, 0.4 * MM); _b.set(1.22 * MM, s * 0.06 * MM, -0.22 * MM);
        _e.lerpVectors(_a, _b, ph); tarsusDir = _d.subVectors(_b, _a).normalize(); L.bulge.set(0.3, s, 0.8);
      } else if (groom === "wings" && L.name === "hind" && s === gs) {
        const ph = 0.5 - 0.5 * Math.cos(TWO_PI * 1.8 * gt);
        _a.set(-0.15 * MM, s * 0.34 * MM, 0.5 * MM); _b.set(-1.6 * MM, s * 0.14 * MM, 0.12 * MM);
        _e.lerpVectors(_a, _b, ph); tarsusDir = _d.subVectors(_b, _a).normalize(); L.bulge.set(-0.2, s, 0.9);
      } else if (groom === "legs" && L.name === "front") {
        const w = TWO_PI * 3 * gt;
        _e.set((1.05 + 0.05 * Math.cos(w)) * MM, s * 0.1 * MM, (-0.5 + s * 0.18 * Math.sin(w)) * MM);
        tarsusDir = _d.set(0.5, -s * 0.4, -0.75).normalize(); L.bulge.set(0.4, s, 0.5);
      }
      if (stand < 0.999) {                              // blend with the tucked foot
        _a.set(L.tuck[0] * MM, s * L.tuck[1] * MM, L.tuck[2] * MM);
        _e.lerp(_a, 1 - stand);
        _b.set(L.tuckDir[0], s * L.tuckDir[1], L.tuckDir[2]).normalize();
        tarsusDir.lerp(_b, 1 - stand);
        _c.set(L.tuckBulge[0], s * L.tuckBulge[1], L.tuckBulge[2]);
        L.bulge.lerp(_c, 1 - stand);
      }
      solveLeg(L, _e, tarsusDir, L.bulge);
    }

    // the proboscis: folded under the head, or extended to the surface and pumping
    P0.set(0.08 * MM, 0, -0.24 * MM).applyMatrix4(head.matrix);
    fold1.copy(P0).add(_a.set(-0.12 * MM, 0, -0.14 * MM)); fold2.copy(fold1).add(_a.set(0.16 * MM, 0, -0.06 * MM));
    const below = surfaceAt(P0.x, P0.y), reach = Math.min(1.1 * MM, Math.max(0.25 * MM, P0.z - below));
    const pump = S.feed > 0.3 ? 0.06 * Math.sin(TWO_PI * 3.2 * t) : 0;
    ext1.copy(P0).add(_a.set(0.06 * MM, 0, -0.42 * reach)); ext2.copy(P0).add(_a.set(0.08 * MM, 0, -reach * (1 - 0.5 * pump)));
    const f = smooth(S.feed);
    P1.lerpVectors(fold1, ext1, f); P2.lerpVectors(fold2, ext2, f);
    putSeg(P0, P1); putSeg(P1, P2); putPt(P2);

    // the antennae: pedicel, funiculus, arista, twitching from their bases
    for (const A of antennae) {
      A.next -= dt;
      if (A.next <= 0) { A.next = 0.5 + 1.5 * Math.random(); A.twitch = 0.25; }
      A.twitch = Math.max(0, A.twitch - dt);
      const k = Math.sin(Math.PI * Math.min(1, A.twitch / 0.25)), s = A.side;
      hp.set(0.24 * MM, s * 0.085 * MM, -0.02 * MM);
      hq.set((0.3 + 0.03 * k) * MM, s * (0.1 + 0.05 * k) * MM, (-0.07 + 0.03 * k) * MM);
      hr.set((0.35 + 0.04 * k) * MM, s * (0.11 + 0.07 * k) * MM, (-0.15 + 0.05 * k) * MM);
      hs.copy(hr).add(_a.set(0.75, s * 0.35, 0.55).normalize().multiplyScalar(0.28 * MM));
      for (const v of [hp, hq, hr, hs]) v.applyMatrix4(head.matrix);
      putSeg(hp, hq); putSeg(hq, hr); putSeg(hr, hs); putPt(hr);
    }
    // the ocelli on the top of the head
    for (const [ox, oy, oz] of [[0.02, 0, 0.285], [-0.06, 0.06, 0.27], [-0.06, -0.06, 0.27]]) putPt(_a.set(ox * MM, oy * MM, oz * MM).applyMatrix4(head.matrix));

    // the halteres, beating against the wings while flying
    const phase = flight.phase, flicker = 0.5 + 0.5 * Math.sin(2.0 * phase);
    const wingsOn = (c.aL + c.aR) > 0.02;
    for (const H of halteres) {
      const ang = -H.side * 0.8 * Math.sin(phase) * flying * (wingsOn ? 1 : 0), ca = Math.cos(ang), sa = Math.sin(ang);
      _a.set(H.dir.x, H.dir.y * ca - H.dir.z * sa, H.dir.y * sa + H.dir.z * ca).multiplyScalar(0.36 * MM).add(H.base);
      putSeg(H.base, _a); putPt(_a);
    }
    for (const B of bristles) putSeg(B.a, B.b);
    lineGeo.attributes.position.needsUpdate = true; ptGeo.attributes.position.needsUpdate = true;

    // the wings: beating while flying, folded flat over the abdomen while sitting
    for (const w of wings) {
      const a = w.side > 0 ? c.aL : c.aR, beta = w.side > 0 ? c.betaL : c.betaR, sft = w.side > 0 ? c.sL : c.sR;
      const sweep = Math.min(SWEEP_MAX, SWEEP_HOVER * a);
      const groomed = groom === "wings" && w.side === gs ? 1 : 0;
      w.pivot.position.x = sft * flying;
      w.tilt.rotation.y = beta * flying + stand * (-0.08 - 0.3 * groomed);
      const mid = w.side * Math.PI / 2;
      // the sweep wedge, its angle following the amplitude
      fan[0] = 0; fan[1] = 0; fan[2] = 0;
      for (let k = 0; k <= FAN_SEGMENTS; k++) {
        const th = mid - w.side * sweep * (2 * k / FAN_SEGMENTS - 1);
        fan[3 * (k + 1)] = WING_LENGTH * Math.cos(th); fan[3 * (k + 1) + 1] = WING_LENGTH * Math.sin(th); fan[3 * (k + 1) + 2] = 0;
      }
      w.fan.geometry.attributes.position.array.set(fan);
      w.fan.geometry.attributes.position.needsUpdate = true;
      w.fan.material.opacity = wingsOn ? (0.07 + 0.06 * Math.min(1, a) + 0.03 * flicker) * flying : 0;
      w.fan.visible = w.fan.material.opacity > 0.005;
      // the wing itself: its stroke angle in flight, the folded angle at rest
      const strokeAngle = -w.side * sweep * Math.sin(phase);
      const foldAngle = w.side * (Math.PI / 2 + (w.side > 0 ? 0.34 : 0.18) - 0.3 * groomed);
      w.blade.rotation.z = strokeAngle * flying + foldAngle * stand;
      w.blade.position.z = stand * (w.side > 0 ? 0.06 : 0.03) * MM;
      w.lines.material.opacity = (0.55 + 0.25 * flicker) * flying + 0.9 * stand;
      w.fill.material.opacity = 0.09 * flying + 0.14 * stand;
    }

    // the light on the surface and the locator
    shadow.position.set(x, y, zs + 0.0004);
    const r = 0.0022 + 0.004 * h;
    shadow.scale.set(r, r, 1);
    shadow.material.opacity = 0.45 / (1 + 5 * h);
    locator.position.set(x, y, z);
    locator.visible = showLocator;
  }

  return { group, shadow, locator, wings, update, legs };
}
