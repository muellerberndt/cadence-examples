// A fly in the Matrix: the room as green wireframe in a black void, the two fruits and the
// sugar, the cameras and the inset renderer. three.js, metres.
//
// Physics coordinates are x along the long wall, y along the short wall, z up. Everything in the
// room lives inside `world`, a group rotated so those coordinates apply unchanged; the cameras
// live in three.js coordinates, converted with toThree().
//
// The room animates itself: the digital rain, the dust motes, the sugar's sparkle and the
// highlight ring advance in an onBeforeRender hook, so a page does not have to call anything
// per frame. `room.update(dt)` is there for pages that prefer to drive it.
import * as THREE from "three";
import { EffectComposer } from "three/addons/postprocessing/EffectComposer.js";
import { RenderPass } from "three/addons/postprocessing/RenderPass.js";
import { UnrealBloomPass } from "three/addons/postprocessing/UnrealBloomPass.js";
import { OutputPass } from "three/addons/postprocessing/OutputPass.js";
import { ROOM } from "./body.js";

export const toThree = (x, y, z) => new THREE.Vector3(x, z, -y);
export const fromThree = (v) => [v.x, -v.z, v.y];

export const MATRIX = {
  background: 0x010603, edge: 0x39ff6a, bright: 0x9dffb0, dim: 0x1f8a3c, fill: 0x03150a,
  banana: 0xb8ff5a, bread: 0xa8ffc8, sugar: 0xe8fff0, glow: 0x39ff6a, rain: "#39ff6a", rainHead: "#d8ffe4",
};
// the old name, kept for imports
export const PALETTE = MATRIX;

const TWO_PI = Math.PI * 2;

// -- materials --------------------------------------------------------------------------------
// Lines are additive and never write depth; fills are dim, write depth and draw first, so the
// body of a shape hides the lines behind it.

const lineCache = new Map(), fillCache = new Map();
export function lineMaterial(color = MATRIX.edge, opacity = 0.9) {
  const key = color + ":" + opacity;
  if (!lineCache.has(key)) lineCache.set(key, new THREE.LineBasicMaterial({ color, transparent: true, opacity, blending: THREE.AdditiveBlending, depthWrite: false }));
  return lineCache.get(key);
}
export function fillMaterial(color = MATRIX.fill, opacity = 0.7) {
  const key = color + ":" + opacity;
  if (!fillCache.has(key)) fillCache.set(key, new THREE.MeshLambertMaterial({ color, emissive: color, emissiveIntensity: 0.6, transparent: true, opacity, depthWrite: true }));
  return fillCache.get(key);
}

// A shape as a dim fill and glowing edges. Returns the group with .fill and .edges.
export function wire(geometry, edgeColor = MATRIX.edge, edgeOpacity = 0.9, fillColor = MATRIX.fill, fillOpacity = 0.7, threshold = 1) {
  const g = new THREE.Group();
  g.fill = new THREE.Mesh(geometry, fillMaterial(fillColor, fillOpacity)); g.fill.renderOrder = 0;
  g.edges = new THREE.LineSegments(new THREE.EdgesGeometry(geometry, threshold), lineMaterial(edgeColor, edgeOpacity)); g.edges.renderOrder = 1;
  g.add(g.fill, g.edges);
  return g;
}

// Line segments from a flat array of xyz pairs.
export function segments(points, color = MATRIX.edge, opacity = 0.9, material = null) {
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(points, 3));
  const l = new THREE.LineSegments(g, material || lineMaterial(color, opacity)); l.renderOrder = 1;
  return l;
}

const seg = (out, a, b) => { out.push(a[0], a[1], a[2], b[0], b[1], b[2]); };
const ring = (out, cx, cy, z, r, n = 32) => { for (let i = 0; i < n; i++) { const a0 = i / n * TWO_PI, a1 = (i + 1) / n * TWO_PI; seg(out, [cx + r * Math.cos(a0), cy + r * Math.sin(a0), z], [cx + r * Math.cos(a1), cy + r * Math.sin(a1), z]); } };
const boxEdges = (out, x0, x1, y0, y1, z0, z1) => {
  for (const z of [z0, z1]) { seg(out, [x0, y0, z], [x1, y0, z]); seg(out, [x1, y0, z], [x1, y1, z]); seg(out, [x1, y1, z], [x0, y1, z]); seg(out, [x0, y1, z], [x0, y0, z]); }
  for (const [x, y] of [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) seg(out, [x, y, z0], [x, y, z1]);
};

// -- textures ---------------------------------------------------------------------------------

export function canvasTexture(w, h, draw, srgb = true) {
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  draw(c.getContext("2d"), w, h);
  const t = new THREE.CanvasTexture(c);
  if (srgb) t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

export function hash(n) { const s = Math.sin(n * 127.1 + 311.7) * 43758.5453; return s - Math.floor(s); }

export function radialTexture(inner, outer) {
  return canvasTexture(128, 128, (g, w, h) => {
    const r = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    r.addColorStop(0, inner); r.addColorStop(1, outer);
    g.fillStyle = r; g.fillRect(0, 0, w, h);
  });
}

export function dotTexture() {
  return canvasTexture(64, 64, (g, w, h) => {
    const r = g.createRadialGradient(w / 2, h / 2, 0, w / 2, h / 2, w / 2);
    r.addColorStop(0, "rgba(255,255,255,1)"); r.addColorStop(0.3, "rgba(255,255,255,.7)"); r.addColorStop(1, "rgba(255,255,255,0)");
    g.fillStyle = r; g.fillRect(0, 0, w, h);
  }, false);
}

// The digital rain: columns of glyphs falling down a canvas, stepped about ten times a second.
function digitalRain(cols = 72, rows = 44) {
  const cw = 10, ch = 14, W = cols * cw, H = rows * ch;
  const c = document.createElement("canvas"); c.width = W; c.height = H;
  const g = c.getContext("2d");
  g.fillStyle = "#000"; g.fillRect(0, 0, W, H);
  const glyphs = "ｱｲｳｴｵｶｷｸｹｺｻｼｽｾｿﾀﾁﾂﾃﾄﾅﾆﾇﾈﾉﾊﾋﾌﾍﾎﾏﾐﾑﾒﾓﾔﾕﾖﾗﾘﾙﾚﾛﾜﾝ0123456789Z:・=*+-<>";
  const heads = new Float32Array(cols), speeds = new Float32Array(cols), tails = new Float32Array(cols);
  for (let i = 0; i < cols; i++) { heads[i] = -Math.floor(hash(i) * rows * 2); speeds[i] = 0.6 + 0.9 * hash(i + 99); tails[i] = 8 + 14 * hash(i + 199); }
  const tex = new THREE.CanvasTexture(c); tex.colorSpace = THREE.SRGBColorSpace;
  const pick = () => glyphs[Math.floor(Math.random() * glyphs.length)];
  const step = () => {
    g.fillStyle = "rgba(0,0,0,0.14)"; g.fillRect(0, 0, W, H);
    g.font = "bold 12px ui-monospace, Menlo, monospace"; g.textAlign = "center"; g.textBaseline = "top";
    for (let i = 0; i < cols; i++) {
      const row = Math.floor(heads[i]);
      if (row >= 0 && row < rows) {
        g.fillStyle = MATRIX.rain; g.fillText(pick(), i * cw + cw / 2, (row - 1) * ch);
        g.fillStyle = MATRIX.rainHead; g.fillText(pick(), i * cw + cw / 2, row * ch);
      }
      if (Math.random() < 0.06) { const rr = Math.floor(Math.random() * rows); g.fillStyle = "rgba(57,255,106,.55)"; g.fillText(pick(), i * cw + cw / 2, rr * ch); }
      heads[i] += speeds[i];
      if (heads[i] > rows + tails[i]) { heads[i] = -Math.floor(Math.random() * rows); speeds[i] = 0.6 + 0.9 * Math.random(); }
    }
    tex.needsUpdate = true;
  };
  // a stutter: the canvas jumps a few cells and a bright bar tears across it
  const stutter = () => {
    const dx = Math.floor((Math.random() - 0.5) * 4) * cw, dy = Math.floor((Math.random() - 0.5) * 6) * ch;
    g.drawImage(c, dx, dy);
    if (Math.random() < 0.6) { g.fillStyle = "rgba(57,255,106,.3)"; g.fillRect(0, Math.random() * H, W, 4 + Math.random() * 18); }
    tex.needsUpdate = true;
  };
  for (let k = 0; k < 40; k++) step();
  return { tex, step, stutter };
}

// -- geometry ---------------------------------------------------------------------------------

// A tube swept along a curve whose radius varies along it and around it: radiusAt(t, theta).
export function sweptTube(curve, radiusAt, segs = 14, radial = 8) {
  const pts = curve.getPoints(segs), frames = curve.computeFrenetFrames(segs, false);
  const pos = [], uv = [], idx = [];
  for (let i = 0; i <= segs; i++) {
    const t = i / segs, P = pts[i], N = frames.normals[i], B = frames.binormals[i];
    for (let j = 0; j <= radial; j++) {
      const th = j / radial * TWO_PI, r = radiusAt(t, th);
      pos.push(P.x + r * (Math.cos(th) * N.x + Math.sin(th) * B.x), P.y + r * (Math.cos(th) * N.y + Math.sin(th) * B.y), P.z + r * (Math.cos(th) * N.z + Math.sin(th) * B.z));
      uv.push(t, j / radial);
    }
  }
  for (let i = 0; i < segs; i++) for (let j = 0; j < radial; j++) {
    const a = i * (radial + 1) + j, b = a + radial + 1;
    idx.push(a, b, a + 1, b, b + 1, a + 1);
  }
  const g = new THREE.BufferGeometry();
  g.setAttribute("position", new THREE.Float32BufferAttribute(pos, 3));
  g.setAttribute("uv", new THREE.Float32BufferAttribute(uv, 2));
  g.setIndex(idx); g.computeVertexNormals();
  return g;
}

// An ellipsoid as rings and meridians, its bottom flattened; local z up.
function ellipsoidWire(rx, ry, rz, rings = 6, meridians = 12, floor = -0.4) {
  const out = [];
  const P = (u, v) => { const ph = -Math.PI / 2 + Math.PI * u, th = TWO_PI * v; let z = Math.sin(ph); if (z < floor) z = floor + (z - floor) * 0.25; const c = Math.cos(ph); return [rx * c * Math.cos(th), ry * c * Math.sin(th), rz * z]; };
  for (let i = 1; i < rings; i++) for (let j = 0; j < meridians * 2; j++) seg(out, P(i / rings, j / (meridians * 2)), P(i / rings, (j + 1) / (meridians * 2)));
  for (let j = 0; j < meridians; j++) for (let i = 0; i < rings * 2; i++) seg(out, P(i / (rings * 2), j / meridians), P((i + 1) / (rings * 2), j / meridians));
  return out;
}

// -- the room ---------------------------------------------------------------------------------

export function createRoom(world) {
  const [W, D, H] = ROOM;
  const room = { table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: 0.75 }, window: { x0: 1.3, x1: 2.7, z0: 0.9, z1: 2.1 } };
  const animations = [];                                   // (dt, time) => void, run once per frame
  const glitchables = [];                                  // { name, objects }: what a glitch may hit

  // the void: a black shell far outside the room, so the page's background never shows
  const voidBox = new THREE.Mesh(new THREE.BoxGeometry(60, 60, 60), new THREE.MeshBasicMaterial({ color: MATRIX.background, side: THREE.BackSide }));
  voidBox.position.set(W / 2, D / 2, H / 2); world.add(voidBox);

  // the floor: a grid, minor lines every 10 cm, major every 50 cm
  {
    const minor = [], major = [];
    for (let i = 0; i <= Math.round(W / 0.1); i++) { const x = i * 0.1; seg(i % 5 ? minor : major, [x, 0, 0], [x, D, 0]); }
    for (let j = 0; j <= Math.round(D / 0.1); j++) { const y = j * 0.1; seg(j % 5 ? minor : major, [0, y, 0], [W, y, 0]); }
    const a = segments(minor, MATRIX.edge, 0.16), b = segments(major, MATRIX.edge, 0.5);
    world.add(a, b); glitchables.push({ name: "floor", objects: [a, b] });
  }
  // the walls: the room's edges, and sparse lines across the walls
  {
    const edges = [], sparse = [];
    boxEdges(edges, 0, W, 0, D, 0, H);
    for (const z of [0.5, 1.0, 1.5, 2.0]) { seg(sparse, [0, 0, z], [W, 0, z]); seg(sparse, [0, D, z], [W, D, z]); seg(sparse, [0, 0, z], [0, D, z]); seg(sparse, [W, 0, z], [W, D, z]); }
    for (let x = 1; x < W; x++) { seg(sparse, [x, 0, 0], [x, 0, H]); seg(sparse, [x, D, 0], [x, D, H]); }
    for (let y = 1; y < D; y++) { seg(sparse, [0, y, 0], [0, y, H]); seg(sparse, [W, y, 0], [W, y, H]); }
    const a = segments(edges, MATRIX.edge, 0.7), b = segments(sparse, MATRIX.edge, 0.14);
    world.add(a, b); glitchables.push({ name: "walls", objects: [a, b] });
  }
  // the window in the far wall: a bright frame, mullions, a sill, and a faint pane
  {
    const wn = room.window, out = [], wx = (wn.x0 + wn.x1) / 2, wz = (wn.z0 + wn.z1) / 2;
    boxEdges(out, wn.x0, wn.x1, D - 0.04, D, wn.z0, wn.z1);
    seg(out, [wx, D - 0.02, wn.z0], [wx, D - 0.02, wn.z1]); seg(out, [wn.x0, D - 0.02, wz + 0.12], [wn.x1, D - 0.02, wz + 0.12]);
    seg(out, [wn.x0 - 0.15, D - 0.1, wn.z0 - 0.06], [wn.x1 + 0.15, D - 0.1, wn.z0 - 0.06]); seg(out, [wn.x0 - 0.15, D, wn.z0 - 0.06], [wn.x1 + 0.15, D, wn.z0 - 0.06]);
    const frame = segments(out, MATRIX.bright, 0.95); world.add(frame); glitchables.push({ name: "window", objects: [frame] });
    const pane = new THREE.Mesh(new THREE.PlaneGeometry(wn.x1 - wn.x0, wn.z1 - wn.z0), new THREE.MeshBasicMaterial({ color: MATRIX.edge, transparent: true, opacity: 0.05, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide }));
    pane.rotation.x = Math.PI / 2; pane.position.set(wx, D - 0.01, wz); pane.renderOrder = 1; world.add(pane);
  }
  // the digital rain on the two walls the cameras face most: x = W and y = 0
  {
    const rain = digitalRain();
    const mat = new THREE.MeshBasicMaterial({ map: rain.tex, transparent: true, opacity: 0.55, blending: THREE.AdditiveBlending, depthWrite: false, side: THREE.DoubleSide });
    const a = new THREE.Mesh(new THREE.PlaneGeometry(D - 0.1, H - 0.1), mat);
    a.rotation.set(Math.PI / 2, 0, Math.PI / 2, "ZYX"); a.position.set(W - 0.01, D / 2, H / 2); a.renderOrder = 1; world.add(a);
    const b = new THREE.Mesh(new THREE.PlaneGeometry(W - 0.1, H - 0.1), mat);
    b.rotation.x = Math.PI / 2; b.position.set(W / 2, 0.01, H / 2); b.renderOrder = 1; world.add(b);
    let due = 0;
    animations.push((dt) => { due += dt; if (due >= 0.1) { due = 0; rain.step(); } });
    room.rain = rain;
    glitchables.push({ name: "rain", objects: [a, b] });
  }

  // the table: a wireframe slab on four legs, its top a fine grid and a dim fill to raycast against
  const tb = room.table, tw = tb.x1 - tb.x0, td = tb.y1 - tb.y0, tx = (tb.x0 + tb.x1) / 2, ty = (tb.y0 + tb.y1) / 2;
  {
    const out = [], grid = [];
    boxEdges(out, tb.x0, tb.x1, tb.y0, tb.y1, tb.z - 0.035, tb.z);
    for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) { const x = tx + sx * (tw / 2 - 0.06), y = ty + sy * (td / 2 - 0.06); seg(out, [x, y, 0], [x, y, tb.z - 0.035]); }
    for (let i = 1; i < Math.round(tw / 0.05); i++) seg(grid, [tb.x0 + i * 0.05, tb.y0, tb.z + 0.0005], [tb.x0 + i * 0.05, tb.y1, tb.z + 0.0005]);
    for (let j = 1; j < Math.round(td / 0.05); j++) seg(grid, [tb.x0, tb.y0 + j * 0.05, tb.z + 0.0005], [tb.x1, tb.y0 + j * 0.05, tb.z + 0.0005]);
    const edgesL = segments(out, MATRIX.edge, 0.8), gridL = segments(grid, MATRIX.edge, 0.22);
    world.add(edgesL, gridL);
    const top = new THREE.Mesh(new THREE.BoxGeometry(tw, td, 0.035), fillMaterial(MATRIX.fill, 0.85));
    top.position.set(tx, ty, tb.z - 0.0175); top.renderOrder = 0; world.add(top);
    room.tableMesh = top;
    glitchables.push({ name: "table", objects: [edgesL, gridL, top] });
  }
  // the chair at the near side of the table
  {
    const out = [], cx = tx, cy = tb.y0 - 0.42;
    boxEdges(out, cx - 0.21, cx + 0.21, cy - 0.21, cy + 0.21, 0.43, 0.46);
    for (const [sx, sy] of [[-1, -1], [1, -1], [-1, 1], [1, 1]]) seg(out, [cx + sx * 0.18, cy + sy * 0.18, 0], [cx + sx * 0.18, cy + sy * 0.18, 0.43]);
    for (const sx of [-1, 1]) seg(out, [cx + sx * 0.18, cy - 0.19, 0.46], [cx + sx * 0.18, cy - 0.19, 0.95]);
    for (const z of [0.7, 0.85]) seg(out, [cx - 0.18, cy - 0.19, z], [cx + 0.18, cy - 0.19, z]);
    world.add(segments(out, MATRIX.edge, 0.55));
  }

  // the ceiling lamp: cord, a wireframe shade, a bright bulb, a soft green light
  {
    const lamp = new THREE.Group(); lamp.position.set(W / 2, D / 2, H);
    const shade = new THREE.LineSegments(new THREE.EdgesGeometry(new THREE.CylinderGeometry(0.07, 0.2, 0.17, 14, 1, true), 1), lineMaterial(MATRIX.edge, 0.8));
    shade.rotation.x = Math.PI / 2; shade.position.z = -0.47; shade.renderOrder = 1; lamp.add(shade);
    lamp.add(segments([0, 0, 0, 0, 0, -0.39], MATRIX.edge, 0.5));
    const bulb = new THREE.Points(new THREE.BufferGeometry().setAttribute("position", new THREE.Float32BufferAttribute([0, 0, -0.5], 3)), new THREE.PointsMaterial({ color: MATRIX.bright, size: 9, sizeAttenuation: false, map: dotTexture(), transparent: true, blending: THREE.AdditiveBlending, depthWrite: false }));
    bulb.renderOrder = 2; lamp.add(bulb);
    const glow = new THREE.Sprite(new THREE.SpriteMaterial({ map: radialTexture("rgba(57,255,106,.5)", "rgba(57,255,106,0)"), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.8 }));
    glow.scale.set(0.6, 0.6, 1); glow.position.z = -0.52; lamp.add(glow);
    const light = new THREE.PointLight(MATRIX.edge, 6, 0, 2); light.position.z = -0.52; lamp.add(light);
    world.add(lamp);
    room.lamp = { group: lamp, light, position: [W / 2, D / 2, H - 0.5] };
  }
  world.add(new THREE.AmbientLight(MATRIX.edge, 0.5));

  // dust motes drifting through the room
  {
    const N = 360, pos = new Float32Array(3 * N), vel = new Float32Array(3 * N), ph = new Float32Array(N);
    for (let i = 0; i < N; i++) { pos[3 * i] = hash(i) * W; pos[3 * i + 1] = hash(i + N) * D; pos[3 * i + 2] = hash(i + 2 * N) * H; vel[3 * i] = 0.01 * (hash(i + 3 * N) - 0.5); vel[3 * i + 1] = 0.01 * (hash(i + 4 * N) - 0.5); vel[3 * i + 2] = -0.004 - 0.01 * hash(i + 5 * N); ph[i] = TWO_PI * hash(i + 6 * N); }
    const geo = new THREE.BufferGeometry(); geo.setAttribute("position", new THREE.BufferAttribute(pos, 3));
    const motes = new THREE.Points(geo, new THREE.PointsMaterial({ color: MATRIX.edge, size: 0.012, map: dotTexture(), transparent: true, opacity: 0.55, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true }));
    motes.frustumCulled = false; motes.renderOrder = 2; world.add(motes);
    animations.push((dt, time) => {
      for (let i = 0; i < N; i++) {
        pos[3 * i] += (vel[3 * i] + 0.004 * Math.sin(time * 0.2 + ph[i])) * dt; pos[3 * i + 1] += (vel[3 * i + 1] + 0.004 * Math.cos(time * 0.17 + ph[i])) * dt; pos[3 * i + 2] += vel[3 * i + 2] * dt;
        if (pos[3 * i] < 0) pos[3 * i] += W; if (pos[3 * i] > W) pos[3 * i] -= W; if (pos[3 * i + 1] < 0) pos[3 * i + 1] += D; if (pos[3 * i + 1] > D) pos[3 * i + 1] -= D; if (pos[3 * i + 2] < 0) pos[3 * i + 2] += H;
      }
      geo.attributes.position.needsUpdate = true;
    });
    let lastTime = -1;                                        // the room's clock, once per frame
    motes.onBeforeRender = () => {
      const now = performance.now() / 1000;
      if (lastTime < 0) { lastTime = now; return; }
      const dt = Math.min(0.05, now - lastTime);
      if (dt <= 0) return;
      lastTime = now;
      for (const a of animations) a(dt, now);
    };
    room.update = (dt) => { for (const a of animations) a(dt, performance.now() / 1000); };
  }

  // -- the fruits ------------------------------------------------------------------------------
  const fruits = {};
  const saucer = (r) => { const out = []; ring(out, 0, 0, 0.0, r * 0.55, 24); ring(out, 0, 0, 0.012, r, 32); for (let k = 0; k < 8; k++) { const a = k / 8 * TWO_PI; seg(out, [r * 0.55 * Math.cos(a), r * 0.55 * Math.sin(a), 0], [r * Math.cos(a), r * Math.sin(a), 0.012]); } return segments(out, MATRIX.edge, 0.6); };

  // the banana: a bent wireframe capsule lying on its side on a saucer
  const bananaGroup = new THREE.Group();
  const bananaCurve = new THREE.CubicBezierCurve3(new THREE.Vector3(-0.085, 0, 0), new THREE.Vector3(-0.03, 0.05, 0), new THREE.Vector3(0.03, 0.05, 0), new THREE.Vector3(0.085, 0, 0));
  const BANANA_R = 0.0165;
  const bananaRadius = (u) => {
    let s = 1;
    if (u < 0.1) s = 0.32 + 0.68 * (0.5 - 0.5 * Math.cos(Math.PI * u / 0.1));
    else if (u > 0.84) s = 1 - 0.72 * Math.pow((u - 0.84) / 0.16, 1.4);
    return BANANA_R * s;
  };
  const bananaGeo = sweptTube(bananaCurve, bananaRadius, 16, 7);
  const bananaWire = wire(bananaGeo, MATRIX.banana, 0.85, 0x0a1a03, 0.7);
  bananaWire.position.z = 0.012 + BANANA_R; bananaGroup.add(bananaWire); bananaGroup.add(saucer(0.11));
  bananaGroup.rotation.z = -0.35;
  world.add(bananaGroup);
  const bananaTopOffset = 0.012 + 2 * BANANA_R;
  const bananaMidLocal = bananaCurve.getPoint(0.5);        // the landing spot, in the banana's frame
  let bananaSamples = [];
  const bananaFootprint = () => { bananaGroup.updateMatrixWorld(true); bananaSamples = []; for (let i = 0; i <= 24; i++) { const u = i / 24, p = bananaCurve.getPoint(u).applyMatrix4(bananaGroup.matrix); bananaSamples.push([p.x, p.y, bananaRadius(u)]); } };
  fruits.banana = { pos: [0, 0, tb.z + bananaTopOffset], group: bananaGroup, label: "banana", odour: "decaying fruit", offset: new THREE.Vector3(bananaMidLocal.x, bananaMidLocal.y, 0).applyAxisAngle(new THREE.Vector3(0, 0, 1), bananaGroup.rotation.z) };

  // the bread: a wireframe loaf on its saucer
  const breadGroup = new THREE.Group();
  const LOAF = [0.05, 0.038, 0.032];
  const loafLines = segments(ellipsoidWire(LOAF[0], LOAF[1], LOAF[2], 6, 12), MATRIX.bread, 0.85);
  const loafGeo = new THREE.SphereGeometry(1, 12, 8); loafGeo.rotateX(Math.PI / 2);
  { const p = loafGeo.attributes.position; for (let i = 0; i < p.count; i++) { const z = p.getZ(i); if (z < -0.4) p.setZ(i, -0.4 + (z + 0.4) * 0.25); } }
  const loafFill = new THREE.Mesh(loafGeo, fillMaterial(0x07160c, 0.7)); loafFill.scale.set(...LOAF); loafFill.renderOrder = 0;
  const loaf = new THREE.Group(); loaf.add(loafFill, loafLines); loaf.rotation.z = 0.4; loaf.position.z = 0.012 + 0.4 * LOAF[2];
  breadGroup.add(loaf, saucer(0.075));
  world.add(breadGroup);
  const breadTopOffset = loaf.position.z + LOAF[2];
  fruits.bread = { pos: [0, 0, tb.z + breadTopOffset], group: breadGroup, label: "bread", odour: "yeasty", offset: new THREE.Vector3(0, 0, 0) };
  room.fruits = fruits;
  glitchables.push({ name: "banana", objects: [bananaGroup] }, { name: "bread", objects: [breadGroup] });

  // moves a fruit so that its landing spot is at (x, y) on the table; the pos array is mutated
  // in place so that holders of the reference see the move
  room.placeFruit = (name, xy) => {
    const F = fruits[name]; if (!F) return;
    const x = Math.min(tb.x1 - 0.1, Math.max(tb.x0 + 0.1, xy[0])), y = Math.min(tb.y1 - 0.08, Math.max(tb.y0 + 0.08, xy[1]));
    F.group.position.set(x - F.offset.x, y - F.offset.y, tb.z);
    F.pos[0] = x; F.pos[1] = y;
    if (name === "banana") bananaFootprint();
    if (room.sugar === name) room.setSugar(name);
    if (room.highlighted === name) room.highlightFruit(name);
  };

  // the height of a fruit's surface at (x, y): the banana as a tube around its curve, the loaf as
  // an ellipsoid; the saucer where the point misses the fruit
  room.fruitSurface = (name, x, y) => {
    if (name === "banana") {
      let best = null;
      for (const [sx, sy, rr] of bananaSamples) { const d2 = (x - sx) ** 2 + (y - sy) ** 2; if (best === null || d2 < best[0]) best = [d2, rr]; }
      const [d2, rr] = best, zc = tb.z + 0.012 + BANANA_R;
      return d2 < rr * rr ? zc + Math.sqrt(rr * rr - d2) : tb.z + 0.012;
    }
    const bx = x - breadGroup.position.x, by = y - breadGroup.position.y;
    const c = Math.cos(loaf.rotation.z), sn = Math.sin(loaf.rotation.z), lx = (c * bx + sn * by) / LOAF[0], ly = (-sn * bx + c * by) / LOAF[1];
    const q = 1 - lx * lx - ly * ly;
    return q > 0 ? tb.z + loaf.position.z + LOAF[2] * Math.sqrt(q) : tb.z + 0.012;
  };

  // -- the sugar: a sparkle heap beside the landing spot, crystals scattered on the fruit --------
  const sugar = new THREE.Group(); sugar.visible = false;
  const N_SUGAR = 320, HEAP = { x: 0.009, y: 0.003, r: 0.0045, h: 0.0025 };
  const sugarPos = new Float32Array(3 * N_SUGAR), sugarGeo = new THREE.BufferGeometry(); sugarGeo.setAttribute("position", new THREE.BufferAttribute(sugarPos, 3));
  const sugarPts = new THREE.Points(sugarGeo, new THREE.PointsMaterial({ color: MATRIX.sugar, size: 0.0004, map: dotTexture(), transparent: true, opacity: 0.85, depthWrite: false, blending: THREE.AdditiveBlending, sizeAttenuation: true }));
  sugarPts.renderOrder = 2; sugar.add(sugarPts);
  const heapLines = [];
  for (let k = 1; k <= 3; k++) ring(heapLines, HEAP.x, HEAP.y, HEAP.h * Math.sqrt(1 - (k / 4) ** 2), HEAP.r * k / 4, 16);
  ring(heapLines, HEAP.x, HEAP.y, 0, HEAP.r, 20);
  const heap = segments(heapLines, MATRIX.sugar, 0.7); sugar.add(heap);
  const sugarGlow = new THREE.Sprite(new THREE.SpriteMaterial({ map: radialTexture("rgba(232,255,240,.6)", "rgba(232,255,240,0)"), transparent: true, depthWrite: false, blending: THREE.AdditiveBlending, opacity: 0.8 }));
  sugarGlow.scale.set(0.018, 0.018, 1); sugarGlow.position.set(HEAP.x * 0.8, HEAP.y * 0.8, 0.003); sugar.add(sugarGlow);
  world.add(sugar);
  let sugarFruit = null;
  const surfaceHere = (lx, ly) => (sugarFruit ? room.fruitSurface(sugarFruit, sugar.position.x + lx, sugar.position.y + ly) - sugar.position.z : 0);
  const placeSugar = () => {
    const base = surfaceHere(HEAP.x, HEAP.y); heap.position.z = base;
    for (let i = 0; i < N_SUGAR; i++) {
      let x, y, z;
      if (i < 200) { const u = Math.sqrt(hash(i)), a = TWO_PI * hash(i + 300), rr = HEAP.r * u; x = HEAP.x + rr * Math.cos(a); y = HEAP.y + rr * Math.sin(a); z = base + HEAP.h * Math.sqrt(Math.max(0, 1 - u * u)); }
      else { const rr = 0.003 + 0.011 * Math.sqrt(hash(i + 900)), a = TWO_PI * hash(i + 1200); x = rr * Math.cos(a); y = rr * Math.sin(a); z = surfaceHere(x, y); }
      sugarPos[3 * i] = x; sugarPos[3 * i + 1] = y; sugarPos[3 * i + 2] = z + 0.0002;
    }
    sugarGeo.attributes.position.needsUpdate = true;
  };
  room.sugar = null;
  room.setSugar = (name) => {
    if (!name || !fruits[name]) { sugar.visible = false; room.sugar = null; sugarFruit = null; return; }
    const p = fruits[name].pos; sugar.position.set(p[0], p[1], p[2]); sugarFruit = name; placeSugar(); sugar.visible = true; room.sugar = name;
  };
  animations.push((dt, time) => {
    if (!sugar.visible) return;
    sugarPts.material.opacity = 0.75 + 0.25 * Math.sin(time * 6.3);
    sugarPts.material.size = 0.00032 + 0.00016 * (0.5 + 0.5 * Math.sin(time * 4.1));
    sugarGlow.material.opacity = 0.45 + 0.15 * Math.sin(time * 2.1);
  });

  // the highlight: a bright ring on the table around a fruit, pulsing
  const ringLines = []; ring(ringLines, 0, 0, 0, 1, 48);
  const highlightMat = new THREE.LineBasicMaterial({ color: MATRIX.bright, transparent: true, opacity: 0.9, blending: THREE.AdditiveBlending, depthWrite: false });
  const highlight = segments(ringLines, MATRIX.bright, 0.9, highlightMat); highlight.visible = false; world.add(highlight);
  room.highlighted = null;
  room.highlightFruit = (name) => {
    if (!name || !fruits[name]) { highlight.visible = false; room.highlighted = null; return; }
    const F = fruits[name], p = F.pos, r = name === "banana" ? 0.13 : 0.095;
    highlight.position.set(p[0] - F.offset.x, p[1] - F.offset.y, tb.z + 0.001); highlight.scale.set(r, r, 1);
    highlight.visible = true; room.highlighted = name;
  };
  animations.push((dt, time) => { if (highlight.visible) highlightMat.opacity = 0.6 + 0.35 * Math.sin(time * 3.0); });

  // glitches: every 15 to 50 s one element of the room jitters by a few centimetres or flickers
  // for 80 to 200 ms, or the digital rain stutters; room.glitch(name) fires one on demand
  let glitchState = null, nextGlitch = 15 + 35 * Math.random();
  room.glitching = null;
  room.glitch = (name = null, kind = null, duration = null) => {
    const G = name ? glitchables.find((g) => g.name === name) : glitchables[Math.floor(Math.random() * glitchables.length)];
    if (!G) return null;
    const kinds = G.name === "rain" ? ["stutter"] : ["jitter", "jitter", "flicker"];
    kind = kind || kinds[Math.floor(Math.random() * kinds.length)];
    if (glitchState) endGlitch();
    glitchState = { G, kind, left: duration || 0.08 + 0.12 * Math.random(), saved: G.objects.map((o) => o.position.clone()), tick: 0 };
    room.glitching = G.name + ":" + kind;
    return room.glitching;
  };
  const endGlitch = () => { const g = glitchState; g.G.objects.forEach((o, i) => { o.position.copy(g.saved[i]); o.visible = true; }); glitchState = null; room.glitching = null; };
  animations.push((dt) => {
    if (!glitchState) { nextGlitch -= dt; if (nextGlitch <= 0) { room.glitch(); nextGlitch = 15 + 35 * Math.random(); } return; }
    const g = glitchState; g.left -= dt; g.tick++;
    if (g.kind === "jitter") g.G.objects.forEach((o, i) => { o.position.copy(g.saved[i]); if (g.tick % 2) { o.position.x += (Math.random() - 0.5) * 0.06; o.position.y += (Math.random() - 0.5) * 0.06; o.position.z += (Math.random() - 0.5) * 0.02; } });
    else if (g.kind === "flicker") g.G.objects.forEach((o) => { o.visible = Math.random() < 0.55; });
    else if (g.kind === "stutter") { if (g.tick % 2) room.rain.stutter(); }
    if (g.left <= 0) endGlitch();
  });

  // the fruits' places on the table, 25 cm apart
  room.placeFruit("banana", [tx - 0.12, ty + 0.03]);
  room.placeFruit("bread", [tx + 0.13, ty - 0.02]);

  // the height of the surface under a point: the table top, and the fruits' tops over their footprints
  room.surfaceZ = (x, y) => {
    if (x < tb.x0 || x > tb.x1 || y < tb.y0 || y > tb.y1) return 0;
    for (const [sx, sy, r] of bananaSamples) { const dx = x - sx, dy = y - sy; if (dx * dx + dy * dy < (r + 0.003) * (r + 0.003)) return fruits.banana.pos[2]; }
    const bx = x - breadGroup.position.x, by = y - breadGroup.position.y;
    const c = Math.cos(loaf.rotation.z), s = Math.sin(loaf.rotation.z), lx = c * bx + s * by, ly = -s * bx + c * by;
    if ((lx / LOAF[0]) ** 2 + (ly / LOAF[1]) ** 2 < 1) return fruits.bread.pos[2];
    if (bx * bx + by * by < 0.075 * 0.075) return tb.z + 0.012;
    const ax = x - bananaGroup.position.x, ay = y - bananaGroup.position.y;
    if (ax * ax + ay * ay < 0.11 * 0.11) return tb.z + 0.012;
    return tb.z;
  };
  return room;
}

// -- the hand: gone from the Matrix; an empty group keeps old imports working ---------------------

export function createHand() {
  const g = new THREE.Group();
  g.visible = false;
  return g;
}

// -- renderer and cameras ---------------------------------------------------------------------

export function createRenderer(canvas) {
  const renderer = new THREE.WebGLRenderer({ canvas, antialias: true, powerPreference: "high-performance" });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 1.5));
  renderer.shadowMap.enabled = false;
  renderer.toneMapping = THREE.ACESFilmicToneMapping; renderer.toneMappingExposure = 1.0;
  renderer.setClearColor(MATRIX.background, 1);
  return renderer;
}

// Bloom tuned for lines: the dim grid stays below the threshold, the bright edges glow.
export function createComposer(renderer, scene, camera) {
  const composer = new EffectComposer(renderer);
  const render = new RenderPass(scene, camera);
  const bloom = new UnrealBloomPass(new THREE.Vector2(1, 1), 0.5, 0.4, 0.46);
  composer.addPass(render); composer.addPass(bloom); composer.addPass(new OutputPass());
  return { composer, render, bloom };
}

// Draws the scene with `camera` into the pixel rectangle (x, y, w, h), measured from the
// bottom-left in CSS pixels as WebGL counts. Made to run after a composer has drawn the main
// view: the rectangle is cleared to the scene's background colour and depth, drawn straight to
// the screen (so no bloom), and the renderer's viewport, scissor and clear state are put back.
const _vp = new THREE.Vector4(), _sc = new THREE.Vector4(), _clear = new THREE.Color();
export function renderInset(renderer, scene, camera, x, y, w, h) {
  const aspect = w / h;
  if (Math.abs(camera.aspect - aspect) > 1e-6) { camera.aspect = aspect; camera.updateProjectionMatrix(); }
  const autoClear = renderer.autoClear, scissorTest = renderer.getScissorTest(), clearAlpha = renderer.getClearAlpha();
  renderer.getViewport(_vp); renderer.getScissor(_sc); renderer.getClearColor(_clear);
  renderer.setRenderTarget(null);
  renderer.setScissorTest(true); renderer.setScissor(x, y, w, h); renderer.setViewport(x, y, w, h);
  renderer.autoClear = false;
  renderer.setClearColor(scene.background && scene.background.isColor ? scene.background : MATRIX.background, 1);
  renderer.clear(true, true, true);
  renderer.render(scene, camera);
  renderer.setClearColor(_clear, clearAlpha);
  renderer.autoClear = autoClear;
  renderer.setScissorTest(scissorTest); renderer.setScissor(_sc); renderer.setViewport(_vp);
}

// Three ways of looking: a chase camera 12 cm behind and above the fly, the room from a corner, and
// the fly's own eyes. Beside them a permanent close-up (`rig.closeup`) that frames the fly from a
// three-quarter front view at about 8 mm, follows it with a critically damped offset, and orbits
// slowly when the fly sits.
export class CameraRig {
  constructor(aspect) {
    this.camera = new THREE.PerspectiveCamera(46, aspect, 0.0006, 80);
    this.mode = "follow";
    this.zoom = 0.12;                   // m, chase distance: the fly at its true size against the room
    this.pos = new THREE.Vector3(); this.look = new THREE.Vector3();
    this.fwd = new THREE.Vector3(1, 0, 0);
    this.first = true;
    this.closeup = new THREE.PerspectiveCamera(35, 1.6, 0.0005, 80);
    this.cu = { d: 0.008, off: new THREE.Vector3(), vel: new THREE.Vector3(), want: new THREE.Vector3(), phi: 0, sit: 0, still: 0, first: true };
  }

  setMode(mode) { this.mode = mode; this.first = true; }

  wheel(dy) { this.zoom = Math.min(0.5, Math.max(0.012, this.zoom * Math.exp(dy * 0.0015))); }

  closeupZoom(dy) { this.cu.d = Math.min(0.05, Math.max(0.005, this.cu.d * Math.exp(dy * 0.0015))); }

  // `sitting` says whether the fly stands on a surface; when it is not given, a fly that has
  // been still for 0.4 s counts as sitting.
  update(flight, dt, sitting = null) {
    const cam = this.camera, [x, y, z] = flight.p, R = flight.rotation();
    if (this.mode === "room") {
      cam.fov = 60; cam.near = 0.01;
      cam.position.copy(toThree(0.32, 0.3, 2.35)); cam.up.set(0, 1, 0); cam.lookAt(toThree(2.4, 1.8, 0.85));
    } else if (this.mode === "eye") {
      cam.fov = 110; cam.near = 0.0004;
      const head = toThree(x + R[0] * 1.2e-3 + R[2] * 2e-4, y + R[3] * 1.2e-3 + R[5] * 2e-4, z + R[6] * 1.2e-3 + R[8] * 2e-4);
      cam.position.copy(head); cam.up.copy(toThree(R[2], R[5], R[8])); cam.lookAt(head.clone().add(toThree(R[0], R[3], R[6])));
    } else {
      cam.fov = 46; cam.near = 0.0006;
      const hx = R[0], hy = R[3], hn = Math.hypot(hx, hy) || 1;   // the heading, smoothed
      const k = this.first ? 1 : 1 - Math.exp(-dt / 0.25);
      this.fwd.lerp(new THREE.Vector3(hx / hn, hy / hn, 0), k).normalize();
      const d = this.zoom;
      const want = new THREE.Vector3(x - this.fwd.x * d, y - this.fwd.y * d, z + 0.42 * d);
      const kp = this.first ? 1 : 1 - Math.exp(-dt / 0.12), kl = this.first ? 1 : 1 - Math.exp(-dt / 0.05);
      this.pos.lerp(want, kp); this.look.lerp(new THREE.Vector3(x, y, z), kl);
      cam.position.copy(toThree(this.pos.x, this.pos.y, this.pos.z)); cam.up.set(0, 1, 0);
      cam.lookAt(toThree(this.look.x, this.look.y, this.look.z));
    }
    cam.updateProjectionMatrix();
    this.first = false;
    this.updateCloseup(flight, dt, sitting, R);
  }

  updateCloseup(flight, dt, sitting, R) {
    const cu = this.cu, cam = this.closeup, [x, y, z] = flight.p;
    if (sitting === null || sitting === undefined) { cu.still = flight.speed() < 0.01 ? cu.still + dt : 0; sitting = cu.still > 0.4; }
    cu.sit += ((sitting ? 1 : 0) - cu.sit) * (cu.first ? 1 : 1 - Math.exp(-dt / 0.6));
    if (sitting) cu.phi += 0.1 * dt;                          // the slow orbit while the fly sits
    const heading = Math.atan2(R[3], R[0]);
    const az = heading + 0.8 + cu.phi, el = 0.16 + 0.26 * cu.sit;
    cu.want.set(Math.cos(el) * Math.cos(az), Math.cos(el) * Math.sin(az), Math.sin(el)).multiplyScalar(cu.d);
    if (cu.first) { cu.off.copy(cu.want); cu.vel.set(0, 0, 0); cu.first = false; }
    else {                                                     // critically damped spring on the offset
      const w0 = 7.0, ax = w0 * w0 * (cu.want.x - cu.off.x) - 2 * w0 * cu.vel.x, ay = w0 * w0 * (cu.want.y - cu.off.y) - 2 * w0 * cu.vel.y, az_ = w0 * w0 * (cu.want.z - cu.off.z) - 2 * w0 * cu.vel.z;
      cu.vel.x += ax * dt; cu.vel.y += ay * dt; cu.vel.z += az_ * dt;
      cu.off.x += cu.vel.x * dt; cu.off.y += cu.vel.y * dt; cu.off.z += cu.vel.z * dt;
    }
    const lift = cu.d * (0.105 * (1 - cu.sit) + 0.01);       // the fly in the lower third when flying
    cam.near = Math.max(0.0003, cu.d * 0.04);
    cam.position.copy(toThree(x + cu.off.x, y + cu.off.y, z + cu.off.z)); cam.up.set(0, 1, 0);
    cam.lookAt(toThree(x, y, z + lift));
    cam.updateProjectionMatrix();
  }
}
