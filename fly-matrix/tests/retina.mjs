// Pixel transduction and offscreen capture contract. No browser/GPU or large fixture.
// The renderer stub checks actual buffer plumbing/state restoration, not optical fidelity.
// Run: node tests/retina.mjs
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";
import { createRetina, validateRetinalFrame, RETINA_WIDTH, RETINA_HEIGHT,
  RETINA_MAX_WIDTH, RETINA_MAX_HEIGHT, RETINA_LUMINANCE, RETINA_MAPPING } from "../web/retina.js";

const close = (a, b, epsilon = 1e-7) => assert.ok(Math.abs(a - b) <= epsilon, `${a} != ${b}`);
const frame = (width, height, pixel, origin = "top-left") => {
  const rgba = new Uint8Array(width * height * 4);
  for (let y = 0; y < height; y++) for (let x = 0; x < width; x++) rgba.set([...pixel(x, y), 255], 4 * (y * width + x));
  return { width, height, rgba, origin };
};
const encoder = createRetina({ n: 20, photoreceptors: [12, 2, 16, 4, 18, 6, 14, 8] });
assert.equal(encoder.mapping.scheme, RETINA_MAPPING);
assert.deepEqual(encoder.encode(frame(1, 1, () => [0, 0, 0])).indices, Int32Array.from([2, 4, 6, 8, 12, 14, 16, 18]));
assert.equal(encoder.mapping.rows, 2);
for (const [rgb, expected] of [[[0, 0, 0], 0], [[255, 255, 255], 1],
  [[255, 0, 0], RETINA_LUMINANCE[0]], [[0, 255, 0], RETINA_LUMINANCE[1]], [[0, 0, 255], RETINA_LUMINANCE[2]]]) {
  const output = encoder.encode(frame(3, 2, () => rgb));
  output.levels.forEach(level => close(level, expected));
}

// Equal total image brightness is insufficient: spatially moved pixels change
// which fixed photoreceptors receive drive, rather than broadcasting a mean.
const left = frame(8, 4, x => x < 4 ? [255, 255, 255] : [0, 0, 0]);
const right = frame(8, 4, x => x < 4 ? [0, 0, 0] : [255, 255, 255]);
const l = encoder.encode(left).levels, r = encoder.encode(right).levels;
assert.notDeepEqual(l, r);
for (let row = 0; row < 2; row++) for (let col = 0; col < 4; col++) {
  close(l[4 * row + col], r[4 * row + 3 - col]);
  assert.equal(l[4 * row + col] > r[4 * row + col], col < 2);
}
const top = frame(4, 4, (_x, y) => y < 2 ? [255, 255, 255] : [0, 0, 0]);
const flipped = frame(4, 4, (_x, y) => y < 2 ? [0, 0, 0] : [255, 255, 255], "bottom-left");
assert.deepEqual(encoder.encode(top), encoder.encode(flipped), "WebGL bottom-up row order must not invert retinal elevation");
assert.ok(encoder.encode(top).levels.slice(0, 4).every(x => x > 0.5));
assert.ok(encoder.encode(top).levels.slice(4).every(x => x < 0.5));

// The encoder is stateless and image-only; hidden scene metadata has no effect.
const original = encoder.encode(left);
const observed = encoder.encode({ ...left, fruit: [100, -20, 30], heading: -2, target: "bread" });
assert.deepEqual(observed, original);
observed.indices.fill(0); observed.levels.fill(0);
assert.deepEqual(encoder.encode(left), original, "caller mutation must not corrupt mapping or future outputs");
const membership = [9, 1, 5];
const copied = createRetina({ n: 10, photoreceptors: membership });
membership.fill(0);
assert.deepEqual(copied.encode(left).indices, Int32Array.from([1, 5, 9]));
const fullCount = createRetina({ n: 2000, photoreceptors: Array.from({ length: 1831 }, (_, k) => k + 10) });
assert.equal(fullCount.mapping.rows, 30);
assert.equal(fullCount.mapping.minColumns, 61); assert.equal(fullCount.mapping.maxColumns, 62);
assert.equal(fullCount.encode(left).levels.length, 1831);

for (const input of [undefined, {}, { n: 2, photoreceptors: [] }, { n: 3, photoreceptors: [1, 1] },
  { n: 3, photoreceptors: [-1] }, { n: 3, photoreceptors: [3] }, { n: 3, photoreceptors: [1.5] },
  { n: 3, photoreceptors: [NaN] }, { n: 3, photoreceptors: new DataView(new ArrayBuffer(4)) },
  { n: 3.5, photoreceptors: [1] }]) assert.throws(() => createRetina(input));
for (const input of [undefined, {}, { ...left, width: 0 }, { ...left, width: 1.5 },
  { ...left, width: RETINA_MAX_WIDTH + 1 }, { ...left, height: RETINA_MAX_HEIGHT + 1 },
  { ...left, origin: "unknown" }, { ...left, origin: undefined },
  { ...left, indices: [0] }, { ...left, levels: [1] },
  { ...left, rgba: left.rgba.slice(1) }, { ...left, rgba: Array.from(left.rgba) },
  { ...left, rgba: new Float32Array(left.rgba) }]) assert.throws(() => validateRetinalFrame(input));
assert.equal(validateRetinalFrame({ ...left, rgba: new Uint8ClampedArray(left.rgba) }).width, left.width);

// Execute the actual eye.js with small Three/renderer stubs. Capture must render
// and read a dedicated raw target even if render() has never been called.
class Vector {
  constructor(x = 0, y = 0, z = 0, w = 0) { this.set(x, y, z, w); this.isVector4 = true; }
  set(x, y, z = 0, w = 0) { Object.assign(this, { x, y, z, w }); return this; }
  copy(v) { return this.set(v.x, v.y, v.z, v.w); }
}
class Camera {
  constructor(fov, aspect, near, far) { Object.assign(this, { fov, aspect, near, far }); this.position = new Vector(); this.up = new Vector(); }
  lookAt(value) { this.look = new Vector().copy(value); }
  updateProjectionMatrix() {}
}
class Scene { constructor() { this.children = []; } add(object) { this.children.push(object); } }
class Target { constructor(width, height) { this.width = width; this.height = height; this.texture = {}; } }
const THREE = {
  Vector2: Vector, Vector4: Vector, WebGLRenderTarget: Target, PerspectiveCamera: Camera,
  OrthographicCamera: Camera, Scene, ShaderMaterial: class { constructor(options) { Object.assign(this, options); } },
  Mesh: class { constructor(geometry, material) { Object.assign(this, { geometry, material }); } },
  PlaneGeometry: class {}, Color: class {}, LinearFilter: 1, RGBAFormat: 2, UnsignedByteType: 3,
};
const eyeSource = readFileSync(new URL("../web/eye.js", import.meta.url), "utf8")
  .replace(/^import .*;\n/gm, "").replace("export function createFlyEye", "function createFlyEye");
const createFlyEye = runInNewContext(eyeSource + "\ncreateFlyEye", {
  THREE, RETINA_WIDTH, RETINA_HEIGHT, Uint8Array,
  toThree: (x, y, z) => new Vector(x, z, -y),
});
function captureFixture() {
  const scene = new Scene(), hidden = [{ visible: true }, { visible: false }];
  const renderer = {
    target: { name: "prior target" }, viewport: new Vector(5, 7, 400, 300),
    scissor: new Vector(8, 9, 200, 150), scissorTest: true, autoClear: false,
    renders: [], reads: [], pixel: 40, fail: null,
    getRenderTarget() { return this.target; },
    setRenderTarget(value) {
      this.target = value;
      // Binding a target installs its viewport/scissor state in Three. Restore
      // the prior explicit viewport only AFTER restoring the previous target.
      this.viewport = new Vector(0, 0, value?.width ?? 640, value?.height ?? 480);
      this.scissor = new Vector().copy(this.viewport); this.scissorTest = false;
    },
    getViewport(out) { return out.copy(this.viewport); }, getScissor(out) { return out.copy(this.scissor); },
    getScissorTest() { return this.scissorTest; }, setScissorTest(value) { this.scissorTest = value; },
    setViewport(x, y, z, w) { this.viewport = typeof x === "object" ? new Vector().copy(x) : new Vector(x, y, z, w); },
    setScissor(x, y, z, w) { this.scissor = typeof x === "object" ? new Vector().copy(x) : new Vector(x, y, z, w); },
    clear() {},
    render(which, camera) {
      if (which === scene) {
        assert.ok(hidden.every(o => !o.visible), "body decorations must not contaminate the raw view");
        assert.equal(this.scissorTest, false);
      }
      this.renders.push({ which, camera, target: this.target, aspect: camera.aspect });
      if (this.fail === "render") throw Error("render failed");
    },
    readRenderTargetPixels(target, x, y, width, height, rgba) {
      this.reads.push({ target, x, y, width, height });
      if (this.fail === "read") throw Error("read failed");
      rgba.fill(this.pixel);
    },
  };
  const flight = { p: [1, 2, 3], rotation: () => [1, 0, 0, 0, 1, 0, 0, 0, 1] };
  return { scene, hidden, renderer, flight, eye: createFlyEye(renderer, scene, { hide: hidden }) };
}
const rendererState = ({ renderer, hidden }) => ({ target: renderer.target, viewport: { ...renderer.viewport },
  scissor: { ...renderer.scissor }, scissorTest: renderer.scissorTest, autoClear: renderer.autoClear,
  visibility: hidden.map(o => o.visible) });
{
  const f = captureFixture(), before = rendererState(f), bodyBefore = f.flight.p.slice();
  const captured = f.eye.capture(f.flight);
  assert.equal(captured.width, RETINA_WIDTH); assert.equal(captured.height, RETINA_HEIGHT);
  assert.equal(captured.origin, "bottom-left"); assert.equal(captured.rgba.length, RETINA_WIDTH * RETINA_HEIGHT * 4);
  assert.equal(f.renderer.renders.length, 1); assert.equal(f.renderer.renders[0].which, f.scene);
  assert.equal(f.renderer.reads.length, 1); assert.notEqual(f.renderer.reads[0].target, f.eye.target);
  assert.equal(f.renderer.reads[0].target.width, RETINA_WIDTH);
  assert.deepEqual(rendererState(f), before); assert.deepEqual(f.flight.p, bodyBefore);
  const initial = encoder.encode(captured).levels;
  f.renderer.pixel = 120;
  const changed = f.eye.capture(f.flight);
  assert.ok(encoder.encode(changed).levels.every((v, k) => v > initial[k]));
  assert.ok(captured.rgba.every(x => x === 40), "later captures must not mutate a prior sensory snapshot");
  f.eye.render(f.flight, 10, 20, 100, 100, 9999);
  assert.equal(f.eye.camera.aspect, 2, "display rectangle cannot alter retinal projection");
  assert.deepEqual(rendererState(f), before);
  assert.deepEqual(f.eye.capture(f.flight), changed, "cosmetic display/time must not affect raw capture");
}
for (const failure of ["render", "read"]) {
  const f = captureFixture(), before = rendererState(f);
  f.renderer.fail = failure;
  assert.throws(() => f.eye.capture(f.flight), new RegExp(failure + " failed"));
  assert.deepEqual(rendererState(f), before, "capture failure must restore target, viewport, scissor and visibility");
}
{
  const f = captureFixture(), before = rendererState(f);
  f.renderer.fail = "render";
  assert.throws(() => f.eye.render(f.flight, 0, 0, 100, 50), /render failed/);
  assert.deepEqual(rendererState(f), before);
  f.renderer.fail = null; f.flight.p[0] = NaN;
  assert.throws(() => f.eye.capture(f.flight), /invalid_retinal_body_pose/);
  assert.deepEqual(rendererState(f), before);
}
console.log("retina: pixel/color/spatial causality, fixed membership, bounded validation, raw capture, stable projection and failure restoration passed");
