// Exercise the actual CameraRig without a browser or WebGL. These small Three stubs cover
// camera placement and target coordinates; projection/rasterization belong to browser tests.
// Run: node fly-matrix/tests/camera.mjs from cadence-examples/.
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { runInNewContext } from "node:vm";

class Vector3 {
  constructor(x = 0, y = 0, z = 0) { this.set(x, y, z); }
  set(x, y, z) { this.x = x; this.y = y; this.z = z; return this; }
  copy(v) { return this.set(v.x, v.y, v.z); }
  clone() { return new Vector3(this.x, this.y, this.z); }
  add(v) { return this.set(this.x + v.x, this.y + v.y, this.z + v.z); }
  multiplyScalar(s) { return this.set(this.x * s, this.y * s, this.z * s); }
  lerp(v, k) { return this.set(this.x + (v.x - this.x) * k, this.y + (v.y - this.y) * k, this.z + (v.z - this.z) * k); }
  normalize() { return this.multiplyScalar(1 / (Math.hypot(this.x, this.y, this.z) || 1)); }
}
class PerspectiveCamera {
  constructor(fov, aspect, near, far) {
    Object.assign(this, { fov, aspect, near, far });
    this.position = new Vector3(); this.up = new Vector3(); this.target = new Vector3();
  }
  lookAt(v) { this.target.copy(v); }
  updateProjectionMatrix() {}
}
const source = readFileSync(new URL("../web/room.js", import.meta.url), "utf8");
const begin = source.indexOf("export class CameraRig {");
assert.ok(begin >= 0, "CameraRig export exists");
const CameraRig = runInNewContext(source.slice(begin).replace("export class CameraRig", "class CameraRig") + "\nCameraRig", {
  THREE: { Vector3, PerspectiveCamera }, toThree: (x, y, z) => new Vector3(x, z, -y),
});
const body = () => ({
  p: [1.2, 1, 1.2], q: [1, 0, 0, 0], v: [0, 0, 0], w: [0, 0, 0], heading: 0,
  touching: 0, onFloor: false,
  speed() { return Math.hypot(...this.v); },
  rotation() { const c = Math.cos(this.heading), s = Math.sin(this.heading); return [c, -s, 0, s, c, 0, 0, 0, 1]; },
});
const vector = v => [v.x, v.y, v.z];
const view = rig => [rig.camera, rig.closeup].flatMap(cam => [
  ...vector(cam.position), ...vector(cam.target), ...vector(cam.up), cam.fov, cam.near,
]);
const stable = (rig, fl, sitting = null) => {
  rig.update(fl, 1 / 60, sitting, { orbit: false });
  const first = view(rig), physical = JSON.stringify(fl);
  for (let n = 0; n < 300; n++) rig.update(fl, 1 / 60, sitting, { orbit: false });
  assert.deepEqual(view(rig), first, "five seconds of wall time cannot move a frozen body's camera");
  assert.equal(JSON.stringify(fl), physical, "the camera never changes physical state");
  assert.ok(view(rig).every(Number.isFinite));
};

// Zero speed in mid-air must not be promoted to a surface-contact pose.
{
  const rig = new CameraRig(1.6), fl = body(); stable(rig, fl);
  assert.equal(rig.cu.sit, 0); assert.equal(rig.cu.phi, 0);
}
// A supported body gets the standing view, but no artificial idle orbit.
{
  const rig = new CameraRig(1.6), fl = body(); fl.onFloor = true; fl.touching = 1;
  stable(rig, fl); assert.equal(rig.cu.sit, 1); assert.equal(rig.cu.phi, 0);
}
// Track actual pose changes immediately; no residual spring motion after they stop.
{
  const rig = new CameraRig(1.6), fl = body(); stable(rig, fl, false);
  const before = view(rig); fl.p = [1.3, 1.1, 1.0]; fl.heading = 0.4;
  stable(rig, fl, false); assert.notDeepEqual(view(rig), before);
  assert.deepEqual(vector(rig.look), fl.p);
  assert.deepEqual(vector(rig.cu.vel), [0, 0, 0]);
}
// Changing from the cinematic presentation clears stored camera velocity.
{
  const rig = new CameraRig(1.6), fl = body(); rig.update(fl, 0.02, true);
  fl.heading = 0.5; rig.update(fl, 0.02, true);
  assert.ok(Math.hypot(...vector(rig.cu.vel)) > 0);
  const orbit = rig.cu.phi; stable(rig, fl, true);
  assert.equal(rig.cu.phi, orbit); assert.deepEqual(vector(rig.cu.vel), [0, 0, 0]);
}
// The old three-argument API retains the intended slow orbit.
{
  const rig = new CameraRig(1.6), fl = body(); rig.update(fl, 0.02, true);
  const first = view(rig), phi = rig.cu.phi;
  for (let n = 0; n < 100; n++) rig.update(fl, 0.02, true);
  assert.ok(rig.cu.phi > phi); assert.notDeepEqual(view(rig), first);
}
// In the public position-following view, physical yaw stays visible: the
// viewing offset remains fixed in world space while the body and position turn.
{
  const rig = new CameraRig(1.6), fl = body();
  const options = { orbit: false, trackHeading: false };
  rig.update(fl, .02, false, options);
  const cameraBefore = vector(rig.camera.position), targetBefore = vector(rig.camera.target);
  fl.heading = Math.PI; const physical = JSON.stringify(fl);
  rig.update(fl, .02, false, options);
  assert.deepEqual(vector(rig.camera.position), cameraBefore, "a real half turn must not orbit the viewing camera");
  assert.deepEqual(vector(rig.camera.target), targetBefore);
  assert.equal(JSON.stringify(fl), physical, "the fixed view has no physical authority");
  fl.p = [1.3, 1.2, 1.5]; rig.update(fl, .02, false, options);
  const cameraDelta = vector(rig.camera.position).map((x, i) => x - cameraBefore[i]);
  [.1, .3, -.2].forEach((x, i) => assert.ok(Math.abs(cameraDelta[i] - x) < 1e-12));
}
for (const mode of ["room", "eye"]) {
  const rig = new CameraRig(1.6), fl = body(); rig.setMode(mode); stable(rig, fl, false);
}
console.log("camera: strict frozen-state stability, contact semantics, actual pose tracking, fixed viewing azimuth, transition and legacy orbit passed");
