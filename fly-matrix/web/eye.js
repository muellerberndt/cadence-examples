// A head-mounted camera supplies actual rendered RGB pixels to the retinal encoder.
// Capture is independent of the visible eye panel and its cosmetic hexagon/scan shader.
// The camera projection and retina.js index-grid are supplied candidate transduction,
// not recovered optical axes, biological ommatidia or demonstrated visual competence.
import * as THREE from "three";
import { toThree } from "./room.js";
import { RETINA_WIDTH, RETINA_HEIGHT } from "./retina.js";

const MM = 1e-3;
const VERT = `varying vec2 vUv; void main(){ vUv = uv; gl_Position = vec4(position.xy, 0.0, 1.0); }`;
const FRAG = `
precision highp float; varying vec2 vUv;
uniform sampler2D tex; uniform vec2 cells; uniform vec3 tint; uniform float time;
vec2 hexCentre(vec2 p) {                       // the centre of the hexagonal cell containing p (cell units)
  const vec2 s = vec2(1.0, 1.7320508);
  vec4 hC = floor(vec4(p, p - vec2(0.5, 1.0)) / s.xyxy) + 0.5;
  vec4 h = vec4(p - hC.xy * s, p - (hC.zw + 0.5) * s);
  return dot(h.xy, h.xy) < dot(h.zw, h.zw) ? hC.xy * s : (hC.zw + 0.5) * s;
}
void main(){
  vec2 p = vUv * cells;
  vec2 c = hexCentre(p);
  vec3 col = texture2D(tex, c / cells).rgb;
  float lum = dot(col, vec3(0.30, 0.59, 0.11));
  float edge = smoothstep(0.40, 0.50, length(p - c));   // the dark rim of each facet
  float vignette = 1.0 - 0.55 * pow(length(vUv - 0.5) * 1.35, 2.0);
  vec3 out3 = tint * (0.06 + 1.6 * lum) * (1.0 - 0.75 * edge) * vignette;
  out3 += tint * 0.03 * sin(time * 3.0 + vUv.y * 40.0);  // a faint scan
  gl_FragColor = vec4(out3, 1.0);
}`;

export function createFlyEye(renderer, scene, options = {}) {
  const facetsAcross = options.facets || 56;
  const target = new THREE.WebGLRenderTarget(192, 96, { minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: true });
  const retinalTarget = new THREE.WebGLRenderTarget(RETINA_WIDTH, RETINA_HEIGHT, {
    minFilter: THREE.LinearFilter, magFilter: THREE.LinearFilter, depthBuffer: true,
    format: THREE.RGBAFormat, type: THREE.UnsignedByteType,
  });
  const camera = new THREE.PerspectiveCamera(options.fov || 150, 2, 0.6 * MM, 30);
  const quadScene = new THREE.Scene();
  const material = new THREE.ShaderMaterial({ uniforms: { tex: { value: target.texture }, cells: { value: new THREE.Vector2(facetsAcross, facetsAcross / 2) }, tint: { value: new THREE.Color(options.tint || 0x39ff6a) }, time: { value: 0 } }, vertexShader: VERT, fragmentShader: FRAG, depthTest: false, depthWrite: false });
  quadScene.add(new THREE.Mesh(new THREE.PlaneGeometry(2, 2), material));
  const quadCamera = new THREE.OrthographicCamera(-1, 1, 1, -1, 0, 1);
  const hidden = options.hide || [];

  function poseCamera(flight) {
    const [px, py, pz] = flight.p, R = flight.rotation();
    if (![px, py, pz, ...R].every(Number.isFinite) || R.length !== 9) throw new RangeError("invalid_retinal_body_pose");
    const fwd = [R[0], R[3], R[6]], up = [R[2], R[5], R[8]];
    const head = [px + 0.9 * MM * fwd[0], py + 0.9 * MM * fwd[1], pz + 0.9 * MM * fwd[2]];
    camera.position.copy(toThree(...head));
    camera.up.copy(toThree(up[0], up[1], up[2]));
    camera.lookAt(toThree(head[0] + fwd[0], head[1] + fwd[1], head[2] + fwd[2]));
    // A resized inset must never change the sensory projection.
    if (camera.aspect !== 2) { camera.aspect = 2; camera.updateProjectionMatrix(); }
  }

  function withRendererState(operation) {
    const saved = { target: renderer.getRenderTarget(), viewport: renderer.getViewport(new THREE.Vector4()),
      scissor: renderer.getScissor(new THREE.Vector4()), scissorTest: renderer.getScissorTest(), autoClear: renderer.autoClear };
    const visible = hidden.map(o => o.visible);
    try { return operation(saved); }
    finally {
      hidden.forEach((o, k) => { o.visible = visible[k]; });
      renderer.autoClear = saved.autoClear;
      renderer.setRenderTarget(saved.target);
      renderer.setViewport(saved.viewport); renderer.setScissor(saved.scissor); renderer.setScissorTest(saved.scissorTest);
    }
  }

  function drawScene(flight, destination) {
    poseCamera(flight);
    const visible = hidden.map(o => o.visible);
    hidden.forEach(o => { o.visible = false; });
    renderer.autoClear = true;
    renderer.setRenderTarget(destination);
    renderer.setScissorTest(false);
    try { renderer.clear(); renderer.render(scene, camera); }
    finally { hidden.forEach((o, k) => { o.visible = visible[k]; }); }
  }

  /** Raw offscreen RGB, bottom row first. Caller chooses when to sample; the
   * visible inset, cosmetic scan time and debug-view flags are never inputs. */
  function capture(flight) {
    return withRendererState(() => {
      drawScene(flight, retinalTarget);
      const rgba = new Uint8Array(RETINA_WIDTH * RETINA_HEIGHT * 4);
      renderer.readRenderTargetPixels(retinalTarget, 0, 0, RETINA_WIDTH, RETINA_HEIGHT, rgba);
      return { width: RETINA_WIDTH, height: RETINA_HEIGHT, rgba, origin: "bottom-left" };
    });
  }

  /** Display the same fixed projection with cosmetic facets. The destination
   * rect uses bottom-left CSS pixels; Three applies the display pixel ratio. */
  function render(flight, x, y, w, h, now = 0) {
    return withRendererState(saved => {
      drawScene(flight, target);
      renderer.setRenderTarget(saved.target);
      material.uniforms.time.value = now / 1000;
      renderer.autoClear = false;
      renderer.setScissorTest(true); renderer.setScissor(x, y, w, h); renderer.setViewport(x, y, w, h);
      renderer.clear(true, true, false);
      renderer.render(quadScene, quadCamera);
    });
  }
  return { render, capture, camera, target };
}
