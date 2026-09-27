// Cadence brain scan: the standard whole-brain view.
// Every neuron is a point, every synapse a line, laid out by the atlas the library exports.
// Option `labelCount` (0: all) labels only the largest regions, for a small card; `dpr` caps the
// device pixel ratio the canvases draw at (2 by default; 1 halves the cost of every pass twice over).
// Options `hot` and `cool` (RGB in 0..1) recolour the glow of neurons that changed and of
// inhibitory messages; a page in another palette (green on black, say) sets them with the
// region colours of its atlas.
// The brain is drawn as a scan: a field of tissue in each region's colour whose brightness
// is the activation, a hot glow where neurons changed in the last steps (the scan colour
// map runs violet, magenta, orange, white), synapses that light up when their presynaptic
// neuron changes, and messages that travel along synapses as they are sent, so a settling
// reads as a wave through the brain. The traces below the map are an EEG-style montage of
// every region's own activity and change over the last steps. WebGL2 draws it; a Canvas2D
// fallback draws the neurons only. No activity is invented here: every value drawn comes
// from the activations the page feeds in. A second style, `style: "brain"`, wraps the same
// net into the volume of a stylised animal brain in three dimensions: glowing somata by
// anatomy, every synapse a curved path lit by the messages that travel on it, a translucent
// shell, bloom, fog and a slow rotation; `setStyle` switches between the two.
// The brain style also draws a real anatomy: an atlas that carries `positions3` (a measured
// soma position per neuron, three floats each, centred and scaled so the longest extent
// spans the unit ball) is drawn at those positions instead of the generic lobes, with no
// shell unless asked for, the region centres and extents for the labels taken from the
// members, the point sizes from `spacing3` (the distance to a neuron's neighbours, from the
// local density in three dimensions). `layoutAtlas({positions3: {'*': array}})` builds such
// an atlas in the browser and the atlas exporter writes one; `decodeAtlas` reads both keys
// when present, so every earlier payload still draws as before.
export const VERSION = "cadence.brain-scan/v4.2";  // v4.1: the glow colours (`hot`, `cool`) are options; v4.2: `labelCount`, `dpr`

const TYPES = { f4: Float32Array, f8: Float64Array, u4: Uint32Array, i4: Int32Array, u2: Uint16Array, i2: Int16Array, u1: Uint8Array, i1: Int8Array };
const FIT = 0.94; // world span shown at zoom 1

export function decodeArray(spec) {
  const T = TYPES[spec.dtype];
  if (!T) throw Error(`Unsupported array dtype ${spec.dtype}`);
  const bytes = Uint8Array.from(atob(spec.b64), (c) => c.charCodeAt(0));
  const copy = new Uint8Array(bytes.length);
  copy.set(bytes);
  return new T(copy.buffer, 0, copy.byteLength / T.BYTES_PER_ELEMENT);
}

export function decodeAtlas(atlas) {
  if (atlas.format !== "cadence.atlas/v1") throw Error("Not a cadence atlas payload");
  return {
    n: atlas.n,
    synapses: atlas.synapses,
    regions: atlas.regions,
    palette: atlas.palette,
    seed: atlas.seed ?? 0,
    region: decodeArray(atlas.region),
    positions: decodeArray(atlas.positions),
    pre: decodeArray(atlas.pre),
    post: decodeArray(atlas.post),
    weight: decodeArray(atlas.weight),
    // a measured anatomy, optional: three floats per neuron in the brain style's frame and the neighbour spacing
    positions3: atlas.positions3 ? decodeArray(atlas.positions3) : null,
    spacing3: atlas.spacing3 ? decodeArray(atlas.spacing3) : null,
    note: atlas.note ?? null,
  };
}

export function decodeFrames(frames) {
  const activation = decodeArray(frames.activation);
  const out = { steps: frames.steps, n: frames.n, activation: [] };
  for (let t = 0; t < frames.steps; t++) {
    const row = new Float32Array(frames.n);
    for (let i = 0; i < frames.n; i++) row[i] = (activation[t * frames.n + i] / 255) * 2 - 1;
    out.activation.push(row);
  }
  return out;
}

// Passes: 0 resting synapses (rasterised once per camera), 5 the field (tissue, activity,
// heat), 4 the composite, 3 synapses lit by change, 2 messages, 1 neurons.
// State texture per neuron: x the level shown, y the heat, z the message (activation sent
// along synapses), w the potential.
const VERTEX = `#version 300 es
precision highp float; precision highp int;
layout(location=0) in vec3 edge;
uniform sampler2D layoutTex; uniform sampler2D colorTex; uniform sampler2D stateTex;
uniform int pass; uniform float clock; uniform float baseAlpha; uniform float dpr; uniform vec3 camera; uniform vec2 aspect; uniform int mode;
uniform float pixelsPerUnit; uniform float maxPoint; uniform float glow; uniform float fit; uniform vec3 frame; uniform float screenPPU; uniform int dust; uniform vec3 hotColor; uniform vec3 coolColor;
out vec4 color; out vec2 uv;
vec4 item(sampler2D t,int id){return texelFetch(t,ivec2(id%512,id/512),0);}
vec2 clip(vec2 p){return ((p-frame.xy)*frame.z*camera.z*fit+camera.xy)*aspect;}
float level(vec4 s){if(mode==2)return clamp(abs(s.w),0.0,1.0);if(mode==3)return clamp(s.y,0.0,1.0);return clamp(abs(s.x),0.0,1.0);}
void main(){
 uv=vec2(0.0);
 if(pass==4){uv=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(uv*2.0-1.0,0.0,1.0);color=vec4(0.0);return;}
 int a=int(edge.x), b=int(edge.y);
 if(pass==1||pass==5){a=gl_VertexID;b=a;}
 vec4 la=item(layoutTex,a), lb=item(layoutTex,b);
 vec4 sa=item(stateTex,a);
 vec3 ca=item(colorTex,a).rgb, cb=item(colorTex,b).rgb;
 float shown=mode==2?sa.w:sa.x; float heat=clamp(sa.y,0.0,1.0); float lev=level(sa); float msg=sa.z;
 vec3 hot=hotColor;
 vec2 p=la.xy;
 float along=dust==1?fract(float(gl_InstanceID)*0.6180339887):(gl_VertexID==0?0.0:1.0);
 if(pass==0){
   p=mix(la.xy,lb.xy,along);
   color=vec4(mix(ca,cb,0.15+0.7*along),baseAlpha*(0.5+min(2.5,abs(edge.z)))*(dust==1?4.0:1.0));
   gl_PointSize=dpr;
 }else if(pass==3){
   p=mix(la.xy,lb.xy,along);
   float w=clamp(heat*min(1.0,abs(edge.z)),0.0,1.0);
   vec3 tint=edge.z<0.0?vec3(0.45,0.7,1.0):mix(ca,hot,0.6);
   color=vec4(tint,w*0.6*(1.0-0.7*along)*(dust==1?3.0:1.0));
   gl_PointSize=dpr*1.5;
 }else if(pass==2){
   float travel=fract(clock*0.7+float(gl_InstanceID%37)/37.0);
   p=mix(la.xy,lb.xy,travel);
   float strength=clamp(abs(msg*edge.z),0.0,1.0);
   color=vec4(msg*edge.z<0.0?vec3(0.45,0.7,1.0):mix(ca,hot,0.5),min(0.9,strength*2.0));
   if(strength<0.02)color.a=0.0;
   gl_PointSize=dpr*(1.4+2.6*strength);
 }else if(pass==5){
   float spacing=max(1e-4,la.z*pixelsPerUnit);
   float radius=clamp(glow*spacing,2.5,maxPoint*0.5);
   float ratio=radius/spacing;
   float weight=1.0/max(1.0,3.1416*ratio*ratio*0.35);
   vec3 tissue=ca*(0.08+1.1*lev);
   if(shown<0.0&&mode!=3)tissue=mix(tissue,vec3(0.35,0.5,1.0)*lev,0.5);
   color=vec4(tissue*weight,heat*weight);
   gl_PointSize=2.0*radius;
 }else{
   vec3 tint=mix(ca*(0.35+0.65*lev),hot,heat*0.9);
   if(shown<0.0)tint=mix(tint,vec3(0.4,0.6,1.0),0.5*lev);
   float size=dpr*(2.4+3.0*lev+3.0*heat)*clamp(camera.z,0.8,2.5);
   float spacingPx=la.z*screenPPU;
   float density=clamp(spacingPx*spacingPx/(size*size),0.06,1.0);
   color=vec4(tint,(0.45+0.55*lev+0.5*heat)*density);
   gl_PointSize=size;
 }
 color.a*=la.w*lb.w;
 gl_Position=vec4(clip(p),0.0,1.0);
}`;
const FRAGMENT = `#version 300 es
precision highp float; precision highp int;
in vec4 color; in vec2 uv;
uniform sampler2D cachedTex; uniform sampler2D fieldTex; uniform int pass; uniform vec3 background;
uniform float edgeGain; uniform float fieldGain; uniform float heatGain;
out vec4 result;
vec3 scan(float t){
 vec3 c1=vec3(0.16,0.06,0.5),c2=vec3(0.8,0.12,0.42),c3=vec3(1.0,0.5,0.12),c4=vec3(1.0,0.96,0.8);
 t=clamp(t,0.0,1.0);
 if(t<0.25)return mix(vec3(0.0),c1,t/0.25);
 if(t<0.5)return mix(c1,c2,(t-0.25)/0.25);
 if(t<0.75)return mix(c2,c3,(t-0.5)/0.25);
 return mix(c3,c4,(t-0.75)/0.25);
}
void main(){
 if(pass==4){
   vec3 e=texture(cachedTex,uv).rgb; vec4 f=texture(fieldTex,uv);
   vec3 rgb=background+(1.0-exp(-e*edgeGain))+(1.0-exp(-f.rgb*fieldGain))+scan(1.0-exp(-f.a*heatGain));
   result=vec4(rgb,1.0);return;
 }
 if(pass==5){float r=length(gl_PointCoord-0.5)*2.0;if(r>1.0)discard;float g=(exp(-r*r*3.0)-0.0498)/0.9502;result=color*g;return;}
 float alpha=color.a;
 if(pass==1||pass==2){float r=length(gl_PointCoord-0.5)*2.0;if(r>1.0)discard;alpha*=pow(1.0-r,0.55);}
 if(pass==3&&gl_PointCoord!=vec2(0.0)){float r=length(gl_PointCoord-0.5)*2.0;if(r>1.0)discard;}
 result=vec4(color.rgb,alpha);
}`;

function compile(gl, kind, source) {
  const s = gl.createShader(kind);
  gl.shaderSource(s, source);
  gl.compileShader(s);
  if (!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw Error(gl.getShaderInfoLog(s));
  return s;
}

const rgba = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

// ---------------------------------------------------------------------------------------
// The brain style: the patch net wrapped into the volume of a stylised animal brain, in
// three dimensions. One generic brain for every net: two smooth lobed hemispheres, a
// cerebellum behind and below, a short stem, drawn as a translucent shell with a soft rim
// light. Every atlas region is assigned a lobe by its role and name (sensory at the back in
// the occipital area, memory deep and low as the hippocampus, association across the
// parietal and frontal cortex, motor in the frontal strip, a governor or steering patch at
// the front as the prefrontal area); regions of the same lobe are tiled inside it, and each
// region's neurons keep the atlas's own arrangement (connected neurons near each other) on
// the lobe's two long axes with the third axis scattered. Neurons are glowing somata,
// every synapse a cubic Bezier ribbon from the presynaptic soma to the postsynaptic one,
// bowing outward (deterministic per edge from the atlas seed), thin and tapering with a
// bouton at the end, very dark at rest and lit by the messages that travel on it: a glow
// along the path and a bright pulse with a tail, brightness by message times weight,
// strongest synapses first under the line and particle budgets. Additive blending into a
// floating-point scene, a bloom pass, a depth fog so the far side recedes, slow
// self-rotation with drag to rotate and scroll to zoom. The static geometry (the 3D
// positions, the bow of every edge, the shell mesh) is uploaded once per atlas and the
// vertex shader evaluates the curves; nothing is rebuilt per frame.
const DIST = 4.0; // the camera's distance from the brain's centre
const FOCAL = 2.8; // the projection's focal length: the whole shell fits at zoom 1
const LOBES = {
  occipital: { c: [-0.66, 0.12, 0], a: [0.26, 0.32, 0.6] },
  hippocampus: { c: [-0.12, -0.3, 0], a: [0.46, 0.12, 0.34] },
  cortex: { c: [0.02, 0.42, 0], a: [0.44, 0.16, 0.64] },
  motor: { c: [0.5, 0.38, 0], a: [0.13, 0.18, 0.6] },
  prefrontal: { c: [0.82, 0.08, 0], a: [0.18, 0.28, 0.4] },
  temporal: { c: [0.08, -0.08, 0], a: [0.36, 0.14, 0.72] },
};
const LOBE_RULES = [
  ["prefrontal", /govern|steer|monitor|prefrontal|readback|critic|reward|salience/],
  ["occipital", /retina|sense|sensor|evidence|input|eye|ear\b|vision|visual|whisker|touch|odou?r|smell|cue|auditory/],
  ["hippocampus", /record|context|memory|hippocamp|trace|recall|notebook|afterglow|echo/],
  ["motor", /motor|habit|action|actuator|joint|muscle|output|gain|efferen|cord/],
  ["cortex", /belief|expect|predict|imagin|assoc|hidden|cortex/],
];
const LOBE_OF_ROLE = { value: "prefrontal", motor: "motor", memory: "hippocampus", association: "cortex", sensory: "occipital", vision: "occipital" };
export function lobeOf(region) {
  const name = String(region.name ?? "").toLowerCase();
  for (const [lobe, rule] of LOBE_RULES) if (rule.test(name)) return lobe;
  return LOBE_OF_ROLE[region.role] || "temporal";
}
function hash01(x) {
  x = Math.imul(x ^ (x >>> 16), 0x45d9f3b); x = Math.imul(x ^ (x >>> 16), 0x45d9f3b); x ^= x >>> 16;
  return (x >>> 0) / 4294967296;
}

/** The 3D layout: every region in a lobe by role and name, regions of one lobe tiled along
 *  its longest axis, every neuron inside its region's sub-ellipsoid with the atlas's own
 *  two-dimensional arrangement on the two long axes. Returns positions (3 per neuron), a
 *  centre and half-extents per region, and the mean spacing between neighbours per neuron. */
export function brainLayout(atlas) {
  const n = atlas.n, regions = atlas.regions, seed = (atlas.seed || 0) >>> 0;
  const positions = new Float32Array(3 * n), spacing = new Float32Array(n);
  const centers = regions.map(() => [0, 0, 0]), extents = regions.map(() => [0.05, 0.05, 0.05]);
  const members = new Map();
  regions.forEach((region, k) => { const lobe = lobeOf(region); if (!members.has(lobe)) members.set(lobe, []); members.get(lobe).push(k); });
  for (const [lobe, ks] of members) {
    const L = LOBES[lobe], axis = L.a.indexOf(Math.max(...L.a));
    ks.sort((i, j) => regions[j].count - regions[i].count || i - j);
    const share = ks.map((k) => Math.cbrt(Math.max(2, regions[k].count)));
    const total = share.reduce((s, v) => s + v, 0);
    let at = -1;
    ks.forEach((k, idx) => {
      const width = (2 * share[idx]) / total, mid = at + width / 2;
      at += width;
      const c = L.c.slice(), a = L.a.slice();
      if (ks.length > 1) { c[axis] += mid * L.a[axis]; a[axis] = (width / 2) * L.a[axis]; const shrink = Math.sqrt(Math.max(0.15, 1 - mid * mid)); for (let d = 0; d < 3; d++) if (d !== axis) a[d] *= shrink; }
      for (let d = 0; d < 3; d++) a[d] *= 0.92;
      centers[k] = c; extents[k] = a;
    });
  }
  regions.forEach((region, k) => {
    const c = centers[k], a = extents[k];
    const order = [0, 1, 2].sort((i, j) => a[j] - a[i]); // the longest axis first
    const volume = (4 / 3) * Math.PI * a[0] * a[1] * a[2];
    // the spacing on the screen: the ellipsoid's mean cross-section shared by its neurons
    const gap = Math.sqrt((Math.PI * Math.cbrt(volume / ((4 / 3) * Math.PI)) ** 2) / Math.max(1, region.count));
    const rng = mulberry((seed * 7919 + k * 104729 + 3) >>> 0);
    for (let i = 0; i < n; i++) {
      if (atlas.region[i] !== k) continue;
      let u = (atlas.positions[2 * i] - region.center[0]) / Math.max(1e-6, region.extent[0]);
      let v = (atlas.positions[2 * i + 1] - region.center[1]) / Math.max(1e-6, region.extent[1]);
      const rr = Math.hypot(u, v);
      if (rr > 0.96) { u *= 0.96 / rr; v *= 0.96 / rr; }
      const w = (rng() * 2 - 1) * Math.sqrt(Math.max(0, 1 - u * u - v * v)) * 0.9;
      const local = [0, 0, 0];
      local[order[0]] = u * a[order[0]]; local[order[1]] = v * a[order[1]]; local[order[2]] = w * a[order[2]];
      positions[3 * i] = c[0] + local[0]; positions[3 * i + 1] = c[1] + local[1]; positions[3 * i + 2] = c[2] + local[2];
      spacing[i] = gap;
    }
  });
  return { positions, spacing, centers, extents };
}

// The anatomical mode: measured positions instead of the generic lobes.
const ANATOMY_VIEW = { yaw: 0.35, pitch: 0.2 }; // a frontal view, turned a little so the depth reads
const ANATOMY_RADIUS = 1.0; // the longest half-extent of a measured anatomy in the brain's frame

/** Measured 3D positions (an array of [x, y, z] or a flat array of 3n floats) centred and
 *  scaled uniformly so their longest extent spans `ANATOMY_RADIUS` each way: the frame the
 *  brain style's projection assumes. Returns a Float32Array of 3n. */
export function fitPositions3(raw, n) {
  const out = new Float32Array(3 * n), flat = typeof raw[0] === "number";
  if ((flat ? raw.length : raw.length * 3) !== 3 * n) throw Error(`positions3['*'] must hold ${n} points`);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) {
    const v = flat ? raw[3 * i + d] : raw[i][d];
    out[3 * i + d] = v; lo[d] = Math.min(lo[d], v); hi[d] = Math.max(hi[d], v);
  }
  const scale = (2 * ANATOMY_RADIUS) / Math.max(1e-9, hi[0] - lo[0], hi[1] - lo[1], hi[2] - lo[2]);
  for (let i = 0; i < n; i++) for (let d = 0; d < 3; d++) out[3 * i + d] = (out[3 * i + d] - (lo[d] + hi[d]) / 2) * scale;
  return out;
}

/** The distance to a neuron's neighbours as seen from the front, from the local density of
 *  the (x, y) projection on a grid over the anatomy (the counterpart of the scan's
 *  `_spacing`, in the brain's frame). What the additive somata sum to on the screen is the
 *  column density along the view, so the projected spacing is the one the point sizes and
 *  alphas need, and a thin anatomy turned to its side reads a little brighter, as a scan
 *  does edge-on. */
export function spacing3Of(positions, n) {
  const out = new Float32Array(n);
  if (!n) return out;
  let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
  for (let i = 0; i < n; i++) { const x = positions[3 * i], y = positions[3 * i + 1]; x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y); }
  const G = 64, w = Math.max(1e-6, x1 - x0), h = Math.max(1e-6, y1 - y0), cell = Math.max(w, h) / G;
  const cols = Math.max(1, Math.ceil(w / cell)), rows = Math.max(1, Math.ceil(h / cell));
  const counts = new Float32Array(cols * rows), cx = new Int32Array(n), cy = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    cx[i] = Math.min(cols - 1, Math.floor((positions[3 * i] - x0) / cell)); cy[i] = Math.min(rows - 1, Math.floor((positions[3 * i + 1] - y0) / cell));
    counts[cy[i] * cols + cx[i]]++;
  }
  for (let i = 0; i < n; i++) {
    let sum = 0, cells = 0;
    for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
      const X = cx[i] + dx, Y = cy[i] + dy;
      if (X < 0 || Y < 0 || X >= cols || Y >= rows) continue;
      sum += counts[Y * cols + X]; cells++;
    }
    out[i] = Math.sqrt((cells * cell * cell) / Math.max(1, sum));
  }
  return out;
}

/** The brain style's layout of an atlas that carries measured positions: the positions as
 *  given, the spacing from the atlas or from the local density, every region's centre and
 *  half-extents from the bulk of its members (the 5th to 95th percentile on each axis, so a
 *  label hangs over the tissue and a few strays or jittered placements do not carry it
 *  away), and the bounds of the whole anatomy for the frame. */
export function anatomyLayout(atlas) {
  const n = atlas.n, positions = atlas.positions3, regions = atlas.regions;
  const spacing = atlas.spacing3 && atlas.spacing3.length === n ? atlas.spacing3 : spacing3Of(positions, n);
  const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity], members = regions.map(() => []);
  for (let i = 0; i < n; i++) { members[atlas.region[i]].push(i); for (let d = 0; d < 3; d++) { const v = positions[3 * i + d]; if (v < lo[d]) lo[d] = v; if (v > hi[d]) hi[d] = v; } }
  const centers = [], extents = [];
  for (const ids of members) {
    const m = ids.length, c = [0, 0, 0], e = [0.05, 0.05, 0.05];
    for (let d = 0; d < 3 && m; d++) {
      const v = new Float32Array(m);
      for (let j = 0; j < m; j++) v[j] = positions[3 * ids[j] + d];
      v.sort();
      const a = v[Math.floor(0.05 * (m - 1))], b = v[Math.ceil(0.95 * (m - 1))];
      c[d] = (a + b) / 2; e[d] = Math.max(0.02, (b - a) / 2);
    }
    centers.push(c); extents.push(e);
  }
  return { positions, spacing, centers, extents, bounds: n ? [lo, hi] : null };
}

/** The bow of every synapse: a vector perpendicular to its chord, outward from the brain's
 *  centre and turned by a per-edge hash, scaled by the chord's length; the cubic Bezier's
 *  control points are the chord's thirds plus this vector. */
export function edgeBows(atlas, positions) {
  const E = atlas.pre.length, seed = (atlas.seed || 0) >>> 0, bows = new Float32Array(3 * E);
  for (let e = 0; e < E; e++) {
    const a = 3 * atlas.pre[e], b = 3 * atlas.post[e];
    const dx = positions[b] - positions[a], dy = positions[b + 1] - positions[a + 1], dz = positions[b + 2] - positions[a + 2];
    const len = Math.hypot(dx, dy, dz);
    if (len < 1e-6) continue;
    const ux = dx / len, uy = dy / len, uz = dz / len;
    const h1 = hash01((seed * 31 + e * 3 + 1) >>> 0), h2 = hash01((seed * 31 + e * 3 + 2) >>> 0), h3 = hash01((seed * 31 + e * 3 + 3) >>> 0);
    const th = h1 * Math.PI * 2, ph = Math.acos(2 * h2 - 1);
    let rx = Math.sin(ph) * Math.cos(th), ry = Math.sin(ph) * Math.sin(th), rz = Math.cos(ph);
    const mx = (positions[a] + positions[b]) / 2, my = (positions[a + 1] + positions[b + 1]) / 2 - 0.05, mz = (positions[a + 2] + positions[b + 2]) / 2;
    const ml = Math.hypot(mx, my, mz) || 1;
    rx = 0.55 * rx + 0.45 * (mx / ml); ry = 0.55 * ry + 0.45 * (my / ml); rz = 0.55 * rz + 0.45 * (mz / ml);
    const dot = rx * ux + ry * uy + rz * uz;
    rx -= dot * ux; ry -= dot * uy; rz -= dot * uz;
    const rl = Math.hypot(rx, ry, rz) || 1;
    const bend = (0.16 + 0.26 * h3) * len;
    bows[3 * e] = (rx / rl) * bend; bows[3 * e + 1] = (ry / rl) * bend; bows[3 * e + 2] = (rz / rl) * bend;
  }
  return bows;
}

/** The shell: one stylised brain for every net. Two lobed hemispheres, a cerebellum behind
 *  and below, a short stem, as interleaved position and normal triples: triangles for the
 *  translucent surface and line segments for the faint mesh with its hint of sulci. */
export function brainShell() {
  const tris = [], lines = [];
  const push = (list, p, nrm) => list.push(p[0], p[1], p[2], nrm[0], nrm[1], nrm[2]);
  const ellipsoid = (c, a, rings, segs, bump, medial) => {
    const grid = [];
    for (let i = 0; i <= rings; i++) {
      const th = (i / rings) * Math.PI, row = [];
      for (let j = 0; j <= segs; j++) {
        const ph = (j / segs) * Math.PI * 2;
        const r = 1 + bump(th, ph);
        let x = c[0] + a[0] * r * Math.sin(th) * Math.cos(ph), y = c[1] + a[1] * r * Math.cos(th), z = c[2] + a[2] * r * Math.sin(th) * Math.sin(ph);
        if (medial > 0 && z < 0.05) z = 0.05; else if (medial < 0 && z > -0.05) z = -0.05;
        const nrm = [(x - c[0]) / (a[0] * a[0]), (y - c[1]) / (a[1] * a[1]), (z - c[2]) / (a[2] * a[2])];
        const nl = Math.hypot(...nrm) || 1;
        row.push([[x, y, z], nrm.map((v) => v / nl)]);
      }
      grid.push(row);
    }
    for (let i = 0; i < rings; i++) for (let j = 0; j < segs; j++) {
      const p00 = grid[i][j], p01 = grid[i][j + 1], p10 = grid[i + 1][j], p11 = grid[i + 1][j + 1];
      push(tris, p00[0], p00[1]); push(tris, p10[0], p10[1]); push(tris, p11[0], p11[1]);
      push(tris, p00[0], p00[1]); push(tris, p11[0], p11[1]); push(tris, p01[0], p01[1]);
      if (i > 0) { push(lines, p00[0], p00[1]); push(lines, p01[0], p01[1]); }
      if (j % 2 === 0) { push(lines, p00[0], p00[1]); push(lines, p10[0], p10[1]); }
    }
  };
  const gyri = (th, ph) => 0.035 * Math.sin(7 * ph + 0.4) * Math.sin(2.6 * th) + 0.025 * Math.sin(5 * th + 1.7) * Math.cos(3 * ph) + 0.02 * Math.sin(11 * ph + 2.0 * th);
  const folia = (th, ph) => 0.03 * Math.sin(9 * th) + 0.015 * Math.sin(13 * ph + th);
  ellipsoid([0.02, 0.1, 0.37], [1.02, 0.6, 0.44], 18, 44, gyri, 1);
  ellipsoid([0.02, 0.1, -0.37], [1.02, 0.6, 0.44], 18, 44, gyri, -1);
  ellipsoid([-0.82, -0.36, 0], [0.4, 0.3, 0.52], 14, 32, folia, 0);
  // the stem: a tapered tube from under the cerebellum down and back
  const stem = { from: [-0.6, -0.5, 0], to: [-0.86, -1.0, 0], r0: 0.16, r1: 0.11 }, rings = 6, segs = 18;
  const grid = [];
  for (let i = 0; i <= rings; i++) {
    const t = i / rings, row = [], rad = stem.r0 + (stem.r1 - stem.r0) * t;
    const cx = stem.from[0] + (stem.to[0] - stem.from[0]) * t, cy = stem.from[1] + (stem.to[1] - stem.from[1]) * t;
    for (let j = 0; j <= segs; j++) { const ph = (j / segs) * Math.PI * 2, nx = Math.cos(ph) * 0.9, nz = Math.sin(ph); row.push([[cx + rad * nx, cy, rad * nz], [nx, 0.3, nz]]); }
    grid.push(row);
  }
  for (let i = 0; i < rings; i++) for (let j = 0; j < segs; j++) {
    const p00 = grid[i][j], p01 = grid[i][j + 1], p10 = grid[i + 1][j], p11 = grid[i + 1][j + 1];
    push(tris, p00[0], p00[1]); push(tris, p10[0], p10[1]); push(tris, p11[0], p11[1]);
    push(tris, p00[0], p00[1]); push(tris, p11[0], p11[1]); push(tris, p01[0], p01[1]);
    push(lines, p00[0], p00[1]); push(lines, p01[0], p01[1]);
    if (j % 3 === 0) { push(lines, p00[0], p00[1]); push(lines, p10[0], p10[1]); }
  }
  return { tris: Float32Array.from(tris), lines: Float32Array.from(lines) };
}

// The brain program. Passes: 10 shell surface, 11 shell mesh, 12 the resting web (ribbons),
// 13 the lit paths (glow ribbons), 14 the pulses (a short ribbon with a tail), 15 the pulse
// heads, 16 the boutons, 17 the somata; 18 and 19 the bloom blur, 20 the composite. Per
// instance: `edge` (pre, post, weight) and `bow`; for the shell passes the same slots carry
// the position and the normal.
const BRAIN_VERTEX = `#version 300 es
precision highp float; precision highp int;
layout(location=0) in vec3 edge; layout(location=1) in vec3 bow;
uniform sampler2D posTex; uniform sampler2D colorTex; uniform sampler2D stateTex;
uniform int pass; uniform int mode; uniform mat3 rot; uniform vec3 camera; uniform vec2 aspect; uniform vec2 viewport; uniform vec2 fogRange; uniform vec3 hotColor; uniform vec3 coolColor;
uniform float dist; uniform float focal; uniform float clock; uniform float dpr; uniform float restAlpha; uniform float edgeGain; uniform float scale; uniform float fade; uniform float segments; uniform float pulseSegments; uniform float sizeScale; uniform float densityLaw;
out vec4 color; out vec2 uv; out float across;
vec4 item(sampler2D t,int id){return texelFetch(t,ivec2(id%512,id/512),0);}
vec3 project(vec3 v){float d=max(0.2,dist-v.z);return vec3((focal*v.xy/d*camera.z+camera.xy)*aspect,d);}
float fog(float d){return mix(1.0,0.3,smoothstep(fogRange.x,fogRange.y,d));}
vec3 bez(vec3 p0,vec3 p1,vec3 p2,vec3 p3,float t){float u=1.0-t;return u*u*u*p0+3.0*u*u*t*p1+3.0*u*t*t*p2+t*t*t*p3;}
float level(vec4 s){if(mode==2)return clamp(abs(s.w),0.0,1.0);if(mode==3)return clamp(s.y,0.0,1.0);return clamp(abs(s.x),0.0,1.0);}
void main(){
 uv=vec2(0.0);across=0.0;color=vec4(0.0);gl_PointSize=1.0;
 if(pass>=18){uv=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(uv*2.0-1.0,0.0,1.0);return;}
 vec3 hot=hotColor,cool=coolColor;
 float zoomSize=pow(camera.z,0.8)*sizeScale;
 if(pass==10||pass==11){
   vec3 v=rot*edge; vec3 nrm=normalize(rot*bow); vec3 pr=project(v);
   float rim=pow(1.0-abs(nrm.z),2.5), f=fog(pr.z);
   vec3 tint=vec3(0.6,0.72,1.0);
   color=pass==10?vec4(tint,(0.01+0.11*rim)*f):vec4(tint,(0.03+0.16*rim)*f);
   gl_Position=vec4(pr.xy,0.0,1.0);return;
 }
 int a=int(edge.x), b=int(edge.y);
 if(pass==17){a=gl_VertexID;b=a;}
 vec4 pa=item(posTex,a), pb=item(posTex,b), sa=item(stateTex,a), cA=item(colorTex,a), cB=item(colorTex,b);
 vec3 ca=cA.rgb, cb=cB.rgb;
 float visible=pa.w*pb.w;
 float heat=clamp(sa.y,0.0,1.0), lev=level(sa), msg=sa.z, shown=mode==2?sa.w:sa.x;
 if(pass==17){
   vec3 pr=project(rot*pa.xyz); float f=fog(pr.z), persp=dist/pr.z;
   float size=dpr*(2.8+3.2*lev+2.4*heat)*persp*zoomSize;
   vec3 tint=mix(ca*(0.35+0.65*lev),hot,heat*0.85);
   if(shown<0.0&&mode!=3)tint=mix(tint,cool,0.5*lev);
   float spacingPx=cA.a*focal/pr.z*camera.z*min(viewport.x,viewport.y)*0.5;
   float density=clamp(pow(spacingPx*spacingPx/(size*size),densityLaw),0.02,1.0); // somata packed tighter than their size share their light; a measured anatomy (law 0.5) keeps its dense tissue brighter than its sparse tissue without saturating
   color=vec4(tint*f*1.4,(0.3+0.5*lev+0.5*heat)*density*visible);
   gl_PointSize=size*(1.0+1.6*heat);
   gl_Position=vec4(pr.xy,0.0,1.0);return;
 }
 vec3 p0=pa.xyz, p3=pb.xyz, p1=mix(p0,p3,0.33)+bow, p2=mix(p0,p3,0.67)+bow*0.8;
 float strength=clamp(abs(msg*edge.z)/scale,0.0,1.0), wn=clamp(abs(edge.z)/scale,0.0,1.0), lit=strength*fade;
 bool negative=msg*edge.z<0.0;
 float phase=fract(float(gl_InstanceID)*0.6180339887);
 if(pass==16){
   vec3 pr=project(rot*bez(p0,p1,p2,p3,0.96)); float f=fog(pr.z);
   vec3 tint=mix(mix(ca,cb,0.5),negative?cool:hot,0.5*lit);
   color=vec4(tint*f,(restAlpha*2.0*(0.3+0.7*wn)+0.3*heat*wn*fade)*edgeGain*visible);
   gl_PointSize=dpr*(1.0+1.2*wn+1.6*heat*wn)*(dist/pr.z)*zoomSize;
   gl_Position=vec4(pr.xy,0.0,1.0);return;
 }
 if(pass==15){
   float t=fract(clock*(0.35+0.65*strength)*0.6+phase);
   vec3 pr=project(rot*bez(p0,p1,p2,p3,t)); float f=fog(pr.z);
   color=vec4(mix(ca,negative?cool:hot,0.6)*f*2.2,min(0.9,lit*lit*1.6)*visible);
   if(lit<0.15)color.a=0.0;
   gl_PointSize=dpr*(2.0+4.0*lit)*(dist/pr.z)*zoomSize;
   gl_Position=vec4(pr.xy,0.0,1.0);return;
 }
 int vid=gl_VertexID; float side=float(vid&1)*2.0-1.0, j=float(vid>>1);
 float t, along=0.0;
 if(pass==14){
   float head=fract(clock*(0.35+0.65*strength)*0.6+phase), tail=0.1+0.12*strength;
   along=j/pulseSegments;
   t=clamp(head-tail*(1.0-along),0.0,1.0);
 }else t=j/segments;
 float ta=max(0.0,t-0.01), tb=min(1.0,t+0.01);
 vec3 pr=project(rot*bez(p0,p1,p2,p3,t)), prA=project(rot*bez(p0,p1,p2,p3,ta)), prB=project(rot*bez(p0,p1,p2,p3,tb));
 vec2 tangent=(prB.xy-prA.xy)*viewport; float tl=length(tangent);
 vec2 normal=tl>1e-6?vec2(-tangent.y,tangent.x)/tl:vec2(0.0,1.0);
 float f=fog(pr.z), widthPx, alpha; vec3 tint;
 if(pass==12){widthPx=dpr*(0.6+0.9*wn)*mix(1.0,0.45,t);tint=mix(mix(ca,cb,t),vec3(0.7,0.78,0.95),0.5);alpha=restAlpha*(0.35+0.65*wn)*edgeGain;}
 else if(pass==13){float glow=heat*sqrt(wn)*fade;widthPx=dpr*(1.0+1.6*glow)*mix(1.0,0.6,t);tint=mix(mix(ca,cb,t),negative?cool:hot,0.5*glow);alpha=0.4*glow*(1.0-0.5*t)*edgeGain;if(glow<0.02)alpha=0.0;}
 else{widthPx=dpr*(0.7+1.6*lit)*(0.35+0.65*along);tint=mix(ca,negative?cool:hot,0.65);alpha=0.9*lit*lit*along*along*edgeGain;if(lit<0.15)alpha=0.0;}
 widthPx*=pow(camera.z,0.5)*sizeScale;
 vec2 offset=normal*side*widthPx*2.0/viewport;
 across=side;
 color=vec4(tint*f,alpha*visible);
 gl_Position=vec4(pr.xy+offset,0.0,1.0);
}`;
const BRAIN_FRAGMENT = `#version 300 es
precision highp float; precision highp int;
in vec4 color; in vec2 uv; in float across;
uniform sampler2D sceneTex; uniform sampler2D bloomTex; uniform int pass; uniform vec3 background; uniform vec2 texel; uniform float exposure; uniform float bloomGain;
out vec4 result;
void main(){
 if(pass==18||pass==19){
   vec2 dir=pass==18?vec2(texel.x,0.0):vec2(0.0,texel.y);
   float w[5]=float[5](0.227,0.194,0.121,0.054,0.016);
   vec3 acc=vec3(0.0);
   for(int i=-4;i<=4;i++){vec3 s=texture(sceneTex,uv+dir*float(i)).rgb;if(pass==18)s=max(vec3(0.0),s-0.35);acc+=s*w[i<0?-i:i];}
   result=vec4(acc,1.0);return;
 }
 if(pass==20){
   vec3 s=texture(sceneTex,uv).rgb, b=texture(bloomTex,uv).rgb;
   vec3 rgb=background+(1.0-exp(-(s+b*bloomGain)*exposure));
   result=vec4(rgb,1.0);return;
 }
 float alpha=color.a;
 if(pass==15||pass==16||pass==17){
   float r=length(gl_PointCoord-0.5)*2.0; if(r>1.0)discard;
   alpha*=exp(-r*r*6.0)+0.35*exp(-r*r*1.6)*(1.0-r);
 }else if(pass>=12&&pass<=14){float a2=across*across;alpha*=pass==12?(1.0-a2):exp(-3.0*a2);}
 result=vec4(color.rgb*alpha,alpha);
}`;

export class BrainScan {
  constructor(canvas, atlas, options = {}) {
    this.canvas = canvas;
    this.options = { style: "scan", particles: true, edges: true, field: true, glow: 2.2, heatDecay: 0.86, background: [0.03, 0.055, 0.085], hot: [1.0, 0.93, 0.78], cool: [0.45, 0.7, 1.0], mode: "activity", montageRows: 16, particleBudget: 300000, lineBudget: 400000, labelTop: 9, labelCount: 0, dpr: 2, restAlpha: 0.025, bloom: 0.5, shell: true, spin: true, spinRate: 0.12, exposure: null, ...options };
    this.given = options; // the options the page set itself: a measured anatomy drops the shell unless the page asked for it
    this.brain = this.options.style === "brain";
    this.anatomical = false; // set by setAtlas: the atlas carries measured positions
    this.view = { yaw: 0.5, pitch: 0.25, spin: this.options.spin, rate: this.options.spinRate, lastDrag: -1e9 }; // the brain style's rotation
    this.dt = 0;
    this.stateClock = 0;
    this.frameMs = 0;
    this.camera = { x: 0, y: 0, zoom: 1 };
    this.frame = { x: 0, y: 0, scale: 1 }; // the layout's centre and the scale that fills the canvas at zoom 1
    this.fitted = true;
    this.pointers = new Map();
    this.clock = 0;
    this.lastTime = 0;
    this.pending = true;
    this.labels = options.labels || null;
    this.strip = options.strip || null;
    this.labelElements = [];
    this.rows = 0;
    this.gl = canvas.getContext("webgl2", { alpha: false, antialias: false, premultipliedAlpha: false, powerPreference: "high-performance" });
    this.enabled = !!this.gl;
    if (this.enabled) this._setupGL();
    else this.ctx = canvas.getContext("2d");
    this.setAtlas(atlas);
    this._setupInteraction(options.interaction || canvas);
  }

  /** Load a (new) atlas into this view: the same canvas and context draw the new brain. */
  setAtlas(atlas) {
    this.atlas = atlas.format ? decodeAtlas(atlas) : atlas;
    this.n = this.atlas.n;
    this.edges = this.atlas.pre.length;
    this.anatomical = !!(this.atlas.positions3 && this.atlas.positions3.length === 3 * this.n);
    if (!("shell" in this.given)) this.options.shell = !this.anatomical;
    const rows = Math.max(1, Math.ceil(this.n / 512));
    const resized = rows !== this.rows;
    this.rows = rows;
    this.layout = new Float32Array(512 * rows * 4);
    this.colors = new Float32Array(this.layout.length);
    this.state = new Float32Array(this.layout.length);
    this.activation = new Float32Array(this.n);
    this.previous = null;
    this.heat = new Float32Array(this.n);
    this.change = new Float32Array(this.n);
    this.message = null;
    this.potential = null;
    this.visible = null;
    this.scale = 1e-6;
    this.hot = 0;
    this.stepCount = 0;
    this.boundaries = [];
    this.history = this.atlas.regions.map(() => ({ activity: [], change: [] }));
    this.global = { change: [], mean: [], activity: [] };
    this._montage = null;
    const spacing = this._spacing();
    for (let i = 0; i < this.n; i++) {
      const k = this.atlas.region[i], region = this.atlas.regions[k];
      const c = region.color;
      this.layout.set([this.atlas.positions[2 * i], this.atlas.positions[2 * i + 1], spacing[i], 1], i * 4);
      this.colors.set([c[0] / 255, c[1] / 255, c[2] / 255, 1], i * 4);
    }
    if (this.enabled) { this._uploadAtlas(resized); if (this.brain) this._uploadBrain(); }
    this._setupLabels();
    this.bounds = [Infinity, Infinity, -Infinity, -Infinity];
    for (let i = 0; i < this.n; i++) {
      const x = this.atlas.positions[2 * i], y = this.atlas.positions[2 * i + 1];
      if (x < this.bounds[0]) this.bounds[0] = x; if (y < this.bounds[1]) this.bounds[1] = y;
      if (x > this.bounds[2]) this.bounds[2] = x; if (y > this.bounds[3]) this.bounds[3] = y;
    }
    if (!this.n) this.bounds = [-1, -1, 1, 1];
    this._frame();
    this.dirty = true;
  }

  /** The frame: centre the layout and scale its bounding box to fill the canvas at zoom 1. */
  _frame() {
    const [x0, y0, x1, y1] = this.bounds, a = this.aspect();
    const w = Math.max(1e-6, x1 - x0), h = Math.max(1e-6, y1 - y0);
    // 0.9: room for the region labels above and beside the outermost regions
    this.frame = { x: (x0 + x1) / 2, y: (y0 + y1) / 2, scale: 0.9 * Math.min(2 / (w * FIT * a[0]), 2 / (h * FIT * a[1])) };
  }

  /** The distance to a neuron's neighbours, from the local density on a grid over the layout:
   *  the tissue field's sprite radius follows it, so dense sheets, sparse regions and regions
   *  sharing one anatomical frame all read as tissue of even brightness. */
  _spacing() {
    const n = this.n, pos = this.atlas.positions, out = new Float32Array(n);
    if (!n) return out;
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    for (let i = 0; i < n; i++) { x0 = Math.min(x0, pos[2 * i]); x1 = Math.max(x1, pos[2 * i]); y0 = Math.min(y0, pos[2 * i + 1]); y1 = Math.max(y1, pos[2 * i + 1]); }
    const G = 64, w = Math.max(1e-6, x1 - x0), h = Math.max(1e-6, y1 - y0), cell = Math.max(w, h) / G;
    const cols = Math.max(1, Math.ceil(w / cell)), rows = Math.max(1, Math.ceil(h / cell));
    const counts = new Float32Array(cols * rows), cx = new Int32Array(n), cy = new Int32Array(n);
    for (let i = 0; i < n; i++) {
      cx[i] = Math.min(cols - 1, Math.floor((pos[2 * i] - x0) / cell)); cy[i] = Math.min(rows - 1, Math.floor((pos[2 * i + 1] - y0) / cell));
      counts[cy[i] * cols + cx[i]]++;
    }
    let occupied = 0;
    for (const c of counts) if (c > 0) occupied++;
    this.occupied = Math.max(1e-6, occupied * cell * cell); // the area of the square the tissue covers, for the synapse light budget
    for (let i = 0; i < n; i++) {
      let sum = 0, cells = 0;
      for (let dy = -1; dy <= 1; dy++) for (let dx = -1; dx <= 1; dx++) {
        const X = cx[i] + dx, Y = cy[i] + dy;
        if (X < 0 || Y < 0 || X >= cols || Y >= rows) continue;
        sum += counts[Y * cols + X]; cells++;
      }
      out[i] = Math.sqrt((cells * cell * cell) / Math.max(1, sum));
    }
    return out;
  }

  _setupGL() {
    const gl = this.gl;
    this.program = gl.createProgram();
    gl.attachShader(this.program, compile(gl, gl.VERTEX_SHADER, VERTEX));
    gl.attachShader(this.program, compile(gl, gl.FRAGMENT_SHADER, FRAGMENT));
    gl.linkProgram(this.program);
    if (!gl.getProgramParameter(this.program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(this.program));
    gl.useProgram(this.program);
    const names = ["layoutTex", "colorTex", "stateTex", "cachedTex", "fieldTex", "pass", "clock", "baseAlpha", "dpr", "camera", "aspect", "mode", "background", "pixelsPerUnit", "maxPoint", "glow", "fit", "frame", "screenPPU", "dust", "edgeGain", "fieldGain", "heatGain", "hotColor", "coolColor"];
    this.uniform = Object.fromEntries(names.map((k) => [k, gl.getUniformLocation(this.program, k)]));
    this.textures = [0, 1, 2].map((unit) => {
      const t = gl.createTexture();
      gl.activeTexture(gl.TEXTURE0 + unit);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
      return t;
    });
    gl.uniform1i(this.uniform.layoutTex, 0);
    gl.uniform1i(this.uniform.colorTex, 1);
    gl.uniform1i(this.uniform.stateTex, 2);
    gl.uniform1i(this.uniform.cachedTex, 3);
    gl.uniform1i(this.uniform.fieldTex, 4);
    this.vao = gl.createVertexArray();
    gl.bindVertexArray(this.vao);
    this.buffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.enableVertexAttribArray(0);
    gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
    gl.vertexAttribDivisor(0, 1);
    this.floatCache = !!gl.getExtension("EXT_color_buffer_float");
    const range = gl.getParameter(gl.ALIASED_POINT_SIZE_RANGE);
    this.maxPoint = Math.min(256, range ? range[1] : 64);
    // A software renderer (a CI browser without a GPU) gets the same picture at a lower budget.
    const info = gl.getExtension("WEBGL_debug_renderer_info");
    const vendor = info ? String(gl.getParameter(info.UNMASKED_RENDERER_WEBGL)) : "";
    this.software = /swiftshader|llvmpipe|software|mesa offscreen/i.test(vendor);
    if (this.software) { this.options.particleBudget = Math.min(this.options.particleBudget, 60000); this.options.lineBudget = Math.min(this.options.lineBudget, 120000); }
    gl.enable(gl.BLEND);
    this.cache = gl.createFramebuffer();
    this.cacheTexture = gl.createTexture();
    this.fieldBuffer = gl.createFramebuffer();
    this.fieldTexture = gl.createTexture();
    this.fieldSize = [0, 0];
    this.sized = false;
    this.scanSize = null;
    if (this.brain) this._setupBrainGL();
  }

  _uploadAtlas(resized) {
    const gl = this.gl;
    if (resized) {
      for (let unit = 0; unit < 3; unit++) {
        gl.activeTexture(gl.TEXTURE0 + unit);
        gl.bindTexture(gl.TEXTURE_2D, this.textures[unit]);
        gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 512, this.rows, 0, gl.RGBA, gl.FLOAT, null);
      }
    }
    this._upload(0, this.layout);
    this._upload(1, this.colors);
    this._upload(2, this.state);
    // Edges go to the GPU in a shuffled order, so drawing a prefix draws a uniform sample:
    // the messages of a brain with more synapses than the particle budget stay representative.
    this.order = null;
    if (this.brain) {
      // Strongest first: the line and particle budgets then take the strongest synapses.
      const mag = new Float32Array(this.edges), order = new Uint32Array(this.edges);
      for (let i = 0; i < this.edges; i++) { mag[i] = Math.abs(this.atlas.weight[i]); order[i] = i; }
      order.sort((i, j) => mag[j] - mag[i]);
      this.order = order;
      this.weightScale = this.edges ? Math.max(1e-9, mag[order[Math.floor(this.edges * 0.02)]]) : 1; // the 98th percentile of |w|
    } else if (this.edges > this.options.particleBudget) {
      const order = new Uint32Array(this.edges), rng = mulberry(7);
      for (let i = 0; i < this.edges; i++) order[i] = i;
      for (let i = this.edges - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
      this.order = order;
    }
    this.flat = new Float32Array(Math.max(1, this.edges) * 3);
    let length = 0, magnitude = 0;
    for (let k = 0; k < this.edges; k++) {
      const i = this.order ? this.order[k] : k, a = this.atlas.pre[i], b = this.atlas.post[i];
      this.flat.set([a, b, this.atlas.weight[i]], k * 3);
      length += Math.hypot(this.atlas.positions[2 * a] - this.atlas.positions[2 * b], this.atlas.positions[2 * a + 1] - this.atlas.positions[2 * b + 1]);
      magnitude += Math.abs(this.atlas.weight[i]);
    }
    // the scan's synapse light budget: line length per unit of covered area, and the mean weight factor of a line
    this.lineLoad = this.edges ? length / (this.occupied || 1) : 0;
    this.lineWeight = this.edges ? 0.5 + Math.min(2.5, magnitude / this.edges) : 1;
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(gl.ARRAY_BUFFER, this.flat, gl.DYNAMIC_DRAW);
  }

  _upload(unit, data) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE0 + unit);
    gl.bindTexture(gl.TEXTURE_2D, this.textures[unit]);
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 512, this.rows, gl.RGBA, gl.FLOAT, data);
  }

  /** Replace the synaptic weights (same synapses): the resting raster and the messages follow. */
  setWeights(weights) {
    for (let i = 0; i < this.edges; i++) this.atlas.weight[i] = weights[i];
    if (!this.enabled) return;
    if (this.brain) { this._uploadAtlas(false); this._uploadBows(); this.dirty = true; this.pending = true; return; }
    for (let k = 0; k < this.edges; k++) this.flat[k * 3 + 2] = weights[this.order ? this.order[k] : k];
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferSubData(gl.ARRAY_BUFFER, 0, this.flat);
    this.dirty = true;
  }

  /** Hide neurons (and their synapses): mask[i] === 0 hides neuron i; null shows all. */
  setVisible(mask) {
    let changed = false;
    for (let i = 0; i < this.n; i++) {
      const v = mask && mask[i] === 0 ? 0 : 1;
      if (this.layout[i * 4 + 3] !== v) { this.layout[i * 4 + 3] = v; changed = true; }
    }
    if (!changed) return;
    this.visible = mask ? Array.from(mask) : null;
    if (this.enabled) this._upload(0, this.layout);
    if (this.enabled && this.brain) { for (let i = 0; i < this.n; i++) this.pos[i * 4 + 3] = this.layout[i * 4 + 3]; this._uploadPos(); }
    this.dirty = true; this.pending = true;
  }

  _target(framebuffer, texture, width, height, linear) {
    const gl = this.gl;
    gl.bindTexture(gl.TEXTURE_2D, texture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, linear ? gl.LINEAR : gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    gl.texImage2D(gl.TEXTURE_2D, 0, this.floatCache ? gl.RGBA16F : gl.RGBA8, width, height, 0, gl.RGBA, this.floatCache ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE, null);
    gl.bindFramebuffer(gl.FRAMEBUFFER, framebuffer);
    gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, texture, 0);
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  }

  _setupInteraction(target) {
    target.style.touchAction = "none";
    target.addEventListener("wheel", (e) => {
      e.preventDefault();
      const r = target.getBoundingClientRect();
      this.zoomAt(Math.exp(-e.deltaY * 0.0015), ((e.clientX - r.left) / r.width) * 2 - 1, 1 - ((e.clientY - r.top) / r.height) * 2);
      this.draw();
    }, { passive: false });
    target.addEventListener("pointerdown", (e) => { target.setPointerCapture(e.pointerId); this.pointers.set(e.pointerId, [e.clientX, e.clientY]); });
    target.addEventListener("pointermove", (e) => {
      if (!this.pointers.has(e.pointerId)) { if (this.onhover) this.onhover(this.inspect(e.clientX, e.clientY)); return; }
      const old = this.pointers.get(e.pointerId), r = target.getBoundingClientRect();
      if (this.pointers.size === 1) {
        if (this.brain && !e.shiftKey && e.buttons !== 2) { this.view.yaw += (e.clientX - old[0]) * 0.008; this.view.pitch = Math.max(-1.3, Math.min(1.3, this.view.pitch + (e.clientY - old[1]) * 0.008)); this.view.lastDrag = performance.now(); this.pending = true; }
        else { this.camera.x += ((e.clientX - old[0]) * 2) / r.width / this.aspect()[0]; this.camera.y -= ((e.clientY - old[1]) * 2) / r.height / this.aspect()[1]; this.fitted = false; this.dirty = true; this.pending = true; }
      }
      this.pointers.set(e.pointerId, [e.clientX, e.clientY]);
      this.draw();
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) target.addEventListener(type, (e) => this.pointers.delete(e.pointerId));
    target.addEventListener("contextmenu", (e) => e.preventDefault());
  }

  _setupLabels() {
    if (!this.labels) return;
    this.labels.innerHTML = "";
    this.labelElements = this.atlas.regions.map((region) => {
      const el = document.createElement("span");
      el.className = "brain-scan-label";
      el.textContent = `${region.label ?? region.name} · ${region.count.toLocaleString()}`;
      el.style.position = "absolute";
      el.style.color = `rgb(${region.color.join(",")})`;
      el.style.pointerEvents = "none";
      el.style.font = "11px system-ui, sans-serif";
      el.style.opacity = "0.85";
      el.style.textShadow = "0 0 4px #000";
      el.style.transform = "translate(-50%, -50%)";
      this.labels.append(el);
      return el;
    });
  }

  aspect() {
    const r = this.canvas.getBoundingClientRect();
    if (!r.width || !r.height) return [1, 1];
    return r.width > r.height ? [r.height / r.width, 1] : [1, r.width / r.height];
  }

  /** World coordinates (the atlas square) to CSS pixels inside the canvas. */
  toScreen(px, py) {
    const r = this.canvas.getBoundingClientRect(), a = this.aspect(), f = this.frame, k = f.scale * this.camera.zoom * FIT;
    return [
      ((((px - f.x) * k + this.camera.x) * a[0] + 1) / 2) * r.width,
      ((1 - ((py - f.y) * k + this.camera.y) * a[1]) / 2) * r.height,
    ];
  }

  /** Screen position of neuron i in CSS pixels inside the canvas. */
  screen(i) { return this.brain ? this._project3(this.positions3[3 * i], this.positions3[3 * i + 1], this.positions3[3 * i + 2]).slice(0, 2) : this.toScreen(this.atlas.positions[2 * i], this.atlas.positions[2 * i + 1]); }

  /** Screen positions of every neuron (CSS pixels inside the canvas) as [x0, y0, x1, y1, ...]. */
  screenAll() {
    if (this.brain) {
      const out = new Float32Array(2 * this.n), R = this._rotation(), a = this.aspect(), r = this.canvas.getBoundingClientRect(), p = this.positions3;
      for (let i = 0; i < this.n; i++) {
        const x = p[3 * i], y = p[3 * i + 1], z = p[3 * i + 2];
        const vx = R[0] * x + R[3] * y + R[6] * z, vy = R[1] * x + R[4] * y + R[7] * z, d = Math.max(0.2, DIST - (R[2] * x + R[5] * y + R[8] * z));
        out[2 * i] = (((((FOCAL * vx) / d) * this.camera.zoom + this.camera.x) * a[0] + 1) / 2) * r.width;
        out[2 * i + 1] = ((1 - (((FOCAL * vy) / d) * this.camera.zoom + this.camera.y) * a[1]) / 2) * r.height;
      }
      return out;
    }
    const r = this.canvas.getBoundingClientRect(), a = this.aspect(), out = new Float32Array(2 * this.n), f = this.frame, k = f.scale * this.camera.zoom * FIT;
    const zx = k * a[0], zy = k * a[1], ox = this.camera.x * a[0], oy = this.camera.y * a[1];
    for (let i = 0; i < this.n; i++) {
      out[2 * i] = (((this.atlas.positions[2 * i] - f.x) * zx + ox + 1) / 2) * r.width;
      out[2 * i + 1] = ((1 - ((this.atlas.positions[2 * i + 1] - f.y) * zy + oy)) / 2) * r.height;
    }
    return out;
  }

  zoomAt(factor, x = 0, y = 0) {
    const z = Math.max(0.5, Math.min(60, this.camera.zoom * factor)), ratio = z / this.camera.zoom;
    const a = this.aspect();
    const wx = x / a[0], wy = y / a[1];
    this.camera.x = wx - (wx - this.camera.x) * ratio;
    this.camera.y = wy - (wy - this.camera.y) * ratio;
    this.camera.zoom = z;
    this.fitted = false;
    this.dirty = true;
    this.pending = true;
  }

  fit() { this.camera = { x: 0, y: 0, zoom: 1 }; this.fitted = true; this._frame(); if (this.brain) { Object.assign(this.view, this._defaultView()); this.view.lastDrag = -1e9; this._frameBrain(); } this.dirty = true; this.pending = true; this.draw(); }

  /** The brain style's resting view: the generic brain from the side at a slight angle, a measured anatomy from the front. */
  _defaultView() { return this.anatomical ? { ...ANATOMY_VIEW } : { yaw: 0.5, pitch: 0.25 }; }

  /** Frame a measured anatomy: the zoom and pan that put the projected bounds of the whole
   *  anatomy (or, without them, every region's box) inside the canvas with room for the
   *  labels, at the current rotation. The generic brain fits at zoom 1 by construction. */
  _frameBrain() {
    if (!this.anatomical || !this.centers3) return;
    const R = this._rotation(), a = this.aspect();
    let x0 = Infinity, y0 = Infinity, x1 = -Infinity, y1 = -Infinity;
    const boxes = this.bounds3 ? [[[0, 1, 2].map((d) => (this.bounds3[0][d] + this.bounds3[1][d]) / 2), [0, 1, 2].map((d) => (this.bounds3[1][d] - this.bounds3[0][d]) / 2)]] : this.centers3.map((c, k) => [c, this.extents3[k]]);
    boxes.forEach(([c, e]) => {
      for (let corner = 0; corner < 8; corner++) {
        const x = c[0] + (corner & 1 ? e[0] : -e[0]), y = c[1] + (corner & 2 ? e[1] : -e[1]), z = c[2] + (corner & 4 ? e[2] : -e[2]);
        const d = Math.max(0.2, DIST - (R[2] * x + R[5] * y + R[8] * z));
        const px = (FOCAL * (R[0] * x + R[3] * y + R[6] * z)) / d, py = (FOCAL * (R[1] * x + R[4] * y + R[7] * z)) / d;
        x0 = Math.min(x0, px); x1 = Math.max(x1, px); y0 = Math.min(y0, py); y1 = Math.max(y1, py);
      }
    });
    if (x0 === Infinity) return;
    // 0.9: room for the labels above the topmost regions
    const zoom = Math.max(0.5, Math.min(60, 0.9 * Math.min(2 / Math.max(1e-6, (x1 - x0) * a[0]), 2 / Math.max(1e-6, (y1 - y0) * a[1]))));
    this.camera = { x: (-(x0 + x1) / 2) * zoom, y: (-(y0 + y1) / 2) * zoom, zoom };
    this.dirty = true; this.pending = true;
  }

  /** Start a new settling from a new stimulus: the next step measures change against this state. */
  reset(activation = null) {
    if (activation) this.activation.set(activation);
    this.previous = Float32Array.from(this.activation);
    this.boundaries.push(this.stepCount);
    if (this.boundaries.length > 64) this.boundaries.shift();
  }

  /** One settling step: the new activations of every neuron (and optionally potentials); `{draw: false}` defers the frame. */
  step(activation, extra = {}) {
    const previous = this.previous || Float32Array.from(this.activation);
    let peak = 1e-9, total = 0;
    for (let i = 0; i < this.n; i++) {
      const a = activation[i];
      const d = Math.abs(a - previous[i]);
      this.change[i] = d;
      this.activation[i] = a;
      if (d > peak) peak = d;
      total += d;
    }
    this.scale = Math.max(peak, this.scale * 0.9);
    const decay = this.options.heatDecay;
    for (let i = 0; i < this.n; i++) this.heat[i] = Math.max(this.heat[i] * decay, Math.min(1, this.change[i] / this.scale));
    this.previous = Float32Array.from(this.activation);
    this.message = null;
    if (extra.potential) this.potential = Float32Array.from(extra.potential);
    this.stepCount++;
    this._record(total / this.n, peak);
    this._uploadState();
    this.pending = true;
    if (extra.draw !== false) this.draw(); // a page applying several steps per frame draws once
    return { peak, mean: total / this.n };
  }

  /** Show a state without measuring change (a snapshot). */
  set(activation, extra = {}) {
    this.activation.set(activation);
    this.previous = Float32Array.from(this.activation);
    this.heat.fill(0);
    this.message = null;
    if (extra.potential) this.potential = Float32Array.from(extra.potential);
    this._uploadState();
    this.pending = true;
    if (extra.draw !== false) this.draw();
  }

  /** Show a page's own signals: `activation` drives the messages, `heat` the glow (0..1 per
   *  neuron), and `extra.level` (default the activation) is the brightness shown. */
  show(activation, heat, extra = {}) {
    this.message = Float32Array.from(activation);
    this.activation.set(extra.level ?? activation);
    for (let i = 0; i < this.n; i++) { const h = heat ? heat[i] : 0; this.heat[i] = h > 1 ? 1 : h < 0 ? 0 : h; this.change[i] = this.heat[i]; }
    this.previous = null;
    this.scale = 1;
    if (extra.potential) this.potential = Float32Array.from(extra.potential);
    this._uploadState();
    this.pending = true;
    if (extra.draw !== false) this.draw();
  }

  _record(meanChange, peak) {
    const regions = this.atlas.regions;
    const sum = new Float64Array(regions.length), moved = new Float64Array(regions.length);
    let active = 0;
    for (let i = 0; i < this.n; i++) { const r = this.atlas.region[i], a = Math.abs(this.activation[i]); sum[r] += a; moved[r] += this.change[i]; active += a; }
    regions.forEach((region, k) => {
      const h = this.history[k];
      h.activity.push(sum[k] / Math.max(1, region.count));
      h.change.push(moved[k] / Math.max(1, region.count));
      if (h.activity.length > 320) { h.activity.shift(); h.change.shift(); }
    });
    this.global.change.push(peak);
    this.global.mean.push(meanChange);
    this.global.activity.push(active / Math.max(1, this.n));
    if (this.global.change.length > 320) { this.global.change.shift(); this.global.mean.shift(); this.global.activity.shift(); }
  }

  _uploadState() {
    let span = 1e-9, hot = 0;
    if (this.potential) for (let i = 0; i < this.n; i++) span = Math.max(span, Math.abs(this.potential[i]));
    for (let i = 0; i < this.n; i++) {
      const msg = this.message ? this.message[i] : this.activation[i];
      if (this.heat[i] > hot) hot = this.heat[i];
      this.state.set([this.activation[i], this.heat[i], msg, this.potential ? this.potential[i] / span : 0], i * 4);
    }
    this.hot = hot;
    this.stateClock = this.clock;
    if (this.enabled) this._upload(2, this.state);
  }

  draw(time = performance.now()) {
    // A software renderer draws a new frame at most every 200 ms; a state change always draws.
    if (this.software && !this.pending && time - this.lastTime < 200) return;
    this.pending = false;
    const t0 = performance.now();
    this.dt = Math.min(0.1, (time - this.lastTime) / 1000);
    this.clock += this.dt;
    this.lastTime = time;
    this._drawLabels();
    this._drawStrip();
    if (!this.enabled) return this._drawFallback();
    const gl = this.gl, r = this.canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, this.options.dpr || 2);
    const width = Math.max(1, Math.round(r.width * dpr)), height = Math.max(1, Math.round(r.height * dpr));
    if (!this.sized || this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width; this.canvas.height = height; this.dirty = true; this.sized = true;
      if (this.fitted) { this._frame(); if (this.brain) this._frameBrain(); }
    }
    if (this.brain) { this._drawBrain(width, height, dpr); this._time(t0); return; }
    // A viewer built on a canvas another viewer already sized must still allocate its own targets.
    if (!this.scanSize || this.scanSize[0] !== width || this.scanSize[1] !== height) {
      this.scanSize = [width, height]; this.dirty = true;
      this._target(this.cache, this.cacheTexture, width, height, false);
      this.fieldSize = [Math.max(1, Math.round(width / 2)), Math.max(1, Math.round(height / 2))];
      this._target(this.fieldBuffer, this.fieldTexture, this.fieldSize[0], this.fieldSize[1], true);
    }
    const a = this.aspect(), screenPPU = (this.frame.scale * this.camera.zoom * FIT * Math.min(width, height)) / 2;
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniform3f(this.uniform.camera, this.camera.x, this.camera.y, this.camera.zoom);
    gl.uniform2f(this.uniform.aspect, a[0], a[1]);
    gl.uniform1f(this.uniform.clock, this.clock);
    gl.uniform1f(this.uniform.dpr, dpr);
    gl.uniform1f(this.uniform.fit, FIT);
    gl.uniform3f(this.uniform.frame, this.frame.x, this.frame.y, this.frame.scale);
    gl.uniform1f(this.uniform.screenPPU, screenPPU);
    gl.uniform1i(this.uniform.mode, this.options.mode === "potential" ? 2 : this.options.mode === "change" ? 3 : 0);
    gl.uniform3f(this.uniform.background, ...this.options.background);
    gl.uniform3f(this.uniform.hotColor, ...this.options.hot); gl.uniform3f(this.uniform.coolColor, ...this.options.cool);
    // the resting web: fainter on a brain with more synapses, and capped where the lines crowd into a small tissue area (a measured anatomy)
    const crossings = (this.lineLoad || 0) / Math.max(1e-6, screenPPU) * this.lineWeight; // line pixels per pixel of tissue at this zoom
    gl.uniform1f(this.uniform.baseAlpha, Math.max(this.floatCache ? 0.00005 : 0.008, Math.min(0.09 / Math.pow(Math.max(1, this.edges / 2000), 0.6), crossings > 0 ? 0.35 / crossings : 1)));
    gl.uniform1f(this.uniform.maxPoint, this.maxPoint);
    gl.uniform1f(this.uniform.glow, this.options.glow);
    gl.uniform1f(this.uniform.edgeGain, 1.1);
    gl.uniform1f(this.uniform.fieldGain, this.options.field ? 1.2 : 0);
    gl.uniform1f(this.uniform.heatGain, this.options.field ? 1.6 : 0);
    for (let i = 0; i < 3; i++) { gl.activeTexture(gl.TEXTURE0 + i); gl.bindTexture(gl.TEXTURE_2D, this.textures[i]); }
    gl.enable(gl.BLEND);
    gl.activeTexture(gl.TEXTURE3); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE4); gl.bindTexture(gl.TEXTURE_2D, null);
    // Beyond the line budget every synapse is one point along its line (dust) unless zoomed in.
    const dust = this.edges > this.options.lineBudget && this.camera.zoom < 3;
    gl.uniform1i(this.uniform.dust, dust ? 1 : 0);
    const synapses = (count) => (dust ? gl.drawArraysInstanced(gl.POINTS, 0, 1, count) : gl.drawArraysInstanced(gl.LINES, 0, 2, count));
    if (this.options.edges && this.dirty) {
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.cache);
      gl.viewport(0, 0, width, height);
      gl.clearColor(0, 0, 0, 0);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ONE, gl.ONE);
      gl.uniform1i(this.uniform.pass, 0);
      if (this.edges) synapses(this.edges);
      gl.bindFramebuffer(gl.FRAMEBUFFER, null);
      this.dirty = false;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.fieldBuffer);
    gl.viewport(0, 0, this.fieldSize[0], this.fieldSize[1]);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    if (this.options.field) {
      gl.blendFunc(gl.ONE, gl.ONE);
      gl.uniform1f(this.uniform.pixelsPerUnit, (this.frame.scale * this.camera.zoom * FIT * Math.min(this.fieldSize[0], this.fieldSize[1])) / 2);
      gl.uniform1i(this.uniform.pass, 5);
      gl.drawArrays(gl.POINTS, 0, this.n);
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.cacheTexture);
    gl.activeTexture(gl.TEXTURE4);
    gl.bindTexture(gl.TEXTURE_2D, this.fieldTexture);
    gl.disable(gl.BLEND);
    gl.uniform1i(this.uniform.pass, 4);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.BLEND);
    gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ONE, gl.ONE);
    if (this.edges) {
      if (this.hot > 0.01 && !(this.software && dust)) { gl.uniform1i(this.uniform.pass, 3); synapses(this.edges); }
      if (this.options.particles) { gl.uniform1i(this.uniform.pass, 2); gl.drawArraysInstanced(gl.POINTS, 0, 1, Math.min(this.edges, this.options.particleBudget)); }
    }
    gl.uniform1i(this.uniform.pass, 1);
    gl.drawArrays(gl.POINTS, 0, this.n);
    this._time(t0);
  }

  // ---- The brain style: the program, the static uploads, the view, the frame ----

  _setupBrainGL() {
    if (this.brainProgram) return;
    const gl = this.gl, program = gl.createProgram();
    gl.attachShader(program, compile(gl, gl.VERTEX_SHADER, BRAIN_VERTEX));
    gl.attachShader(program, compile(gl, gl.FRAGMENT_SHADER, BRAIN_FRAGMENT));
    gl.linkProgram(program);
    if (!gl.getProgramParameter(program, gl.LINK_STATUS)) throw Error(gl.getProgramInfoLog(program));
    this.brainProgram = program;
    gl.useProgram(program);
    const names = ["posTex", "colorTex", "stateTex", "sceneTex", "bloomTex", "pass", "mode", "rot", "camera", "aspect", "viewport", "fogRange", "dist", "focal", "clock", "dpr", "restAlpha", "edgeGain", "scale", "fade", "segments", "pulseSegments", "sizeScale", "densityLaw", "background", "texel", "exposure", "bloomGain", "hotColor", "coolColor"];
    this.brainUniform = Object.fromEntries(names.map((k) => [k, gl.getUniformLocation(program, k)]));
    gl.uniform1i(this.brainUniform.posTex, 5);
    gl.uniform1i(this.brainUniform.colorTex, 1);
    gl.uniform1i(this.brainUniform.stateTex, 2);
    gl.uniform1i(this.brainUniform.sceneTex, 6);
    gl.uniform1i(this.brainUniform.bloomTex, 7);
    this.posTexture = gl.createTexture();
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.posTexture);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
    this.posRows = 0;
    // the bow of every synapse: a second instanced attribute beside the edges
    gl.bindVertexArray(this.vao);
    this.bowBuffer = gl.createBuffer();
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bowBuffer);
    gl.enableVertexAttribArray(1);
    gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 12, 0);
    gl.vertexAttribDivisor(1, 1);
    // the shell, one static upload: the surface as triangles, the mesh as lines
    const shell = brainShell();
    const upload = (data) => {
      const vao = gl.createVertexArray(), buffer = gl.createBuffer();
      gl.bindVertexArray(vao);
      gl.bindBuffer(gl.ARRAY_BUFFER, buffer);
      gl.bufferData(gl.ARRAY_BUFFER, data, gl.STATIC_DRAW);
      gl.enableVertexAttribArray(0); gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 24, 0);
      gl.enableVertexAttribArray(1); gl.vertexAttribPointer(1, 3, gl.FLOAT, false, 24, 12);
      return { vao, count: data.length / 6 };
    };
    this.shellSurface = upload(shell.tris);
    this.shellMesh = upload(shell.lines);
    gl.bindVertexArray(this.vao);
    this.sceneBuffer = gl.createFramebuffer(); this.sceneTexture = gl.createTexture();
    this.bloomBuffers = [gl.createFramebuffer(), gl.createFramebuffer()];
    this.bloomTextures = [gl.createTexture(), gl.createTexture()];
    this.bloomSize = [1, 1];
    this.brainSize = null;
  }

  /** The 3D layout of the loaded atlas and the bow of every synapse, uploaded once: the
   *  measured anatomy when the atlas carries one, the generic lobes otherwise. */
  _uploadBrain() {
    const gl = this.gl, layout = this.anatomical ? anatomyLayout(this.atlas) : brainLayout(this.atlas);
    this.positions3 = layout.positions; this.centers3 = layout.centers; this.extents3 = layout.extents; this.bounds3 = layout.bounds || null;
    if (this.view.lastDrag < 0) Object.assign(this.view, this._defaultView()); // a view never turned by hand takes the mode's own
    if (this.fitted) this._frameBrain();
    this.pos = new Float32Array(512 * this.rows * 4);
    for (let i = 0; i < this.n; i++) {
      this.pos.set([layout.positions[3 * i], layout.positions[3 * i + 1], layout.positions[3 * i + 2], this.layout[i * 4 + 3]], i * 4);
      this.colors[i * 4 + 3] = layout.spacing[i];
    }
    this._uploadPos(true);
    this._upload(1, this.colors);
    this.bows = edgeBows(this.atlas, layout.positions);
    this._uploadBows();
  }

  _uploadPos(resize = false) {
    const gl = this.gl;
    gl.activeTexture(gl.TEXTURE5);
    gl.bindTexture(gl.TEXTURE_2D, this.posTexture);
    if (resize || this.posRows !== this.rows) { gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA32F, 512, this.rows, 0, gl.RGBA, gl.FLOAT, null); this.posRows = this.rows; }
    gl.texSubImage2D(gl.TEXTURE_2D, 0, 0, 0, 512, this.rows, gl.RGBA, gl.FLOAT, this.pos);
  }

  /** The bows in the edge buffer's order (strongest synapses first in the brain style). */
  _uploadBows() {
    const gl = this.gl, flat = new Float32Array(Math.max(1, this.edges) * 3);
    for (let k = 0; k < this.edges; k++) { const i = this.order ? this.order[k] : k; flat[k * 3] = this.bows[3 * i]; flat[k * 3 + 1] = this.bows[3 * i + 1]; flat[k * 3 + 2] = this.bows[3 * i + 2]; }
    gl.bindVertexArray(this.vao);
    gl.bindBuffer(gl.ARRAY_BUFFER, this.bowBuffer);
    gl.bufferData(gl.ARRAY_BUFFER, flat, gl.DYNAMIC_DRAW);
  }

  /** Switch the view between `"scan"` (every neuron a point, every synapse a line, the
   *  tissue field) and `"brain"` (the net wrapped into a transparent brain, in 3D). The same
   *  context, state, camera and history carry over. */
  setStyle(style) {
    const brain = style === "brain";
    if (brain === !!this.brain) return;
    this.brain = brain;
    this.options.style = brain ? "brain" : "scan";
    if (this.fitted) this.camera = { x: 0, y: 0, zoom: 1 }; // each style frames itself: the scan by its square, a measured anatomy in _uploadBrain
    if (this.enabled) {
      if (brain) this._setupBrainGL();
      this._uploadAtlas(false);
      if (brain) this._uploadBrain();
    }
    this.dirty = true; this.pending = true;
    this.draw();
  }

  /** The view rotation, column-major: pitch about x after yaw about y. */
  _rotation() {
    const cy = Math.cos(this.view.yaw), sy = Math.sin(this.view.yaw), cp = Math.cos(this.view.pitch), sp = Math.sin(this.view.pitch);
    return [cy, sp * sy, -cp * sy, 0, cp, sp, sy, -sp * cy, cp * cy];
  }

  /** A point of the brain volume to CSS pixels inside the canvas, with its view depth. */
  _project3(x, y, z) {
    const R = this._rotation(), a = this.aspect(), r = this.canvas.getBoundingClientRect();
    const vx = R[0] * x + R[3] * y + R[6] * z, vy = R[1] * x + R[4] * y + R[7] * z, vz = R[2] * x + R[5] * y + R[8] * z;
    const d = Math.max(0.2, DIST - vz);
    const nx = (((FOCAL * vx) / d) * this.camera.zoom + this.camera.x) * a[0], ny = (((FOCAL * vy) / d) * this.camera.zoom + this.camera.y) * a[1];
    return [((nx + 1) / 2) * r.width, ((1 - ny) / 2) * r.height, d];
  }

  _drawBrain(width, height, dpr) {
    const gl = this.gl, u = this.brainUniform, a = this.aspect();
    if (!this.brainSize || this.brainSize[0] !== width || this.brainSize[1] !== height) {
      this._target(this.sceneBuffer, this.sceneTexture, width, height, true);
      this.bloomSize = [Math.max(1, Math.round(width / 4)), Math.max(1, Math.round(height / 4))];
      for (let i = 0; i < 2; i++) this._target(this.bloomBuffers[i], this.bloomTextures[i], this.bloomSize[0], this.bloomSize[1], true);
      this.brainSize = [width, height];
    }
    const view = this.view;
    if (view.spin && this.lastTime - view.lastDrag > 4000) view.yaw += this.dt * view.rate;
    gl.useProgram(this.brainProgram);
    gl.uniformMatrix3fv(u.rot, false, this._rotation());
    gl.uniform3f(u.camera, this.camera.x, this.camera.y, this.camera.zoom);
    gl.uniform2f(u.aspect, a[0], a[1]);
    gl.uniform2f(u.viewport, width, height);
    gl.uniform2f(u.fogRange, DIST - 1.3, DIST + 1.3);
    gl.uniform1f(u.dist, DIST);
    gl.uniform1f(u.focal, FOCAL);
    gl.uniform1f(u.clock, this.clock);
    gl.uniform1f(u.dpr, dpr);
    gl.uniform1f(u.restAlpha, this.options.restAlpha);
    gl.uniform1f(u.scale, this.weightScale || 1);
    gl.uniform1f(u.fade, Math.exp(-Math.max(0, this.clock - this.stateClock) * 1.2));
    // the curve's segments: fewer on a software renderer and on a brain with very many synapses, the curves kept
    const S = this.software ? 4 : this.edges > 60000 ? 6 : 10, PS = this.software ? 4 : 6;
    gl.uniform1f(u.segments, S);
    gl.uniform1f(u.pulseSegments, PS);
    gl.uniform1f(u.sizeScale, Math.max(0.7, Math.min(1.6, Math.min(width, height) / (600 * dpr))));
    gl.uniform1i(u.mode, this.options.mode === "potential" ? 2 : this.options.mode === "change" ? 3 : 0);
    gl.uniform3f(u.background, ...this.options.background);
    gl.uniform3f(u.hotColor, ...this.options.hot); gl.uniform3f(u.coolColor, ...this.options.cool);
    gl.uniform1f(u.densityLaw, this.anatomical ? 0.5 : 1.0);
    gl.uniform1f(u.exposure, this.options.exposure ?? (this.anatomical ? 1.0 : 0.6)); // a measured anatomy has no shell to carry its shape: more light on the tissue
    gl.uniform1f(u.bloomGain, this.options.bloom);
    gl.activeTexture(gl.TEXTURE1); gl.bindTexture(gl.TEXTURE_2D, this.textures[1]);
    gl.activeTexture(gl.TEXTURE2); gl.bindTexture(gl.TEXTURE_2D, this.textures[2]);
    gl.activeTexture(gl.TEXTURE5); gl.bindTexture(gl.TEXTURE_2D, this.posTexture);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, null);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, null);
    // the scene, additive into the floating-point target: shell, web, lit paths, pulses, somata
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.sceneBuffer);
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    gl.clear(gl.COLOR_BUFFER_BIT);
    gl.enable(gl.BLEND);
    gl.blendFunc(gl.ONE, gl.ONE);
    if (this.options.shell) {
      gl.bindVertexArray(this.shellSurface.vao); gl.uniform1i(u.pass, 10); gl.drawArrays(gl.TRIANGLES, 0, this.shellSurface.count);
      gl.bindVertexArray(this.shellMesh.vao); gl.uniform1i(u.pass, 11); gl.drawArrays(gl.LINES, 0, this.shellMesh.count);
    }
    gl.bindVertexArray(this.vao);
    const lines = Math.min(this.edges, this.options.lineBudget), pulses = Math.min(this.edges, this.options.particleBudget);
    gl.uniform1f(u.edgeGain, Math.min(1, Math.sqrt(3000 / Math.max(1, lines)))); // the light budget: a brain with more synapses draws each one fainter
    const strip = (count, segs) => gl.drawArraysInstanced(gl.TRIANGLE_STRIP, 0, 2 * (segs + 1), count);
    if (lines && this.options.edges) {
      gl.uniform1i(u.pass, 12); strip(lines, S);
      if (!this.software) { gl.uniform1i(u.pass, 13); strip(lines, S); } // a software renderer keeps the web and the pulses, not the glow
      gl.uniform1i(u.pass, 16); gl.drawArraysInstanced(gl.POINTS, 0, 1, lines);
    }
    if (pulses && this.options.particles) {
      gl.uniform1i(u.pass, 14); strip(pulses, PS);
      gl.uniform1i(u.pass, 15); gl.drawArraysInstanced(gl.POINTS, 0, 1, pulses);
    }
    gl.uniform1i(u.pass, 17); gl.drawArrays(gl.POINTS, 0, this.n);
    // the bloom: a separable blur of the scene at a quarter of the size
    gl.disable(gl.BLEND);
    gl.viewport(0, 0, this.bloomSize[0], this.bloomSize[1]);
    gl.uniform2f(u.texel, 1 / this.bloomSize[0], 1 / this.bloomSize[1]);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomBuffers[0]);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, this.sceneTexture);
    gl.uniform1i(u.pass, 18); gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.bindFramebuffer(gl.FRAMEBUFFER, this.bloomBuffers[1]);
    gl.bindTexture(gl.TEXTURE_2D, this.bloomTextures[0]);
    gl.uniform1i(u.pass, 19); gl.drawArrays(gl.TRIANGLES, 0, 3);
    // the composite: scene plus bloom, tone-mapped over the background
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.viewport(0, 0, width, height);
    gl.activeTexture(gl.TEXTURE6); gl.bindTexture(gl.TEXTURE_2D, this.sceneTexture);
    gl.activeTexture(gl.TEXTURE7); gl.bindTexture(gl.TEXTURE_2D, this.bloomTextures[1]);
    gl.uniform1i(u.pass, 20); gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.BLEND);
    if (view.spin) this.pending = true; // the brain turns on its own: every frame is new
  }

  _time(t0) {
    const ms = performance.now() - t0;
    this.frameMs = this.frameMs ? this.frameMs * 0.9 + ms * 0.1 : ms;
    return ms;
  }

  _drawFallback() {
    const ctx = this.ctx, r = this.canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, this.options.dpr || 2);
    this.canvas.width = Math.max(1, r.width * dpr); this.canvas.height = Math.max(1, r.height * dpr);
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const [bg0, bg1, bg2] = this.options.background.map((v) => Math.round(v * 255));
    ctx.fillStyle = `rgb(${bg0},${bg1},${bg2})`;
    ctx.fillRect(0, 0, r.width, r.height);
    for (let i = 0; i < this.n; i++) {
      if (this.layout[i * 4 + 3] === 0) continue;
      const [x, y] = this.screen(i);
      const region = this.atlas.regions[this.atlas.region[i]], level = Math.min(1, Math.abs(this.activation[i])), heat = this.heat[i];
      const c = region.color.map((v) => Math.round(v * (0.3 + 0.7 * level) + (255 - v * 0.3) * heat * 0.6));
      ctx.fillStyle = `rgba(${c[0]},${c[1]},${c[2]},${0.35 + 0.65 * level})`;
      ctx.beginPath(); ctx.arc(x, y, 1 + 1.5 * level + 2 * heat, 0, Math.PI * 2); ctx.fill();
    }
  }

  _drawLabels() {
    if (!this.labels) return;
    const r = this.canvas.getBoundingClientRect();
    const placed = [];
    const anchor = (region, k) => {
      if (!this.brain) return { at: this.toScreen(region.center[0], region.center[1] + region.extent[1] * 1.08), centre: this.toScreen(region.center[0], region.center[1]), depth: 0 };
      const c = this.centers3[k], e = this.extents3[k], centre = this._project3(c[0], c[1], c[2]);
      return { at: this._project3(c[0], c[1] + e[1] * 1.15, c[2]), centre, depth: Math.max(0, Math.min(1, (centre[2] - (DIST - 1.3)) / 2.6)) };
    };
    const order = this.atlas.regions.map((region, k) => ({ k, region, ...anchor(region, k) }))
      .sort((a, b) => b.region.count - a.region.count);
    const shown = this.options.labelCount > 0 ? this.options.labelCount : order.length;  // a small card labels only its largest regions
    order.forEach(({ k }, rank) => { if (rank >= shown) this.labelElements[k].style.display = "none"; });
    for (const { k, region, at, centre, depth } of order.slice(0, shown)) {
      const el = this.labelElements[k]; el.style.display = "";
      let [x, y] = at;
      const w = el.offsetWidth || 8 * (region.label ?? region.name).length, h = 14;
      // a label stays on the canvas: clamped to the edges, and only hidden when its region is off screen
      const [cx, cy] = centre;
      el.style.opacity = this.brain ? String(0.92 - 0.55 * depth) : "0.85";
      const inside = cx > -40 && cx < r.width + 40 && cy > -40 && cy < r.height + 40;
      x = Math.max(w / 2 + 4, Math.min(r.width - w / 2 - 4, x));
      y = Math.max(this.options.labelTop, Math.min(r.height - 9, y));
      // larger regions label first; a colliding label steps down until it is clear
      for (let tries = 0; tries < 12; tries++) {
        const hit = placed.some((p) => Math.abs(p.x - x) < (p.w + w) / 2 + 6 && Math.abs(p.y - y) < h);
        if (!hit) break;
        y += h;
      }
      placed.push({ x, y, w });
      el.style.left = `${x}px`; el.style.top = `${y}px`;
      el.style.display = inside ? "" : "none";
    }
  }

  /** The montage rows: the whole brain first, then the largest regions. */
  montage() {
    if (!this._montage) {
      const regions = this.atlas.regions;
      const order = regions.map((_, k) => k).sort((i, j) => regions[j].count - regions[i].count || i - j);
      this._montage = order.slice(0, Math.max(0, this.options.montageRows - 1));
    }
    const rows = [{ label: "brain", color: [255, 255, 255], change: this.global.change, activity: this.global.activity }];
    for (const k of this._montage) rows.push({ label: this.atlas.regions[k].label ?? this.atlas.regions[k].name, color: this.atlas.regions[k].color, change: this.history[k].change, activity: this.history[k].activity });
    return rows;
  }

  _drawStrip() {
    if (!this.strip) return;
    const canvas = this.strip, r = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, this.options.dpr || 2);
    if (!r.width) return;
    canvas.width = r.width * dpr; canvas.height = r.height * dpr;
    const ctx = canvas.getContext("2d");
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    const w = r.width, h = r.height;
    ctx.fillStyle = "#0a1119"; ctx.fillRect(0, 0, w, h);
    const rows = this.montage();
    const font0 = Math.min(11, Math.max(8, ((h - 6) / rows.length) * 0.62));
    ctx.font = `${font0}px system-ui, sans-serif`;
    let longest = 0;
    for (const row of rows) longest = Math.max(longest, ctx.measureText(row.label.length > 16 ? row.label.slice(0, 15) + "…" : row.label).width);
    const gutter = Math.min(w * 0.3, Math.max(64, longest + 14)), x0 = gutter, x1 = w - 10;
    const rowH = (h - 6) / rows.length, span = 319;
    const x = (t) => x0 + (t / span) * (x1 - x0);
    const steps = this.global.change.length;
    const first = this.stepCount - steps;
    ctx.strokeStyle = "rgba(255,255,255,0.08)";
    for (const b of this.boundaries) { const t = b - first; if (t >= 0 && t < steps) { ctx.beginPath(); ctx.moveTo(x(t), 3); ctx.lineTo(x(t), h - 3); ctx.stroke(); } }
    const font = Math.min(11, Math.max(8, rowH * 0.62));
    rows.forEach((row, k) => {
      const top = 3 + k * rowH, base = top + rowH * 0.82, amp = rowH * 0.74;
      ctx.strokeStyle = "rgba(255,255,255,0.05)"; ctx.beginPath(); ctx.moveTo(x0, base + 0.5); ctx.lineTo(x1, base + 0.5); ctx.stroke();
      ctx.fillStyle = rgba(row.color, 0.9); ctx.font = `${font}px system-ui, sans-serif`; ctx.textAlign = "right";
      const label = row.label.length > 16 ? row.label.slice(0, 15) + "…" : row.label;
      ctx.fillText(label, gutter - 8, base);
      if (!steps) return;
      let topA = 1e-9, topC = 1e-9;
      for (const v of row.activity) topA = Math.max(topA, v);
      for (const v of row.change) topC = Math.max(topC, v);
      ctx.fillStyle = rgba(row.color, 0.13);
      ctx.beginPath(); ctx.moveTo(x(0), base);
      row.activity.forEach((v, t) => ctx.lineTo(x(t), base - amp * (v / topA)));
      ctx.lineTo(x(row.activity.length - 1), base); ctx.closePath(); ctx.fill();
      ctx.strokeStyle = rgba(row.color, 0.95); ctx.lineWidth = k === 0 ? 1.2 : 1;
      ctx.beginPath();
      row.change.forEach((v, t) => { const yy = base - amp * (v / topC); t ? ctx.lineTo(x(t), yy) : ctx.moveTo(x(t), yy); });
      ctx.stroke();
    });
    ctx.fillStyle = "#6f8397"; ctx.font = "9px system-ui, sans-serif"; ctx.textAlign = "right";
    ctx.fillText(`change per region (line) · activity (fill) · last ${span + 1} steps`, x1, h - 3);
  }

  inspect(clientX, clientY) {
    if (this.brain) {
      const r = this.canvas.getBoundingClientRect(), px = clientX - r.left, py = clientY - r.top, all = this.screenAll();
      let best = -1, dist = 144;
      for (let i = 0; i < this.n; i++) { if (this.layout[i * 4 + 3] === 0) continue; const d = (all[2 * i] - px) ** 2 + (all[2 * i + 1] - py) ** 2; if (d < dist) { dist = d; best = i; } }
      if (best < 0) return null;
      const region = this.atlas.regions[this.atlas.region[best]];
      return { neuron: best, region: region.name, role: region.role, activation: this.activation[best], change: this.change[best], heat: this.heat[best], potential: this.potential ? this.potential[best] : null };
    }
    const r = this.canvas.getBoundingClientRect(), a = this.aspect();
    const cx = (((clientX - r.left) / r.width) * 2 - 1) / a[0], cy = (1 - ((clientY - r.top) / r.height) * 2) / a[1];
    const k = this.frame.scale * this.camera.zoom * FIT;
    const wx = (cx - this.camera.x) / k + this.frame.x, wy = (cy - this.camera.y) / k + this.frame.y;
    let best = -1, dist = (0.02 / (this.camera.zoom * this.frame.scale)) ** 2;
    for (let i = 0; i < this.n; i++) {
      const d = (this.atlas.positions[2 * i] - wx) ** 2 + (this.atlas.positions[2 * i + 1] - wy) ** 2;
      if (d < dist) { dist = d; best = i; }
    }
    if (best < 0) return null;
    const region = this.atlas.regions[this.atlas.region[best]];
    return { neuron: best, region: region.name, role: region.role, activation: this.activation[best], change: this.change[best], heat: this.heat[best], potential: this.potential ? this.potential[best] : null };
  }

  snapshot() {
    return { renderer: this.enabled ? "WebGL2" : "canvas fallback", software: !!this.software, version: VERSION, style: this.brain ? "brain" : "scan", neurons: this.n, synapses: this.edges, regions: this.atlas.regions.length, steps: this.stepCount, zoom: this.camera.zoom, scale: this.scale, frameMs: this.frameMs, view: this.brain ? { yaw: this.view.yaw, pitch: this.view.pitch } : null, allEdgesSubmitted: this.enabled && (!this.brain || this.edges <= this.options.lineBudget) };
  }
}

/** Replay quantised frames from the atlas exporter at a given rate; returns a stop function. */
export function playFrames(scan, frames, { fps = 30, loop = true, onstep = null } = {}) {
  const decoded = frames.activation instanceof Array ? frames : decodeFrames(frames);
  let t = 0, stopped = false, last = 0;
  scan.set(decoded.activation[0]);
  scan.reset();
  function tick(now) {
    if (stopped) return;
    if (now - last >= 1000 / fps) {
      last = now;
      t++;
      if (t >= decoded.steps) { if (!loop) return; t = 0; scan.set(decoded.activation[0]); scan.reset(); }
      else scan.step(decoded.activation[t]);
      if (onstep) onstep(t, decoded.steps);
    } else scan.draw(now);
    requestAnimationFrame(tick);
  }
  requestAnimationFrame(tick);
  return () => { stopped = true; };
}

// ---------------------------------------------------------------------------------------
// The layout, in the browser: the same atlas the library computes, for pages that build
// their brains at run time. Deterministic under `seed`; regions by the force layout of the
// region graph, neurons by a whitened spectral embedding of their synapses, sheets by
// declared shapes, supplied coordinates kept in shape.

export const PALETTE = { vision: [89, 183, 255], sensory: [143, 225, 157], memory: [171, 153, 255], association: [89, 229, 203], motor: [255, 191, 112], value: [237, 129, 182], other: [143, 163, 184] };
const ROLE_RULES = [
  ["memory", /afterglow|afterimage|context|trace|record|recall|notebook|route|memory|echo|prefrontal|hippocamp/],
  ["value", /value|reward|critic|dopamine|valence|monitor|salience/],
  ["motor", /motor|action|actuator|joint|pencil|pen\b|body|efferen|output|slot|cord|muscle/],
  ["vision", /retina|visual|vision|eye|fovea|periph|pixel|sheet|v1\b|gaze/],
  ["sensory", /sensor|sense|input|cue|key|smell|odou?r|touch|whisker|auditory|ear\b|heard|nose|taste|vestib|place/],
  ["association", /associat|hidden|cortex|assoc|belt|phrase|harmon|rhythm|melody|timbre|intention|interneuron/],
];
export function roleOf(name, roles = {}) {
  if (roles[name]) return roles[name];
  const key = String(name).toLowerCase();
  for (const [role, rule] of ROLE_RULES) if (rule.test(key)) return role;
  return "other";
}
function mulberry(seed) {
  let a = seed >>> 0;
  return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; };
}
function gaussian(rng) { const u = Math.max(1e-12, rng()), v = rng(); return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v); }

function fitInto(points, ids, region, rng, spread = false) {
  const count = ids.length;
  if (!count) return;
  const [ex, ey] = region.extent;
  let mx = 0, my = 0;
  for (const i of ids) { mx += points[2 * i]; my += points[2 * i + 1]; }
  mx /= count; my /= count;
  const qx = new Float64Array(count), qy = new Float64Array(count);
  for (let k = 0; k < count; k++) { const i = ids[k]; qx[k] = points[2 * i] - mx; qy[k] = points[2 * i + 1] - my; }
  if (spread && count >= 3) {
    // whiten: equal variance along both principal axes, then rank the radii to a uniform disc
    for (let k = 0; k < count; k++) { qx[k] += gaussian(rng) * 1e-4 * ex; qy[k] += gaussian(rng) * 1e-4 * ey; }
    let a = 0, b = 0, c = 0;
    for (let k = 0; k < count; k++) { a += qx[k] * qx[k]; b += qx[k] * qy[k]; c += qy[k] * qy[k]; }
    a /= count; b /= count; c /= count;
    const disc = Math.sqrt(Math.max(0, ((a - c) * (a - c)) / 4 + b * b));
    const l1 = (a + c) / 2 + disc, l2 = (a + c) / 2 - disc;
    let vx = 1, vy = 0;
    if (Math.abs(b) > 1e-30) { vx = l1 - c; vy = b; const len = Math.hypot(vx, vy); vx /= len; vy /= len; } else if (c > a) { vx = 0; vy = 1; }
    const s1 = 1 / Math.sqrt(Math.max(l1, 1e-18)), s2 = 1 / Math.sqrt(Math.max(l2, 1e-18));
    const radius = new Float64Array(count), angle = new Float64Array(count);
    for (let k = 0; k < count; k++) {
      const u = (qx[k] * vx + qy[k] * vy) * s1, w = (-qx[k] * vy + qy[k] * vx) * s2;
      radius[k] = Math.hypot(u, w); angle[k] = Math.atan2(w, u);
    }
    const order = Array.from(radius.keys()).sort((i, j) => radius[i] - radius[j] || i - j);
    const rank = new Float64Array(count);
    order.forEach((k, pos) => { rank[k] = pos; });
    for (let k = 0; k < count; k++) { const rad = Math.sqrt((rank[k] + 0.5) / count) * 0.94; qx[k] = rad * Math.cos(angle[k]) * ex; qy[k] = rad * Math.sin(angle[k]) * ey; }
  } else {
    // supplied coordinates keep their shape: one uniform scale fits them into the region's box
    let wx = 1e-9, wy = 1e-9;
    for (let k = 0; k < count; k++) { wx = Math.max(wx, Math.abs(qx[k])); wy = Math.max(wy, Math.abs(qy[k])); }
    const scale = Math.min(ex / wx, ey / wy) * 0.96;
    for (let k = 0; k < count; k++) { qx[k] *= scale; qy[k] *= scale; }
  }
  for (let k = 0; k < count; k++) {
    const i = ids[k];
    points[2 * i] = region.center[0] + qx[k] + gaussian(rng) * 0.004 * ex;
    points[2 * i + 1] = region.center[1] + qy[k] + gaussian(rng) * 0.004 * ey;
  }
}

export function layoutAtlas({ n, pre, post, weight = null, groups, shapes = {}, roles = {}, positions = {}, positions3 = {}, labels = {}, seed = 0, iterations = 24 }) {
  const rng = mulberry(seed);
  const names = [];
  const members = new Map();
  for (let i = 0; i < n; i++) { const g = groups[i] ?? "other"; if (!members.has(g)) { members.set(g, []); names.push(g); } members.get(g).push(i); }
  const regions = names.map((name) => ({ name, label: labels[name] ?? name, role: roleOf(name, roles), indices: members.get(name), center: [0, 0], radius: 0.1, extent: [0.1, 0.1], shape: shapes[name] ? Array.from(shapes[name]) : null, count: members.get(name).length }));
  regions.forEach((r) => { r.color = PALETTE[r.role] || PALETTE.other; });
  const regionIndex = new Uint16Array(n);
  regions.forEach((r, k) => { for (const i of r.indices) regionIndex[i] = k; });
  const K = regions.length, E = pre.length;
  const strength = new Float64Array(E);
  for (let e = 0; e < E; e++) strength[e] = Math.abs(weight ? weight[e] : 1);
  const finish = (pos, pos3 = null, spacing3 = null) => ({ n, synapses: E, seed, regions: regions.map(({ indices, ...rest }) => rest), region: regionIndex, positions: pos, positions3: pos3, spacing3, pre: pre instanceof Uint32Array ? pre : Uint32Array.from(pre), post: post instanceof Uint32Array ? post : Uint32Array.from(post), weight: weight ? Float32Array.from(weight) : new Float32Array(E).fill(1), palette: PALETTE, memberIndices: regions.map((r) => r.indices) });
  // regions of a shared frame are wherever their neurons are: centre, half-extents and radius from the members' bounding box
  const boxRegions = (pos) => {
    for (const r of regions) {
      let mn = [Infinity, Infinity], mx = [-Infinity, -Infinity];
      for (const i of r.indices) { mn[0] = Math.min(mn[0], pos[2 * i]); mn[1] = Math.min(mn[1], pos[2 * i + 1]); mx[0] = Math.max(mx[0], pos[2 * i]); mx[1] = Math.max(mx[1], pos[2 * i + 1]); }
      r.center = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2];
      r.extent = [Math.max(0.02, (mx[0] - mn[0]) / 2), Math.max(0.02, (mx[1] - mn[1]) / 2)];
      r.radius = Math.hypot(...r.extent);
    }
  };
  if (positions3['*']) {
    // A measured anatomy in three dimensions: centred and scaled into the brain style's
    // frame; the scan draws its (x, y) projection scaled into the square.
    const pos3 = fitPositions3(positions3['*'], n), pos = new Float32Array(2 * n);
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (let i = 0; i < n; i++) { lo[0] = Math.min(lo[0], pos3[3 * i]); hi[0] = Math.max(hi[0], pos3[3 * i]); lo[1] = Math.min(lo[1], pos3[3 * i + 1]); hi[1] = Math.max(hi[1], pos3[3 * i + 1]); }
    const scale = 1.84 / Math.max(1e-9, hi[0] - lo[0], hi[1] - lo[1]);
    for (let i = 0; i < n; i++) { pos[2 * i] = (pos3[3 * i] - (lo[0] + hi[0]) / 2) * scale; pos[2 * i + 1] = (pos3[3 * i + 1] - (lo[1] + hi[1]) / 2) * scale; }
    boxRegions(pos);
    return finish(pos, pos3, spacing3Of(pos3, n));
  }
  if (positions['*']) {
    // One shared frame for every neuron (an anatomy): kept as given, scaled into the square;
    // regions are then wherever their neurons are.
    const all = positions['*'];
    if (all.length !== n) throw Error(`positions['*'] must hold ${n} points`);
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    for (const [x, y] of all) { lo[0] = Math.min(lo[0], x); lo[1] = Math.min(lo[1], y); hi[0] = Math.max(hi[0], x); hi[1] = Math.max(hi[1], y); }
    const scale = 1.84 / Math.max(1e-9, hi[0] - lo[0], hi[1] - lo[1]);
    const pos = new Float32Array(2 * n);
    for (let i = 0; i < n; i++) { pos[2 * i] = (all[i][0] - (lo[0] + hi[0]) / 2) * scale; pos[2 * i + 1] = (all[i][1] - (lo[1] + hi[1]) / 2) * scale; }
    boxRegions(pos);
    return finish(pos);
  }
  // region graph
  const mass = new Float64Array(K * K);
  for (let e = 0; e < E; e++) { const a = regionIndex[pre[e]], b = regionIndex[post[e]]; if (a !== b) { mass[a * K + b] += strength[e]; mass[b * K + a] += strength[e]; } }
  let massMax = 1e-12; for (const m of mass) massMax = Math.max(massMax, m);
  regions.forEach((r) => {
    const rad = Math.min(0.5, Math.max(0.05, 0.58 * Math.sqrt(r.count / Math.max(n, 1))));
    let aspect = 1;
    if (r.shape && r.shape.length >= 2) aspect = Math.sqrt(Math.max(r.shape[1], 1) / Math.max(r.shape[0], 1));
    else if (positions[r.name] && positions[r.name].length > 1) {
      let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
      for (const [x, y] of positions[r.name]) { lo[0] = Math.min(lo[0], x); lo[1] = Math.min(lo[1], y); hi[0] = Math.max(hi[0], x); hi[1] = Math.max(hi[1], y); }
      aspect = Math.sqrt(Math.max(1e-6, hi[0] - lo[0]) / Math.max(1e-6, hi[1] - lo[1]));
    }
    aspect = Math.min(3, Math.max(1 / 3, aspect));
    r.extent = [rad * aspect, rad / aspect]; r.radius = Math.hypot(...r.extent);
  });
  if (K === 1) regions[0].center = [0, 0];
  else {
    const order = [regions.reduce((best, r, k) => (r.count > regions[best].count ? k : best), 0)];
    while (order.length < K) {
      let bestK = -1, bestPull = -1;
      for (let k = 0; k < K; k++) { if (order.includes(k)) continue; let pull = 0; for (const o of order) pull += mass[k * K + o]; if (pull > bestPull) { bestPull = pull; bestK = k; } }
      order.push(bestK);
    }
    const centre = regions.map(() => [0, 0]);
    order.forEach((k, rank) => { const angle = (2 * Math.PI * rank) / K; centre[k] = [0.6 * Math.cos(angle), 0.6 * Math.sin(angle)]; });
    const velocity = regions.map(() => [0, 0]);
    for (let it = 0; it < 400; it++) {
      const force = centre.map((c) => [-0.03 * c[0], -0.03 * c[1]]);
      for (let i = 0; i < K; i++) for (let j = i + 1; j < K; j++) {
        const dx = centre[j][0] - centre[i][0], dy = centre[j][1] - centre[i][1];
        const dist = Math.hypot(dx, dy) + 1e-9, ux = dx / dist, uy = dy / dist;
        const want = regions[i].radius + regions[j].radius + 0.1, s = mass[i * K + j] / massMax;
        let f;
        if (dist < want) f = -(want - dist) * 1.5;
        else if (s > 0) f = (dist - want) * (0.15 + 0.85 * s);
        else f = -0.002 / dist;
        force[i][0] += f * ux; force[i][1] += f * uy; force[j][0] -= f * ux; force[j][1] -= f * uy;
      }
      for (let k = 0; k < K; k++) { velocity[k][0] = 0.6 * velocity[k][0] + 0.08 * force[k][0]; velocity[k][1] = 0.6 * velocity[k][1] + 0.08 * force[k][1]; centre[k][0] += velocity[k][0]; centre[k][1] += velocity[k][1]; }
    }
    let lo = [Infinity, Infinity], hi = [-Infinity, -Infinity];
    regions.forEach((r, k) => { for (const a of [0, 1]) { lo[a] = Math.min(lo[a], centre[k][a] - r.radius); hi[a] = Math.max(hi[a], centre[k][a] + r.radius); } });
    const scale = Math.min(1.84 / Math.max(1e-9, hi[0] - lo[0]), 1.84 / Math.max(1e-9, hi[1] - lo[1]));
    regions.forEach((r, k) => { r.center = [(centre[k][0] - (lo[0] + hi[0]) / 2) * scale, (centre[k][1] - (lo[1] + hi[1]) / 2) * scale]; r.radius *= scale; r.extent = [r.extent[0] * scale, r.extent[1] * scale]; });
  }
  const pos = new Float32Array(2 * n);
  const free = new Uint8Array(n);
  for (const r of regions) {
    if (r.shape) {
      let [h, w] = [r.shape[0], r.shape[1]]; let channels = r.shape.slice(2).reduce((a, b) => a * b, 1) || 1;
      if (h * w * channels !== r.count) { w = Math.ceil(Math.sqrt(r.count)); h = Math.ceil(r.count / w); channels = 1; }
      r.indices.forEach((i, idx) => { const ch = idx % channels, cell = Math.floor(idx / channels), row = Math.floor(cell / w), col = cell % w; pos[2 * i] = r.center[0] + (((col + 0.5) / w) * 2 - 1) * r.extent[0] + (ch - (channels - 1) / 2) * (0.4 * r.extent[0] / w); pos[2 * i + 1] = r.center[1] + (1 - ((row + 0.5) / h) * 2) * r.extent[1]; });
    } else if (positions[r.name]) {
      const given = positions[r.name];
      r.indices.forEach((i, idx) => { pos[2 * i] = given[idx][0]; pos[2 * i + 1] = given[idx][1]; });
      fitInto(pos, r.indices, r, rng);
    } else {
      for (const i of r.indices) { const rad = Math.sqrt(rng()), ang = rng() * 2 * Math.PI; pos[2 * i] = r.center[0] + rad * Math.cos(ang) * r.extent[0] * 0.9; pos[2 * i + 1] = r.center[1] + rad * Math.sin(ang) * r.extent[1] * 0.9; free[i] = 1; }
    }
  }
  if (E && free.some((f) => f)) {
    const den = new Float64Array(n);
    for (let e = 0; e < E; e++) { den[post[e]] += strength[e]; den[pre[e]] += strength[e]; }
    const numX = new Float64Array(n), numY = new Float64Array(n), mixed = new Float32Array(2 * n);
    for (let step = 0; step < iterations; step++) {
      numX.fill(0); numY.fill(0);
      for (let e = 0; e < E; e++) { const a = pre[e], b = post[e], w = strength[e]; numX[b] += w * pos[2 * a]; numY[b] += w * pos[2 * a + 1]; numX[a] += w * pos[2 * b]; numY[a] += w * pos[2 * b + 1]; }
      for (let i = 0; i < n; i++) { const tx = den[i] > 0 ? numX[i] / den[i] : pos[2 * i], ty = den[i] > 0 ? numY[i] / den[i] : pos[2 * i + 1]; mixed[2 * i] = 0.45 * pos[2 * i] + 0.55 * tx; mixed[2 * i + 1] = 0.45 * pos[2 * i + 1] + 0.55 * ty; }
      const jitter = mulberry(seed + 1000 + step);
      for (const r of regions) { if (!r.indices.length || !free[r.indices[0]]) continue; for (const i of r.indices) { pos[2 * i] = mixed[2 * i]; pos[2 * i + 1] = mixed[2 * i + 1]; } fitInto(pos, r.indices, r, jitter, true); }
    }
  }
  return finish(pos);
}
