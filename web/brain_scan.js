// Cadence brain scan: the standard whole-brain view.
// Every neuron is a point, every synapse a line, laid out by the atlas the library exports.
// The brain is drawn as a scan: a field of tissue in each region's colour whose brightness
// is the activation, a hot glow where neurons changed in the last steps (the scan colour
// map runs violet, magenta, orange, white), synapses that light up when their presynaptic
// neuron changes, and messages that travel along synapses as they are sent, so a settling
// reads as a wave through the brain. The traces below the map are an EEG-style montage of
// every region's own activity and change over the last steps. WebGL2 draws it; a Canvas2D
// fallback draws the neurons only. No activity is invented here: every value drawn comes
// from the activations the page feeds in.
export const VERSION = "cadence.brain-scan/v2";

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
    region: decodeArray(atlas.region),
    positions: decodeArray(atlas.positions),
    pre: decodeArray(atlas.pre),
    post: decodeArray(atlas.post),
    weight: decodeArray(atlas.weight),
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
uniform float pixelsPerUnit; uniform float maxPoint; uniform float glow; uniform float fit; uniform vec3 frame; uniform float screenPPU; uniform int dust;
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
 vec3 hot=vec3(1.0,0.93,0.78);
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

export class BrainScan {
  constructor(canvas, atlas, options = {}) {
    this.canvas = canvas;
    this.options = { particles: true, edges: true, field: true, glow: 2.2, heatDecay: 0.86, background: [0.03, 0.055, 0.085], mode: "activity", montageRows: 16, particleBudget: 300000, lineBudget: 400000, ...options };
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
    if (this.enabled) this._uploadAtlas(resized);
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
    const names = ["layoutTex", "colorTex", "stateTex", "cachedTex", "fieldTex", "pass", "clock", "baseAlpha", "dpr", "camera", "aspect", "mode", "background", "pixelsPerUnit", "maxPoint", "glow", "fit", "frame", "screenPPU", "dust", "edgeGain", "fieldGain", "heatGain"];
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
    if (this.edges > this.options.particleBudget) {
      const order = new Uint32Array(this.edges), rng = mulberry(7);
      for (let i = 0; i < this.edges; i++) order[i] = i;
      for (let i = this.edges - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); const t = order[i]; order[i] = order[j]; order[j] = t; }
      this.order = order;
    }
    this.flat = new Float32Array(Math.max(1, this.edges) * 3);
    for (let k = 0; k < this.edges; k++) { const i = this.order ? this.order[k] : k; this.flat.set([this.atlas.pre[i], this.atlas.post[i], this.atlas.weight[i]], k * 3); }
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
    this.dirty = true;
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
      if (this.pointers.size === 1) { this.camera.x += ((e.clientX - old[0]) * 2) / r.width / this.aspect()[0]; this.camera.y -= ((e.clientY - old[1]) * 2) / r.height / this.aspect()[1]; this.fitted = false; this.dirty = true; this.pending = true; }
      this.pointers.set(e.pointerId, [e.clientX, e.clientY]);
      this.draw();
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) target.addEventListener(type, (e) => this.pointers.delete(e.pointerId));
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
  screen(i) { return this.toScreen(this.atlas.positions[2 * i], this.atlas.positions[2 * i + 1]); }

  /** Screen positions of every neuron (CSS pixels inside the canvas) as [x0, y0, x1, y1, ...]. */
  screenAll() {
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

  fit() { this.camera = { x: 0, y: 0, zoom: 1 }; this.fitted = true; this._frame(); this.dirty = true; this.pending = true; this.draw(); }

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
    if (this.enabled) this._upload(2, this.state);
  }

  draw(time = performance.now()) {
    // A software renderer draws a new frame at most every 200 ms; a state change always draws.
    if (this.software && !this.pending && time - this.lastTime < 200) return;
    this.pending = false;
    this.clock += Math.min(0.1, (time - this.lastTime) / 1000);
    this.lastTime = time;
    this._drawLabels();
    this._drawStrip();
    if (!this.enabled) return this._drawFallback();
    const gl = this.gl, r = this.canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(r.width * dpr)), height = Math.max(1, Math.round(r.height * dpr));
    if (this.canvas.width !== width || this.canvas.height !== height) {
      this.canvas.width = width; this.canvas.height = height; this.dirty = true;
      if (this.fitted) this._frame();
      this._target(this.cache, this.cacheTexture, width, height, false);
      this.fieldSize = [Math.max(1, Math.round(width / 2)), Math.max(1, Math.round(height / 2))];
      this._target(this.fieldBuffer, this.fieldTexture, this.fieldSize[0], this.fieldSize[1], true);
    }
    const a = this.aspect();
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    gl.uniform3f(this.uniform.camera, this.camera.x, this.camera.y, this.camera.zoom);
    gl.uniform2f(this.uniform.aspect, a[0], a[1]);
    gl.uniform1f(this.uniform.clock, this.clock);
    gl.uniform1f(this.uniform.dpr, dpr);
    gl.uniform1f(this.uniform.fit, FIT);
    gl.uniform3f(this.uniform.frame, this.frame.x, this.frame.y, this.frame.scale);
    gl.uniform1f(this.uniform.screenPPU, (this.frame.scale * this.camera.zoom * FIT * Math.min(width, height)) / 2);
    gl.uniform1i(this.uniform.mode, this.options.mode === "potential" ? 2 : this.options.mode === "change" ? 3 : 0);
    gl.uniform3f(this.uniform.background, ...this.options.background);
    gl.uniform1f(this.uniform.baseAlpha, Math.max(this.floatCache ? 0.00005 : 0.008, 0.09 / Math.pow(Math.max(1, this.edges / 2000), 0.6)));
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
  }

  _drawFallback() {
    const ctx = this.ctx, r = this.canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
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
    const order = this.atlas.regions.map((region, k) => ({ k, region, at: this.toScreen(region.center[0], region.center[1] + region.extent[1] * 1.08) }))
      .sort((a, b) => b.region.count - a.region.count);
    for (const { k, region, at } of order) {
      const el = this.labelElements[k];
      let [x, y] = at;
      const w = el.offsetWidth || 8 * (region.label ?? region.name).length, h = 14;
      // a label stays on the canvas: clamped to the edges, and only hidden when its region is off screen
      const [cx, cy] = this.toScreen(region.center[0], region.center[1]);
      const inside = cx > -40 && cx < r.width + 40 && cy > -40 && cy < r.height + 40;
      x = Math.max(w / 2 + 4, Math.min(r.width - w / 2 - 4, x));
      y = Math.max(9, Math.min(r.height - 9, y));
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
    const canvas = this.strip, r = canvas.getBoundingClientRect(), dpr = Math.min(devicePixelRatio || 1, 2);
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
    return { renderer: this.enabled ? "WebGL2" : "canvas fallback", software: !!this.software, version: VERSION, neurons: this.n, synapses: this.edges, regions: this.atlas.regions.length, steps: this.stepCount, zoom: this.camera.zoom, scale: this.scale, allEdgesSubmitted: this.enabled };
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

export function layoutAtlas({ n, pre, post, weight = null, groups, shapes = {}, roles = {}, positions = {}, labels = {}, seed = 0, iterations = 24 }) {
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
  const finish = (pos) => ({ n, synapses: E, regions: regions.map(({ indices, ...rest }) => rest), region: regionIndex, positions: pos, pre: pre instanceof Uint32Array ? pre : Uint32Array.from(pre), post: post instanceof Uint32Array ? post : Uint32Array.from(post), weight: weight ? Float32Array.from(weight) : new Float32Array(E).fill(1), palette: PALETTE, memberIndices: regions.map((r) => r.indices) });
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
    for (const r of regions) {
      let mn = [Infinity, Infinity], mx = [-Infinity, -Infinity];
      for (const i of r.indices) { mn[0] = Math.min(mn[0], pos[2 * i]); mn[1] = Math.min(mn[1], pos[2 * i + 1]); mx[0] = Math.max(mx[0], pos[2 * i]); mx[1] = Math.max(mx[1], pos[2 * i + 1]); }
      r.center = [(mn[0] + mx[0]) / 2, (mn[1] + mx[1]) / 2];
      r.extent = [Math.max(0.02, (mx[0] - mn[0]) / 2), Math.max(0.02, (mx[1] - mn[1]) / 2)];
      r.radius = Math.hypot(...r.extent);
    }
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
