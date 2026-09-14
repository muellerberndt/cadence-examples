// Full topology rendering. Every supplied owner and seam reaches the GPU.
// Dense edges overlap on screen; zoom resolves them. No simulated activity is invented here.
export function regionColor(name) {
  const key = name.toLowerCase();
  if (/retina|visual|vision|eye/.test(key)) return [218, 146, 246];
  if (/motor|motion|joint|pencil|action|actuator/.test(key)) return [85, 210, 232];
  if (/rhythm|timing|tempo/.test(key)) return [107, 166, 255];
  if (/harmon|chord/.test(key)) return [244, 186, 101];
  if (/memory|recall|record|motif|phrase/.test(key)) return [176, 147, 255];
  if (/future|plan|reason|monitor|intention|value|candidate|option/.test(key))
    return [246, 145, 173];
  if (/sensory|sensor|smell|auditory|heard|input|cue|key/.test(key)) return [116, 220, 172];
  if (/place|spatial/.test(key)) return [106, 194, 255];
  if (/instrument|ensemble/.test(key)) return [246, 157, 101];
  return [143, 200, 190];
}

const vertex = `#version 300 es
precision highp float;
precision highp int;
layout(location=0) in vec3 edge;
layout(location=1) in float edgeChange;
uniform sampler2D layoutTex;
uniform sampler2D stateTex;
uniform sampler2D colorTex;
uniform int pass;
uniform int channel;
uniform float clock;
uniform float moving;
uniform float baseAlpha;
uniform float dpr;
uniform float changeScale;
uniform vec3 camera;
out vec4 color;
out vec2 uv;
vec4 item(sampler2D tex,int id){return texelFetch(tex,ivec2(id%256,id/256),0);}
void main(){
 uv=vec2(0.0);
 if(pass==4){uv=vec2(float((gl_VertexID<<1)&2),float(gl_VertexID&2));gl_Position=vec4(uv*2.0-1.0,0.0,1.0);color=vec4(0.0);return;}
 int a=int(edge.x), b=int(edge.y);
 if(pass==1){a=gl_VertexID;b=a;}
 vec4 la=item(layoutTex,a), lb=item(layoutTex,b);
 vec4 sa=item(stateTex,a), sb=item(stateTex,b);
 vec3 role=item(colorTex,a).rgb;
 float value=channel==0?sa.x:channel==1?sa.y:channel==2?sa.z:sa.w;
 float strength=clamp(abs(sa.y*edge.z),0.0,1.0);
 vec2 p=la.xy;
 if(pass==0){
   p=gl_VertexID==0?la.xy:lb.xy;
   color=vec4(mix(role,item(colorTex,b).rgb,.35),baseAlpha*(.45+min(3.0,abs(edge.z))));
 }else if(pass==3){
   p=gl_VertexID==0?la.xy:lb.xy;
   color=vec4(edgeChange<0.0?vec3(.55,.7,1.0):vec3(1.0,.72,.35),clamp(abs(edgeChange)/changeScale,0.0,1.0)*.45);
 }else if(pass==1){
   float intensity=clamp(abs(value),0.0,1.0);
   vec3 tint=value<0.0?vec3(.45,.66,1.0):vec3(1.0,.94,.76);
   color=vec4(mix(role,tint,intensity*.65),.22+intensity*.78);
   gl_PointSize=dpr*(2.0+5.0*intensity);
 }else{
   float travel=fract(clock*.85+float(gl_InstanceID%23)/23.0);
   p=mix(la.xy,lb.xy,travel);
   color=vec4(sa.y*edge.z<0.0?vec3(.45,.7,1.0):vec3(1.0,.8,.42),min(.85,strength*2.0)*moving);
   if(strength<.025)color.a=0.0;
   gl_PointSize=dpr*(1.5+2.5*strength);
 }
 color.a*=la.w*lb.w;
 gl_Position=vec4(p*camera.z+camera.xy,0.0,1.0);
}`;
const fragment = `#version 300 es
precision highp float;
precision highp int;
in vec4 color;
in vec2 uv;
uniform sampler2D cachedTex;
uniform int pass;
out vec4 result;
void main(){
 if(pass==4){vec3 rgb=texture(cachedTex,uv).rgb;result=vec4(vec3(.035,.075,.09)+1.0-exp(-rgb*.65),1.0);return;}
 float alpha=color.a;
 if(pass==1||pass==2){float r=length(gl_PointCoord-.5)*2.0;if(r>1.0)discard;alpha*=pow(1.0-r,.55);}
 result=vec4(color.rgb,alpha);
}`;

export class CircuitMap {
  constructor(canvas, interaction = canvas) {
    this.canvas = canvas;
    this.camera = { x: 0, y: 0, zoom: 1 };
    this.pointers = new Map();
    this.n = 0;
    this.edges = 0;
    this.topology = null;
    this.gl = canvas.getContext("webgl2", {
      alpha: true,
      antialias: false,
      premultipliedAlpha: true,
    });
    this.enabled = !!this.gl;
    if (this.enabled) {
      const gl = this.gl;
      const shader = (kind, source) => {
        const s = gl.createShader(kind);
        gl.shaderSource(s, source);
        gl.compileShader(s);
        if (!gl.getShaderParameter(s, gl.COMPILE_STATUS))
          throw Error(gl.getShaderInfoLog(s));
        return s;
      };
      this.program = gl.createProgram();
      gl.attachShader(this.program, shader(gl.VERTEX_SHADER, vertex));
      gl.attachShader(this.program, shader(gl.FRAGMENT_SHADER, fragment));
      gl.linkProgram(this.program);
      if (!gl.getProgramParameter(this.program, gl.LINK_STATUS))
        throw Error(gl.getProgramInfoLog(this.program));
      gl.useProgram(this.program);
      this.uniform = Object.fromEntries(
        [
          "layoutTex",
          "stateTex",
          "colorTex",
          "pass",
          "channel",
          "clock",
          "moving",
          "baseAlpha",
          "dpr",
          "camera",
          "changeScale",
          "cachedTex",
        ].map((k) => [k, gl.getUniformLocation(this.program, k)]),
      );
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
      gl.uniform1i(this.uniform.stateTex, 1);
      gl.uniform1i(this.uniform.colorTex, 2);
      gl.uniform1i(this.uniform.cachedTex, 3);
      this.vao = gl.createVertexArray();
      gl.bindVertexArray(this.vao);
      this.buffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
      gl.enableVertexAttribArray(0);
      gl.vertexAttribPointer(0, 3, gl.FLOAT, false, 12, 0);
      gl.vertexAttribDivisor(0, 1);
      this.changeBuffer = gl.createBuffer();
      gl.bindBuffer(gl.ARRAY_BUFFER, this.changeBuffer);
      gl.enableVertexAttribArray(1);
      gl.vertexAttribPointer(1, 1, gl.FLOAT, false, 4, 0);
      gl.vertexAttribDivisor(1, 1);
      this.floatCache = !!gl.getExtension("EXT_color_buffer_float");
      gl.enable(gl.BLEND);
      gl.blendFuncSeparate(gl.SRC_ALPHA, gl.ONE, gl.ONE, gl.ONE);
      this.cache = gl.createFramebuffer();
      this.cacheTexture = gl.createTexture();
      this.dirty = true;
    }
    interaction.style.touchAction = "none";
    interaction.addEventListener(
      "wheel",
      (e) => {
        e.preventDefault();
        const r = interaction.getBoundingClientRect();
        const x = ((e.clientX - r.left) / r.width) * 2 - 1,
          y = 1 - ((e.clientY - r.top) / r.height) * 2;
        this.zoomAt(Math.exp(-e.deltaY * 0.0015), x, y);
      },
      { passive: false },
    );
    interaction.addEventListener("pointerdown", (e) => {
      interaction.setPointerCapture(e.pointerId);
      this.pointers.set(e.pointerId, [e.clientX, e.clientY]);
    });
    interaction.addEventListener("pointermove", (e) => {
      if (!this.pointers.has(e.pointerId)) return;
      const old = this.pointers.get(e.pointerId),
        r = interaction.getBoundingClientRect();
      if (this.pointers.size === 1) {
        this.camera.x += ((e.clientX - old[0]) * 2) / r.width;
        this.camera.y -= ((e.clientY - old[1]) * 2) / r.height;
      } else {
        const other = [...this.pointers.entries()].find(
          ([id]) => id !== e.pointerId,
        )[1];
        const before = Math.hypot(old[0] - other[0], old[1] - other[1]);
        const after = Math.hypot(e.clientX - other[0], e.clientY - other[1]);
        if (before > 5) this.zoomAt(after / before, 0, 0);
      }
      this.pointers.set(e.pointerId, [e.clientX, e.clientY]);
    });
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"])
      interaction.addEventListener(type, (e) =>
        this.pointers.delete(e.pointerId),
      );
  }
  zoomAt(factor, x = 0, y = 0) {
    const z = Math.max(0.55, Math.min(30, this.camera.zoom * factor)),
      ratio = z / this.camera.zoom;
    this.camera.x = x - (x - this.camera.x) * ratio;
    this.camera.y = y - (y - this.camera.y) * ratio;
    this.camera.zoom = z;
  }
  fit() {
    this.camera = { x: 0, y: 0, zoom: 1 };
  }
  transformContext(ctx, w, h) {
    ctx.translate(
      ((1 - this.camera.zoom + this.camera.x) * w) / 2,
      ((1 - this.camera.zoom - this.camera.y) * h) / 2,
    );
    ctx.scale(this.camera.zoom, this.camera.zoom);
  }
  worldPoint(x, y, w, h) {
    return [
      (x - ((1 - this.camera.zoom + this.camera.x) * w) / 2) / this.camera.zoom,
      (y - ((1 - this.camera.zoom - this.camera.y) * h) / 2) / this.camera.zoom,
    ];
  }
  graph(n, edges, id = null) {
    const flat =
      edges instanceof Float32Array ? edges : Float32Array.from(edges.flat());
    this.n = n;
    this.edges = flat.length / 3;
    this.topology = flat;
    this.id = id;
    this.dirty = true;
    if (!this.enabled) return;
    const gl = this.gl;
    gl.bindBuffer(gl.ARRAY_BUFFER, this.buffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      flat.length ? flat : new Float32Array(3),
      gl.STATIC_DRAW,
    );
    gl.bindBuffer(gl.ARRAY_BUFFER, this.changeBuffer);
    gl.bufferData(
      gl.ARRAY_BUFFER,
      new Float32Array(Math.max(1, this.edges)),
      gl.DYNAMIC_DRAW,
    );
    this.changes = null;
    this.changeScale = 1;
    this.rows = Math.max(1, Math.ceil(n / 256));
    this.layout = new Float32Array(256 * this.rows * 4);
    this.state = new Float32Array(this.layout.length);
    this.colors = new Float32Array(this.layout.length);
    this.textures.forEach((t, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, t);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        gl.RGBA32F,
        256,
        this.rows,
        0,
        gl.RGBA,
        gl.FLOAT,
        null,
      );
    });
  }
  plasticity(changes) {
    if (changes.length !== this.edges)
      throw Error("Plasticity must cover every seam");
    this.changes = changes;
    this.changeScale = 1e-9;
    for (const x of changes)
      this.changeScale = Math.max(this.changeScale, Math.abs(x));
    if (this.enabled) {
      const gl = this.gl;
      gl.bindBuffer(gl.ARRAY_BUFFER, this.changeBuffer);
      gl.bufferData(gl.ARRAY_BUFFER, changes, gl.DYNAMIC_DRAW);
    }
  }
  draw({
    positions,
    groups,
    activation,
    repair,
    mismatch,
    plasticity,
    scales,
    mask,
    channel = 0,
    time = 0,
    moving = true,
    nodes = true,
  }) {
    if (!this.enabled || !this.n || positions.length !== this.n) return;
    const gl = this.gl,
      r = this.canvas.getBoundingClientRect(),
      dpr = Math.min(devicePixelRatio, 2);
    const width = Math.round(r.width * dpr),
      height = Math.round(r.height * dpr);
    if (
      !this.cacheWidth ||
      this.canvas.width !== width ||
      this.canvas.height !== height
    ) {
      this.cacheWidth = width;
      this.canvas.width = width;
      this.canvas.height = height;
      this.dirty = true;
      gl.bindTexture(gl.TEXTURE_2D, this.cacheTexture);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.NEAREST);
      gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MAG_FILTER, gl.NEAREST);
      gl.texImage2D(
        gl.TEXTURE_2D,
        0,
        this.floatCache ? gl.RGBA16F : gl.RGBA8,
        width,
        height,
        0,
        gl.RGBA,
        this.floatCache ? gl.HALF_FLOAT : gl.UNSIGNED_BYTE,
        null,
      );
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.cache);
      gl.framebufferTexture2D(
        gl.FRAMEBUFFER,
        gl.COLOR_ATTACHMENT0,
        gl.TEXTURE_2D,
        this.cacheTexture,
        0,
      );
    }
    const view = [this.camera.x, this.camera.y, this.camera.zoom].join(":");
    if (view !== this.view) {
      this.view = view;
      this.dirty = true;
    }
    gl.viewport(0, 0, width, height);
    gl.clearColor(0, 0, 0, 0);
    const channels = [activation, repair, mismatch, plasticity];
    for (let i = 0; i < this.n; i++) {
      const x = (positions[i][0] / r.width) * 2 - 1,
        y = 1 - (positions[i][1] / r.height) * 2,
        keep = mask?.[i] === 0 ? 0 : 1;
      if (
        this.layout[i * 4] !== Math.fround(x) ||
        this.layout[i * 4 + 1] !== Math.fround(y) ||
        this.layout[i * 4 + 3] !== keep
      )
        this.dirty = true;
      this.layout.set([x, y, 0, keep], i * 4);
      this.colors.set(
        [...regionColor(groups[i]).map((v) => v / 255), 1],
        i * 4,
      );
      for (let k = 0; k < 4; k++)
        this.state[i * 4 + k] =
          (channels[k]?.[i] ?? 0) / Math.max(0.00001, scales?.[k]?.[i] ?? 1);
    }
    gl.useProgram(this.program);
    gl.bindVertexArray(this.vao);
    [this.layout, this.state, this.colors].forEach((data, i) => {
      gl.activeTexture(gl.TEXTURE0 + i);
      gl.bindTexture(gl.TEXTURE_2D, this.textures[i]);
      gl.texSubImage2D(
        gl.TEXTURE_2D,
        0,
        0,
        0,
        256,
        this.rows,
        gl.RGBA,
        gl.FLOAT,
        data,
      );
    });
    gl.uniform3f(
      this.uniform.camera,
      this.camera.x,
      this.camera.y,
      this.camera.zoom,
    );
    gl.uniform1i(this.uniform.channel, channel);
    gl.uniform1f(this.uniform.clock, time);
    gl.uniform1f(this.uniform.moving, moving ? 1 : 0);
    gl.uniform1f(this.uniform.dpr, dpr);
    gl.uniform1f(
      this.uniform.baseAlpha,
      Math.max(
        this.floatCache ? 0.00004 : 0.006,
        0.05 / Math.pow(Math.max(1, this.edges / 3000), 0.65),
      ),
    );
    if (this.dirty) {
      gl.activeTexture(gl.TEXTURE3);
      gl.bindTexture(gl.TEXTURE_2D, this.textures[2]);
      gl.bindFramebuffer(gl.FRAMEBUFFER, this.cache);
      gl.clear(gl.COLOR_BUFFER_BIT);
      gl.uniform1i(this.uniform.pass, 0);
      gl.drawArraysInstanced(gl.LINES, 0, 2, this.edges);
      this.dirty = false;
    }
    gl.bindFramebuffer(gl.FRAMEBUFFER, null);
    gl.activeTexture(gl.TEXTURE3);
    gl.bindTexture(gl.TEXTURE_2D, this.cacheTexture);
    gl.disable(gl.BLEND);
    gl.uniform1i(this.uniform.pass, 4);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
    gl.enable(gl.BLEND);
    if (channel === 3 && this.changes) {
      gl.uniform1f(this.uniform.changeScale, this.changeScale);
      gl.uniform1i(this.uniform.pass, 3);
      gl.drawArraysInstanced(gl.LINES, 0, 2, this.edges);
    } else if (moving) {
      gl.uniform1i(this.uniform.pass, 2);
      gl.drawArraysInstanced(gl.POINTS, 0, 1, this.edges);
    }
    if (nodes) {
      gl.uniform1i(this.uniform.pass, 1);
      gl.drawArrays(gl.POINTS, 0, this.n);
    }
  }
  snapshot() {
    return {
      renderer: this.enabled ? "WebGL2" : "canvas fallback",
      owners: this.n,
      seams: this.edges,
      zoom: this.camera.zoom,
      allEdgesSubmitted: this.enabled,
    };
  }
}
