// Every vertex and edge is submitted. No sampled neuron or synapse subset.
const palette={vision:[.35,.72,1],memory:[.68,.59,1],association:[.35,.9,.79],motor:[1,.75,.44],reward:[.93,.51,.72]};
const areas={
 retina:[20,65,220,225,'vision','Retina'],afterimage:[20,355,220,220,'memory','Afterglow'],
 v1_fovea:[300,85,155,190,'vision','Foveal vision'],v1_periphery:[300,365,155,190,'vision','Peripheral vision'],
 context:[535,65,180,80,'memory','Reverberating context'],association:[545,210,190,195,'association','Association'],
 recall:[540,485,190,55,'memory','Route recall'],motor:[835,110,105,85,'motor','Motor decisions'],
 body:[825,305,110,90,'motor','Body'],efference:[825,475,110,60,'memory','Motor trace'],
 habit:[975,390,95,80,'motor','Fixed habit'],habit_cue:[975,280,95,70,'motor','Habit cue']};
function compile(gl,type,source){const s=gl.createShader(type);gl.shaderSource(s,source);gl.compileShader(s);if(!gl.getShaderParameter(s,gl.COMPILE_STATUS))throw Error(gl.getShaderInfoLog(s));return s;}
function program(gl,vs,fs){const p=gl.createProgram();gl.attachShader(p,compile(gl,gl.VERTEX_SHADER,vs));gl.attachShader(p,compile(gl,gl.FRAGMENT_SHADER,fs));gl.linkProgram(p);if(!gl.getProgramParameter(p,gl.LINK_STATUS))throw Error(gl.getProgramInfoLog(p));return p;}
const common=`precision highp float; precision highp int;
uniform sampler2D positions; uniform sampler2D colors; uniform sampler2D states; uniform sampler2D auxiliary;
uniform vec2 viewport; uniform vec4 camera; uniform float gain; uniform int mode; uniform int core; uniform int live; uniform int critic; uniform int overview;
vec4 item(sampler2D t,int i){return texelFetch(t,ivec2(i%1024,i/1024),0);}
vec2 loc(int i){vec2 p=item(positions,i).xy*camera.zw+camera.xy;return p/viewport*2.-1.;}
float exists(int i){return i<live||i>=critic?1.:0.;}
float intensity(float x){return clamp(log(1.+abs(x)*gain)/3.,0.,1.);}
`;
export class Circuit {
 constructor(canvas,labels,g){
  this.canvas=canvas;this.labels=labels;this.g=g;this.n=g.meta.neurons;this.e=g.meta.synapses;
  this.addresses=g.meta.addresses;this.r=this.addresses.length;this.critic=this.n+this.r*29;this.total=this.critic+2;
  this.totalEdges=this.e+this.r*56+g.meta.critic_index.length;
  this.position=new Float32Array(Math.ceil(this.total/1024)*1024*4);
  this.color=new Float32Array(this.position.length);this.state=new Float32Array(this.position.length);this.aux=new Float32Array(this.position.length);
  this.weight=new Float32Array(this.totalEdges*2);this.names=[];this.regions=[];
  const place=(ids,box,name)=>{const [x,y,w,h,role,title]=box,cols=Math.max(1,Math.ceil(Math.sqrt(ids.length*w/h))),rows=Math.ceil(ids.length/cols);
   for(let k=0;k<ids.length;k++){let id=ids[k];this.position.set([x+(k%cols+.5)*w/cols,y+(Math.floor(k/cols)+.5)*h/rows,0,0],id*4);this.color.set([...palette[role],1],id*4);this.names[id]=name;}
   this.regions.push({x,y:y-18,w,title:title||name,count:ids.length,role});};
  for(const [i,p] of g.meta.populations.entries())place(p.indices,areas[p.name]||[960,65+i*45,100,35,'association',p.name],p.name);
  const bankGroups=new Map();
  this.addresses.forEach((a,i)=>{let key=`${a[0]} ${a[1]?'mirrored':'played'}`;if(!bankGroups.has(key))bankGroups.set(key,[]);bankGroups.get(key).push(i);});
  let b=0;for(const [name,addresses] of bankGroups){const ids=[];for(const i of addresses)for(let k=0;k<29;k++)ids.push(this.n+i*29+k);place(ids,[20+b*1080/bankGroups.size,655,1050/bankGroups.size,95,'memory',`Route store · ${name}`],`route ${name}`);b++;}
  place([this.critic,this.critic+1],[995,135,80,100,'reward','Value / dopamine'],'value / dopamine');
  this.gl=canvas.getContext('webgl2',{antialias:true,alpha:false,powerPreference:'high-performance'});
  if(!this.gl)throw Error('This whole-circuit view needs WebGL 2. Try a browser with hardware acceleration.');
  const gl=this.gl;
  this.nodeProgram=program(gl,`#version 300 es\n${common}
out vec4 tint; void main(){int id=gl_VertexID;vec4 s=item(states,id);float x=mode==0?s.y:mode==1?s.x:mode==2?s.z:item(auxiliary,id).x;float a=intensity(x);vec3 c=item(colors,id).rgb; if(mode==0||mode==2)c=mix(c,x<0.?vec3(1.,.38,.3):vec3(.5,1.,.87),a*.7);gl_Position=vec4(loc(id)*vec2(1.,-1.),0.,1.);gl_PointSize=clamp(camera.z*1.4,1.,5.)+a*2.;tint=vec4(c*(.26+.74*a),exists(id));}`,
   `#version 300 es\nprecision highp float;in vec4 tint;out vec4 outColor;void main(){if(tint.a==0.)discard;outColor=tint;}`);
  this.edgeProgram=program(gl,`#version 300 es\n${common}
layout(location=0) in uvec2 ends;layout(location=1) in vec2 weight;out vec4 tint;
void main(){int a=int(ends.x),b=int(ends.y),id=gl_VertexID==0?a:b;vec4 s=item(states,a);float x=mode==0?weight.x*item(auxiliary,a).y:mode==2?weight.y:weight.x*s.x;float level=intensity(x);vec3 c=item(colors,a).rgb;if(mode==2)c=x<0.?vec3(1.,.38,.3):vec3(.5,1.,.87);vec2 pos=overview==1?mix(loc(a),loc(b),float((uint(overview==1?gl_VertexID:gl_InstanceID)*2654435761u)&65535u)/65535.):loc(id);gl_Position=vec4(pos*vec2(1.,-1.),0.,1.);gl_PointSize=1.+level; tint=vec4(c,(overview==1?(.015+level*.45):(.0015+level*.09))*exists(a)*exists(b));}`,
   `#version 300 es\nprecision highp float;in vec4 tint;out vec4 outColor;void main(){outColor=tint;}`);
  const ends=new Uint32Array(this.totalEdges*2);for(let i=0;i<this.e;i++){ends[2*i]=g.arrays.pre[i];ends[2*i+1]=g.arrays.post[i];}
  let at=this.e;const recall=g.meta.populations.find(p=>p.name==='recall')?.indices||[];
  for(let i=0;i<this.r;i++)for(let j=0;j<28;j++){const node=this.n+i*29;ends.set([node,node+j+1],2*at++);ends.set([node+j+1,recall[j]??node+j+1],2*at++);}
  for(const id of g.meta.critic_index)ends.set([id,this.critic],2*at++);
  this.ends=ends;this.edgeVAO=gl.createVertexArray();gl.bindVertexArray(this.edgeVAO);
  const eb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,eb);gl.bufferData(gl.ARRAY_BUFFER,ends,gl.STATIC_DRAW);gl.enableVertexAttribArray(0);gl.vertexAttribIPointer(0,2,gl.UNSIGNED_INT,0,0);gl.vertexAttribDivisor(0,1);
  this.wb=gl.createBuffer();gl.bindBuffer(gl.ARRAY_BUFFER,this.wb);gl.bufferData(gl.ARRAY_BUFFER,this.weight,gl.DYNAMIC_DRAW);gl.enableVertexAttribArray(1);gl.vertexAttribPointer(1,2,gl.FLOAT,false,0,0);gl.vertexAttribDivisor(1,1);gl.bindVertexArray(null);
  this.textures=[this.texture(this.position),this.texture(this.color),this.texture(this.state),this.texture(this.aux)];
  this.camera=[0,0,1,1];this.mode='repair';this.scale=1;this.live=this.n;this.last=null;
  for(const r of this.regions){const el=document.createElement('span');el.className='region';el.textContent=`${r.title} · ${r.count.toLocaleString()}`;labels.append(el);r.el=el;el.title=`${r.title} · ${r.count.toLocaleString()}`;}
  new ResizeObserver(()=>this.fit()).observe(canvas.parentElement);
  canvas.addEventListener('wheel',e=>{e.preventDefault();const r=canvas.getBoundingClientRect(),x=e.clientX-r.left,y=e.clientY-r.top,f=Math.exp(-e.deltaY*.001);this.camera[0]=x-(x-this.camera[0])*f;this.camera[1]=y-(y-this.camera[1])*f;this.camera[2]*=f;this.camera[3]*=f;this.draw();},{passive:false});
  let drag=null;canvas.addEventListener('pointerdown',e=>{drag=[e.clientX,e.clientY];canvas.setPointerCapture(e.pointerId);});canvas.addEventListener('pointerup',()=>drag=null);canvas.addEventListener('pointercancel',()=>drag=null);
  canvas.addEventListener('pointermove',e=>{if(drag){this.camera[0]+=e.clientX-drag[0];this.camera[1]+=e.clientY-drag[1];drag=[e.clientX,e.clientY];this.draw();return;}this.inspect(e);});
 }
 texture(array){const gl=this.gl,t=gl.createTexture();gl.bindTexture(gl.TEXTURE_2D,t);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MIN_FILTER,gl.NEAREST);gl.texParameteri(gl.TEXTURE_2D,gl.TEXTURE_MAG_FILTER,gl.NEAREST);gl.texImage2D(gl.TEXTURE_2D,0,gl.RGBA32F,1024,array.length/4096,0,gl.RGBA,gl.FLOAT,array);return t;}
 fit(){const r=this.canvas.getBoundingClientRect();if(!r.width||!r.height)return;const sy=Math.min((r.width-24)/1120,(r.height-60)/770),sx=Math.min((r.width-24)/1120,sy*1.8);this.camera=[(r.width-1120*sx)/2,44,sx,sy];this.draw();}
 update({v,s,previous,previousActivation,adaptation,weights,oldWeights,bias,oldBias,memory,critic,oldCritic,criticBias=0,dopamine=0}){
  this.previousActivation=previousActivation;this.state.fill(0);this.live=this.n+29*(memory.arrays.memory.length/28);let maxRepair=0,changed=0;
  for(let i=0;i<this.n;i++){const d=previous?v[i]-previous[i]:0;this.state.set([s[i],d,oldBias?bias[i]-oldBias[i]:0,adaptation?.[i]||0],i*4);this.aux[i*4]=adaptation?.[i]||0;this.aux[i*4+1]=this.previousActivation?s[i]-this.previousActivation[i]:0;maxRepair=Math.max(maxRepair,Math.abs(d));}
  for(let i=0;i<this.e;i++){const w=weights[i],d=oldWeights?w-oldWeights[i]:0;this.weight[2*i]=w;this.weight[2*i+1]=d;if(d!==0){changed++;const a=this.g.arrays.pre[i],b=this.g.arrays.post[i];if(Math.abs(d)>Math.abs(this.state[a*4+2]))this.state[a*4+2]=d;if(Math.abs(d)>Math.abs(this.state[b*4+2]))this.state[b*4+2]=d;}}
  const m=memory.arrays,read=memory.meta.read;let edge=this.e;
  for(let i=0;i<this.r;i++){const cue=this.n+i*29;this.state[cue*4]=i===read?1:0;for(let j=0;j<28;j++){
    const k=i*28+j,w=m.memory[k]||0,delta=(m.consolidated[k]||0)-(m.before_consolidated[k]||0),id=cue+j+1;
    this.state[id*4]=i===read?m.recall[j]:0;this.state[id*4+2]=delta;
    this.weight[2*edge]=w;this.weight[2*edge+1]=delta;edge++;this.weight[2*edge]=1;this.weight[2*edge+1]=0;edge++;
  }}
  let value=criticBias;for(let i=0;i<this.g.meta.critic_index.length;i++){const w=critic?.[i]||0;this.weight[2*edge]=w;this.weight[2*edge+1]=oldCritic?w-oldCritic[i]:0;edge++;value+=w*s[this.g.meta.critic_index[i]];}
  this.state[this.critic*4]=value;this.state[(this.critic+1)*4]=dopamine;
  this.last={v,s,previous,weights,bias,memory,dopamine};this.maxRepair=maxRepair;this.changed=changed;
  const gl=this.gl;gl.bindTexture(gl.TEXTURE_2D,this.textures[2]);gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,1024,this.state.length/4096,gl.RGBA,gl.FLOAT,this.state);
  gl.bindTexture(gl.TEXTURE_2D,this.textures[3]);gl.texSubImage2D(gl.TEXTURE_2D,0,0,0,1024,this.aux.length/4096,gl.RGBA,gl.FLOAT,this.aux);
  gl.bindBuffer(gl.ARRAY_BUFFER,this.wb);gl.bufferSubData(gl.ARRAY_BUFFER,0,this.weight);this.draw();
 }
 draw(){const gl=this.gl,r=this.canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio,2),w=Math.round(r.width*dpr),h=Math.round(r.height*dpr);if(!w||!h)return;if(this.canvas.width!==w||this.canvas.height!==h){this.canvas.width=w;this.canvas.height=h;}gl.viewport(0,0,w,h);gl.clearColor(.035,.06,.09,1);gl.clear(gl.COLOR_BUFFER_BIT);gl.enable(gl.BLEND);gl.blendFunc(gl.SRC_ALPHA,gl.ONE_MINUS_SRC_ALPHA);
  const modes=['repair','activity','plasticity','adaptation'],mode=modes.indexOf(this.mode);let max=0;for(let i=0;i<this.total;i++)max=Math.max(max,Math.abs(this.state[i*4+[1,0,2,3][mode]]));this.scale=max>0?20/max:1;
  const setup=p=>{gl.useProgram(p);for(let i=0;i<4;i++){gl.activeTexture(gl.TEXTURE0+i);gl.bindTexture(gl.TEXTURE_2D,this.textures[i]);gl.uniform1i(gl.getUniformLocation(p,['positions','colors','states','auxiliary'][i]),i);}gl.uniform2f(gl.getUniformLocation(p,'viewport'),r.width,r.height);gl.uniform4fv(gl.getUniformLocation(p,'camera'),this.camera);gl.uniform1f(gl.getUniformLocation(p,'gain'),this.scale);gl.uniform1i(gl.getUniformLocation(p,'mode'),mode);gl.uniform1i(gl.getUniformLocation(p,'core'),this.n);gl.uniform1i(gl.getUniformLocation(p,'live'),this.live);gl.uniform1i(gl.getUniformLocation(p,'critic'),this.critic);gl.uniform1i(gl.getUniformLocation(p,'overview'),Math.min(this.camera[2],this.camera[3])<1.3?1:0);};
  setup(this.edgeProgram);gl.bindVertexArray(this.edgeVAO);if(Math.min(this.camera[2],this.camera[3])<1.3){gl.vertexAttribDivisor(0,0);gl.vertexAttribDivisor(1,0);gl.drawArrays(gl.POINTS,0,this.totalEdges);}else{gl.vertexAttribDivisor(0,1);gl.vertexAttribDivisor(1,1);gl.drawArraysInstanced(gl.LINES,0,2,this.totalEdges);}gl.bindVertexArray(null);setup(this.nodeProgram);gl.drawArrays(gl.POINTS,0,this.total);
  for(const a of this.regions){const short={'Foveal vision':'Near vision','Peripheral vision':'Far vision','Reverberating context':'Context','Motor decisions':'Motor','Route recall':'Recall','Value / dopamine':'Value'};a.el.textContent=r.width<500?(short[a.title]||(a.title.startsWith('Route store')?'Route memory':a.title)):`${a.title} · ${a.count.toLocaleString()}`;a.el.style.left=`${a.x*this.camera[2]+this.camera[0]}px`;a.el.style.top=`${a.y*this.camera[3]+this.camera[1]}px`;}
 }
 inspect(e){const r=this.canvas.getBoundingClientRect(),x=(e.clientX-r.left-this.camera[0])/this.camera[2],y=(e.clientY-r.top-this.camera[1])/this.camera[3];let best=-1,dist=100/this.camera[2]**2;for(let i=0;i<this.total;i++){if(i>=this.live&&i<this.critic)continue;const d=(this.position[i*4]-x)**2+(this.position[i*4+1]-y)**2;if(d<dist){dist=d;best=i;}}if(best<0||!this.last)return;const id=best,l=this.last,fmt=x=>Number(x||0).toExponential(5);let text=`${this.names[id]} · ${id<this.n?'neuron':'port'} ${id}\n`;
  if(id<this.n){let n=0,net=0;for(let i=0;i<this.e;i++)if(this.g.arrays.post[i]===id){n++;net+=l.weights[i]*l.s[this.g.arrays.pre[i]];}text+=`v ${fmt(l.v[id])}  activity ${fmt(l.s[id])}  bias ${fmt(l.bias?.[id])}  repair ${fmt(this.state[id*4+1])}\n${n} incoming synapses · derived input ${fmt(net)}`;}
  else if(id<this.critic){const a=Math.floor((id-this.n)/29),k=(id-this.n)%29;const address=this.addresses[a];text+=`address ${address[2]} · ${k?'value '+(k-1):'cue'} · ${a===l.memory.meta.read?'reading':'inactive'}`;if(k)text+=`\nweight ${fmt(l.memory.arrays.memory[a*28+k-1])} · consolidated ${fmt(l.memory.arrays.consolidated[a*28+k-1])}`;}
  else text+=`value ${fmt(this.state[id*4])} · linear readout / global modulator`;
  document.getElementById('inspector').textContent=text;
 }
}
