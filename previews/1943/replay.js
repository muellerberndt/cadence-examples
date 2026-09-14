import {packet} from './packet.js';
import {Circuit} from './circuit.js';
const $=id=>document.getElementById(id),fmt=(n,d=3)=>Number(n||0).toFixed(d);
let dopamineScale=1;
let manifest,graph,circuit,current=0,iteration=0,playing=false,busy=false,sequence=0,lastTick=0;
const caches={events:new Map(),weights:new Map(),memory:new Map()};
async function cached(kind,path){const cache=caches[kind];if(cache.has(path)){const v=cache.get(path);cache.delete(path);cache.set(path,v);return v;}const data=await packet(path);cache.set(path,data);while(cache.size>3)cache.delete(cache.keys().next().value);return data;}
const eventAt=i=>manifest.events[Math.max(0,Math.min(manifest.events.length-1,i))];
function precedingState(i){for(let j=i;j>=0;j--)if(['settle','action'].includes(eventAt(j).kind))return j;return 0;}
function controls(){for(const id of ['play','back','step'])$(id).disabled=!manifest||(id!=='play'&&busy); $('play').textContent=playing?'Pause':'Play';}
function drawWaves(data,step){const canvas=$('waves'),r=canvas.getBoundingClientRect(),dpr=Math.min(devicePixelRatio,2);canvas.width=r.width*dpr;canvas.height=r.height*dpr;const ctx=canvas.getContext('2d');ctx.scale(dpr,dpr);const w=r.width,h=r.height;ctx.fillStyle='#101923';ctx.fillRect(0,0,w,h);
 const shape=data.meta.arrays.potential?.shape;if(!shape||shape.length!==2){ctx.fillStyle='#91a4b6';ctx.font='11px system-ui';ctx.fillText('Motor commitment / parameter update · choose a settling event to inspect its waveform.',14,40);return;}
 const steps=shape[0],n=shape[1],groups=graph.meta.populations.filter(p=>['v1_fovea','v1_periphery','association','motor','context','afterimage'].includes(p.name));
 const colors=['#ab99ff','#59e5cb','#ffbf70','#ed81b6','#58b7ff','#8fe19d'];let top=1e-12,lines=[];
 for(const group of groups){let values=[];for(let t=0;t<steps;t++){let sum=0;for(const id of group.indices)sum+=data.arrays.potential[t*n+id];const value=sum/group.indices.length;values.push(value);top=Math.max(top,Math.abs(value));}lines.push(values);}
 ctx.strokeStyle='#263441';ctx.beginPath();ctx.moveTo(12,h*.63);ctx.lineTo(w-12,h*.63);ctx.stroke();
 for(let k=0;k<lines.length;k++){ctx.strokeStyle=colors[k%colors.length];ctx.lineWidth=1.4;ctx.beginPath();lines[k].forEach((y,t)=>{const x=12+t/Math.max(steps-1,1)*(w-24),v=h*.63-y/top*h*.4;t?ctx.lineTo(x,v):ctx.moveTo(x,v);});ctx.stroke();ctx.fillStyle=ctx.strokeStyle;ctx.font='9px system-ui';ctx.fillText(groups[k].name,14+(k%3)*w/3,h-18+Math.floor(k/3)*11);}
 ctx.strokeStyle='#eaf2f8';ctx.beginPath();const x=12+step/Math.max(steps-1,1)*(w-24);ctx.moveTo(x,5);ctx.lineTo(x,h-28);ctx.stroke();
}
async function show(index,step=0){
 const ticket=++sequence;busy=true;controls();current=Math.max(0,Math.min(manifest.events.length-1,index));const event=eventAt(current),frame=manifest.frames[event.frame];iteration=event.kind==='settle'?Math.max(0,Math.min(event.steps,step)):0;
 $('status').textContent='Loading aligned game and neural state…';
 try{
  const si=precedingState(current),source=eventAt(si);
  const [data,memory,parameters]=await Promise.all([cached('events',source.packet),cached('memory',frame.memory),cached('weights',event.weights||eventAt(frame.events.findLast(id=>id<=current&&eventAt(id).weights)??si).weights)]);
  let learningEvent=event.kind==='plasticity'?event:null;
  const gameSummary=$('clock').value==='game'&&['action','end'].includes(event.kind);
  if(gameSummary){const id=frame.events.findLast(id=>eventAt(id).kind==='plasticity');if(id!==undefined)learningEvent=eventAt(id);}
  let eventData=data,old=null;if(learningEvent){[eventData,old]=await Promise.all([cached('events',learningEvent.packet),cached('weights',learningEvent.previous_weights)]);}
  let waveData=data,waveStep=null;
  if(event.kind==='action'){const id=frame.events.findLast(id=>eventAt(id).kind==='settle'&&eventAt(id).beta===null);if(id!==undefined){waveData=await cached('events',eventAt(id).packet);waveStep=eventAt(id).steps;}}
  const image=new Image();image.src='./recording/'+frame.image;await image.decode();
  if(ticket!==sequence)return;
  const n=graph.meta.neurons,rows=source.kind==='settle',at=si===current?iteration:(rows?source.steps:0);
  const vector=name=>rows?data.arrays[name].subarray(at*n,(at+1)*n):data.arrays[name];
  const previous=rows&&at>0?data.arrays.potential.subarray((at-1)*n,at*n):null;
  const previousActivation=rows&&at>0?data.arrays.activation.subarray((at-1)*n,at*n):null;
  const plastic=learningEvent!==null,dopamine=plastic?learningEvent.learning.dopamine:0;
  const criticBias=learningEvent?.critic_bias??event.critic_bias??source.critic_bias;
  circuit.update({v:vector('potential'),s:vector('activation'),previous:si===current?previous:null,
   previousActivation:si===current?previousActivation:null,adaptation:vector('adaptation'),
   weights:parameters.arrays.weights,oldWeights:old?.arrays.weights,bias:parameters.arrays.bias,oldBias:old?.arrays.bias,memory,
   critic:eventData.arrays.critic,oldCritic:plastic?waveData.arrays.critic:null,criticBias,dopamine});
  $('screen').src=image.src;$('decision').textContent=frame.decision;$('points').textContent=frame.points.toLocaleString();$('energy').textContent=frame.energy??'—';
  $('game-clock').textContent=`${Math.floor(frame.game_seconds/60).toString().padStart(2,'0')}:${fmt(frame.game_seconds%60,2).padStart(5,'0')}`;
  $('phase').textContent=event.label;$('step-count').textContent=event.kind==='settle'?`Repair ${iteration} / ${event.steps} · event ${current+1} / ${manifest.events.length}`:`${event.kind} · event ${current+1} / ${manifest.events.length}`;
  $('timeline').value=event.frame;$('duration').textContent=`${event.frame} / ${manifest.frames.length-1}`;
  $('intent').textContent=event.kind==='action'?['↑ · ↓'.split(' ')[event.intent[0]],['←','·','→'][event.intent[1]],['rest','fire',manifest.game==='1942'?'loop':'special'][event.intent[2]]].join('  '):event.kind==='end'?'Episode recording ended':'Thinking · no new motor command';
  $('dopamine').textContent=plastic?fmt(dopamine,5):'—';$('learning-state').textContent=plastic?(gameSummary?'This decision’s broadcast':'Broadcast now'):'Between updates';
  $('dopamine-bar').style.left=`${50+Math.min(0,dopamine)/dopamineScale*50}%`;$('dopamine-bar').style.width=`${Math.min(50,Math.abs(dopamine)/dopamineScale*50)}%`;$('dopamine-bar').style.background=dopamine<0?'#ff7666':'#59e5cb';
  $('changed').textContent=plastic?circuit.changed.toLocaleString():'—';$('repair').textContent=circuit.maxRepair.toExponential(3);
  drawWaves(waveData,waveStep??at);$('wave-note').textContent=`${event.kind==='action'?'Last free settlement for this decision.':'Step 0 is the actual initial state.'} Colors distinguish populations; traces are simulated potential, not EEG.`;
  $('status').textContent=`${manifest.mode} · ${event.kind==='settle'?'all iterations retained; brightness normalized within this step':'exact recorded event'}${manifest.complete?'':' · partial recording'}`;
  window.replayState={ready:true,event:current,step:iteration,frame:event.frame,neurons:circuit.n,synapses:circuit.e,totalVertices:circuit.total,totalEdges:circuit.totalEdges,gameImage:frame.image,kind:event.kind,criticBias,waveformSteps:waveData.meta.arrays.potential.shape.length===2?waveData.meta.arrays.potential.shape[0]-1:0};
 }catch(error){if(ticket===sequence){playing=false;$('status').textContent=error.message;console.error(error);}}finally{if(ticket===sequence){busy=false;controls();}}
}
async function advance(direction=1){
 const e=eventAt(current);if($('clock').value==='game'){
  const frame=Math.max(0,Math.min(manifest.frames.length-1,e.frame+direction));if(frame===e.frame){playing=false;controls();return;}await seekFrame(frame,true);return;}
 if(e.kind==='settle'&&((direction>0&&iteration<e.steps)||(direction<0&&iteration>0))){await show(current,iteration+direction);return;}
 const i=current+direction;if(i<0||i>=manifest.events.length){playing=false;controls();return;}await show(i,direction<0?eventAt(i).steps||0:0);
}
async function seekFrame(i,game=false){const events=manifest.frames[i].events;const id=game?(events.findLast(id=>eventAt(id).kind==='action')??events.at(-1)):events[0];await show(id);}
async function tick(time){if(playing&&!busy&&time-lastTick>=1000/(($('clock').value==='game'?15:30)*Number($('speed').value))){lastTick=time;await advance();}else if(circuit)circuit.animate(time);requestAnimationFrame(tick);}
$('play').onclick=()=>{playing=!playing;controls();};$('step').onclick=()=>{playing=false;controls();advance();};$('back').onclick=()=>{playing=false;controls();advance(-1);};
let seekTimer;$('timeline').oninput=()=>{playing=false;controls();clearTimeout(seekTimer);const target=Number($('timeline').value);seekTimer=setTimeout(()=>seekFrame(target,$('clock').value==='game'),100);};
$('clock').onchange=()=>{if($('clock').value==='game'&&$('mode').value==='repair'){$('mode').value='activity';circuit.mode='activity';circuit.draw();}};
$('mode').onchange=()=>{circuit.mode=$('mode').value;circuit.draw();};$('fit').onclick=()=>circuit.fit();
document.addEventListener('keydown',e=>{if(['INPUT','SELECT','BUTTON'].includes(e.target.tagName))return;if(e.code==='Space'){e.preventDefault();$('play').click();}if(e.code==='ArrowRight')$('step').click();if(e.code==='ArrowLeft')$('back').click();});
async function start(){const r=await fetch('./recording/manifest.json');if(!r.ok)throw Error('Recording manifest is unavailable');manifest=await r.json();if(!manifest.events.length)throw Error('This recording has no completed events');graph=await packet('graph.bin.gz');circuit=new Circuit($('map'),$('labels'),graph);window.__flightCircuit=circuit;
 $('counts').textContent=`${circuit.n.toLocaleString()} neurons · ${circuit.e.toLocaleString()} synapses\n+ ${circuit.r.toLocaleString()} memory addresses + value / dopamine`;
 $('game-title').textContent=`${manifest.game} · trained pilot`;$('timeline').max=manifest.frames.length-1;
 $('provenance').textContent=`Cadence ${manifest.cadence}. Checkpoint SHA-256: ${manifest.checkpoint_sha256||'test fixture'}. Seed ${manifest.seed}. ${manifest.stop_reason}. ${manifest.parameter_versions} parameter versions. ${fmt(manifest.bytes/2**20,1)} MiB recorded.`;
 dopamineScale=Math.max(1e-12,...manifest.events.filter(e=>e.kind==='plasticity').map(e=>Math.abs(e.learning.dopamine)));$('dopamine-scale').textContent=`Bar scale ±${dopamineScale.toPrecision(3)} · largest recorded signal`;
 controls();await show(0);requestAnimationFrame(tick);
}
start().catch(error=>{$('status').textContent=error.message;console.error(error);});
