// Live diagnostic snapshots must preserve exact solver/learner arithmetic.
import assert from 'node:assert/strict';
import { SettlingBrain } from '../web/brain.js';
import { ActorCriticLearner } from '../web/learner.js';
import { SETTLED_MOTOR_GROUPS } from '../web/motor.js';
const b64=a=>Buffer.from(a.buffer,a.byteOffset,a.byteLength).toString('base64');
const fixture=()=>({n:5,edges:4,model:{dt:.2,slope:1,threshold:0,leak:1,gain:1,stimulus_amplitude:1},
 arrays:{row_ptr:b64(new Int32Array([0,0,0,2,4,4])),pre:b64(new Int32Array([0,1,0,1])),weight:b64(new Float64Array([.7,.1,-.3,.4])),sign:b64(new Float64Array([.7,.1,-.3,.4])),count:b64(new Uint16Array([1,1,1,1]))},
 populations:{'haltere:left':[0],'orn:decaying_fruit:left':[0],'orn:decaying_fruit:right':[0],'orn:yeasty:left':[1],'orn:yeasty:right':[1],kc:[0,1],mbon:[2,3],
 'mbon:MBON11:right':[2],'mbon:MBON05:left':[3],...Object.fromEntries(SETTLED_MOTOR_GROUPS.map((n,k)=>[n,[2+k%2]]))}});
const make=()=>{const b=new SettlingBrain(fixture());b.stimulate('haltere:left',1);return b;};
const snapshot=b=>({v:[...b.v],s:[...b.s],steps:b.steps,w:[...b.w],bias:[...b.bias],drive:[...b.drive]});
const copy=p=>({...p,activity:Float32Array.from(p.activity),deltaV:Float32Array.from(p.deltaV)});
let count=0;const test=async(name,f)=>{await f();console.log('ok '+name);count++;};
await test('free progress and replay preserve exact state and report actual one-step changes',()=>{
 for(const replay of [null,true]){
  const plain=make(),live=make(),events=[];live.onSolveProgress=p=>events.push(copy(p));
  const a=plain.settleControl(128,1e-10,null,replay),b=live.settleControl(128,1e-10,null,replay);
  assert.deepEqual(b,a);assert.deepEqual(snapshot(live),snapshot(plain));assert.ok(events.length>3);
  const reference=make();let previous=reference.v.slice(),iteration=0;
  for(const e of events){while(iteration<e.iteration){previous=reference.v.slice();reference.step();iteration++;}
   assert.equal(e.phase,'free');assert.deepEqual(e.activity,Float32Array.from(reference.s));
   assert.deepEqual(e.deltaV,Float32Array.from(reference.v,(v,i)=>iteration?v-previous[i]:0));
   assert.equal(e.residual,reference.equationResidual());}
 }
});
await test('nudged progress uses copied states, exact last update and the actual nudged residual',()=>{
 for(const beta of [.1,-.1]){
  const b=make();b.settleControl(128,1e-10);const before=snapshot(b),events=[];
  const plain=b.settleNudgedResidual([2,3],[1,0],beta,.3,128,1e-10);
  b.onSolveProgress=p=>events.push(copy(p));
  const live=b.settleNudgedResidual([2,3],[1,0],beta,.3,128,1e-10);
  assert.deepEqual(live,plain);assert.deepEqual(snapshot(b),before);
  b.onSolveProgress=null;
  for(const e of events){assert.equal(e.phase,beta>0?'plus':'minus');
   if(e.iteration===0){assert.ok(e.deltaV.every(v=>v===0));continue;}
   const current=b.settleNudgedResidual([2,3],[1,0],beta,.3,e.iteration,1e-30);
   const previous=e.iteration===1?{v:b.v}:b.settleNudgedResidual([2,3],[1,0],beta,.3,e.iteration-1,1e-30);
   assert.deepEqual(e.activity,Float32Array.from(current.s));assert.equal(e.residual,current.residual);
   assert.deepEqual(e.deltaV,Float32Array.from(current.v,(v,i)=>v-previous.v[i]));
  }
 }
});
await test('a broken diagnostic observer cannot abort or alter numerical completion',()=>{
 const a=make(),b=make();b.onSolveProgress=()=>{throw Error('display failed');};
 assert.deepEqual(b.settleControl(128,1e-10),a.settleControl(128,1e-10));assert.deepEqual(snapshot(b),snapshot(a));
 assert.deepEqual(b.settleNudgedResidual([2,3],[1,0],.1,.3,128,1e-10),a.settleNudgedResidual([2,3],[1,0],.1,.3,128,1e-10));
});
const original={self:globalThis.self,fetch:globalThis.fetch,performance:Object.getOwnPropertyDescriptor(globalThis,'performance'),stats:ActorCriticLearner.prototype.stats,control:SettlingBrain.prototype.settleControl};
let learner,brain,clock=0,tick=250;const replies=[],transfers=[];
globalThis.self={postMessage(m,list=[]){if(m.type==='settlement_progress'){
 assert.equal(list.length,2);assert.ok(list[0]!==list[1]);assert.equal(m.activity.buffer,list[0]);assert.equal(m.deltaV.buffer,list[1]);
 assert.ok(!list.includes(brain.v.buffer)&&!list.includes(brain.s.buffer));transfers.push(list);
 }replies.push(structuredClone(m,{transfer:list}));}};
globalThis.fetch=async()=>({json:async()=>fixture()});
Object.defineProperty(globalThis,'performance',{configurable:true,value:{now:()=>{clock+=tick;return clock;}}});
ActorCriticLearner.prototype.stats=function(...a){learner=this;brain=this.brain;return original.stats.apply(this,a);};
SettlingBrain.prototype.settleControl=function(...a){brain=this;return original.control.apply(this,a);};
try{
 await import('../web/worker.js');
 const send=async m=>{const start=replies.length;await self.onmessage({data:m});return replies.slice(start);};
 const config={outputs:['mbon:MBON11:right','mbon:MBON05:left'],actions:[0,1],plastic:'all',critic:'kc',beta:.1,temperature:.3,eta:.1,etaBias:0,etaCritic:.05,gamma:.95,lam:.9,cap:3,dopamineCap:1};
 const senses={'orn:decaying_fruit:left':.8,'orn:decaying_fruit:right':.8,'orn:yeasty:left':.7,'orn:yeasty:right':.7};
 const initialize=async()=>{await send({type:'init',url:'fixture',generation:7});await send({type:'learn:init',config,seed:3});};
 const decision=progress=>({type:'assist:decide',requestId:2,generation:7,searchToken:'7:1',steps:128,tolerance:1e-8,senses,selectTarget:true,progress});
 const parameters=()=>({state:snapshot(brain),trace:[...learner.trace],traceBias:[...learner.traceBias],traceCritic:[...learner.traceCritic],efficacy:[...learner.efficacy]});
 await test('worker progress phases match real masks and exact terminal choice/eligibility',async()=>{
  await initialize();const plain=(await send(decision(false))).at(-1),before=parameters();
  await initialize();const all=await send(decision(true)),final=all.at(-1),events=all.filter(m=>m.type==='settlement_progress');
  assert.deepEqual({...final,ms:0},{...plain,ms:0});assert.deepEqual(parameters(),before);
  for(const phase of ['target:banana','target:bread','free','plus','minus'])assert.ok(events.some(m=>m.phase===phase),phase);
  for(const m of events){assert.equal(m.authoritative,false);assert.equal(m.requestId,2);assert.equal(m.generation,7);assert.equal(m.searchToken,'7:1');assert.equal(m.n,5);
   assert.ok(!Object.hasOwn(m,'accepted')&&!Object.hasOwn(m,'converged')&&!Object.hasOwn(m,'decision')&&!Object.hasOwn(m,'readouts'));
   assert.ok(m.valid&&m.activity instanceof Float32Array&&m.deltaV instanceof Float32Array);assert.equal(m.maxAbsDeltaV,Math.max(...m.deltaV.map(Math.abs)));
   assert.equal(m.attention,m.phase.startsWith('target:')?m.phase.split(':')[1]:final.targetSelection.fruit);
  }
  assert.ok(transfers.every(list=>list.every(b=>b.byteLength===0)));assert.equal(brain.s.byteLength,40);assert.equal(brain.onSolveProgress,null);
 });
 await test('wall throttle spans phases and progress opt-out has no leftover callback',async()=>{
  tick=0;await initialize();const all=await send(decision(true));assert.equal(all.filter(x=>x.type==='settlement_progress').length,1);
  const next=await send({type:'assist:observe',requestId:3,generation:7,steps:128,tolerance:1e-8,senses});assert.ok(next.every(x=>x.type!=='settlement_progress'));
  assert.equal(brain.onSolveProgress,null);tick=250;
 });
 await test('capped and malformed attempts keep progress diagnostic and restore/no-mutate state',async()=>{
  await initialize();const before=snapshot(brain);const all=await send({type:'assist:observe',requestId:4,generation:7,senses,steps:1,tolerance:1e-8,progress:true});
  const final=all.at(-1);assert.equal(final.converged,false);assert.equal(final.stateRestored,true);assert.ok(!final.readouts);
  assert.deepEqual([...brain.v],before.v);assert.deepEqual([...brain.s],before.s);assert.equal(brain.onSolveProgress,null);
  assert.ok(all.slice(0,-1).every(m=>m.type==='settlement_progress'&&m.authoritative===false));
  const invalid=await send({type:'assist:observe',requestId:5,generation:7,senses:{'mbon:MBON11:right':1},progress:true});assert.equal(invalid.length,1);assert.equal(invalid[0].converged,false);assert.equal(brain.onSolveProgress,null);
  const failed=await send({...decision(true),steps:1});assert.equal(failed.at(-1).accepted,false);assert.equal(learner.pending,null);assert.equal(brain.onSolveProgress,null);
 });
}finally{globalThis.self=original.self;globalThis.fetch=original.fetch;Object.defineProperty(globalThis,'performance',original.performance);ActorCriticLearner.prototype.stats=original.stats;SettlingBrain.prototype.settleControl=original.control;}
console.log(`${count} live settlement progress groups passed`);
