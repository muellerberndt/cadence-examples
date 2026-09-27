// Tiny deterministic differential test of the actual worker optimization.
// The reference repeats the selected free solve from the common initial state.
import assert from 'node:assert/strict';
import { SettlingBrain } from '../web/brain.js';
import { ActorCriticLearner } from '../web/learner.js';
import { SETTLED_MOTOR_GROUPS } from '../web/motor.js';
import { attendedSenses } from '../web/neural-policy.js';

const b64 = a => Buffer.from(a.buffer).toString('base64');
const payload = { n:7, edges:8, synapses:8, whole:{neurons:7},
  model:{dt:.5,slope:1,threshold:0,leak:1,gain:1,stimulus_amplitude:1,adaptation:null},
  arrays:{row_ptr:b64(new Int32Array([0,0,0,0,0,2,4,8])),
    pre:b64(new Int32Array([5,6,4,6,0,1,2,3])),
    weight:b64(new Float64Array([.05,.6,.05,-.4,.3,-.2,.9,-.9])),
    sign:b64(new Float64Array([.05,.6,.05,-.4,.3,-.2,.9,-.9])),count:b64(new Uint16Array(8).fill(1))},
  populations:{photoreceptor:[0,1],kc:[6],mbon:[4,5],motor:[4,5],
    'orn:decaying_fruit:left':[2],'orn:decaying_fruit:right':[2],
    'orn:yeasty:left':[3],'orn:yeasty:right':[3],
    'mbon:MBON11:right':[4],'mbon:MBON05:left':[5],
    ...Object.fromEntries(SETTLED_MOTOR_GROUPS.map((name,i)=>[name,[4+i%2]]))}};
const cfg={outputs:['mbon:MBON11:right','mbon:MBON05:left'],actions:[0,1],
  plastic:{pre:['kc'],post:['mbon']},critic:'kc',beta:.1,temperature:.3,
  eta:.1,etaBias:0,etaCritic:.05,gamma:.95,lam:.9,cap:3,dopamineCap:1};
const senses={'orn:decaying_fruit:left':.9,'orn:decaying_fruit:right':.9,
  'orn:yeasty:left':.7,'orn:yeasty:right':.7};
const original={self:globalThis.self,fetch:globalThis.fetch,
  act:ActorCriticLearner.prototype.actSettled,stats:ActorCriticLearner.prototype.stats};
let actual=null,actor=null,id=0;const replies=[];
globalThis.self={postMessage:m=>replies.push(m)};
globalThis.fetch=async()=>({json:async()=>payload});
ActorCriticLearner.prototype.stats=function(...args){actual=this.brain;actor=this;return original.stats.apply(this,args);};
const send=async m=>{const n=replies.length;await self.onmessage({data:m});assert.equal(replies.length,n+1);return replies.at(-1);};
const init=async()=>{await send({type:'init',url:'tiny fixture',generation:3});await send({type:'learn:init',config:cfg,seed:1});};
const propose=()=>send({type:'assist:decide',requestId:++id,generation:3,searchToken:`3:${id}`,
  selectTarget:true,senses,steps:256,tolerance:1e-6});
const input=(b,fruit)=>{b.clearStimuli();for(const[n,v]of Object.entries(attendedSenses(senses,fruit)))b.stimulate(n,v);};
try {
  await import('../web/worker.js');
  await init();
  const result=await propose();assert.ok(result.accepted);
  assert.equal(result.decision.solves.free.iterations,0);
  const reference=new SettlingBrain(payload),learner=new ActorCriticLearner(reference,cfg);
  input(reference,result.targetSelection.fruit);
  const repeated=learner.actSettled({u:result.decision.draw,maxSteps:256,tolerance:1e-6});
  assert.ok(repeated.accepted);assert.ok(repeated.solves.free.iterations>0);
  assert.deepEqual(result.decision.p,repeated.p);assert.equal(result.decision.action,repeated.action);
  assert.deepEqual(actual.v,reference.v);assert.deepEqual(actual.s,reference.s);
  assert.deepEqual(actual.drive,reference.drive);assert.equal(actual.steps,reference.steps);
  for(const name of ['trace','traceBias','traceCritic'])assert.deepEqual(actor[name],learner[name]);
  for(const name of ['plus','minus','plusV','minusV'])assert.deepEqual(actor.pending[name],learner.pending[name]);
  assert.deepEqual(result.decision.solves.plus,repeated.solves.plus);
  assert.deepEqual(result.decision.solves.minus,repeated.solves.minus);
  console.log('ok selected-state reuse exactly matches repeated free solve and both nudged states');

  await init();
  // Begin at nonzero state to distinguish restoration from resetting to zero.
  await send({type:'assist:observe',requestId:++id,generation:3,senses,steps:256,tolerance:1e-6});
  const before={v:actual.v.slice(),s:actual.s.slice(),steps:actual.steps,
    weights:actual.w.slice(),trace:actor.trace.slice()};
  let injected=false;
  ActorCriticLearner.prototype.actSettled=function(options){
    const settle=this.brain.settleNudgedResidual;
    this.brain.settleNudgedResidual=function(...args){
      // Exercise an actual capped plus solve after both candidates qualify.
      args[4]=1;injected=true;return settle.apply(this,args);
    };
    try{return original.act.call(this,options);}finally{this.brain.settleNudgedResidual=settle;}
  };
  const denied=await propose();
  assert.ok(injected);assert.equal(denied.accepted,false);assert.equal(denied.reason,'plus_phase_failed');
  assert.equal(denied.decision.solves.free.iterations,0);
  assert.ok(denied.targetSelection.candidates.every(c=>c.solve.converged));
  assert.deepEqual(actual.v,before.v);assert.deepEqual(actual.s,before.s);assert.equal(actual.steps,before.steps);
  assert.deepEqual(actual.w,before.weights);assert.deepEqual(actor.trace,before.trace);assert.equal(actor.pending,null);
  assert.ok(!denied.readouts&&!denied.s);
  console.log('ok actual capped actor nudge restores pre-probe live state and eligibility');
} finally {
  globalThis.self=original.self;globalThis.fetch=original.fetch;
  ActorCriticLearner.prototype.actSettled=original.act;ActorCriticLearner.prototype.stats=original.stats;
}
