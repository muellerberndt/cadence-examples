// Explicit body-support and impulse mechanics. Synthetic qualified decisions
// test handover/custody; none of the waiting motion is attributed to the brain.
import assert from 'node:assert/strict';
import { GoalLife, GOAL_WAITING, GOAL_DECISION_TTL, GOAL_OBSERVATION_TTL } from '../web/goal-life.js';
import { Flight, DT, ROOM, RADIUS, CONTROL_KEYS } from '../web/body.js';
import { POLICY_OUTPUTS, targetProbabilities } from '../web/neural-policy.js';
import { SETTLED_MOTOR_GROUPS } from '../web/motor.js';
const room=()=>({table:{x0:2.45,x1:3.55,y0:1.95,y1:2.65,z:.75},
  fruits:{banana:{pos:[2.88,2.33,.795]},bread:{pos:[3.13,2.28,.8068]}},surfaceZ:()=>.75});
const create=({p=[1.2,1,1.2],heading=.3,environment=room(),options={}}={})=>
  new GoalLife(new Flight(p,heading),environment,7,{generation:2,...options});
const solve=()=>({converged:true,iterations:20,residual:1e-8,tolerance:1e-6,reason:'residual_tolerance'});
function reply(r,fruit='bread'){
 const candidates=['banana','bread'].map(f=>{const v=f===fruit?[.9,.1]:[.1,.9];return {fruit:f,score:v[0]-v[1],
  readouts:Object.fromEntries(POLICY_OUTPUTS.map((n,i)=>[n,v[i]])),solve:solve()};});
 const w=[.9,.1].map(v=>Math.exp(v/.3)),p=w.map(v=>v/(w[0]+w[1]));
 return {...r,type:'assisted_decision',kind:'decision',accepted:true,...solve(),
  readouts:Object.fromEntries(POLICY_OUTPUTS.map((n,i)=>[n,[.9,.1][i]])),
  targetSelection:{fruit,candidates,p:targetProbabilities(candidates,.3),draw:.5,temperature:.3},
  decision:{accepted:true,action:0,choice:0,p,draw:.5,greedy:false,solves:{free:solve(),plus:solve(),minus:solve()}}};
}
function advance(fly,seconds,check=()=>{}){for(let k=0;k<Math.round(seconds/DT);k++){if(fly.step(DT)){fly.decide();fly.odourTick();}check(fly);}}
const physical=f=>[...f.p,...f.v,...f.q,...f.w];
let tests=0;const test=(name,run)=>{run();tests++;console.log(`ok ${name}`);};

test('sixty seconds without a brain choice keeps moving through real room-clearance turns without actor credit',()=>{
 const fly=create(),speeds=[];let path=0,previous=[...fly.flight.p];
 fly.rng=()=>{throw Error('waiting must not consume random route draws');};
 advance(fly,60,f=>{path+=Math.hypot(...f.flight.p.map((x,i)=>x-previous[i]));previous=[...f.flight.p];
  if(f.clock>1)speeds.push(Math.hypot(f.flight.v[0],f.flight.v[1]));
  assert.ok(f.flight.p[0]>.03&&f.flight.p[0]<ROOM[0]-.03&&f.flight.p[1]>.03&&f.flight.p[1]<ROOM[1]-.03);});
 const median=speeds.sort((a,b)=>a-b)[Math.floor(speeds.length/2)],slow=speeds.filter(v=>v<.1).length/speeds.length;
 assert.ok(path>15,`path ${path}`);assert.ok(median>.25&&median<.35);assert.ok(slow<.02,`slow share ${slow}`);
 assert.ok(fly.authority.waitingSupport.turns>=3);assert.equal(fly.authority.choices.executed,0);
 assert.equal(fly.authority.goal.framesApplied,0);assert.equal(fly.authority.navigation.framesApplied,0);assert.equal(fly.pendingReward,null);
 console.log(`  body-support path ${path.toFixed(3)}m, median ${median.toFixed(4)}m/s, slow fraction ${slow.toFixed(4)}`);
});

test('wall and corner support is deterministic and moves away without fruit lookup',()=>{
 for(const [p,heading] of [[[.18,1.5,1.2],Math.PI],[[3.82,1.5,1.2],0],[[2,.18,1.2],-Math.PI/2],
  [[2,2.82,1.2],Math.PI/2],[[.18,.18,1.2],-3*Math.PI/4],[[3.82,2.82,1.2],Math.PI/4],
  [[.18,2.82,1.2],3*Math.PI/4],[[3.82,.18,1.2],-Math.PI/4]]){
  const fly=create({p,heading,options:{startup:false}});advance(fly,3);
  assert.ok(fly.flight.p[0]>.2&&fly.flight.p[0]<ROOM[0]-.2);assert.ok(fly.flight.p[1]>.2&&fly.flight.p[1]<ROOM[1]-.2);
  assert.ok(fly.authority.waitingSupport.turns>0);assert.equal(fly.authority.goal.fruit,null);
 }
 const changed=room();changed.fruits.banana.pos=[.3,.3,.2];changed.fruits.bread.pos=[3.5,1,.5];
 const a=create(),b=create({environment:changed});advance(a,8);advance(b,8);assert.deepEqual(physical(a.flight),physical(b.flight));
});

test('hard pause/revoke stops support and explicit resume preserves body continuity without renewing actor credit',()=>{
 const fly=create();advance(fly,8);fly.revokeNeural('paused');const count=fly.authority.waitingSupport.framesApplied,time=fly.clock;
 assert.equal(fly.authority.waitingSupport.active,false);assert.equal(fly.authority.physicalReaction.active,false);
 advance(fly,2);assert.equal(fly.authority.waitingSupport.framesApplied,count);assert.ok(fly.flight.speed()<.005);
 const before=physical(fly.flight);fly.resumeBodySupport();assert.deepEqual(physical(fly.flight),before);advance(fly,1);
 assert.ok(fly.flight.speed()>.2);assert.ok(fly.clock>time);assert.equal(fly.authority.choices.executed,0);
 const control=create({options:{startup:false,waiting:false}}),p=[...control.flight.p];advance(control,2);
 assert.ok(Math.hypot(control.flight.p[0]-p[0],control.flight.p[1]-p[1])<1e-6);
});

test('waiting survives a rejected current choice and hands off only to a qualified selected goal',()=>{
 const fly=create(),r=fly.openDecision(1),bad=reply(r);bad.accepted=false;bad.decision.solves.plus.converged=false;
 assert.equal(fly.applyAssistedDecision(bad),false);advance(fly,2);assert.ok(fly.authority.waitingSupport.active);
 const q=fly.openDecision(2),before=physical(fly.flight),count=fly.authority.waitingSupport.framesApplied;
 assert.ok(fly.applyAssistedDecision(reply(q)));assert.deepEqual(physical(fly.flight),before);
 assert.equal(fly.authority.choices.executed,0);advance(fly,.3);
 assert.equal(fly.authority.waitingSupport.framesApplied,count);assert.equal(fly.authority.goal.fruit,'bread');
 assert.equal(fly.authority.choices.executed,1);
});

test('twenty-second historical goal admission is separate from unchanged one-second motor freshness',()=>{
 const fly=create(),r=fly.openDecision(1);assert.equal(r.deadline,GOAL_DECISION_TTL);
 advance(fly,11);assert.ok(fly.applyAssistedDecision(reply(r)));assert.equal(fly.authority.choices.executed,0);
 assert.equal(fly.authority.limits.ttlSeconds,1);
 const stale=create(),old=stale.openDecision(1);advance(stale,20.1);
 assert.equal(stale.applyAssistedDecision(reply(old)),false);assert.equal(stale.authority.choices.executed,0);
});

test('aged checked observations unblock goal scheduling while their raw motor trim stays expired',()=>{
 const observe=r=>({...r,type:'control',kind:'control',...solve(),readouts:{
  ...Object.fromEntries(SETTLED_MOTOR_GROUPS.map(name=>[name,0])),'mn:wing:b2:left':.8}});
 const fly=create(),r=fly.expectObservation(1);assert.equal(r.deadline,GOAL_OBSERVATION_TTL);assert.equal(r.freshUntil,1);
 advance(fly,5);assert.ok(fly.applyObservation(observe(r)));fly.step(DT);
 assert.equal(fly.lastQualifiedObservation.requestId,1);assert.equal(fly._observation,null);
 assert.ok(Object.values(fly.authority.neuralTrim).every(v=>v===0));
 assert.ok(fly.odourTick());assert.ok(fly.openDecision(2));assert.equal(fly.authority.choices.executed,0);
 const late=create(),q=late.expectObservation(1);advance(late,10.1);
 assert.equal(late.applyObservation(observe(q)),false);assert.equal(late.lastQualifiedObservation,null);
 assert.equal(late._observation,null);assert.equal(late.authority.choices.executed,0);
});

test('a perched nudge releases holds and integrates exactly the actual impulse with no added teleport or kick',()=>{
 const fly=create({p:[2.88,2.33,.795+RADIUS],options:{startup:false}});
 fly.mode='feeding';fly.perch=[2.88,2.33,.795];fly.flight.touching=1;fly.flight.landed=true;fly._perchedAt=0;
 fly.flight.swat([.2,0,.6],[0,0,2],[1,1,10]);const before=physical(fly.flight);
 const expected=new Flight(fly.flight.p,0);for(const k of ['p','v','q','w','wind'])expected[k]=fly.flight[k].slice();
 fly.respondToNudge();assert.deepEqual(physical(fly.flight),before);assert.equal(fly.perch,null);assert.equal(fly.mode,'flying');
 assert.ok(Object.values(fly.authority.executedControls).every(v=>v===0));assert.equal(fly.controls.f,0);
 const off=Object.fromEntries(CONTROL_KEYS.map(k=>[k,0]));
 for(let k=0;k<100;k++){expected.step(DT,off);fly.step(DT);}
 assert.deepEqual(physical(fly.flight),physical(expected));
 assert.ok(Math.hypot(...fly.flight.p.map((x,i)=>x-before[i]))>.01);
 assert.equal(fly.controls.f,0);assert.equal(fly.pose().feed,0);assert.equal(fly.authority.choices.executed,0);
 advance(fly,.3);assert.equal(fly.authority.physicalReaction.active,false);assert.ok(fly.controls.f>0);
 advance(fly,1);assert.ok(fly.flight.p[2]>.3);assert.equal(fly.authority.goal.framesApplied,0);
});

test('airborne nudge does not return to the old hold target; pausing cancels the reaction timer',()=>{
 const fly=create();advance(fly,1);fly.flight.swat([.15,.1,.6],[0,0,2],[0,0,8]);fly.respondToNudge();
 const p=[...fly.flight.p];advance(fly,.5);assert.ok(Math.hypot(...fly.flight.p.map((x,i)=>x-p[i]))>.04);
 assert.ok(fly.authority.waitingSupport.active);assert.equal(fly.authority.choices.executed,0);
 fly.respondToNudge();fly.revokeNeural('paused');assert.equal(fly.authority.physicalReaction.active,false);
 assert.equal(fly.authority.physicalReaction.until,null);assert.equal(fly.authority.waitingSupport.active,false);
 assert.equal(fly.search,null);assert.equal(fly.authority.goal.fruit,null);
});
console.log(`${tests} waiting, latency and physical-nudge groups passed`);
