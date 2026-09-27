// One-shot supplied launch and checked-intent takeover, with actual body physics.
// Synthetic qualified replies test custody, not spontaneous neural competence.
import assert from 'node:assert/strict';
import { GoalLife, GOAL_STARTUP, GOAL_BODY } from '../web/goal-life.js';
import { Flight, DT, ROOM } from '../web/body.js';
import { POLICY_OUTPUTS, targetProbabilities } from '../web/neural-policy.js';
const room = () => ({ table:{x0:2.45,x1:3.55,y0:1.95,y1:2.65,z:.75},
  fruits:{banana:{pos:[2.88,2.33,.795]},bread:{pos:[3.13,2.28,.8068]}},surfaceZ:()=>.75 });
const create = ({ p=[1.2,1,1.2], heading=.3, options={}, environment=room(), seed=7 }={}) =>
  new GoalLife(new Flight(p,heading),environment,seed,{generation:2,waiting:false,...options});
const solve = () => ({converged:true,iterations:20,residual:1e-8,tolerance:1e-6,reason:'residual_tolerance'});
function reply(request,fruit='bread') {
  const candidates=['banana','bread'].map(name=>{const v=name===fruit?[.9,.1]:[.1,.9];return {fruit:name,score:v[0]-v[1],
    readouts:Object.fromEntries(POLICY_OUTPUTS.map((n,i)=>[n,v[i]])),solve:solve()};});
  const w=[.9,.1].map(x=>Math.exp(x/.3)),p=w.map(x=>x/(w[0]+w[1]));
  return {...request,type:'assisted_decision',kind:'decision',accepted:true,...solve(),
    readouts:Object.fromEntries(POLICY_OUTPUTS.map((n,i)=>[n,[.9,.1][i]])),
    targetSelection:{fruit,candidates,p:targetProbabilities(candidates,.3),draw:.5,temperature:.3},
    decision:{accepted:true,action:0,choice:0,p,draw:.5,greedy:false,solves:{free:solve(),plus:solve(),minus:solve()}}};
}
function advance(fly,seconds,check=()=>{}) { for(let i=0;i<Math.round(seconds/DT);i++){if(fly.step(DT)){fly.decide();fly.odourTick();}check(fly);} }
const state = fly => [...fly.flight.p,...fly.flight.v,...fly.flight.w,...fly.flight.q];
let tests=0;function test(name,run){run();tests++;console.log(`ok ${name}`);}

test('default launch moves immediately along initial heading while all neural execution counters remain zero',()=>{
  const fly=create(),origin=[...fly.flight.p];
  advance(fly,.5);
  const dx=fly.flight.p[0]-origin[0],dy=fly.flight.p[1]-origin[1];
  assert.ok(Math.hypot(dx,dy)>.07); assert.ok(Math.abs(Math.atan2(dy,dx)-.3)<.02);
  assert.equal(fly.authority.startup.active,true);assert.equal(fly.authority.startup.framesApplied,1000);
  assert.equal(fly.authority.goal.framesApplied,0);assert.equal(fly.authority.choices.executed,0);
  assert.equal(fly.authority.navigation.active,false);assert.equal(fly.pendingReward,null);
  fly.outcome(1,'synthetic contact during supplied launch','bread');assert.equal(fly.pendingReward.neuralCredit,false);
});

test('same body pose gives identical launch despite fruit coordinates, strongest smell or random seed',()=>{
  const different=room();different.fruits.banana.pos=[.5,2,.2];different.fruits.bread.pos=[3,.5,.3];
  const a=create(),b=create({environment:different,seed:987});
  a.smelled='banana';b.smelled='bread';
  a.rng=b.rng=()=>{throw Error('launch must not draw a route');};
  advance(a,.5);advance(b,.5);assert.deepEqual(state(a),state(b));
  assert.deepEqual(a.authority.startup.target,b.authority.startup.target);
});

test('endpoint/time bounds stop the one-shot launch and later waiting never restarts it',()=>{
  const fly=create(),origin=[...fly.flight.p];advance(fly,8);
  const startup=fly.authority.startup,frames=startup.framesApplied;
  assert.equal(startup.phase,'finished');assert.equal(startup.active,false);
  assert.ok(startup.endedAt<=GOAL_STARTUP.maxSeconds);
  assert.ok(Math.hypot(fly.flight.p[0]-origin[0],fly.flight.p[1]-origin[1])<=GOAL_STARTUP.distance+.02);
  assert.ok(fly.flight.speed()<.005);advance(fly,2);
  assert.equal(startup.framesApplied,frames);assert.equal(fly.authority.goal.framesApplied,0);
});

test('outward wall and corner starts shorten the ray instead of introducing another direction',()=>{
  for(const [p,heading] of [[[.2,1.5,1.2],Math.PI],[[3.8,1.5,1.2],0],[[2,.2,1.2],-Math.PI/2],
    [[2,2.8,1.2],Math.PI/2],[[3.8,2.8,1.2],Math.PI/4]]) {
    const fly=create({p,heading}),target=fly.authority.startup.target;
    assert.ok(target[0]>=GOAL_STARTUP.clearance-1e-12&&target[0]<=ROOM[0]-GOAL_STARTUP.clearance+1e-12);
    assert.ok(target[1]>=GOAL_STARTUP.clearance-1e-12&&target[1]<=ROOM[1]-GOAL_STARTUP.clearance+1e-12);
    advance(fly,3);assert.equal(fly.flight.touching,0);assert.ok(fly.flight.speed()<.01);
  }
  const blocked=create({p:[.1,1.5,1.2],heading:Math.PI});
  assert.equal(blocked.authority.startup.phase,'blocked');advance(blocked,.5);
  assert.equal(blocked.authority.startup.framesApplied,0);
});

test('pause, sugar change, disabled brain and startup opt-out cannot restart the launch',()=>{
  for(const reason of ['paused','sugar moved']) {
    const fly=create();advance(fly,.3);fly.revokeNeural(reason);const frames=fly.authority.startup.framesApplied;
    assert.equal(fly.authority.startup.active,false);assert.deepEqual(fly.pilot.target,fly.flight.p);
    fly.setSugar(null);fly.revokeNeural('resumed');advance(fly,2);
    assert.equal(fly.authority.startup.framesApplied,frames);assert.ok(fly.flight.speed()<.005);
  }
  for(const options of [{startup:false},{neuralEnabled:false}]) {
    const fly=create({options}),p=[...fly.flight.p];advance(fly,1);
    assert.equal(fly.authority.startup.framesApplied,0);
    assert.ok(Math.hypot(fly.flight.p[0]-p[0],fly.flight.p[1]-p[1])<1e-6);
  }
});

test('qualified target takes over through real forces with no teleport or credit for the launch',()=>{
  const fly=create();advance(fly,.5);
  const request=fly.openDecision(1),before=state(fly),frames=fly.authority.startup.framesApplied;
  assert.ok(fly.applyAssistedDecision(reply(request)));
  assert.deepEqual(state(fly),before);assert.equal(fly.authority.startup.phase,'handoff');
  assert.equal(fly.authority.choices.executed,0);
  fly.step(DT);assert.equal(fly.authority.choices.executed,1);
  assert.ok(Math.hypot(...fly.flight.p.map((x,i)=>x-before[i]))<.001);
  assert.ok(Math.hypot(...fly.flight.v.map((x,i)=>x-before[i+3]))<.05);
  advance(fly,.5);assert.equal(fly.authority.startup.framesApplied,frames);
  assert.equal(fly.authority.goal.fruit,'bread');assert.ok(fly.authority.goal.framesApplied>0);
});

test('matching rejected candidate or actor phase clears dead search and allows a bounded retry',()=>{
  for(const mutate of [m=>{m.targetSelection.candidates[0].solve.converged=false;},
    m=>{m.accepted=false;m.decision.solves.plus.converged=false;}]) {
    const fly=create(),request=fly.openDecision(1),message=reply(request);mutate(message);
    assert.equal(fly.applyAssistedDecision(message),false);
    assert.equal(fly.search,null);assert.equal(fly._pendingDecision,null);assert.equal(fly.odourTick(),false);
    assert.equal(fly.authority.startup.active,false);
    advance(fly,GOAL_BODY.retrySeconds+.01);assert.ok(fly.odourTick());
    const next=fly.openDecision(2);assert.ok(next);assert.ok(fly.applyAssistedDecision(reply(next)));
    assert.equal(fly.authority.startup.framesApplied,0);assert.equal(fly.authority.choices.executed,0);
  }
});

test('unrelated stale failed replies cannot revoke an in-flight current decision or launch',()=>{
  for(const mutate of [m=>{m.requestId++;},m=>{m.generation++;},m=>{m.searchToken+='old';}]) {
    const fly=create(),request=fly.openDecision(1),message=reply(request);mutate(message);message.accepted=false;
    const pending=fly._pendingDecision,search=fly.search;
    assert.equal(fly.applyAssistedDecision(message),false);
    assert.equal(fly._pendingDecision,pending);assert.equal(fly.search,search);
    assert.equal(fly.authority.startup.phase,'launching');
    assert.ok(fly.applyAssistedDecision(reply(request)));
  }
});

console.log(`${tests} startup and retry groups passed`);
