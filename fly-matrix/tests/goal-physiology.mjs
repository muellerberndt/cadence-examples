// Actual body/food-contact regressions with synthetic qualified target replies.
// These test supplied physiology and action custody, not learned natural choice.
import assert from 'node:assert/strict';
import { GoalLife } from '../web/goal-life.js';
import { Flight, DT } from '../web/body.js';
import { POLICY_OUTPUTS, targetProbabilities } from '../web/neural-policy.js';
import { APPETITE, HUNGER_RATE } from '../web/life.js';

const room = () => ({ table: { x0: 2.45, x1: 3.55, y0: 1.95, y1: 2.65, z: .75 },
  fruits: { banana: { pos: [2.88, 2.33, .795] }, bread: { pos: [3.13, 2.28, .8068] } }, surfaceZ: () => .75 });
const create = (hunger = .6) => { const fly = new GoalLife(new Flight([2.65,2.33,1], .3), room(), 7, { generation: 2 }); fly.hunger = hunger; return fly; };
const solve = () => ({ converged: true, iterations: 20, residual: 1e-8, tolerance: 1e-6, reason: 'residual_tolerance' });
function admit(fly, fruit = 'banana', id = 1) {
  const request = fly.openDecision(id, 2, { steps: 1024, tolerance: 1e-6, temperature: .3 });
  assert.ok(request?.selectTarget);
  const candidates = ['banana','bread'].map(name => {
    const v = name === fruit ? [.9,.1] : [.1,.9];
    return { fruit: name, score: v[0] - v[1], readouts: Object.fromEntries(POLICY_OUTPUTS.map((n,i) => [n,v[i]])), solve: solve() };
  });
  const w = [.9,.1].map(x => Math.exp(x/.3)), p = w.map(x => x/(w[0]+w[1]));
  assert.ok(fly.applyAssistedDecision({ ...request, type:'assisted_decision', kind:'decision', accepted:true, ...solve(),
    readouts:Object.fromEntries(POLICY_OUTPUTS.map((n,i) => [n,[.9,.1][i]])),
    targetSelection:{fruit,candidates,p:targetProbabilities(candidates,.3),draw:.5,temperature:.3},
    decision:{accepted:true,action:0,choice:0,p,draw:.5,greedy:false,solves:{free:solve(),plus:solve(),minus:solve()}} }));
}
function advance(fly, seconds, check = () => {}) {
  for (let k = 0; k < Math.round(seconds/DT); k++) {
    if (fly.step(DT)) { fly.senses(null,.1); fly.decide(); fly.odourTick(); }
    check(fly);
  }
}
function until(fly, condition, cap = 20) {
  for (let k = 0; k < Math.round(cap/DT) && !condition(fly); k++) advance(fly, DT);
  assert.ok(condition(fly), `condition not met: ${fly.mode}, hunger ${fly.hunger}, t=${fly.clock}`);
}
function feeding(hunger = .6) { const fly = create(hunger); admit(fly); until(fly, f => f.mode === 'feeding'); return fly; }
const close = (a,b,tol=1e-8) => assert.ok(Math.abs(a-b)<tol, `${a} != ${b}`);
let passed=0,failed=0;
function test(name, run) { try { run(); passed++; console.log(`ok ${name}`); } catch(e) { failed++; console.error(`FAIL ${name}\n${e.stack}`); } }

test('hunger rises at the declared rate and satiation prevents a new food decision', () => {
  const fly = create(.15); assert.equal(fly.openDecision(1), null); assert.equal(fly.odourTick(), false);
  advance(fly,2); close(fly.hunger,.15+2*HUNGER_RATE); assert.ok(fly.hunger<APPETITE);
  assert.equal(fly.authority.goal.fruit,null); assert.equal(fly.odourTick(),false);
  advance(fly,2); assert.ok(fly.hunger>APPETITE); assert.ok(fly.odourTick());
});

test('a near-threshold feeding bout cannot create negative hunger or feed past satiety', () => {
  const fly=feeding(.31); let reachedZero=false;
  advance(fly,4.2,f=>{ assert.ok(f.hunger>=0&&f.hunger<=1, `hunger ${f.hunger}`);
    if(f.hunger===0) reachedZero=true; if(f.hunger===0)assert.notEqual(f.mode,'feeding'); });
  assert.ok(reachedZero); assert.notEqual(fly.mode,'feeding');
});

test('removing sugar during feeding stops calories and mouth extension on the next body step', () => {
  const fly=feeding(); advance(fly,1); const before=fly.hunger;
  fly.revokeNeural('sugar moved'); fly.setSugar(null); fly.senses(null,0);
  advance(fly,.2);
  assert.notEqual(fly.mode,'feeding'); assert.equal(fly.pose().feed,0);
  close(fly.hunger,before+.2*HUNGER_RATE);
});

test('feeding requires physical support as well as a sugar label and nearby fruit', () => {
  const fly=feeding(); const before=fly.hunger;
  fly.perch=null; fly.flight.touching=0;
  fly.step(DT);
  assert.notEqual(fly.mode,'feeding'); assert.ok(fly.hunger>=before);
});

test('moving the occupied fruit uses the existing physical takeoff and cannot keep eating', () => {
  const fly=feeding(), from=[...fly.fruits.banana.pos], before=fly.hunger;
  fly.fruits.banana.pos[0]+=.4;
  assert.ok(fly.fruitMoved('banana',from)); advance(fly,.2);
  assert.equal(fly.mode,'flying'); assert.equal(fly.pose().feed,0);
  close(fly.hunger,before+.2*HUNGER_RATE);
});

test('empty food yields no feeding or positive reward while preserving selected-goal custody', () => {
  const fly=create(); fly.setSugar(null); admit(fly,'bread');
  until(fly,f=>f.mode==='landed'); assert.equal(fly.visits.bread,1);
  assert.equal(fly.pendingReward,null); assert.ok(fly.search.neuralChoice.executed);
  until(fly,f=>f.mode==='grooming'); assert.equal(fly.ethogram.feeds,0);
  until(fly,f=>f.mode==='flying'); assert.equal(fly.pendingReward.reward,0);
  assert.ok(fly.pendingReward.neuralCredit);
});

test('repeated selected goals keep distinct credit tokens and rotate only declared grooming primitives', () => {
  const fly=create(.9), events=[];
  const grooms=[];
  for(const [i,fruit] of ['banana','bread','banana'].entries()) {
    fly.pendingReward=null;
    until(fly,f=>f.hunger>APPETITE);
    fly.setSugar(i<2?fruit:null); admit(fly,fruit,i+1);
    const token=fly.search.neuralToken;
    until(fly,f=>f.mode==='landed');
    if(i<2) { assert.equal(fly.pendingReward.searchToken,token); assert.ok(fly.pendingReward.neuralCredit); }
    else assert.equal(fly.pendingReward,null);
    events.push({fruit,token});
    until(fly,f=>f.mode==='grooming'); grooms.push(fly.pose().groom);
    until(fly,f=>f.mode==='flying');
    if(i===2) { assert.equal(fly.pendingReward.searchToken,token); assert.equal(fly.pendingReward.reward,0); }
    fly.pendingReward=null;
    assert.ok(fly.hunger>=0&&fly.hunger<=1);
    assert.ok([...fly.flight.p,...fly.flight.v,...fly.flight.q].every(Number.isFinite));
  }
  assert.equal(new Set(events.map(x=>x.token)).size,3);
  assert.equal(fly.visits.banana,2); assert.equal(fly.visits.bread,1);
  assert.equal(fly.ethogram.feeds,2); assert.equal(fly.ethogram.grooms,3);
  assert.equal(fly.authority.choices.executed,3);
  assert.deepEqual(grooms,['head','legs','wings']);
  const choiceCount=fly.authority.choices.executed;
  fly.rng=()=>{throw Error('cosmetic grooming must not sample a neural or route choice');};
  fly.startGrooming(); assert.equal(fly.pose().groom,'head');
  assert.equal(fly.authority.choices.executed,choiceCount);
});

test('pause revocation spends no hunger or routine time; reset restores initial physiology', () => {
  const fly=feeding(), before={hunger:fly.hunger,time:fly.clock,episode:fly.episode};
  fly.revokeNeural('paused'); fly.senses(null,0);
  assert.deepEqual({hunger:fly.hunger,time:fly.clock,episode:fly.episode},before);
  fly.revokeNeural('resumed'); advance(fly,.2);
  close(fly.hunger,before.hunger-.2*.12);
  const reset=create();
  assert.equal(reset.hunger,.6); assert.equal(reset.mode,'flying'); assert.equal(reset.ethogram.feeds,0);
  assert.equal(reset.search,null); assert.equal(reset.pendingReward,null);
  reset.startGrooming(); assert.equal(reset.pose().groom,'head');
});

console.log(`${passed} goal physiology groups passed; ${failed} failed`);
if(failed)process.exitCode=1;
