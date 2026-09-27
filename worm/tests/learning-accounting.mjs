// Synthetic learner proposals test final adapter custody without running a life.
import assert from 'node:assert/strict';
import { Life } from '../web/life.js';
const copy=p=>Object.fromEntries(Object.entries(p).map(([k,v])=>[k,v.slice()]));
function fixture(growth, accepted=true) {
  let params=Object.fromEntries(['A','B','C'].map(k=>[k,Float64Array.of(0)]));
  const brain={H:1,updates:7,parameters:()=>copy(params),setParameters:p=>{params=copy(p)},growth,
    observe(){const before=copy(params);if(accepted){params=Object.fromEntries(['A','B','C'].map(k=>[k,Float64Array.of(2)]));this.updates++;}return {updated:accepted,reason:accepted?'updated':'phase_failed',before};}};
  const life=Object.assign(Object.create(Life.prototype),{brain,p:{teach:1,teach_level:.8,beta:.01,rate:1,stability:.97},path:{T:1,output:Float64Array.of(0,0)},stretch:[Float64Array.of(1)],lessons:0,rejected:0,t:0});
  return life;
}
{
 const life=fixture(A=>Math.abs(A[0])),r=life.learn(['food']);
 assert.equal(r.updated,true);assert.equal(r.reason,'updated_after_growth_filter');assert.equal(r.halvings,2);
 assert.deepEqual(Array.from(r.applied),[.5]);assert.equal(life.lessons,1);assert.equal(life.rejected,0);assert.equal(life.brain.updates,8);
 for(const a of Object.values(life.brain.parameters()))assert.deepEqual(Array.from(a),[.5]);
}
for(const growth of [A=>A[0]===0?0:1,()=>NaN,()=>Infinity]) {
 const life=fixture(growth),r=life.learn(['food']);
 assert.equal(r.updated,false);assert.equal(r.reason,'growth_filter_rejected');assert.equal(r.halvings,12);
 assert.deepEqual(Array.from(r.applied),[0]);assert.equal(life.lessons,0);assert.equal(life.rejected,1);assert.equal(life.brain.updates,7);
 for(const a of Object.values(life.brain.parameters()))assert.deepEqual(Array.from(a),[0]);
}
{
 const life=fixture(()=>{throw Error('growth checked rejected solver');},false),r=life.learn(['pain']);
 assert.equal(r.updated,false);assert.equal(r.reason,'phase_failed');assert.equal(r.applied,null);
 assert.equal(life.lessons,0);assert.equal(life.rejected,1);assert.equal(life.brain.updates,7);
}
console.log('worm learning accounting: 5 scenarios passed');
