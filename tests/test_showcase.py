"""Public demo contracts: numerical parity, causal observation, and evidence failures."""

import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "showcase"))
from benchmark import hashes
from build_composites import sources
from embodied import motor_settlement
from model import OnlineMLP, fast_memory, key_bank, reference, samples, worm_engine
from verify import verify


def node(code):
    return json.loads(
        subprocess.check_output(
            ["node", "--input-type=module", "-e", code], cwd=ROOT, text=True
        )
    )


def test_worm_cadence_matches_independent_dynamics_under_lesions():
    d, m = samples(911, 8, lesions=64)
    actual = (
        worm_engine().settle_batch(d, mask=m, steps=200, tolerance=1e-10).activation
    )
    np.testing.assert_allclose(actual, reference(d, m), atol=1e-9, rtol=0)


def test_browser_kernels_match_current_cadence_and_online_mlp():
    result = node("""
import {FastMemory,MLP,keys,Worm} from './showcase/engine.js';
import {motorSettlement} from './showcase/embodied.js';
import fs from 'node:fs';
const ev=JSON.parse(fs.readFileSync('showcase/evidence.json')),data=JSON.parse(fs.readFileSync('showcase/worm.json'));
const f=new FastMemory(),net=new MLP(ev.browser.online_mlp),bank=keys(.5);
for(let k=0;k<20;k++){const key=bank[k%8],v=Array(4).fill(0);v[(k+2)%4]=1;f.observe(key,v);net.observe(key,v,10);}
const w=new Worm(data),d=Array(w.n).fill(0),m=Array(w.n).fill(1);d[23]=1;d[71]=.7;m[16]=0;
console.log(JSON.stringify({fast:f.w,mlp:bank.map(k=>net.predict(k)),worm:w.settle(d,m).state,drive:d,mask:m,motor:motorSettlement([-1.8,1.1],[.45,.35]).state}));
""")
    f = fast_memory()
    net = OnlineMLP()
    bank = key_bank(0.5)
    for k in range(20):
        v = np.eye(4)[(k + 2) % 4]
        f.observe(bank[k % 8 : k % 8 + 1], v[None])
        net.observe(bank[k % 8], v, 10)
    np.testing.assert_allclose(result["fast"], f.strength[0], atol=1e-12, rtol=0)
    np.testing.assert_allclose(
        result["mlp"], [net.predict(k) for k in bank], atol=1e-12, rtol=0
    )
    actual = (
        worm_engine()
        .settle(result["drive"], mask=result["mask"], steps=200, tolerance=1e-10)
        .activation
    )
    np.testing.assert_allclose(result["worm"], actual, atol=1e-9, rtol=0)
    np.testing.assert_allclose(
        result["motor"], motor_settlement([-1.8, 1.1], [0.45, 0.35]), atol=1e-12, rtol=0
    )


def test_forager_cannot_read_nectar_before_contact():
    result = node("""
import {Forager,FastMemory,flowers} from './showcase/engine.js';
const a=new Forager(new FastMemory()),b=new Forager(new FastMemory()),f=flowers(),g=flowers();g.forEach(x=>x.value=3-x.value);
for(let k=0;k<10;k++){a.step(f);b.step(g);}
console.log(JSON.stringify({a:[a.x,a.y,a.target,a.encounters],b:[b.x,b.y,b.target,b.encounters]}));
""")
    assert result["a"] == result["b"] and result["a"][-1] == 0


def test_new_task_revises_without_erasing_old_tasks_and_restores():
    result = node("""
import {TaskLessons} from './showcase/embodied.js';
const m=new TaskLessons(),before=m.recall(3);m.teach(3,3);const after=m.recall(3);m.teach(3,2);const restored=new TaskLessons(m.records);
console.log(JSON.stringify({before,after,answers:[0,1,2,3].map(i=>restored.recall(i))}));
""")
    assert result == {"before": None, "after": 3, "answers": [0, 1, 2, 2]}


def test_mouse_stops_when_goal_is_disconnected():
    result = node("""
import {Mouse,neighbors} from './showcase/embodied.js';
const m=new Mouse();neighbors(m.world,m.world.goal).forEach(i=>m.world.grid[i]=1);m.repair();for(let k=0;k<500;k++)m.step();
console.log(JSON.stringify({moves:m.moves,next:m.next()}));
""")
    assert result["next"] == -1


def test_receipts_are_complete_and_source_bound():
    verify(json.loads((ROOT / "showcase/evidence.json").read_text()), hashes())
    verify(
        json.loads((ROOT / "showcase/composite_evidence.json").read_text()),
        sources(),
        True,
    )


@pytest.mark.parametrize("mutation", ["source", "accuracy", "control"])
def test_receipt_gate_rejects_forged_results_even_with_recomputed_digest(mutation):
    body = json.loads((ROOT / "showcase/evidence.json").read_text())
    if mutation == "source":
        body["sources"].pop(next(iter(body["sources"])))
    if mutation == "accuracy":
        body["memory"][0]["arms"]["cadence"]["accuracy"] = 0.5
    if mutation == "control":
        body["worm"][0]["rows"][0]["arms"].pop("unrolled32")
    body.pop("digest")
    body["digest"] = hashlib.sha256(
        json.dumps(body, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    with pytest.raises(ValueError):
        verify(body, hashes())


def test_habitat_circuit_matches_cadence():
    result = node("""
import {WormArena} from './showcase/worm_arena.js';
import {readFileSync} from 'node:fs';
const w=new WormArena(JSON.parse(readFileSync('./showcase/worm.json')));
w.step();
console.log(JSON.stringify({state:w.state,drive:w.drive,mask:w.mask,moves:w.moves}));
""")
    actual = (
        worm_engine()
        .settle(result["drive"], mask=result["mask"], steps=200, tolerance=1e-10)
        .activation
    )
    np.testing.assert_allclose(result["state"], actual, atol=1e-9, rtol=0)
    assert result["moves"] == 1


def test_diagnostic_replays_match_executed_settlements_and_release_decays():
    result = node("""
import {readFileSync} from 'node:fs';
import {Worm,zeros} from './showcase/engine.js';
import {Mouse,motorSettlement} from './showcase/embodied.js';
import {repairTrace} from './showcase/telemetry.js';
const data=JSON.parse(readFileSync('./showcase/worm.json')),w=new Worm(data),drive=zeros(w.n),mask=Array(w.n).fill(1);
data.stimuli.odor.forEach(i=>drive[i]=1);
const r=w.settle(drive,mask),m=new Mouse(),a=motorSettlement([-1.8,1.1],[.45,.35]);
const sources=[{...r,drive,mask,edges:data.edges},{...m.circuit,edges:m.circuit.engine.data.edges},{...a,steps:80,dt:.25}];
const errors=sources.map(s=>Math.max(...repairTrace(s).frames.at(-1).map((v,i)=>Math.abs(v-s.state[i]))));
const release=repairTrace(sources[0],true);
console.log(JSON.stringify({errors,initial:Math.max(...release.frames[0]),released:Math.max(...release.frames.at(-1))}));
""")
    assert max(result["errors"]) < 1e-12
    assert result["initial"] > 0
    assert result["released"] < result["initial"] * 1e-6
