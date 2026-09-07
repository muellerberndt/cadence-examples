"""09 C. elegans: the published connectome, four training facts, seventeen held-out ablation phenotypes.

Run:  python train.py                      (a few minutes)
      python train.py --verify receipt.json

The wiring is the hermaphrodite connectome (Cook et al. 2019 as distributed by the
Varshney/Chklovskii toolbox, cell classes from c302): 300 neurons, 3,638 chemical
connections with synapse counts, 1,093 gap junctions. Each neuron is an owner; each
chemical connection a seam in one direction, each gap junction a seam in both. Sensory
neurons are clamped by a stimulus; the readout is textbook: B-type ventral cord motor
neurons (DB, VB) drive forward locomotion and A-type (DA, VA) backward, so speed is the
B-type mean minus the A-type mean.

The net learns its seam strengths and signs from four facts, with the free/nudged rule:
at rest the worm moves forward, anterior touch and the aversive chemical channel cause a
reversal, posterior touch an acceleration. The nudge pulls the motor owners toward the
fact's target pattern. Nothing else is shown to it. Then seventeen classical laser-ablation
phenotypes are scored, each by removing the named neurons and settling again, and the same
protocol is run on a wiring whose postsynaptic endpoints were shuffled (every count, sign,
and out-degree kept). What the wiring passes and the shuffle does not is what the wiring
predicted. Eight seeds each.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd

HERE = Path(__file__).resolve().parent
SOURCES = [("09_celegans/train.py", Path(__file__).resolve()), ("09_celegans/fixture.json", HERE / "fixture.json")]

STIMULI = {
    "baseline": (),
    "anterior_touch": ("ALML", "ALMR", "AVM"),
    "posterior_touch": ("PLML", "PLMR"),
    "ash": ("ASHL", "ASHR"),
    "nose_touch": ("ASHL", "ASHR", "FLPL", "FLPR", "OLQDL", "OLQDR", "OLQVL", "OLQVR"),
    "tap": ("ALML", "ALMR", "AVM", "PLML", "PLMR"),
}
FORWARD, BACKWARD = ("DB", "VB"), ("DA", "VA")
TRAINING = (("baseline", "forward"), ("anterior_touch", "reversal"), ("posterior_touch", "acceleration"), ("ash", "reversal"))
MARGIN = 0.05
ROWS = (
    ("R01", ("ALML", "ALMR", "AVM"), "anterior_touch", "no_reversal", "Chalfie et al. 1985 J Neurosci 5:956", True),
    ("R02", ("AVAL", "AVAR"), "anterior_touch", "no_reversal", "Chalfie et al. 1985", False),
    ("R03", ("AVDL", "AVDR"), "anterior_touch", "reversal_reduced", "Chalfie et al. 1985; Wicks & Rankin 1995 J Neurosci 15:2434", False),
    ("R04", ("AVBL", "AVBR"), "baseline", "forward_impaired", "Chalfie et al. 1985", False),
    ("R05", ("PVCL", "PVCR"), "posterior_touch", "no_acceleration", "Chalfie et al. 1985", False),
    ("R06", ("AVM",), "anterior_touch", "reversal_retained", "Chalfie et al. 1985", False),
    ("R07", ("ALML", "ALMR"), "anterior_touch", "reversal_reduced", "Chalfie et al. 1985", False),
    ("R08", ("PLML", "PLMR"), "tap", "reversal_enhanced", "Wicks & Rankin 1995", False),
    ("R09", ("PVCL", "PVCR"), "tap", "reversal_enhanced", "Wicks & Rankin 1995", False),
    ("R10", ("DVA",), "tap", "reversal_reduced", "Wicks & Rankin 1995", False),
    ("R11", (), "tap", "reversal", "Wicks & Rankin 1995; Chalfie & Sulston 1981", False),
    ("R12", (), "nose_touch", "reversal", "Kaplan & Horvitz 1993 PNAS 90:2227", False),
    ("R13", ("ASHL", "ASHR"), "nose_touch", "reversal_reduced", "Kaplan & Horvitz 1993", False),
    ("R14", ("AVAL", "AVAR", "AVDL", "AVDR"), "anterior_touch", "no_reversal", "Chalfie et al. 1985", False),
    ("R15", ("AVAL", "AVAR"), "ash", "no_reversal", "Piggott et al. 2011 Cell 147:922", False),
    ("R16", ("PLML", "PLMR"), "posterior_touch", "no_acceleration", "Chalfie et al. 1985", True),
    ("R17", ("AVBL", "AVBR"), "posterior_touch", "no_acceleration", "Chalfie et al. 1985", False),
)
PREDICATES = {
    "forward": "speed(cond) >= margin",
    "reversal": "speed(cond) <= -margin",
    "acceleration": "speed(cond) - speed(baseline) >= margin",
    "no_reversal": "intact reverses; speed_ablated(cond) >= -margin",
    "reversal_retained": "intact reverses; speed_ablated(cond) <= -margin",
    "reversal_reduced": "intact reverses; speed_ablated(cond) >= speed_intact(cond) + margin",
    "reversal_enhanced": "intact reverses; speed_ablated(cond) <= speed_intact(cond) - margin",
    "forward_impaired": "intact moves forward; speed_ablated(baseline) <= speed_intact(baseline) - margin",
    "no_acceleration": "intact accelerates; speed_ablated(cond) - speed_ablated(baseline) <= margin",
}
SEEDS, SHUFFLED_SEEDS = 8, 8
UPDATES = 600
CONVENTIONS = ("count", "fan_in")  # seam strength per contact, or per owner normalised by its total input
FAN_IN_GAINS = (0.3, 0.6, 1.2)  # for the fan-in convention the gain is chosen by training facts passed
CONFIG = cd.LearnerConfig(beta=0.1, eta=0.2, eta_bias=0.05, nudge="quadratic", tolerance=1e-4, free_steps=200, nudged_steps=60, scale_floor=0.0, scale_cap=8.0)
GAIN_GRID = [0.002 * 1.4**k for k in range(16)]
SPARSITY_CAP = 0.10  # at most this fraction of neurons active under a training stimulus
CLAMP = 3.0  # drive on a stimulated sensory neuron
STEPS = 200


def evaluate_predicate(predicate: str, condition: str, speeds: dict[str, float], intact: dict[str, float], margin: float = MARGIN) -> bool:
    s, i = speeds, intact
    if predicate == "forward":
        return s[condition] >= margin
    if predicate == "reversal":
        return s[condition] <= -margin
    if predicate == "acceleration":
        return s[condition] - s["baseline"] >= margin
    if predicate == "no_reversal":
        return i[condition] <= -margin and s[condition] >= -margin
    if predicate == "reversal_retained":
        return i[condition] <= -margin and s[condition] <= -margin
    if predicate == "reversal_reduced":
        return i[condition] <= -margin and s[condition] >= i[condition] + margin
    if predicate == "reversal_enhanced":
        return i[condition] <= -margin and s[condition] <= i[condition] - margin
    if predicate == "forward_impaired":
        return i["baseline"] >= margin and s["baseline"] <= i["baseline"] - margin
    if predicate == "no_acceleration":
        return i[condition] - i["baseline"] >= margin and s[condition] - s["baseline"] <= margin
    raise ValueError(predicate)


def load() -> tuple[cd.Wiring, dict[str, int], dict[str, Any]]:
    raw = (HERE / "fixture.json").read_bytes()
    fixture = json.loads(raw)
    names = [n["name"] for n in fixture["neurons"]]
    index = {name: i for i, name in enumerate(names)}
    inhibitory = {n["name"] for n in fixture["neurons"] if "GABA" in (n.get("neurotransmitter") or "")}
    pre, post, count, sign = [], [], [], []
    for s in fixture["chemical"]:
        pre.append(index[s["pre"]])
        post.append(index[s["post"]])
        count.append(float(s["count"]))
        sign.append(-1.0 if s["pre"] in inhibitory else 1.0)
    for g in fixture["gap"]:
        for a, b in ((g["a"], g["b"]), (g["b"], g["a"])):
            pre.append(index[a])
            post.append(index[b])
            count.append(float(g["count"]))
            sign.append(1.0)
    sets = {"forward": [i for i, n in enumerate(names) if n.startswith(FORWARD)], "backward": [i for i, n in enumerate(names) if n.startswith(BACKWARD)]}
    for stimulus, cells in STIMULI.items():
        sets[stimulus] = [index[c] for c in cells]
    wiring = cd.Wiring.from_edges(len(names), pre=pre, post=post, count=count, sign=sign, sets=sets, label="celegans-cook2019")
    meta = {"fixture_sha256": hashlib.sha256(raw).hexdigest(), "fixture_digest": fixture["fixture_digest"], "sources": fixture["sources"], "neurons": len(names), "chemical": len(fixture["chemical"]), "gap_junctions": len(fixture["gap"]), "inhibitory_neurons": len(inhibitory), "readout": {"forward": FORWARD, "backward": BACKWARD}}
    return wiring, index, meta


class Worm:
    def __init__(self, wiring: cd.Wiring, index: dict[str, int], seed: int, convention: str = "count") -> None:
        self.wiring = wiring
        self.index = index
        self.convention = convention
        rng = np.random.default_rng(seed)
        scale = wiring.sign * rng.uniform(0.5, 1.5, wiring.edges)  # a jittered start, signs from the neurotransmitter
        if convention == "fan_in":  # each owner scales its inbox by the square root of its total contact count
            total = np.zeros(wiring.n)
            np.add.at(total, wiring.post, wiring.count)
            scale = scale / np.sqrt(total[wiring.post])
        engine = cd.Settlement(wiring, cd.learning_rule(dt=0.5, clamp_amplitude=CLAMP), edge_scale=scale)
        outputs = list(wiring.sets["forward"]) + list(wiring.sets["backward"])
        self.learner = cd.Learner(engine, outputs, CONFIG, symmetric=False)
        self.forward = np.asarray(wiring.sets["forward"])
        self.backward = np.asarray(wiring.sets["backward"])

    def drive(self, stimulus: str) -> np.ndarray:
        return self.learner.engine.clamp_vector(list(self.wiring.sets[stimulus]))[None, :]

    def speeds(self, mask: np.ndarray | None = None) -> dict[str, float]:
        out = {}
        for stimulus in STIMULI:
            state = self.learner.engine.settle_batch(self.drive(stimulus), steps=STEPS, mask=mask, tolerance=1e-5)
            s = state.activation[0]
            out[stimulus] = float(s[self.forward].mean() - s[self.backward].mean())
        return out

    def calibrate(self) -> float:
        """The gain, chosen on the training stimuli and facts alone.

        Count convention: the largest gain at which no training stimulus lights more than
        the sparsity cap of the net. Fan-in convention: the gain from a declared grid that
        passes the most training facts after learning, ties to the smaller.
        """
        if self.convention == "fan_in":
            best_gain, best = FAN_IN_GAINS[0], -1
            for gain in FAN_IN_GAINS:
                probe = Worm(self.wiring, self.index, 0, self.convention)
                probe.learner.engine = probe.learner._with_gain(gain)
                probe.train(UPDATES // 2, np.random.default_rng(0))
                speeds = probe.speeds()
                passed = sum(evaluate_predicate(t, c, speeds, speeds) for c, t in TRAINING)
                if passed > best:
                    best_gain, best = gain, passed
            self.learner.engine = self.learner._with_gain(best_gain)
            return best_gain
        drives = np.concatenate([self.drive(c) for c, _ in TRAINING if c != "baseline"])
        chosen = GAIN_GRID[0]
        for gain in GAIN_GRID:
            engine = self.learner._with_gain(gain)
            state = engine.settle_batch(drives, steps=STEPS, tolerance=1e-5)
            if (state.activation > 0.5).mean(axis=1).max() > SPARSITY_CAP:
                break
            chosen = gain
        self.learner.engine = self.learner._with_gain(chosen)
        return chosen

    def target(self, kind: str) -> np.ndarray:
        t = np.zeros((1, self.wiring.n))
        if kind in ("forward", "acceleration"):
            t[0, self.forward], t[0, self.backward] = 1.0, 0.0
        else:  # reversal
            t[0, self.forward], t[0, self.backward] = 0.0, 1.0
        return t

    def train(self, updates: int, rng: np.random.Generator) -> list[dict[str, float]]:
        """Every update contrasts all four training facts at once, so a seam must serve all of them."""
        learner = self.learner
        history = []
        drive = np.concatenate([self.drive(c) for c, _ in TRAINING])
        target = np.concatenate([self.target(k) for _, k in TRAINING])
        for k in range(updates):
            free = learner.free(drive)
            plus = learner.nudged(drive, free, target)
            minus = learner.nudged(drive, free, target, sign=-1.0)
            learner.update(free, plus, minus)
            if (k + 1) % 25 == 0:
                speeds = self.speeds()
                passed = sum(evaluate_predicate(t, c, speeds, speeds) for c, t in TRAINING)
                history.append({"update": k + 1, "training_passed": passed, **{f"speed_{c}": v for c, v in speeds.items()}})
        return history

    def mask_for(self, cells: tuple[str, ...]) -> np.ndarray | None:
        if not cells:
            return None
        mask = np.ones(self.wiring.n)
        for c in cells:
            mask[self.index[c]] = 0.0
        return mask

    def active_fraction(self) -> float:
        drives = np.concatenate([self.drive(c) for c, _ in TRAINING if c != "baseline"])
        state = self.learner.engine.settle_batch(drives, steps=STEPS, tolerance=1e-5)
        return float((state.activation > 0.5).mean())

    def score(self) -> dict[str, Any]:
        intact = self.speeds()
        training = [{"condition": c, "target": t, "passed": evaluate_predicate(t, c, intact, intact)} for c, t in TRAINING]
        cache: dict[tuple[str, ...], dict[str, float]] = {(): intact}
        rows = []
        for rid, ablate, stimulus, predicate, reference, trivial in ROWS:
            if ablate not in cache:
                cache[ablate] = self.speeds(self.mask_for(ablate))
            ablated = cache[ablate]
            rows.append({"id": rid, "ablate": list(ablate), "stimulus": stimulus, "predicate": predicate, "reference": reference, "trivial": trivial, "speed_intact": intact[stimulus], "speed_ablated": ablated[stimulus], "speed_intact_baseline": intact["baseline"], "speed_ablated_baseline": ablated["baseline"], "passed": evaluate_predicate(predicate, stimulus, ablated, intact)})
        nontrivial = [r for r in rows if not r["trivial"]]
        return {"intact_speeds": intact, "training": training, "training_passed": sum(t["passed"] for t in training), "rows": rows, "held_out_passed": sum(r["passed"] for r in rows), "held_out_total": len(rows), "nontrivial_passed": sum(r["passed"] for r in nontrivial), "nontrivial_total": len(nontrivial)}


def arm(wiring: cd.Wiring, index: dict[str, int], seed: int, label: str, convention: str) -> dict[str, Any]:
    t0 = time.perf_counter()
    worm = Worm(wiring, index, seed, convention)
    gain = worm.calibrate()
    history = worm.train(UPDATES, np.random.default_rng(seed))
    result = worm.score()
    result.update({"wiring": label, "convention": convention, "seed": seed, "gain": gain, "active_fraction_after": worm.active_fraction(), "history": history, "seconds": time.perf_counter() - t0, "parameters": worm.learner.parameters()})
    print(f"{label}/{convention} seed {seed}: gain {gain:.3f}, training {result['training_passed']}/4, held out {result['held_out_passed']}/{result['held_out_total']} (nontrivial {result['nontrivial_passed']}/{result['nontrivial_total']}), active {result['active_fraction_after']:.2f}, {result['seconds']:.0f}s", flush=True)
    return result


def run(out: Path, seeds: int, shuffled_seeds: int) -> dict[str, Any]:
    wiring, index, meta = load()
    print(f"{meta['neurons']} neurons, {wiring.edges} seams ({meta['chemical']} chemical, {meta['gap_junctions']} gap junctions both ways), {meta['inhibitory_neurons']} GABAergic", flush=True)
    measured, controls, summary, per_row = [], [], {}, {}
    for convention in CONVENTIONS:
        m = [arm(wiring, index, s, "measured", convention) for s in range(seeds)]
        c = [arm(cd.shuffled(wiring, 1000 + s), index, s, "shuffled", convention) for s in range(shuffled_seeds)]
        measured += m
        controls += c
        summary[convention] = {
            "measured_held_out": [r["held_out_passed"] for r in m], "shuffled_held_out": [r["held_out_passed"] for r in c],
            "measured_mean": float(np.mean([r["held_out_passed"] for r in m])), "shuffled_mean": float(np.mean([r["held_out_passed"] for r in c])),
            "measured_nontrivial_mean": float(np.mean([r["nontrivial_passed"] for r in m])), "shuffled_nontrivial_mean": float(np.mean([r["nontrivial_passed"] for r in c])),
            "measured_training_mean": float(np.mean([r["training_passed"] for r in m])), "shuffled_training_mean": float(np.mean([r["training_passed"] for r in c])),
            "measured_active_fraction": float(np.mean([r["active_fraction_after"] for r in m])), "shuffled_active_fraction": float(np.mean([r["active_fraction_after"] for r in c])),
        }
        per_row[convention] = {rid: {"measured": sum(r["rows"][k]["passed"] for r in m), "shuffled": sum(r["rows"][k]["passed"] for r in c)} for k, (rid, *_) in enumerate(ROWS)}
        print(f"{convention}: held out measured {summary[convention]['measured_mean']:.2f}/17 vs shuffled {summary[convention]['shuffled_mean']:.2f}/17; training {summary[convention]['measured_training_mean']:.2f}/4 vs {summary[convention]['shuffled_training_mean']:.2f}/4; active fraction {summary[convention]['measured_active_fraction']:.2f}", flush=True)
    conformance = cd.conformance(Worm(wiring, index, 0).learner.engine, list(wiring.sets["anterior_touch"]), steps=STEPS)
    body = {
        "connectome": meta, "wiring": wiring.summary(), "stimuli": STIMULI, "training": TRAINING, "rows": [dict(zip(("id", "ablate", "stimulus", "predicate", "reference", "trivial"), r, strict=True)) for r in ROWS],
        "predicates": PREDICATES, "margin": MARGIN, "config": CONFIG.to_dict(), "conventions": list(CONVENTIONS), "gain_grid": GAIN_GRID, "fan_in_gains": list(FAN_IN_GAINS), "sparsity_cap": SPARSITY_CAP, "updates": UPDATES, "steps": STEPS,
        "measured": measured, "shuffled": controls, "summary": summary, "per_row": per_row, "conformance": conformance,
        "boundary": {"learning_rule": "free/nudged contrastive Hebbian, centered, owner-local; quadratic nudge on the motor owners", "goal_enters_only_through_the_nudge": True, "learned": "seam strengths and signs, starting from neurotransmitter signs; the wiring itself is fixed", "conventions": "count: drive per contact times synapse count; fan_in: each owner's inbox normalised by the square root of its total contact count", "readout_is_declared_textbook_physiology": True, "training_facts": 4, "held_out_rows": len(ROWS), "control": "postsynaptic endpoints shuffled, counts, signs, out-degrees and sets kept", "gain_calibrated_on_training_stimuli_only": True},
    }
    receipt = cd.Receipt.build("cadence-examples/09-celegans/v1", body, sources=SOURCES)
    receipt.write(out)
    print(f"conformance {conformance['max_abs_deviation']:.1e}; receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    for arm_ in body["measured"] + body["shuffled"]:
        intact = arm_["intact_speeds"]
        for row in arm_["rows"]:
            speeds = {row["stimulus"]: row["speed_ablated"], "baseline": row["speed_ablated_baseline"]}
            if row["passed"] != evaluate_predicate(row["predicate"], row["stimulus"], speeds, intact, body["margin"]):
                return f"row {row['id']} of {arm_['wiring']} seed {arm_['seed']}: pass flag does not follow from the speeds"
        if arm_["held_out_passed"] != sum(r["passed"] for r in arm_["rows"]):
            return "held-out tally does not follow from the rows"
    for convention, s in body["summary"].items():
        if abs(float(np.mean(s["measured_held_out"])) - s["measured_mean"]) > 1e-9:
            return f"{convention}: measured mean does not follow from the arms"
    if not body["conformance"]["ledger"]["clean"]:
        return "reference ledger is not clean"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seeds", type=int, default=SEEDS)
    parser.add_argument("--shuffled-seeds", type=int, default=SHUFFLED_SEEDS)
    parser.add_argument("--output", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    run(args.output, args.seeds, args.shuffled_seeds)
    return 0


if __name__ == "__main__":
    sys.exit(main())
