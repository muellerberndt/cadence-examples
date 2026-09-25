#!/usr/bin/env python3
"""The gain of the antennal lobe's local neurons, selected by protocol.

At the one global gain the flight circuits select, the antennal lobe ignites: 104 of its 425
local neurons are predicted cholinergic, they are the broad ones (a median of 88 projection
neuron targets each against 16 for the GABAergic ones), and they excite each other through
194,044 synapses, so any input lights the lobe into one state and the Kenyon cell code is the
same for every odour (cosine 0.98). A threshold does not tame the loop (at a bias of -6 the
local neurons stay a third active); a gain does. This tool selects the local neurons' output
gain the way the global gain was selected: the smallest attenuation at which the training facts
pass, under a sparsity cap, with shuffled wirings selecting their own, and writes the receipt.

The facts are the animal's: about 5 percent of Kenyon cells answer an odour (Turner, Bazhenov
and Laurent 2008) and the codes of two odours are specific (Honegger, Campbell and Turner 2011).

    python tools/class_gains.py            # writes receipts/class_gains.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cadence  # noqa: E402
from cadence import Brain, NeuronModel, Protocol, Row, select_gain  # noqa: E402
from cadence.protocol import Levels, shuffled  # noqa: E402
from fruitfly.banc import MANIFEST_PATH  # noqa: E402
from fruitfly.brain import GAIN, load_fly  # noqa: E402

POPULATION = "ln"  # the antennal lobe's local neurons, every transmitter
ATTENUATIONS = (0.0, 0.7, 1.4, 2.1, 3.0, 4.6)  # -log gain: 1, 1/2, 1/4, 1/8, 1/20, 1/100
LEVELS = Levels(sparse_min=0.02, sparse_max=0.15, specific_max=0.25, code_level=0.05)
SHUFFLED_SEEDS = (1, 2)


def protocol() -> Protocol:
    stimuli = {
        "fruit": ("orn:decaying_fruit:left", "orn:decaying_fruit:right"),
        "yeast": ("orn:yeasty:left", "orn:yeasty:right"),
    }
    rows = [
        Row("fruit recruits a sparse Kenyon cell code", "fruit", "kc", "sparse", tier="physiology"),
        Row("yeast recruits a sparse Kenyon cell code", "yeast", "kc", "sparse", tier="physiology"),
        Row("the two codes are specific", "fruit", "kc", "specific", versus="yeast", tier="physiology"),
    ]
    training = [("fruit", "kc", "sparse"), ("yeast", "kc", "sparse"), ("fruit", "kc", "specific", "yeast")]
    return Protocol(stimuli=stimuli, rows=rows, training=training, levels=LEVELS, steps=100)


def main() -> None:
    fly = load_fly()
    C = fly.connectome  # the whole brain, as the global gain was selected: a sub-net's own stimulated receptors exceed the sparsity cap
    ln = np.array(C.populations[POPULATION])
    proto = protocol()

    def factory(connectome):
        def make(attenuation: float) -> Brain:
            log_gain = np.zeros(connectome.n)
            log_gain[ln] = -attenuation
            return Brain(connectome, NeuronModel(gain=GAIN), log_gain=log_gain)
        return make

    out = {"tool": "tools/class_gains.py", "library": cadence.__version__, "fixture": json.loads(MANIFEST_PATH.read_text())["fixture_sha256"],
           "population": POPULATION, "global_gain": GAIN, "attenuations": list(ATTENUATIONS), "levels": LEVELS.__dict__ if hasattr(LEVELS, "__dict__") else {k: getattr(LEVELS, k) for k in LEVELS.__slots__},
           "brain": {"neurons": int(C.n), "classes": int(C.synapses)}, "protocol": proto.to_dict(), "wirings": {}}
    for name, connectome in [("measured", C)] + [(f"shuffled:{s}", shuffled(C, s)) for s in SHUFFLED_SEEDS]:
        t0 = time.time()
        try:
            chosen, table = select_gain(factory(connectome), proto, ATTENUATIONS, sparsity_cap=0.05)
        except ValueError as e:
            chosen, table = None, [{"error": str(e)}]
            print(f"{name}: {e}", flush=True)
        score = proto.score(factory(connectome)(chosen)) if chosen is not None else None
        out["wirings"][name] = {"selected_attenuation": chosen, "selected_log_gain": None if chosen is None else -chosen, "table": table, "score": score}
        best = max((r.get("facts_passed", 0) for r in table), default=0)
        print(f"{name}: attenuation {chosen} (log gain {None if chosen is None else -chosen}); facts passed by candidate {[(r.get('gain'), r.get('facts_passed'), round(r.get('fraction_active', 0), 3), r.get('admissible')) for r in table]}; best {best}/3 ({time.time() - t0:.0f} s)", flush=True)
        if score: print("   rows:", [(r["id"], r["passed"], {k: round(v, 3) for k, v in r["reading"].items()}) for r in score["rows"]], flush=True)
    (ROOT / "receipts" / "class_gains.json").write_text(json.dumps(out, indent=1))
    print("written receipts/class_gains.json")


if __name__ == "__main__":
    main()
