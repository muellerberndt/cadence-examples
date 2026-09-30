#!/usr/bin/env python3
"""Export what the browser needs and the parity cases that bind it to cadence.

  web/data/brain.json       the newborn brain: the compiled connectome graph
                            with the weights, biases and live state it hatches
                            with after the prenatal reflex lessons
  web/data/connectome.json  neurons, positions, transmitters, synapses
  web/data/params.json      the world and learning constants
  tests/parity_cases.json   lived ticks and lessons with the library's results
  tests/body_cases.json     scripted crawls the browser body must retrace
"""
import json
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import cadence  # noqa: E402
from worm.brain import WormBrain  # noqa: E402


def spec(brain: WormBrain) -> dict:
    """The complete compiled brain, plus the neuron-to-neuron view the page
    draws. A entries are (post, pre) in connectome order with each value's
    global edge index; `pos` maps a connectome index to its patch. Synapses
    running toward the commands in the declaration order are live state edges;
    the rest arrive from the previous tick's inputs."""
    info = brain.brain.inspect()
    H, order = brain.H, brain.order
    A = {"rows": [], "cols": [], "vals": [], "edge": []}
    B, C = [], []
    weights = brain.brain.weights
    for k, (kind, source, target) in enumerate(info["edges"]):
        if kind == "input" and source < H and target < H:      # from the previous tick
            A["rows"].append(order[target])
            A["cols"].append(source)
            A["vals"].append(weights[k])
            A["edge"].append(k)
        elif kind == "input" and target < H:                   # a sense
            B.append([order[target], source - H, weights[k], k])
        elif kind == "state" and target < H:                   # live, within the tick
            A["rows"].append(order[target])
            A["cols"].append(order[source])
            A["vals"].append(weights[k])
            A["edge"].append(k)
        elif kind == "state":                                  # a command drive readout
            C.append([target - H, order[source], weights[k], k])
    return {
        "cadence": cadence.__version__,
        "H": H,
        "names": brain.names,
        "senses": brain.senses,
        "readouts": brain.readouts,
        "pos": list(brain.pos),
        "config": {k: v for k, v in info["config"].items() if k not in ("device", "dtype")},
        "n_inputs": info["input_samples"],
        "edges": [list(e) for e in info["edges"]],
        "weights": list(weights),
        "biases": list(brain.brain.biases),
        "state": list(brain.brain.state),
        "outputs": [{"name": o["name"], "patch": H + k} for k, o in enumerate(info["outputs"])],
        "event_id": info["last_event_id"],
        "admissions": info["admissions"],
        "A": A,
        "B": B,
        "C": C,
    }


def main() -> None:
    web = ROOT / "web" / "data"
    web.mkdir(parents=True, exist_ok=True)
    brain = WormBrain(seed=0)
    prenatal = brain.born()
    s = spec(brain)
    s["prenatal"] = {"lessons": len(prenatal), "admitted": sum(l.updated for l in prenatal)}
    (web / "brain.json").write_text(json.dumps(s, separators=(",", ":")))
    for name in ("connectome.json", "params.json"):
        shutil.copy(ROOT / "data" / name, web / name)

    # Parity: the library's own answers along one continued life. Each case's
    # ticks feed the brain's own previous settled state back in, so the browser
    # engine must retrace the whole trajectory, then learn the same lessons.
    cases = []
    scripts = [
        {"ticks": [[0, 0, 0, 0]] * 2 + [[0.4, 0, 0, 0], [0.8, 0, 0, 0], [1, 0, 1, 0], [1, 0, 1, 0]],
         "lesson": ["food"]},
        {"ticks": [[0, 0.2, 0, 0], [0.1, 0.6, 0, 0], [0, 0.9, 0, 0], [0, 1, 0, 1], [0, 1, 0, 1]],
         "lesson": ["pain"]},
        {"ticks": [[0.5, 0.5, 0, 0]] * 3 + [[0.7, 0.7, 1, 1]],
         "lesson": ["food", "pain"]},
    ]
    for script in scripts:
        outputs = []
        for u in script["ticks"]:
            y = brain.sense(u)
            outputs.append([float(v) for v in y])
        lesson = brain.learn(*script["lesson"])
        cases.append({
            "ticks": script["ticks"],
            "kinds": script["lesson"],
            "outputs": outputs,
            "state": list(brain.brain.state),
            "lesson": {
                "accepted": lesson.updated,
                "reason": lesson.reason,
                "rows": lesson.rows,
                "energy": lesson.energy,
                "weights": list(brain.brain.weights),
                "biases": list(brain.brain.biases),
            },
        })
    (ROOT / "tests").mkdir(exist_ok=True)
    (ROOT / "tests" / "parity_cases.json").write_text(json.dumps(cases))

    # The body: scripted crawls, reversals, omega turns and wall turns; web/life.js must lay the same track.
    from worm.rng import Rng
    from worm.world import World
    bodies = []
    for seed, script in (
        (1, [["step", 200], ["reverse", 2.0, -2.4], ["step", 160], ["steer", 0.5], ["step", 120], ["reverse", 1.2, 1.7], ["step", 90]]),
        (2, [["move", 0.5, 0.6], ["step", 400], ["reverse", 3.0, -3.0], ["step", 300]]),
        (3, [["move", 11.6, 7.7], ["step", 250], ["reverse", 0.6, 2.9], ["step", 8], ["reverse", 0.6, -2.9], ["step", 200]]),
    ):
        w = World(brain.p, Rng(seed * 7919 + 17))
        for what, *args in script:
            if what == "step":
                for _ in range(args[0]):
                    w.physics(brain.p["physics_step"], False)
            elif what == "reverse":
                w.reverse(*args)
            elif what == "steer":
                w.body.heading += args[0]
            elif what == "move":
                dx, dy = args[0] - w.body.x, args[1] - w.body.y
                w.body.trail = [(x + dx, y + dy) for x, y in w.body.trail]
                w.body.x, w.body.y = args
        bodies.append({"seed": seed, "script": script, "trail": [list(q) for q in w.body.trail]})
    (ROOT / "tests" / "body_cases.json").write_text(json.dumps(bodies))
    print(f"brain.json: {len(s['weights'])} connections, prenatal {s['prenatal']}; "
          f"{len(cases)} parity cases")


if __name__ == "__main__":
    main()
