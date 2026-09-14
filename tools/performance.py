"""Small fresh checks of the shipped players. Full multi-opponent results are in receipts."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).parent))
from pages import engine_from, engine_outputs


def main() -> int:
    from sklearn.datasets import load_digits
    from sklearn.model_selection import train_test_split

    dataset = load_digits()
    _, test, _, labels = train_test_split(
        dataset.data / 16,
        dataset.target,
        test_size=0.2,
        random_state=0,
        stratify=dataset.target,
    )
    engine, net = engine_from(ROOT / "01_digits/net.json")
    drive = np.zeros((len(test), net["n"]))
    drive[:, net["sets"]["input"]] = test * net["rule"]["clamp"]
    accuracy = float(
        (engine_outputs(engine, net, drive).argmax(axis=1) == labels).mean()
    )
    assert accuracy >= 0.95, f"digits: {accuracy}"
    sys.path.insert(0, str(ROOT / "04_pong"))
    from evaluate import ExportedPolicy, evaluate

    policy = ExportedPolicy(ROOT / "04_pong/net.json")
    pong = evaluate(policy, 81234, 500)
    assert policy.net["field"]["frames"] == 1 and policy.trace is not None
    assert pong["win_rate"] >= 0.85, f"pong: {pong}"
    # Read the exact exported C4 weights, with the same bounded planning as the page.
    sys.path.insert(0, str(ROOT / "03_connect_four"))
    spec = importlib.util.spec_from_file_location(
        "c4_performance", ROOT / "03_connect_four/train.py"
    )
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    c4_engine, c4_net = engine_from(ROOT / "03_connect_four/net.json")

    class Readout:
        def outputs(self, x):
            d = np.zeros((len(x), c4_net["n"]))
            d[:, c4_net["sets"]["input"]] = x * c4_net["rule"]["clamp"]
            return engine_outputs(c4_engine, c4_net, d)

    c4 = m.play_match(m.DeliberatingPlayer(Readout()), "random", 20, 81234)
    assert c4["wins"] >= 18, f"connect four: {c4}"
    print(
        json.dumps(
            {"digits_accuracy": accuracy, "pong": pong, "connect_four": c4}, indent=2
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
