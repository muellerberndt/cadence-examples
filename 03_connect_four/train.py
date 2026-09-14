"""03 Connect Four: a patch net learns a search's move choices and plays you in the browser.

Run:  python dataset.py              (once; self-play positions labelled by a depth-4 search)
      python train.py                 (a few minutes: selection, training, play-strength, receipt)
      python train.py --verify receipt.json

The net sees the board as 84 input owners (my discs, their discs), settles, and the most
active of 7 output owners is the column it plays. It learns with the free/nudged rule from
positions labelled by a shallow alpha-beta search; the receipt records how often it agrees
with that search on held-out positions and how it fares against fixed opponents, next to
an MLP trained on the same positions.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
import warnings
from pathlib import Path
from typing import Any

import numpy as np

import cadence as cd
from connect4 import COLS, Position, mirror_col, search
from dataset import DATA, load

HERE = Path(__file__).resolve().parent
SOURCES = [
    ("03_connect_four/train.py", Path(__file__).resolve()),
    ("03_connect_four/connect4.py", HERE / "connect4.py"),
    ("03_connect_four/dataset.py", HERE / "dataset.py"),
]
SOURCES += [(f"cadence/{p.name}", p) for p in sorted(Path(cd.__file__).parent.glob("*.py"))]

INPUTS, OUTPUTS = 84, COLS
GRID = [{"hidden": 64}, {"hidden": 128}]
SCHEDULE = {"epochs": 15, "batch": 64, "decay": 0.8, "eta": 3.0}
CONFIG = cd.LearnerConfig(
    beta=0.1, eta_bias=0.03, temperature=0.1, tolerance=3e-3, nudged_steps=12, free_steps=100
)
READOUT_TOLERANCE = 1e-4
GAMES_PER_OPPONENT = 100
OPENING_PLIES = 2  # every match game opens with one random ply per side, so no two games need repeat
CONTROL_EPOCHS = 5


def mirror_planes(x: np.ndarray) -> np.ndarray:
    """Flip both 6x7 planes left to right."""
    planes = x.reshape(len(x), 2, 6, 7)[:, :, :, ::-1]
    return planes.reshape(len(x), INPUTS)


def grouped_split(x: np.ndarray, fraction: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Keep a board and its reflection together, including duplicates from augmentation."""
    reflected = mirror_planes(x)
    groups = [min(row.tobytes(), mirror.tobytes()) for row, mirror in zip(x, reflected, strict=True)]
    unique = sorted(set(groups))
    order = np.random.default_rng(seed).permutation(len(unique))
    selected = {unique[i] for i in order[:int(fraction * len(unique))]}
    mask = np.asarray([key in selected for key in groups])
    return np.flatnonzero(mask), np.flatnonzero(~mask)


def prepare(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """Group board/reflection equivalents before the 90/10 split, then augment training."""
    x, y, meta = load()
    train, test = grouped_split(x, 0.9, seed)
    x_train = np.concatenate([x[train], mirror_planes(x[train])]).astype(float)
    y_train = np.concatenate([y[train], np.array([mirror_col(int(c)) for c in y[train]])]).astype(int)
    return x_train, y_train, x[test].astype(float), y[test].astype(int), meta


class Net:
    def __init__(self, hidden: int, eta: float, seed: int) -> None:
        self.wiring = cd.layered(INPUTS, hidden, OUTPUTS, density=1.0, seed=seed)
        self.learner = cd.Learner(
            cd.Settlement(self.wiring, cd.learning_rule(dt=1.0)),
            self.wiring.sets["output"],
            dataclasses.replace(CONFIG, eta=eta),
        )
        self.seed = seed
        self.steps: list[tuple[float, float]] = []

    def drive(self, x: np.ndarray) -> np.ndarray:
        return self.learner.engine.clamp_levels(np.pad(x, ((0, 0), (0, self.wiring.n - INPUTS))))

    def fit(self, x: np.ndarray, y: np.ndarray, *, epochs: int, batch: int, decay: float, eta: float) -> float:
        drive = self.drive(x)
        rng = np.random.default_rng(self.seed)
        learner = self.learner
        eta_bias = learner.config.eta_bias
        t0 = time.perf_counter()
        for epoch in range(epochs):
            learner.config = dataclasses.replace(
                learner.config, eta=eta * decay**epoch, eta_bias=eta_bias * decay**epoch
            )
            order = rng.permutation(len(y))
            for start in range(0, len(order), batch):
                idx = order[start : start + batch]
                _, report = learner.step(drive[idx], y[idx])
                self.steps.append((report["free_steps"], report["nudged_steps"]))
        return time.perf_counter() - t0

    def _readout(self):
        learner = self.learner
        learning = learner.config
        learner.config = dataclasses.replace(learning, tolerance=READOUT_TOLERANCE)
        return learning

    def outputs(self, x: np.ndarray) -> np.ndarray:
        """Output-owner activations after a free settlement, one row per position."""
        learning = self._readout()
        try:
            out = []
            for start in range(0, len(x), 512):
                free = self.learner.free(self.drive(x[start : start + 512]))
                out.append(free.activation[:, self.learner.output_index])
            return np.concatenate(out)
        finally:
            self.learner.config = learning

    def agreement(self, x: np.ndarray, y: np.ndarray) -> float:
        return float((self.outputs(x).argmax(axis=1) == y).mean())

    def move(self, pos: Position) -> int:
        scores = self.outputs(pos.planes()[None, :])[0]
        legal = pos.legal()
        return max(legal, key=lambda c: (scores[c], -abs(c - 3)))


class DeliberatingPlayer:
    """The learned policy breaks ties between depth-four counterfactual outcomes.

    Planning uses the supplied game rules. Its strength is reported separately from
    the raw network, and a search-only control receives the same planning budget.
    """
    def __init__(self, net: Net, depth: int = 4) -> None:
        self.net, self.depth = net, depth

    def move(self, pos: Position) -> int:
        _, values = search(pos, self.depth)
        scores = self.net.outputs(pos.planes()[None])[0]
        return max(values, key=lambda c: (values[c], scores[c], -abs(c - 3)))


def practice(net: Net, x: np.ndarray, y: np.ndarray, seed: int, games: int = 30, forbidden: np.ndarray | None = None) -> dict:
    """Play, ask the teacher about mistakes, and rehearse old examples alongside them."""
    rng = np.random.default_rng(seed)
    excluded = set() if forbidden is None else {row.tobytes() for row in np.concatenate([forbidden, mirror_planes(forbidden)])}
    frames, labels = [], []
    for game in range(games):
        pos = Position()
        mine = game % 2 == 0
        while not pos.terminal():
            if mine:
                action = net.move(pos)
                target = search(pos, 4)[0]
                if action != target and pos.planes().tobytes() not in excluded:
                    frames.append(pos.planes())
                    labels.append(target)
            else:
                action = search(pos, 2)[0]
            pos = pos.play(action)
            mine = not mine
    if frames:
        replay = rng.choice(len(y), min(len(y), max(512, len(labels))), replace=False)
        net.fit(np.concatenate([x[replay], frames]), np.concatenate([y[replay], labels]),
                epochs=3, batch=64, decay=.9, eta=.3)
    return {"games": games, "corrective_examples": len(labels),
            "rehearsal": "old training positions mixed with teacher corrections on visited positions"}


class MLPPlayer:
    """The sklearn baseline behind the same ``move`` interface."""

    def __init__(self, model: Any) -> None:
        self.model = model

    def move(self, pos: Position) -> int:
        scores = self.model.predict_proba(pos.planes()[None, :])[0]
        legal = pos.legal()
        return max(legal, key=lambda c: (scores[c], -abs(c - 3)))


def opponent_move(pos: Position, kind: str, rng: np.random.Generator) -> int:
    if kind == "random":
        return int(rng.choice(pos.legal()))
    depth = int(kind.split("-")[1])
    return search(pos, depth)[0]


def play_match(player: Any, kind: str, games: int, seed: int) -> dict[str, Any]:
    """``games`` games against an opponent, alternating who starts, from random two-ply openings.

    The net and the searches are deterministic, so without the random opening every game
    on the same side would be the same game.
    """
    rng = np.random.default_rng(seed)
    wins = draws = losses = 0
    lengths = []
    for g in range(games):
        pos = Position()
        for _ in range(OPENING_PLIES):
            pos = pos.play(int(rng.choice(pos.legal())))
        player_to_move = g % 2 == 0
        while not pos.terminal():
            col = player.move(pos) if player_to_move else opponent_move(pos, kind, rng)
            pos = pos.play(col)
            player_to_move = not player_to_move
        lengths.append(pos.plies)
        if pos.full() and not pos.just_won():
            draws += 1
        elif player_to_move:  # the side that just moved was the opponent
            losses += 1
        else:
            wins += 1
    return {
        "opponent": kind,
        "games": games,
        "opening_plies": OPENING_PLIES,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "score": (wins + 0.5 * draws) / games,
        "mean_plies": float(np.mean(lengths)),
    }


def baselines(x: np.ndarray, y: np.ndarray, x_test: np.ndarray, y_test: np.ndarray, hidden: int, seed: int) -> list[dict]:
    from sklearn.neural_network import MLPClassifier

    out = []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for epochs in (SCHEDULE["epochs"], 50):
            t0 = time.perf_counter()
            mlp = MLPClassifier((hidden,), batch_size=SCHEDULE["batch"], max_iter=epochs, random_state=seed).fit(x, y)
            seconds = time.perf_counter() - t0
            row = {
                "model": f"MLP {INPUTS}-{hidden}-{OUTPUTS} (adam, batch {SCHEDULE['batch']})",
                "parameters": INPUTS * hidden + hidden + hidden * OUTPUTS + OUTPUTS,
                "epochs": int(mlp.n_iter_),
                "seconds": seconds,
                "agreement": float(mlp.score(x_test, y_test)),
                "strength": [
                    play_match(MLPPlayer(mlp), kind, GAMES_PER_OPPONENT, seed)
                    for kind in ("random", "search-1", "search-2", "search-4")
                ],
            }
            out.append(row)
            print(f"baseline {row['model']} {epochs} epochs: agreement {row['agreement']:.3f} " + " ".join(f"{s['opponent']} {s['score']:.2f}" for s in row["strength"]))
    return out


def export(net: Net, path: Path, meta: dict) -> None:
    """The settled engine as JSON for the page: dense overlap matrix, biases, rule, sets."""
    engine = net.learner.engine
    rule = engine.rule
    payload = {
        "n": net.wiring.n,
        "sets": {k: [int(i) for i in v] for k, v in net.wiring.sets.items()},
        "W": [round(float(w), 5) for w in engine.dense().ravel()],
        "bias": [round(float(b), 5) for b in engine.bias],
        "rule": {"slope": rule.slope, "threshold": rule.threshold, "leak": rule.leak, "dt": rule.dt, "clamp": rule.clamp_amplitude, "rest": rule.rest_emission},
        "readout_tolerance": READOUT_TOLERANCE,
        "meta": meta,
    }
    path.write_text(json.dumps(payload, separators=(",", ":")))


def run(seed: int, out: Path) -> dict[str, Any]:
    x_train, y_train, x_test, y_test, meta = prepare(seed)
    print(f"{meta['positions']} positions (digest {meta['digest'][:16]}...); {len(y_train)} training rows after mirroring, {len(y_test)} test positions")

    # 1. Select the hidden size on a validation split of the training rows.
    rng = np.random.default_rng(seed)
    fit_idx, val_idx = grouped_split(x_train, 0.9, seed + 1)
    table = []
    for candidate in GRID:
        net = Net(candidate["hidden"], SCHEDULE["eta"], seed)
        seconds = net.fit(x_train[fit_idx], y_train[fit_idx], **SCHEDULE)
        table.append({**candidate, "parameters": net.learner.parameters(), "validation_agreement": net.agreement(x_train[val_idx], y_train[val_idx]), "seconds": seconds})
        print(f"validation: hidden {candidate['hidden']:3d} -> agreement {table[-1]['validation_agreement']:.3f} ({table[-1]['parameters']} parameters, {seconds:.0f}s)")
    best = max(table, key=lambda row: (row["validation_agreement"], -row["parameters"]))
    selected = {"hidden": best["hidden"]}

    # 2. Train the selected net on all training rows; read the test positions once.
    net = Net(selected["hidden"], SCHEDULE["eta"], seed)
    seconds = net.fit(x_train, y_train, **SCHEDULE)
    practice_report = practice(net, x_train, y_train, seed + 1000, forbidden=x_test)
    steps = np.asarray(net.steps).mean(axis=0)
    agreement = net.agreement(x_test, y_test)
    print(f"selected hidden {selected['hidden']}: test agreement {agreement:.3f} ({seconds:.0f}s, {net.learner.parameters()} parameters)")

    # 3. Play strength against fixed opponents, alternating who starts.
    strength = []
    for kind in ("random", "search-1", "search-2", "search-4"):
        strength.append(play_match(net, kind, GAMES_PER_OPPONENT, seed))
        print(f"vs {kind}: {strength[-1]['wins']}-{strength[-1]['draws']}-{strength[-1]['losses']} (score {strength[-1]['score']:.2f})")

    deployed = [play_match(DeliberatingPlayer(net), kind, GAMES_PER_OPPONENT, seed + 100)
                for kind in ("random", "search-2", "search-4")]
    class SearchOnly:
        def move(self, pos):
            return search(pos, 4)[0]
    planning_control = [play_match(SearchOnly(), kind, GAMES_PER_OPPONENT, seed + 100)
                        for kind in ("random", "search-2", "search-4")]
    print("with deliberation:", deployed, flush=True)
    # 4. Controls and conformance.
    shuffled = Net(selected["hidden"], SCHEDULE["eta"], seed)
    shuffled.fit(x_train, rng.permutation(y_train), **{**SCHEDULE, "epochs": CONTROL_EPOCHS})
    label_control = shuffled.agreement(x_test, y_test)
    conformance = cd.conformance(net.learner.engine, net.drive(x_test[:1])[0], steps=100)
    confusion = np.zeros((OUTPUTS, OUTPUTS), dtype=int)
    for truth, guess in zip(y_test, net.outputs(x_test).argmax(axis=1), strict=True):
        confusion[truth, guess] += 1

    comparison = baselines(x_train, y_train, x_test, y_test, selected["hidden"], seed)
    out.parent.mkdir(parents=True, exist_ok=True)
    export(net, out.parent / "net.json", {"agreement": agreement, "strength": strength, "deployed_strength": deployed, "parameters": net.learner.parameters()})
    net.learner.save(out.parent / "learner.npz")

    body = {
        "dataset": {"file": str(DATA.relative_to(HERE)), **meta, "train_rows_after_mirroring": int(len(y_train)), "test_positions": int(len(y_test)), "split_seed": seed},
        "grid": GRID,
        "schedule": SCHEDULE,
        "config": CONFIG.to_dict(),
        "readout_tolerance": READOUT_TOLERANCE,
        "validation": table,
        "selected": selected,
        "wiring": net.wiring.summary(),
        "rule": net.learner.engine.rule.to_dict(),
        "learner": net.learner.to_dict(),
        "training_seconds": seconds,
        "mean_free_steps": float(steps[0]),
        "mean_nudged_steps": float(steps[1]),
        "test_agreement": agreement,
        "confusion": confusion.tolist(),
        "strength": strength,
        "practice": practice_report, "deployed_strength": deployed, "search_only_strength": planning_control,
        "shuffled_label_control_agreement": label_control,
        "shuffled_label_control_epochs": CONTROL_EPOCHS,
        "conformance": conformance,
        "comparison": comparison,
        "boundary": {
            "learning_rule": "free/nudged contrastive Hebbian, centered, owner-local",
            "goal_enters_only_through_the_nudge": True,
            "teacher": "depth-4 alpha-beta with a threat-count heuristic",
            "deployed_player_uses_depth4_search_with_neural_tie_breaking": True,
            "raw_policy_and_search_only_controls_reported_separately": True,
            "selection_on_board_reflection_grouped_validation_only": True,
            "test_positions_read_once_after_selection": True,
            "strength_measured_by_play_against_fixed_opponents_alternating_first_move": True,
            "baselines_measured_here_on_the_same_positions": True,
        },
    }
    receipt = cd.Receipt.build("cadence-examples/03-connect-four/v1", body, sources=SOURCES)
    receipt.write(out)
    print(f"shuffled-label control {label_control:.3f}; conformance {conformance['max_abs_deviation']:.1e}; receipt {out} ({receipt.digest[:16]}...)")
    return body


def check(body: dict) -> str | None:
    confusion = np.asarray(body["confusion"])
    if confusion.sum() != body["dataset"]["test_positions"]:
        return "confusion matrix does not cover the test positions"
    if abs(np.trace(confusion) / confusion.sum() - body["test_agreement"]) > 1e-9:
        return "test agreement does not follow from the confusion matrix"
    for row in body["strength"]:
        if row["wins"] + row["draws"] + row["losses"] != row["games"]:
            return f"match record against {row['opponent']} does not add up"
        if abs((row["wins"] + 0.5 * row["draws"]) / row["games"] - row["score"]) > 1e-9:
            return f"score against {row['opponent']} does not follow from the record"
    best = max(body["validation"], key=lambda r: (r["validation_agreement"], -r["parameters"]))
    if {"hidden": best["hidden"]} != body["selected"]:
        return "selected configuration is not the best on validation"
    if not body["conformance"]["ledger"]["clean"]:
        return "reference ledger is not clean"
    if DATA.exists():
        _, _, meta = load()
        if meta["digest"] != body["dataset"]["digest"]:
            return "positions.npz on disk is not the dataset the receipt names"
    return None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=HERE / "receipt.json")
    parser.add_argument("--verify", type=Path)
    args = parser.parse_args()
    if args.verify:
        ok, message = cd.Receipt.verify(args.verify, sources=SOURCES, check=check)
        print(message)
        return 0 if ok else 1
    run(args.seed, args.output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
