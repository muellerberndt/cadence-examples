"""Verify an S03 run from its causal event logs, independently of the brain's code.

    python connect_four/verify.py runs/connect_four/pilot

Replays every logged game with a rule checker written for the verifier (gravity, alternation,
lines through the new stone, a full board) and recomputes the paired scores from the replayed
outcomes; judges every logged tactical probe with the same checker and recomputes the tactical
success, the strata balance and the suite digest; rebuilds the positions the candidate decided on
in training and checks that no probe is among them. The predicates that follow from these
numbers are recomputed; the validity, terminal, latency and cap predicates are checked against
their thresholds and per-seed values. The receipt digest, schema, seed schedule and the source,
config and event log digests are checked, and an incomplete run fails.
"""

from __future__ import annotations

import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DIRECTIONS = ((0, 1), (1, 0), (1, 1), (1, -1))
TOL = 1e-9
PAIRED = ("random", "one_ply", "minimax_256")
OLD = ("random", "one_ply")


class VerificationError(Exception):
    pass


class Rules:
    """Gravity by the lowest empty cell of a column; a line by scanning the windows through a cell."""

    def __init__(self, rows: int, cols: int, connect: int) -> None:
        self.rows, self.cols, self.connect = rows, cols, connect
        self.through: list[list[tuple[int, ...]]] = [[] for _ in range(rows * cols)]
        for r in range(rows):
            for c in range(cols):
                for dr, dc in DIRECTIONS:
                    spots = [(r + k * dr, c + k * dc) for k in range(connect)]
                    if all(0 <= rr < rows and 0 <= cc < cols for rr, cc in spots):
                        window = tuple(rr * cols + cc for rr, cc in spots)
                        for i in window:
                            self.through[i].append(window)

    def landing(self, cells: list[int], col: int) -> int | None:
        if not 0 <= col < self.cols:
            return None
        for r in range(self.rows):
            if cells[r * self.cols + col] == 0:
                return r * self.cols + col
        return None

    def completes(self, cells: list[int], index: int) -> bool:
        stone = cells[index]
        return stone != 0 and any(all(cells[i] == stone for i in w) for w in self.through[index])

    def any_line(self, cells: list[int]) -> bool:
        return any(self.completes(cells, i) for i in range(len(cells)) if cells[i])

    def winning_columns(self, cells: list[int], stone: int) -> list[int]:
        out = []
        for col in range(self.cols):
            i = self.landing(cells, col)
            if i is None:
                continue
            cells[i] = stone
            if self.completes(cells, i):
                out.append(col)
            cells[i] = 0
        return out

    def replay(self, moves: list[int]) -> tuple[int, list[bytes]]:
        """The first player's outcome (+1, 0, -1) of a finished legal game, and the key of the
        position before every move (cells then the player to move)."""
        cells = [0] * (self.rows * self.cols)
        keys = []
        for ply, col in enumerate(moves):
            i = self.landing(cells, int(col))
            if i is None:
                raise VerificationError(f"illegal move {col} at ply {ply}")
            keys.append(bytes(cells) + bytes([ply % 2]))
            cells[i] = ply % 2 + 1
            if self.completes(cells, i):
                if ply != len(moves) - 1:
                    raise VerificationError("moves continue after a line")
                return (1 if ply % 2 == 0 else -1), keys
        if any(v == 0 for v in cells):
            raise VerificationError("the game ends before a line or a full board")
        return 0, keys

    def judge(self, cells: list[int], to_move: int) -> tuple[str, set[int]]:
        """The kind of a tactical position and its correct columns, or raises when it is none."""
        stones = [sum(1 for v in cells if v == s) for s in (1, 2)]
        if stones[0] - stones[1] != to_move:
            raise VerificationError("stone counts do not match the player to move")
        for col in range(self.cols):
            filled = [cells[r * self.cols + col] != 0 for r in range(self.rows)]
            if any(filled[r + 1] and not filled[r] for r in range(self.rows - 1)):
                raise VerificationError("a floating stone")
        if self.any_line(cells) or all(v != 0 for v in cells):
            raise VerificationError("the position is over")
        mover, other = to_move + 1, 2 - to_move
        wins = self.winning_columns(cells, mover)
        if wins:
            return "win", set(wins)
        threats = self.winning_columns(cells, other)
        if len(threats) != 1:
            raise VerificationError("neither a win nor a single block")
        i = self.landing(cells, threats[0])
        cells[i] = mover
        try:
            if self.winning_columns(cells, other):
                raise VerificationError("the block leaves an immediate win")
        finally:
            cells[i] = 0
        return "block", set(threats)


def score(outcomes: list[int]) -> float:
    return float(np.mean((np.asarray(outcomes, float) + 1) / 2))


def close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= TOL


def check_log(path: Path, rules: Rules) -> dict:
    """Everything the verifier recomputes from one seed's log."""
    tactical: dict[str, list[dict]] = defaultdict(list)
    paired: dict[tuple[str, str, str], list[tuple[int, int]]] = defaultdict(list)
    training: dict[str, int] = defaultdict(int)
    decided: set[bytes] = set()
    for line in path.read_text().splitlines():
        e = json.loads(line)
        if e["phase"] == "tactical":
            cells = [".xo".index(ch) for ch in e["board"]]
            kind, correct = rules.judge(list(cells), int(e["to_move"]))
            if kind != e["kind"]:
                raise VerificationError(f"probe {e['index']} of {e['life']} is a {kind}, logged as {e['kind']}")
            tactical[e["life"]].append({"index": int(e["index"]), "key": bytes(cells) + bytes([int(e["to_move"])]), "kind": kind, "side": int(e["to_move"]),
                                        "hit": int(e["column"]) in correct})
            continue
        outcome, keys = rules.replay([int(c) for c in e["moves"]])
        candidate = outcome if e["first"] == "candidate" else -outcome
        if candidate != int(e["outcome"]):
            raise VerificationError(f"{e['life']} {e['phase']} game replays to {candidate}, logged {e['outcome']}")
        if "k" in e:
            paired[(e["life"], e["phase"], e["opponent"])].append((int(e["k"]), candidate))
        else:
            training[e["life"]] += 1
            if e["life"] == "candidate" and e["phase"] in ("explore", "mixture"):
                start = 0 if e["first"] == "candidate" else 1
                decided.update(keys[start::2])
    scores = {}
    for key, games in paired.items():
        games.sort()
        if [k for k, _ in games] != list(range(len(games))):
            raise VerificationError(f"paired games of {key} are not numbered 0..n-1")
        scores[key] = (score([o for _, o in games]), len(games))
    return {"tactical": dict(tactical), "paired": scores, "training": dict(training), "decided": decided}


def tactical_of(probes: list[dict], expected_n: int, digest: str, decided: set[bytes] | None) -> float:
    probes = sorted(probes, key=lambda p: p["index"])
    if [p["index"] for p in probes] != list(range(expected_n)):
        raise VerificationError(f"expected {expected_n} probes in order, found {len(probes)}")
    if hashlib.sha256(b"".join(p["key"] for p in probes)).hexdigest() != digest:
        raise VerificationError("the probes are not the suite the receipt names")
    strata = defaultdict(int)
    for p in probes:
        strata[(p["kind"], p["side"])] += 1
    if len(strata) != 4 or max(strata.values()) - min(strata.values()) > 1:
        raise VerificationError(f"the suite is not balanced: {dict(strata)}")
    if len({p["key"] for p in probes}) != len(probes):
        raise VerificationError("a probe position repeats")
    if decided is not None and any(p["key"] in decided for p in probes):
        raise VerificationError("a probe is a position the candidate decided on in training")
    return float(np.mean([p["hit"] for p in probes]))


def verify(run: Path) -> tuple[dict, list[str]]:
    path = run / "receipt.json"
    if not path.exists():
        raise VerificationError("no receipt")
    body = json.loads(path.read_text())
    claimed = body.pop("receipt_sha256", None)
    if claimed != hashlib.sha256(json.dumps(body, sort_keys=True, indent=1, allow_nan=False).encode()).hexdigest():
        raise VerificationError("receipt digest does not match its content")
    if body.get("schema") != "cadence-experience-run/v1" or body.get("stage") != "S03":
        raise VerificationError("unknown schema or stage")
    if body["status"] != "complete":
        raise VerificationError(f"run status is {body['status']}")
    if body["seeds"]["failed"] or set(body["seeds"]["completed"]) != set(body["seeds"]["scheduled"]):
        raise VerificationError("seed schedule incomplete")
    stage_dir = Path(__file__).resolve().parents[1] / body["sources"]["stage_dir"]
    for name, sha in body["sources"]["stage_files"].items():
        file = stage_dir / name
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != sha:
            raise VerificationError(f"source {name} changed since the run")
    config = body["config"]
    if hashlib.sha256(json.dumps(config, sort_keys=True, separators=(",", ":")).encode()).hexdigest() != body["config_sha256"]:
        raise VerificationError("config digest mismatch")
    rules = Rules(config["game"]["rows"], config["game"]["cols"], config["game"]["connect"])
    kind = body["run_kind"]
    per_seed = {int(r["seed"]): r for r in body["metrics"]["per_seed"]}
    if set(per_seed) != set(body["seeds"]["scheduled"]):
        raise VerificationError("per-seed metrics do not cover the schedule")
    for entry in body["artifacts"].get("checkpoints", []):
        for f in entry["files"]:
            file = run / f["path"]
            if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != f["sha256"]:
                raise VerificationError(f"checkpoint {f['path']} is missing or changed")
    if kind == "candidate":
        for seed in per_seed:
            labels = {label for entry in body["artifacts"]["checkpoints"] if entry["seed"] == seed for label in entry["labels"]}
            if not {"after_mixture", "final"} <= labels:
                raise VerificationError(f"seed {seed}: the mixture or the final checkpoint is missing")
    recomputed: dict[int, dict] = {}
    for artifact in body["artifacts"]["events"]:
        file = run / artifact["path"]
        if not file.exists() or hashlib.sha256(file.read_bytes()).hexdigest() != artifact["sha256"]:
            raise VerificationError(f"event log {artifact['path']} changed")
        seed = int(artifact["path"].split("seed")[1].split(".")[0])
        r = per_seed[seed]
        log = check_log(file, rules)
        n_probes = config["evaluation"]["tactical_positions"]
        games = r["budget"]["paired_games"]
        digest = r["suites"]["tactical_sha256"]

        def paired_score(life: str, phase: str, opponent: str, claimed: float, log=log, games=games, seed=seed) -> float:
            found = log["paired"].get((life, phase, opponent))
            if found is None or found[1] != games:
                raise VerificationError(f"seed {seed}: expected {games} paired games of {life} {phase} {opponent}")
            if not close(found[0], claimed):
                raise VerificationError(f"seed {seed}: {life} {phase} {opponent} recomputes to {found[0]}, receipt says {claimed}")
            return found[0]

        def tactical_score(life: str, claimed: float, decided=None, log=log, seed=seed, digest=digest, n=n_probes) -> float:
            value = tactical_of(log["tactical"].get(life, []), n, digest, decided)
            if not close(value, claimed):
                raise VerificationError(f"seed {seed}: {life} tactical success recomputes to {value}, receipt says {claimed}")
            return value

        if kind == "candidate":
            if len(log["decided"]) != r["suites"]["excluded_positions"]:
                raise VerificationError(f"seed {seed}: {len(log['decided'])} decided positions replay, receipt says {r['suites']['excluded_positions']}")
            if log["training"].get("candidate", 0) != r["games"]["total"]:
                raise VerificationError(f"seed {seed}: {log['training'].get('candidate', 0)} training games logged, receipt says {r['games']['total']}")
            out = {
                "tactical": tactical_score("candidate", r["tactical"]["success"], log["decided"]),
                "no_planning_tactical": tactical_score("no-planning", r["no_planning"]["tactical"]["success"], log["decided"]),
                "corrupted_tactical": tactical_score("corrupted", r["corrupted"]["tactical"]["success"], log["decided"]),
                "born_frozen_tactical": tactical_score("born-frozen", r["born_frozen"]["tactical"]["success"], log["decided"]),
            }
            for name in PAIRED:
                out[f"mixture_{name}"] = paired_score("candidate", "paired-mixture", name, r["paired_mixture"][name]["score"])
                out[f"adaptation_{name}"] = paired_score("candidate", "paired-adaptation", name, r["paired_adaptation"][name]["score"])
                out[f"no_planning_{name}"] = paired_score("no-planning", "paired-mixture", name, r["no_planning"]["paired"][name]["score"])
            for name in OLD:
                paired_score("born-frozen", "paired-mixture", name, r["born_frozen"]["paired"][name]["score"])
            if not close(out["mixture_one_ply"] - out["no_planning_one_ply"], r["planning_gain"]) or not close(out["tactical"] - out["corrupted_tactical"], r["corrupted_loss"]):
                raise VerificationError(f"seed {seed}: planning gain or corrupted loss does not recompute")
            for name in OLD:
                if not close(out[f"mixture_{name}"] - out[f"adaptation_{name}"], r["old_loss"][name]):
                    raise VerificationError(f"seed {seed}: the old loss on {name} does not recompute")
        else:
            out = {}
            for name, entry in r["controls"].items():
                out[f"{name}_tactical"] = tactical_score(name, entry["tactical"]["success"])
                for opponent in OLD:
                    out[f"{name}_{opponent}"] = paired_score(name, "paired-mixture", opponent, entry["paired"][opponent]["score"])
        recomputed[seed] = out
    if set(recomputed) != set(per_seed):
        raise VerificationError("an event log is missing")
    return body, check_predicates(body, recomputed)


def check_predicates(body: dict, recomputed: dict[int, dict]) -> list[str]:
    config, kind = body["config"], body["run_kind"]
    g = config["gates"]
    seeds = sorted(recomputed)
    values = {k: [recomputed[s][k] for s in seeds] for k in recomputed[seeds[0]]}
    if kind == "candidate":
        losses = {name: float(np.mean(np.array(values[f"mixture_{name}"]) - np.array(values[f"adaptation_{name}"]))) for name in OLD}
        expected = {
            "tactical_success": (float(np.mean(values["tactical"])), ">="),
            "paired_random": (float(np.mean(values["mixture_random"])), ">="),
            "paired_random_seed_floor": (float(np.min(values["mixture_random"])), ">="),
            "paired_one_ply": (float(np.mean(values["mixture_one_ply"])), ">="),
            "paired_one_ply_seed_floor": (float(np.min(values["mixture_one_ply"])), ">="),
            "planning_gain": (float(np.mean(np.array(values["mixture_one_ply"]) - np.array(values["no_planning_one_ply"]))), ">="),
            "corrupted_dynamics_loss": (float(np.mean(np.array(values["tactical"]) - np.array(values["corrupted_tactical"]))), ">="),
            "continual_old_loss": (max(losses.values()), "<="),
            "next_board_validity": (None, ">="), "terminal_prediction": (None, ">="), "latency_p95_ms": (None, "<="), "games_cap": (None, "<="),
        }
    else:
        minimax = [min(recomputed[s][f"minimax_{b}_tactical"] for b in config["controls"]["minimax_budgets"]) for s in seeds]
        expected = {
            "calibration_control_tactical": (float(np.min(minimax)), ">="),
            "calibration_random_tactical_ceiling": (float(np.max(values["random_tactical"])), "<="),
            "calibration_one_ply_beats_random": (float(np.min(values["one_ply_random"])), ">="),
            "calibration_random_vs_random_band": (float(np.max(np.abs(np.array(values["random_random"]) - 0.5))), "<="),
        }
    predicates = {p["name"]: p for p in body["acceptance"]["predicates"]}
    if set(predicates) != set(expected):
        raise VerificationError(f"predicates {sorted(predicates)} differ from the gates {sorted(expected)}")
    thresholds = {**g, "games_cap": body["metrics"]["per_seed"][0]["budget"]["games_cap"]}
    lines = []
    for name, (value, rule) in expected.items():
        p = predicates[name]
        if p["value"] is None or not np.isfinite(p["value"]):
            raise VerificationError(f"predicate {name} has no finite value")
        if not close(p["threshold"], thresholds[name]):
            raise VerificationError(f"predicate {name} threshold {p['threshold']} differs from the config's {thresholds[name]}")
        if value is not None and not close(value, p["value"]):
            raise VerificationError(f"predicate {name} recomputes to {value}, receipt says {p['value']}")
        if "per_seed" in p and value is None:
            aggregate = {"next_board_validity": np.mean, "terminal_prediction": np.mean, "latency_p95_ms": np.max, "games_cap": np.max}[name]
            if not close(aggregate(list(p["per_seed"].values())), p["value"]):
                raise VerificationError(f"predicate {name} does not aggregate its per-seed values")
        passed = p["value"] >= p["threshold"] if rule == ">=" else p["value"] <= p["threshold"]
        if bool(passed) != bool(p["passed"]):
            raise VerificationError(f"predicate {name} pass flag disagrees with its value")
        lines.append(f"  {'PASS' if passed else 'FAIL'} {name}: {p['value']:.4f} {rule} {p['threshold']}")
    if bool(body["acceptance"]["passed"]) != all(p["passed"] for p in predicates.values()):
        raise VerificationError("the acceptance flag disagrees with the predicates")
    return lines


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    try:
        body, lines = verify(Path(argv[1]))
    except (VerificationError, KeyError, ValueError) as error:
        print("FAIL:", error)
        return 1
    passed = body["acceptance"]["passed"]
    print(f"receipt {body['run_id']}: {len(body['seeds']['completed'])} seeds, {len(lines)} predicates recomputed or checked, {'all passed' if passed else 'some failed'}")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
