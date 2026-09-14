"""Local, persistent lessons for the game pages. No training service or data leaves this machine."""

from __future__ import annotations

import dataclasses
import hashlib
import json
import threading
from pathlib import Path

import cadence as cd
import numpy as np

ROOT = Path(__file__).resolve().parent
GAMES = {"pong": "04_pong", "connect-four": "03_connect_four"}


class Lessons:
    """Small local updates, teacher corrections, rehearsal, then a retained checkpoint."""

    def __init__(self, game: str, root: Path = ROOT) -> None:
        folder = root / GAMES[game]
        self.game = game
        self.template = json.loads((folder / "net.json").read_text())
        version = hashlib.sha256((folder / "learner.npz").read_bytes()).hexdigest()[:12]
        self.directory = root / "runs" / "learning" / f"{game}-{version}"
        self.directory.mkdir(parents=True, exist_ok=True)
        self.path = self.directory / "learner.npz"
        self.learner = cd.Learner.load(
            self.path if self.path.exists() else folder / "learner.npz", backend="cpu"
        )
        with np.load(folder / "rehearsal.npz", allow_pickle=False) as replay:
            self.x, self.y = replay["x"], replay["y"]
        self.lock = threading.RLock()
        self.log = self.directory / "lessons.jsonl"
        self.games = sum(1 for _ in self.log.open()) if self.log.exists() else 0

    def model(self) -> dict:
        # Readers see only a completed, retained update. teach() also calls this.
        with self.lock:
            net = dict(self.template)
            engine = self.learner.engine
            net["W"] = engine.dense().ravel().tolist()
            net["bias"] = engine.bias.tolist()
            return {"enabled": True, "games": self.games, "net": net}

    def teach(self, data: dict) -> dict:
        if not isinstance(data, dict):
            raise TypeError("episode must be an object")
        x = np.asarray(data.get("drives"), dtype=float)
        y = np.asarray(data.get("targets"))
        n = self.learner.engine.wiring.n
        limit = 400 if self.game == "pong" else 42
        if (
            x.ndim != 2
            or x.shape[1] != n
            or not 1 <= len(x) <= limit
            or not np.isfinite(x).all()
            or np.max(np.abs(x)) > 100
        ):
            raise ValueError("invalid episode drives")
        if (
            y.shape != (len(x),)
            or not np.issubdtype(y.dtype, np.integer)
            or np.any(y < 0)
            or np.any(y >= len(self.learner.output_index))
        ):
            raise ValueError("invalid teacher targets")
        actions = np.asarray(data.get("actions", y))
        rewards = np.asarray(data.get("rewards", np.zeros(len(y))), dtype=float)
        if (
            actions.shape != y.shape
            or not np.issubdtype(actions.dtype, np.integer)
            or np.any(actions < 0)
            or np.any(actions >= len(self.learner.output_index))
        ):
            raise ValueError("invalid actions")
        if (
            rewards.shape != y.shape
            or not np.isfinite(rewards).all()
            or np.max(np.abs(rewards)) > 10
        ):
            raise ValueError("invalid rewards")
        with self.lock:
            learner = self.learner
            old_engine, old_config = learner.engine, learner.config
            before = learner.accuracy(self.x, self.y)
            # Each game remains a reusable lesson, even if its first proposed update is rejected.
            episode = self.directory / f"episode-{self.games + 1:06d}.npz"
            np.savez_compressed(episode, x=x, y=y, actions=actions, rewards=rewards)
            if self.game == "pong":
                advantage = np.zeros(len(x))
                running = 0.0
                for t in reversed(range(len(x))):
                    running = rewards[t] + 0.97 * running
                    advantage[t] = running
                advantage = np.clip(advantage - advantage.mean(), -2, 2)
                learner.config = dataclasses.replace(
                    old_config, eta=0.003, eta_bias=0.00003, momentum=0, normalize=0
                )
                learner.step(x, actions, weight=advantage)
            # Revisit the teacher on new positions, interleaved with prior skills.
            learner.config = dataclasses.replace(
                old_config, eta=0.1, eta_bias=0.001, momentum=0, normalize=0
            )
            for start in range(0, len(y), 64):
                learner.step(x[start : start + 64], y[start : start + 64])
            replay_x, replay_y = [self.x], [self.y]
            for previous in sorted(self.directory.glob("episode-*.npz"))[-8:]:
                with np.load(previous, allow_pickle=False) as old:
                    pick = np.linspace(
                        0, len(old["y"]) - 1, min(32, len(old["y"])), dtype=int
                    )
                    replay_x.append(old["x"][pick])
                    replay_y.append(old["y"][pick])
            learner.step(np.concatenate(replay_x), np.concatenate(replay_y))
            after = learner.accuracy(self.x, self.y)
            accepted = (
                after >= before
                and np.isfinite(learner.engine.dense()).all()
                and np.isfinite(learner.engine.bias).all()
            )
            if not accepted:
                learner.engine = old_engine
            learner.config = old_config
            self.games += 1
            temporary = self.directory / "pending.npz"
            learner.save(temporary)
            temporary.replace(self.path)
            row = {
                "game": self.games,
                "rows": len(x),
                "retained": bool(accepted),
                "rehearsal_before": before,
                "rehearsal_after": after,
                "episode": episode.name,
            }
            with self.log.open("a") as f:
                f.write(json.dumps(row) + "\n")
            return {**self.model(), "lesson": row}
