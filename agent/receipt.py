"""Run receipts of an experience stage: sources, seeds, counts, metrics, predicates.

A receipt is written by the stage runner and checked by its verifier, which recomputes
every metric from the causal event log with code independent of the policy. The runner
fills source revisions and dirty-file hashes, package versions, data hashes, every
scheduled seed and every failed trial. ``status`` stays ``incomplete`` until the schedule
is complete; a verifier fails closed on it.
"""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "cadence-experience-run/v1"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def sha256_json(value: Any) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def git_state(repo: Path) -> dict[str, Any]:
    """Commit and the hashes of every dirty or untracked tracked-path file under ``repo``."""
    def run(*args: str) -> str:
        return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()
    try:
        commit = run("rev-parse", "HEAD")
        status = run("status", "--porcelain", "--untracked-files=all")
    except (subprocess.CalledProcessError, FileNotFoundError):
        return {"commit": None, "dirty": {}}
    dirty = {}
    for line in status.splitlines():
        name = line[3:].split(" -> ")[-1]
        path = repo / name
        if path.is_file() and not any(part in (".venv", "__pycache__", "runs") for part in path.parts):
            dirty[name] = sha256_file(path)
    return {"commit": commit, "dirty": dirty}


def sources(stage_dir: Path, core: Path | None = None) -> dict[str, Any]:
    core = core or Path(np.__file__).parent  # replaced below by the cadence source when importable
    try:
        import cadence

        core = Path(cadence.__file__).resolve().parent
    except ImportError:  # pragma: no cover
        pass
    stage_files = {p.name: sha256_file(p) for p in sorted(stage_dir.glob("*.py"))}
    stage_files["config.json"] = sha256_file(stage_dir / "config.json")
    core_files = {str(p.relative_to(core)): sha256_file(p) for p in sorted(core.rglob("*.py"))}
    return {
        "stage_dir": stage_dir.name,
        "stage_files": stage_files,
        "stage_repo": git_state(stage_dir.parent),
        "core": {"version": getattr(sys.modules.get("cadence"), "__version__", None), "files_sha256": sha256_json(core_files), "repo": git_state(core.parents[1])},
        "python": sys.version,
        "numpy": np.__version__,
        "platform": platform.platform(),
        "machine": platform.machine(),
    }


def _sanitize(value: Any) -> Any:
    """NaN and infinities become null; everything else passes through."""
    if isinstance(value, dict):
        return {k: _sanitize(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_sanitize(v) for v in value]
    if isinstance(value, (float, np.floating)):
        return None if not np.isfinite(value) else float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, np.ndarray):
        return _sanitize(value.tolist())
    return value


class Receipt:
    """Builds the receipt incrementally; ``write`` stores canonical JSON."""

    def __init__(self, stage: str, run_id: str, config: dict[str, Any], stage_dir: Path, out: Path) -> None:
        self.started = time.time()
        self.body: dict[str, Any] = {
            "schema": SCHEMA,
            "stage": stage,
            "run_id": run_id,
            "status": "incomplete",
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(self.started)),
            "sources": sources(stage_dir),
            "config_sha256": sha256_json(config),
            "config": config,
            "seeds": {},
            "data_manifest_sha256": None,
            "initialization": {"kind": "random", "parent_receipt": None},
            "supplied_components": [],
            "learned_components": [],
            "counts": {},
            "metrics": {},
            "controls": {},
            "numerics": {},
            "resources": {},
            "artifacts": {},
            "failures": [],
            "acceptance": {"spec_sha256": None, "predicates": []},
        }
        self.out = out
        out.mkdir(parents=True, exist_ok=True)

    def predicate(self, name: str, value: float, threshold: float, rule: str, passed: bool, **extra: Any) -> None:
        self.body["acceptance"]["predicates"].append({"name": name, "value": None if value is None or not np.isfinite(value) else float(value), "threshold": threshold, "aggregation": rule, "passed": bool(passed), **extra})

    def finish(self, complete: bool) -> Path:
        self.body["resources"]["wall_seconds"] = time.time() - self.started
        self.body["status"] = "complete" if complete else "incomplete"
        self.body["acceptance"]["passed"] = complete and all(p["passed"] for p in self.body["acceptance"]["predicates"])
        return self.write()

    def write(self) -> Path:
        path = self.out / "receipt.json"
        self.body.pop("receipt_sha256", None)
        self.body = _sanitize(self.body)  # a nonfinite reading becomes null: not measured, never a number
        text = json.dumps(self.body, sort_keys=True, indent=1, allow_nan=False)
        self.body["receipt_sha256"] = hashlib.sha256(text.encode()).hexdigest()
        path.write_text(json.dumps(self.body, sort_keys=True, indent=1, allow_nan=False) + "\n")
        return path


def peak_rss_mb() -> float:
    try:
        import resource

        usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return float(usage) / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except Exception:  # pragma: no cover
        return float("nan")


class EventLog:
    """Causal event log: one JSON line per real event, plus large arrays elsewhere."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0
        self.file = open(path, "w", encoding="utf-8")

    def write(self, record: dict[str, Any]) -> None:
        self.file.write(json.dumps(record, separators=(",", ":"), allow_nan=False) + "\n")
        self.count += 1

    def close(self) -> dict[str, Any]:
        self.file.close()
        return {"path": self.path.name, "events": self.count, "sha256": sha256_file(self.path)}
