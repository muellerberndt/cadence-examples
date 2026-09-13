"""Smoke runs must preserve already-modified source and artifact bytes."""
import importlib.util
import sys
from pathlib import Path

import cadence as cd

PATH = Path(__file__).resolve().parents[1] / "tools/smoke.py"
spec = importlib.util.spec_from_file_location("smoke_examples", PATH)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_smoke_only_mutates_temporary_copy(tmp_path, monkeypatch):
    rung = tmp_path / "toy"
    rung.mkdir()
    (rung / "train.py").write_text("# already modified source\n")
    (rung / "net.json").write_text("already modified net\n")
    before = {p.name: p.read_bytes() for p in rung.iterdir()}
    monkeypatch.setattr(module, "ROOT", tmp_path)
    monkeypatch.setattr(module, "BUDGETS", {"toy": lambda m: None})

    def run(m, output):
        directory = Path(m.__file__).parent
        assert directory != rung
        (directory / "net.json").write_text("changed by smoke\n")
        (directory / "new.txt").write_text("temporary\n")
        m.SOURCES = []
        cd.Receipt.build("smoke-test", {}).write(output)

    monkeypatch.setattr(module, "ARGS", {"toy": run})
    monkeypatch.setattr(sys, "argv", ["smoke.py", "toy"])
    assert module.main() == 0
    assert {p.name: p.read_bytes() for p in rung.iterdir() if p.is_file()} == before
