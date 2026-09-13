"""Smoke runs must preserve already-modified source and artifact bytes."""
import importlib.util
import sys
from pathlib import Path

import cadence as cd
import pytest

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


@pytest.mark.parametrize("stage", ["import", "prepare"])
def test_smoke_reports_a_broken_rung_and_continues(stage, tmp_path, monkeypatch, capsys):
    for name in ("broken", "working"):
        rung = tmp_path / name
        rung.mkdir()
        source = "raise RuntimeError('broken import')\n" if name == "broken" and stage == "import" else ""
        (rung / "train.py").write_text(source)
    monkeypatch.setattr(module, "ROOT", tmp_path)

    def prepare(m):
        if stage == "prepare":
            raise RuntimeError("broken preparation")

    completed = []

    def run(m, output):
        completed.append(Path(m.__file__).parent.name)
        m.SOURCES = []
        cd.Receipt.build("smoke-test", {}).write(output)

    monkeypatch.setattr(module, "BUDGETS", {"broken": prepare, "working": lambda m: None})
    monkeypatch.setattr(module, "ARGS", {"broken": run, "working": run})
    monkeypatch.setattr(sys, "argv", ["smoke.py"])
    original_path = sys.path.copy()
    assert module.main() == 1
    assert completed == ["working"]
    assert "RuntimeError: broken" in capsys.readouterr().out
    assert sys.path == original_path


def test_smoke_rejects_unknown_rungs_before_copying(monkeypatch, capsys):
    monkeypatch.setattr(sys, "argv", ["smoke.py", "../outside"])
    with pytest.raises(SystemExit) as error:
        module.main()
    assert error.value.code == 2
    assert "unknown rung(s)" in capsys.readouterr().err
