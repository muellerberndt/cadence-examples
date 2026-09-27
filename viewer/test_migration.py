"""The extracted viewer retains runnable docs and names its own source repository."""
from hashlib import sha256
from pathlib import Path
import re
import runpy
import sys

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


@pytest.mark.parametrize("name", ["README.md", "reference.md"])
def test_viewer_documentation_python_runs(name, tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    scope = {"__name__": "viewer_documentation"}
    for code in re.findall(r"```python\n(.*?)```", (ROOT / "viewer" / name).read_text(), re.S):
        exec(compile(code, name, "exec"), scope)
    assert scope["atlas"].n == scope["brain"].connectome.n


def test_export_names_viewer_repository_not_core(monkeypatch):
    module = runpy.run_path(str(ROOT / "dozing-cat/export.py"))
    info = module["viewer_info"]
    calls = []

    def git(repo, *args):
        calls.append((Path(repo), args))
        return "examples-commit" if args[0] == "rev-parse" else ""

    monkeypatch.setitem(info.__globals__, "git", git)
    text = module["brain_scan_script"]()
    result = info(text)
    assert result["repository"] == "cadence-examples"
    assert result["source"] == "viewer/brain_scan.js"
    assert result["sha256"] == sha256(text.encode()).hexdigest()
    assert result["commit"] == "examples-commit"
    assert result["uncommitted_changes"] is False
    assert calls and all(repo == ROOT for repo, _ in calls)
    assert (ROOT / result["source"]).read_text() == text
