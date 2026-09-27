"""The quickstart demos: tiny runs end to end, the brains they draw, and the page they serve."""

import json
import threading
import urllib.request

import numpy as np
import pytest

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quickstart import demo  # noqa: E402


@pytest.mark.parametrize(
    "make",
    [
        lambda: demo.StreamDemo(day_passes=2, night_passes=4),
        lambda: demo.DecideDemo(steps=3),
        lambda: demo.BodyDemo(batches=8, decisions=2),
    ],
)
def test_tiny_run_streams_frames_stats_and_curves_of_the_right_shape(make):
    run = make()
    atlas = json.loads(run.atlas_json())
    n = run._connectome.n
    assert atlas["n"] == n and len(run.weights()) == len(run._connectome.pre)
    result = run.run()
    view = run.snapshot()
    assert view["done"] and view["phase"] == "done" and view["result"] == result
    assert view["total"] == len(run.frames) > 0
    assert all(len(f["activity"]) == n for f in run.frames)
    assert all(len(f["heat"]) == n for f in run.frames if "heat" in f)
    assert all(np.isfinite(f["activity"]).all() for f in run.frames)
    assert view["curves"] and view["stats"]
    assert len(run.weights_json("weights")["weights"]) == len(run._connectome.pre)
    later = run.snapshot(since=view["total"] - 1)
    assert later["first"] == view["total"] - 1 and len(later["frames"]) == 1


def test_stream_demo_recalls_by_day_and_the_records_change_the_synapses():
    run = demo.StreamDemo(day_passes=8, night_passes=2)
    run.atlas_json()
    run.run()
    assert run.snapshot()["curves"]["records"][7][1] == 1.0  # the store recalls after the day
    change = run.weights_json("change")["weights"]
    assert len(change) == len(run._connectome.pre) and max(change) > 0


def test_server_serves_the_page_the_state_and_the_weights():
    run = demo.DecideDemo(steps=1)
    server = demo.serve(run, port=0, open_browser=False)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base = f"http://127.0.0.1:{server.server_address[1]}"
        page = urllib.request.urlopen(base + "/").read().decode()
        assert "class BrainScan" in page and '"name": "decide"' in page
        state = json.loads(urllib.request.urlopen(base + "/state?since=0").read())
        assert state["name"] == "decide" and "phase" in state
        weights = json.loads(urllib.request.urlopen(base + "/weights?what=change").read())
        assert weights["what"] == "change"
    finally:
        run.stop.set()
        server.shutdown()
        server.server_close()
