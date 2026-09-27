"""The brain atlas: one integrated layout, region roles, synapses, frames, and the shipped renderer."""

from __future__ import annotations

import base64
import json

import numpy as np
import pytest

import cadence as cd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from viewer.atlas import PALETTE, Atlas, atlas_of, brain_scan_script, build_atlas, role_of  # noqa: E402


def test_roles_follow_names_and_overrides() -> None:
    assert role_of("retina") == "vision"
    assert role_of("afterglow") == "memory"
    assert role_of("association") == "association"
    assert role_of("motor") == "motor"
    assert role_of("value") == "value"
    assert role_of("hidden") == "association"
    assert role_of("input") == "sensory"
    assert role_of("xyz") == "other"
    assert role_of("xyz", {"xyz": "vision"}) == "vision"
    assert set(PALETTE) >= {"vision", "sensory", "memory", "association", "motor", "value", "other"}


def test_every_neuron_lands_in_one_region_inside_the_square() -> None:
    connectome = cd.layered(6, 12, 3, density=1.0, seed=0)
    atlas = build_atlas(connectome)
    assert atlas.n == connectome.n
    assert atlas.positions.shape == (connectome.n, 2)
    assert np.all(np.abs(atlas.positions) <= 1.0)
    counts = sum(len(r.indices) for r in atlas.regions)
    assert counts == connectome.n
    assert sorted(r.name for r in atlas.regions) == ["hidden", "input", "output"]
    assert atlas.synapses == connectome.synapses
    for r in atlas.regions:
        inside = np.hypot(
            (atlas.positions[r.indices, 0] - r.center[0]) / r.extent[0],
            (atlas.positions[r.indices, 1] - r.center[1]) / r.extent[1],
        )
        assert np.all(inside <= 1.05)


def test_layout_is_deterministic_and_regions_do_not_overlap() -> None:
    connectome = cd.layered(10, 20, 4, density=0.5, seed=3)
    a = build_atlas(connectome, seed=7)
    b = build_atlas(connectome, seed=7)
    assert np.array_equal(a.positions, b.positions)
    for i, r in enumerate(a.regions):
        for s in a.regions[i + 1 :]:
            gap = np.hypot(r.center[0] - s.center[0], r.center[1] - s.center[1])
            assert gap >= 0.8 * (r.radius + s.radius)


def test_sheet_shapes_are_grids_and_supplied_positions_are_kept_in_order() -> None:
    connectome = cd.layered(12, 5, 2, density=1.0, seed=1)
    atlas = build_atlas(connectome, shapes={"input": (3, 4)})
    region = next(r for r in atlas.regions if r.name == "input")
    grid = atlas.positions[region.indices]
    assert np.allclose(grid[0, 1], grid[1, 1])  # same row
    assert grid[0, 0] < grid[1, 0] < grid[2, 0] < grid[3, 0]  # columns left to right
    assert grid[0, 1] > grid[4, 1] > grid[8, 1]  # rows top to bottom
    given = np.stack([np.arange(5.0), np.zeros(5)], axis=1)
    atlas2 = build_atlas(connectome, positions={"hidden": given})
    hidden = next(r for r in atlas2.regions if r.name == "hidden")
    xs = atlas2.positions[hidden.indices, 0]
    assert np.all(np.diff(xs) > 0)


def test_atlas_of_a_brain_uses_effective_weights_and_partitions_overlapping_populations() -> None:
    connectome = cd.layered(4, 6, 2, density=1.0, seed=0).with_populations(cortex=range(4, 12))
    brain = cd.Brain(connectome, cd.learning_neuron_model())
    atlas = atlas_of(brain)
    assert np.allclose(atlas.weight, brain.weights.astype(np.float32))
    names = [r.name for r in atlas.regions]
    assert "cortex" in names
    assert sum(len(r.indices) for r in atlas.regions) == connectome.n


def test_payload_round_trips_and_frames_quantise() -> None:
    connectome = cd.layered(5, 7, 3, density=1.0, seed=2)
    atlas = build_atlas(connectome)
    payload = json.loads(atlas.to_json())
    assert payload["format"] == "cadence.atlas/v1"
    raw = base64.b64decode(payload["positions"]["b64"])
    back = np.frombuffer(raw, dtype="<f4").reshape(payload["positions"]["shape"])
    assert np.array_equal(back, atlas.positions)
    pre = np.frombuffer(base64.b64decode(payload["pre"]["b64"]), dtype="<u4")
    assert np.array_equal(pre, atlas.pre)
    steps = np.linspace(-1.0, 1.0, atlas.n)[None, :] * np.ones((4, 1))
    frames = atlas.frames(steps, potential=steps * 3)
    act = np.frombuffer(base64.b64decode(frames["activation"]["b64"]), dtype="u1").reshape(
        4, atlas.n
    )
    assert act[0, 0] == 0 and act[0, -1] == 255
    assert frames["potential_span"] == pytest.approx(3.0)
    with pytest.raises(ValueError):
        atlas.frames(np.zeros((2, atlas.n + 1)))


def test_subsample_keeps_the_strongest_synapses_preferentially() -> None:
    connectome = cd.layered(20, 30, 5, density=1.0, seed=4)
    atlas = build_atlas(connectome)
    small = atlas.subsample_edges(200, seed=0)
    assert small.synapses == 200
    assert small.extras["synapses_total"] == atlas.synapses
    assert np.mean(np.abs(small.weight)) >= np.mean(np.abs(atlas.weight))


def test_frames_from_a_recording_and_the_shipped_renderer() -> None:
    connectome = cd.layered(4, 6, 2, density=1.0, seed=0)
    brain = cd.Brain(connectome, cd.learning_neuron_model())
    records: list = []
    drive = np.zeros((1, connectome.n))
    drive[0, :4] = 0.7
    with cd.record_settlements(records.append, label="probe"):
        brain.settle_batch(drive, steps=12, tolerance=None)
    atlas = atlas_of(brain)
    frames = atlas.frames_from_record(records[0])
    assert frames["steps"] == records[0].activation.shape[0]
    script = brain_scan_script()
    assert "export class BrainScan" in script and "cadence.brain-scan/v4" in script
    page = atlas.page(frames=frames, title="probe")
    assert "<canvas" in page and "cadence.atlas/v1" in page and "BrainScan" in page


def test_atlas_page_can_settle_live_with_an_embedded_brain() -> None:
    connectome = cd.layered(3, 4, 2, density=1.0, seed=0)
    brain = cd.Brain(connectome, cd.learning_neuron_model())
    atlas = atlas_of(brain)
    page = atlas.page(brain=brain, title="live")
    assert '"W"' in page and '"neuron"' in page and "Detune" in page
    assert isinstance(atlas, Atlas)


def test_a_shared_anatomical_frame_is_kept_as_given() -> None:
    connectome = cd.layered(4, 6, 2, density=1.0, seed=0)
    given = np.stack([np.arange(12.0), (np.arange(12.0) % 3) * 10.0], axis=1)
    atlas = build_atlas(connectome, positions={"*": given})
    xs = atlas.positions[:, 0]
    assert np.all(np.diff(xs) > 0)  # order along x kept
    span = atlas.positions.max(axis=0) - atlas.positions.min(axis=0)
    assert np.isclose(span.max(), 1.84, atol=1e-5)  # the larger side fills the square
    assert np.isclose(span[0] / span[1], 11 / 20, atol=1e-3)  # the aspect is kept
    with pytest.raises(ValueError):
        build_atlas(connectome, positions={"*": given[:5]})


def test_the_viewer_allocates_its_targets_on_its_first_draw() -> None:
    # A page that builds a second viewer on a canvas the first one already sized, or swaps
    # brains with setAtlas, must still get its cache and tissue-field buffers: the guard is
    # the instance's own flag, not the canvas size.
    script = brain_scan_script()
    assert "this.sized = false;" in script
    assert "if (!this.sized || this.canvas.width !== width || this.canvas.height !== height)" in script
    assert "this.sized = true;" in script


def test_the_viewer_ships_the_brain_style_with_its_static_geometry_uploaded_once() -> None:
    script = brain_scan_script()
    assert 'style: "scan"' in script  # the default: pages that do not ask keep the scan
    assert '"brain"' in script and "setStyle(style)" in script
    for name in (
        "BRAIN_VERTEX",
        "BRAIN_FRAGMENT",
        "function brainLayout",
        "function edgeBows",
        "function brainShell",
        "function lobeOf",
    ):
        assert name in script
    # the lobes every region is assigned to by its role and name
    for lobe in ("occipital", "hippocampus", "cortex", "motor", "prefrontal", "temporal"):
        assert f"{lobe}:" in script
    # the static geometry (the 3D positions, the bow of every synapse, the shell mesh) is
    # uploaded once per atlas; a frame evaluates the curves in the vertex shader
    assert "_uploadBrain()" in script and "STATIC_DRAW" in script
    frame = script[script.index("_drawBrain(width, height, dpr) {") : script.index("_time(t0) {")]
    for rebuilt in ("_uploadBrain", "brainShell", "edgeBows", "brainLayout", "bufferData"):
        assert rebuilt not in frame
    assert "drawArraysInstanced(gl.TRIANGLE_STRIP" in frame
    for option in ("restAlpha", "bloom", "shell", "spin", "spinRate"):
        assert f"{option}:" in script


def test_the_viewer_draws_a_measured_anatomy_when_the_atlas_carries_one() -> None:
    # An atlas with `positions3` (a measured soma position per neuron) is drawn at those
    # positions in the brain style instead of the generic lobes; the payload keys are
    # optional, so every earlier payload still draws as before.
    script = brain_scan_script()
    for name in ("function fitPositions3", "function spacing3Of", "function anatomyLayout", "ANATOMY_VIEW"):
        assert name in script
    assert "positions3: atlas.positions3 ? decodeArray(atlas.positions3) : null" in script
    assert "spacing3: atlas.spacing3 ? decodeArray(atlas.spacing3) : null" in script
    assert "this.anatomical ? anatomyLayout(this.atlas) : brainLayout(this.atlas)" in script
    assert "positions3 = {}" in script and "positions3['*']" in script
    assert 'if (!("shell" in this.given)) this.options.shell = !this.anatomical;' in script
    frame = script[script.index("_drawBrain(width, height, dpr) {") : script.index("_time(t0) {")]
    assert "anatomyLayout" not in frame and "spacing3Of" not in frame
