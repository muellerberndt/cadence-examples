"""The whole-brain viewer of the Cadence examples: the atlas that lays a connectome out and
the browser component that draws it. Every page in this repository, and any page of another
lane, takes the viewer from here; the library ships the brains alone."""

from .atlas import PALETTE, Atlas, Region, atlas_of, brain_scan_script, build_atlas, role_of

__all__ = ["Atlas", "Region", "PALETTE", "atlas_of", "brain_scan_script", "build_atlas", "role_of"]
