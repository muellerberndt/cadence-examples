"""The fly's nervous system as one Cadence brain, with its flight populations named.

The connectome is the BANC fixture (``fruitfly.banc``); the neuron model is the library's
graded rate model; the populations below are the declared dictionary between the release's
annotations and the names the protocols, the senses and the muscles speak in.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from cadence import Brain, Connectome, NeuronModel

from .banc import DEFAULT_ROOT, Neurons, load_fixture

__all__ = ["FlyNet", "load_fly", "log_gain_for", "GAIN", "CLASS_LOG_GAIN", "STEERING", "POWER", "TENSION", "WING_MUSCLES"]

# The dictionary's numbers. GAIN is the one global gain, selected on two physiology facts of the
# steering circuit (tools/reflex.py, receipts/g2_reflex_facts.json). CLASS_LOG_GAIN is the gain of
# one cell class relative to it, selected by protocol on the animal's facts about the code downstream
# (tools/class_gains.py, receipts/class_gains.json): the antennal lobe's local neurons, whose mutual
# excitation ignites the lobe at the global gain so that every odour makes the same Kenyon cell code.
GAIN = 0.02
CLASS_LOG_GAIN: dict[str, float] = {"ln": -3.0}


def log_gain_for(connectome: Connectome, class_log_gain: dict[str, float] | None = None) -> np.ndarray:
    """The per-neuron log gain of a connectome (the whole brain or a sub-net with its populations) from the class gains."""
    gains = CLASS_LOG_GAIN if class_log_gain is None else class_log_gain
    out = np.zeros(connectome.n)
    for name, value in gains.items():
        out[list(connectome.populations.get(name, ()))] = value
    return out

# Wing muscles by group, as the release names their motor neurons (cell_type).
STEERING: tuple[str, ...] = ("b1", "b2", "b3", "i1", "i2", "iii1", "iii3", "iii4", "iv1", "iv2", "iv3", "iv4", "tp1", "tp2", "tpn")
POWER: tuple[str, ...] = ("DLM1-4", "DLM5", "DVM1a-c", "DVM2a-b", "DVM3a-b")
TENSION: tuple[str, ...] = ("PS1", "PS2", "PSn_u", "MNxm01")
WING_MUSCLES: tuple[str, ...] = STEERING + POWER + TENSION
SIDES: tuple[str, ...] = ("left", "right")

HALTERE_AFFERENT = ("haltere_campaniform_sensillum_neuron", "haltere_chordotonal_organ_neuron")
WING_AFFERENT = (
    "wing_base_campaniform_sensillum_neuron", "wing_tegula_campaniform_sensillum_neuron",
    "wing_campaniform_sensillum_neuron", "wing_base_chordotonal_organ_neuron",
    "wing_tegula_chordotonal_organ_neuron", "wing_tegula_hair_plate_neuron",
)
# Descending neurons the senses and the page name; every type of the release is also a set.
NAMED_DN = ("DNp01", "DNa02", "DNa01", "DNa03", "DNa08", "DNb01", "DNp03", "DNp09", "DNp31", "MDN", "DNp06", "DNp07", "DNp10", "DNp11", "DNg62", "DNge078")


def populations(neurons: Neurons) -> dict[str, tuple[int, ...]]:
    F = neurons.fields
    where = neurons.where
    sets: dict[str, list[int] | np.ndarray] = {}

    def put(name: str, idx: np.ndarray) -> None:
        if len(idx):
            sets[name] = idx

    for side in SIDES:
        put(f"haltere:{side}", where(cell_sub_class=HALTERE_AFFERENT, side=side))
        put(f"wing_sense:{side}", where(cell_sub_class=WING_AFFERENT, side=side))
        for muscle in WING_MUSCLES:
            put(f"mn:wing:{muscle}:{side}", where(cell_class="wing_motor_neuron", cell_type=muscle, side=side))
        put(f"steering:{side}", where(cell_class="wing_motor_neuron", cell_type=STEERING, side=side))
        put(f"power:{side}", where(cell_class="wing_motor_neuron", cell_type=POWER, side=side))
        put(f"tension:{side}", where(cell_class="wing_motor_neuron", cell_type=TENSION, side=side))
        put(f"mn:haltere:{side}", where(cell_class="haltere_motor_neuron", side=side))
        put(f"neck_mn:{side}", where(cell_function="neck_motor", side=side))
        put(f"lptc:hs:{side}", where(cell_type=("HSE", "HSN", "HSS"), side=side))
        put(f"lptc:vs:{side}", np.flatnonzero(np.char.startswith(F["cell_type"].astype(str), "VS") & (F["region"] == "optic_lobe") & (F["side"] == side)))
        put(f"ocelli:{side}", where(cell_class="ocellar_projection_neuron", side=side))
        put(f"mn:ttm:{side}", where(peripheral_target_type="tergotrochanter_extensor_muscle", side=side))
        for jo in ("JO-A", "JO-B", "JO-C", "JO-D", "JO-E", "JO-F"):
            put(f"jo:{jo[3:]}:{side}", where(cell_type=jo, side=side))
    put("haltere", where(cell_sub_class=HALTERE_AFFERENT))
    put("wing_sense", where(cell_sub_class=WING_AFFERENT))
    put("steering", where(cell_class="wing_motor_neuron", cell_type=STEERING))
    put("power", where(cell_class="wing_motor_neuron", cell_type=POWER))
    put("power:dlm", where(cell_class="wing_motor_neuron", cell_type=("DLM1-4", "DLM5")))
    put("power:dvm", where(cell_class="wing_motor_neuron", cell_type=("DVM1a-c", "DVM2a-b", "DVM3a-b")))
    put("tension", where(cell_class="wing_motor_neuron", cell_type=TENSION))
    put("mn:haltere", where(cell_class="haltere_motor_neuron"))
    put("neck_mn", where(cell_function="neck_motor"))
    put("ocelli", where(cell_class="ocellar_projection_neuron"))
    put("lptc:hs", where(cell_type=("HSE", "HSN", "HSS")))
    put("lptc:vs", np.flatnonzero(np.char.startswith(F["cell_type"].astype(str), "VS") & (F["region"] == "optic_lobe")))
    put("mn:ttm", where(peripheral_target_type="tergotrochanter_extensor_muscle"))
    put("dn", where(super_class="descending"))
    put("an", where(super_class="ascending"))
    for t in NAMED_DN:
        put(f"dn:{t}", where(cell_type=t))
        for side in SIDES:
            put(f"dn:{t}:{side}", where(cell_type=t, side=side))
    put("gf", where(cell_type="DNp01"))
    put("kc", np.flatnonzero(np.char.find(F["cell_type"].astype(str), "KC") >= 0))
    put("dan:pam", np.flatnonzero(np.char.startswith(F["cell_type"].astype(str), "PAM")))
    put("dan:ppl1", np.flatnonzero(np.char.startswith(F["cell_type"].astype(str), "PPL1")))
    put("mbon", np.flatnonzero(np.char.startswith(F["cell_type"].astype(str), "MBON")))
    # The mushroom body output neurons by type and side, and by the transmitter rule of Aso et
    # al. 2014 (eLife 3:e04580): every MBON whose activation repels the fly is glutamatergic,
    # every MBON whose activation attracts it is GABAergic or cholinergic. The rule is declared
    # here as two populations; the predicted transmitter of the release decides the membership.
    types = F["cell_type"].astype(str)
    transmitter = F["neurotransmitter_predicted"].astype(str)
    mbon_mask = np.char.startswith(types, "MBON")
    for t in sorted(set(types[mbon_mask])):
        put(f"mbon:{t}", np.flatnonzero(mbon_mask & (types == t)))
        for side in SIDES:
            put(f"mbon:{t}:{side}", np.flatnonzero(mbon_mask & (types == t) & (F["side"] == side)))
    put("mbon:approach", np.flatnonzero(mbon_mask & np.isin(transmitter, ["gaba", "acetylcholine"])))
    put("mbon:avoid", np.flatnonzero(mbon_mask & (transmitter == "glutamate")))
    put("dan", np.flatnonzero(np.char.startswith(types, "PAM") | np.char.startswith(types, "PPL1")))
    put("apl", where(cell_type="APL"))
    put("pn", where(cell_class="antennal_lobe_projection_neuron"))
    put("ln", where(cell_class="antennal_lobe_local_neuron"))
    put("ln:acetylcholine", where(cell_class="antennal_lobe_local_neuron", neurotransmitter_predicted="acetylcholine"))
    put("ln:gaba", where(cell_class="antennal_lobe_local_neuron", neurotransmitter_predicted="gaba"))
    put("ln:glutamate", where(cell_class="antennal_lobe_local_neuron", neurotransmitter_predicted="glutamate"))
    put("orn", where(cell_class="olfactory_receptor_neuron"))
    detail = F["cell_function_detailed"].astype(str)
    orn_mask = F["cell_class"] == "olfactory_receptor_neuron"
    for odour in ("decaying_fruit_volatile", "fruity_volatile", "yeasty_volatile", "alcoholic_fermentation_volatile", "aversive_volatile", "carbon_dioxide_volatile", "pheromone_volatile"):
        for side in SIDES:
            put(f"orn:{odour.replace('_volatile', '')}:{side}", np.flatnonzero(orn_mask & (detail == odour) & (F["side"] == side)))
        put(f"orn:{odour.replace('_volatile', '')}", np.flatnonzero(orn_mask & (detail == odour)))
    gust = F["cell_function"] == "gustatory"
    for taste, needle in (("sugar", "sugar"), ("bitter", "bitter"), ("water", "water")):
        for part in ("labellum", "front_leg", "middle_leg", "hind_leg"):
            put(f"grn:{taste}:{part}", np.flatnonzero(gust & (np.char.find(detail, needle) >= 0) & (F["body_part_sensory"] == part)))
        put(f"grn:{taste}", np.flatnonzero(gust & (np.char.find(detail, needle) >= 0)))
    put("mn9", np.flatnonzero(np.char.find(F["peripheral_target_type"].astype(str), "m9_muscle") >= 0))
    for vis in ("LC4", "LPLC2", "LC6", "LPLC1", "LC16", "LC11"):
        put(f"vis:{vis}", where(cell_type=vis))
        for side in SIDES:
            put(f"vis:{vis}:{side}", where(cell_type=vis, side=side))
    put("dn:landing", where(cell_type=("DNp07", "DNp10")))
    put("dn:grooming", where(cell_type=("DNg62", "DNge078")))
    for side in SIDES:
        put(f"dn:landing:{side}", where(cell_type=("DNp07", "DNp10"), side=side))
        put(f"leg_touch:{side}", np.flatnonzero((F["cell_function"] == "tactile") & np.isin(F["body_part_sensory"], ["front_leg", "middle_leg", "hind_leg"]) & (F["side"] == side)))
    put("leg_touch", np.flatnonzero((F["cell_function"] == "tactile") & np.isin(F["body_part_sensory"], ["front_leg", "middle_leg", "hind_leg"])))
    put("thermo", where(cell_class="thermosensory_receptor_neuron"))
    put("nociceptor", where(cell_function="nociception"))
    put("photoreceptor", where(peripheral_target_type="photoreceptor"))
    put("leg_mn", where(cell_function="leg_motor"))
    put("motor", where(super_class="motor"))
    put("sensory", where(flow="afferent"))
    return {k: tuple(int(i) for i in v) for k, v in sets.items()}


@dataclass(frozen=True)
class FlyNet:
    neurons: Neurons
    connectome: Connectome

    def brain(self, gain: float = GAIN, *, backend: str = "cpu", class_log_gain: dict[str, float] | None = None, **model: Any) -> Brain:
        """The whole brain at the dictionary's gains: one global gain, and the class gains as per-neuron log gains."""
        return Brain(self.connectome, NeuronModel(gain=gain, **model), log_gain=log_gain_for(self.connectome, class_log_gain), backend=backend)

    def side(self) -> np.ndarray:
        return self.neurons.fields["side"]


def load_fly(root: Path = DEFAULT_ROOT) -> FlyNet:
    neurons, pre, post, count, sign = load_fixture(root)
    connectome = Connectome(neurons.n, pre, post, count, sign, populations(neurons), label="banc888-min5")
    return FlyNet(neurons, connectome)
