#!/usr/bin/env python3
"""Build data/connectome.json: the 302 hermaphrodite neurons with positions,
transmitters and synapses, read by both the Python reference and the browser.

Sources (all in data/, see data/SOURCES.md):
  c302_A_Full.net.nml   OpenWorm c302: chemical synapses with transmitter
                        identity and synapse counts, gap junctions
  celegans277.mat       Kaiser & Hilgetag: measured soma positions of 277 neurons
                        in the lateral plane (anterior-posterior, dorsal-ventral, mm)
  all_cell_info.csv     WormAtlas cell classes
"""
import collections
import csv
import hashlib
import json
import re
from pathlib import Path

import scipy.io as sio

DATA = Path(__file__).resolve().parents[1] / "data"

# The 25 neurons without a measured position are placed by anatomy.
# Pharyngeal neurons sit along the pharynx between the tip (+0.137 mm) and the
# terminal bulb; the nerve ring encircles the isthmus near +0.09 mm.
PLACED = {
    "I1L": (0.122, 0.004), "I1R": (0.122, -0.004), "I2L": (0.118, 0.005),
    "I2R": (0.118, -0.005), "I3": (0.126, 0.0), "M1": (0.114, 0.003),
    "M2L": (0.110, 0.005), "M2R": (0.110, -0.005), "MCL": (0.112, 0.006),
    "MCR": (0.112, -0.006), "MI": (0.108, 0.0), "NSML": (0.098, 0.005),
    "NSMR": (0.098, -0.005), "M3L": (0.094, 0.004), "M3R": (0.094, -0.004),
    "I4": (0.090, 0.002), "I5": (0.084, -0.002), "M4": (0.080, 0.0),
    "I6": (0.068, 0.003), "M5": (0.064, -0.002),
    # canal-associated neurons: mid-body, along the excretory canal
    "CANL": (-0.440, 0.010), "CANR": (-0.440, -0.010),
}
MIRROR = {"AIBL": "AIBR", "AIYL": "AIYR", "SMDVL": "SMDVR"}

# Transmitters folded into four visual families (the render uses the family,
# the data keeps the exact label).
FAMILY = {
    "Glutamate": "excitatory", "Acetylcholine": "excitatory",
    "GABA": "inhibitory",
    "Dopamine": "dopamine",
    "Serotonin": "modulatory", "Octapamine": "modulatory", "Tyramine": "modulatory",
    "FMRFamide": "modulatory",
}


def family(label: str) -> str:
    parts = label.split("_")
    if "Dopamine" in parts:
        return "dopamine"
    if "GABA" in parts:
        return "inhibitory"
    if parts[0] in ("Glutamate", "Acetylcholine"):
        return "excitatory"
    return "modulatory"


def role(kind: str, name: str) -> str:
    k = kind.lower()
    if name in ("AVAL", "AVAR", "AVBL", "AVBR", "AVDL", "AVDR", "AVEL", "AVER", "PVCL", "PVCR"):
        return "command"
    if "pharyn" in k or name in PLACED and name not in ("CANL", "CANR"):
        return "pharyngeal"
    if "motor" in k:
        return "motor"
    if "sensory" in k or k in ("amphid", "cephalic", "inner labial", "outer labial") or "sens" in k:
        return "sensory"
    return "interneuron"


def main() -> None:
    nml = (DATA / "c302_A_Full.net.nml").read_text()
    neurons = [n for n, c in re.findall(r'<population id="([^"]+)" component="([^"]+)"', nml)
               if "neuron" in c]
    index = {n: i for i, n in enumerate(neurons)}

    m = sio.loadmat(DATA / "celegans277.mat")
    labels = [str(x[0][0]).strip() for x in m["celegans277labels"]]
    measured = {l: tuple(float(v) for v in p) for l, p in zip(labels, m["celegans277positions"])}

    kinds = {r["Cell name"].strip(): r["Type"].strip() for r in csv.DictReader(open(DATA / "all_cell_info.csv"))}

    proj = re.findall(
        r'<projection id="NC_[^"]*?_[^_"]+_([^"]+)" presynapticPopulation="([^"]+)" '
        r'postsynapticPopulation="([^"]+)" synapse="([^"]+)">\s*<connectionWD[^>]*weight="([^"]+)"', nml)
    chemical, gap = [], []
    outgoing = collections.defaultdict(collections.Counter)
    for label, pre, post, syn, weight in proj:
        if pre not in index or post not in index:
            continue  # neuromuscular and muscle entries are outside the brain
        count = int(round(float(weight)))
        if "elec" in syn:
            gap.append([index[pre], index[post], count])
            continue
        sign = -1 if "inh" in syn else 1
        chemical.append([index[pre], index[post], count, sign, label])
        outgoing[pre][label] += count

    cells = []
    for n in neurons:
        if n in measured:
            (ap, dv), source = measured[n], "measured"
        elif n in MIRROR:
            (ap, dv), source = measured[MIRROR[n]], "mirrored"
        else:
            (ap, dv), source = PLACED[n], "placed"
        nt = outgoing[n].most_common(1)[0][0] if outgoing[n] else None
        cells.append({
            "name": n, "ap": round(ap, 5), "dv": round(dv, 5), "position": source,
            "role": role(kinds.get(n, ""), n), "kind": kinds.get(n, ""),
            "transmitter": nt, "family": family(nt) if nt else None,
        })

    out = {
        "neurons": cells,
        "chemical": chemical,           # [pre, post, synapse count, sign, transmitter]
        "gap": gap,                     # [a, b, gap junction count]
        "sources": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                    for p in (DATA / "c302_A_Full.net.nml", DATA / "celegans277.mat",
                              DATA / "all_cell_info.csv")},
    }
    path = DATA / "connectome.json"
    path.write_text(json.dumps(out, separators=(",", ":")))
    roles = collections.Counter(c["role"] for c in cells)
    fams = collections.Counter(c["family"] for c in cells)
    pos = collections.Counter(c["position"] for c in cells)
    print(f"wrote {path.name}: {len(cells)} neurons, {len(chemical)} chemical, {len(gap)} gap")
    print("roles", dict(roles)); print("families", dict(fams)); print("positions", dict(pos))


if __name__ == "__main__":
    main()
