"""Rebuild the teaching graph from pinned public OpenWorm data, never private fixtures."""

import csv
import hashlib
import io
import json
import re
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

WORM = Path(__file__).resolve().parents[1] / "worm"
BASE = "https://raw.githubusercontent.com/openworm/ConnectomeToolbox/b9c0b4a7bc2ccf47d3ce7aac624e1b3e2ea86254/"
FILES = {
    "herm_full_edgelist.csv": "142693f17556148d7f962835b18ac6dd5af18b7467eef61815ebc1dd5474c0ca",
    "all_cell_info.csv": "e467c065342cafe8be7df2b6d781756fe1cfbae5fe7d74352a36682dea6b5fc9",
}


def main():
    tables = {}
    for name, digest in FILES.items():
        raw = urllib.request.urlopen(BASE + "cect/data/" + name, timeout=30).read()
        assert hashlib.sha256(raw).hexdigest() == digest, name
        tables[name] = list(csv.DictReader(io.StringIO(raw.decode())))
    nml_url = "https://raw.githubusercontent.com/openworm/c302/6cd861f8ca4d3241ee9cf4627884caa930dab53c/examples/c302_A_Full.net.nml"
    raw = urllib.request.urlopen(nml_url, timeout=30).read()
    nml_sha = "bb017a418de7ae8091e92eb586045dff83cbb5f3a649914fae3e7381646ea54a"
    assert hashlib.sha256(raw).hexdigest() == nml_sha
    root = ET.fromstring(raw)
    names_all = {
        p.attrib["id"]
        for p in root.iter()
        if p.tag.endswith("}population") and "neuron" in p.attrib["component"]
    }

    def canonical(name):
        return re.sub(r"0+(\d+)$", lambda m: str(int(m[0])), name.strip())

    cells = {r["Cell name"].strip(): r for r in tables["all_cell_info.csv"]}
    edges = [
        (canonical(r["Source"]), canonical(r["Target"]), int(r["Weight"]))
        for r in tables["herm_full_edgelist.csv"]
        if r["Type"].strip() == "chemical"
        and canonical(r["Source"]) in names_all
        and canonical(r["Target"]) in names_all
        and canonical(r["Source"]) != canonical(r["Target"])
    ]
    names = sorted({x for a, b, _ in edges for x in (a, b)})
    ids = {name: i for i, name in enumerate(names)}
    groups = []
    for name in names:
        description = cells.get(name, {}).get("Classification", "").lower()
        groups.append(
            "motor"
            if "motor" in description
            else "sensory"
            if "sensory" in description or "amphid" in description
            else "interneuron"
        )
    incoming = [0.0] * len(names)
    for _, b, count in edges:
        incoming[ids[b]] += count
    # Positive effective transport is imposed, not inferred transmitter physiology.
    weighted = [
        [ids[a], ids[b], 0.85 * count / incoming[ids[b]]] for a, b, count in edges
    ]
    data = {
        "names": names,
        "groups": groups,
        "edges": weighted,
        "contraction": 0.85,
        "stimuli": {
            k: [ids[n] for n in ns if n in ids]
            for k, ns in {
                "anterior": ["ALML", "ALMR", "AVM"],
                "posterior": ["PLML", "PLMR"],
                "nose": ["ASHL", "ASHR", "FLPL", "FLPR"],
                "odor": ["AWCL", "AWCR", "AWAL", "AWAR"],
            }.items()
        },
        "source_urls": {n: BASE + "cect/data/" + n for n in FILES},
        "source_sha256": {**FILES, "c302_A_Full.net.nml": nml_sha},
        "neuron_list_url": nml_url,
        "citation": "Cook et al. 2019, Nature 571:63–71; public OpenWorm ConnectomeToolbox tables.",
        "boundary": "Chemical topology only. Positive normalized weights and tanh dynamics are imposed. No gap junctions, validated physiology, or full animal behavior.",
    }
    (WORM / "worm.json").write_text(json.dumps(data, separators=(",", ":")) + "\n")
    license_text = urllib.request.urlopen(BASE + "LICENSE", timeout=30).read().decode()
    (WORM / "OPENWORM_LICENSE.txt").write_text(license_text)
    print(f"{len(names)} annotated owners, {len(edges)} chemical edges")


if __name__ == "__main__":
    main()
