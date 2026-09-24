"""Gate 3, the instinct facts: what the physiology reports about the fly's innate behaviours.

Every row names its source. Two facts train the one global gain; the rest are held out and
scored on the connectome and on the shuffled control, as in the reflex protocol.
"""

from __future__ import annotations

from cadence.protocol import Levels, Protocol, Row

__all__ = ["instinct_protocol", "SOURCES"]

SOURCES = {
    "vonreyn2014": "von Reyn CR et al. (2014) A spike-timing mechanism for action selection. Nat Neurosci 17:962-970 (looming: LC4 and LPLC2 onto the giant fibre; the giant fibre takeoff).",
    "klapoetke2017": "Klapoetke NC et al. (2017) Ultra-selective looming detection from radial motion opponency. Nature 551:237-241 (LPLC2 onto the giant fibre).",
    "ache2019": "Ache JM et al. (2019) State-dependent decoupling of sensory and motor circuits underlies behavioral flexibility in Drosophila. Nat Neurosci 22:1132-1139 (DNp07 and DNp10 drive landing; LC4 input).",
    "gordon2009": "Gordon MD, Scott K (2009) Motor control in a Drosophila taste circuit. Neuron 61:373-384 (sugar sensing to the proboscis motor neuron MN9).",
    "shiu2024": "Shiu PK et al. (2024) A Drosophila computational brain model reveals sensorimotor processing. Nature 634:210-219 (sugar onto MN9; bitter suppresses; water onto MN9; predictions on the FlyWire connectome).",
    "hampel2015": "Hampel S et al. (2015) Simultaneous activation of parallel sensory pathways promotes a grooming sequence in Drosophila. eLife 4:e08758 (Johnston's organ onto the antennal grooming descending neurons aDN1 and aDN2).",
    "bidaye2014": "Bidaye SS et al. (2014) Neuronal control of Drosophila walking direction. Science 344:97-101 (MDN backward walking).",
    "namiki2018": "Namiki S et al. (2018) The functional organization of descending sensory-motor pathways in Drosophila. eLife 7:e34272 (descending neuron types).",
    "dolan2019": "Dolan MJ et al. (2019) Neurogenetic dissection of the Drosophila lateral horn reveals major outputs, diverse behavioural functions, and interactions with the mushroom body. eLife 8:e43079 (odour to the lateral horn and descending neurons).",
    "stensmyr2012": "Stensmyr MC et al. (2012) A conserved dedicated olfactory circuit for detecting harmful microbes in Drosophila. Cell 151:1345-1357 (the aversive geosmin line, DA2).",
}


def instinct_protocol(steps: int = 60) -> Protocol:
    stimuli = {
        "loom_lplc2": ("vis:LPLC2",),
        "loom_lc4": ("vis:LC4",),
        "loom_both": ("vis:LC4", "vis:LPLC2"),
        "loom_left": ("vis:LC4:left", "vis:LPLC2:left"),
        "sugar_labellum": ("grn:sugar:labellum",),
        "sugar_front_leg": ("grn:sugar:front_leg",),
        "bitter_labellum": ("grn:bitter:labellum",),
        "sugar_and_bitter": ("grn:sugar:labellum", "grn:bitter:labellum"),
        "water_labellum": ("grn:water:labellum",),
        "jo_ce": ("jo:C:left", "jo:C:right", "jo:E:left", "jo:E:right"),
        "fruit_odour": ("orn:decaying_fruit", "orn:fruity"),
        "aversive_odour": ("orn:aversive",),
        "leg_touch": ("leg_touch",),
    }
    training = [
        ("loom_lplc2", "gf", "active"),  # klapoetke2017
        ("sugar_labellum", "mn9", "active"),  # gordon2009
    ]
    rows = [
        Row("I01", "loom_both", "gf", "active", "vonreyn2014"),
        Row("I02", "loom_lc4", "dn:landing", "sparse", "ache2019"),
        Row("I03", "loom_both", "dn:landing", "sparse", "ache2019"),
        Row("I04", "loom_left", "gf", "active", "vonreyn2014"),
        Row("I05", "sugar_front_leg", "mn9", "active", "shiu2024"),
        Row("I06", "water_labellum", "mn9", "active", "shiu2024"),
        Row("I07", "bitter_labellum", "mn9", "inactive", "shiu2024"),
        Row("I08", "sugar_and_bitter", "mn9", "reduced", "shiu2024", relative_to="sugar_labellum"),
        Row("I09", "jo_ce", "dn:grooming", "active", "hampel2015"),
        Row("I10", "fruit_odour", "dn", "sparse", "dolan2019"),
        Row("I11", "aversive_odour", "dn", "sparse", "stensmyr2012; dolan2019"),
        Row("I12", "fruit_odour", "kc", "sparse", "dolan2019"),
        Row("I13", "leg_touch", "dn:landing", "inactive", "ache2019"),
        Row("I14", "loom_both", "mn:ttm", "active", "vonreyn2014"),
    ]
    return Protocol(stimuli=stimuli, rows=rows, training=training, levels=Levels(), steps=steps)
