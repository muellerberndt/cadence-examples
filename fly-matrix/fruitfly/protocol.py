"""Gate 2, the reflex facts: what the physiology reports about the wing steering circuit.

Every row names its source. Two facts train the one global gain; the rest are held out and
scored on the connectome and on a connectome whose postsynaptic endpoints were permuted
(counts, signs and out-degrees kept). Nothing here is tuned against the held-out rows.
Facts that rest on electrical synapses (the giant fibre to the jump and flight motor
neurons) are outside the reach of a chemical synapse table and are not asked.
"""

from __future__ import annotations

from cadence.protocol import Levels, Protocol, Row

__all__ = ["reflex_protocol", "SOURCES"]

SOURCES = {
    "fayyazuddin1996": "Fayyazuddin A, Dickinson MH (1996) Haltere afferents provide direct, electrotonic input to a steering motor neuron in the blowfly, Calliphora. J Neurosci 16:5225-5232.",
    "fayyazuddin1999": "Fayyazuddin A, Dickinson MH (1999) Convergent mechanosensory input structures the firing phase of a steering motor neuron in the blowfly, Calliphora. J Neurophysiol 82:1916-1926.",
    "dickinson1999": "Dickinson MH (1999) Haltere-mediated equilibrium reflexes of the fruit fly, Drosophila melanogaster. Phil Trans R Soc B 354:903-916.",
    "chan1998": "Chan WP, Prete F, Dickinson MH (1998) Visual input to the efferent control system of a fly's gyroscope. Science 280:289-292.",
    "suver2016": "Suver MP, Huda A, Iwasaki N, Safarik S, Dickinson MH (2016) An array of descending visual interneurons encoding self-motion in Drosophila. J Neurosci 36:11768-11780.",
    "parsons2010": "Parsons MM, Krapp HG, Laughlin SB (2010) Sensor fusion in identified visual interneurons. Curr Biol 20:624-628.",
    "strausfeld1985": "Strausfeld NJ, Seyan HS (1985) Convergence of visual, haltere, and prosternal inputs at neck motor neurons of Calliphora. Cell Tissue Res 240:601-615.",
    "huston2008": "Huston SJ, Krapp HG (2008) Visuomotor transformation in the fly gaze stabilization system. PLoS Biol 6:e173.",
    "suver2019": "Suver MP, Matheson AMM, Sarkar S, Damiata M, Schoppik D, Nagel KI (2019) Encoding of wind direction by central neurons in Drosophila. Neuron 102:828-842.",
    "ehrhardt2025": "Ehrhardt E et al. (2025) Single-cell type analysis of wing premotor circuits in the ventral nerve cord of Drosophila melanogaster.",
}


def reflex_protocol(steps: int = 60) -> Protocol:
    stimuli = {
        "haltere_left": ("haltere:left",),
        "haltere_right": ("haltere:right",),
        "wing_left": ("wing_sense:left",),
        "wing_right": ("wing_sense:right",),
        "hs_left": ("lptc:hs:left",),
        "vs_left": ("lptc:vs:left",),
        "ocelli": ("ocelli",),
        "jo_e_left": ("jo:E:left",),
    }
    training = [
        ("haltere_left", "mn:wing:b1:left", "active"),  # fayyazuddin1996: haltere afferents onto b1
        ("ocelli", "neck_mn", "sparse"),  # strausfeld1985: ocellar input onto a subset of the neck motor neurons
    ]
    rows = [
        Row("R01", "haltere_left", "steering:left", "lateralized", "fayyazuddin1996; dickinson1999", relative_to="steering:right"),
        Row("R02", "haltere_right", "steering:right", "lateralized", "fayyazuddin1996; dickinson1999", relative_to="steering:left"),
        Row("R03", "haltere_right", "mn:wing:b1:right", "active", "fayyazuddin1996"),
        Row("R04", "wing_left", "mn:wing:b1:left", "active", "fayyazuddin1999"),
        Row("R05", "wing_right", "mn:wing:b1:right", "active", "fayyazuddin1999"),
        Row("R06", "wing_left", "steering:left", "lateralized", "fayyazuddin1999", relative_to="steering:right"),
        Row("R07", "hs_left", "dn", "sparse", "suver2016"),
        Row("R08", "vs_left", "dn", "sparse", "suver2016"),
        Row("R09", "ocelli", "dn", "sparse", "parsons2010"),
        Row("R10", "vs_left", "neck_mn", "sparse", "huston2008"),
        Row("R11", "hs_left", "mn:haltere", "sparse", "chan1998"),
        Row("R12", "haltere_left", "mn:haltere:left", "lateralized", "chan1998; dickinson1999", relative_to="mn:haltere:right"),
        Row("R13", "jo_e_left", "dn", "sparse", "suver2019"),
    ]
    return Protocol(stimuli=stimuli, rows=rows, training=training, levels=Levels(), steps=steps)
