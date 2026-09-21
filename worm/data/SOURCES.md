# Sources

| File | What it holds | Source | Licence | sha256 |
|---|---|---|---|---|
| `c302_A_Full.net.nml` | The 302 hermaphrodite neurons; 2,279 chemical synapse classes with transmitter and synapse count; 1,084 gap junctions | [openworm/c302](https://github.com/openworm/c302) at `6cd861f8ca4d3241ee9cf4627884caa930dab53c`, `examples/c302_A_Full.net.nml` | MIT | `bb017a418de7ae8091e92eb586045dff83cbb5f3a649914fae3e7381646ea54a` |
| `all_cell_info.csv` | Cell classes (sensory, interneuron, motor) and neurotransmitter labels | [openworm/ConnectomeToolbox](https://github.com/openworm/ConnectomeToolbox) at `b9c0b4a7bc2ccf47d3ce7aac624e1b3e2ea86254`, `cect/data/all_cell_info.csv` | MIT | `e467c065342cafe8be7df2b6d781756fe1cfbae5fe7d74352a36682dea6b5fc9` |
| `celegans277.mat` | Measured soma positions of 277 neurons in the lateral plane (anterior–posterior and dorsal–ventral, mm) | Kaiser M., Hilgetag C. C. (2006), *Nonoptimal component placement, but short processing paths, due to long-distance projections in neural systems*, PLoS Comput Biol 2(7): e95; positions from Choe Y., McCormick B. H., Koh W. (2004), *Network connectivity analysis on the temporally augmented C. elegans web*, Neurocomputing 58–60; file as distributed with the Cornell spatial *C. elegans* data set | academic data set, cited as above | `cf9cdce6e81a3b01f6691a8e7b5ca1cd692672ea48253973e82606a573090fb1` |

`tools/build_connectome.py` reads the three files and writes `connectome.json`, recording
these hashes in it. Positions: 277 measured, 3 mirrored from their bilateral partner (AIBL,
AIYL, SMDVL), 22 placed by anatomy (the pharyngeal neurons along the pharynx, CANL and CANR
mid-body). Transmitters are kept exactly in the data and folded into four families for
drawing: excitatory (acetylcholine, glutamate), inhibitory (GABA), dopamine, and modulatory
(serotonin, octopamine, tyramine, the FMRFamide peptides).

`params.json` holds the body, world and learning constants; every number the page and the
Python reference use comes from it.
