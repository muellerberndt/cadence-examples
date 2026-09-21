# The worm

The nervous system of the hermaphrodite *C. elegans*, 302 neurons wired as measured, as one
Cadence `TemporalPatchNet`. It lives on a plate, smells, eats, gets hurt, and learns during its
life what the smells around it predict. The page draws the whole nervous system inside the
transparent body at every zoom: every neuron, every active synapse and the transmitter it
releases, pulses running down the nerve cords, and every lesson as a repair wave spreading out
from the command neurons. A second view shows the head at scale, straightened: the nerve ring,
the ganglia around it, and the sensory and command neurons by name. Brightness is activity on a
log scale from 10⁻⁶ to 1, because a signal fades by orders of magnitude as it spreads.

## The brain is the connectome

- Neurons: the 302 of `data/c302_A_Full.net.nml`, placed at their measured soma positions
  (`data/SOURCES.md`).
- Connections: `cadence.experimental.PartitionedTemporalPatchNet` with the connectome as its
  mask. Each chemical synapse class and each gap junction is one permitted entry of the context
  matrix `A`; nothing else can grow. At birth an entry is `sign · log(1 + count)`, with the sign
  from the transmitter (GABA inhibits), gap junctions symmetric, and `A` scaled to spectral
  radius 0.5.
- Senses (`B`): smell A on AWA, smell B on AWC, bacteria on the dopaminergic CEP, ADE and PDE,
  pain on the ASH nociceptors.
- Readouts (`C`): the command interneurons. AVB and PVC drive the body forward; AVA, AVD and
  AVE drive it backward. `C` averages each group and is fixed.
- Body: when backward drive wins, the worm reverses and turns (a pirouette); forward drive
  sets its speed. It turns more often while forward drive falls and bends toward the side of
  its head swing on which forward drive rose, so smells steer it only through what the brain
  makes of them.

## How it learns, by the letter

The brain is used exactly as the library documents it:

1. Every tick, `sense` appends what the sensory neurons receive to the stretch not yet
   committed and `imagine`s the stretch from the live state. The last readout drives the body.
2. When the stretch grows past two windows, the first window is committed with `advance`.
3. When food or pain arrives, `observe` runs on the stretch that led there. The target keeps
   the free prediction everywhere except the last four ticks, where the wanted command group is
   asked to reach 0.8 and its rival 0. `observe` learns by centred equilibrium detuning with
   backtracking.
4. A lesson is kept only if it leaves the context stable. If the growth rate of `A` would pass
   0.97, the change is halved (with `set_parameters`) until it no longer does.

Before birth the worm receives twelve lessons in which pain drives the reverse group: the
withdrawal reflex it hatches with. What smells mean is not supplied; it is learned from what
the smells come before.

A treat on the page is food arriving; a poke is pain. Both are lessons like any other.

## The page

`web/` runs the same brain in the browser. `web/brain.js` solves the same energy with Newton-CG
on the sparse connectome; `tests/parity.mjs` checks it against the library on three lessons of
5, 8 and 16 ticks (worst relative difference 8.5e-12).

```bash
python -m pip install "cadence-net @ git+https://github.com/muellerberndt/cadence.git@3d655c84b131388c4ef05d945947fd3e9e786d45" scipy
python worm/tools/build_connectome.py      # data/connectome.json from the sources
python worm/tools/export_web.py            # the newborn brain, web data and parity cases
node worm/tests/parity.mjs
python -m http.server -d worm/web 8801     # then open http://127.0.0.1:8801
```

`?seed=N` chooses the plate. `worm/` (`brain.py`, `life.py`, `world.py`, `rng.py`) is the
Python reference of the brain loop, the world and the life that the page ports.

## Supplied and learned

- Supplied: the wiring and its signs, which neurons sense what, which neurons command the body,
  the body mechanics, the moments at which lessons happen, the target level, the stability bound.
- Learned: the weights of every connection, from twelve reflex lessons and then from the life.

## Limits

- `PartitionedTemporalPatchNet` is on Cadence `main` after 0.11.0, not in the 0.11.0 release;
  the worm pins that commit.
- Neurons are rate units, not spiking cells; the body is a trail-following curve, not a
  muscle model.
- Food and pain are the only outcomes; a smell learns only by coming before one of them.
- A receipt of conditioning with paired, unpaired, frozen and lesioned controls across seeds is
  not yet in this directory.
