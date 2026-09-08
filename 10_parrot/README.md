# 10 · Grey parrot: hear, remember, imitate

An African grey lives in a household. Five sounds happen there: a doorbell, a phone, a
microwave, a whistle, a siren. Three of them happen forty or fifty times a day, two a
handful of times. The parrot has a cochlea, a syrinx with three muscles, and one
equilibrium net for a brain. By the end of the day it can say the doorbell.

**Play with it:** the page lets you play the household sounds to the parrot, let it speak,
make it babble, and draw a whistle of your own and repeat it until the parrot picks it up.
Every settlement and every learning update in the page is the same rule as in training.

```bash
python train.py            # one day (about ten minutes of wall-clock); writes receipt.json, net.json, bouts.json, imitations/*.wav
python train.py --verify receipt.json
python build_page.py       # embeds net.json and the cochlea's filters into index.html
```

## The biology, and where we took shortcuts

| in the bird | here | shortcut |
|---|---|---|
| cochlea and auditory nerve: a tonotopic map of the sound's envelope | 24 band-pass filters on a log scale from 200 Hz to 5 kHz, an RMS envelope per 10 ms, log-compressed to levels in [0, 1] | linear filters, no adaptation, no phase |
| auditory memory of familiar sounds (caudomedial nidopallium, NCM), which forms by repeated exposure and fades without it | the same net's *memory group*: 24 owners nudged toward the next cochlear frame after the frames it has just heard; every update decays every seam a little, so what repeats stays and what does not fades | the memory is a next-frame expectation over a context window rather than a recurrent trace |
| the song system's mirror neurons: cells that fire both when the bird produces a sound and when it hears it, the inverse model that maps a heard sound onto the command that makes it | the same net's *motor group*: 16 owners nudged, while the parrot hears itself, toward the command it issued one frame earlier | one frame of delay between command and sound |
| the anterior forebrain pathway (LMAN): the source of vocal variability, and of the drive to practise when the bird is alone | an arousal integrator that fills in quiet and opens a bout; babbling bouts are smooth random muscle commands; during imitation the commands carry smooth noise | a scalar arousal, a coin for babble against imitate |
| dopamine from the ventral tegmental area, signalling whether a rendition came out better or worse than usual | while imitating, each command taken is pulled toward, in the context that chose it, by how much better than the running average the heard frame matched the memory's expectation, and pushed from when worse | the critic is the memory's own expectation; a scalar running average is the baseline |
| the syrinx: two labia in an airflow, tension setting pitch, air-sac pressure switching phonation on | the Amador–Mindlin labial oscillator `x'' = -eps x - C x² x' + beta x'`: tension sets `eps` (pitch 300 Hz to 3.5 kHz), pressure sets `beta`, and phonation starts through a Hopf bifurcation at pressure level 0.3 | one sound source, not two |
| the vocal tract, beak and tongue (parrots shape formants with the tongue) | one resonance whose centre a third muscle sets between 800 Hz and 4 kHz | one resonance |
| what the bird attends to | the first 80 ms after an onset that follows a quiet spell is kept as a cue (up to 24); a replay draws among cues in proportion to how well the memory recalls them | the cue is a snapshot, not a memory of its own |
| time | the context holds the last 80 ms at full resolution, plus six 50 ms bins and six 200 ms bins behind it; how many bins a sound has filled is the parrot's clock | a fixed window instead of an internal chain of time-locked bursts (HVC) |

Names never reach the brain. The receipt's per-sound scores use them only to look up which
sound was which.

## How it learns, exactly

One net: 480 auditory owners (the context), 160 hidden owners, 40 output owners, tied seams
between neighbouring layers and nothing else. Two learning signals, both the free/nudged
rule with the quadratic nudge on one output group:

1. **Listening.** While a household sound is on (and for 300 ms after it stops), each 80 ms
   chunk gives eight rows: the context before each frame, and the frame. Settle free;
   settle with the memory group pulled toward the frame (`beta · (frame − s)`), settle with
   it pushed away; move each seam by the difference of the two Hebbian products; decay every
   seam by 0.03%. The memory is switched off while the parrot sings, as auditory responses
   in the song system are.
2. **Singing.** During a bout, each frame the parrot hears (its own voice, one frame after
   the command) gives a row: the context, and the command that produced it as a bump over
   the motor owners. The same update, on the motor group. Babbling bouts are how it collects
   these rows; every bout ends with eight frames of a closed air sac, so quiet maps to closed.
3. **Getting better.** During an imitation bout the commands carry smooth exploratory noise.
   For each frame, the mismatch between what the parrot heard and what its memory expected
   is compared with the running average of that mismatch; the command it took is nudged
   toward, in the context the mirror saw, with a weight equal to the advantage, or pushed
   away when the rendition was worse than usual. This is the reward rung's rule (Pong,
   cart-pole) with the parrot's own memory as the critic.

To say something, the parrot takes a cue, and each frame: the memory expects the next frame
from the context it hears; the mirror is shown a context whose latest frame is that
expectation and answers with a command; the syrinx sounds; the cochlea hears it, and that
frame is the next context. The loop runs through the body. It ends when the mirror keeps the
air sac closed for six frames.

## What the receipt measures

- **recall**: the memory alone, replayed from a sound's first 80 ms on its own expectations,
  correlated with the sound's cochleagram. A sound the memory holds replays; one it does not
  does not.
- **imitation**: the loop through the body from the same cue, the produced sound heard through
  the cochlea and correlated with the original.
- both for the sounds heard often, the sounds heard rarely, and an untrained brain.

<!-- results -->
<!-- /results -->

## What we learned building it

- A constant learning rate of 1 with momentum saturated the memory group within a minute
  of listening: every expected level pinned at its ceiling or floor. A stream has no
  epochs; the rate has to be gentle (0.3, no momentum).
- Next-frame error is a poor measure of memory: a net soon predicts "more of the same" for
  any sound, and rare sounds look as familiar as frequent ones. Replaying from the onset
  is what tells a held sound from a merely smooth one.
- An 80 ms context has no clock: a steady note cannot know when to become the next note.
  The averaged bins behind it are the cheapest clock we found.
- A memory that never heard how a sound ends replays it forever; one that replays on its
  own damped expectations stops after 100 ms. Running the loop through the body, on what
  the parrot actually hears, is both the biological arrangement and the one that works.
