# 10 · Grey parrots: hear, remember, imitate

Two African greys live in a household. Six sounds happen there: a doorbell, a phone, a
microwave, a whistle, a siren, and someone saying hello. Three of them happen well over a
hundred times a day, three a handful of times. Each parrot has a cochlea, a syrinx with
three muscles, and one equilibrium net for a brain. By the end of the day it can say the
microwave, and hello.

**Play with them:** the page shows both birds and both brains settling live. Play the
household to them, record a noise of your own or draw a whistle and repeat it until a parrot
picks it up, let one speak and watch the other listen and learn from it. Every settlement
and every learning update in the page is the same rule as in training; the brain view shows
every owner's activation and, after each update, the seams that moved.

```bash
python train.py                     # two simulated hours (about twenty minutes); writes receipt.json, net.json, bouts.json, imitations/*.wav
python train.py --seed 1 --tag _1   # the second parrot: receipt_1.json, net_1.json, ...
python train.py --verify receipt.json
python build_page.py                # embeds both nets and the cochlea's filters into index.html, and the receipt's numbers into this README
```

## The biology, and where we took shortcuts

| in the bird | here | shortcut |
|---|---|---|
| cochlea and auditory nerve: a tonotopic map of the sound's envelope | 24 band-pass filters on a log scale from 200 Hz to 5 kHz, an RMS envelope per 10 ms, log-compressed to levels in [0, 1] | linear filters, no adaptation, no phase |
| auditory memory of familiar sounds (caudomedial nidopallium, NCM), which forms by repeated exposure and fades without it | the same net's *memory group*: 24 owners nudged toward the next cochlear frame after the frames it has just heard; every update decays every seam a little, so what repeats stays and what does not fades | the memory is a next-frame expectation over a context window rather than a recurrent trace |
| the song system's mirror neurons: cells that fire both when the bird produces a sound and when it hears it, the inverse model that maps a heard sound onto the command that makes it | the same net's *motor group*: 16 owners nudged, while the parrot hears itself, toward the command it issued one frame earlier | one frame of delay between command and sound |
| the anterior forebrain pathway (LMAN): the source of vocal variability, and of the drive to practise when the bird is alone | an arousal integrator that fills in quiet and opens a bout; babbling bouts are smooth random muscle commands; during imitation the commands carry smooth noise | a scalar arousal, a coin for babble against imitate |
| dopamine from the ventral tegmental area, signalling whether a rendition came out better or worse than usual | while imitating, each command taken is pulled toward, in the context that chose it, by how much better than the running average the heard frame matched the memory's expectation, and pushed from when worse | the critic is the memory's own expectation; a scalar running average is the baseline |
| the syrinx: two labia in an airflow, tension setting pitch, air-sac pressure switching phonation on | the Amador–Mindlin labial oscillator `x'' = -eps x - C x² x' + beta x'`: tension sets `eps` (pitch 300 Hz to 3.5 kHz), pressure sets `beta`, and phonation starts through a Hopf bifurcation at pressure level 0.3. The sound is the airflow the labia gate: they close once a cycle and cut the flow, so the source is a pulse train (its rate of change, as sound radiates) rich in harmonics, with turbulence noise in proportion to the pressure | one sound source, not two |
| the vocal tract, beak and tongue (parrots shape formants with the tongue) | two resonances: the trachea's, fixed at 1.5 kHz, and one a third muscle sets between 800 Hz and 4 kHz; the radiated sound is soft-limited | two resonances |
| what the bird attends to | the first 80 ms after an onset that follows a quiet spell is kept as a cue (up to 24); a replay draws among cues in proportion to how well the memory recalls them | the cue is a snapshot, not a memory of its own |
| time | the context holds the last 80 ms at full resolution, plus six 50 ms bins and six 200 ms bins behind it; how many bins a sound has filled is the parrot's clock | a fixed window instead of an internal chain of time-locked bursts (HVC) |

Names never reach the brain. The receipt's per-sound scores use them only to look up which
sound was which.

## How it learns, exactly

One net: 480 context owners, two hidden populations (128 auditory owners under the memory
group read the whole context, since the memory needs the clock; 96 vocal owners under the
motor group read only the last 80 ms, since the command that makes a sound is a function of
the sound now), 40 output owners, tied seams between neighbouring layers and nothing else.
The separation is the caudomedial nidopallium's from the song system's, and it matters
twice over: with one shared hidden layer every motor nudge also moved the memory's seams
(recall 0.55 to 0.44), and a mirror that read the long context answered with an average of
the previous syllables' pitches. Three learning signals, all the free/nudged rule with the
quadratic nudge on one output group:

1. **Listening.** While a household sound is on (and for 300 ms after it stops), each 80 ms
   chunk gives eight rows: the context before each frame, and the frame. Settle free;
   settle with the memory group pulled toward the frame (`beta · (frame − s)`), settle with
   it pushed away; move each seam by the difference of the two Hebbian products; decay every
   seam by 0.03%. The memory is switched off while the parrot sings, as auditory responses
   in the song system are.
2. **Babbling.** During a babbling bout, each frame the parrot hears (its own voice, one
   frame after the command) gives a row: the recent frames, and the command that produced
   them as a bump over the motor owners. The same update, on the motor group. Babbling jumps
   to a new pitch and tract setting every 300 ms, as subsong is varied, so the mirror sees the
   whole range; every bout ends with eight frames of a closed air sac, so quiet maps to
   closed. The mirror never learns from imitation bouts: their commands are its own answers,
   and a mirror taught its own answers settles on one command for everything (measured: it
   did).
3. **Getting better.** During an imitation bout the commands carry smooth exploratory noise.
   For each frame, the mismatch between what the parrot heard and what its memory expected
   is compared with the running average of that mismatch; the command it took is nudged
   toward, in the context the mirror saw, with a weight equal to the advantage, or pushed
   away when the rendition was worse than usual. This is the reward rung's rule (Pong,
   cart-pole) with the parrot's own memory as the critic.

To say something, the parrot takes a cue and lets the memory replay: each frame, the memory
expects the next frame from the context so far; the mirror is shown a context whose latest
frame is that expectation and answers with a command; the syrinx sounds; and the
expectation, not the heard sound, becomes the next context. The template drives, as the
bird's memory of the tutor drives its song; what the parrot hears of itself goes to the
critic. The bout ends when the memory expects quiet, or the mirror keeps the air sac
closed, for six frames. (Feeding the heard voice back as context, which we tried first,
derails the memory: it never learned its own voice, and shown it as context it expected
nothing useful.)

## What the receipt measures

- **recall**: the memory alone, replayed from a sound's first 80 ms on its own expectations,
  correlated with the sound's cochleagram. A sound the memory holds replays; one it does not
  does not.
- **imitation**: the loop through the body from the same cue, the produced sound heard through
  the cochlea and correlated with the original.
- both for the sounds heard often, the sounds heard rarely, and an untrained brain.

<!-- results -->
Two 60-minute days, the second with the roles swapped so every sound is scored once heard often and once heard rarely (day A: 439 household sounds, 162 spontaneous bouts, 71 babbles and 91 imitations; 329 s of wall-clock a day). The first parrot, seed 0:

| sound | heard often / rarely | recall, heard often | recall, heard rarely | recall, untrained | imitation, heard often | imitation, heard rarely | imitation, untrained | pitch track, heard often | pitch track, untrained |
|---|---|---|---|---|---|---|---|---|---|
| beeps | 153 / 6 | 0.153 | 0.484 | -0.128 | 0.416 | 0.410 | 0.000 | 0.000 | 0.000 |
| doorbell | 129 / 7 | 0.631 | -0.157 | -0.000 | 0.253 | 0.318 | -0.166 | 0.084 | -0.806 |
| hello | 137 / 8 | 0.842 | 0.069 | -0.067 | 0.312 | 0.359 | 0.014 | -0.074 | -0.301 |
| ring | 150 / 4 | 0.551 | 0.538 | 0.107 | 0.522 | 0.513 | -0.179 | 0.000 | 0.000 |
| siren | 139 / 9 | 0.650 | 0.148 | -0.010 | 0.549 | 0.500 | 0.508 | 0.489 | 0.115 |
| whistle | 141 / 7 | 0.925 | 0.150 | -0.054 | 0.658 | 0.670 | 0.000 | 0.248 | 0.000 |

Means: recall 0.625 when a sound was heard often against 0.206 when it was rare and -0.026 untrained; 5 of 6 sounds are recalled better after the day that repeated them. Imitation (whole cochleagram) 0.452 / 0.462 / 0.030; spectral shape 0.495 / 0.512 / 0.100; pitch track 0.125 / 0.085 / -0.165. Brain: 744 owners, 65,256 parameters. The second parrot (seed 1, its own receipt) recalls 0.426 / 0.357 and imitates at 0.407 / 0.434 against 0.428 untrained. The receipts bind these numbers to the code and the seeds.
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
- Two learners in one net need their own decay. With one decay on every update, the
  mirror's input seams, learned only during bouts, faded five times as often as they
  learned, and the mirror settled on the mean babble command for any input; the plain
  imitation score hid it, the pitch-track score exposed it. Now each population moves and
  decays only on its own updates (`Learner.trainable_owners` beside `trainable_overlaps`).
- The imitation score matters. Correlating whole cochleagrams rewards being loud in the
  right places; a broadband voice scores 0.44 untrained. The receipt therefore also carries
  the spectral shape (each frame's mean level removed) and the pitch track (the dominant
  channel where both are loud), which is the one that tells melody from noise.
