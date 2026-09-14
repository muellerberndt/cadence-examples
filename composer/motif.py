"""Graft one-trial motif recall onto the trained brain, in the SAME settlement.

Six explicit position cues write a 6x61 delta-rule memory. Recall owners and note
intention owners send messages both ways. The addressing schedule is supplied;
the remembered content comes only from a committed phrase.
"""

import cadence as cd
import numpy as np
from cadence.brains import couple
from cadence.stream import FastSeams

from .encoding import OFFSETS
from .intuition import Intuition


class MotifBrain:
    def __init__(self, learner, intuition=None):
        base = learner.engine
        w = base.wiring
        n = w.n
        self.base = learner
        self.output_index = learner.output_index
        self.config = learner.config
        self.memory = FastSeams(np.arange(6), np.arange(6, 67), rule="delta")
        self.memory.reset(1)
        self.intuition = intuition or Intuition()
        trained = cd.Wiring.from_edges(n, pre=w.pre, post=w.post, sign=base.weights)
        recall = cd.Wiring.from_edges(
            67,
            pre=np.repeat(np.arange(6), 61),
            post=np.tile(np.arange(6, 67), 6),
            sign=np.zeros(366),
        )
        bridges = [
            item
            for k in range(61)
            for item in [
                ("motif", 6 + k, "music", int(self.output_index[k]), 2.5),
                ("music", int(self.output_index[k]), "motif", 6 + k, 0.04),
            ]
        ]
        components = {"music": trained, "motif": recall}
        self.expectation_parameters = 0
        extra_sets = {}
        self.expectation_cues = None
        if self.intuition.tables is not None:
            # Learned association strengths are actual seams in the joint settlement.
            chord_prob = self.intuition.probability(
                self.intuition.tables["chords"]
            ).reshape(48, 24)
            rhythm_prob = self.intuition.probability(
                self.intuition.tables["rhythm"]
            ).reshape(144, 12)
            pre = np.r_[np.repeat(np.arange(48), 24), np.repeat(np.arange(48, 192), 12)]
            post = np.r_[
                np.tile(np.arange(192, 216), 48), np.tile(np.arange(216, 228), 144)
            ]
            weights = []
            for prob in [chord_prob, rhythm_prob]:
                logp = np.log(prob)
                weights.extend(
                    np.clip(
                        (logp - logp.mean(1, keepdims=True)) * 0.25, -1.2, 1.2
                    ).ravel()
                )
            components["expectation"] = cd.Wiring.from_edges(
                228, pre=pre, post=post, sign=weights
            )
            for slot, width, start in [(2, 24, 192), (1, 12, 216)]:
                for k in range(width):
                    output = int(self.output_index[OFFSETS[slot] + k])
                    bridges += [
                        ("expectation", start + k, "music", output, 0.35),
                        ("music", output, "expectation", start + k, 0.04),
                    ]
            offset = n + 67
            extra_sets = {
                "expectation_cue": tuple(range(offset, offset + 192)),
                "harmonic_expectation": tuple(range(offset + 192, offset + 216)),
                "rhythmic_expectation": tuple(range(offset + 216, offset + 228)),
            }
            self.expectation_cues = np.arange(offset, offset + 192)
            self.expectation_parameters = len(weights)
        joint = couple(components, bridges)
        # Flat, disjoint region labels preserve the trained components and add memory ports.
        wiring = cd.Wiring(
            joint.n,
            joint.pre,
            joint.post,
            joint.count,
            joint.sign,
            {
                **w.sets,
                "motif_cue": tuple(range(n, n + 6)),
                "motif_recall": tuple(range(n + 6, n + 67)),
                **extra_sets,
            },
            "composer-with-motif",
        )
        self.engine = cd.Settlement(
            wiring, base.rule, bias=np.r_[base.bias, np.zeros(joint.n - n)]
        )
        self.cues = np.arange(n, n + 6)
        self.recall = np.arange(n + 6, n + 67)
        self.memory_edges = np.flatnonzero(
            (wiring.pre >= n) & (wiring.pre < n + 6) & (wiring.post >= n + 6)
        )
        self.count = 0

    def remember(self, pitches):
        for i, pitch in enumerate(pitches[:6]):
            key = np.zeros((1, 6))
            key[0, i] = 1
            target = np.zeros((1, 61))
            target[0, int(pitch) - 36] = 1
            self.memory.observe(key, target)
        self.count = min(6, len(pitches))
        w = self.engine.wiring
        scales = self.engine.edge_scale.copy()
        for edge in self.memory_edges:
            scales[edge] = (
                3.5
                * self.memory.strength[
                    0, w.pre[edge] - self.cues[0], w.post[edge] - self.recall[0]
                ]
            )
        self.engine = self.engine.with_parameters(edge_scale=scales)

    def cue(self, drive, position):
        # Recall a recognizable opening, then release it so the phrase can develop.
        if self.count and position < self.count:
            drive[:, self.cues[position]] = 2.5 if position < 3 else 1.2

    def expect(self, drive, brief, contexts, positions):
        if self.expectation_cues is None:
            return
        for row, (context, position) in enumerate(zip(contexts, positions)):
            previous_chord = min(23, int(context[-1][2]))
            previous_duration = int(context[-1][1])
            drive[row, self.expectation_cues[brief.mode * 24 + previous_chord]] = 2.5
            rhythm = (brief.arousal * 4 + int(position) % 4) * 12 + previous_duration
            drive[row, self.expectation_cues[48 + rhythm]] = 2.5

    def parameters(self):
        return self.base.parameters() + 366 + self.expectation_parameters
