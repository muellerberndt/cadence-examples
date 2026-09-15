"""Two online world models behind one interface: a multilayer perceptron and a transformer.

A model reads what the candidate's world head reads: every observation field scaled by its
declared bounds, an unobserved value read as zero, one missing flag per field where the
stage declares flags, the goal, the executed action and the reads of the declared stores.
It predicts every prediction field of the stage: class probabilities for a categorical
field and a value inside the field's bounds for a continuous field.

``learn`` takes one witnessed transition and makes one optimizer step on it. With a replay
ring it also stores the transition and makes one more step on each of ``replay``
transitions drawn uniformly from the ones stored before it, which is the protocol of
``arm.brain.ModelController``. The loss is read per field: softmax cross entropy for a
categorical field, squared error over the observed entries for a continuous field, and
their mean over the fields the transition exposes.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import pairwise
from typing import Any

import numpy as np
import torch

from agent.brain import Field, GraphSpec
from agent.life import Moment

Observation = Mapping[str, np.ndarray]
Prediction = dict[str, np.ndarray]


@dataclass(frozen=True)
class Reading:
    """The fields a model reads and the fields it predicts."""

    observation: tuple[Field, ...]
    prediction: tuple[Field, ...]
    actions: int
    goal: int = 0
    reads: int = 0
    flags: bool = False

    @classmethod
    def of(cls, spec: GraphSpec, *, goal: bool = False, flags: bool = False, reads: bool = False) -> Reading:
        """The reading of a stage's graph: the same fields, actions, goal and store reads."""
        return cls(
            observation=spec.observation,
            prediction=spec.prediction,
            actions=int(np.prod(spec.action_fields)),
            goal=spec.goal if goal else 0,
            reads=spec.episodic if reads else 0,
            flags=flags,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "observation_fields": [f.name for f in self.observation],
            "prediction_fields": [f.name for f in self.prediction],
            "actions": self.actions,
            "goal": self.goal,
            "reads": self.reads,
            "flags": self.flags,
        }


class Encoder:
    """One reading as numbers: a flat row for the perceptron, one token per field for the
    transformer. The tokens are the observation fields with their missing flag, the goal,
    the action and the store reads, in that order; the flat row is their concatenation."""

    def __init__(self, reading: Reading) -> None:
        self.reading = reading
        self.names = [f.name for f in reading.observation]
        self.lo = np.concatenate([np.full(f.width, f.lo if f.kind == "continuous" else 0.0) for f in reading.observation])
        span = np.concatenate([np.full(f.width, f.hi - f.lo if f.kind == "continuous" else 1.0) for f in reading.observation])
        self.scale = 1.0 / span
        widths = [f.width for f in reading.observation]
        self.widths = widths
        self.starts = np.concatenate([[0], np.cumsum(widths)[:-1]]).astype(np.int64)
        token_widths = [w + int(reading.flags) for w in widths]
        if reading.goal:
            token_widths.append(reading.goal)
        token_widths.append(reading.actions)
        if reading.reads:
            token_widths.append(reading.reads)
        self.token_widths = token_widths
        self.tokens = len(token_widths)
        self.width = int(sum(token_widths))
        bases = np.concatenate([[0], np.cumsum(token_widths)[:-1]]).astype(np.int64)
        self.values = np.concatenate([np.arange(bases[k], bases[k] + widths[k]) for k in range(len(widths))])
        self.flags = bases[: len(widths)] + np.asarray(widths) if reading.flags else np.zeros(0, np.int64)
        at = len(widths)
        self.goal_at = int(bases[at]) if reading.goal else 0
        at += 1 if reading.goal else 0
        self.action_at = int(bases[at])
        at += 1
        self.reads_at = int(bases[at]) if reading.reads else 0
        # the flat row scattered into tokens: one index pair per entry of the row
        self.token_of = np.concatenate([np.full(w, t) for t, w in enumerate(token_widths)])
        self.slot_of = np.concatenate([np.arange(w) for w in token_widths])

    def encode(
        self,
        observations: Sequence[Observation],
        actions: Sequence[int],
        observed: Sequence[Mapping[str, np.ndarray] | None] | None,
        goal: np.ndarray | None,
        reads: np.ndarray | None,
    ) -> np.ndarray:
        """The rows of a batch: ``(len(observations), width)``."""
        r = self.reading
        out = np.zeros((len(observations), self.width))
        for i, observation in enumerate(observations):
            values = (np.concatenate([np.asarray(observation[n], dtype=float) for n in self.names]) - self.lo) * self.scale
            flags = None if observed is None else observed[i]
            if flags is not None:
                seen = np.concatenate([np.asarray(flags[n], dtype=bool) if n in flags else np.ones(w, bool) for n, w in zip(self.names, self.widths, strict=True)])
                values = values * seen
                if r.flags:
                    out[i, self.flags] = ~np.logical_and.reduceat(seen, self.starts)
            out[i, self.values] = values
            if r.goal and goal is not None:
                out[i, self.goal_at : self.goal_at + r.goal] = np.asarray(goal, dtype=float)
            out[i, self.action_at + int(actions[i])] = 1.0
            if r.reads and reads is not None:
                out[i, self.reads_at : self.reads_at + r.reads] = np.asarray(reads, dtype=float)
        return out

    def tokenize(self, rows: np.ndarray) -> np.ndarray:
        """A batch of flat rows as ``(batch, tokens, widest token)``, zero where a token is short."""
        out = np.zeros((len(rows), self.tokens, max(self.token_widths)), dtype=np.float32)
        out[:, self.token_of, self.slot_of] = rows
        return out


def softmax(values: np.ndarray) -> np.ndarray:
    z = np.exp(values - values.max())
    return z / z.sum()


def targets_from_moment(fields: Sequence[Field], moment: Moment) -> dict[str, Any]:
    """What a moment exposes for each prediction field, the way the candidate reads it:
    a categorical field only when every entry is observed, a continuous field with the mask
    of its observed entries, the reward only when the moment carries one."""
    out: dict[str, Any] = {}
    for f in fields:
        if f.source == "reward":
            if moment.reward_known:
                out[f.name] = np.array([moment.reward])
            continue
        if f.source == "terminal":
            out[f.name] = np.array([1 if moment.terminated else 0])
            continue
        if f.source not in moment.observation:
            continue
        value = np.asarray(moment.observation[f.source], dtype=float)
        known = np.asarray(moment.observed[f.source], dtype=bool)
        if f.kind == "categorical":
            if known.all():
                out[f.name] = value
        elif known.any():
            out[f.name] = (value, known)
    return out


class OnlineModel:
    """The interface both networks share: encoding, decoding, the replay ring and the
    bookkeeping of updates. A subclass supplies ``_outputs`` and ``_update``."""

    kind = "model"

    def __init__(self, reading: Reading, *, replay: int = 0, capacity: int = 4096, seed: int = 0) -> None:
        self.reading = reading
        self.encoder = Encoder(reading)
        self.seed = int(seed)
        self.slices: dict[str, slice] = {}
        at = 0
        for f in reading.prediction:
            self.slices[f.name] = slice(at, at + f.width)
            at += f.width
        self.outputs = at
        self.replay = int(replay)
        self.capacity = int(capacity)
        self.ring: list[tuple[np.ndarray, list]] = []
        self.ring_at = 0
        self.ring_rng = np.random.default_rng([self.seed, 7919])
        self.updates = 0
        self.replay_updates = 0
        self.losses: list[float] = []

    # -- reading and writing

    def _encode(self, observations, actions, observed, goal, reads) -> np.ndarray:
        return self.encoder.encode(observations, actions, observed, goal, reads)

    def _encode_targets(self, targets: Mapping[str, Any]) -> list:
        """Per exposed field: its slice, whether it is categorical, the class or the scaled
        values, and the mask of observed entries."""
        out = []
        for f in self.reading.prediction:
            if f.name not in targets:
                continue
            value = targets[f.name]
            known = None
            if isinstance(value, tuple):
                value, known = value
            array = np.asarray(value, dtype=float)
            if f.kind == "categorical":
                out.append((self.slices[f.name], True, int(array.argmax()) if array.ndim else int(array), None))
            else:
                scaled = (array - f.lo) / (f.hi - f.lo)
                mask = np.ones(f.width) if known is None else np.asarray(known, dtype=float)
                out.append((self.slices[f.name], False, scaled, mask))
        return out

    def _decode(self, row: np.ndarray) -> Prediction:
        out: Prediction = {}
        for f in self.reading.prediction:
            value = row[self.slices[f.name]]
            out[f.name] = softmax(value) if f.kind == "categorical" else f.lo + np.clip(value, 0.0, 1.0) * (f.hi - f.lo)
        return out

    # -- the interface the planners use

    def predict_batch(
        self,
        observations: Sequence[Observation],
        actions: Sequence[int],
        *,
        observed: Sequence[Mapping[str, np.ndarray] | None] | None = None,
        goal: np.ndarray | None = None,
        reads: np.ndarray | None = None,
    ) -> list[Prediction]:
        rows = self._outputs(self._encode(observations, actions, observed, goal, reads))
        return [self._decode(row) for row in rows]

    def predict(self, observation: Observation, action: int, *, observed=None, goal=None, reads=None) -> Prediction:
        return self.predict_batch([observation], [action], observed=None if observed is None else [observed], goal=goal, reads=reads)[0]

    def learn(self, observation: Observation, action: int, targets: Mapping[str, Any], *, observed=None, goal=None, reads=None) -> float:
        """One step on the witnessed transition, then one step on each replayed transition."""
        encoded = self._encode_targets(targets)
        if not encoded:
            return float("nan")
        row = self._encode([observation], [action], None if observed is None else [observed], goal, reads)[0]
        loss = self._update(row, encoded)
        self.updates += 1
        self.losses.append(loss)
        for _ in range(self.replay if self.ring else 0):
            stored, stored_targets = self.ring[int(self.ring_rng.integers(len(self.ring)))]
            self._update(stored, stored_targets)
            self.replay_updates += 1
        if self.replay:
            self._store(row, encoded)
        return loss

    def _store(self, row: np.ndarray, targets: list) -> None:
        if len(self.ring) < self.capacity:
            self.ring.append((row, targets))
            return
        self.ring[self.ring_at] = (row, targets)
        self.ring_at = (self.ring_at + 1) % self.capacity

    def loss(self, observation: Observation, action: int, targets: Mapping[str, Any], *, observed=None, goal=None, reads=None) -> float:
        """The loss of one transition under the current parameters; nothing moves."""
        encoded = self._encode_targets(targets)
        row = self._encode([observation], [action], None if observed is None else [observed], goal, reads)[0]
        return self._loss_of(self._outputs(row[None])[0], encoded)

    def _loss_of(self, out: np.ndarray, targets: list) -> float:
        losses = []
        for where, categorical, target, mask in targets:
            value = out[where]
            if categorical:
                losses.append(-float(np.log(max(softmax(value)[target], 1e-12))))
            else:
                error = (value - target) * mask
                losses.append(float((error**2).sum() / max(mask.sum(), 1.0)))
        return float(np.mean(losses)) if losses else float("nan")

    def _gradient(self, out: np.ndarray, targets: list) -> tuple[float, np.ndarray]:
        """The loss of one row and its gradient at the outputs."""
        gradient = np.zeros_like(out)
        losses = []
        fields = len(targets)
        for where, categorical, target, mask in targets:
            value = out[where]
            if categorical:
                p = softmax(value)
                losses.append(-float(np.log(max(p[target], 1e-12))))
                p[target] -= 1.0
                gradient[where] = p / fields
            else:
                error = (value - target) * mask
                known = max(mask.sum(), 1.0)
                losses.append(float((error**2).sum() / known))
                gradient[where] = error * 2.0 / (known * fields)
        return float(np.mean(losses)), gradient

    # -- copies

    def copy(self, *, ring: bool = True) -> OnlineModel:
        """An isolated copy; without ``ring`` the stored transitions stay behind."""
        held = self.ring
        if not ring:
            self.ring = []
        try:
            out = copy.deepcopy(self)
        finally:
            self.ring = held
        return out

    # -- what a receipt records

    def parameters(self) -> int:
        raise NotImplementedError

    def describe(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "parameters": self.parameters(),
            "inputs": self.encoder.width,
            "tokens": self.encoder.tokens,
            "outputs": self.outputs,
            "replay": self.replay,
            "capacity": self.capacity,
            "seed": self.seed,
            "reading": self.reading.to_dict(),
        }

    def counts(self) -> dict[str, int]:
        return {"updates": self.updates, "replay_updates": self.replay_updates, "stored": len(self.ring)}

    def _outputs(self, rows: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def _update(self, row: np.ndarray, targets: list) -> float:
        raise NotImplementedError


class OnlineMLP(OnlineModel):
    """The arm's forward model generalised: one or two tanh hidden layers, a linear readout
    per prediction field, Adam on one transition at a time. With one hidden layer, no goal,
    no flags and no store reads it is ``arm.brain.OnlineMLP``."""

    kind = "mlp"

    def __init__(self, reading: Reading, *, hidden: Sequence[int] = (64,), lr: float = 1e-3, replay: int = 0, capacity: int = 4096, seed: int = 0) -> None:
        super().__init__(reading, replay=replay, capacity=capacity, seed=seed)
        self.hidden = [int(h) for h in hidden]
        self.lr = float(lr)
        rng = np.random.default_rng(seed)
        sizes = [self.encoder.width, *self.hidden]
        self.weights = [rng.normal(0.0, np.sqrt(2 / a), (a, b)) for a, b in pairwise(sizes)]
        self.biases = [np.zeros(b) for b in sizes[1:]]
        self.weights.append(np.zeros((sizes[-1], self.outputs)))
        self.biases.append(np.zeros(self.outputs))
        self.params = [p for layer in zip(self.weights, self.biases, strict=True) for p in layer]
        self.first = [np.zeros_like(p) for p in self.params]
        self.second = [np.zeros_like(p) for p in self.params]
        self.steps = 0

    def _outputs(self, rows: np.ndarray) -> np.ndarray:
        h = rows
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            h = np.tanh(h @ weight + bias)
        return h @ self.weights[-1] + self.biases[-1]

    def _update(self, row: np.ndarray, targets: list) -> float:
        states = [row]
        h = row
        for weight, bias in zip(self.weights[:-1], self.biases[:-1], strict=True):
            h = np.tanh(h @ weight + bias)
            states.append(h)
        out = h @ self.weights[-1] + self.biases[-1]
        loss, delta = self._gradient(out, targets)
        gradients: list[np.ndarray] = []
        for layer in range(len(self.weights) - 1, -1, -1):
            gradients = [np.outer(states[layer], delta), delta, *gradients]
            if layer:
                delta = (delta @ self.weights[layer].T) * (1 - states[layer] ** 2)
        self.steps += 1
        for p, g, first, second in zip(self.params, gradients, self.first, self.second, strict=True):
            first *= 0.9
            first += 0.1 * g
            second *= 0.999
            second += 0.001 * g * g
            p -= self.lr * (first / (1 - 0.9**self.steps)) / (np.sqrt(second / (1 - 0.999**self.steps)) + 1e-8)
        return loss

    def parameters(self) -> int:
        return int(sum(p.size for p in self.params))

    def describe(self) -> dict[str, Any]:
        return {**super().describe(), "hidden": list(self.hidden), "lr": self.lr}


class _Block(torch.nn.Module):
    """One pre-norm transformer layer: self attention, then a feedforward."""

    def __init__(self, width: int, heads: int, feedforward: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = torch.nn.LayerNorm(width)
        self.qkv = torch.nn.Linear(width, 3 * width)
        self.out = torch.nn.Linear(width, width)
        self.norm2 = torch.nn.LayerNorm(width)
        self.up = torch.nn.Linear(width, feedforward)
        self.down = torch.nn.Linear(feedforward, width)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, width = x.shape
        parts = self.qkv(self.norm1(x)).view(batch, tokens, 3, self.heads, width // self.heads)
        q, k, v = parts.permute(2, 0, 3, 1, 4)
        attended = torch.nn.functional.scaled_dot_product_attention(q, k, v)
        x = x + self.out(attended.transpose(1, 2).reshape(batch, tokens, width))
        return x + self.down(torch.nn.functional.gelu(self.up(self.norm2(x))))


class _Tokens(torch.nn.Module):
    """One token per field: a linear embedding of the field's entries plus a learned field
    embedding, two pre-norm layers, and every prediction field read from the pooled token."""

    def __init__(self, token_widths: Sequence[int], outputs: int, width: int, heads: int, layers: int, feedforward: int) -> None:
        super().__init__()
        count, widest = len(token_widths), max(token_widths)
        mask = torch.zeros(count, widest)
        weight = torch.zeros(count, widest, width)
        for t, w in enumerate(token_widths):
            mask[t, :w] = 1.0
            bound = 1.0 / np.sqrt(w)
            torch.nn.init.uniform_(weight[t, :w], -bound, bound)
        self.embedding = torch.nn.Parameter(weight)
        self.register_buffer("mask", mask[..., None])
        self.field = torch.nn.Parameter(torch.randn(count, width) * 0.02)
        self.blocks = torch.nn.ModuleList(_Block(width, heads, feedforward) for _ in range(layers))
        self.norm = torch.nn.LayerNorm(width)
        self.head = torch.nn.Linear(width, outputs)
        torch.nn.init.zeros_(self.head.weight)
        torch.nn.init.zeros_(self.head.bias)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = torch.einsum("btk,tkd->btd", tokens, self.embedding * self.mask) + self.field
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x).mean(dim=1))


class OnlineTransformer(OnlineModel):
    """A small transformer over the reading's tokens, trained by Adam on one transition at
    a time. The tokens are the observation fields, the goal, the action and the store reads."""

    kind = "transformer"

    def __init__(
        self,
        reading: Reading,
        *,
        width: int = 64,
        heads: int = 4,
        layers: int = 2,
        feedforward: int = 256,
        lr: float = 3e-4,
        replay: int = 0,
        capacity: int = 4096,
        seed: int = 0,
    ) -> None:
        super().__init__(reading, replay=replay, capacity=capacity, seed=seed)
        self.width, self.heads, self.layers, self.feedforward, self.lr = width, heads, layers, feedforward, float(lr)
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(self.seed)
            self.net = _Tokens(self.encoder.token_widths, self.outputs, width, heads, layers, feedforward)
        try:  # the fused step is half the cost of the loop over parameter tensors
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr, fused=True)
            self.adam = "fused"
        except (RuntimeError, ValueError):
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
            self.adam = "loop"

    def _tokens(self, rows: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(self.encoder.tokenize(rows))

    def _outputs(self, rows: np.ndarray) -> np.ndarray:
        with torch.inference_mode():
            return self.net(self._tokens(rows)).to(torch.float64).numpy()

    def _torch_loss(self, out: torch.Tensor, targets: list) -> torch.Tensor:
        losses = []
        for where, categorical, target, mask in targets:
            value = out[where]
            if categorical:
                losses.append(-torch.log_softmax(value, 0)[target])
            else:
                error = (value - torch.from_numpy(target).float()) * torch.from_numpy(mask).float()
                losses.append((error**2).sum() / max(float(mask.sum()), 1.0))
        return torch.stack(losses).mean()

    def _update(self, row: np.ndarray, targets: list) -> float:
        loss = self._torch_loss(self.net(self._tokens(row[None]))[0], targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        return float(loss.detach())

    def parameters(self) -> int:
        """Trainable entries; the padding of a short token's embedding never takes part."""
        entries = int(sum(p.numel() for name, p in self.net.named_parameters() if name != "embedding"))
        return entries + int(sum(self.encoder.token_widths)) * self.width

    def describe(self) -> dict[str, Any]:
        return {
            **super().describe(),
            "width": self.width,
            "heads": self.heads,
            "layers": self.layers,
            "feedforward": self.feedforward,
            "lr": self.lr,
            "adam": self.adam,
        }
