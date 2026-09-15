"""One brain that observes, remembers, predicts, acts and learns through real events.

The graph is developed from a ``Genome`` (sensory, goal, efference, workspace, context,
recall, dynamics, prediction and motor regions) and settled as one ``Brain``. The world
head is a records cortex (``cadence.Records``) when ``AgentConfig.records`` is set: the
reading of the sensory, goal, action and recall neurons is coded sparsely, each prediction
field is read from the active cells' records, and the witnessed consequence is written into
those records by the delta rule, with no settling and no replay. Without records the world
head repairs the prediction neurons by masked free and nudged contrasts. The motor head moves
only the workspace-to-motor synapses, from reward, through per-synapse eligibility and a
broadcast prediction error. Carried context is a ``Trace`` of the workspace; declared stores
enter the recall neurons through ``recall_drive``.

``Agent.step`` is the event transaction: validate, score the stored prediction, credit the
executed action, repair the world head, store evidence, advance context, decide, predict,
commit. Imagination (``predict``, ``predict_batch``) reads the current records, parameters
and memories and writes nothing. Every settling phase records its steps and its equation
residual.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from typing import Any

import cadence as cd
import numpy as np
from cadence.brain import BrainState, Nudge

from .life import Decision, Moment

M32 = 0xFFFFFFFF
CONTROLLERS = ("actor", "planner", "exploration", "demonstration")


class Mulberry32(cd.Mulberry32):
    """The generator of ``brain_scan.js``, with a categorical draw, so a page can replay a life."""

    def choice(self, probabilities: np.ndarray) -> int:
        u = self.random()
        total = 0.0
        for k, p in enumerate(probabilities):
            total += float(p)
            if u < total:
                return k
        return int(len(probabilities) - 1)


# -- specification


@dataclass(frozen=True)
class Field:
    """One named group of neurons: ``categorical`` (one neuron per class) or ``continuous``
    (one neuron per value, with fixed sensor bounds ``lo``..``hi``)."""

    name: str
    kind: str
    width: int
    lo: float = 0.0
    hi: float = 1.0
    source: str = ""  # prediction fields: the observation field (or "reward"/"terminal") they predict

    def __post_init__(self) -> None:
        if self.kind not in ("categorical", "continuous"):
            raise ValueError("kind must be categorical or continuous")
        if isinstance(self.width, bool) or not isinstance(self.width, int) or self.width < 1:
            raise ValueError("width must be a positive integer")
        if not np.isfinite([self.lo, self.hi]).all() or self.hi <= self.lo:
            raise ValueError("continuous bounds must be finite with hi above lo")


@dataclass(frozen=True)
class GraphSpec:
    observation: tuple[Field, ...]
    prediction: tuple[Field, ...]
    action_fields: tuple[int, ...] = (2,)
    goal: int = 0
    perceptual: int = 0  # 0: sources project straight into the workspace (symbolic input)
    workspace: int = 64
    dynamics: int = 32
    episodic: int = 0
    intention: int = 0
    missing_flags: bool = True  # extra sensory neurons driven when a value is unobserved
    flags_per_field: bool = False  # one flag neuron per field (any entry unobserved) instead of one per value
    density: float = 1.0  # W-D and other large association projections
    scale: float = 1.0  # reciprocal (loop-forming) projections, fan-scaled
    source_scale: float = 1.0  # one-way projections out of source ports, fan-scaled
    sensory_to_dynamics: bool = True  # the dynamics cortex also reads the sensory copy directly
    bias: float = 0.25  # declared prior on free neurons; source neurons stay at 0
    input_gain: float = 2.0  # drive per unit observed value on a source neuron
    field_gains: dict[str, float] = field(default_factory=dict)  # per observation field, a factor on input_gain (its flag keeps input_gain)
    action_gain: float | None = None  # drive of the chosen action's neurons; None: input_gain
    readout_init: float = 0.0  # initial magnitude of the readout pairs (dynamics-prediction, workspace-motor)
    dt: float = 0.5
    slope: float = 1.0  # the neuron model's sigmoid slope: larger bends the units sooner (products need bends)
    lateral: float = 0.0  # fixed inhibitory synapses within workspace and dynamics (fan-scaled magnitude): sparser codes

    @property
    def actions(self) -> int:
        return int(sum(self.action_fields))

    @property
    def observation_width(self) -> int:
        values = sum(f.width for f in self.observation)
        if not self.missing_flags:
            return int(values)
        return int(values + (len(self.observation) if self.flags_per_field else values))

    @property
    def prediction_width(self) -> int:
        return int(sum(f.width for f in self.prediction))

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["observation"] = [asdict(f) for f in self.observation]
        out["prediction"] = [asdict(f) for f in self.prediction]
        out["action_fields"] = list(self.action_fields)
        return out

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> GraphSpec:
        d = dict(d)
        d["observation"] = tuple(Field(**f) for f in d["observation"])
        d["prediction"] = tuple(Field(**f) for f in d["prediction"])
        d["action_fields"] = tuple(int(k) for k in d["action_fields"])
        return cls(**d)


@dataclass(frozen=True)
class WorldConfig:
    beta: float = 0.05
    eta: float = 0.02
    eta_bias: float = 0.002
    momentum: float = 0.0
    normalize: float = 0.0
    temperature: float = 0.2  # softmax temperature of categorical prediction fields
    label_smoothing: float = 0.05  # categorical targets: (1 - s) one-hot + s / classes
    reject_unconverged: bool = True  # an update whose phases missed the residual tolerance is dropped
    max_phase_gap: float = 0.3  # plus/minus phases further apart than this sit in different attractors: rejected
    step_clip: float = 0.1  # largest change of one efficacy in one update
    fan_in_rate: bool = True  # each synapse's step is divided by its postsynaptic neuron's plastic fan-in
    free_steps: int = 400  # step budget of a free phase
    nudged_steps: int = 400  # step budget of a nudged phase
    chunk: int = 16  # steps between residual checks
    tolerance: float = 1e-7  # activation-movement tolerance of the raw-settle readouts (imagination)
    imagine_tolerance: float | None = None  # a looser tolerance for imagined consequences; None: tolerance
    residual_tolerance: float = 1e-6  # the equilibrium claim, checked on every phase

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ActorConfig:
    beta: float = 0.05
    eta: float = 0.5
    eta_bias: float = 0.05
    eta_critic: float = 0.05
    temperature: float = 0.2
    gamma: float = 0.99
    lam: float = 0.9
    delta_cap: float = 5.0
    step_clip: float = 0.1  # largest change of one efficacy in one update
    epsilon: float = 0.0  # probability of a uniformly random legal action (controller "exploration")
    critic_normalize: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class EpisodicConfig:
    key: str = "observation"  # the port whose drive is the key: observation | workspace
    decay: float = 1.0
    rate: float = 1.0
    amplitude: float = 1.0
    reset_on_episode: bool = True

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ReplayConfig:
    """A bounded ring of raw observed transitions (drive, warm state, observed targets).

    Each real transition's world repair also replays ``per_transition`` stored transitions
    drawn uniformly from the ring, in the same batch; the stored drive carries the exact
    context of its moment. Replay teaches the world head only; it never rewards the actor."""

    capacity: int = 4096
    per_transition: int = 1

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RecordsConfig:
    """The world head as records (Marr 1969, Albus 1971): a mean-free reading of the source
    neurons, a fixed sparse expansion with ``active`` winners, and delta-rule records per
    prediction field. A reading touches few records; the outcome is written into exactly those."""

    granules: int = 8000
    active: int = 40  # winners per reading (lateral inhibition)
    rate: float = 0.2  # consequence records
    reward_rate: float = 1.0  # reward and terminal records (one exposure writes)
    habituation: float = 1e-5  # the slowest rate of the input units' running mean (the plain average until 1/n reaches it); 0 subtracts nothing
    bias_scale: float = 0.3
    context: bool = False  # read the context trace too (the stores carry memory; the trace varies the code)
    normalize_blocks: bool = True  # the valued code: divisive normalisation per input pathway (observation, goal, action, recall) so the goal and the recall reads have an equal say in the code the reward and terminal records read
    block_rate: float = 0.002  # running-norm rate of the pathways
    task_sets: bool = False  # the goal port's active unit selects the group of cells the valued code draws from (one group per goal value)
    pathways: str = "ports"  # the input pathways of the code: "ports" (observation, goal, action, recall, context) or "fields" (each observation field with its flag, then goal, action, recall, context)
    fan_in: int = 0  # each cell reads this many pathways, drawn at birth, and every input outside them; 0 reads every input

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentConfig:
    graph: GraphSpec
    world: WorldConfig = field(default_factory=WorldConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    episodic: EpisodicConfig | None = None
    replay: ReplayConfig | None = None
    records: RecordsConfig | None = None  # a records world head instead of the settled prediction port
    context_decay: float = 0.8
    context_amplitude: float = 1.0
    controller: str = "actor"  # actor | planner
    learning: bool = True  # False: the identical architecture, frozen from birth
    world_learning: bool = True
    motor_learning: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "graph": self.graph.to_dict(),
            "world": self.world.to_dict(),
            "actor": self.actor.to_dict(),
            "episodic": None if self.episodic is None else self.episodic.to_dict(),
            "replay": None if self.replay is None else self.replay.to_dict(),
            "records": None if self.records is None else self.records.to_dict(),
            "context_decay": self.context_decay,
            "context_amplitude": self.context_amplitude,
            "controller": self.controller,
            "learning": self.learning,
            "world_learning": self.world_learning,
            "motor_learning": self.motor_learning,
        }

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> AgentConfig:
        return cls(
            graph=GraphSpec.from_dict(d["graph"]),
            world=WorldConfig(**d.get("world", {})),
            actor=ActorConfig(**d.get("actor", {})),
            episodic=None if d.get("episodic") is None else EpisodicConfig(**d["episodic"]),
            replay=None if d.get("replay") is None else ReplayConfig(**d["replay"]),
            records=None if d.get("records") is None else RecordsConfig(**d["records"]),
            context_decay=float(d.get("context_decay", 0.8)),
            context_amplitude=float(d.get("context_amplitude", 1.0)),
            controller=str(d.get("controller", "actor")),
            learning=bool(d.get("learning", True)),
            world_learning=bool(d.get("world_learning", True)),
            motor_learning=bool(d.get("motor_learning", True)),
        )


# -- the graph


@dataclass
class Ports:
    """Index arrays of every port and field of the developed connectome."""

    observation: np.ndarray
    goal: np.ndarray
    action: np.ndarray
    perception: np.ndarray
    workspace: np.ndarray
    context: np.ndarray
    recall: np.ndarray
    dynamics: np.ndarray
    prediction: np.ndarray
    motor: np.ndarray
    intention: np.ndarray
    observation_fields: dict[str, np.ndarray]
    observation_flags: dict[str, np.ndarray]
    prediction_fields: dict[str, np.ndarray]
    action_slots: list[np.ndarray]
    motor_slots: list[np.ndarray]

    @property
    def sources(self) -> np.ndarray:
        return np.concatenate([self.observation, self.goal, self.action, self.context, self.recall])


ROLES = {
    "sensory": "sensory",
    "goal": "value",
    "efference": "motor",
    "perception": "association",
    "workspace": "association",
    "context": "memory",
    "recall": "memory",
    "dynamics": "association",
    "prediction": "sensory",
    "motor": "motor",
    "intention": "association",
}

COLORS = {  # one colour per cortex on a page; roles alone would merge the association areas
    "sensory": (143, 225, 157),
    "goal": (237, 129, 182),
    "efference": (255, 140, 90),
    "perception": (120, 200, 255),
    "workspace": (89, 229, 203),
    "context": (171, 153, 255),
    "recall": (205, 170, 255),
    "dynamics": (255, 225, 120),
    "prediction": (255, 250, 200),
    "motor": (255, 191, 112),
    "intention": (190, 240, 180),
}


def build_connectome(spec: GraphSpec, seed: int = 0) -> tuple[cd.Connectome, Ports]:
    """Develop the foundation graph: sources feed a perceptual cortex and the workspace;
    the workspace couples reciprocally to dynamics and motor; dynamics reads the action
    copy and couples to the prediction readout."""
    one_way = spec.source_scale
    regions = [cd.Region("sensory", spec.observation_width)]
    projections = []
    if spec.perceptual:
        projections.append(cd.Projection("sensory", "perception", scale=one_way, reciprocal=False))
    else:
        projections.append(cd.Projection("sensory", "workspace", scale=one_way, reciprocal=False))
    if spec.sensory_to_dynamics:
        projections.append(cd.Projection("sensory", "dynamics", scale=one_way, reciprocal=False))
    if spec.goal:
        regions.append(cd.Region("goal", spec.goal))
        projections.append(cd.Projection("goal", "workspace", scale=one_way, reciprocal=False))
    regions.append(cd.Region("efference", spec.actions))
    if spec.perceptual:
        regions.append(cd.Region("perception", spec.perceptual))
        projections.append(cd.Projection("perception", "workspace", scale=spec.scale))
    regions += [cd.Region("workspace", spec.workspace), cd.Region("context", spec.workspace)]
    projections += [
        cd.Projection("efference", "dynamics", scale=one_way, reciprocal=False),
        cd.Projection("context", "workspace", scale=one_way, reciprocal=False),
    ]
    if spec.episodic:
        regions.append(cd.Region("recall", spec.episodic))
        projections.append(cd.Projection("recall", "workspace", scale=one_way, reciprocal=False))
    regions += [cd.Region("dynamics", spec.dynamics)]
    projections += [cd.Projection("workspace", "dynamics", density=spec.density, scale=spec.scale)]
    if spec.lateral:
        # lateral inhibition, fixed: the plastic masks name port pairs, never a region with itself
        projections += [
            cd.Projection("workspace", "workspace", sign=-1.0, scale=spec.lateral, reciprocal=False),
            cd.Projection("dynamics", "dynamics", sign=-1.0, scale=spec.lateral, reciprocal=False),
        ]
    if spec.prediction_width:
        regions.append(cd.Region("prediction", spec.prediction_width))
        projections.append(cd.Projection("dynamics", "prediction", scale=spec.scale))
    regions.append(cd.regions.motor_cortex(spec.actions, name="motor"))
    projections.append(cd.Projection("workspace", "motor", scale=spec.scale))
    if spec.intention:
        regions.append(cd.Region("intention", spec.intention))
        projections += [
            cd.Projection("intention", target, scale=spec.scale)
            for target in ("workspace", "dynamics", "motor")
        ]
    connectome = cd.develop(cd.Genome(tuple(regions), tuple(projections), label="experience"), seed=seed)
    pops = connectome.populations

    def port(name: str) -> np.ndarray:
        return np.asarray(pops[name], dtype=np.int64) if name in pops else np.zeros(0, np.int64)

    observation = port("sensory")
    fields: dict[str, np.ndarray] = {}
    flags: dict[str, np.ndarray] = {}
    at = 0
    for f in spec.observation:
        fields[f.name] = observation[at : at + f.width]
        at += f.width
    if spec.missing_flags:
        for f in spec.observation:
            width = 1 if spec.flags_per_field else f.width
            flags[f.name] = observation[at : at + width]
            at += width
    prediction = port("prediction")
    heads: dict[str, np.ndarray] = {}
    at = 0
    for f in spec.prediction:
        heads[f.name] = prediction[at : at + f.width]
        at += f.width
    action, motor = port("efference"), port("motor")
    action_slots, motor_slots = [], []
    at = 0
    for k in spec.action_fields:
        action_slots.append(action[at : at + k])
        motor_slots.append(motor[at : at + k])
        at += k
    ports = Ports(
        observation=observation,
        goal=port("goal"),
        action=action,
        perception=port("perception"),
        workspace=port("workspace"),
        context=port("context"),
        recall=port("recall"),
        dynamics=port("dynamics"),
        prediction=prediction,
        motor=motor,
        intention=port("intention"),
        observation_fields=fields,
        observation_flags=flags,
        prediction_fields=heads,
        action_slots=action_slots,
        motor_slots=motor_slots,
    )
    if np.isin(connectome.post, ports.sources).any():
        raise ValueError("a source port received synapses")
    return connectome, ports


def _between(connectome: cd.Connectome, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    pre, post = connectome.pre, connectome.post
    return (np.isin(pre, a) & np.isin(post, b)) | (np.isin(pre, b) & np.isin(post, a))


def plasticity_masks(connectome: cd.Connectome, p: Ports) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """World synapses (sensory to prediction, symmetric across every reciprocal pair) and
    motor synapses (workspace/intention to motor); free biases follow their owner."""
    world = np.zeros(connectome.synapses, bool)
    for a, b in (
        (p.observation, p.perception),
        (p.observation, p.workspace),
        (p.observation, p.dynamics),
        (p.perception, p.workspace),
        (p.goal, p.workspace),
        (p.context, p.workspace),
        (p.recall, p.workspace),
        (p.workspace, p.dynamics),
        (p.action, p.dynamics),
        (p.dynamics, p.prediction),
        (p.intention, p.workspace),
        (p.intention, p.dynamics),
    ):
        if len(a) and len(b):
            world |= _between(connectome, a, b)
    motor = np.zeros(connectome.synapses, bool)
    for a, b in ((p.workspace, p.motor), (p.intention, p.motor)):
        if len(a) and len(b):
            motor |= _between(connectome, a, b)
    n = connectome.n
    world_neurons = np.zeros(n, bool)
    world_neurons[np.concatenate([p.perception, p.workspace, p.dynamics, p.prediction, p.intention])] = True
    motor_neurons = np.zeros(n, bool)
    motor_neurons[p.motor] = True
    if (world & motor).any():
        raise ValueError("world and motor plasticity overlap")
    return world, motor, world_neurons, motor_neurons


# -- the records world head


VALUED_SOURCES = ("reward", "terminal")


class RecordsHead:
    """The world head as a records cortex: ``cadence.Records`` with the agent's field decoding.

    The consequence fields read the plain code; the reward and terminal fields read the valued
    code, in which every input pathway (observation, goal, action, recall) has an equal say,
    and are written at the fast rate. ``predict`` turns each read into class probabilities
    (the positive part, normalised) or a value in the field's bounds; ``learn`` writes a
    witnessed class as a one-hot target and a witnessed value through its observed mask."""

    def __init__(self, inputs: int, fields: tuple[Field, ...], config: RecordsConfig, seed: int, blocks: Sequence[np.ndarray] | None = None, tasks: np.ndarray | None = None) -> None:
        self.config = config
        self.fields = fields
        self.cortex = cd.Records(
            inputs, {f.name: f.width for f in fields}, cells=config.granules, active=config.active,
            rate=config.rate, valued=[f.name for f in fields if f.source in VALUED_SOURCES],
            valued_rate=config.reward_rate, habituation=config.habituation, bias=config.bias_scale,
            pathways=list(blocks or []) if config.normalize_blocks else (), pathway_rate=config.block_rate,
            tasks=tasks if config.task_sets and tasks is not None else (),
            fan_in=config.fan_in,
            seed=(seed * 7919 + 13) & M32,
        )

    @property
    def mean(self) -> np.ndarray:
        return self.cortex.mean

    @mean.setter
    def mean(self, value: np.ndarray) -> None:
        self.cortex.mean = np.asarray(value, dtype=float)

    @property
    def block_norm(self) -> np.ndarray:
        return self.cortex.pathway_norm

    @block_norm.setter
    def block_norm(self, value: np.ndarray) -> None:
        self.cortex.pathway_norm = np.asarray(value, dtype=float)

    @property
    def blocks(self) -> list[np.ndarray]:
        return self.cortex.pathways

    @property
    def records(self) -> dict[str, np.ndarray]:
        return self.cortex.tables

    @property
    def seen(self) -> int:
        return self.cortex.seen

    @seen.setter
    def seen(self, value: int) -> None:
        self.cortex.seen = int(value)

    @property
    def writes(self) -> int:
        return self.cortex.writes

    @writes.setter
    def writes(self, value: int) -> None:
        self.cortex.writes = int(value)

    def code(self, readings: np.ndarray, *, adapt: bool, valued: bool = True) -> np.ndarray:
        """The plain and valued codes of one or more readings: ``(2, batch, cells)``; with
        ``valued=False`` the plain code alone, for imagined readings whose consequences are read."""
        return self.cortex.code(readings, adapt=adapt, valued=valued)

    def predict(self, code: np.ndarray, *, valued: bool = True) -> dict[str, np.ndarray]:
        return self.predict_many(code[:, None, :], valued=valued)[0]

    def predict_many(self, codes: np.ndarray, *, valued: bool = True) -> list[dict[str, np.ndarray]]:
        """The predictions of ``(2, batch, cells)`` codes, one dict per row: every field is read once
        for the whole batch, then decoded per row."""
        reads = self.cortex.read(codes)
        batch = codes.shape[1]
        out: list[dict[str, np.ndarray]] = [{} for _ in range(batch)]
        for f in self.fields:
            if not valued and f.source in VALUED_SOURCES:
                continue
            y = reads[f.name]
            if f.kind == "categorical":
                p = np.maximum(y, 0.0)
                total = p.sum(axis=1, keepdims=True)
                decoded = np.where(total > 0, p / np.where(total > 0, total, 1.0), 1.0 / f.width)
            else:
                decoded = f.lo + (np.clip(y, 0.05, 0.8) - 0.05) / 0.75 * (f.hi - f.lo)
            for k in range(batch):
                out[k][f.name] = decoded[k]
        return out

    def learn(self, code: np.ndarray, targets: Mapping[str, tuple[np.ndarray, np.ndarray]]) -> int:
        values, known = {}, {}
        for f in self.fields:
            if f.name not in targets:
                continue
            target, observed = targets[f.name]
            if f.kind == "categorical":
                values[f.name] = np.eye(f.width)[int(target[0])]
            else:
                values[f.name] = np.asarray(target, dtype=float)
                known[f.name] = np.asarray(observed, dtype=bool)
        return self.cortex.write(code, values, known)

    def parameters(self) -> int:
        return self.cortex.parameters()


# -- per-stream records


@dataclass
class Pending:
    """A committed decision awaiting its real outcome, with everything credit needs."""

    decision: Decision
    drive: np.ndarray  # the pre-action drive (n,), with the executed action's copy set
    state_v: np.ndarray  # the prediction phase's potentials, to warm the repair phase
    state_a: np.ndarray
    value: float
    score_edges: np.ndarray | None  # actor eligibility of this decision (synapses,)
    score_bias: np.ndarray | None
    workspace: np.ndarray  # the workspace activation the critic read at decision time
    code: np.ndarray | None = None  # the records head's code of the executed reading


@dataclass
class Ledger:
    phases: dict[str, int] = field(default_factory=dict)
    steps: dict[str, int] = field(default_factory=dict)
    max_residual: dict[str, float] = field(default_factory=dict)
    unconverged: dict[str, int] = field(default_factory=dict)
    rejected_updates: int = 0
    clipped_deltas: int = 0
    deltas: int = 0
    real_transitions: int = 0
    replay_writes: int = 0
    episodic_writes: int = 0
    record_writes: int = 0
    imagined: int = 0

    def record(self, phase: str, steps: int, residual: float, tolerance: float) -> None:
        self.phases[phase] = self.phases.get(phase, 0) + 1
        self.steps[phase] = self.steps.get(phase, 0) + int(steps)
        self.max_residual[phase] = max(self.max_residual.get(phase, 0.0), float(residual))
        if not residual <= tolerance:
            self.unconverged[phase] = self.unconverged.get(phase, 0) + 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "phases": dict(self.phases),
            "mean_steps": {k: self.steps[k] / max(1, self.phases[k]) for k in self.phases},
            "max_residual": dict(self.max_residual),
            "unconverged_phases": dict(self.unconverged),
            "rejected_updates": self.rejected_updates,
            "clipped_deltas": self.clipped_deltas,
            "deltas": self.deltas,
            "real_transitions": self.real_transitions,
            "replay_writes": self.replay_writes,
            "episodic_writes": self.episodic_writes,
            "record_writes": self.record_writes,
            "imagined": self.imagined,
        }


def _rows(state: BrainState, rows: np.ndarray) -> BrainState:
    return BrainState(state.v[rows], state.activation[rows], state.adaptation[rows], state.steps)


Planner = Callable[["Agent", int, Moment, np.ndarray, BrainState], tuple[int | None, dict[str, np.ndarray], dict[str, int]]]


# -- the agent


class Agent:
    """One life: fixed streams (parallel bodies), one brain, one parameter set, all memories."""

    def __init__(
        self,
        config: AgentConfig,
        *,
        streams: int = 1,
        seed: int = 0,
        stream_seeds: Sequence[int] | None = None,
        planner: Planner | None = None,
        backend: str = "cpu",
    ) -> None:
        if isinstance(streams, bool) or not isinstance(streams, int) or streams < 1:
            raise ValueError("streams must be a positive integer")
        if config.controller not in ("actor", "planner"):
            raise ValueError("controller must be actor or planner")
        if config.controller == "planner" and planner is None:
            raise ValueError("a planner controller needs a planner callable")
        self.config = config
        self.streams = streams
        self.seed = int(seed)
        self.planner = planner
        spec = config.graph
        self.connectome, self.ports = build_connectome(spec, seed=self.seed)
        n = self.connectome.n
        bias = np.full(n, spec.bias)
        bias[self.ports.sources] = 0.0
        efficacy = self.connectome.sign.copy()
        readout = _between(self.connectome, self.ports.workspace, self.ports.motor)
        if len(self.ports.prediction):
            readout |= _between(self.connectome, self.ports.dynamics, self.ports.prediction)
        efficacy[readout] *= spec.readout_init  # calibrated start: uniform predictions and policy
        brain = cd.Brain(self.connectome, cd.learning_neuron_model(dt=spec.dt, slope=spec.slope), bias=bias, efficacy=efficacy, backend=backend)
        world_edges, motor_edges, world_neurons, motor_neurons = plasticity_masks(self.connectome, self.ports)
        self.world_edges, self.motor_edges = world_edges, motor_edges
        w, a = config.world, config.actor
        rate = None
        if w.fan_in_rate:
            fan_in = np.bincount(self.connectome.post, weights=(world_edges | motor_edges).astype(float), minlength=n)
            rate = 1.0 / np.maximum(fan_in[self.connectome.post], 1.0)
        # Without prediction fields there is no world head; the motor head then owns the brain.
        self.world: cd.Learner | None = None
        if spec.prediction:
            self.world = cd.Learner(
                brain,
                list(self.ports.prediction),
                cd.LearnerConfig(
                    beta=w.beta, eta=w.eta, eta_bias=w.eta_bias, free_steps=w.free_steps,
                    nudged_steps=w.nudged_steps, tolerance=w.tolerance, momentum=w.momentum,
                    normalize=w.normalize, temperature=w.temperature,
                ),
                plastic_synapses=world_edges,
                plastic_neurons=world_neurons,
                synapse_rate=rate,
            )
        self.motor = cd.Learner(
            brain,
            list(self.ports.motor),
            cd.LearnerConfig(
                beta=a.beta, eta=a.eta, eta_bias=a.eta_bias, free_steps=w.free_steps,
                nudged_steps=w.nudged_steps, tolerance=w.tolerance, temperature=a.temperature,
            ),
            plastic_synapses=motor_edges,
            plastic_neurons=motor_neurons,
            synapse_rate=rate,
            slots=list(spec.action_fields),
        )
        self.context = cd.Trace(
            self.connectome, decay=config.context_decay, amplitude=config.context_amplitude,
            source="workspace", target="context",
        )
        self.context.reset(streams)
        self.episodic: cd.FastSynapses | None = None
        if config.episodic is not None:
            if not spec.episodic:
                raise ValueError("an episodic store needs recall neurons")
            key = self.ports.observation if config.episodic.key == "observation" else self.ports.workspace
            self.episodic = cd.FastSynapses(
                key, self.ports.recall, decay=config.episodic.decay, rate=config.episodic.rate,
                amplitude=config.episodic.amplitude, rule="delta",
            )
            self.episodic.reset(streams)
        self.w_critic = np.zeros(len(self.ports.workspace))
        self.b_critic = 0.0
        seeds = list(stream_seeds) if stream_seeds is not None else [self.seed * 1000 + k for k in range(streams)]
        if len(seeds) != streams:
            raise ValueError("one stream seed per stream")
        self.rng = [Mulberry32(s) for s in seeds]
        self.elig = np.zeros((streams, self.connectome.synapses))
        self.elig_bias = np.zeros((streams, n))
        self.trace_critic = np.zeros((streams, len(self.ports.workspace) + 1))
        self.warm: BrainState | None = None
        self.pending: list[Pending | None] = [None] * streams
        self.last_event = np.full(streams, -1, dtype=np.int64)
        self.episode = np.full(streams, -1, dtype=np.int64)
        self.next_decision_id = 0
        self.parameter_version = 0
        self.context_version = 0
        self.ledger = Ledger()
        self.last_report: dict[str, Any] = {}
        self._halted = False
        self.ring: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, tuple[np.ndarray, np.ndarray]]]] = []
        self.ring_at = 0
        self.replay_rng = Mulberry32((self.seed * 7919 + 17) & 0xFFFFFFFF)
        # a stage's own memories read into the recall port: (moment, row) -> drive over the recall neurons
        self.recall_drive: Callable[[Moment, int], np.ndarray] | None = None
        self.records: RecordsHead | None = None
        self.reading = self.ports.sources  # the neurons a records head reads
        if config.records is not None:
            if not config.records.context:
                self.reading = np.concatenate([self.ports.observation, self.ports.goal, self.ports.action, self.ports.recall]).astype(np.int64)
            position = {int(n): k for k, n in enumerate(self.reading)}

            def block(*ports: np.ndarray) -> np.ndarray:
                return np.array([position[int(n)] for port in ports for n in port if int(n) in position], np.int64)

            if config.records.pathways == "fields":
                blocks = [block(self.ports.observation_fields[name], self.ports.observation_flags.get(name, np.zeros(0, np.int64))) for name in self.ports.observation_fields]
            elif config.records.pathways == "ports":
                blocks = [block(self.ports.observation)]
            else:
                raise ValueError("records.pathways is 'ports' or 'fields'")
            goal_block = block(self.ports.goal)
            blocks += [goal_block, block(self.ports.action), block(self.ports.recall), block(self.ports.context)]
            self.records = RecordsHead(len(self.reading), config.graph.prediction, config.records, seed, blocks, tasks=goal_block)
        self._last_recall = np.zeros((streams, len(self.ports.recall)))

    # -- shared parameter ownership

    @property
    def heads(self) -> list[cd.Learner]:
        return [h for h in (self.world, self.motor) if h is not None]

    @property
    def brain(self) -> cd.Brain:
        return self.heads[0].brain

    def _rebind(self, owner: cd.Learner) -> None:
        for head in self.heads:
            head.brain = owner.brain
        self.parameter_version += 1

    # -- encoding

    def _encode_observation(self, observation: Mapping[str, np.ndarray], observed: Mapping[str, np.ndarray] | None) -> np.ndarray:
        """The sensory drive of one moment: observed values (unobserved read zero) and flags."""
        spec = self.config.graph
        out = np.zeros(self.connectome.n)
        for f in spec.observation:
            if f.name not in observation:
                continue  # an absent field is entirely unobserved
            value = np.asarray(observation[f.name], dtype=float)
            if value.shape != (f.width,):
                raise ValueError(f"observation field {f.name!r} must have shape ({f.width},)")
            flag = np.ones(f.width, bool) if observed is None or f.name not in observed else np.asarray(observed[f.name], bool)
            if f.kind == "continuous":
                value = np.clip((value - f.lo) / (f.hi - f.lo), 0.0, 1.0)
            out[self.ports.observation_fields[f.name]] = spec.input_gain * spec.field_gains.get(f.name, 1.0) * value * flag
            if spec.missing_flags:
                missing = np.array([not flag.all()]) if spec.flags_per_field else ~flag
                out[self.ports.observation_flags[f.name]] = spec.input_gain * missing
        return out

    def _drive(self, moments: Sequence[Moment]) -> np.ndarray:
        """The drive of every stream: sensory values, goal, carried context and recall."""
        drive = np.zeros((self.streams, self.connectome.n))
        for i, m in enumerate(moments):
            drive[i] = self._encode_observation(m.observation, m.observed)
            if len(self.ports.goal):
                if m.goal is None or np.asarray(m.goal).shape != (len(self.ports.goal),):
                    raise ValueError("this brain needs a goal of the declared width on every moment")
                drive[i, self.ports.goal] = self.config.graph.input_gain * np.asarray(m.goal, float)
        drive = self.context.stimulate(drive)
        if self.episodic is not None:
            drive = self.episodic.stimulate(drive)
        if self.recall_drive is not None and len(self.ports.recall):
            for i, m in enumerate(moments):
                read = np.asarray(self.recall_drive(m, i), float)
                if read.shape != (len(self.ports.recall),) or not np.isfinite(read).all():
                    raise ValueError("recall_drive must return a finite vector over the recall neurons")
                self._last_recall[i] = read
                drive[i, self.ports.recall] += read
        return drive

    def _with_action(self, drive: np.ndarray, actions: Sequence[int]) -> np.ndarray:
        out = np.array(drive, dtype=float)
        gain = self.config.graph.input_gain if self.config.graph.action_gain is None else self.config.graph.action_gain
        for i, action in enumerate(actions):
            out[i, self.ports.action] = 0.0
            for slot, k in zip(self.ports.action_slots, self._split_action(int(action)), strict=True):
                out[i, slot[k]] = gain
        return out

    def _split_action(self, action: int) -> list[int]:
        """A joint action index into one index per field (row-major over the fields)."""
        sizes = self.config.graph.action_fields
        if not 0 <= action < int(np.prod(sizes)):
            raise ValueError("action index outside the joint action space")
        out = []
        for k in reversed(sizes):
            out.append(action % k)
            action //= k
        return list(reversed(out))

    def _joint_action(self, parts: Sequence[int]) -> int:
        action = 0
        for k, part in zip(self.config.graph.action_fields, parts, strict=True):
            action = action * k + int(part)
        return action

    # -- settling with a ledger

    def _settle(self, phase: str, drive: np.ndarray, *, warm: BrainState | None = None, nudge: Nudge | None = None, budget: int | None = None) -> BrainState:
        w = self.config.world
        steps = (w.nudged_steps if nudge is not None else w.free_steps) if budget is None else budget
        # Checked settling: chunks of steps until the neuron equations hold to the residual
        # tolerance or the budget is spent. Activation movement alone is not the criterion;
        # a saturated neuron can stand still while its potential equation is far from rest.
        eq = self.brain.equilibrate(drive, budget=steps, chunk=w.chunk, tolerance=w.residual_tolerance, state=warm, nudge=nudge)
        state, residual = eq.state, float(np.max(eq.residual))
        if not np.isfinite(state.activation).all():
            self._halted = True
            raise RuntimeError("nonfinite neural state; the life is halted")
        self.ledger.record(phase, eq.steps, residual, w.residual_tolerance)
        self._last_converged = bool(residual <= w.residual_tolerance)
        return BrainState(state.v, state.activation, state.adaptation, eq.steps)

    _last_converged: bool = True

    def _value(self, state: BrainState) -> np.ndarray:
        return np.asarray(state.activation[:, self.ports.workspace] @ self.w_critic + self.b_critic)

    # -- decoding predictions

    def decode(self, state: BrainState, row: int = 0) -> dict[str, np.ndarray]:
        """Read the prediction fields of a settled state: class probabilities or values."""
        out = {}
        s = state.activation[row]
        for f in self.config.graph.prediction:
            y = s[self.ports.prediction_fields[f.name]]
            if f.kind == "categorical":
                z = y / self.config.world.temperature
                z = np.exp(z - z.max())
                out[f.name] = z / z.sum()
            else:
                out[f.name] = f.lo + (np.clip(y, 0.05, 0.8) - 0.05) / 0.75 * (f.hi - f.lo)
        return out

    def _continuous_target(self, f: Field, value: np.ndarray) -> np.ndarray:
        return 0.05 + 0.75 * np.clip((np.asarray(value, float) - f.lo) / (f.hi - f.lo), 0.0, 1.0)

    def _outcome_targets(self, m: Moment) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """What this moment actually exposes for each prediction field: (target, observed mask)
        in neuron coordinates of the field; a class target is the index of the observed class."""
        out = {}
        for f in self.config.graph.prediction:
            if f.source == "reward":
                if not m.reward_known:
                    continue
                out[f.name] = (self._continuous_target(f, np.array([m.reward])), np.ones(1, bool))
                continue
            if f.source == "terminal":
                out[f.name] = (np.array([1 if m.terminated else 0]), np.ones(1, bool))
                continue
            if f.source not in m.observation:
                continue
            value = np.asarray(m.observation[f.source], float)
            known = np.asarray(m.observed[f.source], bool)
            if f.kind == "categorical":
                if not known.all():
                    continue
                if value.shape != (f.width,) or not np.isclose(value.sum(), 1.0) or value.max() < 0.99:
                    raise ValueError(f"categorical outcome {f.source!r} must be one-hot over {f.width} classes")
                out[f.name] = (np.array([int(value.argmax())]), np.ones(1, bool))
            else:
                if value.shape != (f.width,):
                    raise ValueError(f"continuous outcome {f.source!r} must have {f.width} values")
                if not known.any():
                    continue
                out[f.name] = (self._continuous_target(f, value), known)
        return out

    def score(self, prediction: Mapping[str, np.ndarray], m: Moment) -> dict[str, float]:
        """Score a stored prediction against what the moment actually exposes."""
        report: dict[str, float] = {}
        targets = self._outcome_targets(m)
        for f in self.config.graph.prediction:
            if f.name not in prediction or f.name not in targets:
                continue
            target, known = targets[f.name]
            p = np.asarray(prediction[f.name], float)
            if f.kind == "categorical":
                k = int(target[0])
                report[f"{f.name}/nll"] = float(-np.log(max(p[k], 1e-12)))
                report[f"{f.name}/correct"] = float(int(p.argmax()) == k)
            else:
                predicted = self._continuous_target(f, p)
                err = (predicted - target)[known]
                report[f"{f.name}/mse"] = float(np.mean(err**2)) if err.size else float("nan")
        return report

    # -- the event transaction

    def step(self, moment: Moment) -> Decision | None:
        """One stream's real event; see ``step_batch`` for parallel streams."""
        if self.streams != 1:
            raise ValueError("step takes one moment for a one-stream agent; use step_batch")
        return self.step_batch([moment])[0]

    def step_batch(self, moments: Sequence[Moment]) -> list[Decision | None]:
        if self._halted:
            raise RuntimeError("the life is halted after a numerical failure")
        if len(moments) != self.streams:
            raise ValueError(f"one moment per stream: expected {self.streams}, got {len(moments)}")
        # 1. validate every event before any state changes
        for i, m in enumerate(moments):
            self._validate(i, m)
        new_episode = np.array([m.episode_id != self.episode[i] for i, m in enumerate(moments)])
        for i in np.flatnonzero(new_episode):
            self._reset_stream(i)
            self.episode[i] = moments[i].episode_id
        drive = self._drive(moments)
        # 2. the free phase under the current parameters: the bootstrap value and the scoring state
        free_pre = self._settle("free", drive, warm=self.warm)
        value_next = self._value(free_pre)
        report: dict[str, Any] = {"scores": {}, "learning": {}}
        version_before = self.parameter_version
        for i, m in enumerate(moments):
            if m.has_feedback:
                assert self.pending[i] is not None
                report["scores"][i] = self.score(self.pending[i].decision.prediction, m)
        # 3.-4. reward credit of the executed decisions, through their saved eligibility
        report["learning"].update(self._credit(moments, value_next))
        # 5. repair the world model from (saved drive, executed action, observed consequence)
        report["learning"].update(self._repair(moments))
        # 6. witnessed associations
        self._store(moments)
        # 7. context advances once per real moment; a fresh free phase under the current parameters
        free_post = self._settle("free_post", drive, warm=free_pre) if self.parameter_version != version_before else free_pre
        self.context.update(free_post)
        self.context_version += 1
        closing = np.array([m.terminated or m.truncated for m in moments])
        # 8.-9. decide and predict for the streams that continue
        decisions: list[Decision | None] = [None] * self.streams
        v, a = free_post.v.copy(), free_post.adaptation.copy()
        for i, m in enumerate(moments):
            self.pending[i] = None
            if closing[i]:
                self._reset_stream(i)
                v[i], a[i] = 0.0, 0.0
                continue
            if not m.any_legal:
                raise ValueError("a continuing moment offers no legal action")
            decisions[i] = self._decide(i, m, drive, free_post)
        self.warm = BrainState(v, free_post.activation.copy(), a, free_post.steps)
        # 10. commit
        for i, m in enumerate(moments):
            self.last_event[i] = m.event_id
            if m.has_feedback:
                self.ledger.real_transitions += 1
        self.last_report = report
        return decisions

    def _validate(self, i: int, m: Moment) -> None:
        if m.event_id <= self.last_event[i]:
            raise ValueError(f"out-of-order or duplicate event {m.event_id} on stream {i}")
        if m.replay_of is not None:
            raise ValueError("a replayed observed event cannot enter step as a new real observation")
        pending = self.pending[i]
        if m.has_feedback:
            if pending is None:
                raise ValueError(f"feedback for decision {m.feedback_for} but no decision is pending on stream {i}")
            if m.feedback_for != pending.decision.decision_id:
                raise ValueError(f"feedback names decision {m.feedback_for}; pending is {pending.decision.decision_id}")
            if m.executed != pending.decision.action:
                raise ValueError(f"executed action {m.executed} differs from the committed action {pending.decision.action}")
            if m.episode_id != self.episode[i]:
                raise ValueError("feedback cannot cross an episode boundary")
        elif pending is not None:
            raise ValueError(f"decision {pending.decision.decision_id} on stream {i} awaits feedback; a new observation cannot replace it")
        if m.episode_id != self.episode[i] and m.episode_id < self.episode[i]:
            raise ValueError("episodes must not go backwards")
        if m.action_mask.shape != (int(np.prod(self.config.graph.action_fields)),):
            raise ValueError("action_mask must have one entry per joint action")
        for name in m.observation:
            if name not in self.ports.observation_fields:
                raise ValueError(f"unknown observation field {name!r}")

    def _reset_stream(self, i: int) -> None:
        """An episode boundary: designated transient state only; learned parameters and,
        unless declared, episodic records stay."""
        self.context.reset(self.streams, rows=np.array([i]))
        self.elig[i] = 0.0
        self.elig_bias[i] = 0.0
        self.trace_critic[i] = 0.0
        self.pending[i] = None
        if self.episodic is not None and self.config.episodic is not None and self.config.episodic.reset_on_episode:
            self.episodic.reset(self.streams, rows=np.array([i]))
        if self.warm is not None:
            v, a = self.warm.v.copy(), self.warm.adaptation.copy()
            v[i], a[i] = 0.0, 0.0
            self.warm = BrainState(v, self.warm.activation.copy(), a, self.warm.steps)

    # -- credit

    def _credit(self, moments: Sequence[Moment], value_next: np.ndarray) -> dict[str, float]:
        cfg = self.config.actor
        decay = cfg.gamma * cfg.lam
        credited: list[int] = []
        deltas = np.zeros(self.streams)
        td = np.zeros(self.streams)
        for i, m in enumerate(moments):
            if not m.has_feedback:
                continue
            p = self.pending[i]
            assert p is not None
            # the executed action's eligibility, only for the controller that generated it
            self.elig[i] *= decay
            self.elig_bias[i] *= decay
            if p.score_edges is not None:
                self.elig[i] += p.score_edges
                self.elig_bias[i] += p.score_bias
            self.trace_critic[i] *= decay
            self.trace_critic[i, :-1] += p.workspace
            self.trace_critic[i, -1] += 1.0
            if not m.reward_known:
                continue  # an outcome without a reward event teaches the world model only
            bootstrap = 0.0 if m.terminated else float(value_next[i])
            if m.truncated:
                # a time limit bootstraps from the final observation, never from a reset one
                final = self._hypothetical_drive(m.final_observation if m.final_observation is not None else m.observation, None, m.goal, i)
                bootstrap = float(self._value(self._settle("bootstrap", final))[0])
            error = m.reward + cfg.gamma * bootstrap - p.value
            td[i] = error
            delta = float(np.clip(error, -cfg.delta_cap, cfg.delta_cap))
            self.ledger.deltas += 1
            if delta != error:
                self.ledger.clipped_deltas += 1
            deltas[i] = delta
            credited.append(i)
        out: dict[str, float] = {}
        if credited:
            rows = np.array(credited)
            out["dopamine"] = float(deltas[rows].mean())
            out["td_error"] = float(np.abs(td[rows]).mean())
            if self.config.learning and self.config.motor_learning:
                step_edges = (deltas[rows, None] * self.elig[rows]).mean(axis=0)
                step_bias = (deltas[rows, None] * self.elig_bias[rows]).mean(axis=0)
                if np.any(step_edges) or np.any(step_bias):
                    self.motor.apply(np.clip(cfg.eta * step_edges, -cfg.step_clip, cfg.step_clip), cfg.eta_bias * step_bias)
                    self._rebind(self.motor)
                trace = self.trace_critic[rows]
                if cfg.critic_normalize:
                    trace = trace / (1.0 + (trace**2).sum(axis=1, keepdims=True))
                critic_step = cfg.eta_critic * (td[rows, None] * trace).mean(axis=0)
                self.w_critic += critic_step[:-1]
                self.b_critic += float(critic_step[-1])
        for i, m in enumerate(moments):
            if m.has_feedback and (m.terminated or m.truncated):
                self.elig[i] = 0.0
                self.elig_bias[i] = 0.0
                self.trace_critic[i] = 0.0
        return out

    # -- world repair

    def _repair_records(self, moments: Sequence[Moment]) -> dict[str, float]:
        """The records head's repair: the witnessed outcome written into the records the
        executed reading touched. No settle, no ring, nothing else moves."""
        assert self.records is not None
        written = 0
        for i, m in enumerate(moments):
            if not m.has_feedback:
                continue
            p = self.pending[i]
            assert p is not None
            if p.code is None:
                continue
            targets = self._outcome_targets(m)
            if targets:
                written += self.records.learn(p.code, targets)
        self.ledger.record_writes += written
        return {"records_written": float(written)} if written else {}

    def _repair(self, moments: Sequence[Moment]) -> dict[str, float]:
        if not (self.config.learning and self.config.world_learning) or self.world is None:
            return {}
        if self.records is not None:
            return self._repair_records(moments)
        rows = [i for i, m in enumerate(moments) if m.has_feedback]
        if not rows:
            return {}
        n = self.connectome.n
        cfg = self.config.world
        # the real transitions, stored in the ring, plus replayed stored transitions
        transitions = []
        for i in rows:
            p = self.pending[i]
            assert p is not None
            targets = self._outcome_targets(moments[i])
            transitions.append((p.drive, p.state_v, p.state_a, targets))
        replayed = 0
        if self.config.replay is not None:
            for t in transitions:
                if t[3]:
                    if len(self.ring) < self.config.replay.capacity:
                        self.ring.append(t)
                    else:
                        self.ring[self.ring_at] = t
                        self.ring_at = (self.ring_at + 1) % self.config.replay.capacity
            stored = len(self.ring) - len([t for t in transitions if t[3]])
            for _ in range(self.config.replay.per_transition * len(rows)):
                if stored <= 0:
                    break
                transitions.append(self.ring[int(self.replay_rng.random() * stored)])
                replayed += 1
            self.ledger.replay_writes += replayed
        # group rows by their observed target mask, per nudge kind
        drives, warm_v, warm_a = [], [], []
        cat_target, cat_mask, cat_groups, cat_fields = [], [], [], []
        con_target, con_mask, con_fields = [], [], []
        for drive_row, v_row, a_row, targets in transitions:
            t_cat, m_cat, g_cat = np.zeros(n), np.zeros(n), np.full(n, -1, dtype=np.int64)
            t_con, m_con = np.zeros(n), np.zeros(n)
            k_cat = k_con = 0
            for k, f in enumerate(self.config.graph.prediction):
                if f.name not in targets:
                    continue
                neurons = self.ports.prediction_fields[f.name]
                target, known = targets[f.name]
                if f.kind == "categorical":
                    m_cat[neurons] = 1.0
                    g_cat[neurons] = k
                    t_cat[neurons[int(target[0])]] = 1.0
                    k_cat += 1
                else:
                    m_con[neurons[known]] = 1.0
                    t_con[neurons] = target
                    k_con += 1
            if not m_cat.any() and not m_con.any():
                continue  # all missing: no update from this row
            drives.append(drive_row)
            warm_v.append(v_row)
            warm_a.append(a_row)
            cat_target.append(t_cat)
            cat_mask.append(m_cat)
            cat_groups.append(g_cat)
            cat_fields.append(k_cat)
            con_target.append(t_con)
            con_mask.append(m_con)
            con_fields.append(k_con)
        if not drives:
            return {}
        drive = np.stack(drives)
        warm = BrainState(np.stack(warm_v), self.brain.neuron_model.activation(np.stack(warm_v)), np.stack(warm_a), 0)
        free = self._settle("repair_free", drive, warm=warm)
        converged = self._last_converged
        edges_total = np.zeros(self.connectome.synapses)
        neurons_total = np.zeros(n)
        contributing = 0
        smoothing = cfg.label_smoothing
        for kind, targets_k, masks_k, groups_k, fields_k in (
            ("categorical", cat_target, cat_mask, cat_groups, cat_fields),
            ("continuous", con_target, con_mask, [None] * len(con_target), con_fields),
        ):
            keys: dict[bytes, list[int]] = {}
            for r, mask in enumerate(masks_k):
                if mask.any():
                    keys.setdefault(mask.tobytes() + (groups_k[r].tobytes() if groups_k[r] is not None else b""), []).append(r)
            for group_rows in keys.values():
                sel = np.array(group_rows)
                mask = masks_k[sel[0]]
                groups = groups_k[sel[0]]
                beta = cfg.beta / max(1, fields_k[sel[0]])
                target = np.stack([targets_k[r] for r in group_rows])
                temperature = cfg.temperature if kind == "categorical" else None
                if kind == "categorical" and smoothing > 0:
                    for g in np.unique(groups[groups >= 0]):
                        members = np.flatnonzero(groups == g)
                        target[:, members] = (1.0 - smoothing) * target[:, members] + smoothing / len(members)
                plus = self._settle("repair_plus", drive[sel], warm=_rows(free, sel), nudge=Nudge(target, mask, beta, softmax_temperature=temperature, groups=groups))
                converged &= self._last_converged
                minus = self._settle("repair_minus", drive[sel], warm=_rows(free, sel), nudge=Nudge(target, mask, -beta, softmax_temperature=temperature, groups=groups))
                converged &= self._last_converged
                if float(np.abs(plus.activation - minus.activation).max()) > cfg.max_phase_gap:
                    converged = False  # the nudged phases sit in different attractors: no gradient here
                e, b = self.world.contrast(_rows(free, sel), plus, minus)
                edges_total += e * len(sel)
                neurons_total += b * len(sel)
                contributing += len(sel)
        if not contributing:
            return {}
        if cfg.reject_unconverged and not converged:
            self.ledger.rejected_updates += 1
            return {"world_rejected": 1.0}
        edges, neurons = edges_total / len(drives), neurons_total / len(drives)
        edges, neurons = self._adaptive(self.world, edges, neurons, cfg.momentum, cfg.normalize)
        edges = np.clip(cfg.eta * edges, -cfg.step_clip, cfg.step_clip)
        report = self.world.apply(edges, cfg.eta_bias * neurons)
        self.world.contrast_updates += 1
        self._rebind(self.world)
        return {"world_scale_step": report["scale_step"], "world_bias_step": report["bias_step"], "replayed": float(replayed)}

    def _adaptive(self, head: cd.Learner, edges: np.ndarray, neurons: np.ndarray, momentum: float, normalize: float) -> tuple[np.ndarray, np.ndarray]:
        raw_e, raw_n = edges, neurons
        count = head.contrast_updates + 1
        if momentum > 0:
            head.velocity = momentum * head.velocity + (1 - momentum) * edges
            head.velocity_bias = momentum * head.velocity_bias + (1 - momentum) * neurons
            correction = 1.0 - momentum**count
            edges, neurons = head.velocity / correction, head.velocity_bias / correction
        if normalize > 0:
            head.second_moment = normalize * head.second_moment + (1 - normalize) * raw_e**2
            head.second_moment_bias = normalize * head.second_moment_bias + (1 - normalize) * raw_n**2
            correction = 1.0 - normalize**count
            edges = edges / (np.sqrt(head.second_moment / correction) + head.config.normalize_floor)
            neurons = neurons / (np.sqrt(head.second_moment_bias / correction) + head.config.normalize_floor)
        return edges, neurons

    # -- evidence

    def _store(self, moments: Sequence[Moment]) -> None:
        """Witnessed associations: the executed transition's key and its observed consequence."""
        if self.episodic is None:
            return
        keys = np.zeros((self.streams, len(self.episodic.pre)))
        values = np.zeros((self.streams, len(self.episodic.post)))
        write = np.zeros(self.streams, bool)
        for i, m in enumerate(moments):
            p = self.pending[i]
            if p is None or not m.has_feedback or self.recall_value is None:
                continue
            value = self.recall_value(m)
            if value is None:
                continue
            keys[i] = p.drive[self.episodic.pre]
            values[i] = value
            write[i] = True
        if write.any():
            self.episodic.observe(keys, values, write=write)
            self.ledger.episodic_writes += int(write.sum())

    recall_value: Callable[[Moment], np.ndarray | None] | None = None

    # -- decisions

    def _policy(self, state: BrainState, row: int, mask: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Joint-action probabilities over the legal entries, and per-field marginals."""
        cfg = self.config.actor
        s = state.activation[row]
        sizes = self.config.graph.action_fields
        per_field: list[np.ndarray] = []
        for k, slot in enumerate(self.ports.motor_slots):
            legal = np.zeros(sizes[k], bool)
            for joint in np.flatnonzero(mask):
                legal[self._split_action(int(joint))[k]] = True
            z = np.where(legal, s[slot] / cfg.temperature, -np.inf)
            z = np.exp(z - z[legal].max())
            per_field.append(z / z.sum())
        joint = np.zeros(len(mask))
        for j in np.flatnonzero(mask):
            parts = self._split_action(int(j))
            joint[j] = float(np.prod([per_field[k][parts[k]] for k in range(len(sizes))]))
        joint = joint / joint.sum()
        return joint, per_field

    def _decide(self, i: int, m: Moment, drive: np.ndarray, free: BrainState) -> Decision:
        cfg = self.config.actor
        mask = m.action_mask
        legal = np.flatnonzero(mask)
        controller = self.config.controller
        budget: dict[str, int] = {}
        prediction: dict[str, np.ndarray] = {}
        score_edges = score_bias = None
        probabilities, _ = self._policy(free, i, mask)
        if controller == "planner":
            assert self.planner is not None
            if cfg.epsilon > 0 and self.rng[i].random() < cfg.epsilon:
                action = int(legal[int(self.rng[i].random() * len(legal))])
                controller = "exploration"
            else:
                action, prediction, budget = self.planner(self, i, m, drive, free)
                if action is None:  # the planner delegates this decision to the actor: credit rules apply
                    controller = "actor"
                    prediction = {}
                elif not mask[action]:
                    raise ValueError("the planner chose an illegal action")
        if controller == "actor":
            if cfg.epsilon > 0 and self.rng[i].random() < cfg.epsilon:
                action = int(legal[int(self.rng[i].random() * len(legal))])
                controller = "exploration"
            else:
                action = self.rng[i].choice(probabilities)
                if not mask[action]:
                    raise AssertionError("sampled an illegal action")
                if len(legal) > 1 and self.config.learning and self.config.motor_learning:
                    score_edges, score_bias = self._score(i, m, action, drive, free)
                elif len(legal) > 1:
                    pass
                else:
                    score_edges = np.zeros(self.connectome.synapses)
                    score_bias = np.zeros(self.connectome.n)
        code = None
        drive_a = self._with_action(drive[[i]], [action])
        if self.records is not None:
            code = self.records.code(drive_a[:, self.reading], adapt=self.config.learning and self.config.world_learning)[:, 0]
            if not prediction and self.config.graph.prediction:
                prediction = self.records.predict(code)
            state_v, state_a = free.v[i].copy(), free.adaptation[i].copy()
        elif not prediction and self.config.graph.prediction:
            predicted = self._settle("predict", drive_a, warm=_rows(free, np.array([i])))
            prediction = self.decode(predicted, 0)
            state_v, state_a = predicted.v[0], predicted.adaptation[0]
        else:
            state_v, state_a = free.v[i].copy(), free.adaptation[i].copy()
        log_probability = float(np.log(max(probabilities[action], 1e-300)))
        decision = Decision(
            decision_id=self.next_decision_id,
            event_id=m.event_id,
            stream=i,
            parameter_version=self.parameter_version,
            context_version=self.context_version,
            action=int(action),
            action_mask=mask,
            probabilities=probabilities,
            log_probability=log_probability,
            controller=controller,
            prediction=prediction,
            budget=budget,
        )
        self.next_decision_id += 1
        pending = Pending(
            decision=decision,
            drive=drive_a[0],
            state_v=state_v,
            state_a=state_a,
            value=float(self._value(free)[i]),
            score_edges=score_edges if controller == "actor" else None,
            score_bias=score_bias if controller == "actor" else None,
            workspace=free.activation[i, self.ports.workspace].copy(),
            code=code,
        )
        self.pending[i] = pending
        return decision

    def _score(self, i: int, m: Moment, action: int, drive: np.ndarray, free: BrainState) -> tuple[np.ndarray, np.ndarray]:
        """The actor's per-synapse score of the sampled action: masked centred contrast."""
        cfg = self.config.actor
        n = self.connectome.n
        mask = np.zeros(n)
        groups = np.full(n, -1, dtype=np.int64)
        target = np.zeros((1, n))
        parts = self._split_action(action)
        sizes = self.config.graph.action_fields
        fields = 0
        for k, slot in enumerate(self.ports.motor_slots):
            legal = np.zeros(sizes[k], bool)
            for joint in np.flatnonzero(m.action_mask):
                legal[self._split_action(int(joint))[k]] = True
            if legal.sum() < 2:
                continue  # a forced field has no policy-score derivative
            mask[slot[legal]] = 1.0
            groups[slot[legal]] = k
            target[0, slot[parts[k]]] = 1.0
            fields += 1
        if fields == 0:
            return np.zeros(self.connectome.synapses), np.zeros(n)
        beta = cfg.beta / fields
        rows = np.array([i])
        plus = self._settle("score_plus", drive[rows], warm=_rows(free, rows), nudge=Nudge(target, mask, beta, softmax_temperature=cfg.temperature, groups=groups))
        converged = self._last_converged
        minus = self._settle("score_minus", drive[rows], warm=_rows(free, rows), nudge=Nudge(target, mask, -beta, softmax_temperature=cfg.temperature, groups=groups))
        converged &= self._last_converged
        if float(np.abs(plus.activation - minus.activation).max()) > self.config.world.max_phase_gap:
            converged = False
        if self.config.world.reject_unconverged and not converged:
            self.ledger.rejected_updates += 1
            return np.zeros(self.connectome.synapses), np.zeros(n)
        edges, neurons = self.motor.contrast_rows(_rows(free, rows), plus, minus)
        return edges[0], neurons[0]

    # -- read-only imagination

    def _hypothetical_drive(self, observation: Mapping[str, np.ndarray], observed: Mapping[str, np.ndarray] | None, goal: np.ndarray | None, row: int, *, context: bool = True) -> np.ndarray:
        """A one-row drive for a hypothetical observation in stream ``row``'s current context."""
        drive = np.zeros((1, self.connectome.n))
        drive[0] = self._encode_observation(observation, observed)
        if len(self.ports.goal):
            if goal is None or np.asarray(goal).shape != (len(self.ports.goal),):
                raise ValueError("this brain needs a goal of the declared width to imagine")
            drive[0, self.ports.goal] = self.config.graph.input_gain * np.asarray(goal, float)
        if context and len(self.context.trace) == self.streams:
            drive[0, self.ports.context] = self.context.amplitude * self.context.trace[row]
            if self.episodic is not None and len(self.episodic.strength) == self.streams:
                key = cd.FastSynapses._delta_unit(drive[:, self.episodic.pre])
                drive[0, self.ports.recall] += self.episodic.amplitude * (key[0] @ self.episodic.strength[row])
            if self.recall_drive is not None and len(self.ports.recall):
                drive[0, self.ports.recall] += self._last_recall[row]
        return drive

    def predict(self, observation: Mapping[str, np.ndarray], action: int, *, observed: Mapping[str, np.ndarray] | None = None, goal: np.ndarray | None = None, row: int = 0, context: bool = True) -> dict[str, np.ndarray]:
        """The consequence the model predicts for a hypothetical observation and action.

        Reads the current parameters, the stream's carried context and recall; writes
        nothing: no trace, no memory, no RNG, no pending credit."""
        drive = self._with_action(self._hypothetical_drive(observation, observed, goal, row, context=context), [action])
        if self.records is not None:
            self.ledger.imagined += 1
            return self.records.predict(self.records.code(drive[:, self.reading], adapt=False)[:, 0])
        w = self.config.world
        state = self.brain.settle_batch(drive, steps=w.free_steps, tolerance=w.tolerance if w.imagine_tolerance is None else w.imagine_tolerance)
        self.ledger.imagined += 1
        return self.decode(state, 0)

    def predict_batch(self, observations: Sequence[Mapping[str, np.ndarray]], actions: Sequence[int], *, observed: Sequence[Mapping[str, np.ndarray] | None] | None = None, goal: np.ndarray | None = None, row: int = 0, context: bool = True, valued: bool = True) -> list[dict[str, np.ndarray]]:
        """``predict`` for several hypothetical (observation, action) pairs in one settle.

        ``observed``: one flag map per observation (None: every field observed). The flags are
        part of the reading, so an imagined state must carry the same flags a real moment would.
        ``valued=False`` predicts the consequence fields alone (a records head then skips its
        valued code); the reward and terminal fields are left out of the predictions."""
        if len(observations) != len(actions) or not observations:
            raise ValueError("one action per hypothetical observation")
        if observed is not None and len(observed) != len(observations):
            raise ValueError("one observed map per hypothetical observation")
        drive = np.concatenate([self._hypothetical_drive(obs, None if observed is None else observed[k], goal, row, context=context) for k, obs in enumerate(observations)])
        drive = self._with_action(drive, [int(a) for a in actions])
        self.ledger.imagined += len(actions)
        if self.records is not None:
            codes = self.records.code(drive[:, self.reading], adapt=False, valued=valued)
            return self.records.predict_many(codes, valued=valued)
        w = self.config.world
        state = self.brain.settle_batch(drive, steps=w.free_steps, tolerance=w.tolerance if w.imagine_tolerance is None else w.imagine_tolerance)
        return [self.decode(state, k) for k in range(len(actions))]

    # -- inspection

    def state_hash(self) -> str:
        """A digest of every learned parameter, memory, trace, generator and pending record."""
        h = hashlib.sha256()
        for arr in (self.brain.efficacy, self.brain.bias, self.w_critic, np.array([self.b_critic]), self.elig, self.elig_bias, self.trace_critic, self.context.trace, self.context.last):
            h.update(np.ascontiguousarray(arr).tobytes())
        if self.episodic is not None:
            h.update(np.ascontiguousarray(self.episodic.strength).tobytes())
        if self.records is not None:
            h.update(np.ascontiguousarray(self.records.mean).tobytes())
            h.update(np.ascontiguousarray(self.records.block_norm).tobytes())
            h.update(np.array([self.records.seen], dtype=np.int64).tobytes())
            for name in sorted(self.records.records):
                h.update(np.ascontiguousarray(self.records.records[name]).tobytes())
        h.update(json.dumps([r.state for r in self.rng] + [self.replay_rng.state, len(self.ring), self.ring_at]).encode())
        h.update(json.dumps([None if p is None else p.decision.decision_id for p in self.pending]).encode())
        if self.warm is not None:
            h.update(np.ascontiguousarray(self.warm.v).tobytes())
        return h.hexdigest()

    def parameters(self) -> dict[str, int]:
        return {
            "neurons": int(self.connectome.n),
            "directed_synapses": int(self.connectome.synapses),
            "world_parameters": 0 if self.world is None else int(self.world.parameters()),
            "record_parameters": 0 if self.records is None else self.records.parameters(),
            "motor_parameters": int(self.motor.parameters()),
            "critic_parameters": int(len(self.w_critic) + 1),
            "episodic_parameters": 0 if self.episodic is None else int(self.episodic.strength.size),
            "eligibility_numbers": int(self.elig.size + self.elig_bias.size + self.trace_critic.size),
            "optimizer_numbers": int(sum(x.size for h in self.heads for x in (h.velocity, h.velocity_bias, h.second_moment, h.second_moment_bias))),
        }

    def new_stream(self, row: int = 0) -> None:
        """Attach stream ``row`` to a new environment: fresh cursors and transient state.

        Refused while a decision awaits feedback; learned parameters and memories stay."""
        if self.pending[row] is not None:
            raise ValueError("a pending decision must receive its outcome before the stream changes")
        self._reset_stream(row)
        self.last_event[row] = -1
        self.episode[row] = -1

    def abandon(self, row: int = 0) -> None:
        """Drop a decision awaiting its outcome without feedback: for evaluation copies that
        move to another environment; a live life feeds every decision back instead."""
        self.pending[row] = None

    def frozen(self) -> Agent:
        """A read-only competence copy: learning off, every stream attached afresh.

        The copy drops any decision awaiting its outcome; the live agent keeps it."""
        out = self.clone()
        out.config = replace(self.config, learning=False)
        for row in range(out.streams):
            out.pending[row] = None
            out.new_stream(row)
        return out

    def clone(self) -> Agent:
        """An isolated copy for frozen or adapting evaluation; the live agent is untouched.

        Callables (planner, recall hooks) are shared by reference: a stage that keeps its own
        stores must rebind them on the copy to isolated copies of those stores."""
        planner, recall_value, recall_drive = self.planner, self.recall_value, self.recall_drive
        self.planner = self.recall_value = self.recall_drive = None
        try:
            out = copy.deepcopy(self)
        finally:
            self.planner, self.recall_value, self.recall_drive = planner, recall_value, recall_drive
        out.planner, out.recall_value, out.recall_drive = planner, recall_value, recall_drive
        return out

    # -- checkpoints

    def save(self, path: str | Path) -> Path:
        """Everything a resumed life needs, at a transaction boundary."""
        path = Path(path)
        data: dict[str, Any] = {
            "efficacy": self.brain.efficacy,
            "bias": self.brain.bias,
            "w_critic": self.w_critic,
            "elig": self.elig,
            "elig_bias": self.elig_bias,
            "trace_critic": self.trace_critic,
            "context/trace": self.context.trace,
            "context/last": self.context.last,
            "context/cold": self.context.cold,
            "last_event": self.last_event,
            "episode": self.episode,
            "rng": np.array([r.state for r in self.rng], dtype=np.int64),
        }
        if self.ring:
            data["ring/drive"] = np.stack([t[0] for t in self.ring])
            data["ring/v"] = np.stack([t[1] for t in self.ring])
            data["ring/a"] = np.stack([t[2] for t in self.ring])
            for k, f in enumerate(self.config.graph.prediction):
                present = np.array([f.name in t[3] for t in self.ring])
                data[f"ring/present/{k}"] = present
                data[f"ring/target/{k}"] = np.stack([t[3][f.name][0] if f.name in t[3] else np.zeros(1 if f.kind == "categorical" else f.width) for t in self.ring])
                data[f"ring/known/{k}"] = np.stack([t[3][f.name][1] if f.name in t[3] else np.zeros(1 if f.kind == "categorical" else f.width, bool) for t in self.ring])
        for head, name in ((self.world, "world"), (self.motor, "motor")):
            if head is None:
                continue
            for moment in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
                data[f"{name}/{moment}"] = getattr(head, moment)
        if self.episodic is not None:
            data["episodic/strength"] = self.episodic.strength
            data["episodic/mass"] = self.episodic.mass
        if self.warm is not None:
            for name in ("v", "activation", "adaptation"):
                data[f"warm/{name}"] = getattr(self.warm, name)
        pending_meta = []
        for i, p in enumerate(self.pending):
            if p is None:
                pending_meta.append(None)
                continue
            pending_meta.append({"decision": p.decision.to_dict(), "value": p.value, "scored": p.score_edges is not None})
            data[f"pending/{i}/drive"] = p.drive
            data[f"pending/{i}/state_v"] = p.state_v
            data[f"pending/{i}/state_a"] = p.state_a
            if p.code is not None:
                data[f"pending/{i}/code"] = p.code
            data[f"pending/{i}/workspace"] = p.workspace
            data[f"pending/{i}/mask"] = p.decision.action_mask
            if p.score_edges is not None:
                data[f"pending/{i}/score_edges"] = p.score_edges
                data[f"pending/{i}/score_bias"] = p.score_bias
        meta = {
            "format": "cadence-experience-agent/1",
            "config": self.config.to_dict(),
            "streams": self.streams,
            "seed": self.seed,
            "b_critic": self.b_critic,
            "next_decision_id": self.next_decision_id,
            "parameter_version": self.parameter_version,
            "context_version": self.context_version,
            "world_updates": None if self.world is None else [self.world.updates, self.world.contrast_updates],
            "motor_updates": [self.motor.updates, self.motor.contrast_updates],
            "episodic_writes": None if self.episodic is None else self.episodic.writes,
            "warm_steps": None if self.warm is None else self.warm.steps,
            "ring": [len(self.ring), self.ring_at, self.replay_rng.state],
            "pending": pending_meta,
            "ledger": self.ledger.to_dict(),
            "halted": self._halted,
        }
        if self.records is not None:
            data["records/mean"] = self.records.mean
            data["records/block_norm"] = self.records.block_norm
            for name, table in self.records.records.items():
                data[f"records/{name}"] = table
            meta["record_writes"] = self.records.writes
            meta["record_seen"] = self.records.seen
        data["meta"] = np.array(json.dumps(meta, sort_keys=True))
        tmp = path.with_suffix(path.suffix + ".tmp")
        with open(tmp, "wb") as f:
            np.savez_compressed(f, **data)
        tmp.replace(path)
        return path

    @classmethod
    def load(cls, path: str | Path, *, planner: Planner | None = None, backend: str = "cpu") -> Agent:
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["meta"]))
            if meta.get("format") != "cadence-experience-agent/1":
                raise ValueError("not an experience-agent checkpoint")
            config = AgentConfig.from_dict(meta["config"])
            agent = cls(config, streams=int(meta["streams"]), seed=int(meta["seed"]), planner=planner, backend=backend)
            efficacy, bias = data["efficacy"].copy(), data["bias"].copy()
            if efficacy.shape != agent.brain.efficacy.shape or bias.shape != agent.brain.bias.shape or not np.isfinite(efficacy).all() or not np.isfinite(bias).all():
                raise ValueError("saved parameters do not match the developed connectome")
            restored = agent.brain.with_parameters(efficacy=efficacy, bias=bias)
            for head in agent.heads:
                head.brain = restored
            for head, name in ((agent.world, "world"), (agent.motor, "motor")):
                if head is None:
                    continue
                for moment in ("velocity", "velocity_bias", "second_moment", "second_moment_bias"):
                    setattr(head, moment, data[f"{name}/{moment}"].copy())
            if agent.world is not None:
                agent.world.updates, agent.world.contrast_updates = (int(x) for x in meta["world_updates"])
            agent.motor.updates, agent.motor.contrast_updates = (int(x) for x in meta["motor_updates"])
            agent.w_critic = data["w_critic"].copy()
            agent.b_critic = float(meta["b_critic"])
            agent.elig, agent.elig_bias, agent.trace_critic = data["elig"].copy(), data["elig_bias"].copy(), data["trace_critic"].copy()
            agent.context.trace, agent.context.last, agent.context.cold = data["context/trace"].copy(), data["context/last"].copy(), data["context/cold"].copy()
            agent.last_event, agent.episode = data["last_event"].copy(), data["episode"].copy()
            for r, state in zip(agent.rng, data["rng"], strict=True):
                r.state = int(state)
            if agent.records is not None:
                agent.records.mean = data["records/mean"].copy()
                if "records/block_norm" in data:
                    agent.records.block_norm = data["records/block_norm"].copy()
                for name in agent.records.records:
                    agent.records.records[name] = data[f"records/{name}"].copy()
                agent.records.writes = int(meta.get("record_writes", 0))
                agent.records.seen = int(meta.get("record_seen", 0))
            ring_len, agent.ring_at, replay_state = meta["ring"]
            agent.replay_rng.state = int(replay_state)
            if ring_len:
                fields = agent.config.graph.prediction
                for j in range(int(ring_len)):
                    targets = {}
                    for k, f in enumerate(fields):
                        if data[f"ring/present/{k}"][j]:
                            targets[f.name] = (data[f"ring/target/{k}"][j].copy(), data[f"ring/known/{k}"][j].copy())
                    agent.ring.append((data["ring/drive"][j].copy(), data["ring/v"][j].copy(), data["ring/a"][j].copy(), targets))
            if agent.episodic is not None:
                agent.episodic.strength, agent.episodic.mass = data["episodic/strength"].copy(), data["episodic/mass"].copy()
                agent.episodic.writes = int(meta["episodic_writes"])
            if meta["warm_steps"] is not None:
                agent.warm = BrainState(data["warm/v"].copy(), data["warm/activation"].copy(), data["warm/adaptation"].copy(), int(meta["warm_steps"]))
            for i, pm in enumerate(meta["pending"]):
                if pm is None:
                    continue
                d = pm["decision"]
                decision = Decision(
                    decision_id=d["decision_id"], event_id=d["event_id"], stream=d["stream"],
                    parameter_version=d["parameter_version"], context_version=d["context_version"],
                    action=d["action"], action_mask=data[f"pending/{i}/mask"].copy(),
                    probabilities=np.array(d["probabilities"]), log_probability=d["log_probability"],
                    controller=d["controller"], prediction={k: np.array(v) for k, v in d["prediction"].items()},
                    uncertainty=d["uncertainty"], budget=d["budget"], intention=d["intention"],
                )
                p = Pending(
                    decision=decision, drive=data[f"pending/{i}/drive"].copy(), state_v=data[f"pending/{i}/state_v"].copy(),
                    state_a=data[f"pending/{i}/state_a"].copy(), value=float(pm["value"]),
                    score_edges=data[f"pending/{i}/score_edges"].copy() if pm["scored"] else None,
                    score_bias=data[f"pending/{i}/score_bias"].copy() if pm["scored"] else None,
                    workspace=data[f"pending/{i}/workspace"].copy(),
                    code=data[f"pending/{i}/code"].copy() if f"pending/{i}/code" in data else None,
                )
                agent.pending[i] = p
            agent.next_decision_id = int(meta["next_decision_id"])
            agent.parameter_version = int(meta["parameter_version"])
            agent.context_version = int(meta["context_version"])
            agent._halted = bool(meta["halted"])
            ledger = meta["ledger"]
            agent.ledger.rejected_updates = ledger["rejected_updates"]
            agent.ledger.real_transitions = ledger["real_transitions"]
            agent.ledger.deltas, agent.ledger.clipped_deltas = ledger["deltas"], ledger["clipped_deltas"]
            agent.ledger.episodic_writes, agent.ledger.imagined = ledger["episodic_writes"], ledger["imagined"]
            agent.ledger.phases = {k: int(v) for k, v in ledger["phases"].items()}
            agent.ledger.steps = {k: round(ledger["mean_steps"][k] * v) for k, v in agent.ledger.phases.items()}
            agent.ledger.max_residual = dict(ledger["max_residual"])
            agent.ledger.unconverged = {k: int(v) for k, v in ledger["unconverged_phases"].items()}
        return agent

