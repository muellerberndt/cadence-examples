// The remembered world in JavaScript: a port of world/env.py
// (`World`, `WorldConfig`, the constants) and of world/tasks.py
// (`Curriculum`, `requestDeadline`, `goalVector`, `manhattan`). Every rule of the world is
// the Python rule: the map's edges in the same order, the same observation fields and
// missingness flags, the same act, reward, deadline and truncation logic, the same
// privileged accessors the curriculum uses. A request, a name request and the cue decision
// start a new episode of the same life (`request`, `nameRequest` and `armCue` return its
// first moment), as tasks.py does. The random numbers come from a Mulberry32 stream (the
// Python world draws from numpy, which JavaScript cannot reproduce), so a life on this world
// is its own; `World.fromState` loads the private state that the Python fixture dumps and
// `world.state()` returns it in the same shape, which is how the parity harness compares
// the two worlds step by step. Reward rules and the joint-attention word rule are data
// (`{kind, ...}`) rather than closures, so a state dump carries them. The eighteen fields
// include the displacement of the last action (`dx`, `dy`), as env.py reports it.

import { Mulberry32 } from "../../web/engine.js";

export const SIZE = 6;
export const ACTIONS = ["north", "east", "south", "west", "inspect", "interact"];
export const DIRECTION_NAMES = ["north", "east", "south", "west"];
export const DIRECTIONS = { north: [0, -1], east: [1, 0], south: [0, 1], west: [-1, 0] };
export const PASSAGE = ["open", "wall", "door_closed"];
export const OUTCOMES = ["none", "moved", "blocked", "picked", "dropped", "toggled", "inspected"];
export const OBJECTS = 4;
export const COLORS = 4;
export const WORDS = 8;
export const GOALS = 8;
export const CELLS = SIZE * SIZE;
export const FIELDS = {
  x: SIZE, y: SIZE,
  dx: 3, dy: 3, // the displacement of the last action: -1, 0, +1 (as S01 reports its displacement)
  passage_north: 3, passage_east: 3, passage_south: 3, passage_west: 3,
  here_object: OBJECTS + 1, here_color: COLORS + 1,
  north_object: OBJECTS + 1, east_object: OBJECTS + 1, south_object: OBJECTS + 1, west_object: OBJECTS + 1,
  carrying: OBJECTS + 1, outcome: OUTCOMES.length, word: WORDS + 1, phase: 2,
};
export const FIELD_NAMES = Object.keys(FIELDS);
export const ACTION_INSPECT = ACTIONS.indexOf("inspect");
export const ACTION_INTERACT = ACTIONS.indexOf("interact");
export const WORLD_STATE_FORMAT = "cadence-s02-world/1";

export function oneHot(index, width) { const out = new Float64Array(width); out[index] = 1.0; return out; }
export function cellKey(c) { return `${c[0]},${c[1]}`; }
export function sameCell(a, b) { return a !== null && b !== null && a !== undefined && b !== undefined && a[0] === b[0] && a[1] === b[1]; }
export function inBounds(c) { return c[0] >= 0 && c[0] < SIZE && c[1] >= 0 && c[1] < SIZE; }
export function manhattan(a, b) { return Math.abs(a[0] - b[0]) + Math.abs(a[1] - b[1]); }
/** The first index of the largest value (numpy argmax). */
export function argmaxOf(values) { let arg = 0; for (let k = 1; k < values.length; k++) if (values[k] > values[arg]) arg = k; return arg; }
/** Every cell in row-major order: for y, for x (the order Python's comprehensions use). */
export function allCells() { const out = []; for (let y = 0; y < SIZE; y++) for (let x = 0; x < SIZE; x++) out.push([x, y]); return out; }

export const DEFAULT_WORLD_CONFIG = { doors: 3, walls: 4, horizon: 64, inspect_visible: true, junction: [1, 1] };
export function worldConfig(over = {}) { const c = { ...DEFAULT_WORLD_CONFIG, ...over }; c.junction = [c.junction[0] | 0, c.junction[1] | 0]; return c; }

const edgeKeyOf = (a, b) => `${a[0]},${a[1]}|${b[0]},${b[1]}`;
const cellOfKey = (k) => k.split(",").map(Number);
const copyCell = (c) => (c === null || c === undefined ? null : [c[0] | 0, c[1] | 0]);

/** Fisher-Yates with a Mulberry32 stream (numpy's permutation is a different algorithm; the page runs its own life). */
export function permutation(rng, n) {
  const out = Array.from({ length: n }, (_, k) => k);
  for (let i = n - 1; i > 0; i--) { const j = Math.floor(rng.random() * (i + 1)); const t = out[i]; out[i] = out[j]; out[j] = t; }
  return out;
}

export class World {
  /** One life's private map and one episode at a time; `act` executes a committed decision. */
  constructor(config = {}, { seed = 0, life_id = "world", build = true } = {}) {
    this.config = worldConfig(config);
    this.seed = seed;
    this.life_id = life_id;
    this.rng = new Mulberry32(seed);
    this._edges = new Map(); // canonical edge key -> "open" | "wall" | "door_closed", in the build order
    this._doors = new Set(); // the edges built as doors, whatever their state
    this._colors = Array.from({ length: COLORS }, (_, k) => k);
    this._wordOf = Array.from({ length: WORDS }, (_, k) => k);
    this._objects = new Map(); // object -> cell or null, in first-placement order (Python dict order; object_at scans it)
    this._agent = [0, 0];
    this._carrying = null;
    this._outcome = 0;
    this._move = [0, 0]; // the last action's displacement
    this._inspected = false;
    this._word = 0;
    this._wordRule = null; // {kind: "joint_attention", object, word}: the word shown when the object is visible, else 0
    this._goal = new Float64Array(GOALS);
    this._episode = -1;
    this._event = 0;
    this._tick = 0;
    this._pending = null;
    this._decisionAction = null;
    this._doorRule = "toggle"; // or "locked": interact no longer opens doors
    this._mask = new Uint8Array(ACTIONS.length).fill(1); // the task's legal actions
    this._deadline = null; // a task's own time limit (tick), truncating earlier than the horizon
    this.rewardRule = null; // {kind: "reach", target} | {kind: "cue", paid} | null, set by a task
    if (build) this._build();
  }

  // -- the private map

  _build() {
    const rng = this.rng, cells = allCells(), edges = [];
    for (const [x, y] of cells) {
      if (x + 1 < SIZE) edges.push([[x, y], [x + 1, y]]);
      if (y + 1 < SIZE) edges.push([[x, y], [x, y + 1]]);
    }
    // a random spanning tree stays open so every cell is reachable through doors at most
    const order = permutation(rng, edges.length);
    const parent = new Map(cells.map((c) => [cellKey(c), cellKey(c)]));
    const root = (k) => { while (parent.get(k) !== k) { parent.set(k, parent.get(parent.get(k))); k = parent.get(k); } return k; };
    const tree = new Set();
    for (const k of order) {
      const [a, b] = edges[k], ra = root(cellKey(a)), rb = root(cellKey(b));
      if (ra !== rb) { parent.set(ra, rb); tree.add(k); }
    }
    let rest = []; for (let k = 0; k < edges.length; k++) if (!tree.has(k)) rest.push(k);
    rest = permutation(rng, rest.length).map((i) => rest[i]);
    const kinds = new Array(edges.length).fill("open");
    const [jx, jy] = this.config.junction;
    const indexOf = (e) => edges.findIndex((f) => sameCell(f[0], e[0]) && sameCell(f[1], e[1]));
    const protectedEdges = new Set([indexOf([[jx - 1, jy], [jx, jy]]), indexOf([[jx, jy - 1], [jx, jy]])]);
    rest = rest.filter((k) => !protectedEdges.has(k));
    for (const k of rest.slice(0, this.config.walls)) kinds[k] = "wall";
    let candidates = []; for (let k = 0; k < edges.length; k++) if (kinds[k] === "open" && !protectedEdges.has(k)) candidates.push(k);
    candidates = permutation(rng, candidates.length).map((i) => candidates[i]);
    for (const k of candidates.slice(0, this.config.doors)) kinds[k] = "door_closed";
    this._edges = new Map();
    for (let k = 0; k < edges.length; k++) this._edges.set(edgeKeyOf(edges[k][0], edges[k][1]), kinds[k]);
    this._doors = new Set(candidates.slice(0, this.config.doors).map((k) => edgeKeyOf(edges[k][0], edges[k][1])));
    this._colors = permutation(rng, COLORS);
    this._wordOf = permutation(rng, WORDS);
  }

  _edgeKey(cell, other) { const k = edgeKeyOf(cell, other); return this._edges.has(k) ? k : edgeKeyOf(other, cell); }

  _passage(cell, direction) {
    const [dx, dy] = DIRECTIONS[direction], other = [cell[0] + dx, cell[1] + dy];
    if (!inBounds(other)) return "wall";
    const kind = this._edges.get(this._edgeKey(cell, other));
    if (kind === undefined) throw Error(`no edge between ${cellKey(cell)} and ${cellKey(other)}`);
    return kind;
  }

  _setPassage(cell, direction, kind) {
    const [dx, dy] = DIRECTIONS[direction];
    this._edges.set(this._edgeKey(cell, [cell[0] + dx, cell[1] + dy]), kind);
  }

  objectAt(cell) { for (const [k, where] of this._objects) if (where !== null && sameCell(where, cell)) return k; return null; }

  // -- privileged access for tasks, the page and evaluation only

  /** `World.state` of env.py plus the whole private state: what the fixture dumps and the parity harness compares. */
  state() {
    return {
      format: WORLD_STATE_FORMAT, size: SIZE, config: { ...this.config, junction: [...this.config.junction] }, seed: this.seed, life_id: this.life_id,
      edges: [...this._edges].map(([k, kind]) => { const [a, b] = k.split("|"); return [cellOfKey(a), cellOfKey(b), kind]; }),
      doors: [...this._doors].map((k) => { const [a, b] = k.split("|"); return [cellOfKey(a), cellOfKey(b)]; }),
      colors: [...this._colors], words: [...this._wordOf],
      objects: [...this._objects].map(([k, where]) => [k, copyCell(where)]),
      agent: copyCell(this._agent), carrying: this._carrying, outcome: this._outcome, inspected: this._inspected,
      move: [this._move[0], this._move[1]],
      word: this._word, word_rule: this._wordRule === null ? null : { ...this._wordRule }, goal: Array.from(this._goal),
      episode: this._episode, event: this._event, tick: this._tick, pending: this._pending, decision_action: this._decisionAction,
      door_rule: this._doorRule, mask: Array.from(this._mask), deadline: this._deadline,
      reward_rule: this.rewardRule === null ? null : { ...this.rewardRule, ...(this.rewardRule.target ? { target: copyCell(this.rewardRule.target) } : {}), ...(this.rewardRule.paid ? { paid: copyCell(this.rewardRule.paid) } : {}) },
    };
  }

  /** A world with exactly the private state of a dump (the Python fixture's or `state()`), its own random stream from `seed`. */
  static fromState(s, { seed = null, life_id = null } = {}) {
    if (s.format !== WORLD_STATE_FORMAT) throw Error(`not a ${WORLD_STATE_FORMAT} state`);
    if (s.size !== SIZE) throw Error(`the state is a ${s.size} by ${s.size} world; this port is ${SIZE} by ${SIZE}`);
    const w = new World(s.config, { seed: seed ?? s.seed ?? 0, life_id: life_id ?? s.life_id ?? "world", build: false });
    w._edges = new Map(s.edges.map(([a, b, kind]) => [edgeKeyOf(a, b), kind]));
    w._doors = new Set(s.doors.map(([a, b]) => edgeKeyOf(a, b)));
    w._colors = s.colors.map((c) => c | 0); w._wordOf = s.words.map((c) => c | 0);
    w._objects = new Map(s.objects.map(([k, where]) => [k | 0, copyCell(where)]));
    w._agent = copyCell(s.agent); w._carrying = s.carrying ?? null; w._outcome = s.outcome | 0; w._inspected = !!s.inspected;
    w._move = s.move ? [s.move[0] | 0, s.move[1] | 0] : [0, 0];
    w._word = s.word | 0; w._wordRule = s.word_rule ? { ...s.word_rule } : null; w._goal = Float64Array.from(s.goal);
    w._episode = s.episode | 0; w._event = s.event | 0; w._tick = s.tick | 0; w._pending = s.pending ?? null; w._decisionAction = s.decision_action ?? null;
    w._doorRule = s.door_rule; w._mask = Uint8Array.from(s.mask, (x) => (x ? 1 : 0)); w._deadline = s.deadline ?? null;
    w.rewardRule = s.reward_rule ? { ...s.reward_rule } : null;
    return w;
  }

  /** What `World.state` exposes in env.py (evaluation and rendering; never handed to the agent). */
  get privileged() { return { agent: copyCell(this._agent), objects: Object.fromEntries([...this._objects].map(([k, c]) => [k, copyCell(c)])), carrying: this._carrying, colors: [...this._colors], words: [...this._wordOf], door_rule: this._doorRule }; }
  get agent() { return copyCell(this._agent); }
  get tick() { return this._tick; }
  get episode() { return this._episode; }
  get event() { return this._event; }
  get deadline() { return this._deadline; }
  get carrying() { return this._carrying; }
  get inspected() { return this._inspected; }
  get word() { return this._word; }
  get doorRule() { return this._doorRule; }
  get mask() { return Uint8Array.from(this._mask); }
  get goal() { return Float64Array.from(this._goal); }
  objects() { return Object.fromEntries([...this._objects].map(([k, c]) => [k, copyCell(c)])); }
  colorOf(k) { return this._colors[k]; }
  wordOf(k) { return this._wordOf[k]; }
  passage(cell, direction) { return this._passage(cell, direction); }

  placeObject(k, cell) { this._objects.set(k | 0, copyCell(cell)); }
  setWord(word) { this._word = word | 0; this._wordRule = null; }
  /** Joint attention: the word shown each moment is computed from what is visible. */
  setWordRule(rule) { this._wordRule = rule === null ? null : { kind: "joint_attention", object: rule.object | 0, word: rule.word | 0 }; }

  /** Objects in the cell, plus those in adjacent cells after an inspect. */
  visibleObjects() {
    const [x, y] = this._agent, out = [];
    const here = this.objectAt(this._agent);
    if (here !== null) out.push(here);
    if (this._inspected) {
      for (const direction of DIRECTION_NAMES) {
        const [dx, dy] = DIRECTIONS[direction], cell = [x + dx, y + dy];
        if (this._passage(this._agent, direction) !== "wall" && inBounds(cell)) { const k = this.objectAt(cell); if (k !== null) out.push(k); }
      }
    }
    return out;
  }

  setGoal(goal) { this._goal = Float64Array.from(goal, Number); }
  setDoorRule(rule) { this._doorRule = rule; }
  /** A task's time limit: `steps` more decisions from now, or null for the horizon alone. */
  setDeadline(steps) { this._deadline = steps === null || steps === undefined ? null : this._tick + (steps | 0); }
  /** The legal actions a task allows (a rule of the task, visible to the agent as the mask). */
  setMask(legal) { this._mask = legal === null || legal === undefined ? new Uint8Array(ACTIONS.length).fill(1) : Uint8Array.from(legal, (x) => (x ? 1 : 0)); }
  randomCell() { return [Math.floor(this.rng.random() * SIZE), Math.floor(this.rng.random() * SIZE)]; }

  /** Privileged planner for the upper reference and for evaluation: breadth first, neighbours in north, east, south, west order. */
  shortestPath(start, goal, { doorsOpen = false } = {}) {
    const queue = [copyCell(start)];
    const came = new Map([[cellKey(start), null]]);
    let head = 0;
    while (head < queue.length) {
      let cell = queue[head++];
      if (sameCell(cell, goal)) {
        const path = [];
        let k = cellKey(cell);
        while (came.get(k) !== null) { const [from, action] = came.get(k); path.push(action); k = cellKey(from); }
        return path.reverse();
      }
      for (const direction of DIRECTION_NAMES) {
        const kind = this._passage(cell, direction);
        if (kind === "wall" || (kind === "door_closed" && !doorsOpen)) continue;
        const [dx, dy] = DIRECTIONS[direction], nxt = [cell[0] + dx, cell[1] + dy];
        if (!came.has(cellKey(nxt))) { came.set(cellKey(nxt), [cell, direction]); queue.push(nxt); }
      }
    }
    return null;
  }

  // -- episodes

  reset({ agent = null } = {}) {
    this._episode += 1;
    this._tick = 0;
    this._pending = null;
    this._agent = agent === null ? this.randomCell() : copyCell(agent);
    this._carrying = null;
    this._outcome = 0;
    this._move = [0, 0];
    this._inspected = false;
    this._deadline = null;
    return this._moment({ feedback: false, reward: 0.0, terminated: false, truncated: false });
  }

  /** The observation and its observed flags (the four neighbour fields, off before an inspect; every other field is observed). */
  observation() {
    const [x, y] = this._agent, here = this.objectAt(this._agent);
    if (this._wordRule !== null) this._word = this.visibleObjects().includes(this._wordRule.object) ? this._wordRule.word : 0;
    const obs = {
      x: oneHot(x, SIZE), y: oneHot(y, SIZE),
      dx: oneHot(this._move[0] + 1, 3), dy: oneHot(this._move[1] + 1, 3),
      passage_north: oneHot(PASSAGE.indexOf(this._passage(this._agent, "north")), 3),
      passage_east: oneHot(PASSAGE.indexOf(this._passage(this._agent, "east")), 3),
      passage_south: oneHot(PASSAGE.indexOf(this._passage(this._agent, "south")), 3),
      passage_west: oneHot(PASSAGE.indexOf(this._passage(this._agent, "west")), 3),
      here_object: oneHot(here === null ? 0 : here + 1, OBJECTS + 1),
      here_color: oneHot(here === null ? 0 : this._colors[here] + 1, COLORS + 1),
      carrying: oneHot(this._carrying === null ? 0 : this._carrying + 1, OBJECTS + 1),
      outcome: oneHot(this._outcome, OUTCOMES.length),
      word: oneHot(this._word, WORDS + 1),
      phase: oneHot(this._tick === 0 ? 0 : 1, 2),
    };
    const observed = {};
    for (const direction of DIRECTION_NAMES) {
      const [dx, dy] = DIRECTIONS[direction], cell = [x + dx, y + dy];
      const visible = this._inspected && this._passage(this._agent, direction) !== "wall" && inBounds(cell);
      const obj = visible ? this.objectAt(cell) : null;
      obs[`${direction}_object`] = oneHot(obj === null ? 0 : obj + 1, OBJECTS + 1);
      observed[`${direction}_object`] = new Uint8Array(OBJECTS + 1).fill(visible ? 1 : 0);
    }
    return { observation: obs, observed };
  }

  _moment({ feedback, reward, terminated, truncated }) {
    const { observation, observed } = this.observation();
    const m = {
      life_id: this.life_id, episode_id: this._episode, event_id: this._event, tick: this._tick, dt: 1.0,
      observation, observed, action_mask: Uint8Array.from(this._mask),
      feedback_for: feedback ? this._pending : null, executed: feedback ? this._decisionAction : null,
      reward, reward_known: feedback, terminated, truncated,
      final_observation: truncated ? observation : null, goal: Float64Array.from(this._goal),
    };
    this._event += 1;
    return m;
  }

  /** The reward rule a task set: (reward, terminated) after an action (tasks.py `_reach_reward` and `arm_cue`). */
  _reward(name, moved) {
    const rule = this.rewardRule;
    if (rule === null) return [0.0, false];
    if (rule.kind === "reach") return sameCell(this._agent, rule.target) ? [1.0, true] : [-0.01, false];
    if (rule.kind === "cue") {
      const here = this._agent;
      if (ROOMS.some((r) => sameCell(r, here))) return [sameCell(here, rule.paid) ? 1.0 : 0.0, true];
      return [0.0, false];
    }
    throw Error(`unknown reward rule ${rule.kind}`);
  }

  act(decisionId, action) {
    if (this._pending !== null) throw Error("the previous decision has not been fed back");
    action = action | 0;
    const name = ACTIONS[action];
    if (name === undefined) throw Error("action outside the six actions");
    if (!this._mask[action]) throw Error(`action ${name} is not legal under the task's mask`);
    this._pending = decisionId; this._decisionAction = action;
    this._inspected = false;
    this._move = [0, 0];
    let moved = false;
    if (name in DIRECTIONS) {
      const kind = this._passage(this._agent, name);
      if (kind === "open") {
        const [dx, dy] = DIRECTIONS[name];
        this._agent = [this._agent[0] + dx, this._agent[1] + dy];
        this._move = [dx, dy];
        moved = true;
        this._outcome = OUTCOMES.indexOf("moved");
      } else this._outcome = OUTCOMES.indexOf("blocked");
    } else if (name === "inspect") {
      this._inspected = true;
      this._outcome = OUTCOMES.indexOf("inspected");
    } else {
      const here = this.objectAt(this._agent);
      if (this._carrying === null && here !== null) {
        this._objects.set(here, null);
        this._carrying = here;
        this._outcome = OUTCOMES.indexOf("picked");
      } else if (this._carrying !== null && here === null) {
        this._objects.set(this._carrying, copyCell(this._agent));
        this._carrying = null;
        this._outcome = OUTCOMES.indexOf("dropped");
      } else {
        let toggled = false;
        for (const direction of DIRECTION_NAMES) {
          const kind = this._passage(this._agent, direction);
          if (kind === "door_closed" && this._doorRule === "toggle") { this._setPassage(this._agent, direction, "open"); toggled = true; }
          else if (kind === "open" && this._doorRule === "toggle" && this._wasDoor(direction)) { this._setPassage(this._agent, direction, "door_closed"); toggled = true; }
        }
        this._outcome = OUTCOMES.indexOf(toggled ? "toggled" : "none");
      }
    }
    this._tick += 1;
    const [reward, terminated] = this._reward(name, moved);
    const truncated = !terminated && (this._tick >= this.config.horizon || (this._deadline !== null && this._tick >= this._deadline));
    const m = this._moment({ feedback: true, reward, terminated, truncated });
    this._pending = null;
    return m;
  }

  /** An open passage that was built as a door. */
  _wasDoor(direction) {
    const [dx, dy] = DIRECTIONS[direction];
    return this._doors.has(this._edgeKey(this._agent, [this._agent[0] + dx, this._agent[1] + dy]));
  }
}

// -- the curriculum (tasks.py)

export const GOAL_EXPLORE = 0, GOAL_OBJECT = 1, GOAL_WORD = 5, GOAL_CUE = 6;
export const REQUEST_DISTANCE = Math.floor(SIZE / 2) + 1; // a request starts at least this far (Manhattan) from its target
export const CUE_WORDS = [7, 8]; // the two cue words (1-based in the word field: 0 is "no word")
export const ROOMS = [[0, 1], [1, 0]]; // the two reward rooms of the cue task: west and north of the junction (1, 1)

export function goalVector(index) { return oneHot(index, GOALS); }

/** Decisions allowed to fulfil a request: twice the world's own shortest path plus two. */
export function requestDeadline(world, target) {
  const path = world.shortestPath(world.agent, target, { doorsOpen: true });
  return 2 * (path !== null ? path.length : 6) + 2;
}

/** What one task episode asks, for evaluation: the private answer never reaches the agent. */
export function episodeRecord(task, over = {}) { return { task, target_cell: null, object_id: null, delay: 0, cue: null, rewarded_room: null, old_cell: null, ...over }; }

export class Curriculum {
  /** Runs task episodes on a world; the world's random stream is separate from the agent's. */
  constructor(world, seed, { cueRule = null } = {}) {
    this.world = world;
    this.rng = new Mulberry32(seed);
    this.cueRule = Math.floor(this.rng.random() * 2); // which cue word rewards which room in this life
    if (cueRule !== null) this.cueRule = cueRule | 0;
    this._pendingMove = null;
  }

  _clearObjects() { for (let k = 0; k < OBJECTS; k++) this.world.placeObject(k, null); }

  _freeCell(exclude) { for (;;) { const c = this.world.randomCell(); if (!exclude.has(cellKey(c))) return c; } }

  _reachReward(target) { return { kind: "reach", target: copyCell(target) }; }

  /** Free exploration: objects scattered, no goal, no reward; time limit only. */
  explore() {
    this._clearObjects();
    this.world.setMask(null);
    const cells = new Set();
    for (let k = 0; k < OBJECTS; k++) { const c = this._freeCell(cells); cells.add(cellKey(c)); this.world.placeObject(k, c); }
    this.world.setWord(0);
    this.world.setGoal(goalVector(GOAL_EXPLORE));
    this.world.rewardRule = null;
    return episodeRecord("explore");
  }

  /** See one object once at the start, wander `delay` steps with no goal, then be asked for it. */
  remember(delay, { move = null } = {}) {
    this._clearObjects();
    const k = Math.floor(this.rng.random() * OBJECTS);
    const start = this.world.randomCell();
    this.world.placeObject(k, start); // the agent starts on the object and sees it here
    const others = new Set([cellKey(start)]);
    for (let j = 0; j < OBJECTS; j++) if (j !== k) { const c = this._freeCell(others); others.add(cellKey(c)); this.world.placeObject(j, c); }
    this.world.setWord(0);
    this.world.setGoal(goalVector(GOAL_EXPLORE));
    this.world.rewardRule = null;
    this.world.setMask([1, 1, 1, 1, 1, 0]); // no interact while wandering: the object stays put
    this._pendingMove = move;
    return episodeRecord("remember", { target_cell: copyCell(start), object_id: k, delay, old_cell: copyCell(start) });
  }

  /** The cells a request may start from: free of objects and at least REQUEST_DISTANCE from the target, else every cell but the target (tasks.py `_far_start`). */
  farCandidates(target) {
    const objects = new Set(Object.values(this.world.objects()).filter((c) => c !== null).map(cellKey));
    const far = allCells().filter((c) => !objects.has(cellKey(c)) && manhattan(c, target) >= REQUEST_DISTANCE);
    return far.length ? far : allCells().filter((c) => !sameCell(c, target));
  }

  _farStart(target) { const far = this.farCandidates(target); return far[Math.floor(this.rng.random() * far.length)]; }

  /** After the delay the request starts a new episode of the same life from a distant cell, the goal naming the object; returns its first moment. */
  request(episode) {
    if (episode.object_id === null) throw Error("a request needs the remembered object");
    let target = this.world.objects()[episode.object_id];
    if (target === null || target === undefined) { target = episode.old_cell; this.world.placeObject(episode.object_id, target); } // carried after all: put it back where it was witnessed
    episode.target_cell = copyCell(target);
    this.world.setMask(null);
    this.world.setGoal(goalVector(GOAL_OBJECT + episode.object_id));
    this.world.rewardRule = this._reachReward(target);
    const m = this.world.reset({ agent: this._farStart(target) });
    this.world.setDeadline(requestDeadline(this.world, target));
    return m;
  }

  /** Move the remembered object to a new cell (tasks.py move_object). Visibly means onto the agent's own
   * empty cell, so the next moment shows it; invisibly means a free cell the agent neither stands on nor saw it in. */
  moveObject(episode, { visible }) {
    if (episode.object_id === null) throw Error("a move needs the remembered object");
    const agent = this.world.agent;
    const occupied = new Set(Object.values(this.world.objects()).filter((c) => c !== null).map(cellKey));
    let next;
    if (visible) {
      if (this.world.objectAt(agent) !== null) throw Error("a visible move needs the agent on an empty cell");
      next = agent;
    }
    else next = this._freeCell(new Set([...occupied, cellKey(agent), cellKey(episode.old_cell)]));
    this.world.placeObject(episode.object_id, next);
    episode.target_cell = copyCell(next);
  }

  /** A cue word at the start decides which room pays; the agent is far from both rooms. */
  cue(delay) {
    this._clearObjects();
    const cue = Math.floor(this.rng.random() * 2);
    this.world.setWord(CUE_WORDS[cue]);
    const rewarded = ROOMS[cue ^ this.cueRule];
    this.world.setGoal(goalVector(GOAL_CUE));
    this.world.setMask([1, 1, 1, 1, 1, 0]);
    this.world.rewardRule = null; // nothing pays during the wander
    return episodeRecord("cue", { delay, cue, rewarded_room: copyCell(rewarded) });
  }

  /** The two-choice decision, a new episode of the same life at the junction: only the two room entries are legal, the word is hidden; returns its first moment. */
  armCue(episode, rewarded = null) {
    const paid = rewarded === null ? episode.rewarded_room : rewarded;
    const legal = new Array(ACTIONS.length).fill(0);
    legal[ACTIONS.indexOf("north")] = 1; legal[ACTIONS.indexOf("west")] = 1;
    this.world.setMask(legal);
    this.world.setWord(0);
    this.world.setGoal(goalVector(GOAL_CUE));
    this.world.rewardRule = { kind: "cue", paid: copyCell(paid) };
    return this.world.reset({ agent: this.world.config.junction });
  }

  hideCue() { this.world.setWord(0); }

  /** Word grounding: teaching under joint attention, or the word alone as a request (`nameRequest`). */
  word({ teach = true } = {}) {
    this._clearObjects();
    this.world.setMask([1, 1, 1, 1, 1, 0]); // nothing is carried off while the word is taught
    const k = Math.floor(this.rng.random() * OBJECTS);
    const word = this.world.wordOf(k) + 1; // the life's label of object k, 1-based
    const cells = new Set();
    for (let j = 0; j < OBJECTS; j++) { const c = this._freeCell(cells); cells.add(cellKey(c)); this.world.placeObject(j, c); }
    this.world.setGoal(goalVector(GOAL_EXPLORE));
    this.world.rewardRule = null;
    if (teach) this.world.setWordRule({ object: k, word }); else this.world.setWord(0);
    const target = this.world.objects()[k];
    // the episode starts on the referent: the first scene is guaranteed, its place witnessed
    return episodeRecord("word", { target_cell: copyCell(target), object_id: k, cue: word, old_cell: copyCell(target) });
  }

  /** The word alone, as a request in a new episode from a distant cell: reach its referent within the deadline; returns its first moment. */
  nameRequest(episode) {
    if (episode.object_id === null || episode.cue === null) throw Error("a name request needs the taught word and its referent");
    this.world.setWordRule(null);
    this.world.setWord(episode.cue);
    this.world.setMask(null);
    this.world.setGoal(goalVector(GOAL_WORD));
    let target = this.world.objects()[episode.object_id];
    if (target === null || target === undefined) { target = episode.target_cell; this.world.placeObject(episode.object_id, target); }
    episode.target_cell = copyCell(target);
    this.world.rewardRule = this._reachReward(target);
    const m = this.world.reset({ agent: this._farStart(target) });
    this.world.setDeadline(requestDeadline(this.world, target));
    return m;
  }

  /** A changed consequence: interact no longer opens doors, mid-life. */
  lockDoors() { this.world.setDoorRule("locked"); }
}

/** Random legal-looking actions for the delay phase (moves only, no interact). */
export function scriptedWander(world, rng, steps) { return Array.from({ length: steps }, () => Math.floor(rng.random() * 4)); }
