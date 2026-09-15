// The remembered world's search and its brain wrapper in JavaScript: a port of `compose`,
// `observe_from`, `IMAGINED_OBSERVED`, `WorldPlanner` and `Brain` from
// world/brain.py. The search is best first over the learned
// model's imagined consequences (a binary heap ordered by (f, g, id) exactly as Python's
// heapq orders its tuples, and the ids are unique, so the pop sequence is the same), with the
// same memoisation of imagined consequences per visible state, depth-0 marker and action, the
// same batch composition of every `predictBatch` call (the rows must be the same rows), the
// same pruning of moves through walls and closed doors, the same closed set and best-cost
// table, the same tie-breaking and the same Mulberry32 fallback. The imagined position
// composes from the predicted displacement (`dx`, `dy`); a known next cell takes its
// passages from the witnessed map (`MapStore`); an imagined reading carries the flags a
// real moment would (`IMAGINED_OBSERVED`: every field observed except the four neighbour
// fields) while depth-0 readings carry the real moment's flags. `Brain` witnesses every
// moment into the four stores, reads three of them into the recall port through the
// engine's `recallDrive` hook, chooses the target (remembered place or the nearest
// uninspected cell), reuses the plan of the previous decision while the world follows the
// imagined path, and steps the agent.

import { ExperienceAgent, Mulberry32, normalizeMoment } from "../../web/engine.js";
import { ACTIONS, ACTION_INSPECT, ACTION_INTERACT, DIRECTIONS, DIRECTION_NAMES, FIELDS, FIELD_NAMES, GOAL_CUE, GOAL_EXPLORE, GOAL_OBJECT, GOAL_WORD, OBJECTS, OUTCOMES, PASSAGE, SIZE, allCells, argmaxOf, cellKey, manhattan, oneHot, sameCell } from "./world.js";
import { CueStore, MapStore, PlaceStore, RECALL_WIDTH, WordStore } from "./stores.js";

export const CONSEQUENCES = ["dx", "dy", "passage_north", "passage_east", "passage_south", "passage_west", "here_object", "here_color", "carrying", "outcome"];
const DIRECTION_ACTIONS = DIRECTION_NAMES.map((d) => ACTIONS.indexOf(d));
const MOVES = [...DIRECTION_ACTIONS, ACTION_INTERACT];
const OUTCOME_INSPECTED = OUTCOMES.indexOf("inspected");

/** The flags of an imagined state: every field observed except the four neighbour-object fields (an imagined cell's neighbours are never in view). */
export const IMAGINED_OBSERVED = Object.fromEntries(FIELD_NAMES.map((name) => [name, new Uint8Array(FIELDS[name]).fill(1)]));
for (const d of DIRECTION_NAMES) IMAGINED_OBSERVED[`${d}_object`] = new Uint8Array(OBJECTS + 1);

export function observeFrom(obs) {
  return { cell: [argmaxOf(obs.x), argmaxOf(obs.y)], passages: DIRECTION_NAMES.map((d) => PASSAGE[argmaxOf(obs[`passage_${d}`])]) };
}

/** The imagined next observation: predicted consequences replace the visible fields, the position composes from the predicted displacement, a known cell's passages come from the witnessed map (`mapStore`), neighbours stay unknown, the word stays, the phase moves on. */
export function compose(observation, prediction, mapStore = null) {
  const out = { ...observation };
  for (const name of CONSEQUENCES) { const p = prediction[`next_${name}`]; out[name] = oneHot(argmaxOf(p), p.length); }
  let x = argmaxOf(observation.x), y = argmaxOf(observation.y);
  x = Math.min(Math.max(x + argmaxOf(out.dx) - 1, 0), SIZE - 1);
  y = Math.min(Math.max(y + argmaxOf(out.dy) - 1, 0), SIZE - 1);
  out.x = oneHot(x, SIZE); out.y = oneHot(y, SIZE);
  if (mapStore !== null && mapStore !== undefined) {
    const { passages } = mapStore.recall([x, y]);
    if (passages !== null) DIRECTION_NAMES.forEach((d, k) => { out[`passage_${d}`] = oneHot(PASSAGE.indexOf(passages[k]), 3); });
  }
  for (const d of DIRECTION_NAMES) out[`${d}_object`] = oneHot(0, OBJECTS + 1);
  out.phase = Float64Array.of(0.0, 1.0);
  return out;
}

/** The visible state as a memo key: the argmax of every field in FIELDS order (the Python tuple). */
export function stateKey(obs) { return FIELD_NAMES.map((name) => argmaxOf(obs[name])).join(","); }

/** heapq order for the frontier tuples (f, g, id, ...): the ids are unique, so the order is total. */
class Frontier {
  constructor() { this.items = []; }
  get length() { return this.items.length; }
  static less(a, b) { return a.f !== b.f ? a.f < b.f : a.g !== b.g ? a.g < b.g : a.id < b.id; }
  push(item) {
    const h = this.items; h.push(item);
    let k = h.length - 1;
    while (k > 0) { const parent = (k - 1) >> 1; if (Frontier.less(h[k], h[parent])) { [h[k], h[parent]] = [h[parent], h[k]]; k = parent; } else break; }
  }
  pop() {
    const h = this.items, top = h[0], last = h.pop();
    if (h.length) {
      h[0] = last;
      let k = 0;
      for (;;) {
        const l = 2 * k + 1, r = l + 1;
        let m = k;
        if (l < h.length && Frontier.less(h[l], h[m])) m = l;
        if (r < h.length && Frontier.less(h[r], h[m])) m = r;
        if (m === k) break;
        [h[k], h[m]] = [h[m], h[k]]; k = m;
      }
    }
    return top;
  }
}

export class WorldPlanner {
  /** Best-first search over the learned model toward a target cell, or one-step reward greed. */
  constructor({ max_nodes = 128, depth = 8, seed = 0, rng = null, fallbacks = 0, pruned = 0 } = {}) {
    this.maxNodes = max_nodes | 0; this.depth = depth | 0; this.seed = seed | 0;
    this.rng = new Mulberry32((((seed * 104729) % 4294967296) + 7) % 4294967296);
    if (rng !== null && rng !== undefined) this.rng.state = Number(rng) >>> 0;
    this.fallbacks = fallbacks | 0; this.pruned = pruned | 0;
  }

  /**
   * {action, prediction, expanded, path, cells}: the first action, its prediction, the nodes expanded, the imagined path and its cells.
   * `legal` restricts the moves; `mapStore` supplies the witnessed passages of known cells after a move; `observed` is the real
   * moment's flag map (imagined states carry IMAGINED_OBSERVED).
   */
  search(model, observation, goal, target, legal = null, mapStore = null, observed = null) {
    const allowed = new Set();
    if (legal === null || legal === undefined) for (let a = 0; a < ACTIONS.length; a++) allowed.add(a); else for (let a = 0; a < legal.length; a++) if (legal[a]) allowed.add(a);
    const start = observeFrom(observation).cell;
    const rootFlags = observed === null || observed === undefined ? IMAGINED_OBSERVED : observed;
    if (sameCell(start, target)) {
      const action = allowed.has(ACTION_INSPECT) ? ACTION_INSPECT : Math.min(...allowed);
      const pred = model.predictBatch([observation], [action], goal, { observed: [rootFlags] })[0];
      return { action, prediction: pred, expanded: 1, path: [action], cells: [start] };
    }
    let counter = 0;
    const frontier = new Frontier();
    frontier.push({ f: manhattan(start, target), g: 0, id: counter, obs: observation, first: null, firstPred: null, depth: 0 });
    let best = { h: manhattan(start, target), path: null, pred: null };
    let expanded = 0;
    const closed = new Set(); // cells already expanded (A* over cells: unit costs, consistent heuristic)
    const bestG = new Map([[cellKey(start), 0]]);
    const cache = new Map(); // imagined consequences of one search, keyed by the visible state, the depth-0 marker and the action
    const imagine = (obs, moves, depth) => {
      const flags = depth === 0 ? rootFlags : IMAGINED_OBSERVED;
      const key = `${stateKey(obs)}|${depth === 0 ? 1 : 0}`;
      const missing = moves.filter((a) => !cache.has(`${key}|${a}`));
      if (missing.length) { const preds = model.predictBatch(missing.map(() => obs), missing, goal, { observed: missing.map(() => flags) }); missing.forEach((a, k) => cache.set(`${key}|${a}`, preds[k])); }
      return moves.map((a) => cache.get(`${key}|${a}`));
    };
    while (frontier.length && expanded < this.maxNodes) {
      const node = frontier.pop();
      const { g, obs, first, firstPred, depth } = node;
      const hereCell = observeFrom(obs).cell;
      if (closed.has(cellKey(hereCell)) || depth >= this.depth) continue;
      closed.add(cellKey(hereCell));
      const moves = MOVES.filter((a) => allowed.has(a));
      const predictions = imagine(obs, moves, depth);
      expanded += 1;
      for (let k = 0; k < moves.length; k++) {
        const action = moves[k], pred = predictions[k], name = ACTIONS[action];
        // a move through what the state shows as a wall or a closed door is impossible: the candidate is discarded, never repaired into the right next cell
        if (name in DIRECTIONS && PASSAGE[argmaxOf(obs[`passage_${name}`])] !== "open") { this.pruned += 1; continue; }
        const nxt = compose(obs, pred, name in DIRECTIONS ? mapStore : null);
        const cell = observeFrom(nxt).cell;
        if (name in DIRECTIONS && sameCell(cell, observeFrom(obs).cell)) continue; // the model imagines standing still: no progress along this branch
        const path = first !== null ? [...first, action] : [action];
        const childPred = firstPred === null ? pred : firstPred;
        const h = manhattan(cell, target);
        if (sameCell(cell, target)) return { action: path[0], prediction: childPred, expanded, path: path.slice(), cells: this._cellsOf(observation, path, cache, model, goal, mapStore) };
        if (h < best.h) best = { h, path, pred: childPred };
        if (closed.has(cellKey(cell)) || g + 1 >= (bestG.has(cellKey(cell)) ? bestG.get(cellKey(cell)) : 1e9)) continue;
        bestG.set(cellKey(cell), g + 1);
        counter += 1;
        frontier.push({ f: g + 1 + h, g: g + 1, id: counter, obs: nxt, first: path, firstPred: childPred, depth: depth + 1 });
      }
    }
    if (best.path === null) { // no imagined move improves on the start: the model's least bad move
      let moves = DIRECTION_ACTIONS.filter((a) => allowed.has(a));
      if (!moves.length) moves = [...allowed].sort((x, y) => x - y);
      const predictions = model.predictBatch(moves.map(() => observation), moves, goal, { observed: moves.map(() => rootFlags) });
      const scored = moves.map((a, k) => ({ h: manhattan(observeFrom(compose(observation, predictions[k], mapStore)).cell, target), action: a, pred: predictions[k] }));
      const closest = Math.min(...scored.map((sc) => sc.h));
      const ties = scored.filter((sc) => sc.h === closest);
      const pick = ties[Math.floor(this.rng.random() * ties.length)];
      this.fallbacks += 1;
      return { action: pick.action, prediction: pick.pred, expanded, path: [pick.action], cells: [] };
    }
    return { action: best.path[0], prediction: best.pred, expanded, path: best.path.slice(), cells: this._cellsOf(observation, best.path, cache, model, goal, mapStore) };
  }

  /** The imagined cell after each action of `path` (from the search's own cache; the root reading carries the depth-0 marker). */
  _cellsOf(observation, path, cache, model, goal, mapStore = null) {
    const cells = [];
    let obs = observation;
    for (const a of path) {
      const key = `${stateKey(obs)}|${obs === observation ? 1 : 0}|${a}`;
      let pred = cache.get(key);
      if (pred === undefined) pred = model.predictBatch([obs], [a], goal, { observed: [IMAGINED_OBSERVED] })[0];
      obs = compose(obs, pred, ACTIONS[a] in DIRECTIONS ? mapStore : null);
      cells.push(observeFrom(obs).cell);
    }
    return cells;
  }

  greedyReward(model, observation, goal, legal = null, observed = null) {
    const actions = [];
    if (legal === null || legal === undefined) for (let a = 0; a < ACTIONS.length; a++) actions.push(a); else for (let a = 0; a < legal.length; a++) if (legal[a]) actions.push(a);
    const flags = observed === null || observed === undefined ? IMAGINED_OBSERVED : observed;
    const predictions = model.predictBatch(actions.map(() => observation), actions, goal, { observed: actions.map(() => flags) });
    const rewards = predictions.map((p) => p.reward[0]);
    const best = argmaxOf(rewards);
    return { action: actions[best], prediction: predictions[best], expanded: 1 };
  }

  toJSON() { return { max_nodes: this.maxNodes, depth: this.depth, seed: this.seed, rng: this.rng.state, fallbacks: this.fallbacks, pruned: this.pruned }; }
}

/** The candidate: agent + stores + planner behind the one `step` contract (brain.py `Brain`), around an engine agent built from a snapshot. */
export class Brain {
  /**
   * @param agent an ExperienceAgent (controller "planner") built from the snapshot
   * @param extra the snapshot's `extra` block as parity_fixture.py writes it: planner, stores, brain and recall records (all optional)
   */
  constructor(agent, extra = {}) {
    if (!(agent instanceof ExperienceAgent)) throw Error("the brain wraps an ExperienceAgent");
    this.agent = agent;
    if (agent.ports.recall.length !== RECALL_WIDTH) throw Error(`the recall port has ${agent.ports.recall.length} neurons; the stores need ${RECALL_WIDTH}`);
    this.recallGain = extra.recall && extra.recall.gain !== undefined ? Number(extra.recall.gain) : 2.0;
    this.places = new PlaceStore(); this.cue = new CueStore(); this.words = new WordStore(); this.map = new MapStore();
    if (extra.stores) { this.places.load(extra.stores.places); this.cue.load(extra.stores.cue); this.words.load(extra.stores.words); if (extra.stores.map) this.map.load(extra.stores.map); }
    this.planner = new WorldPlanner(extra.planner || {});
    this.visited = new Set(); this.inspected = new Set();
    this.lastMoment = null; this.lastDecision = null;
    this.nodes = 0;
    this.plan = []; // the remaining imagined path of the last search
    this.planCells = []; // the cell each remaining action should reach
    this.planKey = null; // [goal, target] the plan was made for
    this.searches = 0; this.reused = 0;
    this.lastTarget = null; this.lastSearch = null; // what the page draws: the target chosen and the last search's path
    const b = extra.brain || {};
    for (const c of b.visited || []) this.visited.add(cellKey(c));
    for (const c of b.inspected || []) this.inspected.add(cellKey(c));
    this.plan = (b.plan || []).map((a) => a | 0); this.planCells = (b.plan_cells || []).map((c) => [c[0] | 0, c[1] | 0]);
    this.planKey = b.plan_key ? [b.plan_key[0] | 0, [b.plan_key[1][0] | 0, b.plan_key[1][1] | 0]] : null;
    this.searches = b.searches | 0; this.reused = b.reused | 0; this.nodes = b.nodes | 0;
    agent.setPlanner((a, row, m, drive, free) => this._plan(a, row, m, drive, free));
    agent.setRecallDrive((m, row) => this._recall(m, row), extra.recall && extra.recall.last ? Float64Array.from(extra.recall.last, Number) : null);
  }

  // -- what the moment reveals goes into the stores (supplied bookkeeping of witnessed facts)

  witness(m) {
    const o = m.observation, cell = [argmaxOf(o.x), argmaxOf(o.y)];
    if (m.tick === 0 && m.feedback_for === null) { this.visited.clear(); this.inspected.clear(); } // the cue store holds its word across episodes
    this.visited.add(cellKey(cell));
    if (m.observed.x.every((v) => v === 1) && m.observed.passage_north.every((v) => v === 1)) this.map.see(cell, observeFrom(o).passages);
    const word = argmaxOf(o.word);
    this.cue.see(word);
    const here = argmaxOf(o.here_object) - 1;
    const visible = [];
    if (here >= 0) { this.places.see(here, cell); visible.push(here); }
    else for (let k = 0; k < OBJECTS; k++) { const { where, known } = this.places.recall(k); if (known >= 0.5 && sameCell(where, cell)) this.places.forget(k); } // the remembered place is empty: the record is stale
    if (argmaxOf(o.outcome) === OUTCOME_INSPECTED) {
      this.inspected.add(cellKey(cell));
      for (const d of DIRECTION_NAMES) {
        const [dx, dy] = DIRECTIONS[d];
        if (m.observed[`${d}_object`].every((x) => x === 1)) { const k = argmaxOf(o[`${d}_object`]) - 1; if (k >= 0) { this.places.see(k, [cell[0] + dx, cell[1] + dy]); visible.push(k); } }
      }
    }
    if (argmaxOf(m.goal) === GOAL_EXPLORE) this.words.scene(word, visible); // a word heard as a request is not a naming scene
  }

  _requestedObject(m) {
    const goal = argmaxOf(m.goal);
    if (goal >= GOAL_OBJECT && goal < GOAL_OBJECT + OBJECTS) return goal - GOAL_OBJECT;
    if (goal === GOAL_WORD) return this.words.referent(argmaxOf(m.observation.word)).object;
    return null;
  }

  /** The recall port's drive: the place read of the requested object, the cue held, the word read of the word shown, times the recall gain. */
  _recall(m, row) {
    const k = this._requestedObject(m), word = argmaxOf(m.observation.word);
    const parts = [this.places.drive(k), this.cue.value, this.words.drive(word)];
    const out = new Float64Array(RECALL_WIDTH);
    let at = 0;
    for (const p of parts) { for (let j = 0; j < p.length; j++) out[at + j] = p[j] * this.recallGain; at += p.length; }
    return out;
  }

  // -- decisions

  /** The nearest cell not yet inspected (ties by x, then y); every cell inspected starts the sweep again. */
  explorationTarget(cell) {
    let candidates = allCells().filter((c) => !this.inspected.has(cellKey(c)));
    if (!candidates.length) { this.inspected.clear(); candidates = allCells().filter((c) => !sameCell(c, cell)); }
    let best = null, bestKey = null;
    for (const c of candidates) { const key = [manhattan(cell, c), c[0], c[1]]; if (best === null || key[0] < bestKey[0] || (key[0] === bestKey[0] && (key[1] < bestKey[1] || (key[1] === bestKey[1] && key[2] < bestKey[2])))) { best = c; bestKey = key; } }
    return best;
  }

  _plan(agent, row, m, drive, free) {
    const observation = m.observation, goal = m.goal, which = argmaxOf(goal), cell = observeFrom(observation).cell;
    let action, pred, nodes;
    if (which === GOAL_CUE) {
      this.plan = [];
      const out = this.planner.greedyReward(agent, observation, goal, m.action_mask, m.observed);
      action = out.action; pred = out.prediction; nodes = out.expanded;
      this.lastTarget = null; this.lastSearch = { kind: "greedy", path: [action], cells: [], expanded: nodes, reused: false };
    } else {
      let target = null;
      const k = this._requestedObject(m);
      if (which !== GOAL_EXPLORE && k !== null) target = this.places.recall(k).where;
      if (target === null) target = this.explorationTarget(cell);
      if (sameCell(cell, target) && which !== GOAL_EXPLORE) target = this.explorationTarget(cell);
      const key = [which, target];
      const sameKey = this.planKey !== null && this.planKey[0] === key[0] && sameCell(this.planKey[1], key[1]);
      // the plan of the previous decision is reused while the world follows the imagined path
      if (this.plan.length && sameKey && this.planCells.length && sameCell(cell, this.planCells[0])) { this.plan.shift(); this.planCells.shift(); }
      else { this.plan = []; this.planCells = []; }
      if (this.plan.length && sameKey && m.action_mask[this.plan[0]]) {
        action = this.plan[0];
        pred = agent.predictBatch([observation], [action], goal, { observed: [m.observed] })[0];
        nodes = 0;
        this.reused += 1;
        this.lastSearch = { kind: "reuse", path: this.plan.slice(), cells: this.planCells.slice(), expanded: 0, reused: true };
      } else {
        const out = this.planner.search(agent, observation, goal, target, m.action_mask, this.map, m.observed);
        action = out.action; pred = out.prediction; nodes = out.expanded;
        this.searches += 1;
        this.plan = out.path.slice(); this.planCells = out.cells.slice(); this.planKey = key;
        this.lastSearch = { kind: "search", path: out.path.slice(), cells: out.cells.slice(), expanded: nodes, reused: false };
      }
      this.lastTarget = [target[0], target[1]];
    }
    this.nodes += nodes;
    return { action: action | 0, prediction: pred, budget: { expansions: nodes } };
  }

  /** One real moment: witness it into the stores, then the agent's event transaction; returns Decision.to_dict() or null. */
  step(moment) {
    const m = normalizeMoment(moment);
    this.witness(m);
    const d = this.agent.step(m);
    this.lastMoment = m; this.lastDecision = d;
    return d;
  }

  setEpsilon(epsilon) { this.agent.config.actor.epsilon = Number(epsilon); }
  get epsilon() { return this.agent.config.actor.epsilon; }

  /** The bookkeeping the fixture records, for the parity check and for a page. */
  counters() { return { searches: this.searches, reused: this.reused, nodes: this.nodes, fallbacks: this.planner.fallbacks, pruned: this.planner.pruned, rng: this.planner.rng.state }; }
  stores() { return { places: this.places.toJSON(), cue: this.cue.toJSON(), words: this.words.toJSON(), map: this.map.toJSON() }; }
  bookkeeping() {
    const cells = (set) => [...set].map((k) => k.split(",").map(Number));
    return { visited: cells(this.visited), inspected: cells(this.inspected), plan: this.plan.slice(), plan_cells: this.planCells.map((c) => [c[0], c[1]]), plan_key: this.planKey === null ? null : [this.planKey[0], [this.planKey[1][0], this.planKey[1][1]]], searches: this.searches, reused: this.reused, nodes: this.nodes };
  }
}
