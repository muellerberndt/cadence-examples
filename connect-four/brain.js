import { Circuit, settleTogether, zeros } from "../shared/nervous_system.js";
export const COLS = 7,
  ROWS = 6,
  order = [3, 2, 4, 1, 5, 0, 6];
export const lines = [];
for (let y = 0; y < 6; y++)
  for (let x = 0; x < 7; x++)
    for (const [dx, dy] of [
      [1, 0],
      [0, 1],
      [1, 1],
      [1, -1],
    ])
      if (x + 3 * dx < 7 && y + 3 * dy >= 0 && y + 3 * dy < 6)
        lines.push(
          Array.from({ length: 4 }, (_, i) => (y + i * dy) * 7 + x + i * dx),
        );
export const legal = (b) => order.filter((c) => !b[c]);
export function drop(board, column, player) {
  if (!Number.isInteger(column) || column < 0 || column >= 7 || board[column])
    throw Error("Illegal move");
  const next = board.slice();
  for (let y = 5; y >= 0; y--)
    if (!next[y * 7 + column]) {
      next[y * 7 + column] = player;
      return next;
    }
}
export function winner(b) {
  for (const line of lines) {
    const v = b[line[0]];
    if (v && line.every((i) => b[i] === v)) return v;
  }
  return 0;
}
export function features(b, p) {
  const f = zeros(5);
  for (const line of lines) {
    let us = 0,
      them = 0;
    for (const i of line) {
      us += +(b[i] === p);
      them += +(b[i] === -p);
    }
    if (!them) {
      if (us === 2) f[0]++;
      if (us === 3) f[1]++;
    }
    if (!us) {
      if (them === 2) f[2]++;
      if (them === 3) f[3]++;
    }
  }
  for (let y = 0; y < 6; y++) f[4] += b[y * 7 + 3] * p;
  return [f[0] / 20, f[1] / 10, f[2] / 20, f[3] / 10, f[4] / 7].map((v) =>
    Math.max(-0.99, Math.min(0.99, v)),
  );
}
const weights = [1, 4, -1, -4, 0.3];
// Exact two-step graded circuit: input owners settle first, then the value owner.
export function evaluate(b, p) {
  const f = features(b, p);
  return Math.tanh(f.reduce((s, v, i) => s + v * weights[i], 0)) * 100;
}
export function valueCircuit(b, p) {
  const f = features(b, p),
    state = [...f, evaluate(b, p) / 100];
  return {
    state,
    drive: [...f.map(Math.atanh), 0],
    edges: weights.map((w, i) => [i, 5, w]),
    names: [
      "Own pairs",
      "Own threats",
      "Opponent pairs",
      "Opponent threats",
      "Center control",
      "Predicted value",
    ],
    groups: [...Array(5).fill("features"), "value"],
    recurrent: true,
    steps: 2,
    dt: 1,
  };
}
export class Monitor extends Circuit {
  constructor() {
    super(
      [
        "Activity change",
        "Option ambiguity",
        "Budget pressure",
        "Alternatives present",
        "Integrated uncertainty",
        "Request more thought",
      ],
      [
        "self_input",
        "self_input",
        "self_input",
        "self_input",
        "self_state",
        "self_state",
      ],
      [
        [0, 4, 0.6],
        [1, 4, 0.8],
        [2, 4, -0.4],
        [3, 4, 0.4],
        [4, 5, 1],
        [2, 5, -1],
      ],
    );
    this.previous = null;
  }
  prepare(activity, scores, pressure = 0) {
    const repair = this.previous
      ? Math.max(...activity.map((v, i) => Math.abs(v - this.previous[i]))) /
        Math.max(1, ...activity.map(Math.abs))
      : 0;
    const ranked = scores.slice().sort((a, b) => b - a),
      ambiguity = ranked.length > 1 ? 1 / (1 + ranked[0] - ranked[1]) : 0;
    this.previous = activity.slice();
    return {
      repair,
      ambiguity,
      pressure,
      drive: [
        Math.min(1, repair),
        ambiguity,
        pressure,
        +(scores.length > 1),
        0,
        0,
      ],
    };
  }
  read(activity, scores, pressure = 0) {
    const { drive, ...review } = this.prepare(activity, scores, pressure);
    this.settle(drive, 40);
    return {
      ...review,
      uncertainty: this.state[4],
      request_more: this.state[5] > 0.35 && pressure < 1,
    };
  }
}
export function decisionCircuit(
  board,
  player,
  candidates,
  monitor = new Monitor(),
  pressure = 0,
) {
  const activity = Array.from({ length: 7 }, (_, i) =>
      Math.tanh((candidates.find((c) => c.column === i)?.score ?? 0) / 100),
    ),
    signal = monitor.prepare(
      activity,
      candidates.map((c) => c.score / 100),
      pressure,
    ),
    value = valueCircuit(board, player),
    choices = new Circuit(
      Array.from({ length: 7 }, (_, i) => `Column ${i + 1} commitment`),
      Array(7).fill("futures"),
      [],
    );
  choices.mask = activity.map(
    (_, i) => +candidates.some((c) => c.column === i),
  );
  // Candidate records, current value and own-activity monitor exchange messages.
  // Equal shared offsets preserve candidate ordering while the work gate adapts.
  const bridges = Array.from({ length: 7 }, (_, i) => [
    [0, 5, 1, i, 0.02],
    [1, i, 0, 5, 0.02 / 7],
    [1, i, 2, 3, 0.02 / 7],
    [2, 5, 1, i, -0.015],
  ]).flat();
  const joint = settleTogether(
    [value, choices, monitor],
    [
      value.drive,
      activity.map((v) =>
        Math.atanh(Math.max(-0.999999999, Math.min(0.999999999, v))),
      ),
      signal.drive,
    ],
    bridges,
  );
  const ordered = candidates
    .slice()
    .sort((a, b) => choices.state[b.column] - choices.state[a.column]);
  return {
    joint,
    column: ordered[0]?.column,
    review: {
      repair: signal.repair,
      ambiguity: signal.ambiguity,
      pressure,
      uncertainty: monitor.state[4],
      request_more: monitor.state[5] > 0.35 && pressure < 1,
    },
  };
}
export function reason(
  board,
  player,
  {
    depth = 6,
    maxNodes = 80000,
    monitoring = true,
    onProgress = () => {},
  } = {},
) {
  if (
    !Number.isInteger(depth) ||
    depth < 1 ||
    !Number.isInteger(maxNodes) ||
    maxNodes < 1 ||
    ![1, -1].includes(player)
  )
    throw Error("Positive integer depth/budget and player ±1 required");
  if (winner(board) || !legal(board).length)
    return {
      column: null,
      candidates: [],
      sequence: [],
      depth: 0,
      nodes: 0,
      review: { request_more: false },
      monitor: null,
      budgetExhausted: false,
    };
  const original = board.slice(),
    monitor = new Monitor(),
    cache = new Map();
  let nodes = 0,
    best = null,
    budgetExhausted = false;
  function search(b, p, left, alpha, beta) {
    if (nodes >= maxNodes) throw new Error("budget");
    nodes++;
    const win = winner(b);
    if (win) return { score: win * p * (100000 + left), sequence: [] };
    const moves = legal(b);
    if (!moves.length) return { score: 0, sequence: [] };
    if (!left) return { score: evaluate(b, p), sequence: [] };
    const startAlpha = alpha;
    const key = b.join(",") + ":" + p + ":" + left;
    const cached = cache.get(key);
    if (cached) return cached;
    let result = { score: -Infinity, sequence: [] },
      cut = false;
    for (const c of moves) {
      const child = search(drop(b, c, p), -p, left - 1, -beta, -alpha),
        score = -child.score;
      if (score > result.score)
        result = { score, sequence: [c, ...child.sequence] };
      alpha = Math.max(alpha, score);
      if (alpha >= beta) {
        cut = true;
        break;
      }
    }
    // Only cache fully explored exact values; cutoff bounds are not exact scores.
    if (!cut && result.score > startAlpha && result.score < beta)
      cache.set(key, result);
    return result;
  }
  for (let d = 1; d <= depth; d++) {
    try {
      const candidates = legal(board)
        .map((column) => {
          const r = search(
            drop(board, column, player),
            -player,
            d - 1,
            -Infinity,
            Infinity,
          );
          return { column, score: -r.score, sequence: [column, ...r.sequence] };
        })
        .sort((a, b) => b.score - a.score);
      if (!candidates.length) break;
      const decision = decisionCircuit(
          board,
          player,
          candidates,
          monitor,
          nodes / maxNodes,
        ),
        review = decision.review;
      best = {
        column: decision.column,
        candidates,
        sequence: candidates.find((c) => c.column === decision.column).sequence,
        depth: d,
        nodes,
        review,
        decision: decision.joint,
      };
      onProgress(best);
      if (candidates[0].score > 90000) break;
      // Monitoring has a causal job: ambiguous choices extend the normal four-ply budget.
      if (d >= 4 && (monitoring ? !review.request_more : true)) break;
    } catch (error) {
      if (error.message !== "budget") throw error;
      budgetExhausted = true;
      break;
    }
  }
  if (!best) {
    const column = legal(board)[0];
    best = {
      column,
      candidates: [],
      sequence: column === undefined ? [] : [column],
      depth: 0,
      nodes,
      review: { request_more: false },
    };
  }
  if (board.some((v, i) => v !== original[i]))
    throw Error("Imagination mutated live board");
  return { ...best, nodes, budgetExhausted };
}
export function brainSnapshot(board, player, thinking) {
  const joint = thinking?.decision ?? decisionCircuit(board, player, []).joint;
  return Object.assign(joint, {
    regionLabels: {
      features: "Threat features",
      value: "Value evaluator",
      futures: "Compared futures",
      self_input: "Own activity readback",
      self_state: "Self-monitor / budget",
    },
    adapters:
      "Branch scores ↔ current value ↔ commitment ↔ self-monitor · one shared settlement",
    memory:
      "The board, legal-move rules and isolated search branches supply the decision boundary. Value, candidate and monitor owners settle jointly; their state selects the action and extra-search request. No weight training or consciousness claim.",
  });
}
