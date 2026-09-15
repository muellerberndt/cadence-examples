// The arm's planner in JavaScript: a port of `compose` and `ModelPlanner` from
// arm/brain.py. Bounded beam search through the learned model's
// imagined consequences (Agent.predict_batch, read-only); every candidate torque pair is
// held for `rollout` decisions, and a leaf is scored by how well its imagined hand
// displacement tracks a clipped velocity field toward the target (objective "velocity") or
// by the distance after coasting (objective "coast"). The sort is stable, as in Python.

export const PREDICTED = ["dd_hand", "d_velocity", "d_angles"];
export const TORQUE_COUNT = 9;

const norm2 = (x, y) => Math.sqrt(x * x + y * y);

/** The imagined next observation from predicted deltas: kinematic bookkeeping only. */
export function compose(observation, deltas) {
  const angles = observation.angles;
  const theta0 = Math.atan2(angles[0], angles[1]) + deltas.d_angles[0], theta1 = Math.atan2(angles[2], angles[3]) + deltas.d_angles[1];
  const velocity = Float64Array.from(observation.velocity, (v, k) => Math.min(1.0, Math.max(-1.0, v + deltas.d_velocity[k] / 2.0)));
  const dHand = Float64Array.from(observation.d_hand, (d, k) => d + deltas.dd_hand[k]);
  return {
    angles: Float64Array.of(Math.sin(theta0), Math.cos(theta0), Math.sin(theta1), Math.cos(theta1)),
    velocity,
    hand: Float64Array.from(observation.hand, (h, k) => h + dHand[k]),
    d_hand: dHand,
    dd_hand: Float64Array.from(deltas.dd_hand),
    d_velocity: Float64Array.from(deltas.d_velocity),
    d_angles: Float64Array.from(deltas.d_angles),
    flags: new Float64Array(2),
  };
}

export class ModelPlanner {
  constructor({ depth = 1, beam = 4, coast = 3, rollout = 1, objective = "velocity", gain = 3.0, max_speed = 0.6, decision_seconds = 0.1 } = {}) {
    this.depth = depth | 0; this.beam = beam | 0; this.coast = coast; this.rollout = rollout | 0; this.objective = objective;
    this.gain = gain; this.maxSpeed = max_speed; this.decisionSeconds = decision_seconds;
  }

  leafScore(nxt, target) {
    if (this.objective === "coast") {
      const cx = nxt.hand[0] + this.coast * nxt.d_hand[0], cy = nxt.hand[1] + this.coast * nxt.d_hand[1];
      return -norm2(cx - target[0], cy - target[1]) - 0.5 * norm2(nxt.hand[0] - target[0], nxt.hand[1] - target[1]);
    }
    const ex = target[0] - nxt.hand[0], ey = target[1] - nxt.hand[1];
    const distance = norm2(ex, ey);
    const factor = Math.min(this.gain, this.maxSpeed / Math.max(distance, 1e-9)); // a clipped velocity field
    const dx = ex * factor, dy = ey * factor;
    return -norm2(nxt.d_hand[0] / this.decisionSeconds - dx, nxt.d_hand[1] / this.decisionSeconds - dy);
  }

  /** The first action of the best imagined sequence, its first-step prediction, and the search budget.
   *  `lastImagined` keeps the last level's candidates for display: each torque's imagined hand
   *  positions over the rollout (from the observed hand), its score and its first action, best first. */
  plan(model, observation, goal) {
    const target = Float64Array.from(goal, (g) => g * 2.0 - 1.0);
    let expansions = 0;
    let beam = [{ first: null, obs: observation, pred: null }];
    let imagined = [];
    for (let level = 0; level < this.depth; level++) {
      const observations = [], actions = [], entries = [];
      for (const entry of beam) for (let a = 0; a < TORQUE_COUNT; a++) { observations.push(entry.obs); actions.push(a); entries.push(entry); }
      const predictions = model.predictBatch(observations, actions, goal);
      expansions += actions.length;
      let composed = observations.map((obs, k) => compose(obs, predictions[k]));
      const hands = composed.map((obs, k) => [[observations[k].hand[0], observations[k].hand[1]], [obs.hand[0], obs.hand[1]]]);
      const firstPredictions = predictions.slice();
      for (let r = 0; r < this.rollout - 1; r++) { // hold each candidate torque: its effect accumulates
        const more = model.predictBatch(composed, actions, goal);
        expansions += actions.length;
        composed = composed.map((obs, k) => compose(obs, more[k]));
        composed.forEach((obs, k) => hands[k].push([obs.hand[0], obs.hand[1]]));
      }
      const candidates = entries.map((entry, k) => ({ score: this.leafScore(composed[k], target), first: entry.first === null ? actions[k] : entry.first, obs: composed[k], pred: entry.pred === null ? firstPredictions[k] : entry.pred, action: actions[k], hands: hands[k] }));
      candidates.sort((x, y) => y.score - x.score); // stable, descending
      imagined = candidates.map((c) => ({ action: c.action, first: c.first, score: c.score, hands: c.hands }));
      beam = candidates.slice(0, this.beam);
    }
    const best = beam[0];
    if (best.first === null || best.pred === null) throw Error("the search returned no action");
    this.lastImagined = { chosen: best.first | 0, candidates: imagined };
    return { action: best.first | 0, prediction: best.pred, budget: { expansions, depth: this.depth, beam: this.beam } };
  }

  /** The callable an ExperienceAgent with controller "planner" expects. */
  forAgent() {
    return (agent, row, moment) => {
      if (!moment.goal) throw Error("the arm's planner needs the goal marker on every moment");
      return this.plan(agent, moment.observation, moment.goal);
    };
  }
}
