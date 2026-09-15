// The Connect Four rules and the world that emits S00 moments, ported from connect_four/env.py.
// The page and the parity harness both run games through this file: the board is the same
// engine the Python world uses (gravity, alternating turns, lines in four directions, draws),
// and the moments carry the same fields in the same order, so a moment built here enters the
// engine's step exactly as the moment stream.jsonl carries does.
//
// Cells are a flat array, index = row * cols + col, row 0 at the bottom; 0 empty, 1 the first
// player's stone, 2 the second's. Players are 0 and 1. The candidate is the brain; the
// observation is relative to it (0 empty, 1 its own stone, 2 the other side's).

export const CELL = ["empty", "self", "opponent"];
export const MOVER = ["candidate", "opponent"];
export const TURN = ["candidate", "opponent"];
export const DIRECTIONS = [[0, 1], [1, 0], [1, 1], [1, -1]];

export function gameConfig(config = {}) {
  const rows = config.rows ?? 6, cols = config.cols ?? 7, connect = config.connect ?? 4;
  return { rows, cols, connect, cells: rows * cols };
}

export function fieldNames(config) {
  const out = [];
  for (let i = 0; i < config.cells; i++) out.push(`c${String(i).padStart(2, "0")}`);
  return out;
}

const EYE3 = [Float64Array.from([1, 0, 0]), Float64Array.from([0, 1, 0]), Float64Array.from([0, 0, 1])];
const EYE2 = [Float64Array.from([1, 0]), Float64Array.from([0, 1])];

/** The moment's observation of `cells` (relative to the candidate): one categorical field per cell, the mover and the phase. */
export function observationOf(cells, mover, config) {
  const out = {};
  for (let i = 0; i < config.cells; i++) out[`c${String(i).padStart(2, "0")}`] = EYE3[cells[i]];
  out.mover = EYE2[mover];
  let own = 0;
  for (let i = 0; i < config.cells; i++) if (cells[i] === 1) { own = 1; break; }
  out.phase = EYE2[own];
  return out;
}

/** The candidate-relative cells (0 empty, 1 self, 2 opponent) of an observation. */
export function cellsOf(observation, config) {
  const out = new Int8Array(config.cells);
  for (let i = 0; i < config.cells; i++) {
    const field = observation[`c${String(i).padStart(2, "0")}`];
    let arg = 0;
    for (let k = 1; k < field.length; k++) if (field[k] > field[arg]) arg = k;
    out[i] = arg;
  }
  return out;
}

export function goalVector(toMove, candidateFirst) {
  return Float64Array.from([toMove === 0 ? 1 : 0, toMove === 1 ? 1 : 0, candidateFirst ? 1 : 0]);
}

/** The rules (env.Board). */
export class Board {
  constructor(config) {
    this.config = config;
    this.cells = new Int8Array(config.cells);
    this.heights = new Int32Array(config.cols);
    this.toMove = 0;
    this.winner = null;
    this.moves = [];
  }

  copy() {
    const out = new Board(this.config);
    out.cells = Int8Array.from(this.cells);
    out.heights = Int32Array.from(this.heights);
    out.toMove = this.toMove;
    out.winner = this.winner;
    out.moves = [...this.moves];
    return out;
  }

  get full() { for (const h of this.heights) if (h < this.config.rows) return false; return true; }
  get terminal() { return this.winner !== null || this.full; }

  isLegal(col) { return !this.terminal && col >= 0 && col < this.config.cols && this.heights[col] < this.config.rows; }

  legalColumns() {
    if (this.terminal) return [];
    const out = [];
    for (let c = 0; c < this.config.cols; c++) if (this.heights[c] < this.config.rows) out.push(c);
    return out;
  }

  legalMask() {
    const mask = new Uint8Array(this.config.cols);
    for (const c of this.legalColumns()) mask[c] = 1;
    return mask;
  }

  play(col) {
    if (!this.isLegal(col)) throw Error(`column ${col} is not legal`);
    const row = this.heights[col];
    this.cells[row * this.config.cols + col] = this.toMove + 1;
    this.heights[col] = row + 1;
    if (this.connects(row, col)) this.winner = this.toMove;
    this.moves.push(col);
    this.toMove = 1 - this.toMove;
  }

  /** Whether the stone at (row, col) lies on a line of `connect` equal stones. */
  connects(row, col) {
    const { rows, cols, connect } = this.config, stone = this.cells[row * cols + col];
    if (!stone) return false;
    for (const [dr, dc] of DIRECTIONS) {
      let count = 1;
      for (const sign of [1, -1]) {
        let r = row + sign * dr, c = col + sign * dc;
        while (r >= 0 && r < rows && c >= 0 && c < cols && this.cells[r * cols + c] === stone) { count += 1; r += sign * dr; c += sign * dc; }
      }
      if (count >= connect) return true;
    }
    return false;
  }

  /** The cells of the line the last stone completed, for the page's winning highlight. */
  lineThrough(row, col) {
    const { rows, cols, connect } = this.config, stone = this.cells[row * cols + col];
    if (!stone) return null;
    for (const [dr, dc] of DIRECTIONS) {
      const line = [row * cols + col];
      for (const sign of [1, -1]) {
        let r = row + sign * dr, c = col + sign * dc;
        while (r >= 0 && r < rows && c >= 0 && c < cols && this.cells[r * cols + c] === stone) { line.push(r * cols + c); r += sign * dr; c += sign * dc; }
      }
      if (line.length >= connect) return line.sort((a, b) => a - b);
    }
    return null;
  }

  result(player) {
    if (!this.terminal) throw Error("the game is not over");
    if (this.winner === null) return 0;
    return this.winner === player ? 1 : -1;
  }

  /** The cells as `player` sees them: 0 empty, 1 own stone, 2 the other's. */
  relative(player) {
    const own = player + 1, out = new Int8Array(this.config.cells);
    for (let i = 0; i < this.config.cells; i++) out[i] = this.cells[i] === 0 ? 0 : (this.cells[i] === own ? 1 : 2);
    return out;
  }
}

/** One life's games, one at a time (env.World): the candidate acts through `act`, the other side through `reply`. */
export class World {
  constructor(config, { lifeId = "web", event = 0, episode = -1 } = {}) {
    this.config = config;
    this.lifeId = lifeId;
    this._board = new Board(config);
    this._candidate = 0;
    this._episode = episode;
    this._event = event;
    this._tick = 0;
    this._awaitingReply = false;
    this._closed = true;
    this._pending = null;
    this._executed = null;
  }

  get board() { return this._board; }
  get candidate() { return this._candidate; }
  get closed() { return this._closed; }
  get awaitingReply() { return this._awaitingReply; }
  get episode() { return this._episode; }
  get event() { return this._event; }

  /** 'candidate', 'opponent', 'draw', or null while the game continues. */
  winner() {
    if (!this._board.terminal) return null;
    if (this._board.winner === null) return "draw";
    return this._board.winner === this._candidate ? "candidate" : "opponent";
  }

  outcome() { return this._board.terminal ? this._board.result(this._candidate) : null; }

  /** A new game; `first` names who opens. An opponent opening comes from `opening`. */
  reset({ first = "candidate", opening = null } = {}) {
    if (first !== "candidate" && first !== "opponent") throw Error("first must be 'candidate' or 'opponent'");
    this._board = new Board(this.config);
    this._candidate = first === "candidate" ? 0 : 1;
    this._episode += 1;
    this._tick = 0;
    this._awaitingReply = false;
    this._closed = false;
    this._pending = this._executed = null;
    if (first === "opponent") {
      if (opening === null) throw Error("an opponent opening needs its column");
      this._board.play(opening);
    }
    return this._decisionMoment();
  }

  /** Execute the candidate's committed decision; the intermediate moment carries its feedback and offers no action. */
  act(decisionId, column) {
    if (this._closed) throw Error("the game is over");
    if (this._awaitingReply) throw Error("the opponent has not replied");
    if (this._board.toMove !== this._candidate) throw Error("it is not the candidate's move");
    if (!this._board.isLegal(column)) throw Error(`column ${column} is not legal`);
    this._pending = decisionId | 0;
    this._executed = column | 0;
    this._board.play(column | 0);
    const terminal = this._board.terminal;
    const reward = terminal ? this._board.result(this._candidate) : 0.0;
    const m = this._moment({ feedback: true, reward, terminated: terminal, truncated: false, mask: new Uint8Array(this.config.cols) });
    this._pending = this._executed = null;
    this._awaitingReply = !terminal;
    this._closed = terminal;
    return m;
  }

  /** The other side's move; the next decision moment, or the terminal moment of the game. */
  reply(column) {
    if (this._closed) throw Error("the game is over");
    if (!this._awaitingReply) throw Error("it is the candidate's move");
    if (!this._board.isLegal(column)) throw Error(`the opponent's column ${column} is not legal`);
    this._board.play(column | 0);
    this._awaitingReply = false;
    return this._decisionMoment();
  }

  /** Close an unfinished episode without an outcome. */
  truncate() {
    if (this._closed) throw Error("the episode is already closed");
    this._closed = true;
    this._awaitingReply = false;
    return this._moment({ feedback: false, reward: 0.0, terminated: false, truncated: true, mask: new Uint8Array(this.config.cols) });
  }

  _decisionMoment() {
    const terminal = this._board.terminal;
    const reward = terminal ? this._board.result(this._candidate) : 0.0;
    this._closed = terminal;
    const mask = terminal ? new Uint8Array(this.config.cols) : this._board.legalMask();
    return this._moment({ feedback: false, reward, terminated: terminal, truncated: false, mask });
  }

  _moment({ feedback, reward, terminated, truncated, mask }) {
    const mover = this._board.toMove === this._candidate ? MOVER.indexOf("opponent") : MOVER.indexOf("candidate");
    const obs = observationOf(this._board.relative(this._candidate), mover, this.config);
    const toMove = this._board.toMove === this._candidate ? TURN.indexOf("candidate") : TURN.indexOf("opponent");
    const m = {
      life_id: this.lifeId, episode_id: this._episode, event_id: this._event, tick: this._tick,
      observation: obs, observed: {}, action_mask: mask,
      feedback_for: feedback ? this._pending : null, executed: feedback ? this._executed : null,
      reward, reward_known: feedback, terminated, truncated,
      final_observation: truncated ? obs : null, goal: goalVector(toMove, this._candidate === 0),
    };
    this._event += 1;
    this._tick += 1;
    return m;
  }
}
