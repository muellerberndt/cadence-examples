// The Connect Four brain in the browser: the rules, and the search of connect4/brain.py
// move for move (the same column order, table, budget of read positions and tie-break), so a
// searched position gives the same column, value, depth and number of reads as in Python.

export const WIDTH = 7, HEIGHT = 6, CELLS = 42;
export const ORDER = [3, 2, 4, 1, 5, 0, 6];
export const PROVEN = 2.0;
const EXACT = 0, LOWER = 1, UPPER = 2;
const DIRECTIONS = [[0, 1], [1, 0], [1, 1], [1, -1]];

// A position: cells[row * 7 + column] is 0 empty, 1 the first player's stone, 2 the second's.
export class Position {
  constructor() {
    this.cells = new Int8Array(CELLS);
    this.height = new Int8Array(WIDTH);
    this.plies = 0;
    this.first = 0; this.second = 0;                   // each side's stones as an exact integer below 2^49
    this.firstM = 0; this.secondM = 0;                 // the same boards with their columns reversed
    this.columns = [];
  }
  get side() { return 1 + (this.plies & 1); }          // whose stone lands next
  playable(column) { return this.height[column] < HEIGHT; }
  legal() { const out = []; for (const c of ORDER) if (this.height[c] < HEIGHT) out.push(c); return out; }
  get full() { return this.plies === CELLS; }

  // whether `side` has a line through the cell (row, column), that cell counted as its own
  _line(row, column, side) {
    const cells = this.cells;
    for (const [dr, dc] of DIRECTIONS) {
      let run = 1;
      for (let s = -1; s <= 1; s += 2) {
        let r = row + s * dr, c = column + s * dc;
        while (r >= 0 && r < HEIGHT && c >= 0 && c < WIDTH && cells[r * WIDTH + c] === side) { run++; r += s * dr; c += s * dc; }
      }
      if (run >= 4) return true;
    }
    return false;
  }
  wins(column) { return this._line(this.height[column], column, this.side); }
  play(column) {
    const row = this.height[column], side = this.side;
    this.cells[row * WIDTH + column] = side;
    this.height[column]++; this.plies++;
    if (side === 1) { this.first += BIT[column * 7 + row]; this.firstM += BIT[(6 - column) * 7 + row]; }
    else { this.second += BIT[column * 7 + row]; this.secondM += BIT[(6 - column) * 7 + row]; }
    this.columns.push(column);
    this.lost = this._line(row, column, side);         // the side that just moved completed a line
  }
  undo() {
    const column = this.columns.pop(), row = --this.height[column];
    this.plies--;
    const side = this.side;
    this.cells[row * WIDTH + column] = 0;
    if (side === 1) { this.first -= BIT[column * 7 + row]; this.firstM -= BIT[(6 - column) * 7 + row]; }
    else { this.second -= BIT[column * 7 + row]; this.secondM -= BIT[(6 - column) * 7 + row]; }
    this.lost = false;
  }
  // The position exactly, as Pascal Pons keys it: the stones of the side to move plus every
  // stone. The sum is below 2^50, so a double holds it without rounding, and two positions
  // never share a key. (A hashed key can collide and hand the search another position's entry.)
  get key() { return (this.side === 1 ? this.first : this.second) + this.first + this.second; }

  // What the value patch reads: the side that just moved first, then the other side, each 42
  // cells row-major from the bottom-left. A position and its mirror image are one reading: of
  // the two pairs of boards (own, other), the smaller, compared as connect4/game.py does.
  _mirrored() {
    const firstMoved = (this.plies & 1) === 1;
    const own = firstMoved ? this.first : this.second, other = firstMoved ? this.second : this.first;
    const ownM = firstMoved ? this.firstM : this.secondM, otherM = firstMoved ? this.secondM : this.firstM;
    return ownM < own || (ownM === own && otherM < other);
  }
  get readingKey() {
    const firstMoved = (this.plies & 1) === 1, flip = this._mirrored();
    const a = flip ? this.firstM : this.first, b = flip ? this.secondM : this.second;
    return firstMoved ? a + "|" + b : b + "|" + a;
  }
  reading(out) {
    const own = 2 - (this.plies & 1), flip = this._mirrored();
    out.fill(0);
    for (let r = 0; r < HEIGHT; r++) for (let c = 0; c < WIDTH; c++) {
      const v = this.cells[r * WIDTH + (flip ? WIDTH - 1 - c : c)];
      if (v) out[(v === own ? 0 : CELLS) + r * WIDTH + c] = 1;
    }
    return out;
  }
}

const BIT = new Float64Array(49);
for (let k = 0; k < 49; k++) BIT[k] = 2 ** k;
Position.prototype.lost = false;

export const proven = (value) => Math.abs(value) > PROVEN - 1.0;
class Spent extends Error {}

export class Brain {
  constructor(patch, { reads = 2000, lateStones = 18, lateReads = 200000, visits = 2000000 } = {}) {
    this.patch = patch; this.readsLimit = reads; this.lateStones = lateStones; this.lateReads = lateReads; this.visitsLimit = visits; this.visits = 0;
    this.table = new Map(); this.values = new Map();
    this.reads = 0; this.limit = 0;
    this._u = new Uint8Array(2 * CELLS);
    this.onRead = null;                                // the page's hook: called with each read's trace
  }

  _read(position) {
    position.reading(this._u);
    if (this.onRead) { const trace = {}; const v = this.patch.value(this._u, trace); this.onRead(position, v, trace); return v; }
    return this.patch.value(this._u);
  }

  _frontier(position, ply) {
    const legal = position.legal(), values = new Float64Array(legal.length), unread = [];
    for (let k = 0; k < legal.length; k++) {
      position.play(legal[k]);
      if (position.full) values[k] = 0.0;
      else if (position.legal().some((c) => position.wins(c))) values[k] = -(PROVEN - (ply + 2) / 100);
      else unread.push(k);
      position.undo();
    }
    if (unread.length) {
      // A position is read once in a search and its value kept, as in connect4/brain.py.
      const keys = [], fresh = new Set();
      for (const k of unread) {
        position.play(legal[k]);
        const key = position.readingKey;
        keys.push(key);
        if (!this.values.has(key)) fresh.add(key);
        position.undo();
      }
      if (this.reads + fresh.size > this.limit) throw new Spent();
      this.reads += fresh.size;
      for (let i = 0; i < unread.length; i++) {
        if (!this.values.has(keys[i])) {
          position.play(legal[unread[i]]);
          this.values.set(keys[i], Math.min(0.999, Math.max(-0.999, this._read(position))));
          position.undo();
        }
        values[unread[i]] = this.values.get(keys[i]);
      }
    }
    let best = 0;
    for (let k = 1; k < legal.length; k++) if (values[k] > values[best]) best = k;
    return [values[best], legal[best]];
  }

  _negamax(position, depth, alpha, beta, ply) {
    if (++this.visits > this.visitsLimit) throw new Spent();
    const legal = position.legal();
    for (const column of legal) if (position.wins(column)) return PROVEN - (ply + 1) / 100;
    if (position.plies === 41) return 0.0;
    // What the other side threatens to complete at once decides the move: two such columns
    // cannot both be blocked, and one must be.
    let forced = -1, threats = 0;
    position.plies++;                                    // the other side's view of the same board
    for (const column of legal) if (position.wins(column)) { threats++; if (forced < 0) forced = column; }
    position.plies--;
    if (threats > 1) return -(PROVEN - (ply + 2) / 100);
    const key = position.key, entry = this.table.get(key);
    let first = -1;
    if (entry !== undefined) {
      first = entry[3];
      if (entry[0] >= depth || proven(entry[2])) {
        if (entry[1] === EXACT) return entry[2];
        if (entry[1] === LOWER) alpha = Math.max(alpha, entry[2]); else beta = Math.min(beta, entry[2]);
        if (alpha >= beta) return entry[2];
      }
    }
    if (forced >= 0) {
      position.play(forced);
      const value = -this._negamax(position, Math.max(depth - 1, 1), -beta, -alpha, ply + 1);
      position.undo();
      this.table.set(key, [depth, value <= alpha ? UPPER : value >= beta ? LOWER : EXACT, value, forced]);
      return value;
    }
    if (depth <= 1) {
      const [value, column] = this._frontier(position, ply);
      this.table.set(key, [1, EXACT, value, column]);
      return value;
    }
    const order = legal.slice();
    const at = order.indexOf(first);
    if (at > 0) { order.splice(at, 1); order.unshift(first); }
    let best = -Infinity, bestColumn = order[0];
    const floor = alpha;
    for (const column of order) {
      position.play(column);
      const value = -this._negamax(position, depth - 1, -beta, -alpha, ply + 1);
      position.undo();
      if (value > best) { best = value; bestColumn = column; }
      alpha = Math.max(alpha, value);
      if (alpha >= beta) break;
    }
    this.table.set(key, [depth, best <= floor ? UPPER : best >= beta ? LOWER : EXACT, best, bestColumn]);
    return best;
  }

  // Search the position for the side to move. Reads privately: the patch does not change.
  think(position) {
    const legal = position.legal();
    if (!legal.length) throw new Error("the board is full");
    this.table = new Map(); this.values = new Map(); this.reads = 0; this.visits = 0;
    this.limit = position.plies >= this.lateStones ? this.lateReads : this.readsLimit;
    const deepest = CELLS - position.plies, rootPlies = position.plies;
    let thought = { column: legal[0], value: 0.0, depth: 0, reads: 0, proven: false, columns: {}, line: [] };
    for (let depth = 1; depth <= deepest; depth++) {
      const columns = {};
      const order = legal.includes(thought.column) ? [thought.column, ...legal.filter((c) => c !== thought.column)] : legal.slice();
      try {
        let alpha = -Infinity;
        for (const column of order) {
          let value;
          if (position.wins(column)) value = PROVEN - 1 / 100;
          else if (position.plies === 41) value = 0.0;
          else {
            position.play(column);
            value = depth === 1 ? -this._negamax(position, 1, -Infinity, Infinity, 1) : -this._negamax(position, depth - 1, -Infinity, -alpha, 1);
            position.undo();
          }
          columns[column] = value;
          alpha = Math.max(alpha, value - 1e-9);
        }
      } catch (error) {
        // The search plays and takes back stones on one board; a deepening that runs out of
        // budget leaves its stones on it, so the board is taken back to where the search began.
        while (position.plies > rootPlies) position.undo();
        if (error instanceof Spent) break;
        throw error;
      }
      let top = -Infinity;
      for (const c of order) top = Math.max(top, columns[c]);
      const ties = order.filter((c) => columns[c] >= top - 1e-9);   // in the order searched, as in Python
      // the column held from the shallower search when it is among the best, else the nearest the centre
      let choice = thought.column;
      if (!ties.includes(choice)) { choice = ties[0]; for (const c of ties) if (Math.abs(c - 3) < Math.abs(choice - 3)) choice = c; }
      thought = { column: choice, value: top, depth, reads: this.reads, proven: proven(top), columns, line: this._line(position, choice) };
      if (proven(top) || depth >= deepest) break;
    }
    thought.reads = this.reads; thought.visits = this.visits;
    return thought;
  }

  _line(position, column, length = 8) {
    const line = [column];
    position.play(column);
    let played = 1;
    while (line.length < length && !(position.lost || position.full)) {
      const entry = this.table.get(position.key);
      if (entry === undefined || !position.playable(entry[3])) break;
      line.push(entry[3]); position.play(entry[3]); played++;
    }
    while (played--) position.undo();
    return line;
  }
}
