// The brain thinks here, off the page's thread, and reports what it reads while it searches.
import { ValuePatch } from "./patch.js";
import { Brain, Position } from "./brain.js";

let brain = null, patch = null;
const TRACES = 140;                                    // reads shown per search, spread over the whole search

self.onmessage = async (event) => {
  const message = event.data;
  if (message.type === "load") {
    const state = await (await fetch(message.url)).json();
    patch = new ValuePatch(state);
    brain = new Brain(patch, message.search);
    self.postMessage({ type: "ready", sizes: state.sizes, search: message.search });
    return;
  }
  if (message.type === "think") {
    const position = new Position();
    for (const column of message.columns) position.play(column);
    // A traced read computes the record address as well, which costs far more than a plain
    // read of an empty store, so a sample of the reads is traced, evenly through the search.
    let seen = 0, every = 1, sent = 0;
    const started = performance.now();
    brain.onRead = null;
    const plain = patch.value.bind(patch);
    const u = brain._u;
    brain._read = (p) => {
      p.reading(u);
      seen++;
      if (seen % every === 0 && sent < TRACES) {
        const trace = {};
        const value = plain(u, trace);
        sent++;
        if (sent % 20 === 0) every *= 2;               // later reads are sampled more thinly
        self.postMessage({ type: "read", stones: p.plies, depth: p.plies - message.columns.length, value, slow: trace.slow, read: trace.read,
          reading: Uint8Array.from(u), context: Float32Array.from(trace.context), cells: trace.cells, activity: Float32Array.from(trace.activity) });
        return value;
      }
      return plain(u);
    };
    const thought = brain.think(position);
    self.postMessage({ type: "move", id: message.id, thought, ms: performance.now() - started });
    return;
  }
  if (message.type === "read") {                       // one traced read of a real position, for the resting brain
    const position = new Position();
    for (const column of message.columns) position.play(column);
    if (!position.plies) { self.postMessage({ type: "rest", empty: true }); return; }
    const trace = {}, u = position.reading(new Uint8Array(84));
    const value = patch.value(u, trace);
    self.postMessage({ type: "rest", stones: position.plies, value, slow: trace.slow, read: trace.read, reading: u,
      context: Float32Array.from(trace.context), cells: trace.cells, activity: Float32Array.from(trace.activity) });
  }
};
