import { Reasoner } from "./brain.js";

const reasoner = new Reasoner();
let timer = null, current = null;
let evaluations = [], evaluated = 0;

function pump(id) {
  if (id !== current) return;
  try {
    const previous = reasoner.result;
    const result = reasoner.tick(128);
    if (evaluations.length) {
      self.postMessage({ kind: "activity", id, evaluations, evaluated });
      evaluations = [];
    }
    if (result && (result !== previous || !reasoner.active))
      self.postMessage({ kind: reasoner.active ? "progress" : "done", id, result });
    if (reasoner.active) timer = setTimeout(() => pump(id), 0);
  } catch (error) {
    self.postMessage({ kind: "error", id, message: error.message });
  }
}

self.onmessage = ({ data }) => {
  clearTimeout(timer);
  current = data.id;
  reasoner.cancel();
  evaluations = [];
  evaluated = 0;
  if (data.reset) reasoner.reset();
  if (data.kind === "cancel") return;
  reasoner.start(data.board, data.player, {
    depth: data.depth ?? 6, maxNodes: 80000, monitoring: data.monitoring,
    onEvaluation: state => { evaluations.push(state); evaluated++; },
  });
  timer = setTimeout(() => pump(data.id), 0);
};
