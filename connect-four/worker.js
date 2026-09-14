import { reason } from "./brain.js";
self.onmessage = ({ data }) => {
  try {
    const result = reason(data.board, data.player, {
      depth: data.depth ?? 6,
      maxNodes: 80000,
      monitoring: data.monitoring,
      onProgress: (r) =>
        self.postMessage({ kind: "progress", id: data.id, result: r }),
    });
    self.postMessage({ kind: "done", id: data.id, result });
  } catch (error) {
    self.postMessage({ kind: "error", id: data.id, message: error.message });
  }
};
