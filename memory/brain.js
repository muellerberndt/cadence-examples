import { FastMemory } from "../showcase/engine.js";
import { MemoryReadout, settleTogether } from "../showcase/nervous_system.js";

export class MemoryBrain {
  constructor() {
    this.records = new FastMemory();
    this.readout = new MemoryReadout();
  }
  get w() {
    return this.records.w;
  }
  observe(key, value) {
    this.records.observe(key, value);
  }
  predict(key) {
    this.last = settleTogether(
      [this.readout],
      [this.readout.configure(this.records, key)],
      [],
      {
        metadata: {
          regionLabels: { key: "Cue input", record: "Value memory" },
          adapters: "Cue → associative seams → recall · one shared settlement",
          memory:
            "Observed lessons change FastSeams between settlements. The current cue and recall settle together with fixed weights; the published recall is tanh of the linear memory value, preserving argmax.",
          behavior: { label: "Recalling", tone: "seeking" },
        },
      },
    );
    return this.last.state
      .slice(8)
      .map((v) => Math.atanh(Math.max(-0.999999999, Math.min(0.999999999, v))));
  }
  brain(key) {
    this.predict(key);
    return this.last;
  }
}
