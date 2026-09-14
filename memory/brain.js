import { SynapticMemory } from "../shared/engine.js";
import { MemoryReadout, settleTogether } from "../shared/nervous_system.js";

export class MemoryBrain {
  constructor() {
    this.records = new SynapticMemory();
    this.readout = new MemoryReadout();
  }
  get w() {
    return this.records.w;
  }
  observe(key, value, options = {}) {
    this.records.observe(key, value, 1, options);
  }
  predict(key) {
    this.last = settleTogether(
      [this.readout],
      [this.readout.configure(this.records, key)],
      [],
      {
        metadata: {
          regionLabels: { key: "Cue input", record: "Value memory" },
          adapters: "Cue → associative synapses → recall · one shared equilibrium",
          memory:
            "Observed lessons change transient and persistent synaptic weights in the ongoing interaction loop. Repeated or salient lessons strengthen the persistent component. The current cue and recall settle together with fixed weights; the published recall is tanh of the linear memory value, preserving argmax.",
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
