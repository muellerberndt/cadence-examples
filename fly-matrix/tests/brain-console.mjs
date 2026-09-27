// A small DOM fixture exercises the observer/input independently of the brain.
// Real-browser tests separately check the page dispatcher and visual layout.
// Run: node tests/brain-console.mjs
import assert from "node:assert/strict";
import { CONSOLE_COMMANDS, parseConsoleCommand, formatBrainStatus, createBrainConsole } from "../web/brain-console.js";

class Element {
  constructor(tag, document) {
    Object.assign(this, { tag, ownerDocument: document, children: [], parent: null, listeners: new Map(),
      attrs: new Map(), dataset: {}, hidden: false, value: "", _text: "", _top: 0, height: 0 });
    const names = new Set();
    this.classList = { contains: n => names.has(n), toggle(n, force = !names.has(n)) { if (force) names.add(n); else names.delete(n); } };
  }
  set innerHTML(_) { throw Error("HTML insertion is forbidden"); }
  set textContent(value) { this._text = String(value); this.children = []; }
  get textContent() { return this._text + this.children.map(c => c.textContent).join(""); }
  get firstChild() { return this.children[0] ?? null; }
  get clientHeight() { return this.hidden ? 0 : this.height; }
  get scrollHeight() { return this.hidden ? 0 : this.children.length * 20; }
  get scrollTop() { return this._top; }
  set scrollTop(value) { this._top = Math.max(0, Math.min(value, this.scrollHeight - this.clientHeight)); }
  appendChild(child) { child.parent = this; this.children.push(child); return child; }
  remove() { this.parent.children.splice(this.parent.children.indexOf(this), 1); this.parent = null; }
  replaceChildren() { for (const c of this.children) c.parent = null; this.children = []; }
  setAttribute(name, value) { this.attrs.set(name, String(value)); }
  getAttribute(name) { return this.attrs.get(name); }
  querySelector(selector) { return this.children.find(c => c.id === selector.slice(1)) ?? this.children.map(c => c.querySelector(selector)).find(Boolean) ?? null; }
  focus() { this.ownerDocument.activeElement = this; }
  blur() { if (this.ownerDocument.activeElement === this) this.ownerDocument.activeElement = null; }
  setSelectionRange(start, end) { this.selectionStart = start; this.selectionEnd = end; }
  addEventListener(name, fn) { if (!this.listeners.has(name)) this.listeners.set(name, []); this.listeners.get(name).push(fn); }
  dispatch(type, data = {}) {
    const event = { type, ...data, stopped: false, defaultPrevented: false,
      stopPropagation() { this.stopped = true; }, preventDefault() { this.defaultPrevented = true; } };
    const promises = [];
    for (let node = this; node; node = node.parent) {
      for (const fn of node.listeners.get(type) ?? []) { const p = fn(event); if (p?.then) promises.push(p); }
      if (event.stopped) break;
    }
    return { event, done: Promise.all(promises) };
  }
}
function fixture(options = {}) {
  const document = { createElement(tag) { return new Element(tag, document); }, activeElement: null };
  const outer = document.createElement("body"), root = outer.appendChild(document.createElement("section"));
  const log = root.appendChild(document.createElement("div")); log.id = "console-log"; log.height = 60;
  const form = root.appendChild(document.createElement("form")); form.id = "console-form";
  const input = form.appendChild(document.createElement("input")); input.id = "console-input";
  const toggle = root.appendChild(document.createElement("button")); toggle.id = "console-toggle";
  const calls = [], api = createBrainConsole(root, { execute: raw => { calls.push(raw); }, now: () => 12.5, ...options });
  const submit = raw => { input.value = raw; return form.dispatch("submit").done; };
  return { api, root, outer, log, form, input, toggle, calls, submit, document };
}
let passed = 0, failed = 0;
async function test(name, run) {
  try { await run(); passed++; console.log("ok " + name); }
  catch (error) { failed++; console.error("FAIL " + name + "\n" + error.stack); }
}

await test("parser accepts one slash command without interpreting expressions", () => {
  assert.equal(parseConsoleCommand("  "), null);
  assert.deepEqual(parseConsoleCommand("  /BRAIN settling  "), { command: "brain", args: ["settling"], raw: "/BRAIN settling" });
  assert.deepEqual(parseConsoleCommand("/status <script>alert(1)</script>").args, ["<script>alert(1)</script>"]);
  for (const raw of ["alert(1)", "/status;alert(1)", "/status\n/reset", "/"]) assert.throws(() => parseConsoleCommand(raw));
  assert.throws(() => parseConsoleCommand(null), TypeError);
  assert.throws(() => parseConsoleCommand("/help " + "x".repeat(2048)), RangeError);
  assert.equal(parseConsoleCommand("/unknown").command, "unknown", "page dispatcher owns command authorization");
  assert.ok(Object.isFrozen(CONSOLE_COMMANDS));
});

await test("status uses qualified measurements and marks paused/unqualified records", () => {
  const state = { time: 4.5, paused: false, hunger: .7, neurons: 100, learningUpdates: 0, changedConnections: 0,
    speed: .03, altitude: 1.2, wingHz: 200, attention: "banana", neural: { state: "live", available: true, ageSeconds: .2,
      sample: { converged: true, active: 25, neuronCount: 100, iterations: 89, initialResidual: 2, residual: 3e-8, tolerance: 1e-6, requestId: 7 } } };
  const text = formatBrainStatus(state, { mode: "flying", speedScale: .5,
    navigation: { active: true, command: { forwardSpeed: -.05, yawRate: .1, verticalSpeed: 0 } } }).join("\n");
  for (const part of ["hunger 70%", "25 / 100 cells ≥0.5 (25.0%)", "89 settling steps", "2.00e+0 → 3.00e-8",
    "tolerance 1.00e-6", "input #7", "age 0.20s", "Learning 0 updates", "0.50× time", "forward -0.050m/s"]) assert.ok(text.includes(part), part);
  state.neural.state = "paused"; state.paused = true;
  assert.ok(formatBrainStatus(state).join("\n").includes("RECORDED SETTLED"));
  state.neural.sample.converged = false;
  assert.ok(formatBrainStatus(state).join("\n").includes("UNQUALIFIED · activity unavailable"));
});

await test("stale/off samples cannot masquerade as current values or fabricated zeros", () => {
  for (const mode of ["stale", "off", "waiting"]) {
    const text = formatBrainStatus({ neural: { state: mode, available: false,
      sample: { converged: true, active: 31415, iterations: 927, residual: 8.765e-7 } }, speed: NaN, hunger: Infinity }).join("\n");
    assert.ok(text.includes("Brain " + mode.toUpperCase()));
    assert.ok(text.includes("activity, settling steps and mismatch unavailable"));
    assert.ok(!/31415|927|8.77e-7|NaN|Infinity/.test(text));
    assert.ok(text.includes("Learning — updates"));
  }
});

await test("continued course execution cannot turn expired neural telemetry into a fresh solve", () => {
  const text = formatBrainStatus({ neural: { state: "stale", available: false,
    sample: { converged: true, active: 99, iterations: 81, residual: 1e-8 } } }, {
    navigation: { active: false }, motionSupport: { enabled: true, phase: "held",
      source: { requestId: 32 }, sourceAgeSeconds: 3.4, velocityWorld: [.3, 0, 0], config: { cruiseSpeed: .3 } },
  }).join("\n");
  assert.ok(text.includes("Brain STALE"));
  assert.ok(!text.includes("SETTLED") && !text.includes("81 settling"));
  assert.ok(text.includes("Course execution held") && text.includes("checked input #32"));
  assert.ok(text.includes("age 3.40s") && text.includes("smoothed target 0.300m/s"));
});

await test("log uses safe text, simulation time and detached metadata/snapshots", () => {
  const f = fixture(), metadata = { request: { id: 8 } }, payload = "<img src=x onerror='globalThis.attacked=true'>";
  const returned = f.api.write("settle", payload, metadata);
  metadata.request.id = 9; returned.meta.request.id = 10;
  const snapshot = f.api.snapshot(); snapshot.entries[0].text = "changed"; snapshot.entries[0].meta.request.id = 11;
  assert.equal(f.api.snapshot().entries[0].text, payload); assert.equal(f.api.snapshot().entries[0].meta.request.id, 8);
  assert.equal(f.log.firstChild.children[2].textContent, payload);
  assert.equal(f.log.firstChild.children[0].textContent, "12.50");
  assert.equal(f.log.firstChild.dataset.kind, "settle");
  assert.equal(f.log.getAttribute("aria-live"), "off");
});

await test("bounded pruning keeps a scrolled reader's visible row instead of forcing the tail", () => {
  const f = fixture({ maxEntries: 5 });
  for (let i = 0; i < 5; i++) f.api.write("event", String(i));
  assert.equal(f.log.scrollTop, 40);
  f.log.scrollTop = 20; f.log.dispatch("scroll"); f.api.write("event", "5");
  assert.equal(f.log.scrollTop, 0, "one old row was pruned above the reader");
  assert.deepEqual(f.api.snapshot().entries.map(e => e.text), ["1", "2", "3", "4", "5"]);
  assert.equal(f.log.children.length, 5);
  assert.throws(() => fixture({ maxEntries: 161 }), RangeError);
});

await test("raw command echo, async replies and errors remain text-only", async () => {
  let resolve, called;
  const f = fixture({ execute: raw => { called = raw; return new Promise(r => { resolve = r; }); } });
  const completion = f.submit("  /status  ");
  assert.equal(called, "  /status  ");
  assert.deepEqual(f.api.snapshot().entries.map(e => [e.kind, e.text]), [["command", "  /status  "]]);
  assert.equal(f.input.value, "");
  resolve(["first", "<script>not code</script>"]); await completion;
  assert.deepEqual(f.api.snapshot().entries.map(e => e.kind), ["command", "reply", "reply"]);
  const bad = fixture({ execute: async () => { throw Error("unknown /oops <img>"); } });
  await bad.submit("/oops"); assert.equal(bad.api.snapshot().entries.at(-1).kind, "error");
  assert.equal(bad.api.snapshot().entries.at(-1).text, "unknown /oops <img>");
  await bad.submit("   "); assert.equal(bad.api.snapshot().entries.length, 2);
});

await test("history restores draft, deduplicates adjacent input and retains at most 100 commands", async () => {
  const f = fixture(); await f.submit("/status"); await f.submit("/help"); await f.submit("/help");
  f.input.value = "/bra";
  f.input.dispatch("keydown", { key: "ArrowUp" }); assert.equal(f.input.value, "/help");
  f.input.dispatch("keydown", { key: "ArrowUp" }); assert.equal(f.input.value, "/status");
  f.input.dispatch("keydown", { key: "ArrowDown" }); assert.equal(f.input.value, "/help");
  f.input.dispatch("keydown", { key: "ArrowDown" }); assert.equal(f.input.value, "/bra");
  for (let i = 0; i < 105; i++) await f.submit("/status " + i);
  for (let i = 0; i < 110; i++) f.input.dispatch("keydown", { key: "ArrowUp" });
  assert.equal(f.input.value, "/status 5");
});

await test("Tab cycles known names, Escape blurs and typing cannot reach simulation shortcuts", () => {
  const f = fixture(); let shortcuts = 0;
  for (const type of ["keydown", "keyup", "keypress"]) f.outer.addEventListener(type, () => shortcuts++);
  f.input.value = "/s";
  f.input.dispatch("keydown", { key: "Tab" }); assert.equal(f.input.value, "/status ");
  f.input.dispatch("keydown", { key: "Tab" }); assert.equal(f.input.value, "/senses ");
  f.input.dispatch("keydown", { key: "Tab", shiftKey: true }); assert.equal(f.input.value, "/status ");
  f.input.value = "/n"; f.input.dispatch("input"); f.input.dispatch("keydown", { key: "Tab" }); assert.equal(f.input.value, "/nudge ");
  f.input.value = "/unknown"; f.input.dispatch("input"); f.input.dispatch("keydown", { key: "Tab" }); assert.equal(f.input.value, "/unknown");
  for (const type of ["keydown", "keyup", "keypress"]) for (const key of ["g", "w", "i", "e", " "]) f.input.dispatch(type, { key });
  assert.equal(shortcuts, 0); f.api.focus(); assert.equal(f.document.activeElement, f.input);
  f.input.dispatch("keydown", { key: "Escape" }); assert.equal(f.document.activeElement, null);
  assert.equal(f.input.dispatch("keydown", { key: "Enter" }).event.defaultPrevented, false);
});

await test("collapse preserves command input and clear preserves command history", async () => {
  const f = fixture(); await f.submit("/status"); f.toggle.dispatch("click");
  assert.equal(f.api.snapshot().collapsed, true); assert.equal(f.log.hidden, true);
  assert.equal(f.form.hidden, false); assert.equal(f.input.hidden, false);
  assert.equal(f.toggle.getAttribute("aria-expanded"), "false");
  await f.submit("/help"); f.api.write("learn", "one checked update");
  f.api.setCollapsed(false); assert.equal(f.log.hidden, false);
  f.api.clear(); assert.deepEqual(f.api.snapshot().entries, []); assert.equal(f.log.children.length, 0);
  f.input.dispatch("keydown", { key: "ArrowUp" }); assert.equal(f.input.value, "/help");
});

console.log(passed + " brain console tests passed; " + failed + " failed");
if (failed) process.exitCode = 1;
