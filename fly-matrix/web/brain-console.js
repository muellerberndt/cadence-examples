// A bounded text observer and command input. Execution belongs to the page's
// explicit dispatcher; this module cannot evaluate code or change the brain.
export const CONSOLE_COMMANDS = Object.freeze([
  "/help", "/status", "/pause", "/resume", "/gust", "/nudge", "/brain", "/paths",
  "/senses", "/eye", "/camera", "/speed", "/sugar", "/reset", "/about", "/close", "/clear",
]);
const finite = x => typeof x === "number" && Number.isFinite(x);
const number = (x, digits = 2) => finite(x) ? x.toFixed(digits) : "—";
const count = x => Number.isSafeInteger(x) && x >= 0 ? x.toLocaleString("en-US") : "—";
const scientific = x => finite(x) && x >= 0 ? x.toExponential(2) : "—";

/** Parse one slash command, without interpreting expressions or quoted code. */
export function parseConsoleCommand(raw) {
  if (typeof raw !== "string") throw new TypeError("Command must be text.");
  const text = raw.trim();
  if (!text) return null;
  if (text.length > 2048 || /[\r\n]/.test(text)) throw new RangeError("Enter one short command at a time.");
  const match = /^\/([a-z][a-z0-9-]*)(?:\s+(.*))?$/i.exec(text);
  if (!match) throw new Error("Use a slash command; /help lists the commands.");
  return { command: match[1].toLowerCase(), args: match[2] ? match[2].split(/\s+/) : [], raw: text };
}

/** Format only the measurements stateSnapshot makes available. Old samples in
 * stale/off telemetry are deliberately ignored; paused values are recorded. */
export function formatBrainStatus(state = {}, context = {}) {
  const neural = state.neural ?? {}, sample = neural.available === true ? neural.sample : null;
  const measured = sample && typeof sample === "object", recorded = neural.state === "paused";
  const status = measured ? `${recorded ? "RECORDED " : ""}${sample.converged ? "SETTLED" : "UNQUALIFIED"}`
    : String(neural.state ?? "waiting").toUpperCase();
  const lines = [`t=${number(state.time)}s · ${state.paused ? "PAUSED" : "RUNNING"}${context.mode ? ` · ${context.mode}` : ""}`
    + `${finite(context.speedScale) ? ` · ${number(context.speedScale)}× time` : ""} · hunger ${number(finite(state.hunger) ? state.hunger * 100 : null, 0)}%`];
  if (measured) {
    const n = sample.neuronCount ?? state.neurons;
    const active = sample.converged && Number.isSafeInteger(sample.active) && sample.active >= 0
      && Number.isSafeInteger(n) && n > 0 && sample.active <= n;
    lines.push(`Brain ${status} · activity ${active ? `${count(sample.active)} / ${count(n)} cells ≥0.5 (${number(100 * sample.active / n, 1)}%)` : "unavailable"}`
      + ` · ${count(sample.iterations)} settling steps`);
    lines.push(`Mismatch ${scientific(sample.initialResidual)} → ${scientific(sample.residual)} · tolerance ${scientific(sample.tolerance)}`
      + ` · input #${count(sample.requestId)} · age ${number(neural.ageSeconds)}s`
      + `${finite(context.solveMs) ? ` · solve ${number(context.solveMs, 0)}ms` : ""}`);
  } else lines.push(`Brain ${status} · activity, settling steps and mismatch unavailable${neural.reason ? ` · ${neural.reason}` : ""}`);
  const progress = context.progress;
  if (progress && !progress.completed) lines.push(`Computing input #${count(progress.requestId)} · ${progress.phase} · step ${count(progress.iteration)} · mismatch ${scientific(progress.residual)}${progress.withdrawn ? ' · withdrawn input' : ' · intermediate state'}`);
  lines.push(`Learning ${count(state.learningUpdates)} updates · ${count(state.changedConnections)} connections differ from initial weights`);
  lines.push(`Motion ${number(state.speed, 3)}m/s · altitude ${number(state.altitude, 3)}m · wings ${number(state.wingHz, 0)}Hz · odor context ${state.attention ?? "none"}`);
  if (context.hybrid) {
    lines.push("Brain: odor target, approach/avoid and local learning. Body controller: routes, flight and body routines.");
    const goal = context.goal;
    if (goal) lines.push(`Body execution ${goal.phase} · ${goal.fruit ?? "awaiting neural target"}`
      + `${Number.isSafeInteger(goal.decisionRequestId) ? ` · neural choice #${goal.decisionRequestId}` : ""}`
      + `${goal.clearanceLimited ? " · room clearance limits the supplied route" : ""}`);
    if (context.startup?.active) lines.push("Initial launch: body heading only · no selected fruit or learning credit");
    if (context.waitingSupport?.active) lines.push("Body continues cruising while the next brain goal settles · mechanical wall clearance · no new food selection");
    const target = context.targetSelection;
    if (Array.isArray(target?.p) && target.p.length === 2 && target.p.every(finite))
      lines.push(`Last target odds: banana ${number(100 * target.p[0], 1)}% · bread ${number(100 * target.p[1], 1)}% · selected ${target.fruit}`);
  }
  const navigation = context.navigation;
  if (navigation) {
    const c = navigation.command;
    lines.push(navigation.active && c ? `Neural request: forward ${number(c.forwardSpeed, 3)}m/s · yaw ${number(c.yawRate, 3)}rad/s · climb ${number(c.verticalSpeed, 3)}m/s`
      : "Neural request inactive · waiting for a fresh applied command");
  }
  const support = context.motionSupport;
  if (support?.enabled) {
    const source = support.source;
    lines.push(source ? `Course execution ${support.phase.replaceAll("_", " ")} · checked input #${count(source.requestId)} · age ${number(support.sourceAgeSeconds)}s`
      + ` · smoothed target ${number(Math.hypot(...support.velocityWorld), 3)}m/s · cruise ${number(support.config.cruiseSpeed, 2)}m/s`
      : `Course execution ${support.phase} · awaiting a checked direction`);
  }
  return lines;
}

/** The section supplies the log, form, input and collapse button by their IDs.
 * Entries are bounded, detached in snapshots, and inserted through textContent.
 * No automatic stream announcements: readers can inspect the log on demand. */
export function createBrainConsole(root, { execute, now = () => 0, commands = CONSOLE_COMMANDS, maxEntries = 160 } = {}) {
  if (!root?.querySelector || typeof execute !== "function" || typeof now !== "function") throw new TypeError("Invalid console configuration.");
  if (!Number.isSafeInteger(maxEntries) || maxEntries < 1 || maxEntries > 160) throw new RangeError("Console entry limit must be between 1 and 160.");
  if (!Array.isArray(commands) || !commands.every(c => typeof c === "string" && /^\/[a-z][a-z0-9-]*$/i.test(c))) throw new TypeError("Invalid command completions.");
  const choices = [...new Set(commands.map(c => c.toLowerCase()))];
  const log = root.querySelector("#console-log"), form = root.querySelector("#console-form"),
    input = root.querySelector("#console-input"), toggle = root.querySelector("#console-toggle");
  if (!log || !form || !input || !toggle) throw new TypeError("Console elements are missing.");
  const document = root.ownerDocument, entries = [], history = [];
  let nextId = 1, collapsed = root.classList.contains("collapsed"), following = true;
  let historyIndex = 0, draft = "", completion = null;
  let refreshHeight = () => {};
  log.setAttribute("aria-live", "off");
  input.maxLength = 2048;
  const atBottom = () => log.scrollHeight - log.clientHeight - log.scrollTop <= 8;

  function write(kind, text, meta = null) {
    const time = now(), label = /^[a-z][a-z0-9_-]*$/i.test(String(kind)) ? String(kind).toLowerCase() : "event";
    const entry = { id: nextId++, time: finite(time) && time >= 0 ? time : null,
      kind: label, text: String(text), meta: structuredClone(meta) };
    const oldTop = log.scrollTop;
    if (!collapsed) following = atBottom();
    const line = document.createElement("div"); line.className = "console-line"; line.dataset.kind = label;
    for (const [name, value] of [["time", entry.time === null ? "—" : entry.time.toFixed(2)], ["kind", label], ["text", entry.text]]) {
      const span = document.createElement("span"); span.className = `console-${name}`; span.textContent = value; line.appendChild(span);
    }
    log.appendChild(line); entries.push(entry);
    const beforePrune = log.scrollHeight;
    while (entries.length > maxEntries) { entries.shift(); log.firstChild?.remove(); }
    if (!collapsed) {
      if (following) log.scrollTop = log.scrollHeight;
      else log.scrollTop = Math.max(0, oldTop - (beforePrune - log.scrollHeight));
    }
    return structuredClone(entry);
  }

  function clear() {
    entries.length = 0; log.replaceChildren(); log.scrollTop = 0; following = true;
  }
  function focus() { input.focus({ preventScroll: true }); }
  function setCollapsed(value) {
    if (!collapsed) following = atBottom();
    collapsed = !!value; root.classList.toggle("collapsed", collapsed); log.hidden = collapsed;
    toggle.setAttribute("aria-expanded", String(!collapsed));
    toggle.textContent = collapsed ? "expand" : "collapse";
    toggle.title = collapsed ? "Expand log" : "Collapse log";
    toggle.setAttribute("aria-label", `${collapsed ? "Expand" : "Collapse"} brain activity log`);
    refreshHeight();
    if (!collapsed && following) log.scrollTop = log.scrollHeight;
  }

  // Height is a presentation preference. The top edge grows upward because
  // the console is anchored to the bottom; pointer capture also supports touch.
  const resizer = root.querySelector("#console-resize"), view = document.defaultView;
  if (resizer && view) {
    const style = document.documentElement.style, storageKey = "fly-matrix.console-height.v1";
    let preferred = null, drag = null, expandedHeight = root.getBoundingClientRect().height;
    try {
      const saved = Number(view.localStorage.getItem(storageKey));
      if (Number.isFinite(saved) && saved >= 160) preferred = saved;
    } catch { /* Resizing remains available when browser storage is disabled. */ }
    const limits = () => ({ min: 160, max: Math.max(160, Math.floor(Math.min(view.innerHeight * .7, view.innerHeight - 140))) });
    const remember = () => {
      try {
        if (preferred === null) view.localStorage.removeItem(storageKey);
        else view.localStorage.setItem(storageKey, String(preferred));
      } catch { /* Session-only preference. */ }
    };
    refreshHeight = () => {
      const { min, max } = limits(), clamp = h => Math.round(Math.max(min, Math.min(max, h)));
      if (preferred === null) style.removeProperty("--console-height");
      else style.setProperty("--console-height", `${clamp(preferred)}px`);
      if (!collapsed) {
        expandedHeight = clamp(root.getBoundingClientRect().height);
        style.setProperty("--console-height", `${expandedHeight}px`);
      }
      // Inspectors use the occupied height, including when the log is collapsed.
      style.setProperty("--console-occupied-height", `${root.getBoundingClientRect().height}px`);
      resizer.setAttribute("aria-valuemin", String(min));
      resizer.setAttribute("aria-valuemax", String(Math.floor(max)));
      resizer.setAttribute("aria-valuenow", String(expandedHeight));
      resizer.setAttribute("aria-valuetext", `${expandedHeight} pixels tall`);
      if (!collapsed && following) log.scrollTop = log.scrollHeight;
    };
    const resize = height => {
      following = atBottom();
      const { min, max } = limits();
      preferred = Math.round(Math.max(min, Math.min(max, height))); refreshHeight();
    };
    resizer.addEventListener("pointerdown", event => {
      if (event.button !== 0 || collapsed || drag) return;
      event.preventDefault(); event.stopPropagation();
      drag = { id: event.pointerId, y: event.clientY, height: expandedHeight };
      resizer.setPointerCapture(event.pointerId); resizer.focus({ preventScroll: true });
      root.classList.toggle("resizing", true);
    });
    resizer.addEventListener("pointermove", event => {
      if (drag?.id === event.pointerId) resize(drag.height + drag.y - event.clientY);
    });
    const finishResize = event => {
      if (drag?.id !== event.pointerId) return;
      drag = null; root.classList.toggle("resizing", false); remember();
    };
    for (const type of ["pointerup", "pointercancel", "lostpointercapture"]) resizer.addEventListener(type, finishResize);
    resizer.addEventListener("keydown", event => {
      const { min, max } = limits(), step = event.shiftKey ? 60 : 20;
      const heights = { ArrowUp: expandedHeight + step, ArrowDown: expandedHeight - step, Home: min, End: max };
      if (!Object.hasOwn(heights, event.key)) return;
      event.preventDefault(); resize(heights[event.key]); remember();
    });
    resizer.addEventListener("dblclick", () => { preferred = null; remember(); refreshHeight(); });
    view.addEventListener("resize", refreshHeight);
  }
  function complete(reverse) {
    const value = input.value;
    if (!completion || completion.last !== value) {
      const prefix = value.trimStart().toLowerCase();
      const matches = choices.filter(c => c.startsWith(prefix));
      if (!matches.length) return;
      completion = { matches, index: reverse ? matches.length : -1, last: null };
    }
    const c = completion;
    c.index = (c.index + (reverse ? -1 : 1) + c.matches.length) % c.matches.length;
    input.value = c.last = `${c.matches[c.index]} `;
    input.setSelectionRange?.(input.value.length, input.value.length);
  }
  input.addEventListener("input", () => { completion = null; });
  input.addEventListener("keydown", event => {
    if (event.isComposing) return;
    if (event.key === "Escape") { event.preventDefault(); input.blur(); }
    else if (event.key === "Tab") { event.preventDefault(); complete(event.shiftKey); }
    else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
      event.preventDefault(); completion = null;
      if (!history.length) return;
      if (historyIndex === history.length) draft = input.value;
      historyIndex = Math.max(0, Math.min(history.length, historyIndex + (event.key === "ArrowUp" ? -1 : 1)));
      input.value = historyIndex === history.length ? draft : history[historyIndex];
      input.setSelectionRange?.(input.value.length, input.value.length);
    }
  });
  // Also stop keyup/keypress and keys on console buttons, so typing g/w/i/e or
  // space cannot reach the simulation's window shortcuts. Default form submit
  // is kept; Enter is not intercepted by the input's key handler.
  for (const type of ["keydown", "keyup", "keypress"]) root.addEventListener(type, event => event.stopPropagation());
  log.addEventListener("scroll", () => { if (!collapsed) following = atBottom(); });
  toggle.addEventListener("click", () => setCollapsed(!collapsed));
  form.addEventListener("submit", async event => {
    event.preventDefault(); event.stopPropagation();
    const raw = input.value;
    if (!raw.trim()) return;
    write("command", raw);
    if (history.at(-1) !== raw) history.push(raw);
    if (history.length > 100) history.shift();
    historyIndex = history.length; draft = ""; completion = null; input.value = "";
    try {
      const result = await execute(raw);
      if (Array.isArray(result)) for (const line of result) write("reply", line);
      else if (result !== undefined && result !== null) write("reply", result);
    } catch (error) { write("error", error?.message ?? String(error)); }
  });
  setCollapsed(collapsed);
  return Object.freeze({ write, clear, focus, setCollapsed,
    snapshot: () => ({ collapsed, entries: structuredClone(entries) }) });
}
