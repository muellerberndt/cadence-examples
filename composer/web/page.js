// The S06 composer page: a recorded life, replayed. For each of the seven public-domain sample
// pieces the stage's bundle carries three renderings of the emulated SID (the demonstration, the
// composer's imitation of it and its continuation of it), and, per block of the episode that made
// each, the settled activation of every neuron and what the records cortex did with that block's
// reading. This page plays the audio and replays the rest in step with it: the level lanes fill
// block by block, the whole brain runs through the standard BrainScan renderer, and the records
// cortex runs through the standard records view. Nothing here is computed from a brain running in
// the browser, because the instrument is a cycle-exact emulator with no port in a browser yet;
// when one exists this page plays live, as the arm, the world, the artist and Connect Four
// already do. No activity is invented: every neuron value, every lit cell and every flash comes
// out of the bundle, and the lanes are the level of the file the page is playing.

import { BrainScan, decodeArray, roleOf } from "../../web/brain_scan.js";
import { RecordsView } from "../../web/records_view.js";

const $ = (id) => document.getElementById(id);
const PAGE_OPTIONS = { autoplay: false, ...(window.__COMPOSER_PAGE__ || {}) };
const PAGE_FORMAT = "cadence-composer-page/1";
const FRAMES_FORMAT = "s06-public-frames/1";
const TRACKS = ["demonstration", "imitation", "continuation"];
const TRACK_LABEL = { demonstration: "DEMONSTRATION", imitation: "IMITATION", continuation: "CONTINUATION" };
const TRACK_NOTE = {
  demonstration: "what the chip was given, and what the brain heard",
  imitation: "what the composer played back at it",
  continuation: "what the composer played on, having heard the first half",
};
const PHASE_LABEL = { listen: "LISTENING", gap: "GAP", attempt: "PLAYING", instruct: "READING THE REQUEST" };
const VOICE_COLOR = [[154, 210, 132], [184, 199, 111], [112, 164, 178]]; // light green, yellow, cyan
const DEMO_COLOR = [112, 164, 178]; // cyan: what the chip was given
const BRAIN_COLOR = [154, 210, 132]; // light green: what the composer played
const PLAYHEAD = [255, 255, 255];
const PANE_INK = [184, 174, 240];
const GATE_SECTIONS = ["instrument", "imitation", "continuation", "controls", "latency", "instruction", "theme recall", "revision"];
const NUMBER_LABELS = {
  pitch_within_a_semitone: ["PITCH WITHIN A SEMITONE", "share of the blocks a voice sounds, on this piece"],
  onset_f1: ["ONSET F1", "attacks paired inside 40 ms"],
  imitation_reward: ["IMITATION REWARD", "per block, zero is the demonstration"],
  continuation_surprise_nats: ["CONTINUATION SURPRISE", "nats per note event under the corpus model"],
  continuation_pitch_within_a_semitone: ["CONTINUATION PITCH", "share, against what the piece does there"],
  continuation_onset_f1: ["CONTINUATION ONSET F1", "attacks paired inside 40 ms"],
  median_pitch_error_cents: ["MEDIAN PITCH ERROR", "cents, over the paired notes"],
};
const shade = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(`cannot load ${url}: ${response.status}`);
  return response.json();
}

/** The index the build inlines, or the one beside the page in development. */
async function loadPayload() {
  if (window.__COMPOSER__) {
    const payload = window.__COMPOSER__;
    if (!payload.atlas) payload.atlas = await fetchJSON(payload.base + payload.manifest.brain.atlas);
    return payload;
  }
  const named = String(document.body.dataset.bundle || "./index.json");
  const base = named.replace(/[^/]*$/, "");
  const manifest = await fetchJSON(named);
  return { manifest, base, atlas: await fetchJSON(base + manifest.brain.atlas) };
}

/** A plain array, or a {dtype, shape, b64} array of the kind the atlas exporter writes. */
function values(spec) {
  if (spec == null) return null;
  if (Array.isArray(spec)) return Float32Array.from(spec);
  if (ArrayBuffer.isView(spec)) return spec;
  return decodeArray(spec);
}

function indices(spec) {
  if (spec == null) return null;
  if (Array.isArray(spec)) return Int32Array.from(spec);
  if (ArrayBuffer.isView(spec)) return spec;
  return Int32Array.from(decodeArray(spec));
}

function main(payload) {
  const manifest = payload.manifest;
  if (manifest.format !== PAGE_FORMAT) throw Error(`the page index is ${manifest.format}, not ${PAGE_FORMAT}`);
  const clock = { sample_rate: 22050, block_ms: 20, bar_blocks: 96, sixteenth_blocks: 6, phrase_blocks: 192, ...(manifest.clock || {}) };
  const blockSeconds = clock.block_ms / 1000;
  const brain = manifest.brain || {};
  const readingIndex = Int32Array.from(brain.reading || []);
  const scan = new BrainScan($("scan"), payload.atlas, { labels: $("labels"), strip: $("strip"), labelTop: 10, montageRows: 5 });
  const recordsSpec = brain.records || null;
  const records = recordsSpec && recordsSpec.cells ? new RecordsView($("records"), { granules: recordsSpec.cells, active: recordsSpec.active, inputs: recordsSpec.inputs, groups: recordsSpec.groups || [], rate: recordsSpec.rate, rewardRate: recordsSpec.reward_rate }) : null;
  if (!records) $("recordsPanel").hidden = true;

  // -- the state of the replay
  let piece = null, loaded = null, track = "demonstration";
  let episodes = {}; // the frames files of this piece, by file name
  let replay = {};   // per track: the decoded rows, the records and the phase
  let shownFrames = 0, shownRecords = 0, shownBlock = -1;
  let playing = false, clockSource = "none", internalStart = 0, internalPaused = 0;
  let loadToken = 0;
  const audio = Object.fromEntries(TRACKS.map((name) => {
    const element = new Audio();
    element.preload = "auto";
    element.id = `audio${name[0].toUpperCase()}${name.slice(1)}`;
    element.dataset.track = name;
    // an element already at its end can fire `ended` the moment it is played again; the replay
    // is stopped only once it has actually run
    element.addEventListener("ended", () => { if (track === name && playing && elapsed() > 0.25) finish(); });
    document.body.appendChild(element);
    return [name, element];
  }));

  /** One track's frames and records, cut out of its episode's file by the index's range. */
  function cut(file, range) {
    const body = episodes[file];
    if (!body || !range) return null;
    const steps = body.frames.steps | 0, n = body.frames.n | 0;
    const raw = body.raw || (body.raw = decodeArray(body.frames.activation));
    const start = clamp(range.start | 0, 0, steps), end = clamp(range.end | 0, start, steps);
    const rows = [], blocks = [], entries = [];
    for (let t = start; t < end; t++) {
      const row = new Float32Array(n);
      for (let i = 0; i < n; i++) row[i] = (raw[t * n + i] / 255) * 2 - 1;
      rows.push(row);
      const block = body.blocks[t] || {};
      blocks.push(block);
      const r = block.records;
      entries.push(r ? {
        index: indices(r.index), values: values(r.values),
        valuedIndex: indices(r.valuedIndex), valuedValues: values(r.valuedValues),
        imagined: r.imagined | 0, writes: r.writes | 0, phraseWrites: r.phrase_store_writes | 0,
        readingRow: row,
      } : null);
    }
    return { steps: rows.length, n, rows, entries, blocks, phase: range.phase || "", first: start, last: end - 1 };
  }

  // ---------------------------------------------------------------- the pieces
  const select = $("piece");
  manifest.pieces.forEach((p) => {
    const option = document.createElement("option");
    option.value = p.id;
    option.textContent = p.title || p.name || p.id;
    select.appendChild(option);
  });
  select.addEventListener("change", () => { void choose(select.value); });

  async function choose(id) {
    const chosen = manifest.pieces.find((p) => p.id === id) || manifest.pieces[0];
    const token = ++loadToken;
    stop(false);
    piece = chosen;
    loaded = null;
    select.value = chosen.id;
    setState("LOADING " + (chosen.title || chosen.id).toUpperCase());
    const files = [...new Set(Object.values(chosen.frames || {}).map((range) => range.file))];
    const bodies = await Promise.all(files.map((file) => fetchJSON(payload.base + file)));
    if (token !== loadToken) return;
    episodes = {};
    files.forEach((file, k) => {
      if (bodies[k].format !== FRAMES_FORMAT) throw Error(`${file} is ${bodies[k].format}, not ${FRAMES_FORMAT}`);
      episodes[file] = bodies[k];
    });
    replay = {};
    for (const name of TRACKS) {
      const range = (chosen.frames || {})[name];
      replay[name] = range ? cut(range.file, range) : null;
    }
    for (const name of TRACKS) {
      const source = (chosen.audio || {})[name];
      audio[name].src = source ? payload.base + source : "";
      const button = $(`play${name[0].toUpperCase()}${name.slice(1)}`);
      button.disabled = !source;
      button.title = source ? `${TRACK_LABEL[name]} · ${source}` : "not in this bundle";
    }
    track = (chosen.audio || {}).demonstration ? "demonstration" : TRACKS.find((t) => (chosen.audio || {})[t]) || "demonstration";
    resetReplay();
    fillNumbers();
    fillLedger();
    loaded = chosen.id;
    setState("READY");
    $("hint").hidden = false;
    drawRoll();
    return chosen;
  }

  // ---------------------------------------------------------------- the transport
  function currentAudio() { return audio[track]; }

  /** The time the replay follows: the audio's own clock while it is sounding, and the page's
   *  clock otherwise, so a browser that will not play the file still shows the recorded life. */
  function elapsed() {
    const element = currentAudio();
    const internal = playing ? (performance.now() - internalStart) / 1000 : internalPaused;
    if (element && element.src && !element.paused && element.currentTime > 0) return element.currentTime;
    return playing || internalPaused ? internal : 0;
  }

  function source() {
    const element = currentAudio();
    if (!playing) return "none";
    return element && element.src && !element.paused && element.currentTime > 0 ? "audio" : "internal";
  }

  /** How many blocks a track runs for: its audio's length, and the frames it carries. */
  function trackLength(name) {
    const blocks = ((piece && piece.track_blocks) || {})[name] || 0;
    const frames = replay[name] ? replay[name].steps : 0;
    return Math.max(1, blocks, frames);
  }

  function play(name) {
    if (!piece || !loaded) return false;
    if (name && name !== track) { stop(false); track = name; }
    const element = currentAudio();
    resetReplay();
    playing = true;
    internalStart = performance.now();
    internalPaused = 0;
    $("hint").hidden = true;
    if (element && element.src) {
      try { element.currentTime = 0; } catch (error) { /* a file that is not seekable yet */ }
      if (element.currentTime > 0.01) element.load(); // the seek did not take: it would resume at its end
      const started = element.play();
      if (started && started.catch) started.catch(() => { /* the page's own clock carries the replay */ });
    }
    markButtons();
    setState(`PLAYING ${TRACK_LABEL[track]}`, "live");
    return true;
  }

  function stop(reset = true) {
    playing = false;
    for (const element of Object.values(audio)) { if (element.src) { element.pause(); if (reset) element.currentTime = 0; } }
    if (reset) { internalPaused = 0; resetReplay(); $("hint").hidden = false; }
    markButtons();
    setState("READY");
  }

  /** The track has played to its end: the lanes, the brain and the records hold what they
   *  reached instead of snapping back to the start. */
  function finish() {
    internalPaused = trackLength(track) * blockSeconds;
    stop(false);
    setState(`${TRACK_LABEL[track]} PLAYED`);
  }

  function markButtons() {
    for (const name of TRACKS) {
      const button = $(`play${name[0].toUpperCase()}${name.slice(1)}`);
      button.setAttribute("aria-pressed", String(playing && track === name));
    }
  }

  for (const name of TRACKS) $(`play${name[0].toUpperCase()}${name.slice(1)}`).addEventListener("click", () => { void play(name); });
  $("stop").addEventListener("click", () => stop(true));

  // ---------------------------------------------------------------- the replay
  function resetReplay() {
    shownFrames = 0; shownRecords = 0; shownBlock = -1;
    const f = replay[track];
    if (f && f.steps) { scan.set(f.rows[0]); scan.reset(); shownFrames = 1; }
    if (records) {
      records.level.fill(0); records.valued.fill(0); records.imagined.fill(0); records.glow.fill(0);
      records.counts = { codes: 0, imagined: 0, writes: 0, cells: 0 };
      records.lastWrite = null; records.activeIndex = new Int32Array(0);
    }
    fillRecordsLedger();
  }

  function feedRecord(entry) {
    if (!records || !entry) return;
    if (entry.readingRow && readingIndex.length) {
      const reading = new Float32Array(readingIndex.length);
      for (let k = 0; k < readingIndex.length; k++) reading[k] = entry.readingRow[readingIndex[k]];
      records.setReading(reading);
    }
    if (entry.index && entry.values) records.code({ index: entry.index, values: entry.values, valuedIndex: entry.valuedIndex, valuedValues: entry.valuedValues }, "decide");
    if (entry.imagined) records.counts.imagined += entry.imagined;
    // the head writes the outcome into exactly the cells of that block's code; the bundle counts
    // the writes, so the flash is drawn on those cells and never on any other
    if (entry.writes && entry.index && entry.values) {
      records.write({ index: entry.index, values: entry.values, written: entry.writes,
                      fields: [{ name: "consequence", rate: recordsSpec.rate, reward: false }] });
    }
  }

  /** Advance the frames and the records to the block the audio has reached, counted inside the
   *  track: block 0 is the track's first block. */
  function advance(local) {
    const f = replay[track];
    if (f && f.steps) {
      const target = clamp(local + 1, 0, f.steps);
      if (target < shownFrames) { // a seek backwards: show the frame and measure change from it
        shownFrames = Math.max(1, target);
        scan.set(f.rows[shownFrames - 1]);
        scan.reset();
      } else if (target - shownFrames > 48) { // a long jump: do not run hundreds of steps in one frame
        shownFrames = target;
        scan.set(f.rows[target - 1]);
        scan.reset();
      } else {
        let budget = 8; // at most eight settling steps a frame, so a slow machine stays in time
        while (shownFrames < target && budget-- > 0) scan.step(f.rows[shownFrames++], { draw: false });
      }
      if (f.entries) {
        let fed = 0;
        while (shownRecords < shownFrames && fed++ < 24) feedRecord(f.entries[shownRecords++]);
        shownRecords = Math.max(shownRecords, Math.min(shownFrames, f.entries.length)); // a jump: count, do not flash
      }
    }
    shownBlock = local;
  }

  function frame(now) {
    requestAnimationFrame(frame);
    if (!piece) return;
    const seconds = elapsed();
    clockSource = source();
    const length = trackLength(track);
    const local = clamp(Math.floor(seconds / blockSeconds), 0, length - 1);
    if (playing && clockSource === "internal" && seconds >= length * blockSeconds) finish();
    if (local !== shownBlock || shownFrames === 0) advance(local);
    scan.draw(now);
    if (records) records.draw();
    drawRoll(local, seconds);
    $("clock").textContent = `${timecode(seconds)} · BLOCK ${local + 1} OF ${length}`;
    fillRecordsLedger();
    const f = replay[track];
    $("phase").textContent = f && f.phase ? `${PHASE_LABEL[f.phase] || f.phase.toUpperCase()} · RECORDED` : "RECORDED LIFE";
  }

  function timecode(seconds) {
    const s = Math.max(0, Math.floor(seconds)), m = Math.floor(s / 60);
    return `${String(m).padStart(2, "0")}:${String(s % 60).padStart(2, "0")}`;
  }

  // ---------------------------------------------------------------- the lanes
  const roll = $("roll");
  const rollContext = roll.getContext("2d");

  function laneOf(name) {
    const level = ((piece && piece.level) || {})[name];
    return level && level.length ? level : null;
  }

  function drawRoll(block = -1, seconds = 0) {
    const rect = roll.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.round(rect.width * dpr), h = Math.round(rect.height * dpr);
    if (roll.width !== w || roll.height !== h) { roll.width = w; roll.height = h; }
    const ctx = rollContext;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    ctx.fillStyle = "#2a1f61";
    ctx.fillRect(0, 0, rect.width, rect.height);
    if (!piece) return;
    const second = track === "demonstration" ? "imitation" : track;
    const panes = [
      { name: "demonstration", color: DEMO_COLOR, label: TRACK_LABEL.demonstration, note: TRACK_NOTE.demonstration },
      { name: second, color: BRAIN_COLOR, label: TRACK_LABEL[second], note: TRACK_NOTE[second] },
    ];
    const blocks = Math.max(1, ...panes.map((pane) => (laneOf(pane.name) || []).length), trackLength(track));
    const gutter = Math.min(42, Math.max(22, rect.width * 0.11)), top = 3, gap = 9, labelH = 15, footerH = 12;
    const paneH = (rect.height - top - 2 * labelH - gap - footerH) / 2;
    const x0 = gutter, x1 = rect.width - 8;
    const x = (b) => x0 + (b / blocks) * (x1 - x0);
    panes[0].y = top + labelH;
    panes[1].y = top + 2 * labelH + paneH + gap;
    for (const pane of panes) {
      const lane = laneOf(pane.name);
      ctx.textAlign = "left";
      ctx.font = "600 10px ui-monospace, Menlo, monospace";
      ctx.fillStyle = shade(pane.name === track ? PLAYHEAD : PANE_INK, pane.name === track ? 0.85 : 0.8);
      ctx.fillText(pane.label, 4, pane.y - 4);
      ctx.font = "9px ui-monospace, Menlo, monospace";
      ctx.fillStyle = shade(PANE_INK, 0.45);
      ctx.fillText(lane ? pane.note : "not in this bundle", 8 + ctx.measureText(pane.label).width * 1.16, pane.y - 4);
      // the grid: a line every sixteenth, a brighter one every bar, and the level scale
      ctx.strokeStyle = shade(PANE_INK, 0.09);
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (let b = 0; b <= blocks; b += clock.sixteenth_blocks) { const px = Math.round(x(b)) + 0.5; ctx.moveTo(px, pane.y); ctx.lineTo(px, pane.y + paneH); }
      ctx.stroke();
      ctx.strokeStyle = shade(PANE_INK, 0.28);
      ctx.beginPath();
      for (let b = 0; b <= blocks; b += clock.bar_blocks) { const px = Math.round(x(b)) + 0.5; ctx.moveTo(px, pane.y); ctx.lineTo(px, pane.y + paneH); }
      ctx.stroke();
      ctx.textAlign = "right";
      ctx.font = "9px ui-monospace, Menlo, monospace";
      for (const db of [0, -20, -40]) {
        const py = pane.y + paneH * (-db / 60);
        ctx.strokeStyle = shade(PANE_INK, 0.12);
        ctx.beginPath(); ctx.moveTo(x0, Math.round(py) + 0.5); ctx.lineTo(x1, Math.round(py) + 0.5); ctx.stroke();
        ctx.fillStyle = shade(PANE_INK, 0.45);
        ctx.fillText(`${db} dB`, gutter - 5, py + 7);
      }
      ctx.strokeStyle = shade(PANE_INK, 0.2);
      ctx.strokeRect(Math.round(x0) + 0.5, Math.round(pane.y) + 0.5, Math.round(x1 - x0), Math.round(paneH));
      if (!lane) continue;
      const bw = Math.max(1, (x1 - x0) / blocks);
      for (let b = 0; b < lane.length; b++) {
        const level = lane[b] / 255;
        if (level <= 0) continue;
        const bx = x(b), height = Math.max(1, level * (paneH - 2));
        ctx.fillStyle = shade(pane.color, b <= block ? 0.95 : 0.26);
        ctx.fillRect(bx, pane.y + paneH - height - 1, Math.max(1, bw - 0.4), height);
      }
    }
    // the playhead
    const px = Math.round(x(block + 1)) + 0.5;
    ctx.strokeStyle = shade(PLAYHEAD, playing ? 0.9 : 0.35);
    ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px, panes[0].y); ctx.lineTo(px, rect.height - footerH); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillStyle = shade(PANE_INK, 0.5);
    ctx.font = "9px ui-monospace, Menlo, monospace";
    const scale = `${blocks} BLOCKS · ${(blocks * clock.block_ms / 1000).toFixed(2)} S`;
    ctx.fillText(rect.width < 470 ? scale : `${scale} · ONE BAR ${clock.bar_blocks} BLOCKS · ONE SIXTEENTH ${clock.sixteenth_blocks} · LEVEL PER BLOCK`, x1, rect.height - 3);
  }

  // ---------------------------------------------------------------- the readouts
  function setState(text, kind = "") {
    const element = $("state");
    element.textContent = text;
    element.className = `state ${kind}`;
  }

  function fillNumbers() {
    const box = $("numbers");
    box.textContent = "";
    const numbers = (piece && piece.numbers) || {};
    const keys = Object.keys(NUMBER_LABELS).filter((k) => k in numbers).concat(Object.keys(numbers).filter((k) => !(k in NUMBER_LABELS)));
    if (!keys.length) { box.innerHTML = `<div class="number"><span>numbers</span><b>none</b><em>the bundle carries no number for this piece</em></div>`; return; }
    for (const key of keys.slice(0, 4)) {
      const [label, note] = NUMBER_LABELS[key] || [key.replace(/_/g, " "), ""];
      const value = numbers[key];
      const shown = typeof value === "number" ? (Math.abs(value) >= 100 ? value.toFixed(0) : value.toFixed(3)) : String(value);
      const cell = document.createElement("div");
      cell.className = "number";
      cell.innerHTML = `<span></span><b></b><em></em>`;
      cell.children[0].textContent = label;
      cell.children[1].textContent = shown;
      cell.children[2].textContent = note;
      box.appendChild(cell);
    }
  }

  function fillLedger() {
    const f = replay[track];
    const rows = [
      ["piece", piece.name || piece.id],
      ["split", piece.split || "none"],
      ["blocks", `${piece.blocks || 0}`],
      ["licence", piece.licence || "none"],
      ["frames", `${f ? f.steps : 0}`],
      ["phase", f && f.phase ? f.phase : "none"],
    ];
    $("ledger").innerHTML = rows.map(() => `<span></span><b></b>`).join("");
    const children = $("ledger").children;
    rows.forEach((row, k) => { children[2 * k].textContent = row[0]; children[2 * k + 1].textContent = row[1]; children[2 * k + 1].title = row[1]; });
  }

  function fillRecordsLedger() {
    if (!records) { $("recordsLedger").textContent = ""; return; }
    const c = records.counts;
    const rows = [["cells", `${records.granules}`], ["active", `${c.cells || (recordsSpec ? recordsSpec.active : 0)}`], ["codes", `${c.codes}`], ["imagined", `${c.imagined}`], ["writes", `${c.writes}`]];
    const box = $("recordsLedger");
    if (box.children.length !== 2 * rows.length) box.innerHTML = rows.map(() => `<span></span><b></b>`).join("");
    rows.forEach((row, k) => { box.children[2 * k].textContent = row[0]; box.children[2 * k + 1].textContent = row[1]; });
  }

  function fillGates() {
    // a row without a section joins the controls; it heads no section of its own
    const rows = (manifest.gates || []).map((gate) => ({ ...gate, section: gate.section || "controls" }));
    const render = (table, list) => {
      const body = table.querySelector("tbody");
      body.textContent = "";
      let section = null;
      for (const gate of list) {
        if (gate.section !== section) {
          section = gate.section;
          const head = document.createElement("tr");
          head.className = "section";
          head.innerHTML = `<td colspan="4"></td>`;
          head.children[0].textContent = section;
          body.appendChild(head);
        }
        const tr = document.createElement("tr");
        tr.innerHTML = `<td></td><td class="value"></td><td class="want"></td><td class="mark"></td>`;
        tr.children[0].textContent = gate.name;
        if (gate.note) { const em = document.createElement("em"); em.textContent = gate.note; tr.children[0].appendChild(em); }
        tr.children[1].textContent = gate.value === null || gate.value === undefined ? "not measured" : typeof gate.value === "number" ? formatGate(gate.value) : String(gate.value);
        const measured = gate.measured || gate.threshold === null || gate.threshold === undefined;
        tr.children[2].textContent = measured ? "no threshold declared" : `${gate.rule || "at least"} ${formatGate(gate.threshold)}`;
        const mark = gate.open ? "open" : measured ? "open" : gate.passed ? "pass" : "fail";
        tr.children[3].textContent = gate.open ? (gate.passed ? "OPEN · MET" : "OPEN") : measured ? "MEASURED" : gate.passed ? "PASS" : "NOT MET";
        tr.children[3].className = `mark ${mark}`;
        body.appendChild(tr);
      }
      if (!list.length) body.innerHTML = `<tr><td colspan="4">the run states no gate of this kind</td></tr>`;
    };
    const order = (a, b) => GATE_SECTIONS.indexOf(a.section) - GATE_SECTIONS.indexOf(b.section);
    render($("gatedTable"), rows.filter((g) => !g.open).sort(order));
    render($("openTable"), rows.filter((g) => g.open).sort(order));
  }

  function formatGate(value) {
    if (!Number.isFinite(value)) return String(value);
    if (Number.isInteger(value) && Math.abs(value) >= 1000) return value.toLocaleString("en-US");
    if (Number.isInteger(value)) return String(value);
    return Math.abs(value) >= 100 ? value.toFixed(1) : value.toFixed(3);
  }

  function fillHeader() {
    const run = manifest.run || {};
    $("counts").textContent = [
      `${manifest.pieces.length} PIECES · ${(brain.neurons || payload.atlas.n).toLocaleString("en-US")} NEURONS · ${(brain.synapses || payload.atlas.synapses).toLocaleString("en-US")} SYNAPSES`,
      `${clock.block_ms} MS BLOCK · ${clock.bar_blocks} BLOCKS A BAR · ${clock.sample_rate.toLocaleString("en-US")} HZ`,
      run.run_id ? `RUN ${run.run_id}` : "",
    ].filter(Boolean).join("\n");
    $("bootLine").textContent = `64K RAM SYSTEM  ${(brain.neurons || payload.atlas.n).toLocaleString("en-US")} NEURONS  ${((brain.records || {}).cells || 0).toLocaleString("en-US")} RECORD CELLS FREE`;
    const note = [];
    if (run.run_id) note.push(`The run is ${run.run_id}`);
    if (run.seeds) note.push(`${run.seeds} seeds`);
    if (run.seed !== undefined && run.seed !== null) note.push(`this brain is seed ${run.seed}${run.checkpoint ? ` at the ${run.checkpoint} checkpoint` : ""}`);
    if (run.receipt_sha256) note.push(`receipt ${String(run.receipt_sha256).slice(0, 12)}`);
    if (run.emulator_sha256) note.push(`emulator ${String(run.emulator_sha256).slice(0, 12)}`);
    if ((manifest.bundle || {}).listening) note.push(`the demonstration is heard ${manifest.bundle.listening}`);
    if (manifest.provenance) note.push(manifest.provenance);
    $("receiptNote").textContent = note.length ? note.join(" · ") + "." : "the bundle names no run.";
    $("voiceKey").innerHTML = `<span><i></i></span><span><i></i></span><span><i style="background:rgba(255,255,255,0.8)"></i></span>`;
    const key = $("voiceKey").children;
    key[0].children[0].style.background = `rgb(${DEMO_COLOR.join(",")})`;
    key[0].appendChild(document.createTextNode("WHAT THE CHIP WAS GIVEN"));
    key[1].children[0].style.background = `rgb(${BRAIN_COLOR.join(",")})`;
    key[1].appendChild(document.createTextNode("WHAT THE COMPOSER PLAYED"));
    key[2].appendChild(document.createTextNode("PLAYHEAD"));
    const regions = payload.atlas.regions || [];
    $("legend").innerHTML = regions.slice(0, 9).map(() => `<span><i></i><b></b> <em></em></span>`).join("");
    const legend = $("legend").children;
    regions.slice(0, 9).forEach((region, k) => {
      legend[k].children[0].style.background = `rgb(${(region.color || [140, 140, 160]).join(",")})`;
      legend[k].children[1].textContent = region.name;
      legend[k].children[2].textContent = `${region.count} · ${region.role || roleOf(region.name)}`;
    });
  }

  // ---------------------------------------------------------------- the brain controls
  $("fit").addEventListener("click", () => scan.fit());
  $("view").addEventListener("change", (event) => { scan.options.mode = event.target.value; scan.dirty = true; scan.pending = true; scan.draw(); });
  $("scan").addEventListener("mousemove", (event) => {
    const hit = scan.inspect(event.clientX, event.clientY);
    $("inspector").textContent = hit ? `${hit.region} · neuron ${hit.neuron}\nactivation ${hit.activation.toFixed(3)} · change ${hit.change.toFixed(3)}` : "hover a neuron · scroll to zoom · drag to pan";
  });
  addEventListener("resize", () => { scan.dirty = true; scan.pending = true; drawRoll(); });

  // ---------------------------------------------------------------- what the check reads
  window.__page = {
    ready: false,
    scan,
    records,
    manifest,
    pieces: manifest.pieces.map((p) => p.id),
    tracks: TRACKS,
    select: (id) => choose(id),
    play: (name) => play(name),
    stop: () => stop(true),
    seek(seconds) {
      const element = currentAudio();
      if (element && element.src) { try { element.currentTime = seconds; } catch (error) { /* not seekable */ } }
      internalStart = performance.now() - seconds * 1000;
      internalPaused = seconds;
    },
    get stats() {
      const f = replay[track];
      return {
        piece: piece ? piece.id : null,
        loaded,
        track,
        playing,
        clock_source: source(),
        block: shownBlock,
        seconds: elapsed(),
        frames_shown: shownFrames,
        frames_total: f ? f.steps : 0,
        phase: f ? f.phase : null,
        records_shown: shownRecords,
        records_total: f && f.entries ? f.entries.filter(Boolean).length : 0,
        record_counts: records ? { ...records.counts } : null,
        scan_steps: scan.stepCount,
        audio: Object.fromEntries(TRACKS.map((name) => [name, { src: audio[name].getAttribute("src") || "", duration: audio[name].duration || 0, paused: audio[name].paused }])),
        lanes: Object.fromEntries(TRACKS.map((name) => [name, (laneOf(name) || []).length])),
      };
    },
  };

  fillHeader();
  fillGates();
  void choose(manifest.pieces[0].id).then(() => {
    window.__page.ready = true;
    $("status").textContent = `${scan.snapshot().renderer} · ${payload.atlas.n.toLocaleString("en-US")} neurons`;
    if (PAGE_OPTIONS.autoplay) void play("demonstration");
  });
  requestAnimationFrame(frame);
}

loadPayload().then(main).catch((error) => {
  const box = $("counts");
  if (box) box.textContent = `THE BUNDLE DID NOT LOAD: ${error.message}`;
  const state = $("state");
  if (state) state.textContent = "NO BUNDLE";
  throw error;
});
