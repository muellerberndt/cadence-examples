// The live composer page: the chip, the ear, the records cortex and the search, in the browser.
//
// The loop is the stage's: one 20 ms block at a time, the demonstration on its own chips while the
// body rests, then the answer. The audio of every block is scheduled on the audio clock, so the
// blocks play without gaps as long as the machine computes them faster than it plays them, and the
// page reports how long a block takes. Nothing is scripted: what the composer plays is what the
// search over its records chooses.

// verbatim copies of the shared renderer and records view of cadence-examples/web/
import { BrainScan, decodeArray } from "./brain_scan.js";
import { RecordsView } from "./records_view.js";
import * as cdc from "./js/codec.js";
import * as ear from "./js/hearing.js";
import { Instrument } from "./instrument.js";
import { LiveComposer, gesturesOf } from "./life.js";

const $ = (id) => document.getElementById(id);
const PAGE = { autostart: false, ...(window.__LIVE_PAGE__ || {}) };
const PIECES = ["a-b-a-form", "fast-arpeggio", "one-voice-scale", "ostinato-in-c", "ostinato-in-c-a-third-up", "sparse-answer", "walking-bassline"];
const TITLES = {
  "a-b-a-form": "A-B-A form", "fast-arpeggio": "Fast arpeggio", "one-voice-scale": "One voice, a scale",
  "ostinato-in-c": "Ostinato in C", "ostinato-in-c-a-third-up": "Ostinato in C, a third up",
  "sparse-answer": "Voices that rest apart", "walking-bassline": "Walking bass line", keyboard: "What you played",
};
const HEARD_COLOR = [112, 164, 178], PLAYED_COLOR = [154, 210, 132], PANE_INK = [184, 174, 240], PLAYHEAD = [255, 255, 255];
const KEYS = ["z", "s", "x", "d", "c", "v", "g", "b", "h", "n", "j", "m", ","];
const shade = (c, a) => `rgba(${c[0]},${c[1]},${c[2]},${a})`;

async function fetchJSON(url) {
  const response = await fetch(url);
  if (!response.ok) throw Error(`cannot load ${url}: ${response.status}`);
  return response.json();
}

async function main() {
  const body = document.body.dataset;
  const [spec, ...pieces] = await Promise.all([
    fetchJSON(body.brain),
    ...PIECES.map((name) => fetchJSON(`${body.pieces}/${name}.json`)),
  ]);
  if (spec.format !== "s06-page-brain/1") throw Error(`the brain is ${spec.format}`);
  const library = Object.fromEntries(PIECES.map((name, k) => [name, pieces[k]]));

  const instrument = await Instrument.open(new URL(body.chip, location.href).href);
  const demonstrator = new Instrument(instrument.module);
  const life = new LiveComposer(spec, instrument, demonstrator);
  life.brain.legalMask = life.codec.legalMask();

  // -- the brain beside it
  const scan = new BrainScan($("scan"), spec.atlas, { labels: $("labels"), labelTop: 10 });
  scan.onhover = (hit) => {
    $("inspector").textContent = hit
      ? `NEURON ${hit.neuron} · ${hit.region.toUpperCase()} · DRIVE ${hit.activation.toFixed(3)}`
      : "HOVER A NEURON";
  };
  const records = new RecordsView($("records"), {
    granules: spec.records.cells, active: spec.records.active, inputs: spec.records.inputs,
    groups: readingGroups(spec), rate: spec.planner.rate || 0.2, rewardRate: 1.0,
  });
  const neurons = spec.neurons | 0;
  const reading = Int32Array.from(spec.reading);
  const activation = new Float32Array(neurons);
  const heat = new Float32Array(neurons);

  function readingGroups(spec) {
    const region = decodeArray(spec.atlas.region);
    const names = spec.atlas.regions.map((r) => r.name);
    const out = [];
    spec.reading.forEach((neuron, position) => {
      const name = names[region[neuron]] || "?";
      if (out.length && out[out.length - 1].name === name) out[out.length - 1].end = position + 1;
      else out.push({ name, start: position, end: position + 1 });
    });
    return out;
  }

  // -- the audio clock
  let audio = null, playhead = 0;
  const openAudio = () => {
    if (!audio) audio = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 22050 });
    if (audio.state === "suspended") void audio.resume();
    return audio;
  };
  function schedule(samples) {
    if (!audio || !samples || !samples.length) return;
    const buffer = audio.createBuffer(1, samples.length, audio.sampleRate);
    const channel = buffer.getChannelData(0);
    for (let i = 0; i < samples.length; i++) channel[i] = Math.max(-1, Math.min(1, samples[i]));
    const source = audio.createBufferSource();
    source.buffer = buffer;
    source.connect(audio.destination);
    const now = audio.currentTime;
    if (playhead < now + 0.05) playhead = now + 0.05;
    source.start(playhead);
    playhead += samples.length / audio.sampleRate;
  }

  // -- the phrase the visitor plays
  const keyboard = { notes: [], holding: new Map() };
  function keyboardPiece() {
    const blocks = spec.clock.phrase_blocks | 0;
    const voices = [0, 1, 2].map(() => ({ settings: { attack: 0, decay: 3, sustain: 10, release: 2, waveform: 1 }, notes: [] }));
    for (const note of keyboard.notes) voices[note.voice].notes.push([note.start, note.blocks, note.pitch]);
    return { blocks, shared: { cutoff: 24, resonance: 2, routing: 0, volume: 15 }, voices };
  }

  function pieceOf(name) {
    if (name === "keyboard") return keyboardPiece();
    const piece = library[name];
    return { blocks: piece.blocks, shared: piece.shared, voices: piece.voices };
  }

  // -- the loop
  let running = false, started = 0, blockTimes = [], heardRoll = [], playedRoll = [], lastDecision = null;
  const parts = { demonstration: 0, decision: 0, chips: 0, ear: 0, blocks: 0 };
  const phaseTimes = { listen: [], attempt: [] };
  let lastScan = 0;
  const select = $("piece");
  for (const name of PIECES) {
    const option = document.createElement("option");
    option.value = name;
    option.textContent = TITLES[name];
    select.appendChild(option);
  }
  const option = document.createElement("option");
  option.value = "keyboard";
  option.textContent = TITLES.keyboard;
  select.appendChild(option);

  function start() {
    const name = select.value;
    const piece = pieceOf(name);
    const gestures = gesturesOf(piece).slice(0, spec.clock.phrase_blocks | 0);
    if (!gestures.length) return false;
    openAudio();
    playhead = 0;
    heardRoll = [];
    playedRoll = [];
    blockTimes = [];
    for (const k of Object.keys(parts)) parts[k] = 0;
    phaseTimes.listen = [];
    phaseTimes.attempt = [];
    life.setPhrase(gestures, { passes: spec.clock.listen_passes || 4 });
    running = true;
    started = performance.now();
    $("hint").hidden = true;
    setState(life.label.toUpperCase(), "live");
    return true;
  }

  function stop() {
    running = false;
    setState(life.done ? "DONE" : "READY");
    $("phase").textContent = "IDLE";
    $("phase").className = "state";
  }

  function pump() {
    if (!running) return;
    const ahead = audio ? playhead - audio.currentTime : 0;
    let budget = 8; // blocks a frame, so the page stays responsive while it runs ahead
    while (running && budget-- > 0 && (!audio || ahead + (8 - budget) * 0.02 < 0.5)) {
      const t0 = performance.now();
      const out = life.step();
      if (!out) { stop(); break; }
      const took = performance.now() - t0;
      blockTimes.push(took);
      (out.phase === "attempt" ? phaseTimes.attempt : phaseTimes.listen).push(took);
      if (out.clock) { for (const k of Object.keys(parts)) if (k !== "blocks") parts[k] += out.clock[k] || 0; parts.blocks++; }
      schedule(out.audio);
      if (out.phase === "attempt") {
        playedRoll.push(notesOf(out.values));
        lastDecision = out;
      } else if (out.phase === "listen") {
        heardRoll.push(heardNotesOf(life));
        const blocks = spec.clock.phrase_blocks | 0;
        if (heardRoll.length > blocks) heardRoll = heardRoll.slice(-blocks); // the last pass is the whole phrase
      }
      if (performance.now() - lastScan > 100) { showBrain(); lastScan = performance.now(); }
      if (life.done) { stop(); break; }
    }
    setState(life.done ? "DONE" : life.label.toUpperCase(), life.done ? "" : "live");
  }

  function notesOf(values) {
    const out = [];
    for (let voice = 0; voice < cdc.VOICES; voice++) {
      const pitch = values[cdc.fieldIndex(voice, "pitch")];
      const gate = values[cdc.fieldIndex(voice, "gate")];
      out.push(pitch !== cdc.REST && gate === 1 ? pitch : null);
    }
    return out;
  }

  function heardNotesOf(life) {
    const hearing = life.hearing;
    if (!hearing) return [null, null, null];
    const target = life.reader.read(hearing.subarray(0, ear.WIDTH), [null, null, null], life.fine);
    return target.pitches;
  }

  /** The drive this decision reads, on the neurons that carry it. */
  function showBrain() {
    const drive = life.brain.observationDrive(life.hearing || new Float64Array(4 * ear.WIDTH), Float64Array.from([1, 0]), life.fine);
    const withAction = life.brain.fieldDrive(drive, life.base);
    activation.fill(0);
    let top = 1e-9;
    for (let k = 0; k < reading.length; k++) { const v = Math.abs(withAction[k]); if (v > top) top = v; }
    for (let k = 0; k < reading.length; k++) {
      const value = withAction[k] / top;
      activation[reading[k]] = value;
      heat[reading[k]] = Math.min(1, Math.abs(value));
    }
    scan.show(activation, heat, { draw: false });
    const code = life.brain.cortex.rawDrive(withAction);
    const keep = new Int32Array(life.brain.cortex.active), values = new Float64Array(life.brain.cortex.active);
    life.brain.cortex._winners(code, keep, values);
    records.code({ index: keep, values }, "decide");
    const readingRow = new Float64Array(reading.length);
    for (let k = 0; k < reading.length; k++) readingRow[k] = withAction[k];
    records.setReading(readingRow);
  }

  function setState(text, kind = "") {
    const element = $("state");
    element.textContent = text;
    element.className = `state ${kind}`;
  }

  // -- the roll
  const roll = $("roll"), context = roll.getContext("2d");
  function drawRoll() {
    const rect = roll.getBoundingClientRect();
    if (!rect.width || !rect.height) return;
    const dpr = Math.min(devicePixelRatio || 1, 2);
    const w = Math.round(rect.width * dpr), h = Math.round(rect.height * dpr);
    if (roll.width !== w || roll.height !== h) { roll.width = w; roll.height = h; }
    const ctx = context;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.fillStyle = "#2a1f61";
    ctx.fillRect(0, 0, rect.width, rect.height);
    const blocks = Math.max(1, spec.clock.phrase_blocks | 0);
    const gutter = Math.min(42, Math.max(22, rect.width * 0.11)), top = 3, gap = 9, labelH = 15, footerH = 12;
    const paneH = (rect.height - top - 2 * labelH - gap - footerH) / 2;
    const x0 = gutter, x1 = rect.width - 8;
    const x = (b) => x0 + (b / blocks) * (x1 - x0);
    const panes = [
      { y: top + labelH, label: "HEARD", note: "what the phrase played, as the ear read it", rows: heardRoll, color: HEARD_COLOR },
      { y: top + 2 * labelH + paneH + gap, label: "PLAYED", note: "what the composer answered", rows: playedRoll, color: PLAYED_COLOR },
    ];
    let lo = 127, hi = 0, any = false;
    for (const pane of panes) for (const row of pane.rows) for (const pitch of row) if (pitch) { any = true; lo = Math.min(lo, pitch); hi = Math.max(hi, pitch); }
    if (!any) { lo = 24; hi = 48; }
    if (hi - lo < 12) { const mid = (hi + lo) / 2; lo = Math.round(mid - 6); hi = Math.round(mid + 6); }
    lo -= 1; hi += 1;
    for (const pane of panes) {
      ctx.textAlign = "left";
      ctx.font = "600 10px ui-monospace, Menlo, monospace";
      ctx.fillStyle = shade(PANE_INK, 0.8);
      ctx.fillText(pane.label, 4, pane.y - 4);
      ctx.font = "9px ui-monospace, Menlo, monospace";
      ctx.fillStyle = shade(PANE_INK, 0.45);
      ctx.fillText(pane.note, 8 + ctx.measureText(pane.label).width * 1.2, pane.y - 4);
      ctx.strokeStyle = shade(PANE_INK, 0.09);
      ctx.beginPath();
      for (let b = 0; b <= blocks; b += 6) { const px = Math.round(x(b)) + 0.5; ctx.moveTo(px, pane.y); ctx.lineTo(px, pane.y + paneH); }
      ctx.stroke();
      ctx.strokeStyle = shade(PANE_INK, 0.28);
      ctx.beginPath();
      for (let b = 0; b <= blocks; b += 96) { const px = Math.round(x(b)) + 0.5; ctx.moveTo(px, pane.y); ctx.lineTo(px, pane.y + paneH); }
      ctx.stroke();
      ctx.strokeStyle = shade(PANE_INK, 0.2);
      ctx.strokeRect(Math.round(x0) + 0.5, Math.round(pane.y) + 0.5, Math.round(x1 - x0), Math.round(paneH));
      const rowH = Math.max(2.5, paneH / (hi - lo + 1));
      const y = (pitch) => pane.y + paneH - (pitch - lo + 1) * rowH;
      const bw = Math.max(1.5, (x1 - x0) / blocks);
      for (let b = 0; b < pane.rows.length; b++) {
        for (let voice = 0; voice < pane.rows[b].length; voice++) {
          const pitch = pane.rows[b][voice];
          if (!pitch) continue;
          ctx.fillStyle = shade(pane.color, voice === 0 ? 0.95 : voice === 1 ? 0.75 : 0.55);
          ctx.fillRect(x(b % blocks), y(pitch), bw, Math.max(2, rowH - 1));
        }
      }
    }
    const at = playedRoll.length ? playedRoll.length : heardRoll.length % blocks;
    const px = Math.round(x(Math.min(at, blocks))) + 0.5;
    ctx.strokeStyle = shade(PLAYHEAD, running ? 0.9 : 0.35);
    ctx.beginPath(); ctx.moveTo(px, panes[0].y); ctx.lineTo(px, rect.height - footerH); ctx.stroke();
    ctx.textAlign = "right";
    ctx.fillStyle = shade(PANE_INK, 0.5);
    ctx.font = "9px ui-monospace, Menlo, monospace";
    ctx.fillText(`${blocks} BLOCKS · ${(blocks * 0.02).toFixed(2)} S · 20 MS A BLOCK`, x1, rect.height - 3);
  }

  // -- the keys
  function buildKeyboard() {
    const box = $("keyboard");
    box.textContent = "";
    for (let k = 0; k < KEYS.length; k++) {
      const pitch = 37 + k; // one octave from C, in the codec's pitches
      const button = document.createElement("button");
      button.className = "pianokey";
      button.dataset.pitch = String(pitch);
      button.textContent = KEYS[k].toUpperCase();
      button.addEventListener("pointerdown", (event) => { event.preventDefault(); press(pitch); });
      button.addEventListener("pointerup", () => release(pitch));
      button.addEventListener("pointerleave", () => release(pitch));
      box.appendChild(button);
    }
    const clear = document.createElement("button");
    clear.className = "pianokey clear";
    clear.id = "clearKeys";
    clear.textContent = "CLEAR";
    clear.addEventListener("click", () => { keyboard.notes = []; keyboard.holding.clear(); drawRoll(); });
    box.appendChild(clear);
  }

  function press(pitch) {
    if (keyboard.holding.has(pitch)) return;
    const at = Math.min(spec.clock.phrase_blocks - 1, Math.round((performance.now() - (keyboard.started || (keyboard.started = performance.now()))) / 20));
    keyboard.holding.set(pitch, at);
    select.value = "keyboard";
    markKey(pitch, true);
  }

  function release(pitch) {
    if (!keyboard.holding.has(pitch)) return;
    const start = keyboard.holding.get(pitch);
    keyboard.holding.delete(pitch);
    const end = Math.min(spec.clock.phrase_blocks, Math.max(start + 3, Math.round((performance.now() - keyboard.started) / 20)));
    const voice = keyboard.notes.length % cdc.VOICES;
    keyboard.notes.push({ voice, start, blocks: end - start, pitch });
    markKey(pitch, false);
    drawRoll();
  }

  function markKey(pitch, down) {
    const button = $("keyboard").querySelector(`.pianokey[data-pitch="${pitch}"]`);
    if (button) button.classList.toggle("down", down);
  }

  /** The numbers grid and the ledgers, which live in the model card. */
  function updateCard() {
    if (!$("card").open) return;
    const c = records.counts;
    fill($("numbers"), [
      ["phrase", TITLES[select.value] || select.value],
      ["notes you played", `${keyboard.notes.length}`],
      ["block", `${life.block}`],
      ["phase", life.label],
      ["controller", lastDecision ? lastDecision.controller : "none"],
      ["ms a block", blockTimes.length ? (blockTimes.reduce((a, b) => a + b, 0) / blockTimes.length).toFixed(2) : "none"],
    ], (cell, label, value) => {
      cell.className = "number";
      cell.innerHTML = "<span></span><b></b>";
      cell.children[0].textContent = label;
      cell.children[1].textContent = value;
    });
    ledger($("ledger"), [
      ["neurons", neurons.toLocaleString("en-US")], ["reading", reading.length], ["blocks", life.block],
      ["heard", heardRoll.length], ["answered", playedRoll.length],
      ["ms a block, last 50", blockTimes.length ? (blockTimes.slice(-50).reduce((a, b) => a + b, 0) / Math.min(50, blockTimes.length)).toFixed(2) : "0"],
    ]);
    ledger($("recordsLedger"), [
      ["cells", records.granules], ["active", spec.records.active], ["reading", spec.records.inputs],
      ["codes", c.codes], ["writes", c.writes], ["learning", "off"],
    ]);
    const shot = scan.snapshot();
    ledger($("rendererLedger"), [
      ["renderer", shot.renderer], ["synapses", shot.synapses.toLocaleString("en-US")],
      ["regions", shot.regions], ["audio ahead", running && audio ? `${Math.max(0, playhead - audio.currentTime).toFixed(2)} s` : "not playing"],
    ]);
  }

  function ledger(box, rows) {
    box.innerHTML = rows.map(([k, v]) => `<span>${k}</span><b>${v}</b>`).join("");
  }

  function fill(box, rows, paint) {
    while (box.children.length > rows.length) box.lastChild.remove();
    while (box.children.length < rows.length) box.appendChild(document.createElement("div"));
    rows.forEach(([label, value], k) => paint(box.children[k], label, value));
  }

  /** The gate table of the run this brain comes from, the open measures marked. */
  function showGates() {
    const gates = spec.gates || [];
    const run = spec.run || {};
    $("runNote").textContent = gates.length
      ? `${run.run_id || "the acceptance run"} · ${run.seeds || 0} seeds · checkpoint ${run.checkpoint || "?"} · receipt ${(run.receipt_sha256 || "").slice(0, 12)} · ${gates.filter((g) => !g.open && g.passed).length} of ${gates.filter((g) => !g.open).length} gates passed, and the measures below them are open`
      : "the page asset carries no gate table";
    const body = $("gateTable").querySelector("tbody");
    const number = (v) => (Number.isInteger(v) ? v.toLocaleString("en-US") : Number(v).toFixed(3));
    const rows = [];
    for (const open of [false, true]) {
      const part = gates.filter((g) => !!g.open === open);
      if (!part.length) continue;
      rows.push(`<tr class="section"><td colspan="4">${open ? "open: measured, not gated" : "the gates of this run"}</td></tr>`);
      for (const gate of part) {
        const mark = open ? `<span class="open">OPEN</span>` : gate.passed ? `<span class="pass">PASS</span>` : `<span class="fail">FAIL</span>`;
        rows.push(`<tr><td>${gate.name.replace(/_/g, " ")}${gate.note ? `<em>${gate.note}</em>` : ""}</td><td class="value">${number(gate.value)}</td><td class="want">${number(gate.threshold)}</td><td class="mark">${mark}</td></tr>`);
      }
    }
    body.innerHTML = rows.join("");
  }

  addEventListener("keydown", (event) => {
    const k = KEYS.indexOf(event.key.toLowerCase());
    if (k >= 0) press(37 + k);
  });
  addEventListener("keyup", (event) => {
    const k = KEYS.indexOf(event.key.toLowerCase());
    if (k >= 0) release(37 + k);
  });

  $("listen").addEventListener("click", () => { if (!start()) setState("PLAY SOMETHING FIRST"); });
  $("stop").addEventListener("click", stop);
  $("card").addEventListener("toggle", () => updateCard());
  $("learning").addEventListener("change", (event) => {
    if (!event.target.checked) return;
    event.target.checked = false;
    const note = $("rendererLedger");
    note.innerHTML = `<span>learning</span><b>not in this build</b>` + note.innerHTML;
  });
  addEventListener("resize", () => { scan.fit(); });

  // -- the two clocks: the composer on a timer, the picture on the frames
  //
  // The blocks must not wait for the drawing. A software renderer draws the whole brain in about a
  // second a frame, and the composer would then compute eight blocks a second instead of fifty, so
  // the loop that hears and answers runs on its own timer and the frame only draws what it finds.
  function tick() {
    const t0 = performance.now();
    pump();
    const ahead = audio && running ? playhead - audio.currentTime : 0;
    const spent = performance.now() - t0;
    setTimeout(tick, running ? (ahead > 0.3 ? 20 : spent > 40 ? 0 : 4) : 80);
  }

  let lastCard = 0;
  function frame(now) {
    requestAnimationFrame(frame);
    scan.draw(now);
    records.draw();
    drawRoll();
    $("clock").textContent = `BLOCK ${life.block} · ${((life.block * 0.02) | 0).toString().padStart(2, "0")}:${Math.floor((life.block * 0.02 * 100) % 100).toString().padStart(2, "0")}`;
    $("phase").textContent = running ? `LIVE · ${records.counts.codes} CODES READ` : "IDLE";
    $("phase").className = running ? "state live" : "state";
    if (now - lastCard > 250) { updateCard(); lastCard = now; }
  }

  $("counts").textContent = [
    `${neurons.toLocaleString("en-US")} NEURONS · ${spec.records.cells.toLocaleString("en-US")} RECORD CELLS`,
    `20 MS BLOCK · 22,050 HZ · SEED ${spec.seed}`,
  ].join("\n");
  showGates();
  buildKeyboard();
  showBrain();  // the scan and the records carry the resting drive before the first phrase
  setState("READY");

  window.__page = {
    ready: true,
    life, scan, records, spec,
    pieces: [...PIECES, "keyboard"],
    select: (name) => { select.value = name; updateCard(); },
    start, stop,
    get stats() {
      return {
        piece: select.value, block: life.block, phase: life.label, running,
        played: life.played, listened: life.listened, done: life.done,
        latched_steps: life.latched.size, latched_voice_steps: life.latchedVoice.map((m) => m.size),
        heard_blocks: heardRoll.length, played_blocks: playedRoll.length,
        controller: lastDecision ? lastDecision.controller : null,
        values: lastDecision ? Array.from(lastDecision.values) : null,
        ms_per_block: blockTimes.length ? blockTimes.reduce((a, b) => a + b, 0) / blockTimes.length : 0,
        ms_per_block_p95: blockTimes.length ? [...blockTimes].sort((a, b) => a - b)[Math.floor(blockTimes.length * 0.95)] : 0,
        blocks_timed: blockTimes.length,
        ms_parts: parts.blocks ? Object.fromEntries(Object.entries(parts).filter(([k]) => k !== "blocks").map(([k, v]) => [k, v / parts.blocks])) : null,
        ms_by_phase: Object.fromEntries(Object.entries(phaseTimes).map(([k, v]) => [k, v.length ? {
          blocks: v.length, mean: v.reduce((a, b) => a + b, 0) / v.length,
          p95: [...v].sort((a, b) => a - b)[Math.floor(v.length * 0.95)],
        } : null])),
        audio_seconds: audio ? playhead - audio.currentTime : 0,
      };
    },
    /** One recorded decision, run through this page's own planner: the check's parity hook. */
    replayCase(item) {
      const asFloats = (s) => (s === null || s === undefined ? null : Float64Array.from(decodeArray(s)));
      const asInts = (s) => (s === null || s === undefined ? null : Int32Array.from(s));
      const st = item.planner;
      const planner = life.planner;
      planner.mode = st.mode; planner.turn = st.turn; planner.target_channels = st.target_channels;
      planner.target = asFloats(st.target); planner.fine = asFloats(st.fine); planner.heard_fine = asFloats(st.heard_fine);
      planner.voices = st.voices === null ? null : st.voices.map(asFloats);
      planner.voice_fines = st.voice_fines === null ? null : st.voice_fines.map(asFloats);
      planner.echo = asInts(st.echo); planner.note = asInts(st.note); planner.forced = asInts(st.forced);
      planner.held_target = st.held_target === null ? null : { ...st.held_target, pitches: st.held_target.pitches };
      planner.note_target = null;
      const got = planner.plan(life.brain, {
        hearing: asFloats(item.hearing),
        attending: Float64Array.from(item.attending),
        hypothetical: asFloats(item.hypothetical_reading),
        mask: Uint8Array.from(decodeArray(item.mask)),
        base: item.executed === null ? cdc.silence() : cdc.fromJoint(BigInt(item.executed)),
        heard_fine: asFloats(item.fine_mix),
      });
      return { values: Array.from(got.values), controller: got.controller };
    },
  };
  requestAnimationFrame(frame);
  tick();
  if (PAGE.autostart) start();
}

main().catch((error) => {
  const box = $("counts");
  if (box) box.textContent = `THE PAGE DID NOT LOAD: ${error.message}`;
  const state = $("state");
  if (state) state.textContent = "NO BRAIN";
  throw error;
});
