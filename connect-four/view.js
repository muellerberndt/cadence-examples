import { drop, legal, winner, brainSnapshot } from "./brain.js";
export function mountGame({ $, ctx, metrics, explain, table, evidence }) {
  let board = Array(42).fill(0),
    turn = 1,
    busy = false,
    result = null,
    preview = null,
    previewStep = 0,
    job = 0,
    status = "Your turn · choose a column",
    error = null;
  const worker = new Worker(new URL("./worker.js", import.meta.url), {
    type: "module",
  });
  const bounds = (w, h) => {
    const cell = Math.min((w * 0.64 - 28) / 7, (h - 54) / 6);
    return { cell, x: 18, y: 36 };
  };
  $("headline").innerHTML = "Make your move.<br>Watch it consider the reply.";
  $("intro").textContent =
    "Play Connect Four against a reasoner that compares isolated futures. Inspect the lines it considers and watch its self-monitor request more thought when choices are ambiguous.";
  $("stage-label").textContent =
    "LIVE BOARD → IMAGINED FUTURES → SELF-READING → ACTION";
  $("stage-detail").textContent = "You: coral · Cadence: mint";
  $("stage-hint").textContent =
    "Click a column or use the numbered buttons. Preview lines are hypothetical; only the chosen move changes the board.";
  $("scene").setAttribute(
    "aria-label",
    "Connect Four board on the left; an inspected hypothetical continuation on the right. Numbered buttons also play columns.",
  );
  $("controls").innerHTML =
    `<div><h2>Think before placing a stone.</h2><p>Watch pauses before the move. Choose a candidate to inspect its predicted continuation, then let Cadence play its preferred move.</p></div><div class="buttons" id="game-columns">${Array.from({ length: 7 }, (_, i) => `<button data-column="${i}" aria-label="Play column ${i + 1}">${i + 1}</button>`).join("")}</div><div class="buttons"><button id="new-game">New game</button><button id="cadence-first">Cadence starts</button><button id="watch-thought" aria-pressed="false">Watch before moving</button><button id="play-thought" disabled>Play Cadence’s move</button></div><div><label for="game-depth">Thinking depth</label><select id="game-depth"><option value="4">4 plies · quick</option><option value="6" selected>Up to 6 · adaptive</option></select><div class="buttons" style="margin-top:8px"><button id="self-monitor" aria-pressed="true">Self-monitor on</button></div></div>${metrics(
      [
        ["Imagined positions", "0", "game-nodes"],
        ["Completed depth", "0", "game-depth-readout"],
        ["Monitor request", "—", "game-review"],
      ],
    )}<div id="game-futures" class="buttons"></div><div><label for="future-step">Inspect future ply</label><input id="future-step" type="range" min="0" max="6" value="0"></div><p id="game-status" aria-live="polite"></p>`;
  let watch = false,
    monitoring = true;
  const update = () => {
    $("game-status").textContent = status;
    $("game-columns")
      .querySelectorAll("button")
      .forEach(
        (b) =>
          (b.disabled =
            turn !== 1 ||
            busy ||
            !!winner(board) ||
            !legal(board).includes(+b.dataset.column)),
      );
    $("play-thought").disabled = !(busy && result && watch);
  };
  const playAI = () => {
    if (!result || !legal(board).includes(result.column) || turn !== -1) return;
    board = drop(board, result.column, -1);
    job++; // Ignore deeper worker messages if a completed partial search was played.
    busy = false;
    preview = null;
    turn = 1;
    const won = winner(board);
    status = won
      ? "Cadence connected four."
      : !legal(board).length
        ? "Draw · board full"
        : "Your turn · choose a column";
    update();
  };
  const think = () => {
    busy = true;
    result = null;
    preview = null;
    status = "Comparing possible replies…";
    update();
    worker.postMessage({
      id: ++job,
      board,
      player: -1,
      depth: +$("game-depth").value,
      monitoring,
    });
  };
  const play = (c) => {
    if (turn !== 1 || busy || winner(board) || !legal(board).includes(c))
      return;
    board = drop(board, c, 1);
    if (winner(board)) {
      status = "You connected four!";
      update();
      return;
    }
    if (!legal(board).length) {
      status = "Draw · board full";
      update();
      return;
    }
    turn = -1;
    think();
  };
  $("game-columns")
    .querySelectorAll("button")
    .forEach((b) => (b.onclick = () => play(+b.dataset.column)));
  const reset = (first) => {
    job++;
    board = Array(42).fill(0);
    turn = first ? -1 : 1;
    busy = false;
    result = null;
    preview = null;
    status = "Your turn · choose a column";
    error = null;
    $("game-nodes").textContent = "0";
    $("game-depth-readout").textContent = "0";
    $("game-review").textContent = "—";
    $("game-futures").replaceChildren();
    update();
    if (first) think();
  };
  $("new-game").onclick = () => reset(false);
  $("cadence-first").onclick = () => reset(true);
  $("watch-thought").onclick = () => {
    watch = !watch;
    $("watch-thought").setAttribute("aria-pressed", String(watch));
    update();
    if (!watch && busy && result) playAI();
  };
  $("play-thought").onclick = playAI;
  $("self-monitor").onclick = () => {
    monitoring = !monitoring;
    $("self-monitor").setAttribute("aria-pressed", String(monitoring));
    $("self-monitor").textContent = monitoring
      ? "Self-monitor on"
      : "Self-monitor off";
  };
  $("future-step").oninput = () => (previewStep = +$("future-step").value);
  worker.onmessage = ({ data }) => {
    if (data.id !== job) return;
    if (data.kind === "error") {
      error = data.message;
      status = "Thinking stopped: " + error;
      busy = false;
      update();
      return;
    }
    const r = data.result;
    result = r;
    preview = r.sequence;
    previewStep = preview.length;
    $("future-step").max = preview.length;
    $("future-step").value = previewStep;
    $("game-nodes").textContent = r.nodes.toLocaleString();
    $("game-depth-readout").textContent = r.depth;
    $("game-review").textContent = r.review.request_more
      ? "More thought"
      : "Budget sufficient";
    $("game-futures").replaceChildren(
      ...r.candidates.map((candidate) => {
        const b = document.createElement("button");
        b.dataset.future = candidate.column;
        b.textContent = `${candidate.column + 1}: ${candidate.score > 90000 ? "winning line" : candidate.score < -90000 ? "losing line" : candidate.score.toFixed(1)}`;
        b.onclick = () => {
          preview = candidate.sequence;
          previewStep = preview.length;
          $("future-step").max = preview.length;
          $("future-step").value = previewStep;
        };
        return b;
      }),
    );
    status =
      data.kind === "done"
        ? `Considering column ${r.column + 1} · ${r.depth} plies searched${r.budgetExhausted ? " · node limit reached" : ""}`
        : `Depth ${r.depth} complete · ${r.review.request_more ? "monitor requests a deeper look" : "evaluating budget"}`;
    update();
    if (data.kind === "done" && !watch) playAI();
  };
  worker.onerror = () => {
    error = "Worker unavailable";
    status =
      "Reasoning worker could not start. Reload through the local launcher.";
    busy = false;
    update();
  };
  explain([
    [
      "Counterfactual workspace",
      "Every candidate gets a separate board. A supplied transition model predicts legal moves and replies. The live board stays unchanged while futures are explored.",
    ],
    [
      "Local value evaluator",
      "Threat counts drive a six-owner graded value circuit. Bounded adversarial search compares its predicted outcomes; it does not learn from imagined events. A shallow evaluator-only control locates the value of lookahead.",
    ],
    [
      "Self-reading control",
      "Six monitor owners read changes in predicted option values, ambiguity and budget pressure. Their output permits search beyond four plies. This is metacognitive control, not a claim of consciousness or a uniquely biological mechanism.",
    ],
  ]);
  $("evidence-title").textContent = "What looking ahead changes";
  $("evidence-note").textContent =
    "Four fixed openings, both sides: adaptive search versus the same evaluator at one or four plies. The monitor permits extra work; these results measure a search-budget difference, not an exclusive patch-net advantage. No solved-game or guaranteed-win claim.";
  $("evidence-table").innerHTML = table(
    ["Opponent", "Cadence wins", "Draws", "Cadence losses"],
    ["one_ply", "four_ply"].map((control) => {
      const rows = evidence.games.filter((g) => g.control === control);
      return [
        control === "one_ply" ? "One-ply evaluator" : "Four-ply search",
        `${rows.filter((g) => g.outcome === 1).length}/${rows.length}`,
        rows.filter((g) => g.outcome === 0).length,
        rows.filter((g) => g.outcome === -1).length,
      ];
    }),
  );
  $("boundary").textContent =
    "Conventional minimax with the same evaluator can implement the same planning algorithm. A transformer can also be coupled to search. This demo tests architecture, future comparison and self-monitor feedback; it does not establish a special monopoly on reasoning or consciousness.";
  document.querySelector(".evidence details a").href =
    "connect-four/evidence.json";
  update();
  return {
    draw(w, h) {
      const a = bounds(w, h);
      function drawBoard(b, x, y, cell) {
        ctx.fillStyle = "#1f3d49";
        ctx.beginPath();
        ctx.roundRect(x - 4, y - 4, cell * 7 + 8, cell * 6 + 8, 8);
        ctx.fill();
        for (let row = 0; row < 6; row++)
          for (let col = 0; col < 7; col++) {
            ctx.fillStyle =
              b[row * 7 + col] === 1
                ? "#ff947c"
                : b[row * 7 + col] === -1
                  ? "#87ecc2"
                  : "#0a161d";
            ctx.beginPath();
            ctx.arc(
              x + (col + 0.5) * cell,
              y + (row + 0.5) * cell,
              cell * 0.37,
              0,
              Math.PI * 2,
            );
            ctx.fill();
          }
      }
      drawBoard(board, a.x, a.y, a.cell);
      ctx.fillStyle = "#a3b4b9";
      ctx.font = "11px system-ui";
      ctx.fillText("REAL BOARD", a.x, 20);
      const px = w * 0.68,
        ps = (w - px - 18) / 7;
      ctx.fillText("IMAGINED CONTINUATION", px, 20, w - px - 12);
      let predicted = board.slice(),
        p = turn;
      if (preview && turn === -1)
        for (const c of preview.slice(0, previewStep)) {
          if (winner(predicted) || !legal(predicted).includes(c)) break;
          predicted = drop(predicted, c, p);
          p = -p;
        }
      drawBoard(predicted, px, 48, ps);
      ctx.fillStyle = "#a3b4b9";
      ctx.fillText(
        preview ? `Future ply ${previewStep}` : "Select “Watch before moving”",
        px,
        70 + ps * 6,
        w - px - 12,
      );
      $("stage-readout").textContent = busy
        ? "Thinking in isolated branches"
        : winner(board)
          ? "Game complete"
          : turn === 1
            ? "Your turn"
            : "Cadence turn";
    },
    pointer(x, y, w, h) {
      const a = bounds(w, h);
      if (x >= a.x && x < a.x + 7 * a.cell && y >= a.y && y < a.y + 6 * a.cell)
        play(Math.floor((x - a.x) / a.cell));
    },
    brain() {
      return {
        ...brainSnapshot(board, -1, result),
        behavior: {
          label: busy
            ? "Imagining replies"
            : winner(board)
              ? "Game complete"
              : "Observing board",
          tone: busy ? "correcting" : "seeking",
        },
      };
    },
    snapshot() {
      return {
        board: board.slice(),
        turn,
        busy,
        winner: winner(board),
        result,
        preview,
        previewStep,
        error,
      };
    },
  };
}
