// Reader-facing descriptions. Keep learning distinct from running a supplied controller.
const guides = {
  game: [
    [
      "Ready to reason",
      "Play a column. The agent uses supplied rules, a graded patch evaluator and bounded search; no pretrained game policy is needed.",
    ],
    [
      "Inspect an imagined future",
      "Enable Watch before moving, play, then select a candidate column and scrub its predicted continuation. The live board changes only when you execute the chosen move.",
    ],
    [
      "Why Cadence fits",
      "Separate state copies, local evaluators and a self-reading monitor compose into an inspectable planner. The monitor can request more depth. Conventional search can do this too; no exclusive planning advantage is claimed.",
    ],
  ],
  mouse: [
    [
      "Ready with three lessons",
      "Cheese, home and water are supplied cue/destination demonstrations. The fourth cue starts untaught. Your additional lessons persist in this browser. The maze and navigation rule are supplied.",
    ],
    [
      "Watch a lesson take effect",
      "Select New task, choose Flag, then Teach task and Perform task. Change its destination and teach again; earlier distinct cues retain their lessons. New maze carries the same task memory into another layout.",
    ],
    [
      "Why Cadence fits",
      "Task records and the current spatial field have separate jobs: one local write revises a goal, while recurrent settlement finds a route through the supplied map. A frozen route cannot adapt; BFS and dictionary lookup can. This is a compositional control demo, not a trained-MLP win.",
    ],
  ],
  arm: [
    [
      "Ready to draw",
      "Draw on the left pad or upload a line drawing. The eye reads 24 × 24 pixels. Joint and pencil-lift motors start with supplied weights and geometry; this is not a pretrained drawing skill.",
    ],
    [
      "Watch feedback, not training",
      "Disturb a joint as the arm draws. Visual error and motor correction settle together. Repeat with Feedback off to see what pose readback contributes. These corrections change activity and movement, not learned weights.",
    ],
    [
      "Why Cadence fits",
      "Retinal input, visual/proprioceptive error and joint coordination drive six motor neurons. Motor ablations stop the corresponding actuator; pose feedback repairs disturbances. Classical feedback controllers can also correct motion; no MLP comparison is claimed.",
    ],
  ],
  fly: [
    [
      "Ready body, fresh memory",
      "Both foragers start with supplied sensors, directional motor circuits and exploration. Nectar memory starts fresh; the MLP starts randomly initialized. This is a fly-inspired planar body, not a reconstructed fly nervous system.",
    ],
    [
      "Watch learning on contact",
      "Let the agents visit flowers and watch the encounter readout. Only contact reveals nectar and triggers an update. Change nectar to make old predictions wrong, then watch new encounters revise them. Pause freezes the scene; Reset starts fresh.",
    ],
    [
      "Why Cadence fits",
      "A bounded memory can replace an observed value in one residual write and immediately inform the next target choice. The MLP learns online too, with a selectable update budget. Live nectar totals reflect different experiences; the changing-memory demo supplies the matched-stream comparison.",
    ],
  ],
  worm: [
    [
      "C. elegans is included",
      "The bundled chemical graph has 297 participating annotated neurons and 3,604 edges. Cadence runs a supplied recurrent rule immediately; a trained MLP comparator is bundled too. Dynamics are imposed, not validated whole-worm physiology.",
    ],
    [
      "Build, feed and inspect",
      "Paint food and walls, or erase a passage. Contact consumes a patch; turning smell off stops cue-driven movement. Habitat uses supplied diffusion and a directional sensory/motor circuit that drives the body. Switch to Circuit for stimulation and lesions. Neither view trains weights.",
    ],
    [
      "Why Cadence fits",
      "Local state and readback reuse known interactions after an intervention, avoiding refitting an input/output surrogate. Cadence agrees with the converged reference; the trained MLP is faster per query but less accurate, especially after lesions. Conventional recurrence also reuses the mechanism.",
    ],
  ],
  memory: [
    [
      "Ready to learn from scratch",
      "The store starts blank and the MLP starts randomly initialized. Eight explicit keys select four possible values. No pretrained checkpoint is needed, and Clear restarts the live experiment.",
    ],
    [
      "Watch each write",
      "Choose a key and value, then Teach once. Replace the value and inspect recall of other keys. Run stream automates observations while accuracy updates. Increase key similarity to see interference; it starts a fresh stream.",
    ],
    [
      "Why Cadence fits",
      "One residual write replaces a record and preserves orthogonal-key records. Matched distinct-key trials give exact recall with 32 mutable entries, exceeding the tested MLP update budgets. Dictionary lookup is exact too; strongly overlapping keys can favor the MLP. This measures explicit record storage, not general intelligence.",
    ],
  ],
};

export function showGuide(element, mode) {
  element.dataset.demo = mode;
  element.replaceChildren(
    ...guides[mode].map(([title, copy]) => {
      const card = document.createElement("div");
      const heading = document.createElement("h3");
      const paragraph = document.createElement("p");
      heading.textContent = title;
      paragraph.textContent = copy;
      card.append(heading, paragraph);
      return card;
    }),
  );
}
