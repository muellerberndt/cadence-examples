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
      "Value, candidate and self-monitor neurons settle jointly; search branches stay isolated. The monitor can request more depth. Conventional search can do this too; no exclusive planning advantage is claimed.",
    ],
  ],
  arm: [
    [
      "Ready to draw",
      "Draw on the left pad or upload a line drawing. The eye reads 48 × 48 pixels. Joint and pencil-lift motors start with supplied weights and geometry; this is not a pretrained drawing skill.",
    ],
    [
      "Watch feedback, not training",
      "Disturb a joint as the arm draws. Visual error and motor correction settle together. Repeat with Feedback off to see what pose readback contributes. These corrections change activity and movement, not learned weights.",
    ],
    [
      "Why Cadence fits",
      "Retina, visual/proprioceptive error, coordination and six motor neurons settle in one connected graph. Motor ablations stop the corresponding actuator; pose feedback repairs disturbances. Classical feedback controllers can also correct motion; no MLP comparison is claimed.",
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
      "A contact revises memory in one residual write. Cue, recall and motor regions then settle together before motion. The MLP learns online too, with a selectable update budget. Live nectar totals reflect different experiences; the matched-stream benchmark below measures learning on identical experiences.",
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
      "Chemical and directional motor regions share one equilibrium under the current odor cue; local activity changes propagate across their synapses. Cadence agrees with the converged reference; the trained MLP is faster per query but less accurate, especially after lesions. Conventional recurrence also reuses the mechanism.",
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
