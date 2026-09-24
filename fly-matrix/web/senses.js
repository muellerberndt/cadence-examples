// The senses: the room's quantities onto the afferents. Twin of fruitfly/senses.py.
export const STROKE_PLANE_TILT = 30 * Math.PI / 180;
export const HALTERE_SATURATION = 20.0;   // rad/s
export const HALTERE_TONE = 0.5;
export const OCELLI_GAIN = 2.0;
export const AIRSPEED_SATURATION = 1.0;   // m/s
export const OCELLUS_ELEVATION = 45 * Math.PI / 180, OCELLUS_AZIMUTH = 35 * Math.PI / 180;

export const CHANNELS = [
  { id: "haltere_left", sets: ["haltere:left"], quantity: "the flight tone of the left haltere: one spike per stroke while flying; the rate model carries no timing, so it is a constant in flight" },
  { id: "haltere_right", sets: ["haltere:right"], quantity: "the same on the right haltere" },
  { id: "ocelli_left", sets: ["ocelli:left"], quantity: "brightness the left ocellus sees, up and to the left" },
  { id: "ocelli_right", sets: ["ocelli:right"], quantity: "the same for the right ocellus" },
  { id: "antenna", sets: ["jo:C:left", "jo:C:right", "jo:E:left", "jo:E:right"], quantity: "airspeed through the antennae" },
  { id: "optic_flow_left", sets: ["lptc:hs:left", "lptc:vs:left"], quantity: "rotational optic flow, injected at HS and VS" },
  { id: "optic_flow_right", sets: ["lptc:hs:right", "lptc:vs:right"], quantity: "the same on the right" },
];

export function haltereTone(flying = true) { return flying ? [HALTERE_TONE, HALTERE_TONE] : [0, 0]; }
/** The Coriolis load magnitude per side, for the instrument strip; the brain receives the tone. */
export function haltereDrive(omega) {
  const roll = omega[0], pitch = omega[1], yaw = omega[2];
  const c = Math.cos(STROKE_PLANE_TILT), s = Math.sin(STROKE_PLANE_TILT);
  const left = Math.abs(pitch * c + roll * s) + 0.5 * Math.abs(yaw);
  const right = Math.abs(pitch * c - roll * s) + 0.5 * Math.abs(yaw);
  return [Math.min(1, left / HALTERE_SATURATION), Math.min(1, right / HALTERE_SATURATION)];
}
export function ocellarDrive(up, down) { const t = up + down; if (t <= 0) return 0; return Math.min(1, Math.max(0, OCELLI_GAIN * (up - down) / t)); }
/** Left and right ocellar drive from the body-to-world rotation matrix (row major, 9 numbers). */
export function ocelliLR(R) {
  const ce = Math.cos(OCELLUS_ELEVATION), se = Math.sin(OCELLUS_ELEVATION), ca = Math.cos(OCELLUS_AZIMUTH), sa = Math.sin(OCELLUS_AZIMUTH);
  const out = [];
  for (const side of [1, -1]) { const dx = ce * ca, dy = side * ce * sa, dz = se; const up = R[6] * dx + R[7] * dy + R[8] * dz; out.push(Math.min(1, Math.max(0, up))); }
  return out;
}
export function antennaDrive(airspeed) { return Math.min(1, Math.max(0, airspeed / AIRSPEED_SATURATION)); }
export const FLOW_REST = 0.5, FLOW_SATURATION = 20.0;
/** HS and VS as graded cells: a resting level moved by the flow in their preferred directions. */
export function opticFlowDrive(yawRate, rollRate, pitchRate, side) {
  const sign = side === "left" ? 1 : -1;
  const hs = sign * yawRate, vs = sign * rollRate - pitchRate;
  return Math.min(1, Math.max(0, FLOW_REST + 0.5 * (hs + vs) / FLOW_SATURATION));
}
/** All channel drives from {omega:[roll,pitch,yaw], up, down, airspeed, flow:[yaw,roll,pitch]}. */
export function sense(state) {
  const [l, r] = haltereTone(state.flying ?? true);
  const flow = state.flow || [0, 0, 0];
  const [ol, or_] = state.rotation ? ocelliLR(state.rotation) : [ocellarDrive(state.up ?? 1, state.down ?? 1), ocellarDrive(state.up ?? 1, state.down ?? 1)];
  return {
    haltere_left: l, haltere_right: r,
    ocelli_left: ol, ocelli_right: or_,
    antenna: antennaDrive(state.airspeed || 0),
    optic_flow_left: opticFlowDrive(flow[0], flow[1], flow[2], "left"),
    optic_flow_right: opticFlowDrive(flow[0], flow[1], flow[2], "right"),
  };
}
