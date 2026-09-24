// The body of the fly: the exact twin of fruitfly/body.py. Same constants, same arithmetic in the
// same order, plain numbers only. Read body.py for the model; this file keeps its structure.
//
// Frames. World: x along the long wall, y along the short wall, z up, origin at a floor corner.
// Body: x forward through the head, y left, z up. The quaternion q = [w, x, y, z] rotates body
// vectors into the world. Controls: {aL, aR, betaL, betaR, sL, sR, f}.

// ---------------------------------------------------------------------------------------------
// Constants. fruitfly/body.py carries the same list with the same values.

export const MASS = 1.0e-6;                         // kg, body mass: about one milligram
export const GRAVITY = 9.81;                        // m/s^2
export const INERTIA = [1.25e-13, 5.8e-13, 5.8e-13]; // kg m^2, about body x, y, z: a 2.5 mm by 0.5 mm cylinder
export const F_NOMINAL = 200.0;                     // Hz, nominal wingbeat frequency
export const F_HOVER = MASS * GRAVITY / 2.0;        // N, wing force per wing at hover trim: half the weight
export const HINGE_Y = 0.6e-3;                      // m, the wing hinges sit 0.6 mm left and right of the axis
export const HINGE_Z = 0.3e-3;                      // m, and 0.3 mm above the centre of mass
export const K_WING = 6.0e-6;                       // N s/m, wing damping per wing at hover amplitude and 200 Hz
export const K_FCT = [1.0e-11, 2.0e-12, 5.0e-11];   // N m s/rad, flapping counter-torque damping about x, y, z
export const RHO = 1.2;                             // kg/m^3, air density
export const C_D = 1.0;                             // body drag coefficient
export const AREA = 2.0e-6;                         // m^2, body reference area
export const RADIUS = 1.2e-3;                       // m, the contact sphere of the body
export const ROOM = [4.0, 3.0, 2.6];                // m, the room: x by y by z
export const K_CONTACT = 1.0;                       // N/m, contact spring against floor, walls and ceiling
export const C_CONTACT = 2.0e-3;                    // N s/m, contact damping along the surface normal
export const MU = 0.6;                              // friction coefficient on a surface
export const C_SLIDE = 2.0e-3;                      // N s/m, viscous friction below the Coulomb limit
export const C_SPIN = 2.0e-11;                      // N m s/rad, spin damping while touching a surface (the legs)
export const WIND_TAU = 0.05;                       // s, decay time of a gust
export const DT = 0.0005;                           // s, physics step: 2000 Hz
export const A_MAX = 1.3;                           // stroke amplitude ceiling, 1 is hover
export const BETA_MAX = 0.5;                        // rad, stroke-plane tilt ceiling
export const S_MAX = 0.5e-3;                        // m, mean stroke shift ceiling
export const F_MAX = 250.0;                         // Hz, wingbeat frequency ceiling

// Numbers the model holds to: mass about 1 mg; wingbeat about 200 Hz; hover force equals the
// weight; cruise 0.2 to 0.5 m/s with bursts near 1 m/s; a 90 degree saccade in about 50 ms; an
// open-loop pitch instability that grows within a few wingbeats.

export const PI = 3.141592653589793;
export const TWO_PI = 6.283185307179586;
export const HALF_PI = 1.5707963267948966;
export const QUARTER_PI = 0.7853981633974483;

// ---------------------------------------------------------------------------------------------
// Exact-twin arithmetic: sine, cosine and arctangent from products, sums and square roots only.

const S3 = -1.0 / 6.0, S5 = 1.0 / 120.0, S7 = -1.0 / 5040.0, S9 = 1.0 / 362880.0;
const S11 = -1.0 / 39916800.0, S13 = 1.0 / 6227020800.0, S15 = -1.0 / 1307674368000.0, S17 = 1.0 / 355687428096000.0;
const C2 = -1.0 / 2.0, C4 = 1.0 / 24.0, C6 = -1.0 / 720.0, C8 = 1.0 / 40320.0;
const C10 = -1.0 / 3628800.0, C12 = 1.0 / 479001600.0, C14 = -1.0 / 87178291200.0, C16 = 1.0 / 20922789888000.0;

function sinPoly(x) {
  const x2 = x * x;
  return x * (1.0 + x2 * (S3 + x2 * (S5 + x2 * (S7 + x2 * (S9 + x2 * (S11 + x2 * (S13 + x2 * (S15 + x2 * S17))))))));
}

function cosPoly(x) {
  const x2 = x * x;
  return 1.0 + x2 * (C2 + x2 * (C4 + x2 * (C6 + x2 * (C8 + x2 * (C10 + x2 * (C12 + x2 * (C14 + x2 * C16)))))));
}

function reduce(x) {
  const k = Math.floor(x / TWO_PI + 0.5);
  return x - k * TWO_PI;
}

export function sin_(x) {
  x = reduce(x);
  if (x > HALF_PI) x = PI - x;
  else if (x < -HALF_PI) x = -PI - x;
  if (x > QUARTER_PI) return cosPoly(HALF_PI - x);
  if (x < -QUARTER_PI) return -cosPoly(HALF_PI + x);
  return sinPoly(x);
}

export function cos_(x) {
  x = reduce(x);
  let neg = false;
  if (x > HALF_PI) { x = PI - x; neg = true; }
  else if (x < -HALF_PI) { x = -PI - x; neg = true; }
  let r;
  if (x > QUARTER_PI) r = sinPoly(HALF_PI - x);
  else if (x < -QUARTER_PI) r = sinPoly(HALF_PI + x);
  else r = cosPoly(x);
  return neg ? -r : r;
}

function atanUnit(t) {
  t = t / (1.0 + Math.sqrt(1.0 + t * t));
  t = t / (1.0 + Math.sqrt(1.0 + t * t));
  const t2 = t * t;
  const p = t * (1.0 + t2 * (-1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (-1.0 / 7.0 + t2 * (1.0 / 9.0 + t2 * (-1.0 / 11.0 + t2 * (1.0 / 13.0 + t2 * (-1.0 / 15.0 + t2 * (1.0 / 17.0 + t2 * (-1.0 / 19.0 + t2 * (1.0 / 21.0)))))))))));
  return 4.0 * p;
}

export function atan2_(y, x) {
  const ax = Math.abs(x), ay = Math.abs(y);
  if (ax === 0.0 && ay === 0.0) return 0.0;
  let a;
  if (ax >= ay) a = atanUnit(ay / ax);
  else a = HALF_PI - atanUnit(ax / ay);
  if (x < 0.0) a = PI - a;
  if (y < 0.0) a = -a;
  return a;
}

export function asin_(x) {
  if (x > 1.0) x = 1.0;
  else if (x < -1.0) x = -1.0;
  return atan2_(x, Math.sqrt(1.0 - x * x));
}

export function wrap_(a) { return reduce(a); }

function clamp(x, lo, hi) { return x < lo ? lo : (x > hi ? hi : x); }

// ---------------------------------------------------------------------------------------------
// Controls.

export const CONTROL_KEYS = ["aL", "aR", "betaL", "betaR", "sL", "sR", "f"];

export function hoverTrim() {
  return { aL: 1.0, aR: 1.0, betaL: 0.0, betaR: 0.0, sL: 0.0, sR: 0.0, f: F_NOMINAL };
}

// ---------------------------------------------------------------------------------------------
// The rigid body.

export class Flight {
  constructor(position = [2.0, 1.5, 1.2], heading = 0.0) {
    this.p = [position[0], position[1], position[2]];
    this.v = [0.0, 0.0, 0.0];
    this.q = [cos_(heading * 0.5), 0.0, 0.0, sin_(heading * 0.5)];
    this.w = [0.0, 0.0, 0.0];
    this.phase = 0.0;
    this.wind = [0.0, 0.0, 0.0];
    this.t = 0.0;
    this.steps = 0;
    this.touching = 0;
    this.onFloor = false;
    this.landed = false;
  }

  // The body-to-world rotation matrix, row major.
  rotation() {
    const [w, x, y, z] = this.q;
    return [
      1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
      2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
      2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
    ];
  }

  // [roll, pitch, yaw]: yaw about world z, then pitch about body y, then roll about body x.
  // Pitch is positive nose down, the sense of a positive rate about body y.
  euler() {
    const R = this.rotation();
    return [atan2_(R[7], R[8]), asin_(-R[6]), atan2_(R[3], R[0])];
  }

  setAttitude(roll, pitch, yaw) {
    const cr = cos_(roll * 0.5), sr = sin_(roll * 0.5);
    const cp = cos_(pitch * 0.5), sp = sin_(pitch * 0.5);
    const cy = cos_(yaw * 0.5), sy = sin_(yaw * 0.5);
    this.q = [
      cy * cp * cr + sy * sp * sr,
      cy * cp * sr - sy * sp * cr,
      cy * sp * cr + sy * cp * sr,
      sy * cp * cr - cy * sp * sr,
    ];
  }

  speed() {
    const v = this.v;
    return Math.sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  // A hit: a velocity change dv (m/s, world), a gust (m/s, world) that decays, body rates spin.
  swat(dv, gust, spin) {
    this.v[0] += dv[0]; this.v[1] += dv[1]; this.v[2] += dv[2];
    this.wind[0] += gust[0]; this.wind[1] += gust[1]; this.wind[2] += gust[2];
    this.w[0] += spin[0]; this.w[1] += spin[1]; this.w[2] += spin[2];
  }

  kick(wx, wy, wz) {
    this.w[0] += wx; this.w[1] += wy; this.w[2] += wz;
  }

  step(dt, c) {
    const aL = clamp(c.aL, 0.0, A_MAX), aR = clamp(c.aR, 0.0, A_MAX);
    const bL = clamp(c.betaL, -BETA_MAX, BETA_MAX), bR = clamp(c.betaR, -BETA_MAX, BETA_MAX);
    const sL = clamp(c.sL, -S_MAX, S_MAX), sR = clamp(c.sR, -S_MAX, S_MAX);
    const f = clamp(c.f, 0.0, F_MAX);
    const fr = f / F_NOMINAL;
    const fr2 = fr * fr;
    const p = this.p, v = this.v, w = this.w, wind = this.wind;
    const R = this.rotation();

    // velocity of the body through the air, in the body frame: R^T (v - wind)
    const ux = v[0] - wind[0];
    const uy = v[1] - wind[1];
    const uz = v[2] - wind[2];
    const vbx = R[0] * ux + R[3] * uy + R[6] * uz;
    const vby = R[1] * ux + R[4] * uy + R[7] * uz;
    const vbz = R[2] * ux + R[5] * uy + R[8] * uz;

    let Fx = 0.0, Fy = 0.0, Fz = 0.0, Tx = 0.0, Ty = 0.0, Tz = 0.0;
    const wing = (a, beta, s, ry) => {
      const rx = s;
      const rz = HINGE_Z;
      const Fw = F_HOVER * fr2 * (a * a);
      // hinge velocity through the air: v_body + omega x r
      const hx = vbx + (w[1] * rz - w[2] * ry);
      const hy = vby + (w[2] * rx - w[0] * rz);
      const hz = vbz + (w[0] * ry - w[1] * rx);
      const kd = K_WING * a * fr;
      const fx = Fw * sin_(beta) - kd * hx;
      const fy = -kd * hy;
      const fz = Fw * cos_(beta) - kd * hz;
      Fx += fx;
      Fy += fy;
      Fz += fz;
      Tx += ry * fz - rz * fy;
      Ty += rz * fx - rx * fz;
      Tz += rx * fy - ry * fx;
    };
    wing(aL, bL, sL, HINGE_Y);
    wing(aR, bR, sR, -HINGE_Y);

    // flapping counter-torque damping
    const am = (aL + aR) * 0.5;
    Tx -= K_FCT[0] * fr * am * w[0];
    Ty -= K_FCT[1] * fr * am * w[1];
    Tz -= K_FCT[2] * fr * am * w[2];

    // body forces to the world frame, then drag and gravity
    let Wx = R[0] * Fx + R[1] * Fy + R[2] * Fz;
    let Wy = R[3] * Fx + R[4] * Fy + R[5] * Fz;
    let Wz = R[6] * Fx + R[7] * Fy + R[8] * Fz;
    const speed = Math.sqrt(ux * ux + uy * uy + uz * uz);
    const drag = 0.5 * RHO * C_D * AREA * speed;
    Wx -= drag * ux;
    Wy -= drag * uy;
    Wz -= drag * uz;
    Wz -= MASS * GRAVITY;

    // contacts: a sphere against the six planes of the room
    let touching = 0;
    let onFloor = false;
    for (let axis = 0; axis < 3; axis++) {
      const lo = 0.0, size = ROOM[axis];
      for (const side of [1.0, -1.0]) {
        const d = side > 0.0 ? RADIUS - (p[axis] - lo) : p[axis] + RADIUS - size;
        if (d <= 0.0) continue;
        touching += 1;
        if (axis === 2 && side > 0.0) onFloor = true;
        const vn = v[axis] * side;
        let N = K_CONTACT * d - C_CONTACT * vn;
        if (N < 0.0) N = 0.0;
        const tx = v[0] - (axis === 0 ? vn * side : 0.0);
        const ty = v[1] - (axis === 1 ? vn * side : 0.0);
        const tz = v[2] - (axis === 2 ? vn * side : 0.0);
        const vt = Math.sqrt(tx * tx + ty * ty + tz * tz);
        if (vt > 0.0) {
          let fm = MU * N;
          const slide = C_SLIDE * vt;
          if (slide < fm) fm = slide;
          const k = fm / vt;
          Wx -= k * tx;
          Wy -= k * ty;
          Wz -= k * tz;
        }
        if (axis === 0) Wx += N * side;
        else if (axis === 1) Wy += N * side;
        else Wz += N * side;
      }
    }
    if (touching > 0) {
      Tx -= C_SPIN * w[0];
      Ty -= C_SPIN * w[1];
      Tz -= C_SPIN * w[2];
    }

    // semi-implicit Euler: velocity first, then position
    v[0] += dt * (Wx / MASS);
    v[1] += dt * (Wy / MASS);
    v[2] += dt * (Wz / MASS);
    p[0] += dt * v[0];
    p[1] += dt * v[1];
    p[2] += dt * v[2];

    // Euler's equation in the body frame, with the gyroscopic term
    const Ix = INERTIA[0], Iy = INERTIA[1], Iz = INERTIA[2];
    const Lx = Ix * w[0];
    const Ly = Iy * w[1];
    const Lz = Iz * w[2];
    w[0] += dt * ((Tx - (w[1] * Lz - w[2] * Ly)) / Ix);
    w[1] += dt * ((Ty - (w[2] * Lx - w[0] * Lz)) / Iy);
    w[2] += dt * ((Tz - (w[0] * Ly - w[1] * Lx)) / Iz);

    // quaternion: q += dt/2 q (0, omega), then renormalised
    const [qw, qx, qy, qz] = this.q;
    const h = 0.5 * dt;
    const nw = qw + h * (-qx * w[0] - qy * w[1] - qz * w[2]);
    const nx = qx + h * (qw * w[0] + qy * w[2] - qz * w[1]);
    const ny = qy + h * (qw * w[1] - qx * w[2] + qz * w[0]);
    const nz = qz + h * (qw * w[2] + qx * w[1] - qy * w[0]);
    const n = Math.sqrt(nw * nw + nx * nx + ny * ny + nz * nz);
    this.q = [nw / n, nx / n, ny / n, nz / n];

    // wingbeat phase, for drawing
    this.phase += dt * (TWO_PI * f);
    if (this.phase > TWO_PI) this.phase -= TWO_PI;

    // the gust decays
    const decay = 1.0 - dt / WIND_TAU;
    wind[0] *= decay;
    wind[1] *= decay;
    wind[2] *= decay;

    this.t += dt;
    this.steps += 1;
    this.touching = touching;
    this.onFloor = onFloor;
    this.landed = onFloor && this.speed() < 1.0e-3;
  }
}

// ---------------------------------------------------------------------------------------------
// The hand-written pilot: the control condition of the demo. See HandPilot in body.py.

export class HandPilot {
  static TILT_MAX = 0.5235987755982988;  // rad, 30 degrees of body tilt
  static K_POS = 4.0;        // 1/s, target speed per metre of position error
  static K_VEL = 12.0;       // 1/s, acceleration per m/s of velocity error
  static V_CLIMB = 0.5;      // m/s, climb and descent ceiling
  static K_ATT = 12000.0;    // 1/s^2, angular acceleration per radian of roll or pitch error
  static K_RATE = 170.0;     // 1/s, angular acceleration per rad/s of roll or pitch rate
  static K_HEAD = 20000.0;   // 1/s^2, angular acceleration per radian of heading error
  static K_YAW = 160.0;      // 1/s, angular acceleration per rad/s of yaw rate
  static A_CRUISE = 1.15;    // the amplitude above which the wingbeat frequency rises
  static ARRIVE = 0.08;      // m, a waypoint counts as reached inside this radius

  constructor(target = [2.0, 1.5, 1.2], heading = 0.0, speed = 0.3) {
    this.target = [target[0], target[1], target[2]];
    this.heading = heading;
    this.speed = speed;
    this.route = null;
    this.leg = 0;
    this.last = hoverTrim();
  }

  saccade(angle) { this.heading = wrap_(this.heading + angle); }

  // Waypoints flown in a cycle, the first one the current target.
  flyRoute(waypoints) {
    this.route = waypoints.map((w) => [w[0], w[1], w[2]]);
    this.leg = 0;
    this.target = this.route[0].slice();
  }

  controls(fl) {
    const P = HandPilot;
    const p = fl.p, v = fl.v, w = fl.w;
    const R = fl.rotation();
    const roll = atan2_(R[7], R[8]);
    const pitch = asin_(-R[6]);
    const yaw = atan2_(R[3], R[0]);

    let ex = this.target[0] - p[0];
    let ey = this.target[1] - p[1];
    let ez = this.target[2] - p[2];
    if (this.route !== null && Math.sqrt(ex * ex + ey * ey + ez * ez) < P.ARRIVE) {
      this.leg = (this.leg + 1) % this.route.length;
      this.target = this.route[this.leg].slice();
      ex = this.target[0] - p[0];
      ey = this.target[1] - p[1];
      ez = this.target[2] - p[2];
      this.saccade(wrap_(atan2_(ey, ex) - this.heading));
    }

    // outer loop: target velocity, then acceleration
    let vdx = P.K_POS * ex;
    let vdy = P.K_POS * ey;
    const hm = Math.sqrt(vdx * vdx + vdy * vdy);
    if (hm > this.speed) {
      vdx *= this.speed / hm;
      vdy *= this.speed / hm;
    }
    const vdz = clamp(P.K_POS * ez, -P.V_CLIMB, P.V_CLIMB);
    const ax = P.K_VEL * (vdx - v[0]);
    const ay = P.K_VEL * (vdy - v[1]);
    const az = P.K_VEL * (vdz - v[2]);

    // forces in the heading frame, with the wing damping fed forward
    const cy = cos_(yaw);
    const sy = sin_(yaw);
    const af = cy * ax + sy * ay;
    const al = -sy * ax + cy * ay;
    const ux = v[0] - fl.wind[0];
    const uy = v[1] - fl.wind[1];
    const uz = v[2] - fl.wind[2];
    const vbx = R[0] * ux + R[3] * uy + R[6] * uz;
    const vby = R[1] * ux + R[4] * uy + R[7] * uz;
    const vbz = R[2] * ux + R[5] * uy + R[8] * uz;
    // the wing damping of the last step, fed forward as force and moment; the moment is
    // the nose-up pitching of forward flight that the pilot has to hold against
    const last = this.last;
    const fr = last.f / F_NOMINAL;
    const kmean = K_WING * ((last.aL + last.aR) * 0.5) * fr;
    let tdx = 0.0, tdy = 0.0, tdz = 0.0;
    for (const [a, s, ry] of [[last.aL, last.sL, HINGE_Y], [last.aR, last.sR, -HINGE_Y]]) {
      const rx = s;
      const rz = HINGE_Z;
      const hx = vbx + (w[1] * rz - w[2] * ry);
      const hy = vby + (w[2] * rx - w[0] * rz);
      const hz = vbz + (w[0] * ry - w[1] * rx);
      const kd = K_WING * a * fr;
      const dx = -kd * hx;
      const dy = -kd * hy;
      const dz = -kd * hz;
      tdx += ry * dz - rz * dy;
      tdy += rz * dx - rx * dz;
      tdz += rx * dy - ry * dx;
    }
    // the drag of the last step fed forward in the heading frame
    const uf = cy * ux + sy * uy;
    const ul = -sy * ux + cy * uy;
    const drag = 2.0 * kmean + 0.5 * RHO * C_D * AREA * Math.sqrt(ux * ux + uy * uy + uz * uz);
    const ff = MASS * af + drag * uf;
    const fl_ = MASS * al + drag * ul;
    let fz = MASS * (GRAVITY + az) + drag * uz;
    if (fz < 0.2 * MASS * GRAVITY) fz = 0.2 * MASS * GRAVITY;

    // thrust direction: body tilt first, the forward remainder to the stroke plane
    const gamma = atan2_(ff, fz);
    const pitchCmd = clamp(gamma, -P.TILT_MAX, P.TILT_MAX);
    const beta0 = clamp(gamma - pitch, -BETA_MAX, BETA_MAX);
    const rollCmd = clamp(-atan2_(fl_, fz), -P.TILT_MAX, P.TILT_MAX);
    let ct = cos_(pitch + beta0) * cos_(roll);
    if (ct < 0.5) ct = 0.5;
    const thrust = fz / ct;

    // inner loop: torques
    const tx = INERTIA[0] * (P.K_ATT * (rollCmd - roll) - P.K_RATE * w[0]) - tdx;
    const ty = INERTIA[1] * (P.K_ATT * (pitchCmd - pitch) - P.K_RATE * w[1]) - tdy;
    const tz = INERTIA[2] * (P.K_HEAD * wrap_(this.heading - yaw) - P.K_YAW * w[2]) - tdz;

    // allocation: thrust and roll to the two wing forces
    const F = thrust * 0.5;
    const delta = clamp(tx / (2.0 * F * HINGE_Y), -0.6, 0.6);
    const FL = F * (1.0 + delta);
    const FR = F * (1.0 - delta);
    // the wingbeat frequency rises when the amplitude would exceed A_CRUISE
    const Fbig = FL > FR ? FL : FR;
    let f = F_NOMINAL;
    const need = Fbig / (F_HOVER * (P.A_CRUISE * P.A_CRUISE));
    if (need > 1.0) f = clamp(F_NOMINAL * Math.sqrt(need), F_NOMINAL, F_MAX);
    const fr2 = (f / F_NOMINAL) * (f / F_NOMINAL);
    const aL = clamp(Math.sqrt(FL / (F_HOVER * fr2)), 0.0, A_MAX);
    const aR = clamp(Math.sqrt(FR / (F_HOVER * fr2)), 0.0, A_MAX);
    // yaw from differential stroke-plane tilt
    const bd = asin_(clamp(-tz / (2.0 * HINGE_Y * F * cos_(beta0)), -1.0, 1.0));
    const bL = clamp(beta0 + bd, -BETA_MAX, BETA_MAX);
    const bR = clamp(beta0 - bd, -BETA_MAX, BETA_MAX);
    // pitch from the mean stroke shift, with the thrust above the centre of mass compensated
    const s = clamp((HINGE_Z * (FL * sin_(bL) + FR * sin_(bR)) - ty) / (FL * cos_(bL) + FR * cos_(bR)), -S_MAX, S_MAX);

    this.last = { aL, aR, betaL: bL, betaR: bR, sL: s, sR: s, f };
    return this.last;
  }
}

// ---------------------------------------------------------------------------------------------

export const LAP_ROUTE = [[1.0, 0.8, 1.3], [3.0, 0.8, 1.3], [3.0, 2.2, 1.3], [1.0, 2.2, 1.3]];

// Laps of a rectangle of waypoints around the room. Returns samples every `every` steps as
// [t, position, quaternion, velocity, controls].
export function flyLaps(seconds, flight = null, pilot = null, every = 20, speed = 0.3) {
  const fl = flight !== null ? flight : new Flight([1.0, 0.8, 1.3], 0.0);
  const pt = pilot !== null ? pilot : new HandPilot(LAP_ROUTE[0], 0.0, speed);
  pt.flyRoute(LAP_ROUTE);
  pt.leg = 0;
  const out = [];
  const n = Math.round(seconds / DT);
  for (let k = 0; k < n; k++) {
    const c = pt.controls(fl);
    fl.step(DT, c);
    if ((k + 1) % every === 0) out.push([fl.t, fl.p.slice(), fl.q.slice(), fl.v.slice(), { ...c }]);
  }
  return out;
}
