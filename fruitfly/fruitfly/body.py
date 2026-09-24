"""The body of the fly: a rigid body with stroke-averaged wing aerodynamics, in a room.

Frames. World: x along the long wall, y along the short wall, z up, origin at a floor corner.
Body: x forward through the head, y left, z up. The quaternion q = (w, x, y, z) rotates body
vectors into the world.

State: position (m), velocity (m/s, world), quaternion, angular velocity (rad/s, body frame),
wingbeat phase (rad, for drawing only) and the wind (m/s, world), a gust that decays.

Controls, per wing (left, right): stroke amplitude a in [0, 1.3], with 1 the hover amplitude;
stroke-plane tilt beta in [-0.5, 0.5] rad, positive tilting the wing force forward; mean stroke
shift s in [-0.5, 0.5] mm, moving the point of application of the wing force along x; and one
wingbeat frequency f in [0, 250] Hz for both wings.

Forces, in the body frame, each applied at its wing hinge shifted by s along x. The wing force
F_i = F_HOVER (f / 200)^2 a_i^2 (sin beta_i, 0, cos beta_i). The wing damping force
D_i = -K_WING a_i (f / 200) v_rel with v_rel the velocity of the hinge through the air. The hinge
sits above the centre of mass, so forward speed turns into a pitching moment: this makes hover
unstable open loop, as in the real fly. Body drag, gravity and a flapping-counter-torque damping
-K_FCT (f / 200) mean(a) omega complete the model.

Integrator: semi-implicit Euler at DT = 0.5 ms, quaternion renormalised every step. Every
arithmetic operation has a twin in web/body.js in the same order. Python and JavaScript share
sqrt and the four operations bit for bit and differ by one unit in the last place on the
library sine, cosine and arctangent, so this module carries its own polynomials for those.
"""

from __future__ import annotations

from math import floor, sqrt

__all__ = [
    "MASS", "GRAVITY", "INERTIA", "F_NOMINAL", "F_HOVER", "HINGE_Y", "HINGE_Z", "K_WING", "K_FCT",
    "RHO", "C_D", "AREA", "RADIUS", "ROOM", "DT", "A_MAX", "BETA_MAX", "S_MAX", "F_MAX",
    "hover_trim", "Flight", "HandPilot", "fly_laps", "sin_", "cos_", "atan2_", "asin_", "wrap_",
]

# ---------------------------------------------------------------------------------------------
# Constants. web/body.js carries the same list with the same values.

MASS = 1.0e-6                          # kg, body mass: about one milligram
GRAVITY = 9.81                         # m/s^2
INERTIA = (1.25e-13, 5.8e-13, 5.8e-13)  # kg m^2, about body x, y, z: a 2.5 mm by 0.5 mm cylinder
F_NOMINAL = 200.0                      # Hz, nominal wingbeat frequency
F_HOVER = MASS * GRAVITY / 2.0         # N, wing force per wing at hover trim: half the weight
HINGE_Y = 0.6e-3                       # m, the wing hinges sit 0.6 mm left and right of the axis
HINGE_Z = 0.3e-3                       # m, and 0.3 mm above the centre of mass
K_WING = 6.0e-6                        # N s/m, wing damping per wing at hover amplitude and 200 Hz
K_FCT = (1.0e-11, 2.0e-12, 5.0e-11)    # N m s/rad, flapping counter-torque damping about x, y, z
RHO = 1.2                              # kg/m^3, air density
C_D = 1.0                              # body drag coefficient
AREA = 2.0e-6                          # m^2, body reference area
RADIUS = 1.2e-3                        # m, the contact sphere of the body
ROOM = (4.0, 3.0, 2.6)                 # m, the room: x by y by z
K_CONTACT = 1.0                        # N/m, contact spring against floor, walls and ceiling
C_CONTACT = 2.0e-3                     # N s/m, contact damping along the surface normal
MU = 0.6                               # friction coefficient on a surface
C_SLIDE = 2.0e-3                       # N s/m, viscous friction below the Coulomb limit
C_SPIN = 2.0e-11                       # N m s/rad, spin damping while touching a surface (the legs)
WIND_TAU = 0.05                        # s, decay time of a gust
DT = 0.0005                            # s, physics step: 2000 Hz
A_MAX = 1.3                            # stroke amplitude ceiling, 1 is hover
BETA_MAX = 0.5                         # rad, stroke-plane tilt ceiling
S_MAX = 0.5e-3                         # m, mean stroke shift ceiling
F_MAX = 250.0                          # Hz, wingbeat frequency ceiling

# Numbers the model holds to: mass about 1 mg; wingbeat about 200 Hz; hover force equals the
# weight; cruise 0.2 to 0.5 m/s with bursts near 1 m/s; a 90 degree saccade in about 50 ms; an
# open-loop pitch instability that grows within a few wingbeats.

PI = 3.141592653589793
TWO_PI = 6.283185307179586
HALF_PI = 1.5707963267948966
QUARTER_PI = 0.7853981633974483

# ---------------------------------------------------------------------------------------------
# Exact-twin arithmetic: sine, cosine and arctangent from products, sums and square roots only.

_S3 = -1.0 / 6.0
_S5 = 1.0 / 120.0
_S7 = -1.0 / 5040.0
_S9 = 1.0 / 362880.0
_S11 = -1.0 / 39916800.0
_S13 = 1.0 / 6227020800.0
_S15 = -1.0 / 1307674368000.0
_S17 = 1.0 / 355687428096000.0
_C2 = -1.0 / 2.0
_C4 = 1.0 / 24.0
_C6 = -1.0 / 720.0
_C8 = 1.0 / 40320.0
_C10 = -1.0 / 3628800.0
_C12 = 1.0 / 479001600.0
_C14 = -1.0 / 87178291200.0
_C16 = 1.0 / 20922789888000.0


def _sin_poly(x):
    """Taylor sine on [-pi/4, pi/4], Horner form."""
    x2 = x * x
    return x * (1.0 + x2 * (_S3 + x2 * (_S5 + x2 * (_S7 + x2 * (_S9 + x2 * (_S11 + x2 * (_S13 + x2 * (_S15 + x2 * _S17))))))))


def _cos_poly(x):
    """Taylor cosine on [-pi/4, pi/4], Horner form."""
    x2 = x * x
    return 1.0 + x2 * (_C2 + x2 * (_C4 + x2 * (_C6 + x2 * (_C8 + x2 * (_C10 + x2 * (_C12 + x2 * (_C14 + x2 * _C16)))))))


def _reduce(x):
    """x brought into [-pi, pi]."""
    k = floor(x / TWO_PI + 0.5)
    return x - k * TWO_PI


def sin_(x):
    x = _reduce(x)
    if x > HALF_PI:
        x = PI - x
    elif x < -HALF_PI:
        x = -PI - x
    if x > QUARTER_PI:
        return _cos_poly(HALF_PI - x)
    if x < -QUARTER_PI:
        return -_cos_poly(HALF_PI + x)
    return _sin_poly(x)


def cos_(x):
    x = _reduce(x)
    neg = False
    if x > HALF_PI:
        x = PI - x
        neg = True
    elif x < -HALF_PI:
        x = -PI - x
        neg = True
    if x > QUARTER_PI:
        r = _sin_poly(HALF_PI - x)
    elif x < -QUARTER_PI:
        r = _sin_poly(HALF_PI + x)
    else:
        r = _cos_poly(x)
    return -r if neg else r


def _atan_unit(t):
    """Arctangent for 0 <= t <= 1: the argument halved twice, then a Taylor series."""
    t = t / (1.0 + sqrt(1.0 + t * t))
    t = t / (1.0 + sqrt(1.0 + t * t))
    t2 = t * t
    p = t * (1.0 + t2 * (-1.0 / 3.0 + t2 * (1.0 / 5.0 + t2 * (-1.0 / 7.0 + t2 * (1.0 / 9.0 + t2 * (-1.0 / 11.0 + t2 * (1.0 / 13.0 + t2 * (-1.0 / 15.0 + t2 * (1.0 / 17.0 + t2 * (-1.0 / 19.0 + t2 * (1.0 / 21.0)))))))))))
    return 4.0 * p


def atan2_(y, x):
    ax = abs(x)
    ay = abs(y)
    if ax == 0.0 and ay == 0.0:
        return 0.0
    if ax >= ay:
        a = _atan_unit(ay / ax)
    else:
        a = HALF_PI - _atan_unit(ax / ay)
    if x < 0.0:
        a = PI - a
    if y < 0.0:
        a = -a
    return a


def asin_(x):
    if x > 1.0:
        x = 1.0
    elif x < -1.0:
        x = -1.0
    return atan2_(x, sqrt(1.0 - x * x))


def wrap_(a):
    """An angle brought into [-pi, pi]."""
    return _reduce(a)


def _clamp(x, lo, hi):
    return lo if x < lo else (hi if x > hi else x)


# ---------------------------------------------------------------------------------------------
# Controls.

CONTROL_KEYS = ("aL", "aR", "betaL", "betaR", "sL", "sR", "f")


def hover_trim():
    """The controls that balance the weight with the body level and still."""
    return {"aL": 1.0, "aR": 1.0, "betaL": 0.0, "betaR": 0.0, "sL": 0.0, "sR": 0.0, "f": F_NOMINAL}


def _clamp_controls(c):
    return (
        _clamp(c["aL"], 0.0, A_MAX), _clamp(c["aR"], 0.0, A_MAX),
        _clamp(c["betaL"], -BETA_MAX, BETA_MAX), _clamp(c["betaR"], -BETA_MAX, BETA_MAX),
        _clamp(c["sL"], -S_MAX, S_MAX), _clamp(c["sR"], -S_MAX, S_MAX),
        _clamp(c["f"], 0.0, F_MAX),
    )


# ---------------------------------------------------------------------------------------------
# The rigid body.

class Flight:
    """One fly in the room. Positions in metres, the body a sphere of RADIUS for contacts."""

    def __init__(self, position=(2.0, 1.5, 1.2), heading=0.0):
        self.p = [position[0], position[1], position[2]]
        self.v = [0.0, 0.0, 0.0]
        self.q = [cos_(heading * 0.5), 0.0, 0.0, sin_(heading * 0.5)]
        self.w = [0.0, 0.0, 0.0]
        self.phase = 0.0
        self.wind = [0.0, 0.0, 0.0]
        self.t = 0.0
        self.steps = 0
        self.touching = 0          # number of surfaces in contact after the last step
        self.on_floor = False
        self.landed = False        # on the floor and at rest

    # -- orientation -------------------------------------------------------------------------

    def rotation(self):
        """The body-to-world rotation matrix, row major."""
        w, x, y, z = self.q
        return (
            1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - w * z), 2.0 * (x * z + w * y),
            2.0 * (x * y + w * z), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - w * x),
            2.0 * (x * z - w * y), 2.0 * (y * z + w * x), 1.0 - 2.0 * (x * x + y * y),
        )

    def euler(self):
        """(roll, pitch, yaw) in radians: yaw about world z, then pitch about body y, then roll
        about body x. Pitch is positive nose down, the sense of a positive rate about body y."""
        R = self.rotation()
        return (atan2_(R[7], R[8]), asin_(-R[6]), atan2_(R[3], R[0]))

    def set_attitude(self, roll, pitch, yaw):
        """The quaternion from the same Euler angles as euler() returns."""
        cr, sr = cos_(roll * 0.5), sin_(roll * 0.5)
        cp, sp = cos_(pitch * 0.5), sin_(pitch * 0.5)
        cy, sy = cos_(yaw * 0.5), sin_(yaw * 0.5)
        self.q = [
            cy * cp * cr + sy * sp * sr,
            cy * cp * sr - sy * sp * cr,
            cy * sp * cr + sy * cp * sr,
            sy * cp * cr - cy * sp * sr,
        ]

    def speed(self):
        v = self.v
        return sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2])

    # -- disturbances ------------------------------------------------------------------------

    def swat(self, dv, gust, spin):
        """A hit: a velocity change dv (m/s, world), a gust (m/s, world) that decays, and body
        rates spin (rad/s) added to the angular velocity."""
        self.v[0] += dv[0]
        self.v[1] += dv[1]
        self.v[2] += dv[2]
        self.wind[0] += gust[0]
        self.wind[1] += gust[1]
        self.wind[2] += gust[2]
        self.w[0] += spin[0]
        self.w[1] += spin[1]
        self.w[2] += spin[2]

    def kick(self, wx, wy, wz):
        """Body rates added to the angular velocity."""
        self.w[0] += wx
        self.w[1] += wy
        self.w[2] += wz

    # -- one step ----------------------------------------------------------------------------

    def step(self, dt, controls):
        aL, aR, bL, bR, sL, sR, f = _clamp_controls(controls)
        fr = f / F_NOMINAL
        fr2 = fr * fr
        p, v, w, wind = self.p, self.v, self.w, self.wind
        R = self.rotation()

        # velocity of the body through the air, in the body frame: R^T (v - wind)
        ux = v[0] - wind[0]
        uy = v[1] - wind[1]
        uz = v[2] - wind[2]
        vbx = R[0] * ux + R[3] * uy + R[6] * uz
        vby = R[1] * ux + R[4] * uy + R[7] * uz
        vbz = R[2] * ux + R[5] * uy + R[8] * uz

        Fx = 0.0
        Fy = 0.0
        Fz = 0.0
        Tx = 0.0
        Ty = 0.0
        Tz = 0.0
        for a, beta, s, ry in ((aL, bL, sL, HINGE_Y), (aR, bR, sR, -HINGE_Y)):
            rx = s
            rz = HINGE_Z
            Fw = F_HOVER * fr2 * (a * a)
            # hinge velocity through the air: v_body + omega x r
            hx = vbx + (w[1] * rz - w[2] * ry)
            hy = vby + (w[2] * rx - w[0] * rz)
            hz = vbz + (w[0] * ry - w[1] * rx)
            kd = K_WING * a * fr
            fx = Fw * sin_(beta) - kd * hx
            fy = -kd * hy
            fz = Fw * cos_(beta) - kd * hz
            Fx += fx
            Fy += fy
            Fz += fz
            Tx += ry * fz - rz * fy
            Ty += rz * fx - rx * fz
            Tz += rx * fy - ry * fx

        # flapping counter-torque damping
        am = (aL + aR) * 0.5
        Tx -= K_FCT[0] * fr * am * w[0]
        Ty -= K_FCT[1] * fr * am * w[1]
        Tz -= K_FCT[2] * fr * am * w[2]

        # body forces to the world frame, then drag and gravity
        Wx = R[0] * Fx + R[1] * Fy + R[2] * Fz
        Wy = R[3] * Fx + R[4] * Fy + R[5] * Fz
        Wz = R[6] * Fx + R[7] * Fy + R[8] * Fz
        speed = sqrt(ux * ux + uy * uy + uz * uz)
        drag = 0.5 * RHO * C_D * AREA * speed
        Wx -= drag * ux
        Wy -= drag * uy
        Wz -= drag * uz
        Wz -= MASS * GRAVITY

        # contacts: a sphere against the six planes of the room
        touching = 0
        on_floor = False
        for axis, lo, size in ((0, 0.0, ROOM[0]), (1, 0.0, ROOM[1]), (2, 0.0, ROOM[2])):
            for side in (1.0, -1.0):
                d = RADIUS - (p[axis] - lo) if side > 0.0 else p[axis] + RADIUS - size
                if d <= 0.0:
                    continue
                touching += 1
                if axis == 2 and side > 0.0:
                    on_floor = True
                vn = v[axis] * side
                N = K_CONTACT * d - C_CONTACT * vn
                if N < 0.0:
                    N = 0.0
                tx = v[0] - (vn * side if axis == 0 else 0.0)
                ty = v[1] - (vn * side if axis == 1 else 0.0)
                tz = v[2] - (vn * side if axis == 2 else 0.0)
                vt = sqrt(tx * tx + ty * ty + tz * tz)
                if vt > 0.0:
                    fm = MU * N
                    slide = C_SLIDE * vt
                    if slide < fm:
                        fm = slide
                    k = fm / vt
                    Wx -= k * tx
                    Wy -= k * ty
                    Wz -= k * tz
                if axis == 0:
                    Wx += N * side
                elif axis == 1:
                    Wy += N * side
                else:
                    Wz += N * side
        if touching > 0:
            Tx -= C_SPIN * w[0]
            Ty -= C_SPIN * w[1]
            Tz -= C_SPIN * w[2]

        # semi-implicit Euler: velocity first, then position
        v[0] += dt * (Wx / MASS)
        v[1] += dt * (Wy / MASS)
        v[2] += dt * (Wz / MASS)
        p[0] += dt * v[0]
        p[1] += dt * v[1]
        p[2] += dt * v[2]

        # Euler's equation in the body frame, with the gyroscopic term
        Ix, Iy, Iz = INERTIA
        Lx = Ix * w[0]
        Ly = Iy * w[1]
        Lz = Iz * w[2]
        w[0] += dt * ((Tx - (w[1] * Lz - w[2] * Ly)) / Ix)
        w[1] += dt * ((Ty - (w[2] * Lx - w[0] * Lz)) / Iy)
        w[2] += dt * ((Tz - (w[0] * Ly - w[1] * Lx)) / Iz)

        # quaternion: q += dt/2 q (0, omega), then renormalised
        qw, qx, qy, qz = self.q
        h = 0.5 * dt
        nw = qw + h * (-qx * w[0] - qy * w[1] - qz * w[2])
        nx = qx + h * (qw * w[0] + qy * w[2] - qz * w[1])
        ny = qy + h * (qw * w[1] - qx * w[2] + qz * w[0])
        nz = qz + h * (qw * w[2] + qx * w[1] - qy * w[0])
        n = sqrt(nw * nw + nx * nx + ny * ny + nz * nz)
        self.q = [nw / n, nx / n, ny / n, nz / n]

        # wingbeat phase, for drawing
        self.phase += dt * (TWO_PI * f)
        if self.phase > TWO_PI:
            self.phase -= TWO_PI

        # the gust decays
        decay = 1.0 - dt / WIND_TAU
        wind[0] *= decay
        wind[1] *= decay
        wind[2] *= decay

        self.t += dt
        self.steps += 1
        self.touching = touching
        self.on_floor = on_floor
        self.landed = on_floor and self.speed() < 1.0e-3


# ---------------------------------------------------------------------------------------------
# The hand-written pilot: the control condition of the demo.

class HandPilot:
    """A PD flight controller that sets the six wing controls from a target point and heading.

    Outer loop: position error to a target velocity, capped at `speed`; velocity error to a
    force. The vertical force sets the thrust. The horizontal force sets a body tilt, capped at
    TILT_MAX, and any forward remainder goes to the stroke-plane tilt. Inner loop: roll, pitch
    and heading errors with rate damping give three torques. The thrust and torques become the
    wing controls by inverting the wing model of Flight: roll from differential amplitude, pitch
    from the mean stroke shift, yaw from differential stroke-plane tilt, altitude from the
    common amplitude, with the wingbeat frequency raised when the amplitude runs out.

    `saccade(angle)` turns the heading target by `angle`; the heading loop saturates the yaw
    torque, which is what makes the turn fast. A `route` of waypoints is flown in a cycle: at
    each waypoint the pilot saccades to the next leg and flies it.
    """

    TILT_MAX = 0.5235987755982988   # rad, 30 degrees of body tilt
    K_POS = 4.0        # 1/s, target speed per metre of position error
    K_VEL = 12.0       # 1/s, acceleration per m/s of velocity error
    V_CLIMB = 0.5      # m/s, climb and descent ceiling
    K_ATT = 12000.0    # 1/s^2, angular acceleration per radian of roll or pitch error
    K_RATE = 170.0     # 1/s, angular acceleration per rad/s of roll or pitch rate
    K_HEAD = 20000.0   # 1/s^2, angular acceleration per radian of heading error
    K_YAW = 160.0      # 1/s, angular acceleration per rad/s of yaw rate
    A_CRUISE = 1.15    # the amplitude above which the wingbeat frequency rises
    ARRIVE = 0.08      # m, a waypoint counts as reached inside this radius

    def __init__(self, target=(2.0, 1.5, 1.2), heading=0.0, speed=0.3):
        self.target = [target[0], target[1], target[2]]
        self.heading = heading
        self.speed = speed
        self.route = None
        self.leg = 0
        self.last = hover_trim()

    def saccade(self, angle):
        self.heading = wrap_(self.heading + angle)

    def fly_route(self, waypoints):
        """Waypoints flown in a cycle, the first one the current target."""
        self.route = [[w[0], w[1], w[2]] for w in waypoints]
        self.leg = 0
        self.target = list(self.route[0])

    def controls(self, fl):
        p, v, w = fl.p, fl.v, fl.w
        R = fl.rotation()
        roll = atan2_(R[7], R[8])
        pitch = asin_(-R[6])
        yaw = atan2_(R[3], R[0])

        ex = self.target[0] - p[0]
        ey = self.target[1] - p[1]
        ez = self.target[2] - p[2]
        if self.route is not None and sqrt(ex * ex + ey * ey + ez * ez) < self.ARRIVE:
            self.leg = (self.leg + 1) % len(self.route)
            self.target = list(self.route[self.leg])
            ex = self.target[0] - p[0]
            ey = self.target[1] - p[1]
            ez = self.target[2] - p[2]
            self.saccade(wrap_(atan2_(ey, ex) - self.heading))

        # outer loop: target velocity, then acceleration
        vdx = self.K_POS * ex
        vdy = self.K_POS * ey
        hm = sqrt(vdx * vdx + vdy * vdy)
        if hm > self.speed:
            vdx *= self.speed / hm
            vdy *= self.speed / hm
        vdz = _clamp(self.K_POS * ez, -self.V_CLIMB, self.V_CLIMB)
        ax = self.K_VEL * (vdx - v[0])
        ay = self.K_VEL * (vdy - v[1])
        az = self.K_VEL * (vdz - v[2])

        # forces in the heading frame, with the wing damping fed forward
        cy = cos_(yaw)
        sy = sin_(yaw)
        af = cy * ax + sy * ay
        al = -sy * ax + cy * ay
        ux = v[0] - fl.wind[0]
        uy = v[1] - fl.wind[1]
        uz = v[2] - fl.wind[2]
        vbx = R[0] * ux + R[3] * uy + R[6] * uz
        vby = R[1] * ux + R[4] * uy + R[7] * uz
        vbz = R[2] * ux + R[5] * uy + R[8] * uz
        # the wing damping of the last step, fed forward as force and moment; the moment is
        # the nose-up pitching of forward flight that the pilot has to hold against
        last = self.last
        fr = last["f"] / F_NOMINAL
        kmean = K_WING * ((last["aL"] + last["aR"]) * 0.5) * fr
        tdx = 0.0
        tdy = 0.0
        tdz = 0.0
        for a, s, ry in ((last["aL"], last["sL"], HINGE_Y), (last["aR"], last["sR"], -HINGE_Y)):
            rx = s
            rz = HINGE_Z
            hx = vbx + (w[1] * rz - w[2] * ry)
            hy = vby + (w[2] * rx - w[0] * rz)
            hz = vbz + (w[0] * ry - w[1] * rx)
            kd = K_WING * a * fr
            dx = -kd * hx
            dy = -kd * hy
            dz = -kd * hz
            tdx += ry * dz - rz * dy
            tdy += rz * dx - rx * dz
            tdz += rx * dy - ry * dx
        # the drag of the last step fed forward in the heading frame
        uf = cy * ux + sy * uy
        ul = -sy * ux + cy * uy
        drag = 2.0 * kmean + 0.5 * RHO * C_D * AREA * sqrt(ux * ux + uy * uy + uz * uz)
        ff = MASS * af + drag * uf
        fl_ = MASS * al + drag * ul
        fz = MASS * (GRAVITY + az) + drag * uz
        if fz < 0.2 * MASS * GRAVITY:
            fz = 0.2 * MASS * GRAVITY

        # thrust direction: body tilt first, the forward remainder to the stroke plane
        gamma = atan2_(ff, fz)
        pitch_cmd = _clamp(gamma, -self.TILT_MAX, self.TILT_MAX)
        beta0 = _clamp(gamma - pitch, -BETA_MAX, BETA_MAX)
        roll_cmd = _clamp(-atan2_(fl_, fz), -self.TILT_MAX, self.TILT_MAX)
        ct = cos_(pitch + beta0) * cos_(roll)
        if ct < 0.5:
            ct = 0.5
        thrust = fz / ct

        # inner loop: torques
        tx = INERTIA[0] * (self.K_ATT * (roll_cmd - roll) - self.K_RATE * w[0]) - tdx
        ty = INERTIA[1] * (self.K_ATT * (pitch_cmd - pitch) - self.K_RATE * w[1]) - tdy
        tz = INERTIA[2] * (self.K_HEAD * wrap_(self.heading - yaw) - self.K_YAW * w[2]) - tdz

        # allocation: thrust and roll to the two wing forces
        F = thrust * 0.5
        delta = _clamp(tx / (2.0 * F * HINGE_Y), -0.6, 0.6)
        FL = F * (1.0 + delta)
        FR = F * (1.0 - delta)
        # the wingbeat frequency rises when the amplitude would exceed A_CRUISE
        Fbig = FL if FL > FR else FR
        f = F_NOMINAL
        need = Fbig / (F_HOVER * (self.A_CRUISE * self.A_CRUISE))
        if need > 1.0:
            f = _clamp(F_NOMINAL * sqrt(need), F_NOMINAL, F_MAX)
        fr2 = (f / F_NOMINAL) * (f / F_NOMINAL)
        aL = _clamp(sqrt(FL / (F_HOVER * fr2)), 0.0, A_MAX)
        aR = _clamp(sqrt(FR / (F_HOVER * fr2)), 0.0, A_MAX)
        # yaw from differential stroke-plane tilt
        bd = asin_(_clamp(-tz / (2.0 * HINGE_Y * F * cos_(beta0)), -1.0, 1.0))
        bL = _clamp(beta0 + bd, -BETA_MAX, BETA_MAX)
        bR = _clamp(beta0 - bd, -BETA_MAX, BETA_MAX)
        # pitch from the mean stroke shift, with the thrust above the centre of mass compensated
        s = _clamp((HINGE_Z * (FL * sin_(bL) + FR * sin_(bR)) - ty) / (FL * cos_(bL) + FR * cos_(bR)), -S_MAX, S_MAX)

        self.last = {"aL": aL, "aR": aR, "betaL": bL, "betaR": bR, "sL": s, "sR": s, "f": f}
        return self.last


# ---------------------------------------------------------------------------------------------

LAP_ROUTE = ((1.0, 0.8, 1.3), (3.0, 0.8, 1.3), (3.0, 2.2, 1.3), (1.0, 2.2, 1.3))


def fly_laps(seconds, flight=None, pilot=None, every=20, speed=0.3):
    """Laps of a rectangle of waypoints around the room. Returns samples every `every` steps as
    (t, position, quaternion, velocity, controls)."""
    fl = flight if flight is not None else Flight((1.0, 0.8, 1.3), 0.0)
    pt = pilot if pilot is not None else HandPilot(LAP_ROUTE[0], 0.0, speed)
    pt.fly_route(LAP_ROUTE)
    pt.leg = 0
    out = []
    n = int(round(seconds / DT))
    for k in range(n):
        c = pt.controls(fl)
        fl.step(DT, c)
        if (k + 1) % every == 0:
            out.append((fl.t, list(fl.p), list(fl.q), list(fl.v), dict(c)))
    return out
