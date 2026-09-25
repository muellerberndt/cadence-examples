// The fly's life in the room: modes, drives, what the room does to the senses, and what the
// brain's descending neurons do to the body. The wingbeat-timescale equilibrium reflex is the
// body's own (the HandPilot's inner loop, as the worm's undulation is the worm's); everything
// here happens at the timescale of a rate model: decisions every DECISION_S of simulated time.
//
// Two layers act on the course. The instinct layer is hand-written and declared (bouts of
// flight with saccades, landing, sitting, grooming, wall avoidance): it is the control condition
// of the demo and the fallback when the brain is quiet. The brain layer reads the settled
// descending neurons (deviations from their level-flight rest) and overrides the course: DNa02
// left and right turn, DNp09 drives forward flight, DNp07 and DNp10 land, the giant fibre
// escapes, MN9 feeds, aDN1 and aDN2 groom. Only the brain layer changes between "brain",
// "shuffled" and "instincts" on the page's switch.
import { HandPilot, hoverTrim, wrap_, RADIUS } from "./body.js";
import { haltereTone, ocelliLR, opticFlowDrive, antennaDrive } from "./senses.js";

export const DECISION_S = 0.1;          // s of simulated time between decisions
export const CRUISE = 0.3;              // m/s
export const ALTITUDE = 1.2;            // m, the preferred height away from surfaces
export const SACCADE_RATE = 0.4;        // per s, the spontaneous baseline (Censi 2013)
export const SACCADE_ANGLE = 93 * Math.PI / 180;  // rad (Muijres 2015)
export const BOUT_MEAN_S = 15;          // s, mean flight bout before a landing is sought (compressed from 30 to 90 s, Houot 2017)
export const SIT_MEAN_S = 6;            // s, mean sit before a voluntary takeoff (compressed)
export const GROOM_S = 2.5;             // s, a grooming episode (Mueller 2019: bouts of 0.15 to 2 s in longer events)
export const FEED_S = 4;                // s, a feeding burst on the fruit
export const WALL_MARGIN = 0.35;        // m, the fly turns away from a wall inside this band
export const ODOUR_SIGMA = 0.5;         // m, the faint plume's width: the smell is noticed from across the table
export const ODOUR_CORE = 0.1;          // m, the core's width: at a fruit its own smell dominates the other's (25 cm away) by 5 to 1
export const ODOUR_CORE_WEIGHT = 0.85;  // the core's share of the concentration at the source; one wide Gaussian put both smells near saturation at either fruit
export const ODOUR_BASELINE = 0.05;     // m, the bilateral sampling baseline (declared, stands in for casting)
export const ODOUR_COMPARISON = 1.5;    // drive per unit of log concentration ratio between the antennae
export const HUNGER_RATE = 1 / 60;      // per s, hunger rises from fed to starved in a minute (compressed)
export const HAND_RADIUS = 0.025;       // m, the hand as a disc for looming
export const LOOM_SATURATION = 4.0;     // rad/s of angular expansion at full looming drive
export const K_TURN = 6.0;              // rad/s of turn per unit of DNa02 asymmetry
export const K_SPEED = 0.6;             // fraction of cruise per unit of DNp09 deviation
export const LAND_LEVEL = 0.15;         // DNp07/DNp10 deviation that asks for a landing
export const ESCAPE_LEVEL = 0.5;        // giant fibre activation that fires an escape
export const FEED_LEVEL = 0.2;          // MN9 activation that feeds when on sugar
export const GROOM_LEVEL = 0.15;        // aDN deviation that grooms when sitting
export const ODOUR_DECISION_REACH = 0.06;  // m, the mushroom body decides once per search when the fly hovers within this of the fruit it is drawn to: there the antennal lobe of the rate model is in its odour state for either smell (below a receptor drive of about 0.4 it answers no odour, and between 0.6 and 0.8 the bread's code sits in a third state) and both output cells are inside their range
export const HOVER_HEIGHT = 0.05;       // m above the fruit at which the drawn fly hovers until the decision
export const SMELL_FLOOR = 0.04;        // concentration at the head below which nothing is smelled
export const ODOUR_TURN = Math.PI / 4;  // rad, the largest supplied turn toward or away from a smell per decision (the arena's)
export const APPETITE = 0.2;            // hunger below which smells do not call
export const LAND_REACH = 0.14;         // m, an approached fruit is landed on inside this distance
export const FRUIT_FOOTPRINT = 0.14;    // m, a fruit with its saucer: a standing fly inside this is on it
export const EPISODE_S = 15;            // s, an odour search that reaches no fruit ends here (the arena's time limit)
export const DOPAMINE_S = 0.8;          // s, how long the dopaminergic neurons show a reward or a punishment
// The fidgets of a real fly's flight, declared with their sources: body saccades come in bursts and
// the straight segments between them vary in speed (van Breugel and Dickinson 2012; Censi et al.
// 2013 give 0.4 to 1 saccade per second); in an odour plume the fly casts crosswind, zigzagging
// (van Breugel and Dickinson 2014); hovering flight drifts and jitters (Muijres et al. 2015).
export const SEARCH_SACCADE_RATE = 1.0; // per s, the saccade rate while a smell is searched (casting)
export const MICRO_SACCADE_RATE = 1.5;  // per s, small heading jumps of 8 to 25 degrees
export const SPEED_MIN = 0.12;          // m/s, the slowest straight segment
export const SPEED_MAX = 0.55;          // m/s, the fastest
export const JITTER = 0.12;             // rad, heading noise per decision
export const ALTITUDE_JITTER = 0.06;    // m per decision, the drift in height

export function mulberry(seed) { let a = seed >>> 0; return () => { a = (a + 0x6d2b79f5) >>> 0; let t = a; t = Math.imul(t ^ (t >>> 15), t | 1); t ^= t + Math.imul(t ^ (t >>> 7), t | 61); return ((t ^ (t >>> 14)) >>> 0) / 4294967296; }; }
const expo = (rng, mean) => -mean * Math.log(1 - rng());

export class Life {
  constructor(flight, room, seed = 1) {
    this.flight = flight; this.room = room; this.rng = mulberry(seed);
    this.mode = "flying";                 // flying | landed | grooming | feeding
    this.pilot = new HandPilot(flight.p.slice(), flight.euler()[2], CRUISE);
    this.heading = flight.euler()[2]; this.speed = CRUISE; this.altitude = ALTITUDE;
    this.hunger = 0.6; this.fatigue = 0.0;
    this.bout = expo(this.rng, BOUT_MEAN_S); this.sit = 0; this.episode = 0;
    this.landing = null;                  // the point the fly descends to
    this.escape = 0;                      // s left of an escape manoeuvre
    this.since = 0;                       // s since the last decision
    this.clock = 0;
    this.events = [];                     // (t, text) for the page's log
    this.lastHand = null; this.loom = 0; this.handDist = Infinity;
    this.controls = hoverTrim();
    this.wingsOff = false;
    this.readouts = {}; this.baseline = null; this.source = "instincts";
    // the two fruits and their odour classes; the sugar sits on one of them (the visitor moves it).
    // The positions are the room's own arrays, which placeFruit mutates in place, so the smell, the
    // descent and the perch follow a fruit wherever the visitor sets it down; a copy here left the
    // fly landing and feeding at the fruit's first place after a move.
    const fr = room.fruits || {};
    this.fruits = {
      banana: { pos: fr.banana ? fr.banana.pos : [room.table.x0 + (room.table.x1 - room.table.x0) / 2 - 0.12, room.table.y0 + (room.table.y1 - room.table.y0) / 2, room.table.z + 0.03], odour: "decaying_fruit", sigma: ODOUR_SIGMA },
      bread: { pos: fr.bread ? fr.bread.pos : [room.table.x0 + (room.table.x1 - room.table.x0) / 2 + 0.13, room.table.y0 + (room.table.y1 - room.table.y0) / 2, room.table.z + 0.03], odour: "yeasty", sigma: ODOUR_SIGMA },
    };
    this.sugar = "banana";
    this.fruit = this.fruits.banana.pos;   // kept for older callers
    this.landingFruit = null;             // { name, dx, dy } while the descent is to a fruit: the target follows the fruit
    // the lessons: odour decisions, the search in progress, rewards and punishments
    this.smelled = null;                  // the fruit whose odour is strongest at the head, or null
    this.smellC = 0; this.smellDrive = 0;   // the concentration at the head, and the receptor drive, of the fruit smelled most
    this.valence = null;                  // what the fly does about the search's smell: { fruit, action: 0 approach | 1 avoid, p, innate }
    this.search = null;                   // { since, fruit, asked, landed } while an odour search is open: one mushroom-body decision per search
    this.wantDecision = false;            // the page reads this at the brain tick and asks the worker
    this.pendingReward = null;            // { reward, done, why } for the page to hand to the worker
    this.reward = 0; this.punish = 0;     // s left of the dopaminergic display
    this.lastP = { banana: null, bread: null };
    this.takeoffAt = -1; this.groomTarget = "head";
    this.perch = null;                    // [x, y, z of the surface] while the legs hold the body on a table or a fruit (the physics knows only the room's walls)
    // the ethogram: counts and durations for the G3b measurement against docs/REAL_FLY.md
    this.ethogram = { saccades: 0, microSaccades: 0, bouts: [], sits: [], grooms: 0, feeds: 0, landings: 0, flying_s: 0, sitting_s: 0, boutStart: 0, sitStart: null, decisions: 0 };
    this.visits = { banana: 0, bread: 0, sweet: 0, punished: 0 };
  }

  setSugar(name) { this.sugar = name; this.log(name ? `sugar on the ${name}` : "no sugar anywhere"); }
  /** The visitor set a fruit down elsewhere (`from` is where it stood). A fly standing on it, or
   *  where it now lands, is shaken off; a descent to it follows it (see step). */
  fruitMoved(name, from) {
    const F = this.fruits[name]; if (!F) return false;
    this.log(`the ${name} is moved`);
    if (this.mode === "flying") return false;
    const [x, y] = this.flight.p, reach = FRUIT_FOOTPRINT;
    const wasOn = from && Math.hypot(x - from[0], y - from[1]) < reach, isUnder = Math.hypot(x - F.pos[0], y - F.pos[1]) < reach;
    if (!wasOn && !isUnder) return false;
    if (this.search) this.closeSearch(0, `the ${name} is taken away`);
    this.takeoff(wasOn ? `shaken off the ${name}` : `startled by the ${name}`);
    return true;
  }
  fruitAt(p, reach = 0.09) { for (const [name, f] of Object.entries(this.fruits)) if (Math.hypot(p[0] - f.pos[0], p[1] - f.pos[1]) < reach && Math.abs(p[2] - f.pos[2]) < 0.03) return name; return null; }

  log(text) { this.events.push([this.clock, text]); if (this.events.length > 40) this.events.shift(); }

  // ---- the senses: what the room does to the afferents ----------------------------------
  odour(source, sigma = ODOUR_SIGMA) {
    const [x, y, z] = this.flight.p, R = this.flight.rotation();
    const nl = [R[1], R[4], R[7]]; // the body's y axis (left) in the world
    // a narrow core in a wide faint plume: 1 at the source, 0.17 at the other fruit, 0.09 at half a metre, 0.02 at a metre
    const at = (px, py, pz) => { const d2 = (px - source[0]) ** 2 + (py - source[1]) ** 2 + (pz - source[2]) ** 2; return ODOUR_CORE_WEIGHT * Math.exp(-d2 / (2 * ODOUR_CORE * ODOUR_CORE)) + (1 - ODOUR_CORE_WEIGHT) * Math.exp(-d2 / (2 * sigma * sigma)); };
    const c = at(x, y, z), cl = at(x + ODOUR_BASELINE * nl[0], y + ODOUR_BASELINE * nl[1], z + ODOUR_BASELINE * nl[2]), cr = at(x - ODOUR_BASELINE * nl[0], y - ODOUR_BASELINE * nl[1], z - ODOUR_BASELINE * nl[2]);
    const ratio = Math.log(cl + 1e-9) - Math.log(cr + 1e-9);
    const gain = 0.5 + this.hunger;      // hunger raises the olfactory drive (Root 2011: sNPF gates ORN sensitivity)
    // the bilateral comparison scales the concentration-driven response, so a distant smell stays faint
    return { left: Math.min(1, gain * c * (1 + ODOUR_COMPARISON * Math.max(0, ratio))), right: Math.min(1, gain * c * (1 + ODOUR_COMPARISON * Math.max(0, -ratio))), c };
  }

  senses(handPos, dt) {
    const f = this.flight, flying = this.mode === "flying";
    const [hl, hr] = haltereTone(flying);
    const [ol, or_] = ocelliLR(f.rotation());
    const flow = [f.w[2], f.w[0], f.w[1]];
    // looming: the rate of angular expansion of the hand as seen from the fly
    let loom = 0;
    if (handPos) {
      const d = Math.hypot(handPos[0] - f.p[0], handPos[1] - f.p[1], handPos[2] - f.p[2]);
      const theta = 2 * Math.atan(HAND_RADIUS / Math.max(d, 0.005));
      if (this.lastHand !== null && dt > 0) loom = Math.max(0, (theta - this.lastHand) / dt) / LOOM_SATURATION;
      this.lastHand = theta; this.handDist = d;
    } else { this.lastHand = null; this.loom = 0; this.handDist = Infinity; }
    this.loom = Math.min(1, 0.7 * this.loom + loom);  // a short memory so a swat is seen across brain steps
    const banana = this.odour(this.fruits.banana.pos), bread = this.odour(this.fruits.bread.pos);
    const on = flying ? null : this.fruitAt(f.p);
    const onFruit = !!on;
    const onSugar = onFruit && on === this.sugar;
    // what is smelled most at the head
    const cb = banana.c, cr = bread.c;
    this.smellC = Math.max(cb, cr); this.smelled = this.smellC < SMELL_FLOOR ? null : (cb >= cr ? "banana" : "bread");
    this.smellDrive = this.smelled ? Math.max((this.smelled === "banana" ? banana : bread).left, (this.smelled === "banana" ? banana : bread).right) : 0;
    const s = {
      "haltere:left": hl, "haltere:right": hr, "ocelli:left": ol, "ocelli:right": or_,
      "lptc:hs:left": opticFlowDrive(flow[0], flow[1], flow[2], "left"), "lptc:vs:left": opticFlowDrive(flow[0], flow[1], flow[2], "left"),
      "lptc:hs:right": opticFlowDrive(flow[0], flow[1], flow[2], "right"), "lptc:vs:right": opticFlowDrive(flow[0], flow[1], flow[2], "right"),
      "jo:C:left": antennaDrive(f.speed()), "jo:C:right": antennaDrive(f.speed()), "jo:E:left": antennaDrive(f.speed()), "jo:E:right": antennaDrive(f.speed()),
      "vis:LC4": this.loom, "vis:LPLC2": this.loom,
      "orn:decaying_fruit:left": banana.left, "orn:decaying_fruit:right": banana.right,
      "orn:yeasty:left": bread.left, "orn:yeasty:right": bread.right,
      "grn:sugar:labellum": onSugar ? 1 : 0, "grn:sugar:front_leg": onSugar ? 1 : 0,
      "leg_touch": flying ? 0 : 0.6,
      "dan:pam": this.reward > 0 ? Math.min(1, this.reward / DOPAMINE_S + 0.3) : 0,     // sugar: the PAM neurons (Burke 2012, Liu 2012)
      "dan:ppl1": this.punish > 0 ? Math.min(1, this.punish / DOPAMINE_S + 0.3) : 0,    // a blow: the PPL1 neurons (Claridge-Chang 2009, Aso 2010)
    };
    this.onFruit = onFruit; this.onSugar = onSugar; this.onWhich = on; this.fruitC = this.smellC;
    return s;
  }

  // ---- the brain's readouts, as deviations from their level-flight rest ---------------------
  setBaseline(readouts) { this.baseline = { ...readouts }; }
  dev(name) { const r = this.readouts[name]; if (r === undefined) return 0; return r - (this.baseline ? (this.baseline[name] || 0) : 0); }

  // ---- decisions ----------------------------------------------------------------------------
  decide(useBrain) {
    const f = this.flight, rng = this.rng, [x, y, z] = f.p, W = 4.0, D = 3.0, H = 2.6;
    if (this.mode === "flying") {
      if (this.escape > 0) return;
      // the instinct layer: saccades, wall avoidance, altitude, the end of a bout
      let turn = 0, speed = this.segmentSpeed || CRUISE, land = false;
      const searching = !!this.search;
      if (rng() < (searching ? SEARCH_SACCADE_RATE : SACCADE_RATE) * DECISION_S) { turn = (rng() < 0.5 ? 1 : -1) * SACCADE_ANGLE * (0.8 + 0.4 * rng()); this.segmentSpeed = SPEED_MIN + (SPEED_MAX - SPEED_MIN) * rng() ** 1.5; speed = this.segmentSpeed; this.ethogram.saccades++; this.log("saccade"); }
      else if (rng() < MICRO_SACCADE_RATE * DECISION_S) { turn = (rng() < 0.5 ? 1 : -1) * (0.14 + 0.3 * rng()); this.ethogram.microSaccades++; }
      turn += (rng() - 0.5) * 2 * JITTER;
      if (rng() < 0.3 * DECISION_S) { this.segmentSpeed = SPEED_MIN + (SPEED_MAX - SPEED_MIN) * rng() ** 1.5; speed = this.segmentSpeed; }
      const hx = Math.cos(this.heading), hy = Math.sin(this.heading);
      const ahead = [x + 0.5 * hx, y + 0.5 * hy];
      if (ahead[0] < WALL_MARGIN || ahead[0] > W - WALL_MARGIN || ahead[1] < WALL_MARGIN || ahead[1] > D - WALL_MARGIN) {
        const toCentre = Math.atan2(D / 2 - y, W / 2 - x);
        turn = wrap_(toCentre - this.heading) * 0.6;
      }
      this.altitude = Math.min(H - 0.4, Math.max(0.3, this.altitude + (rng() - 0.5) * 2 * ALTITUDE_JITTER));
      this.bout -= DECISION_S;
      if (this.bout <= 0) land = true;
      // the brain layer overrides
      if (useBrain && this.baseline) {
        const asym = this.dev("dn:DNa02:right") - this.dev("dn:DNa02:left");
        if (Math.abs(asym) > 0.02) turn += K_TURN * asym * DECISION_S;
        const fwd = this.dev("dn:DNp09") || this.dev("dn:DNp09:left");
        speed = CRUISE * Math.min(1.5, Math.max(0.3, 1 + K_SPEED * fwd));
        if (this.dev("dn:landing") > LAND_LEVEL) land = true;
        if ((this.readouts["gf"] || 0) > ESCAPE_LEVEL) this.startEscape("the giant fibre fired");
      }
      // the smell: the open search steers the course toward its fruit by instinct until the mushroom
      // body has decided, then toward or away from it by that decision (the turn itself is supplied,
      // as in the arena); an approached fruit is landed on. Without the brain the instinct layer
      // approaches every smell it is hungry for and learns nothing.
      if (this.search && this.valence && !this.search.landed && this.hunger > APPETITE) {
        const F = this.fruits[this.valence.fruit].pos, dxy = Math.hypot(F[0] - x, F[1] - y);
        const bearing = wrap_(Math.atan2(F[1] - y, F[0] - x) - this.heading);
        const want = this.valence.action === 0 ? bearing : wrap_(bearing + Math.PI);
        turn = Math.max(-ODOUR_TURN, Math.min(ODOUR_TURN, want)) + (rng() - 0.5) * 0.3;
        if (this.valence.action === 0) {
          this.altitude = Math.max(F[2] + HOVER_HEIGHT, this.altitude - 0.25 * DECISION_S / 0.1 * 0.1);
          if (this.valence.innate && useBrain) speed = Math.max(0.05, Math.min(speed, dxy));  // slows to a hover over the fruit until the brain has decided
          if (dxy < LAND_REACH && (!this.valence.innate || !useBrain)) { land = true; }
        }
        else { this.altitude = Math.min(H - 0.4, this.altitude + 0.05); land = false; }
      }
      this.heading = wrap_(this.heading + turn); this.speed = speed;
      if (land) this.chooseLanding();
    } else if (this.mode === "landed") {
      this.sit -= DECISION_S;
      if (useBrain && this.baseline) {
        if ((this.readouts["gf"] || 0) > ESCAPE_LEVEL) { this.takeoff("escape takeoff"); return; }
        if (this.onSugar && (this.readouts["mn9"] || 0) > FEED_LEVEL && this.hunger > 0.15) { this.startFeeding(); return; }
        if (this.dev("dn:grooming") > GROOM_LEVEL) { this.startGrooming(); return; }
      }
      if (this.onSugar && this.hunger > 0.3 && rng() < 0.5) { this.startFeeding(); return; }
      if (this.sit <= 0) { if (rng() < 0.35) this.startGrooming(); else this.takeoff("voluntary takeoff"); }
    } else if (this.mode === "grooming" || this.mode === "feeding") {
      this.episode -= DECISION_S;
      if (useBrain && (this.readouts["gf"] || 0) > ESCAPE_LEVEL) { this.takeoff("escape takeoff"); return; }
      if (this.episode <= 0) { this.mode = "landed"; this.sit = expo(rng, SIT_MEAN_S) * 0.5; this.log("sits"); }
    }
  }

  chooseLanding() {
    const f = this.flight, t = this.room.table, [x, y] = f.p;
    const nearTable = x > t.x0 - 0.4 && x < t.x1 + 0.4 && y > t.y0 - 0.4 && y < t.y1 + 0.4;
    const drawn = this.search && !this.search.landed && this.valence && this.valence.action === 0 && this.hunger > APPETITE ? this.valence.fruit : null;
    this.landingFruit = null;
    if (drawn) { this.landingFruit = { name: drawn, dx: (this.rng() - 0.5) * 0.03, dy: (this.rng() - 0.5) * 0.03 }; this.landing = this.landingPoint(); }
    else if (nearTable) { const lx = Math.min(t.x1 - 0.05, Math.max(t.x0 + 0.05, x)), ly = Math.min(t.y1 - 0.05, Math.max(t.y0 + 0.05, y)); this.landing = [lx, ly, this.room.surfaceZ ? this.room.surfaceZ(lx, ly) : t.z]; }
    else this.landing = [x, y, 0.0];
    this.log(drawn ? `descends to the ${drawn}` : "descends to land");
  }
  /** The landing spot on the fruit the descent is to, where the fruit stands now. */
  landingPoint() { const L = this.landingFruit, F = this.fruits[L.name].pos; return [F[0] + L.dx, F[1] + L.dy, F[2]]; }

  // ---- the lessons: what the page asks the worker, and what the outcome was ------------------
  /** Called at every life decision: whether the brain should take the search's decision now.
   *  A search opens when a hungry fly notices a smell; the instinct orients it toward the fruit; at
   *  ODOUR_DECISION_LEVEL the mushroom body decides once, approach or avoid (the T-maze's one choice
   *  per trial: a decision every 0.3 s mixed the eligibility of approach and avoid nudges along one
   *  flight, and sugar then taught whichever nudge was larger); the outcome ends the search. */
  odourTick() {
    const S = this.search;
    if (S && !S.landed) {
      if (this.clock - S.since > EPISODE_S) { this.closeSearch(0, `the search for the ${S.fruit} ran out of time`); return false; }
      if (S.asked && this.valence && this.valence.action === 1 && this.smellC < SMELL_FLOOR) { this.closeSearch(0, `avoided the ${S.fruit}`); return false; }
    }
    if (this.mode !== "flying" || this.escape > 0 || this.hunger <= APPETITE) return false;
    if (!this.smelled) { if (S && !S.asked) this.dropSearch(`lost the ${S.fruit}`); return false; }
    if (!this.search) { this.search = { since: this.clock, fruit: this.smelled, asked: false, landed: null }; this.valence = { fruit: this.smelled, action: 0, p: null, innate: true }; this.log(`smells the ${this.smelled}`); return false; }
    if (this.search.landed || this.search.asked) return false;
    if (this.smelled !== this.search.fruit) { this.dropSearch(`the ${this.smelled} smells stronger than the ${this.search.fruit}`); return false; }
    const F = this.fruits[this.search.fruit].pos;
    if (Math.hypot(F[0] - this.flight.p[0], F[1] - this.flight.p[1]) >= ODOUR_DECISION_REACH) return false;
    this.search.asked = true;
    return true;
  }
  /** A search that ends before the brain was asked: nothing to learn from, nothing queued. */
  dropSearch(why) { this.search = null; this.valence = null; this.log(why); }
  /** The worker's decision for the open search. */
  applyDecision(d) {
    if (!d || !this.search || this.search.landed) return;
    const fruit = this.search.fruit;
    this.valence = { fruit, action: d.action, p: d.p }; this.search.action = d.action;
    this.lastP[fruit] = d.p[0];
    this.log(`${d.action === 0 ? "approaches" : "avoids"} the ${fruit} (${(100 * d.p[0]).toFixed(0)} % for approach)`);
  }
  /** The outcome that ends a search: sugar (+1), an empty fruit (0), a blow (-1); queued for the worker.
   *  A search that reaches a fruit without sugar stays open while the fly sits there (search.landed): it
   *  ends with nothing when the fly leaves, or with a blow. Closing it at the landing cleared the
   *  learner's eligibility before a blow could arrive, and the blow taught nothing. */
  closeSearch(reward, why) { if (this.search) this.outcome(reward, why); }
  /** An outcome at a fruit, search or no search. With no approach decision about that fruit to credit
   *  (no search open: the fly came to the fruit by chance, or fed there already; or the search was for
   *  the other fruit, or it had decided to avoid) it is marked fresh: the page then asks the brain for
   *  the approach decision the outcome is credited to, before handing the outcome over. */
  outcome(reward, why, on = null) {
    const S = this.search, fresh = !S || !S.asked || (on !== null && (S.action !== 0 || S.fruit !== on));  // an outcome at a fruit needs the approach decision about that fruit; a search ending on its own is credited to its decision
    this.pendingReward = { reward, done: true, why, fresh };
    this.search = null; this.valence = null;
    if (reward > 0) this.reward = DOPAMINE_S; if (reward < 0) this.punish = DOPAMINE_S;
  }
  /** A blow from the hand while the fly sits on or hovers over a fruit: a punishment for that smell. */
  punished(where) {
    const near = this.onWhich || this.fruitAt(this.flight.p, 0.2) || (this.smelled && this.smellC > 0.3 ? this.smelled : null);
    if (!near) return false;
    this.visits.punished++;
    this.outcome(-1, `struck at the ${near}`, near);
    this.log(`struck at the ${near}`);
    return true;
  }

  startEscape(why) { this.escape = 0.25; this.heading = wrap_(this.heading + (this.rng() < 0.5 ? 1 : -1) * Math.PI / 2); this.speed = 1.0; this.altitude = Math.min(2.2, this.altitude + 0.4); this.landing = null; this.landingFruit = null; this.log("escape: " + why); }
  startFeeding() { this.mode = "feeding"; this.episode = FEED_S; this.ethogram.feeds++; this.log("feeds on the fruit"); }
  startGrooming() { this.mode = "grooming"; this.episode = GROOM_S; this.ethogram.grooms++; this.groomTarget = ["head", "head", "wings", "legs"][Math.floor(this.rng() * 4)]; this.log(`grooms its ${this.groomTarget}`); }
  /** What the body shows: the pose the fly model draws. */
  pose() { return { mode: this.takeoffAt >= 0 && this.clock - this.takeoffAt < 0.15 ? "takeoff" : this.mode, t: this.clock, groom: this.mode === "grooming" ? this.groomTarget : null, feed: this.mode === "feeding" ? Math.min(1, (FEED_S - this.episode) / 0.6) : 0, hunger: this.hunger }; }
  /** The body held on its perch: level, still, the surface under its feet. */
  hold() {
    const f = this.flight, P = this.perch; if (!P) return;
    const yaw = f.euler()[2];
    f.p = [P[0], P[1], P[2] + RADIUS]; f.v = [0, 0, 0]; f.w = [0, 0, 0]; f.q = [Math.cos(yaw / 2), 0, 0, Math.sin(yaw / 2)];
    f.landed = true; f.touching = 1; f.onFloor = P[2] < 0.001;
  }
  takeoff(why) { if (this.search && this.search.landed) this.closeSearch(0, `left the ${this.search.landed} with nothing`); this.perch = null; if (this.ethogram.sitStart !== null) { this.ethogram.sits.push(this.clock - this.ethogram.sitStart); this.ethogram.sitStart = null; } this.ethogram.boutStart = this.clock; this.takeoffAt = this.clock; this.mode = "flying"; this.wingsOff = false; this.landing = null; this.landingFruit = null; this.bout = expo(this.rng, BOUT_MEAN_S); this.altitude = Math.max(0.6, this.flight.p[2] + 0.4); this.speed = CRUISE; this.heading = this.flight.euler()[2]; this.flight.v[2] += 0.28; this.log(why); }

  // ---- every physics step: the course into the pilot, the pilot into the wings ---------------
  step(dt) {
    const f = this.flight; this.clock += dt; this.since += dt;
    if (this.mode === "flying") this.ethogram.flying_s += dt; else this.ethogram.sitting_s += dt;
    this.hunger = Math.min(1, this.hunger + (this.mode === "feeding" ? -0.12 : HUNGER_RATE) * dt);
    if (this.reward > 0) this.reward -= dt; if (this.punish > 0) this.punish -= dt;
    if (this.mode === "flying") {
      this.fatigue = Math.min(1, this.fatigue + dt / 60);
      if (this.escape > 0) this.escape -= dt;
      const p = this.pilot;
      if (this.landing) {
        if (this.landingFruit) this.landing = this.landingPoint();  // the fruit may have been moved during the descent
        else if (this.room.surfaceZ) this.landing[2] = this.room.surfaceZ(this.landing[0], this.landing[1]);  // a fruit may have been set down on the spot
        const L = this.landing, dxy = Math.hypot(L[0] - f.p[0], L[1] - f.p[1]);
        p.target = [L[0], L[1], dxy > 0.15 ? Math.max(L[2] + 0.12, f.p[2] - 0.05) : L[2] + RADIUS + 0.0008];
        p.heading = dxy > 0.05 ? Math.atan2(L[1] - f.p[1], L[0] - f.p[0]) : p.heading; p.speed = Math.max(0.05, Math.min(CRUISE, dxy));
        const settled = dxy < 0.03 && Math.abs(f.p[2] - (L[2] + RADIUS)) < 0.004 && f.speed() < 0.1;
        if (settled || f.landed || (this.wingsOff && f.touching > 0 && f.speed() < 0.02)) {
          this.mode = "landed"; this.landing = null; this.landingFruit = null; this.sit = expo(this.rng, SIT_MEAN_S); this.fatigue = 0;
          this.perch = [f.p[0], f.p[1], settled ? L[2] : Math.max(0, f.p[2] - RADIUS)];
          this.hold();
          this.ethogram.landings++; this.ethogram.bouts.push(this.clock - this.ethogram.boutStart); this.ethogram.sitStart = this.clock;
          const on = this.fruitAt(f.p);
          if (on) { this.visits[on]++; if (on === this.sugar) { this.visits.sweet++; this.log(`lands on the ${on}: sugar`); this.outcome(1, `sugar on the ${on}`, on); } else { this.log(`lands on the ${on}: nothing`); if (this.search) { this.search.landed = on; this.valence = null; } } }
          else this.log("lands");
        }
      } else {
        const look = 0.25;
        p.target = [f.p[0] + look * Math.cos(this.heading), f.p[1] + look * Math.sin(this.heading), this.altitude];
        p.heading = this.heading; p.speed = this.speed;
      }
      this.controls = this.wingsOff ? { aL: 0, aR: 0, betaL: 0, betaR: 0, sL: 0, sR: 0, f: 0 } : p.controls(f);
    } else {
      this.fatigue = Math.max(0, this.fatigue - dt / 20);
      this.controls = { aL: 0, aR: 0, betaL: 0, betaR: 0, sL: 0, sR: 0, f: 0 };
    }
    if (this.perch && this.mode !== "flying") this.hold();  // the legs hold the body: no physics step while it stands
    else f.step(dt, this.controls);
    if (this.since >= DECISION_S) { this.since -= DECISION_S; return true; }
    return false;
  }
}
