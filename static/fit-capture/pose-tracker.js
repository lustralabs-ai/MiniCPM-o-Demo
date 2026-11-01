/**
 * pose-tracker.js — MediaPipe pose checks for the 360° fit-capture protocol.
 *
 * Ported from drape/apps/webcam_capture/capture_360.html (orientation scan,
 * framing, turn tracking), plus an arm-pose check ("christmas tree": arms
 * straight, 20–45° out from the body) and arm-droop detection while turning.
 *
 * The tracker draws every camera frame into an upright canvas (rotated by
 * 0/90/180/270°). That canvas feeds the pose model, the agent's video frames
 * and the recording.
 */

// Vendored copies; see vendor/mediapipe/README.md.
import { FilesetResolver, PoseLandmarker } from './vendor/mediapipe/vision_bundle.mjs';

const VENDOR    = '/static/fit-capture/vendor/mediapipe';
const WASM_URL  = `${VENDOR}/wasm`;
const MODEL_URL = `${VENDOR}/pose_landmarker_full.task`;

// Tuning constants: adjust after testing on real devices.
export const POSE_CFG = {
  targetDeg: 360,
  completeDeg: 340,      // accept slightly under a full turn (pose jitter)
  maxFrameTurnDeg: 40,   // larger per-frame jumps are left/right landmark flips, ignored
  stillWindowMs: 1000,   // window used to decide the body has stopped turning
  stillRangeDeg: 12,     // max swing in body angle inside that window that still counts as "still"
  swayLimit: 0.06,       // hip-centre drift, as a fraction of the frame
  minVisibility: 0.5,
  minBodyH: 0.5,         // head-to-feet height as a fraction of the frame; smaller is "too far"
  armMinDeg: 18,         // arm abduction from the torso (spec is 20–45°, with a little slack)
  armMaxDeg: 55,
  elbowMinDeg: 150,      // straight-arm threshold (180 = perfectly straight)
  droopMs: 1500,         // arms out of range for this long during the turn counts as a droop
  armEma: 0.3,           // smoothing for the arm angles
};

const SKELETON = [[11,12],[23,24],[11,23],[12,24],[23,27],[24,28],[11,13],[13,15],[12,14],[14,16]];
const KEY = [0,11,12,13,14,15,16,23,24,27,28];

const deg = r => r * 180 / Math.PI;

function sub(a, b) { return { x: a.x - b.x, y: a.y - b.y, z: (a.z ?? 0) - (b.z ?? 0) }; }
function angleBetween(u, v) {
  const d = u.x * v.x + u.y * v.y + u.z * v.z;
  const n = Math.hypot(u.x, u.y, u.z) * Math.hypot(v.x, v.y, v.z);
  return n > 1e-9 ? deg(Math.acos(Math.max(-1, Math.min(1, d / n)))) : 0;
}
function mid(a, b) { return { x: (a.x + b.x) / 2, y: (a.y + b.y) / 2, z: ((a.z ?? 0) + (b.z ?? 0)) / 2 }; }

async function createPoser(runningMode, preferGpu = true) {
  const fileset = await FilesetResolver.forVisionTasks(WASM_URL);
  const make = delegate => PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: MODEL_URL, delegate },
    runningMode,
    numPoses: 1,
  });
  if (preferGpu) {
    try { return await make('GPU'); }
    catch (e) { console.warn('GPU delegate unavailable, falling back to CPU', e); }
  }
  return make('CPU');
}

// Clockwise angle in degrees of the hips-to-shoulders direction (0 = upright), in pixel space.
function torsoTilt(lm, w, h) {
  const dx = ((lm[11].x + lm[12].x) / 2 - (lm[23].x + lm[24].x) / 2) * w;
  const dy = ((lm[23].y + lm[24].y) / 2 - (lm[11].y + lm[12].y) / 2) * h;
  return deg(Math.atan2(dx, dy));
}

// How plausible is it that this image shows an upright person? Uses only the points that stay visible
// when you are close (nose, shoulders, hips): high visibility, nose above shoulders, torso vertical.
function orientationScore(lm, w, h) {
  if (!lm) return 0;
  const core = [0, 11, 12, 23, 24];
  const vis = core.reduce((sum, i) => sum + (lm[i].visibility ?? 0), 0) / core.length;
  if (lm[0].y >= (lm[11].y + lm[12].y) / 2) return 0;          // nose must be above the shoulders
  return vis * Math.max(0, Math.cos(torsoTilt(lm, w, h) * Math.PI / 180));
}

/**
 * Is the whole body (head to feet) in frame, roughly centred, and big enough?
 * code is a stable id used for prompts; msg is the on-screen banner.
 */
export function framing(lm) {
  if (!lm) return { ok: false, code: 'no_person', msg: 'Step back so your whole body is in view' };
  const need = [0, 11, 12, 23, 24, 27, 28];
  if (need.some(i => (lm[i].visibility ?? 0) < POSE_CFG.minVisibility)) {
    const feet = (lm[27].visibility ?? 0) < POSE_CFG.minVisibility || (lm[28].visibility ?? 0) < POSE_CFG.minVisibility;
    return feet
      ? { ok: false, code: 'feet_hidden', msg: 'Step back so both feet are in view' }
      : { ok: false, code: 'body_hidden', msg: 'Step back so your head and body are in view' };
  }
  const top = lm[0].y;
  const bottom = Math.max(lm[27].y, lm[28].y);
  const hipX = (lm[23].x + lm[24].x) / 2;
  if (top < 0.02) return { ok: false, code: 'head_edge', msg: 'Step back: your head is at the top edge' };
  if (bottom > 0.97) return { ok: false, code: 'feet_edge', msg: 'Step back: your feet are at the bottom edge' };
  if (bottom - top < POSE_CFG.minBodyH) return { ok: false, code: 'too_far', msg: 'Move a little closer to the camera' };
  if (hipX < 0.2 || hipX > 0.8) return { ok: false, code: 'off_centre', msg: 'Move to the centre of the frame' };
  return { ok: true, code: 'ok', msg: 'Good position' };
}

/**
 * Arm pose from 3D world landmarks (metres, hip-centred), so it holds while the
 * person is side-on during the turn. Abduction = angle between the torso's
 * downward direction and shoulder→wrist; elbow = angle at the elbow.
 */
export function armAngles(lm, wl) {
  if (!lm || !wl) return null;
  const vis = i => (lm[i].visibility ?? 0) >= POSE_CFG.minVisibility * 0.6;
  const down = sub(mid(wl[23], wl[24]), mid(wl[11], wl[12]));
  const side = (s, e, w) => (vis(s) && vis(e) && vis(w)) ? {
    abd: angleBetween(down, sub(wl[w], wl[s])),
    elbow: angleBetween(sub(wl[s], wl[e]), sub(wl[w], wl[e])),
  } : null;
  return { left: side(11, 13, 15), right: side(12, 14, 16) };
}

/** Classify smoothed arm angles: ok | low | high | bent | hidden. */
export function armCheck(arms) {
  if (!arms || (!arms.left && !arms.right)) return { ok: false, code: 'arms_hidden', msg: 'Keep both arms in view' };
  const sides = [arms.left, arms.right].filter(Boolean);
  if (sides.some(s => s.elbow < POSE_CFG.elbowMinDeg)) return { ok: false, code: 'arms_bent', msg: 'Straighten your arms' };
  if (sides.some(s => s.abd < POSE_CFG.armMinDeg)) return { ok: false, code: 'arms_low', msg: 'Lift your arms out a little more' };
  if (sides.some(s => s.abd > POSE_CFG.armMaxDeg)) return { ok: false, code: 'arms_high', msg: 'Lower your arms a little' };
  if (sides.length < 2) return { ok: false, code: 'arms_hidden', msg: 'Keep both arms in view' };
  return { ok: true, code: 'ok', msg: 'Arms look good' };
}

/** Yaw tracker over a recording; ported from capture_360 trackTurn(). */
export class TurnTracker {
  constructor() { this.reset(); }

  reset() {
    this.prevYaw = null;
    this.cumDeg = 0;
    this.moves = [];
    this.completedAt = 0;
    this.hipBase = null;
    this.maxSway = 0;
    this.sway = 0;
    this.lostSince = null;
    this.armBadSince = null;
    this.droops = 0;
    this.drooping = false;
  }

  get absDeg() { return Math.abs(this.cumDeg); }

  /** Returns { lost, absDeg, completed, still, sway, swaying, drooping, armCode } for this frame. */
  update(lm, wl, arms, now) {
    const core = lm && wl && [11, 12, 23, 24].every(i => (lm[i].visibility ?? 0) >= POSE_CFG.minVisibility);
    if (!core) {
      if (this.lostSince === null) this.lostSince = now;
      return { lost: true, lostMs: now - this.lostSince, absDeg: this.absDeg, completed: !!this.completedAt };
    }
    this.lostSince = null;

    // Body yaw: direction of the shoulder-and-hip line in the horizontal plane.
    const l = wl[11], r = wl[12], lh = wl[23], rh = wl[24];
    const yaw = Math.atan2((r.z - l.z) + (rh.z - lh.z), (r.x - l.x) + (rh.x - lh.x));
    if (this.prevYaw !== null) {
      let d = yaw - this.prevYaw;
      d = deg(Math.atan2(Math.sin(d), Math.cos(d)));
      if (Math.abs(d) <= POSE_CFG.maxFrameTurnDeg) {
        this.cumDeg += d;
        this.moves.push({ t: now, c: this.cumDeg });
        this.prevYaw = yaw;
      }
    } else {
      this.prevYaw = yaw;
    }
    while (this.moves.length && this.moves[0].t < now - POSE_CFG.stillWindowMs) this.moves.shift();
    let windowDeg = 0;
    if (this.moves.length) {
      let lo = Infinity, hi = -Infinity;
      for (const m of this.moves) { if (m.c < lo) lo = m.c; if (m.c > hi) hi = m.c; }
      windowDeg = hi - lo;
    }
    // Pose angles are noisier when the person is small in the frame, so loosen the threshold for them.
    const bodyH = Math.max(0.2, Math.max(lm[27].y, lm[28].y) - lm[0].y);
    const stillLimit = POSE_CFG.stillRangeDeg * Math.min(2.5, Math.max(1, 0.6 / bodyH));

    const hx = (lm[23].x + lm[24].x) / 2, hy = (lm[23].y + lm[24].y) / 2;
    if (!this.hipBase) this.hipBase = { x: hx, y: hy };
    this.sway = Math.hypot(hx - this.hipBase.x, hy - this.hipBase.y);
    this.maxSway = Math.max(this.maxSway, this.sway);

    if (!this.completedAt && this.absDeg >= POSE_CFG.completeDeg) this.completedAt = now;

    // Droop: arms out of range for a sustained stretch. Hidden arms (side-on) don't count.
    const ac = armCheck(arms);
    const bad = !ac.ok && ac.code !== 'arms_hidden';
    if (bad) {
      if (this.armBadSince === null) this.armBadSince = now;
      if (!this.drooping && now - this.armBadSince >= POSE_CFG.droopMs) { this.drooping = true; this.droops++; }
    } else if (ac.ok) {
      this.armBadSince = null;
      this.drooping = false;
    }

    return {
      lost: false,
      absDeg: this.absDeg,
      completed: !!this.completedAt,
      completedAt: this.completedAt,
      still: windowDeg < stillLimit,
      sway: this.sway,
      swaying: this.sway > POSE_CFG.swayLimit,
      drooping: this.drooping,
      armCode: ac.code,
    };
  }
}

/**
 * Owns the pose model, the upright frame canvas and the preview overlay.
 * Call process(video, now) once per animation frame.
 */
export class PoseTracker {
  constructor({ frameCanvas, overlayCanvas, debug = false }) {
    this.frame = frameCanvas;
    this.fctx = frameCanvas.getContext('2d');
    this.overlay = overlayCanvas;
    this.octx = overlayCanvas.getContext('2d');
    this.debug = debug;
    this.poser = null;
    this.probePoser = null;
    this.probeLoading = false;
    this.probing = false;
    this.probeCanvas = document.createElement('canvas');
    this.pctx = this.probeCanvas.getContext('2d');
    this.rotation = 0;
    this.lastTs = 0;
    this.lastRotChange = -Infinity;
    this.lastScanAt = -Infinity;
    this.noPoseSince = null;
    this.lastScanInfo = '';
    this.allowRotationScan = true;   // set false once recording starts
    this.smArms = null;
  }

  get ready() { return !!this.poser; }

  async load() {
    if (!this.poser) this.poser = await createPoser('VIDEO');
  }

  drawFrame(video, rot = this.rotation, c = this.frame, g = this.fctx) {
    const vw = video.videoWidth, vh = video.videoHeight;
    const swap = rot === 90 || rot === 270;
    const cw = swap ? vh : vw, ch = swap ? vw : vh;
    if (c.width !== cw || c.height !== ch) { c.width = cw; c.height = ch; }
    if (rot === 90)       g.setTransform(0, 1, -1, 0, cw, 0);
    else if (rot === 270) g.setTransform(0, -1, 1, 0, 0, ch);
    else if (rot === 180) g.setTransform(-1, 0, 0, -1, cw, ch);
    else                  g.setTransform(1, 0, 0, 1, 0, 0);
    g.drawImage(video, 0, 0, vw, vh);
    g.setTransform(1, 0, 0, 1, 0, 0);
  }

  /**
   * Draw the upright frame and run pose on it. Returns
   * { lm, wl, framing, arms, armCheck, bodyH, rotated } or null if the video isn't ready.
   */
  process(video, now) {
    if (video.readyState < 2 || !video.videoWidth) return null;
    this.drawFrame(video);
    if (!this.poser) return { lm: null, wl: null, framing: framing(null), arms: null, armCheck: armCheck(null), bodyH: 0, rotated: false };

    this.lastTs = Math.max(this.lastTs + 1, now);   // MediaPipe needs strictly increasing timestamps
    const res = this.poser.detectForVideo(this.frame, this.lastTs);
    const lm = res.landmarks?.[0] || null;
    const wl = res.worldLandmarks?.[0] || null;
    this.drawOverlay(video, lm);

    const before = this.rotation;
    this._updateOrientation(video, lm, now);

    const raw = armAngles(lm, wl);
    this.smArms = this._smooth(this.smArms, raw);
    const bodyH = lm ? Math.max(lm[27].y, lm[28].y) - lm[0].y : 0;
    return {
      lm, wl,
      framing: framing(lm),
      arms: this.smArms,
      armCheck: armCheck(this.smArms),
      bodyH,
      rotated: this.rotation !== before,
    };
  }

  _smooth(prev, cur) {
    if (!cur) return null;
    const a = POSE_CFG.armEma;
    const mix = (p, c) => !c ? null : !p ? { ...c } : { abd: p.abd + a * (c.abd - p.abd), elbow: p.elbow + a * (c.elbow - p.elbow) };
    return { left: mix(prev?.left, cur.left), right: mix(prev?.right, cur.right) };
  }

  _applyRotation(rot, why) {
    if (rot === this.rotation) return;
    this.rotation = rot;
    this.lastRotChange = performance.now();
    this.noPoseSince = null;
    this.smArms = null;
    console.info(`Camera orientation set to ${rot} degrees (${why})`);
  }

  _ensureProbePoser() {
    if (this.probePoser || this.probeLoading) return;
    this.probeLoading = true;
    createPoser('IMAGE', false)
      .then(pp => { this.probePoser = pp; })
      .catch(e => console.warn('Orientation scanner unavailable', e))
      .finally(() => { this.probeLoading = false; });
  }

  // Try each rotation once on a hidden canvas with a separate image-mode detector.
  _scanOrientations(video) {
    if (!this.probePoser) { this._ensureProbePoser(); return; }
    this.probing = true;
    try {
      const scores = {};
      for (const rot of [0, 90, 180, 270]) {
        this.drawFrame(video, rot, this.probeCanvas, this.pctx);
        const res = this.probePoser.detect(this.probeCanvas);
        scores[rot] = orientationScore(res.landmarks?.[0], this.probeCanvas.width, this.probeCanvas.height);
      }
      const best = [0, 90, 180, 270].reduce((a, r) => (scores[r] > scores[a] ? r : a), this.rotation);
      this.lastScanInfo = [0, 90, 180, 270].map(r => `${r}:${scores[r].toFixed(2)}`).join(' ');
      if (best !== this.rotation && scores[best] >= 0.35 && scores[best] > scores[this.rotation] + 0.1) {
        this._applyRotation(best, 'scan');
      }
    } catch (e) {
      console.warn('Orientation scan failed', e);
    } finally {
      this.probing = false;
      this.lastScanAt = performance.now();
    }
  }

  _updateOrientation(video, lm, now) {
    const plausible = orientationScore(lm, this.frame.width, this.frame.height) >= 0.5;
    if (plausible) { this.noPoseSince = null; return; }
    if (this.noPoseSince === null) { this.noPoseSince = now; if (this.allowRotationScan) this._ensureProbePoser(); }
    if (!this.allowRotationScan || now - this.lastRotChange < 1500) return;
    if (now - this.noPoseSince > 1000 && now - this.lastScanAt > 2000 && !this.probing) this._scanOrientations(video);
  }

  // The preview shows the raw camera image; landmarks are mapped back from the upright frame.
  _toRaw(x, y) {
    switch (this.rotation) {
      case 90:  return { x: y, y: 1 - x };
      case 180: return { x: 1 - x, y: 1 - y };
      case 270: return { x: 1 - y, y: x };
      default:  return { x, y };
    }
  }

  drawOverlay(video, lm) {
    const c = this.overlay, g = this.octx;
    if (c.width !== video.videoWidth || c.height !== video.videoHeight) {
      c.width = video.videoWidth;
      c.height = video.videoHeight;
    }
    g.clearRect(0, 0, c.width, c.height);
    if (!lm) return;
    const P = i => { const r = this._toRaw(lm[i].x, lm[i].y); return { x: r.x * c.width, y: r.y * c.height }; };
    const ac = armCheck(this.smArms);
    g.lineWidth = Math.max(2, c.width / 200);
    SKELETON.forEach(([a, b], k) => {
      g.strokeStyle = k >= 6 ? (ac.ok ? 'rgba(34,197,94,0.9)' : 'rgba(245,158,11,0.9)') : 'rgba(59,130,246,0.9)';
      const pa = P(a), pb = P(b);
      g.beginPath(); g.moveTo(pa.x, pa.y); g.lineTo(pb.x, pb.y); g.stroke();
    });
    g.fillStyle = '#fff';
    for (const i of KEY) {
      const p = P(i);
      g.beginPath(); g.arc(p.x, p.y, Math.max(3, c.width / 150), 0, Math.PI * 2); g.fill();
    }
  }

  clearOverlay() { this.octx.clearRect(0, 0, this.overlay.width, this.overlay.height); }
}
