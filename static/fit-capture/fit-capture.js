/**
 * fit-capture.js — agent-guided 360° body-turn video capture.
 *
 * Steps: positioning → arms → [countdown] → recording → finishing → done | failed.
 *
 * Driver "agent": the duplex model starts the recording with a 🎬 marker and may
 * end it with ✅; the pose tracker gates both (optional) and also stops the
 * recording when the turn is complete. Driver "tracker" (default): the pose tracker
 * moves through every step, the page announces each change with a fixed line
 * (STEP_LINES), and the agent only coaches within the step. 🔄 / 🛑 work in both modes.
 * The agent driver let the model judge progress from its own view, and it often
 * talked the user through steps the tracker had not passed.
 *
 * Context: the cpp backend has an 8192-token KV. The page resets the LLM context
 * in the same session (session.reset) with a fresh prompt from prompts.js at each
 * step change, on stable tracker findings, and when the KV passes a threshold.
 * Resets wait until the model is listening, except at a step change: then the page
 * cuts the agent off, resets, and speaks the step line. Resets also wait until the user
 * has stopped talking (mic level), so a question is not wiped from the context.
 *
 * Forced speech: the opening and closing lines are not left to the model. In duplex
 * mode it often waited silently for the user to speak first, and after the capture
 * it kept talking. The page instead sends session.say (cpp backend, see llama.cpp-omni
 * docs/omni-duplex-say.md) with a line from prompts.js: the backend speaks it verbatim
 * and writes it into the context as the assistant's own words. The closing line uses
 * mute_after, so the model only listens afterwards; the page ends the session once
 * session.say.done has arrived and the audio has played out.
 */

import { RealtimeSession } from '/static/duplex/lib/realtime-session.js';
import { arrayBufferToBase64 } from '/static/duplex/lib/duplex-utils.js';
import { buildPrompt, findMarkers, findPhraseMarkers, stripEmoji, pickLine, OPENING_LINES, CLOSING_LINES, FAILED_LINES, STEP_LINES } from './prompts.js';

const SAMPLE_RATE_IN = 16000;
const SEND_MAX_H = 640;          // agent frames are scaled down to at most this height
const SEND_JPEG_Q = 0.7;
const PRESET_AUDIO_URL = '/api/presets/omni/english_call/audio';

const FLOW = {
    readyHoldMs: 1000,       // whole body in view this long → arms step
    armsHoldMs: 1200,        // tracker driver: arms OK this long → countdown
    countdownSec: 3,
    dropoutMs: 500,          // brief framing loss shorter than this is ignored
    lostStepMs: 1500,        // framing lost this long during arms/countdown → back to positioning
    startGateMs: 1500,       // agent 🎬 accepted if framing was OK within this window
    minDoneDeg: 270,         // agent ✅ accepted once the tracker has seen this much turn
    maxFinishMs: 3000,       // after a complete turn, stop even if the pose is still noisy
    hintStableMs: 1500,      // tracker finding must hold this long before it is sent
    hintResetGapMs: 8000,    // minimum gap between tracker-driven resets
    resetTimeoutMs: 6000,    // give up waiting for session.reset.done
    forceResetOverKv: 1500,  // reset even mid-speech once KV passes threshold + this
    lingerMaxMs: 15000,      // longest wait for the closing sentence to be spoken
    speechHoldMs: 700,       // agent counts as speaking this long after playback runs dry
    sayTimeoutMs: 20000,     // give up waiting for session.say.done
    sayTailMs: 1500,         // after say.done, the Talker may still owe this much audio
    userQuietMs: 1000,       // resets wait until the user has been quiet this long
    userHoldMaxMs: 3000,     // a step change waits at most this long for the user (or mic noise)
};

// User voice detection on the mic, so a reset does not wipe a question being asked.
const VOICE = {
    pollMs: 100,
    minRms: 0.01,            // never count quieter than this as voice
    floorRatio: 3,           // voice = RMS this many times the tracked noise floor
};

// Frame gating: a frame costs ~64 KV tokens, so a still scene only gets a keepalive frame.
const GATE = {
    keepaliveMs: 4000,       // send at least this often, and always when there is no pose
    poseMove: 0.06,          // mean landmark shift, as a fraction of body height
    turnDeg: 20,             // during recording, send after this much more turn
};
// Nose, shoulders, elbows, wrists, hips, knees, ankles.
const GATE_LM = [0, 11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28];

const $ = id => document.getElementById(id);
const els = {
    intro: $('intro'), error: $('error'), settings: $('settings'), startBtn: $('startBtn'),
    preview: $('preview'), previewVideo: $('previewVideo'), previewNote: $('previewNote'),
    capture: $('capture'), view: $('view'), video: $('video'), overlay: $('overlay'), frame: $('frame'),
    sendCanvas: $('sendCanvas'), countdown: $('countdown'), banner: $('banner'), subtitle: $('subtitle'),
    chipStep: $('chipStep'), chipAgent: $('chipAgent'), chipKv: $('chipKv'), chipTime: $('chipTime'), chipRec: $('chipRec'),
    progressFill: $('progressFill'), progressText: $('progressText'), debugLog: $('debugLog'),
    restartBtn: $('restartBtn'), stopBtn: $('stopBtn'),
    result: $('result'), resultTitle: $('resultTitle'), playback: $('playback'), stats: $('stats'),
    download: $('download'), again: $('again'),
};
const fctx = els.frame.getContext('2d');
const sctx = els.sendCanvas.getContext('2d');

// ── Settings ────────────────────────────────────────────────────────────────

const SETTINGS_KEY = 'fitCapture.settings';
const SETTINGS_VERSION = 2;   // bump to drop saved values whose defaults changed (loadSettings)
const SETTING_IDS = {
    camera: 'cfgCamera', res: 'cfgRes', driver: 'cfgDriver', timeoutSec: 'cfgTimeout', maxRecSec: 'cfgMaxRec',
    frameEvery: 'cfgFrameEvery', slices: 'cfgSlices', resetKv: 'cfgResetKv', listenScale: 'cfgListenScale', reset: 'cfgReset',
    frameGate: 'cfgFrameGate',
    hints: 'cfgHints', gate: 'cfgGate', phrases: 'cfgPhrases', mirror: 'cfgMirror', debug: 'cfgDebug',
};

function readSettings() {
    const s = {};
    for (const [k, id] of Object.entries(SETTING_IDS)) {
        const el = $(id);
        s[k] = el.type === 'checkbox' ? el.checked : el.type === 'number' ? Number(el.value) : el.value;
    }
    s.frameEvery = Number(s.frameEvery) || 1;
    s.slices = Number(s.slices) || 1;
    s.timeoutSec = Math.max(20, s.timeoutSec || 120);
    s.maxRecSec = Math.max(10, s.maxRecSec || 40);
    s.resetKv = s.resetKv || 5500;
    if (!Number.isFinite(s.listenScale)) s.listenScale = 0;
    if (location.hash === '#debug') s.debug = true;
    return s;
}

function loadSettings() {
    let saved = {};
    try { saved = JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}'); } catch (_) {}
    if ((saved.v || 1) < SETTINGS_VERSION) {
        // v2: driver now defaults to the pose tracker, listen bias to 0; drop the old saved values once.
        delete saved.driver;
        delete saved.listenScale;
    }
    for (const [k, id] of Object.entries(SETTING_IDS)) {
        if (!(k in saved)) continue;
        const el = $(id);
        if (el.type === 'checkbox') el.checked = !!saved[k];
        else if (el.tagName === 'SELECT' && ![...el.options].some(o => o.value === String(saved[k]))) {
            // A device id from a previous visit; add a placeholder until devices are listed.
            if (k === 'camera') el.add(new Option('Saved camera', saved[k]));
            else continue;
            el.value = saved[k];
        } else el.value = saved[k];
    }
}

function saveSettings(s) {
    try { localStorage.setItem(SETTINGS_KEY, JSON.stringify({ ...s, v: SETTINGS_VERSION })); } catch (_) {}
}

// The label of the last camera used. Device ids can change when a camera is replugged
// (e.g. a phone in webcam mode), so a saved device is found again by its label.
const CAMERA_LABEL_KEY = 'fitCapture.cameraLabel';

async function videoInputs() {
    return (await navigator.mediaDevices.enumerateDevices()).filter(d => d.kind === 'videoinput');
}

async function listCameras() {
    try {
        const devs = await videoInputs();
        if (!devs.length || !devs.some(d => d.label)) return;   // no labels until camera permission is granted
        const sel = $('cfgCamera');
        let cur = sel.value;
        const savedLabel = localStorage.getItem(CAMERA_LABEL_KEY);
        if (cur.startsWith('device:') && !devs.some(d => `device:${d.deviceId}` === cur)) {
            const same = devs.find(d => d.label && d.label === savedLabel);
            if (same) cur = `device:${same.deviceId}`;
        }
        [...sel.options].filter(o => o.value.startsWith('device:')).forEach(o => o.remove());
        devs.forEach((d, i) => sel.add(new Option(d.label || `Camera ${i + 1}`, `device:${d.deviceId}`)));
        sel.value = [...sel.options].some(o => o.value === cur) ? cur : sel.options[0].value;
    } catch (_) {}
}

navigator.mediaDevices?.addEventListener?.('devicechange', listCameras);

$('cfgCamera').addEventListener('change', e => {
    if (e.target.value === 'facing:user') $('cfgMirror').checked = true;
    else if (e.target.value === 'facing:environment') $('cfgMirror').checked = false;
});

// ── State ───────────────────────────────────────────────────────────────────

let cfg = readSettings();
let step = 'idle';
let session = null, PT = null, tracker = null, turn = null;
let camKey = '', previewSeq = 0;   // settings camStream was opened with; bumped to drop a stale preview open
let camStream = null, micStream = null, micCtx = null, captureNode = null, sinkNode = null;
let micAnalyser = null, voiceTimer = 0, noiseFloor = 0, lastUserVoiceAt = -Infinity, userVoiceOn = false;
let rafId = 0, tickTimer = 0, lingerTimer = 0;
let t0 = 0, attempt = 1, resets = 0, note = '';
let lingering = false, closingDone = false, closingDoneAt = 0, sayInFlight = false, saySentAt = 0;
let speaking = false, lastAudibleAt = -Infinity, kv = 0, agentLost = false, refAudio = null;
let pendingReset = null, resetInFlight = false, resetSentAt = 0, lastResetAt = 0;
let lineAfterReset = null;   // step line to speak once the reset it belongs to is done
let candCodes = '', candSince = 0, promptCodes = '', liveCodes = [];
let framingOkSince = null, lastFramingOkAt = -Infinity, armsOkSince = null;
let countdownStart = 0, lastCount = null;
let recorder = null, recChunks = [], mimeType = '', recordStart = 0, stopReason = '', canvasStream = null;
let fired = new Set(), lastUrl = null, lastFr = null;
let poseLm = null, sent = null, frameStats = { sent: 0, skipped: 0 };

const ACTIVE_STEPS = new Set(['positioning', 'arms', 'countdown', 'recording']);
const now = () => performance.now();
const elapsedSec = () => t0 ? (now() - t0) / 1000 : 0;

function log(text) {
    console.info('[fit]', text);
    if (!cfg.debug) return;
    const line = `${elapsedSec().toFixed(1).padStart(5)} ${text}\n`;
    els.debugLog.textContent = (els.debugLog.textContent + line).split('\n').slice(-60).join('\n');
    els.debugLog.scrollTop = els.debugLog.scrollHeight;
}

const setBanner = t => { if (els.banner.textContent !== t) els.banner.textContent = t; };

function setChip(el, text, cls = '') {
    el.textContent = text;
    el.className = 'chip' + (cls ? ' ' + cls : '');
}

// ── Prompt state and context resets ─────────────────────────────────────────

function promptState() {
    return {
        step: step === 'finishing' ? 'recording' : step,
        driver: agentLost ? 'tracker' : cfg.driver,
        tracker: cfg.hints ? liveCodes : [],
        turnDeg: turn ? turn.absDeg : null,
        attempt,
        elapsedSec: elapsedSec(),
        timeoutSec: cfg.timeoutSec,
        resets,
        note,
    };
}

// line: a step line to speak right after the reset (session.reset clears any queued say).
function requestReset(reason, line = null) {
    if (!session?.running) return;
    if (!cfg.reset) { if (line) say(line); return; }
    // A step change outranks a tracker or KV reason already waiting; a newer step line replaces an older one.
    if (!pendingReset || reason === 'step' || reason === 'note') {
        pendingReset = { reason, at: now(), line: line ?? pendingReset?.line ?? null };
    }
    maybeReset();
}

function maybeReset() {
    if (!pendingReset || !session?.running) return;
    if (sayBusy()) return;    // a reset would drop the rest of the forced line
    if (resetInFlight) {
        if (now() - resetSentAt < FLOW.resetTimeoutMs) return;
        log('⚠ session.reset.done timed out');
        resetInFlight = false;
        if (lineAfterReset) { say(lineAfterReset); lineAfterReset = null; return; }
    }
    const overKv = kv >= cfg.resetKv + FLOW.forceResetOverKv;
    const line = pendingReset.line;
    // Wait until the reply has finished playing, except for a step change: a chatty agent would
    // otherwise hold the old step's prompt for many seconds while the screen has moved on.
    if (agentAudible() && !overKv && !line) return;
    // Nor while the user is talking: the fresh context would never contain their question.
    // A step change only waits so long, in case mic noise or echo reads as speech.
    if (userTalking() && !overKv && !(line && now() - pendingReset.at > FLOW.userHoldMaxMs)) return;
    if (overKv && agentAudible()) log(`⚠ forced reset mid-speech (kv=${kv})`);
    if (pendingReset.reason === 'tracker' && now() - lastResetAt < FLOW.hintResetGapMs) return;

    const prompt = buildPrompt(promptState());
    if (!session.resetContext(prompt)) return;
    log(`→ reset (${pendingReset.reason}, kv=${kv}, ${prompt.length}ch)`);
    if (line) {
        // Cut what the agent was saying about the old step; the step line follows the reset.
        session.audioPlayer?.stopAll();
        lineAfterReset = line;
    }
    if (cfg.debug) console.debug('[fit] prompt\n' + prompt);
    promptCodes = liveCodes.join(',');
    note = '';
    pendingReset = null;
    resetInFlight = true;
    resetSentAt = lastResetAt = now();
}

// Speak a fixed line as the agent (session.say).
function say(text, opts) {
    if (!session?.say(text, opts)) return false;
    log(`→ say "${text}"${opts?.muteAfter ? ' (mute after)' : ''}`);
    sayInFlight = true;
    saySentAt = now();
    return true;
}

function sayBusy() {
    if (sayInFlight && now() - saySentAt > FLOW.sayTimeoutMs) {
        log('⚠ session.say.done timed out');
        sayInFlight = false;
    }
    return sayInFlight;
}

// The model switches to listen while the Talker may still owe several seconds of audio
// (turn-end flush), so "done speaking" means the audio player has run dry.
// A short hold bridges gaps between flush chunks.
function agentAudible() {
    const p = session?.audioPlayer;
    const t = now();
    if (speaking || p?.turnActive || (p?.ctx && p.playing && p.nextTime > p.ctx.currentTime + 0.05)) {
        lastAudibleAt = t;
    }
    return t - lastAudibleAt < FLOW.speechHoldMs;
}

// Stable tracker findings become a prompt update after FLOW.hintStableMs.
function noteTrackerCodes(codes) {
    const key = codes.join(',');
    const t = now();
    if (key !== candCodes) { candCodes = key; candSince = t; return; }
    if (t - candSince < FLOW.hintStableMs) return;
    liveCodes = codes;
    if (cfg.hints && key !== promptCodes && ACTIVE_STEPS.has(step) && step !== 'countdown') requestReset('tracker');
}

// ── Step changes ────────────────────────────────────────────────────────────

function goStep(next, { reset = true } = {}) {
    if (step === next) return;
    log(`step ${step} → ${next}`);
    const line = stepLine(step, next);
    step = next;
    setChip(els.chipStep, next);
    framingOkSince = null;
    armsOkSince = null;
    if (reset) requestReset('step', line);
    else if (line && !sayBusy() && !userTalking()) say(line);   // e.g. the countdown line; never talk over a line or the user
}

// The fixed line announcing a step change, or null. With the agent driver the model moves
// the user on itself, so only the recording start (which the page performs) is announced.
function stepLine(prev, next) {
    const trackerDrives = cfg.driver === 'tracker' || agentLost;
    let lines = null;
    if (next === 'recording') lines = STEP_LINES.recording;
    else if (!trackerDrives) return null;
    else if (next === 'arms') lines = STEP_LINES.arms;
    else if (next === 'countdown') lines = STEP_LINES.countdown;
    else if (next === 'positioning') lines = prev === 'restarting' ? STEP_LINES.restart : STEP_LINES.lost;
    return lines ? pickLine(lines) : null;
}

function restartAttempt(why) {
    log(`restart (${why})`);
    if (recorder && recorder.state !== 'inactive') {
        recorder.onstop = null;
        recorder.stop();
    }
    recorder = null;
    canvasStream?.getTracks().forEach(t => t.stop());
    canvasStream = null;
    attempt++;
    turn?.reset();
    if (tracker) tracker.allowRotationScan = true;
    els.countdown.classList.add('hidden');
    els.chipRec.classList.add('hidden');
    setProgress(0);
    step = 'restarting';
    goStep('positioning');
}

// ── Markers from the agent ──────────────────────────────────────────────────

function handleAgentText(text) {
    for (const name of findMarkers(text)) {
        if (fired.has(name)) continue;
        fired.add(name);
        onMarker(name, 'emoji');
    }
    if (cfg.phrases && cfg.driver === 'agent' && !agentLost) {
        for (const name of findPhraseMarkers(stripEmoji(text))) {
            if (fired.has(name)) continue;
            fired.add(name);
            onMarker(name, 'phrase');
        }
    }
    els.subtitle.textContent = stripEmoji(text);
}

function onMarker(name, src) {
    log(`★ marker ${name} (${src}) in ${step}`);
    if (lingering) return;   // the result is already on screen; a stray 🛑 must not replace it
    const agentDrives = cfg.driver === 'agent' && !agentLost;

    if (name === 'stop') { cancelAll('Stopped at your request.'); return; }
    if (name === 'restart') { if (ACTIVE_STEPS.has(step)) restartAttempt('agent'); return; }

    if (name === 'start') {
        if (!agentDrives || (step !== 'positioning' && step !== 'arms')) return;
        if (cfg.gate && tracker?.ready && now() - lastFramingOkAt > FLOW.startGateMs) {
            note = `The app did not start recording yet: ${lastFr?.msg || 'the whole body is not in view'}.`;
            log(`  start refused: ${lastFr?.code}`);
            requestReset('note');
            return;
        }
        startRecording();
        return;
    }

    if (name === 'done') {
        if (!agentDrives || step !== 'recording') return;
        if (cfg.gate && tracker?.ready && turn.absDeg < FLOW.minDoneDeg) {
            note = `The turn is not finished: only ${Math.round(turn.absDeg)} of 360 degrees so far.`;
            log(`  done refused at ${Math.round(turn.absDeg)}°`);
            requestReset('note');
            return;
        }
        stopRecording('agent');
    }
}

// ── Session ─────────────────────────────────────────────────────────────────

function createSession() {
    const s = new RealtimeSession('fit', {
        // With resets on, the page keeps KV below the limit itself; this is only a backstop.
        getMaxKvTokens: () => 8150,
        getPlaybackDelayMs: () => 200,
        getStopOnSlidingWindow: () => false,
        getWsUrl: () => {
            const proto = location.protocol === 'https:' ? 'wss' : 'ws';
            const url = `${proto}://${location.host}/v1/realtime?mode=video`;
            return window.ClientIdentity ? window.ClientIdentity.appendToUrl(url) : url;
        },
    });

    s.onSystemLog = text => log(text);
    s.onQueueUpdate = q => { if (q) setBanner(`Waiting for the assistant (position ${q.position})…`); };

    s.onSpeakStart = text => {
        speaking = true;
        fired = new Set();
        setChip(els.chipAgent, 'agent: speaking', 'speaking');
        handleAgentText(text);
        return {};
    };
    s.onSpeakUpdate = (_h, text) => handleAgentText(text);
    s.onSpeakEnd = () => {
        log(`agent: "${s.currentSpeakText}"`);
    };
    s.onListenResult = () => {
        // The LLM is listening again; the chip and pending resets follow playback (tick()).
        speaking = false;
        maybeReset();
    };

    s.onMetrics = data => {
        if (data.type !== 'result' || !data.kvCacheLength) return;
        kv = data.kvCacheLength;
        setChip(els.chipKv, `kv ${kv}`, kv >= cfg.resetKv ? 'warn' : '');
        if (cfg.reset && kv >= cfg.resetKv && !pendingReset && !resetInFlight) requestReset('kv');
    };

    s.onSayDone = msg => {
        sayInFlight = false;
        log(`← say.done muted=${msg.muted}`);
        if (lingering) {
            // The closing line is decoded and the model is muted: stop the mic, let the audio play out.
            closingDone = true;
            closingDoneAt = now();
        }
        maybeReset();
    };

    s.onSessionReset = msg => {
        resetInFlight = false;
        resets++;
        kv = msg.metrics?.kv_cache_length || 0;
        setChip(els.chipKv, `kv ${kv}`);
        log(`← reset.done ok=${msg.ok} kv=${kv} ${Math.round(now() - resetSentAt)}ms`);
        sent = null;    // the fresh context has not seen the scene yet
        if (lineAfterReset && !lingering) say(lineAfterReset);
        lineAfterReset = null;
        maybeReset();   // another reason may have queued meanwhile
    };

    s.onRunningChange = running => {
        if (running || s !== session) return;
        if (ACTIVE_STEPS.has(step)) {
            // Keep going on the tracker alone rather than throwing the attempt away.
            agentLost = true;
            setChip(els.chipAgent, 'agent: disconnected', 'warn');
            log('⚠ assistant disconnected; tracker drives the flow');
        }
    };
    return s;
}

async function fetchRefAudio() {
    try {
        const r = await fetch(PRESET_AUDIO_URL);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const data = await r.json();
        return data?.ref_audio?.data || null;
    } catch (e) {
        log(`ref audio unavailable: ${e.message}`);
        return null;
    }
}

// ── Media ───────────────────────────────────────────────────────────────────

const cameraKey = s => `${s.camera}|${s.res}`;
const camLive = () => camStream?.getVideoTracks()[0]?.readyState === 'live';

function stopCamera() {
    camStream?.getTracks().forEach(t => t.stop());
    camStream = null;
    camKey = '';
}

// Opens the camera chosen in settings s; falls back from a missing saved device.
async function openCamera(s) {
    if (!navigator.mediaDevices?.getUserMedia) {
        throw new Error('Camera access needs HTTPS (or localhost) in a current browser.');
    }
    const h = s.res === '1080' ? 1080 : 720;
    const video = { width: { ideal: Math.round(h * 16 / 9) }, height: { ideal: h } };
    if (s.camera.startsWith('device:')) video.deviceId = { exact: s.camera.slice(7) };
    else video.facingMode = { ideal: s.camera.slice(7) || 'environment' };
    let stream;
    try {
        stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    } catch (e) {
        if (!video.deviceId) throw e;
        const savedLabel = localStorage.getItem(CAMERA_LABEL_KEY);
        const same = (await videoInputs()).find(d => d.label && d.label === savedLabel);
        if (same && same.deviceId !== video.deviceId.exact) {
            log(`saved camera id gone (${e.name}); using "${same.label}"`);
            video.deviceId = { exact: same.deviceId };
        } else {
            log(`saved camera unavailable (${e.name}); using default`);
            delete video.deviceId;
        }
        stream = await navigator.mediaDevices.getUserMedia({ video, audio: false });
    }
    const track = stream.getVideoTracks()[0];
    const st = track?.getSettings() || {};
    if (track?.label) localStorage.setItem(CAMERA_LABEL_KEY, track.label);
    log(`camera "${track?.label}" ${st.width}x${st.height} facing=${st.facingMode || '?'}`);
    videoInputs().then(devs => log(`cameras: ${devs.map(d => `"${d.label}"`).join(', ')}`)).catch(() => {});
    return stream;
}

// Takes over the preview's camera if it matches the settings, otherwise opens it again.
async function startCamera() {
    previewSeq++;   // a preview still opening must not take over
    els.previewVideo.srcObject = null;
    if (!camLive() || camKey !== cameraKey(cfg)) {
        stopCamera();
        camStream = await openCamera(cfg);
        camKey = cameraKey(cfg);
    }
    els.video.srcObject = camStream;
    await els.video.play();
    els.view.classList.toggle('mirror', cfg.mirror);
    listCameras();
}

const MIC_MISSING = 'No microphone found. Connect a microphone or headset; a phone used as a webcam does not send audio.';

function setPreviewNote(text, warn = false) {
    els.previewNote.textContent = text;
    els.previewNote.classList.toggle('warn', warn);
    els.previewNote.classList.toggle('hidden', !text);
}

// Start screen: show the camera from the current settings. Also asks for mic access here,
// so both permission prompts come before Start; the mic is closed again straight away.
async function startPreview() {
    if (step !== 'idle') return;
    const s = readSettings();
    els.preview.classList.toggle('mirror', s.mirror);
    if (camLive() && camKey === cameraKey(s)) return;
    const seq = ++previewSeq;
    stopCamera();
    els.previewVideo.srcObject = null;
    setPreviewNote('Starting camera…');
    let stream;
    try {
        stream = await openCamera(s);
    } catch (e) {
        if (seq !== previewSeq) return;
        setPreviewNote(e.name === 'NotAllowedError'
            ? 'Camera access is blocked. Allow it in the browser\'s site settings, then reload.'
            : `Camera unavailable: ${e.message || e.name}` + (e.name === 'NotReadableError' ? ' (is another app or tab using it?)' : ''), true);
        return;
    }
    if (seq !== previewSeq || step !== 'idle') { stream.getTracks().forEach(t => t.stop()); return; }
    camStream = stream;
    camKey = cameraKey(s);
    els.previewVideo.srcObject = stream;
    setPreviewNote('');
    listCameras();
    try {
        (await navigator.mediaDevices.getUserMedia({ audio: true, video: false })).getTracks().forEach(t => t.stop());
    } catch (e) {
        if (seq !== previewSeq) return;
        if (e.name === 'NotFoundError') setPreviewNote(MIC_MISSING, true);
        else if (e.name === 'NotAllowedError') setPreviewNote('Microphone access is blocked; the assistant needs it to hear you.', true);
    }
}

async function startMic() {
    micStream = await navigator.mediaDevices.getUserMedia({
        audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true, autoGainControl: true },
        video: false,
    });
    if (micCtx.state === 'suspended') await micCtx.resume();
    await micCtx.audioWorklet.addModule('/static/duplex/lib/capture-processor.js');
    const src = micCtx.createMediaStreamSource(micStream);
    captureNode = new AudioWorkletNode(micCtx, 'capture-processor', { processorOptions: { chunkSize: SAMPLE_RATE_IN } });
    // The worklet passes audio through; route it to a muted sink so the graph keeps pulling it.
    sinkNode = micCtx.createGain();
    sinkNode.gain.value = 0;
    src.connect(captureNode);
    captureNode.connect(sinkNode);
    sinkNode.connect(micCtx.destination);
    captureNode.port.onmessage = e => { if (e.data.type === 'chunk') onMicChunk(e.data.audio); };
    micAnalyser = micCtx.createAnalyser();
    micAnalyser.fftSize = 1024;
    src.connect(micAnalyser);
    noiseFloor = 0; lastUserVoiceAt = -Infinity; userVoiceOn = false;
    voiceTimer = setInterval(pollVoice, VOICE.pollMs);
}

// Level only, no recognition. The noise floor follows quiet stretches down fast and
// rises slowly, so steady room noise does not count as the user talking.
function pollVoice() {
    if (!micAnalyser) return;
    const buf = new Float32Array(micAnalyser.fftSize);
    micAnalyser.getFloatTimeDomainData(buf);
    let sum = 0;
    for (const x of buf) sum += x * x;
    const rms = Math.sqrt(sum / buf.length);
    noiseFloor = noiseFloor === 0 || rms < noiseFloor ? rms : noiseFloor + (rms - noiseFloor) * 0.005;
    const voice = rms > Math.max(VOICE.minRms, noiseFloor * VOICE.floorRatio);
    if (voice) lastUserVoiceAt = now();
    const on = userTalking();
    if (on !== userVoiceOn) { userVoiceOn = on; if (cfg.debug) log(on ? 'user: talking' : 'user: quiet'); }
    if (!on) maybeReset();
}

const userTalking = () => now() - lastUserVoiceAt < FLOW.userQuietMs;

function onMicChunk(audio) {
    if (!session?.running || closingDone) return;
    const msg = { type: 'audio_chunk', audio_base64: arrayBufferToBase64(audio.buffer) };
    if (camStream && session.chunksSent % cfg.frameEvery === 0) {
        const why = frameReason();
        const f = why && grabSendFrame();
        if (f) {
            msg.frame_base64_list = [f];
            sent = { at: now(), lm: poseLm, step, codes: liveCodes.join(','), deg: turn?.absDeg ?? 0 };
            frameStats.sent++;
            if (cfg.debug) console.debug(`[fit] frame (${why})`);
        } else {
            frameStats.skipped++;
        }
    }
    if (cfg.slices > 1) msg.max_slice_nums = cfg.slices;
    session.sendChunk(msg);
    maybeReset();
}

// Why this chunk should carry a frame, or '' to send audio only.
function frameReason() {
    if (!cfg.frameGate || !sent) return 'always';
    if (now() - sent.at >= GATE.keepaliveMs) return 'keepalive';
    if (step !== sent.step) return 'step';
    if (liveCodes.join(',') !== sent.codes) return 'tracker';
    if (!tracker?.ready) return 'no tracker';
    if (!poseLm || !sent.lm) return poseLm !== sent.lm ? 'person' : '';
    if (step === 'recording' && Math.abs((turn?.absDeg ?? 0) - sent.deg) >= GATE.turnDeg) return 'turn';
    return poseShift(poseLm, sent.lm) >= GATE.poseMove ? 'pose' : '';
}

// Mean shift of the main landmarks, relative to body height.
function poseShift(a, b) {
    let sum = 0;
    for (const i of GATE_LM) sum += Math.hypot(a[i].x - b[i].x, a[i].y - b[i].y);
    const h = Math.max(0.2, Math.max(a[27].y, a[28].y) - a[0].y);
    return sum / GATE_LM.length / h;
}

// Scale the upright frame to at most SEND_MAX_H tall for the agent.
function grabSendFrame() {
    const fw = els.frame.width, fh = els.frame.height;
    if (!fw || !fh) return null;
    const scale = Math.min(1, SEND_MAX_H / fh);
    const w = Math.round(fw * scale), h = Math.round(fh * scale);
    if (els.sendCanvas.width !== w || els.sendCanvas.height !== h) { els.sendCanvas.width = w; els.sendCanvas.height = h; }
    sctx.drawImage(els.frame, 0, 0, w, h);
    return els.sendCanvas.toDataURL('image/jpeg', SEND_JPEG_Q).split(',')[1];
}

function releaseMedia() {
    cancelAnimationFrame(rafId);
    rafId = 0;
    try { captureNode?.port.postMessage({ command: 'stop' }); } catch (_) {}
    captureNode?.disconnect();
    sinkNode?.disconnect();
    captureNode = sinkNode = null;
    clearInterval(voiceTimer);
    micAnalyser?.disconnect();
    micAnalyser = null;
    micCtx?.close().catch(() => {});
    micCtx = null;
    micStream?.getTracks().forEach(t => t.stop());
    stopCamera();
    canvasStream?.getTracks().forEach(t => t.stop());
    micStream = canvasStream = null;
    els.video.srcObject = null;
    tracker?.clearOverlay();
}

function endSession() {
    clearTimeout(lingerTimer);
    clearInterval(tickTimer);
    const s = session;
    session = null;
    if (s) log(`session: ${Math.round((now() - t0) / 1000)}s, ${resets} resets, frames ${frameStats.sent} sent / ${frameStats.skipped} skipped`);
    s?.stop();
    releaseMedia();
}

// ── Per-frame loop ──────────────────────────────────────────────────────────

function drawPlainFrame(v) {
    if (els.frame.width !== v.videoWidth || els.frame.height !== v.videoHeight) {
        els.frame.width = v.videoWidth; els.frame.height = v.videoHeight;
    }
    fctx.drawImage(v, 0, 0);
}

function loop(t) {
    rafId = requestAnimationFrame(loop);
    const v = els.video;
    if (v.readyState < 2 || !v.videoWidth) return;
    if (!tracker?.ready) {
        drawPlainFrame(v);
        if (ACTIVE_STEPS.has(step)) onFrameNoTracker();
        return;
    }
    const obs = tracker.process(v, t);
    if (!obs || obs.rotated) return;   // the frame changed orientation under us; wait for the next one
    poseLm = obs.lm;
    if (cfg.debug) {
        const a = obs.arms;
        const fmt = s => s ? `${s.abd.toFixed(0)}°/${s.elbow.toFixed(0)}°` : '—';
        els.chipStep.textContent = `${step} · ${obs.framing.code} · L${fmt(a?.left)} R${fmt(a?.right)} · rot${tracker.rotation}`;
    }
    if (ACTIVE_STEPS.has(step)) onFrame(obs, t);
}

function onFrameNoTracker() {
    if (step === 'positioning' || step === 'arms') setBanner('Follow the assistant');
}

function onFrame(obs, t) {
    const fr = obs.framing;
    lastFr = fr;
    if (fr.ok) lastFramingOkAt = t;
    const trackerDrives = cfg.driver === 'tracker' || agentLost;

    if (step === 'positioning') {
        noteTrackerCodes(fr.ok ? ['in_view'] : [fr.code]);
        if (!fr.ok) {
            if (framingOkSince !== null && t - lastFramingOkAt < FLOW.dropoutMs) return;
            framingOkSince = null;
            setBanner(fr.msg);
            return;
        }
        framingOkSince ??= t;
        setBanner('Hold still…');
        if (t - framingOkSince >= FLOW.readyHoldMs) goStep('arms');
        return;
    }

    if (step === 'arms' || step === 'countdown') {
        if (!fr.ok) {
            if (t - lastFramingOkAt > FLOW.lostStepMs) {
                els.countdown.classList.add('hidden');
                goStep('positioning');
            } else setBanner(fr.msg);
            noteTrackerCodes([fr.code]);
            return;
        }
        const ac = obs.armCheck;
        noteTrackerCodes(ac.ok ? ['in_view', 'arms_ok'] : ['in_view', ac.code]);

        if (step === 'arms') {
            setBanner(ac.ok ? 'Great, hold that pose' : ac.msg);
            if (!trackerDrives) return;
            if (!ac.ok) { armsOkSince = null; return; }
            armsOkSince ??= t;
            if (t - armsOkSince >= FLOW.armsHoldMs) {
                goStep('countdown', { reset: false });
                countdownStart = t;
                lastCount = null;
                els.countdown.classList.remove('hidden');
            }
            return;
        }

        // countdown (tracker driver only)
        const remaining = FLOW.countdownSec - Math.floor((t - countdownStart) / 1000);
        if (remaining <= 0) startRecording();
        else if (remaining !== lastCount) {
            lastCount = remaining;
            els.countdown.textContent = remaining;
            setBanner(ac.ok ? 'Get ready to turn' : ac.msg);
        }
        return;
    }

    if (step === 'recording') trackTurn(obs, t);
}

// ── Recording ───────────────────────────────────────────────────────────────

function setProgress(d) {
    els.progressFill.style.width = Math.min(100, d / 360 * 100) + '%';
    els.progressText.textContent = `${Math.min(360, Math.round(d))}° / 360°`;
}

function startRecording() {
    if (step === 'recording') return;
    els.countdown.classList.add('hidden');
    recChunks = [];
    stopReason = '';
    turn?.reset();
    if (tracker) tracker.allowRotationScan = false;   // never rotate mid-recording
    mimeType = ['video/mp4', 'video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm']
        .find(m => MediaRecorder.isTypeSupported(m)) || '';
    // Record the upright canvas, not the raw camera stream, so sideways capture still saves upright.
    canvasStream = els.frame.captureStream(30);
    const opts = { videoBitsPerSecond: 8_000_000, ...(mimeType ? { mimeType } : {}) };
    recorder = new MediaRecorder(canvasStream, opts);
    recorder.ondataavailable = e => { if (e.data?.size) recChunks.push(e.data); };
    recorder.onstop = finishRecording;
    recorder.start(500);
    recordStart = now();
    els.chipRec.classList.remove('hidden');
    setProgress(0);
    setBanner('Now turn slowly');
    goStep('recording');
}

function trackTurn(obs, t) {
    const el = t - recordStart;
    if (el > cfg.maxRecSec * 1000) { stopRecording('max_length'); return; }
    if (turn.completedAt && t - turn.completedAt > FLOW.maxFinishMs) { stopRecording('auto'); return; }

    const r = turn.update(obs.lm, obs.wl, obs.arms, t);
    setProgress(r.absDeg);
    if (r.lost) {
        setBanner('Keep your body in view');
        if (r.lostMs > 1500) noteTrackerCodes(['lost']);
        return;
    }
    const codes = [];
    if (r.swaying) codes.push('swaying');
    if (r.drooping) codes.push('drooping');
    noteTrackerCodes(codes);

    if (r.swaying) setBanner('Try to stay in one spot');
    else if (r.drooping) setBanner('Keep your arms out');
    else if (r.completed) setBanner('Stop and face the camera');
    else if (r.absDeg < 15 && el > 1500) setBanner('Now turn slowly');
    else setBanner('Keep turning…');

    // Auto-stop: turn complete and the body has been still for a moment.
    if (r.completed && el > 1000 && r.still) stopRecording('auto');
}

function stopRecording(reason) {
    if (step !== 'recording') return;
    stopReason = reason;
    goStep('finishing', { reset: false });
    els.chipRec.classList.add('hidden');
    setBanner('Saving…');
    recorder.stop();   // fires onstop → finishRecording
}

function finishRecording() {
    const type = recorder?.mimeType || mimeType || 'video/webm';
    const ext = type.includes('mp4') ? 'mp4' : 'webm';
    const blob = new Blob(recChunks, { type: type.split(';')[0] });
    const secs = (now() - recordStart) / 1000;
    const deg = turn ? turn.absDeg : null;
    const completed = deg === null ? stopReason === 'agent' : deg >= 340;
    canvasStream?.getTracks().forEach(t => t.stop());
    canvasStream = null;

    const failed = stopReason === 'timeout';
    showResult(blob, ext, [
        `Length: ${secs.toFixed(1)} s`,
        deg === null ? 'Turn: not tracked' : `Turn: ${Math.min(360, Math.round(deg))}°${completed ? ' (complete)' : ' (incomplete, consider recording again)'}`,
        turn ? `Max sway: ${(turn.maxSway * 100).toFixed(1)}% of frame · arm droops: ${turn.droops}` : '',
        {
            auto: 'Stopped automatically after the turn',
            agent: 'Stopped by the assistant',
            manual: 'Stopped manually',
            max_length: 'Stopped at the recording length limit',
            timeout: 'Stopped: overall time limit reached',
        }[stopReason] || '',
        `Attempts: ${attempt} · context resets: ${resets}`,
        `Format: ${ext.toUpperCase()}, ${(blob.size / 1048576).toFixed(1)} MB`,
    ], failed ? 'Time is up' : 'Thank you!');
    goStep(failed ? 'failed' : 'done', { reset: false });
    linger(failed);
}

// Speak a fixed closing line, mute the model, and end the session once it has played.
function linger(failed) {
    stopCamera();
    cancelAnimationFrame(rafId);
    clearTimeout(lingerTimer);
    pendingReset = null;    // the context no longer matters, and a reset would drop the line
    lineAfterReset = null;
    lingering = true; closingDone = false;
    const spoke = session?.running && say(pickLine(failed ? FAILED_LINES : CLOSING_LINES), { muteAfter: true });
    // tick() ends the session once the closing line has played; this is the backstop.
    lingerTimer = setTimeout(endSession, spoke ? FLOW.lingerMaxMs : 0);
}

function showResult(blob, ext, lines, title) {
    els.capture.classList.add('hidden');
    els.result.classList.remove('hidden');
    els.resultTitle.textContent = title;
    if (lastUrl) URL.revokeObjectURL(lastUrl);
    if (blob && blob.size) {
        lastUrl = URL.createObjectURL(blob);
        els.playback.src = lastUrl;
        els.playback.classList.remove('hidden');
        els.download.classList.remove('hidden');
        els.download.href = lastUrl;
        els.download.download = `fit-capture-${new Date().toISOString().replace(/[:.]/g, '-')}.${ext}`;
    } else {
        lastUrl = null;
        els.playback.removeAttribute('src');
        els.playback.classList.add('hidden');
        els.download.classList.add('hidden');
    }
    els.stats.textContent = lines.filter(Boolean).join('\n');
}

// ── Timeout ─────────────────────────────────────────────────────────────────

function tick() {
    const e = elapsedSec();
    setChip(els.chipTime, `${Math.floor(e)}s / ${cfg.timeoutSec}s`, e > cfg.timeoutSec - 15 ? 'warn' : '');
    if (session?.running && !agentLost) {
        const audible = agentAudible();
        if (audible !== els.chipAgent.classList.contains('speaking')) {
            setChip(els.chipAgent, audible ? 'agent: speaking' : 'agent: listening', audible ? 'speaking' : '');
        }
    }
    maybeReset();
    if (closingDone && now() - closingDoneAt > FLOW.sayTailMs && !agentAudible()) { endSession(); return; }
    // trackTurn() enforces this too, but only runs when the pose tracker is loaded.
    if (step === 'recording' && now() - recordStart > cfg.maxRecSec * 1000) { stopRecording('max_length'); return; }
    if (!ACTIVE_STEPS.has(step) || e < cfg.timeoutSec) return;
    log('⏱ failure timeout');
    if (step === 'recording') { stopRecording('timeout'); return; }
    els.countdown.classList.add('hidden');
    showResult(null, '', [`No complete recording within ${cfg.timeoutSec} s.`, `Attempts: ${attempt}`], 'Time is up');
    goStep('failed', { reset: false });
    linger(true);
}

// ── Buttons ─────────────────────────────────────────────────────────────────

function showError(msg) {
    els.capture.classList.add('hidden');
    els.result.classList.add('hidden');
    els.intro.classList.remove('hidden');
    els.settings.classList.remove('hidden');
    els.preview.classList.remove('hidden');
    els.startBtn.disabled = false;
    els.error.textContent = msg;
    els.error.classList.toggle('hidden', !msg);
    startPreview();
}

function cancelAll(msg = '') {
    log(`cancel ${msg}`);
    if (recorder && recorder.state !== 'inactive') { recorder.onstop = null; recorder.stop(); }
    recorder = null;
    step = 'idle';
    endSession();
    showError(msg);
}

els.startBtn.onclick = async () => {
    cfg = readSettings();
    saveSettings(cfg);
    els.startBtn.disabled = true;
    els.debugLog.textContent = '';
    els.debugLog.classList.toggle('hidden', !cfg.debug);

    // Audio contexts must be created inside the tap on iOS.
    session = createSession();
    session.audioPlayer.init();
    session.audioPlayer._ctx?.resume?.();
    micCtx = new AudioContext({ sampleRate: SAMPLE_RATE_IN });
    micCtx.resume().catch(() => {});

    step = 'connecting';
    attempt = 1; resets = 0; note = '';
    lingering = closingDone = sayInFlight = false; kv = 0; agentLost = false;
    pendingReset = null; resetInFlight = false; lastResetAt = 0; lineAfterReset = null;
    candCodes = promptCodes = ''; liveCodes = []; lastFramingOkAt = -Infinity; lastFr = null;
    poseLm = sent = null; frameStats = { sent: 0, skipped: 0 }; lastAudibleAt = -Infinity;
    els.error.classList.add('hidden');
    els.intro.classList.add('hidden');
    els.settings.classList.add('hidden');
    els.preview.classList.add('hidden');
    els.capture.classList.remove('hidden');
    els.subtitle.textContent = '';
    setChip(els.chipStep, 'connecting');
    setChip(els.chipAgent, 'agent: —');
    setChip(els.chipKv, 'kv 0');
    setProgress(0);

    try {
        setBanner('Starting camera…');
        await startCamera();
        await startMic();
        rafId = requestAnimationFrame(loop);

        setBanner('Loading…');
        const [ref] = await Promise.all([refAudio ? refAudio : fetchRefAudio(), loadTracker()]);
        refAudio = ref;

        setBanner('Connecting to the assistant…');
        const s = session;
        // The cpp model's default listen bias (1.0) leaves it silent until someone speaks to it,
        // and it often ignores speech from across the room. In a quiet room 0 says little;
        // −0.5 coaches unprompted but talks over the user, and −1 gets chatty and claims turns
        // that never happened. The default is 0 now that the page speaks the opening and step lines.
        const payload = { max_slice_nums: cfg.slices, use_tts: true, config: { listen_prob_scale: cfg.listenScale } };
        if (refAudio) payload.ref_audio_base64 = refAudio;
        await s.start(buildPrompt({ ...promptState(), step: 'positioning', resets: 0 }), payload, async () => {
            captureNode.port.postMessage({ command: 'start' });
        });
        if (s !== session) return;   // cancelled while connecting
        t0 = now();
        step = 'positioning';
        setChip(els.chipStep, step);
        setBanner('Step back until your whole body is in view');
        // Open with a fixed line rather than waiting for the model to decide to speak.
        say(pickLine(OPENING_LINES));
        tickTimer = setInterval(tick, 500);
        log(`session ${s.sessionId} driver=${cfg.driver} tracker=${tracker?.ready ? 'on' : 'off'}`);
    } catch (err) {
        console.error(err);
        const hint = err.name === 'NotReadableError' ? ' (is another app or tab using the camera?)' : '';
        if (err.name === 'NotFoundError' && camStream && !micStream) {
            // A phone in webcam mode sends video only; the mic has to come from elsewhere.
            err = new Error(MIC_MISSING);
        }
        if (step !== 'idle') cancelAll('Could not start: ' + (err.message || err.name) + hint);
    }
};

async function loadTracker() {
    if (tracker?.ready) { turn?.reset(); tracker.allowRotationScan = true; return; }
    try {
        PT = PT || await import('./pose-tracker.js');
        tracker = tracker || new PT.PoseTracker({ frameCanvas: els.frame, overlayCanvas: els.overlay, debug: cfg.debug });
        setBanner('Loading pose model (first time only)…');
        await tracker.load();
        turn = new PT.TurnTracker();
    } catch (e) {
        log(`⚠ pose tracker unavailable: ${e.message}`);
        tracker = null;
        turn = null;
        if (cfg.driver === 'tracker') throw new Error('Pose tracker failed to load: ' + e.message);
    }
}

els.stopBtn.onclick = () => {
    if (step === 'recording') stopRecording('manual');
    else cancelAll();
};

els.restartBtn.onclick = () => {
    if (ACTIVE_STEPS.has(step)) restartAttempt('button');
};

els.again.onclick = () => {
    els.result.classList.add('hidden');
    if (lastUrl) { URL.revokeObjectURL(lastUrl); lastUrl = null; }
    endSession();
    step = 'idle';
    showError('');
};

// ── Init ────────────────────────────────────────────────────────────────────

loadSettings();
listCameras();
startPreview();
// Camera, resolution or mirror changed: update the preview (a no-op for other settings).
els.settings.addEventListener('change', () => startPreview());
window.fitCapture = { get session() { return session; }, get step() { return step; }, get tracker() { return tracker; }, buildPrompt, promptState };
