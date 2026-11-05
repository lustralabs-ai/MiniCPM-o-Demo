/**
 * prompts.js — system prompts for the fit-capture agent.
 *
 * The cpp backend's context is 8192 tokens and every 1 s chunk costs roughly
 * 64 vision tokens plus audio. The page therefore resets the LLM context
 * (session.reset) at step changes and when the KV fills up. Each reset
 * re-prefills a prompt built from the current protocol state, so the prompt
 * has to carry everything the agent needs to pick up where it left off.
 *
 * Markers are single emoji at the start of a reply. The cpp backend strips
 * emoji before TTS (OMNI_TTS_STRIP_EMOJI). Everything here is English: Chinese
 * tags tend to switch the model into Chinese.
 *
 * The opening and closing lines are spoken verbatim through session.say (forced
 * speech, see fit-capture.js) instead of being left to the model, which often
 * failed to start the conversation and kept talking after the capture.
 *
 * With the tracker driver the page also announces every step change with a fixed
 * line (STEP_LINES), and the prompt tells the model the app owns the steps. Left to
 * itself the model judged progress from its own 1 fps view and talked the user
 * through steps the tracker had not passed, so voice and screen disagreed.
 */

// Spoken at session start. Each greets and gives step 1, matching STEP_TEXT.positioning.
export const OPENING_LINES = [
    // One sentence each: the model cannot answer the user while a forced line is playing.
    "Hi, I'm Fit. Step back until I can see all of you.",
    "Hello! Step back until your whole body is in view.",
    "Hi there! Step back so I can see you head to toe.",
    "Hi! Please step back until you're fully in view.",
    "Hey, I'm Fit. Step back until I see your feet.",
];

// Spoken when the recording is saved; the model is muted afterwards.
export const CLOSING_LINES = [
    "That's it, all done. Thank you!",
    "Perfect, your video is saved. Thanks!",
    "Great job, we're all done. Thank you!",
    "All done, thanks for your help!",
];

// Spoken when the time limit ends the session without a complete recording.
export const FAILED_LINES = [
    "Sorry, we ran out of time. Press Start to try again.",
    "Time's up, sorry. You can press Start to have another go.",
];

// Spoken when the page changes step (tracker driver; recording also with the agent driver).
// Short, since each 12 characters takes about a second and blocks the next step's reset.
export const STEP_LINES = {
    arms: [
        "Great. Now hold both arms out, like a Christmas tree.",
        "Good. Now hold your arms out, angled down.",
        "That's it. Now arms out straight, a little away from you.",
    ],
    lost: [
        "I can't see all of you any more. Please step back into view.",
        "Step back a little, I've lost part of you.",
    ],
    restart: [
        "OK, let's start again. Step back until I can see all of you.",
        "No problem, from the top. Step back until you're fully in view.",
    ],
    countdown: ["Perfect, hold that.", "Great, keep that pose."],
    recording: [
        "Now turn slowly in place, all the way round.",
        "Start turning now, nice and slow.",
        "Now turn slowly, one full circle.",
    ],
};

export const pickLine = lines => lines[Math.floor(Math.random() * lines.length)];

export const MARKERS = {
    start:   { emoji: '🎬', alt: ['🎥'], desc: 'the user is in position with arms out, and you are telling them to start turning now' },
    done:    { emoji: '✅', alt: ['✔'], desc: 'the user has finished the full turn' },
    restart: { emoji: '🔄', alt: ['🔁'], desc: 'start over from the beginning' },
    stop:    { emoji: '🛑', alt: ['⛔'], desc: 'the user wants to quit' },
};

// What the agent should be doing at each step. Keep each one to a sentence or two.
// Agent driver: the model moves the user on itself (markers).
const STEP_TEXT = {
    positioning: 'Step 1 of 3. Help the user step back until their whole body, head to feet, is in view and centred.',
    arms: 'Step 2 of 3. The user is in view. Ask them to hold both arms out straight, angled down and away from the body like a Christmas tree.',
    countdown: 'Step 3 of 3 is about to begin. Tell the user to get ready to turn.',
    recording: 'Step 3 of 3. Recording is running. Encourage one slow, steady full turn in place. Only speak if they need a reminder: keep arms out, feet in one spot, finish facing the camera.',
    done: 'The recording is complete and saved. Thank the user warmly in one sentence. Do not ask anything.',
    failed: 'Time is up and the recording did not complete. Apologise in one sentence and tell them they can press Start to try again.',
};

// Tracker driver: the app moves the user on and announces it; the model only helps within the step.
const TRACKER_STEP_TEXT = {
    positioning: 'Step 1 of 3: the user is stepping back until their whole body, head to feet, is in view and centred. Help them with that, using the pose tracker below.',
    arms: 'Step 2 of 3: the user is holding both arms out straight, angled down and away from the body. Help them get the arms right, using the pose tracker below.',
    countdown: 'Step 2 of 3: the arms are right and the app is about to start recording. Say nothing.',
    recording: 'Step 3 of 3: recording is running and the user is turning slowly in place. Only speak if they need a reminder from the pose tracker below: keep arms out, feet in one spot.',
    done: STEP_TEXT.done,
    failed: STEP_TEXT.failed,
};

const TRACKER_RULES =
    'The app watches the pose tracker and decides when a step is complete. It tells the user itself when to move on, and it starts and stops the recording. ' +
    'Trust the pose tracker over your own view of the picture. ' +
    'Never say that a step is complete, never tell the user to move on to the next step or to start turning, and never say the turn is finished. ' +
    'If the pose tracker reports nothing wrong, stay quiet.';

// Tracker findings, keyed by pose-tracker codes. Phrased as facts for the agent.
const TRACKER_TEXT = {
    no_person: 'No person visible.',
    feet_hidden: 'Feet are not in view.',
    body_hidden: 'Head or body is not fully in view.',
    head_edge: 'Head is at the top edge of the frame.',
    feet_edge: 'Feet are at the bottom edge of the frame.',
    too_far: 'The user is too far away; they should come a little closer.',
    off_centre: 'The user is not centred in the frame.',
    arms_hidden: 'Arms are not clearly visible.',
    arms_bent: 'Elbows are bent; arms should be straight.',
    arms_low: 'Arms are too close to the body; they should move further out.',
    arms_high: 'Arms are too high; they should come down a little.',
    swaying: 'The user is drifting from their spot.',
    drooping: 'Arms have drooped during the turn.',
    lost: 'The tracker lost sight of the body.',
    in_view: 'Whole body is in view.',
    arms_ok: 'Arm position is good.',
};

/**
 * @param {object} st
 *   step: positioning | arms | countdown | recording | done | failed
 *   driver: 'agent' | 'tracker'
 *   tracker: array of TRACKER_TEXT codes (may be empty)
 *   turnDeg: degrees turned so far (recording only)
 *   attempt: 1-based attempt number
 *   elapsedSec, timeoutSec: seconds since Start, and the overall limit
 *   resets: how many context resets so far
 *   note: optional one-off event, e.g. why a start request was refused
 */
export function buildPrompt(st) {
    const lines = [];
    lines.push(
        'You are Fit, a friendly voice assistant helping someone record a short video of their body for clothing-size measurement. ' +
        'You can see the camera and hear the room. The phone is on a stand or held by a friend, so you may be talking to either of them. ' +
        'Always speak English, in one or two short sentences. Never comment on the person\'s body, weight or looks.'
    );
    const agentDrives = st.driver === 'agent';
    lines.push(
        'How it works: 1. Step back until the whole body, head to feet, is in view. ' +
        '2. Hold both arms out straight, 20 to 45 degrees from the body, like a Christmas tree. ' +
        '3. Turn slowly in place through one full circle, keeping arms out and feet in one spot, and finish facing the camera. ' +
        'The recording stops automatically.'
    );

    if (agentDrives) {
        lines.push(
            'Signals: start a reply with exactly one of these emoji when the event happens. They are silent signals for the app and are never spoken. ' +
            Object.values(MARKERS).map(m => `${m.emoji} means ${m.desc}.`).join(' ')
        );
    } else {
        lines.push(
            TRACKER_RULES + ' ' +
            `If the user asks to start over, begin your reply with ${MARKERS.restart.emoji}. If they want to quit, begin with ${MARKERS.stop.emoji}. These emoji are silent signals, never spoken.`
        );
    }

    const texts = agentDrives ? STEP_TEXT : TRACKER_STEP_TEXT;
    let stepText = texts[st.step] || texts.positioning;
    // Probe runs: the model emits 🎬 unprompted but tends to forget ✅ unless it is named for this step.
    if (agentDrives && st.step === 'recording') {
        stepText += ` When they have turned all the way round and face the camera again, begin your reply with ${MARKERS.done.emoji}.`;
    } else if (agentDrives && st.step === 'arms') {
        stepText += ` When their arms are right and they are ready, begin your reply with ${MARKERS.start.emoji} and tell them to start turning.`;
    }
    lines.push(`Current step: ${stepText}`);

    const facts = (st.tracker || []).map(c => TRACKER_TEXT[c]).filter(Boolean);
    if (st.step === 'recording' && st.turnDeg != null) facts.push(`Turned ${Math.round(st.turnDeg)} of 360 degrees.`);
    if (facts.length) lines.push(`Pose tracker: ${facts.join(' ')}`);

    const progress = [];
    if (st.attempt > 1) progress.push(`This is attempt ${st.attempt}.`);
    if (st.timeoutSec) progress.push(`${Math.round(st.elapsedSec || 0)} of ${st.timeoutSec} seconds used.`);
    if (st.note) progress.push(st.note);
    if (progress.length) lines.push(progress.join(' '));

    if (st.resets > 0) {
        // Quoting the agent's last line here made it repeat that line verbatim after the reset.
        // With the tracker driver the page speaks a STEP_LINES line right after a step reset.
        lines.push('The conversation is already under way; do not greet or introduce yourself again, and do not repeat earlier instructions.');
    } else {
        // The page has just spoken an OPENING_LINES greeting as the agent (session.say).
        lines.push('You have already greeted the user and asked them to step back; do not greet them again. Speak when they need help.');
    }
    return lines.join('\n');
}

const MARKER_LOOKUP = new Map();
for (const [name, m] of Object.entries(MARKERS)) {
    MARKER_LOOKUP.set(m.emoji, name);
    for (const a of m.alt) MARKER_LOOKUP.set(a, name);
}

/** All marker names found in text, in order. */
export function findMarkers(text) {
    const out = [];
    for (const ch of text) {
        const name = MARKER_LOOKUP.get(ch);
        if (name) out.push(name);
    }
    return out;
}

/** Text with emoji (and their joiners/selectors) and stray <think> tags removed, for subtitles. */
export function stripEmoji(text) {
    return text.replace(/<\/?think>/g, '').replace(/[\p{Extended_Pictographic}\u{FE0F}\u{200D}\u{20E3}]/gu, '').replace(/\s{2,}/g, ' ').trim();
}

// Optional fallback for experiments: phrases that imply a marker the model forgot to emit.
const PHRASES = {
    start: /\b(start|begin) (turning|to turn|your turn)\b|\bturn (slowly|around) now\b/i,
    done: /\b(you'?re|all) done\b|\bthat was (perfect|great)\b.*\bturn\b/i,
};

export function findPhraseMarkers(text) {
    return Object.entries(PHRASES).filter(([, re]) => re.test(text)).map(([name]) => name);
}
