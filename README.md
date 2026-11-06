# MiniCPM-o demo for Lustra: the fit-capture voice guide

This is the lustra fork of the [MiniCPM-o 4.5 demo](https://github.com/OpenBMB/MiniCPM-o) (upstream
[docs](https://minicpmo45.modelbest.cn/docs/en/), [Realtime API](https://minicpmo45.modelbest.cn/docs/en/realtime-api/overview/),
[中文简介](README_zh.md)). It runs the MiniCPM-o 4.5 omni model on the llama.cpp-omni backend.
Lustra uses it for one job: **a voice guide that talks a person through recording a turn video**
for clothing-size measurement. It sees the camera at about one frame a second, hears the room and
speaks back.

The guide is used in two places:

- **Standalone**, at `/fit-capture/` on the gateway. This is the prototype the flow was developed
  on. It records, plays back and downloads the video in the browser.
- **From Drape** (`~/lustra/drape`), where the same flow is part of tester enrolment. Drape's api
  relays the browser's session to this gateway; see [Using it from Drape](#using-it-from-drape).

Contents: [Running it](#running-it) · [Using it from Drape](#using-it-from-drape) ·
[The fit-capture page](#the-fit-capture-page) · [Privacy: what is kept on disk](#privacy-what-is-kept-on-disk) ·
[Configuration](#configuration) · [Project structure](#project-structure)

## Running it

`omni.sh` starts the three components from local checkouts, without Docker:

```text
llama-omni-server :22501  ->  worker :22401  ->  gateway :8016 (internal :8017)
   (the model)                (protocol)         (HTTP, WebSocket, the pages)
```

```sh
./omni.sh start      # backend, then worker, then gateway; waits for each to be healthy
./omni.sh status     # components, and whether sessions are being recorded
./omni.sh logs gateway
./omni.sh stop
```

Paths, ports and options are environment variables listed at the top of `omni.sh`. The defaults
expect `~/lustra/llama.cpp-omni` built (`build/bin/llama-omni-server`), the Q8_0 GGUF in
`~/models/MiniCPM-o-4_5-gguf/`, the token2wav assets in `~/models/MiniCPM-o-4.5/assets/`, and a
Python environment in `~/py` with `requirements.txt` installed.

`omni.sh` is cpp-only: fit-capture needs two llama.cpp-omni features that the PyTorch backend
lacks: same-session context resets (`session.reset`) and forced speech (`session.say`, documented
in llama.cpp-omni `docs/omni-duplex-say.md`). The Docker Compose files from upstream are still here
but are not maintained for this use.

**One session at a time.** There is one worker, so a second person who connects is queued
(`session.queued`) until the first session ends.

### Reaching it from other machines

The gateway binds to 127.0.0.1 only. To serve it over HTTPS on the tailnet:

```sh
sudo tailscale serve --bg 8016
```

It is then at `https://aleatoric.tailfc2739.ts.net/` (this machine's tailnet name). Phones on the
tailnet can open `/fit-capture/` there; browsers allow camera and microphone only over HTTPS.

## Using it from Drape

Drape's tester app (`apps/tester-web`) has a guided turn video on `/videos/turn`. With the voice
guide on, the browser opens a WebSocket to Drape's own api, which signs the tester in, checks her
consent, and relays the session to this gateway. Browsers never connect here directly (Drape ADR
0001), and nothing identifying is forwarded: no cookies, client ids, user agent or Drape ids.

On the Drape side (see Drape's `docs/development.md`):

| Drape setting | Meaning |
|---|---|
| `CAPTURE_AGENT=minicpm` | The default. The voice guide is offered, with the on-device pose tracker driving the steps. |
| `CAPTURE_AGENT=off` | No voice guide: the pose tracker drives the steps and on-screen text replaces speech. Use it for A/B comparisons or when this machine is off. |
| `CAPTURE_AGENT_URL` | This gateway's base URL, e.g. `https://aleatoric.tailfc2739.ts.net`. The api calls `GET /health` and then opens `wss://…/v1/realtime?mode=video`. |

What Drape relies on here:

1. **`GET /health` reports `"recording": false`.** Drape's relay checks this before every session
   and refuses to connect otherwise, so a tester's voice and camera frames are never left in
   `data/sessions/`. `omni.sh` starts the gateway with recording off; only `OMNI_RECORDING=1`
   turns it on. A gateway started before this setting existed reports no `recording` field, and
   Drape treats that as recording.
2. **The realtime protocol on `/v1/realtime?mode=video`**, as used by
   `static/duplex/lib/realtime-session.js`:
   - client sends `session.init` (`system_prompt`, `ref_audio_base64`, `use_tts`, `max_slice_nums`,
     `config`), `input.append` (one second of 16 kHz audio plus optional `video_frames`),
     `session.reset` (`system_prompt`), `session.say` (`text`, `mute_after`) and `session.close`;
   - server sends `session.queued` / `session.queue_update` / `session.queue_done`,
     `session.created`, `response.output_audio.delta`, `response.listen`, `response.metrics`
     (`kv_cache_length`), `session.reset.done`, `session.say.done`, `session.closed` and `error`.
3. **The reference voice** at `GET /api/presets/omni/english_call/audio`, fetched by the relay.
4. **An 8k context.** Each second costs about 64 vision tokens plus audio, so the page resets the
   context at every step change and when the KV cache nears full (`CPP_CTX=8192`).

The prompts, fixed lines and the pose tracker live in Drape as a port of `static/fit-capture/`.
Changes to the flow should be made in both places, or the standalone page left as the reference.

### Checking it

```sh
curl -s http://127.0.0.1:8016/health                         # {"status":"healthy",...,"recording":false}
curl -s https://aleatoric.tailfc2739.ts.net/health           # the same, over the tailnet
./omni.sh status                                             # recording: false
```

If `recording` is `true` or missing, restart with `./omni.sh restart` (and without
`OMNI_RECORDING=1`).

## The fit-capture page

`static/fit-capture/` (served at `/fit-capture/`):

- `pose-tracker.js`: MediaPipe pose landmarks (vendored tasks-vision 0.10.14 and
  `pose_landmarker_full.task`), whole-body framing, the arm position check (arms straight, 18–55°
  from the body), turn tracking from the shoulder and hip lines (complete at 340°), sway and
  droop, and an upright canvas that copes with phones held at any of four rotations.
- `fit-capture.js`: the flow (positioning, arms, countdown, recording, done or failed), the
  realtime session, context resets, forced lines, frame gating (frames are sent only when the pose
  changes, at most every 4 s otherwise), mic-level voice detection, and the recording itself
  (MediaRecorder on the upright canvas, up to 40 s).
- `prompts.js`: the system prompt rebuilt at each reset, the fixed opening, step and closing lines,
  and the tracker findings phrased for the model.

Two drivers can move the flow on. With **tracker** (the default, and the only one Drape uses) the
page decides when each step is done and announces it with a fixed line; the model only helps
within a step. With **agent** the model moves the flow on itself by starting a reply with an emoji
marker (🎬 start, ✅ done, 🔄 restart, 🛑 stop). Settings on the page cover the rest.

## Privacy: what is kept on disk

| What | Where | Default |
|---|---|---|
| Session recordings: input audio and frames, model output, timeline | `data/sessions/<session_id>/` (gateway) | **Off** under `omni.sh` (`OMNI_RECORDING=0`). On under Docker Compose and when run by hand, unless `OMNI_RECORDING=0` or `recording.enabled: false` in `config.json`. |
| Per-step input media (wav, jpg) | `/tmp/omni_ws/<session_id>/` (backend) | Deleted once read; the directory is removed when the session ends. Kept only with `OMNI_DEBUG_DUMP=1`. |
| Debug dumps (hidden states, logits, TTS tokens and wavs) | under `/tmp/omni_ws/` or the directory named | Off. Only with `OMNI_DEBUG_DUMP=1`, `TTS_SAVE_HIDDEN_STATES_DIR`, `TTS_LOGITS_DEBUG_DIR` or `TTS_OUTPUT_DIR`. |
| Speaker prompt for the reference voice | `~/.cache/llama-omni/voices/` | Cached by content hash. Built from the preset voice, not from the person's audio. |
| Component logs | `$OMNI_STATE_DIR/cpp/*.log` (default `/tmp/minicpm-omni`) | Session start and end, timings, errors. Prompts and speech text are not logged unless `OMNI_LOG_VERBOSE=1`. |
| Front-end diagnostics | `.run-logs/mobile-record-trace.jsonl` | Written only by pages that post to it; fit-capture and Drape do not. |

The recordings already in `data/sessions/` are kept; turning recording off only stops new ones.

## Configuration

`config.json` sits next to `config.py`. It is gitignored and optional; create it with
`cp config.example.json config.json` and write only the fields you want to change. Missing fields
take their defaults, and CLI arguments (which `omni.sh` passes for ports) take precedence.
Environment variables are applied last:

| Variable | Effect |
|---|---|
| `OMNI_RECORDING=0` / `1` | Overrides `recording.enabled`. `omni.sh` sets `0` unless told otherwise. |

### `config.json` fields

`config.json` sits next to `config.py` in the repo root. It is gitignored; create it with `cp config.example.json config.json`. Every field is optional, so write only the ones you want to change. If the file is missing, every field takes its default. Unknown keys are ignored. The schema, with descriptions, is in `config.py`.

| Field | Default | Meaning |
|---|---|---|
| `model.model_path` | `""` | HuggingFace model directory. Only the PyTorch backend needs it, and `--model-path` overrides it. |
| `model.pt_path` | `null` | Optional extra `.pt` weights. |
| `model.attn_implementation` | `"auto"` | `auto` uses `flash_attention_2` when available and falls back to `sdpa`. Also accepts `flash_attention_2`, `sdpa` and `eager`. |
| `audio.ref_audio_path` | `assets/ref_audio/ref_minicpm_signature.wav` | Default reference voice for TTS voice cloning. |
| `audio.playback_delay_ms` | `200` | How long the frontend buffers after the first speech chunk before it starts playback (0–2000). |
| `audio.chat_vocoder` | `"token2wav"` | Vocoder for non-streaming chat, either `token2wav` or `cosyvoice2`. Streaming and duplex always use token2wav. |
| `service.gateway_port` | `8006` | Gateway port. `--port` overrides it, and `omni.sh` always passes `--port`. |
| `service.worker_base_port` | `22400` | Port of worker 0. Worker *n* uses base + *n*. |
| `service.max_queue_size` | `1000` | Maximum number of queued requests. |
| `service.request_timeout` | `300.0` | Request timeout in seconds. |
| `service.compile` | `false` | Applies `torch.compile` to the core modules (PyTorch backend). |
| `service.data_dir` | `"data"` | Data directory, relative to the repo root. Session recordings go in `<data_dir>/sessions/`. |
| `service.eta_chat_s`, `eta_half_duplex_s`, `eta_audio_duplex_s`, `eta_omni_duplex_s` | 15 / 180 / 120 / 90 | Starting estimates, in seconds, for the queue ETA of each mode. Admins can adjust them at runtime. |
| `service.eta_ema_alpha` | `0.3` | EMA smoothing factor for the measured ETA. |
| `service.eta_ema_min_samples` | `3` | Number of samples needed before the EMA replaces the starting estimate. |
| `duplex.pause_timeout` | `60.0` | How many seconds a paused duplex session holds its worker before releasing it. |
| `recording.enabled` | `true` | The gateway records every session (model input and output plus raw media) to `<data_dir>/sessions/<session_id>/`. `OMNI_RECORDING` overrides it, and `omni.sh` sets it off. |
| `recording.session_retention_days` | `-1` | Deletes recordings older than this many days. `-1` keeps them forever. |
| `recording.max_storage_gb` | `-1` | Total size cap for recordings. When it's exceeded, the oldest are deleted first. `-1` means no cap. |

The gateway runs the retention and size cleanup once a day, starting a minute after it boots. If both limits are `-1`, it does nothing. To stop recording sessions altogether:

```json
{ "recording": { "enabled": false } }
```

### llama.cpp backend logging and temp files

The C++ backend (`llama-omni-server`) is controlled by environment variables rather than `config.json`. `omni.sh` passes them through, for example `OMNI_LOG_VERBOSE=1 ./omni.sh start`.

| Variable | Default | Effect |
|---|---|---|
| `OMNI_LOG_VERBOSE=1` | off | Turns on the per-step trace: encoder, prefill and decode `[prof]` lines, TTS chunks and T2W windows. When it is off, the log shows only session start and end, model init, KV slides, turn-end flush trims, first-audio latency, warnings and errors. |
| `OMNI_DEBUG_DUMP=1` | off | Keeps the debug dumps under `/tmp/omni_ws/<session_id>/`. Each step's input wav and jpg stays there until the session ends. The dumps are `llm_debug/`, `tts_txt/`, and in `tts_wav/` the per-window `wav_N.wav`, token files and merged wavs. When it is off, nothing is dumped, each input file is deleted once the encoder has read it, and the session directory is removed when the session ends. |

## Project structure

```text
├── omni.sh                   # start / stop / status / logs for the cpp stack
├── config.py                 # config schema (Pydantic), config.json loading, env overrides
├── config.example.json
├── gateway.py                # routing, queueing, WebSocket proxy, recording, the pages
├── gateway_modules/          # gateway modules (session recording, ref audio registry, ...)
├── worker.py                 # worker: runtime protocol proxy to the backend
├── runtime/                  # backend protocol client and session layer
├── static/
│   ├── fit-capture/          # the fit-capture page (see above)
│   ├── duplex/lib/           # realtime-session.js and the duplex audio pipeline
│   └── ...                   # upstream demo pages: chat, omni and audio duplex, admin
├── py_backend/, core/, MiniCPMO45/   # PyTorch backend (upstream; not used by omni.sh)
├── docker*/                  # upstream Docker deployment (not maintained here)
├── resources/, assets/       # reference audio, presets
└── data/sessions/            # session recordings, when recording is on
```

The other upstream pages (turn-based chat, omni and audio full-duplex, the admin dashboard) still
work against this stack; see the upstream documentation for them.
