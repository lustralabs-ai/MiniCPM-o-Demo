# Marker probe for static/fit-capture: does the model emit the emoji markers, and are they kept out of TTS?
#
# usage: e2e_fit_markers.py SCENARIO [--port 8016] [--secs 30] [--wav user.wav --at 5] [--reset-at S --reset-step STEP]
#                           [--driver agent|tracker] [--out out.wav] [--slices 1] [--listen 0.0] [--gate]
#   SCENARIO: positioning | arms | recording   (prompt step; frames are chosen to match)
#     positioning/arms: front-view frames;  recording: front→right→back→left→front, one view per --view-secs
#   --reset-at S: send session.reset at S seconds with the prompt for --reset-step (needs the rebuilt cpp server)
#   --listen X: init config.listen_prob_scale (the page's 'Listen bias'; omit for the server default)
#   --gate: send a frame only when the view changes or every 4 s, like the page's frame gating
#   --no-frames-after-reset: audio only after --reset-at, as the page does once the camera stops (done/failed)
#
# The prompt is built by static/fit-capture/prompts.js (via node) so this tests what the page sends.
import asyncio, base64, io, json, subprocess, sys, time
import numpy as np, soundfile as sf, websockets
from PIL import Image

args = sys.argv[1:]
opt = lambda k, d: type(d)(args[args.index(k) + 1]) if k in args else d
scenario = args[0]
port = opt("--port", "8016"); secs = opt("--secs", 30); driver = opt("--driver", "agent")
wav = opt("--wav", ""); wav_at = opt("--at", 5.0); out = opt("--out", "")
reset_at = opt("--reset-at", -1.0); reset_step = opt("--reset-step", "recording")
view_secs = opt("--view-secs", 5); slices = opt("--slices", 1)
listen = opt("--listen", -99.0); gate = "--gate" in args; blind_after_reset = "--no-frames-after-reset" in args
POSE = "/home/carl/lustra/drape/apps/tester-web/public/pose-guides"
MARKERS = {"🎬": "start", "🎥": "start", "✅": "done", "✔": "done", "🔄": "restart", "🔁": "restart", "🛑": "stop", "⛔": "stop"}


def prompt_for(step, resets=0, turn=None):
    st = {"step": step, "driver": driver, "tracker": ["in_view"] + (["arms_ok"] if step != "positioning" else []),
          "turnDeg": turn, "attempt": 1, "elapsedSec": 10, "timeoutSec": 120, "resets": resets, "lastSaid": ""}
    js = ("import { buildPrompt } from '/home/carl/lustra/MiniCPM_app/static/fit-capture/prompts.js';"
          f"process.stdout.write(buildPrompt({json.dumps(st)}));")
    return subprocess.run(["node", "--input-type=module", "-e", js], capture_output=True, text=True, check=True).stdout


def frame(name):
    # Letterbox the portrait guide image into a 360x640 portrait frame, as the page sends (max 640 tall).
    img = Image.open(f"{POSE}/{name}.jpg").convert("RGB")
    img.thumbnail((360, 640))
    canvas = Image.new("RGB", (360, 640), (128, 128, 128)); canvas.paste(img, ((360 - img.width) // 2, (640 - img.height) // 2))
    buf = io.BytesIO(); canvas.save(buf, "JPEG", quality=70)
    return base64.b64encode(buf.getvalue()).decode()


def load16(p):
    a, sr = sf.read(p, dtype="float32"); a = a if a.ndim == 1 else a[:, 0]
    return np.interp(np.linspace(0, len(a), int(len(a) * 16000 / sr), endpoint=False), np.arange(len(a)), a).astype(np.float32)


async def main():
    frames = {n: frame(n) for n in ("front", "right", "back", "left")}
    seq = ["front"] if scenario != "recording" else ["front", "right", "back", "left", "front"]
    audio = np.zeros(16000 * secs, np.float32)
    if wav:
        u = load16(wav); i = int(wav_at * 16000); audio[i:i + len(u)] = u[: len(audio) - i]
    ref = load16("/home/carl/lustra/MiniCPM_app/assets/ref_audio/ref_en_dlc_1.wav")
    prompt = prompt_for(scenario)
    print("PROMPT", len(prompt), "chars\n" + prompt + "\n")
    payload = {"system_prompt": prompt, "max_slice_nums": slices, "ref_audio_base64": base64.b64encode(ref.tobytes()).decode()}
    if listen > -99: payload["config"] = {"listen_prob_scale": listen}
    pcm = []; utter = []; cur = ""; found = []; kv = [0]; nframes = 0; last_view = None; last_frame_at = -99
    async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime?mode=video", max_size=None) as ws:
        t0 = time.monotonic(); created = asyncio.Event()

        async def reader():
            nonlocal cur
            async for raw in ws:
                m = json.loads(raw); t = m.get("type"); k = m.get("kind"); el = time.monotonic() - t0
                kv[0] = (m.get("metrics") or {}).get("kv_cache_length") or m.get("kv_cache_length") or kv[0]
                if t == "session.queue_done":
                    await ws.send(json.dumps({"type": "session.init", "payload": payload}))
                if t == "session.created": created.set()
                if k == "audio": pcm.append(np.frombuffer(base64.b64decode(m["audio"]), np.float32))
                if k == "text":
                    cur += m["text"]
                    for ch in m["text"]:
                        if ch in MARKERS: found.append((round(el, 1), MARKERS[ch])); print(f"{el:6.1f} ★ MARKER {ch} {MARKERS[ch]}")
                if k == "listen" and cur:
                    print(f"{el:6.1f} AGENT: {cur!r}"); utter.append(cur); cur = ""
                if t in ("session.reset.done", "session.closed", "error"):
                    print(f"{el:6.1f} {t} {json.dumps({x: m.get(x) for x in ('ok', 'reason', 'error', 'metrics') if x in m})}")

        r = asyncio.create_task(reader())
        await asyncio.wait_for(created.wait(), 120)
        ts = time.monotonic()
        for n in range(secs):
            if 0 <= reset_at <= n < reset_at + 1:
                await ws.send(json.dumps({"type": "session.reset", "payload": {"system_prompt": prompt_for(reset_step, resets=1, turn=0)}}))
                print(f"{time.monotonic() - t0:6.1f} → session.reset ({reset_step})")
            view = seq[min(len(seq) - 1, n // view_secs)]
            inp = {"audio": base64.b64encode(audio[n * 16000:(n + 1) * 16000].tobytes()).decode()}
            if blind_after_reset and 0 <= reset_at <= n:
                pass
            elif not gate or view != last_view or n - last_frame_at >= 4:
                inp["video_frames"] = [frames[view]]; last_view = view; last_frame_at = n; nframes += 1
            if slices > 1: inp["max_slice_nums"] = slices
            await ws.send(json.dumps({"type": "input.append", "input": inp}))
            await asyncio.sleep(max(0, ts + n + 1 - time.monotonic()))
        await asyncio.sleep(3)
        if cur: print("AGENT (open):", repr(cur)); utter.append(cur)
        await ws.send(json.dumps({"type": "session.close"})); await asyncio.sleep(1); r.cancel()
    print("MARKERS", found)
    print("UTTERANCES", json.dumps(utter, ensure_ascii=False))
    print(f"SUMMARY listen={listen if listen > -99 else 'default'} gate={gate} secs={secs} replies={len(utter)} "
          f"per_min={len(utter) * 60 / secs:.1f} frames={nframes} kv_end={kv[0]} kv_per_min={kv[0] * 60 / secs:.0f} "
          f"audio_s={sum(len(x) for x in pcm) / 24000:.1f}")
    if out and pcm: sf.write(out, np.concatenate(pcm), 24000); print("wrote", out)

asyncio.run(main())
