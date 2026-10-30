# usage: e2e_voice.py question.wav out.wav [--video] [--noref] [--port 8006] [--frame-every N] [--repeat K] [--gap S]
import asyncio, base64, io, json, sys, time
import numpy as np, soundfile as sf, websockets
from PIL import Image
args = sys.argv[1:]
video = "--video" in args; noref = "--noref" in args
opt = lambda k, d: type(d)(args[args.index(k)+1]) if k in args else d
port = opt("--port", "8006"); every = opt("--frame-every", 1); repeat = opt("--repeat", 1); gap = opt("--gap", 20)
def load16(p):
    a, sr = sf.read(p, dtype="float32"); a = a if a.ndim == 1 else a[:, 0]
    return np.interp(np.linspace(0, len(a), int(len(a)*16000/sr), endpoint=False), np.arange(len(a)), a).astype(np.float32)
async def main(wav, out):
    q = load16(wav); z = lambda s: np.zeros(int(16000*s), np.float32)
    a = np.concatenate([z(1)] + [q, z(gap)] * repeat)
    ref = load16(opt("--ref", "/home/carl/lustra/MiniCPM-o-Demo/assets/ref_audio/ref_minicpm_signature.wav"))
    img = Image.open("/home/carl/lustra/MiniCPM-o-Demo/assets/images/demo_preview.png").convert("RGB").resize((640, 480))
    buf = io.BytesIO(); img.save(buf, "JPEG", quality=70); frame = base64.b64encode(buf.getvalue()).decode()
    payload = {"system_prompt": "Streaming Omni Conversation." if video else "You are a helpful assistant.", "max_slice_nums": 1}
    if not noref: payload["ref_audio_base64"] = base64.b64encode(ref.tobytes()).decode()
    mode = "video" if video else "audio"
    pcm = []; arr = []
    async with websockets.connect(f"ws://127.0.0.1:{port}/v1/realtime?mode={mode}", max_size=None) as ws:
        t0 = time.monotonic(); text = ""; ts = None
        async def reader():
            nonlocal text
            async for raw in ws:
                m = json.loads(raw); t = m.get("type")
                if t == "session.queue_done":
                    await ws.send(json.dumps({"type": "session.init", "payload": payload}))
                k = m.get("kind")
                if k == "audio":
                    p = np.frombuffer(base64.b64decode(m["audio"]), np.float32); pcm.append(p)
                    arr.append((time.monotonic() - t0, len(p) / 24000))
                if k == "text": text += m["text"]
                if t == "session.created": created.set()
                if t in ("session.created", "session.closed", "error") or k == "text":
                    print(f"{time.monotonic()-t0:6.2f} {t} {k or ''} {m.get('text','')!r} {m.get('reason','')}")
        created = asyncio.Event(); r = asyncio.create_task(reader())
        await asyncio.wait_for(created.wait(), 120)
        ts = time.monotonic() - t0
        for i in range(0, len(a), 16000):
            inp = {"audio": base64.b64encode(a[i:i+16000].tobytes()).decode()}
            if video and (i // 16000) % every == 0: inp["video_frames"] = [frame]
            await ws.send(json.dumps({"type": "input.append", "input": inp}))
            await asyncio.sleep(1.0)
        await ws.send(json.dumps({"type": "session.close"}))
        await asyncio.sleep(2); r.cancel()
    # playback simulation: underruns > 0.15 s inside a burst (bursts separated by > 3 s)
    gaps = []; play_end = None
    for t, d in arr:
        if play_end is not None and 0.15 < t - play_end < 3.0: gaps.append((round(play_end - ts, 2), round(t - play_end, 2)))
        play_end = max(play_end or 0, t) + d
    # audio-player.js: the first chunk of a turn waits D, later chunks queue back to back; no rebuffer after a gap
    for D in (0.2, 0.3, 0.5, 1.0):
        nt = None; last = None; gl = []
        for t, d in arr:
            if last is None or t - last > 3.0: nt = t + D
            elif nt < t:
                gl.append(round(t - nt, 2)); nt = t
            nt += d; last = t
        print(f"UNDERRUN@{D}", len(gl), round(sum(gl), 2), gl[:12])
    audio = np.concatenate(pcm) if pcm else np.zeros(1, np.float32)
    sf.write(out, audio, 24000)
    print("STREAM_START", round(ts, 2), "Q_ENDS", [round(1 + k*(len(q)/16000 + gap) + len(q)/16000, 2) for k in range(repeat)])
    print("GAPS", len(gaps), gaps)
    print("ARR", [(round(t - ts, 2), round(d, 2)) for t, d in arr])
    print("AUDIO_S", round(len(audio)/24000, 2)); print("TEXT", text)
asyncio.run(main(args[0], args[1]))
