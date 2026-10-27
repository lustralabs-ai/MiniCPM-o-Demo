"""Backend protocol server that forwards to a vllm-omni MiniCPM-o 4.5 server.

Speaks the same protocol as ``py_backend.server`` (``GET /health``,
``WS /backend``, ``POST /sessions/{id}/close``) so the existing worker,
gateway and web UI can run unchanged on top of vllm-omni:

* full_duplex -> vllm-omni realtime WS (``/v1/realtime?duplex=1``)
* turn_based  -> vllm-omni ``/v1/chat/completions`` (SSE streaming)

Usage:
    python -m py_backend.vllm_omni_bridge --port 22500 --upstream http://127.0.0.1:8099
"""

from __future__ import annotations

import argparse
import asyncio
import base64
import hashlib
import io
import json
import logging
import time
import uuid
from contextlib import suppress
from typing import Any, Dict, List, Optional

import httpx
import numpy as np
import soundfile as sf
import websockets
from fastapi import FastAPI, HTTPException, Request, WebSocket, WebSocketDisconnect

logger = logging.getLogger("vllm_omni_bridge")

OUTPUT_SR = 24000
INPUT_SR = 16000

BRIDGE_CONFIG: Dict[str, Any] = {
    "upstream": "http://127.0.0.1:8099",
    "model": "openbmb/MiniCPM-o-4_5",
    "ref_audio_path": None,
}

# vllm's wording when a prompt (plus max_tokens) does not fit max_model_len.
_CONTEXT_ERROR_MARKERS = ("maximum context length", "maximum model length")

# Turn-based pages resend the whole history every turn, and every turn is a new
# session. Remember how many leading exchanges a conversation (keyed by its first
# message) had to drop, so later turns start there instead of re-probing.
_TRIM_HINTS: Dict[str, int] = {}


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _first_dict(*values: Any) -> Dict[str, Any]:
    for value in values:
        if isinstance(value, dict):
            return value
    return {}


def _coalesce(*values: Any, default: Any = None) -> Any:
    for value in values:
        if value is not None and value != "":
            return value
    return default


def _ws_url(base: str, path: str) -> str:
    if base.startswith("https://"):
        return "wss://" + base[len("https://"):].rstrip("/") + path
    if base.startswith("http://"):
        return "ws://" + base[len("http://"):].rstrip("/") + path
    return base.rstrip("/") + path


def _f32_b64_to_wav_data_url(audio_b64: str, sample_rate: int = INPUT_SR) -> str:
    audio = np.frombuffer(base64.b64decode(audio_b64), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _file_to_wav_data_url(path: str) -> str:
    audio, sr = sf.read(path, dtype="float32", always_2d=True)
    audio = audio[:, 0]
    if sr != INPUT_SR:
        audio = _resample(audio, sr, INPUT_SR)
    buf = io.BytesIO()
    sf.write(buf, audio, INPUT_SR, format="WAV", subtype="PCM_16")
    return "data:audio/wav;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def _resample(audio: np.ndarray, src_sr: int, dst_sr: int) -> np.ndarray:
    if src_sr == dst_sr or audio.size == 0:
        return audio.astype(np.float32)
    n_out = int(round(audio.size * dst_sr / src_sr))
    x_old = np.linspace(0.0, 1.0, audio.size, endpoint=False)
    x_new = np.linspace(0.0, 1.0, n_out, endpoint=False)
    return np.interp(x_new, x_old, audio).astype(np.float32)


def _pcm16_b64_to_f32_b64(pcm_b64: str, sample_rate: int) -> str:
    pcm = np.frombuffer(base64.b64decode(pcm_b64), dtype="<i2").astype(np.float32) / 32768.0
    pcm = _resample(pcm, sample_rate, OUTPUT_SR)
    return base64.b64encode(pcm.tobytes()).decode("ascii")


def _wav_b64_to_f32(wav_b64: str) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(base64.b64decode(wav_b64)), dtype="float32", always_2d=True)
    return _resample(audio[:, 0], sr, OUTPUT_SR)


def _image_mime(data: bytes) -> str:
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:3] == b"GIF":
        return "image/gif"
    return "image/jpeg"


def _join_segment(prev: str, piece: str) -> str:
    """vllm-omni duplex decodes each 1 s unit separately, so word-boundary spaces
    between units are unreliable (dropped, or inserted mid-word).  Normalise to
    a single space between units unless the join is clearly inside a word."""
    piece = piece.lstrip()
    if not prev or not piece:
        return piece
    if ord(prev[-1]) >= 0x2E80 or ord(piece[0]) >= 0x2E80:  # CJK: no spaces
        return piece
    if prev[-1].isspace() or prev[-1] in "'’-([\"" or not (piece[0].isalnum() or piece[0] in "(\"'"):
        return piece
    return " " + piece


def _conversation_key(messages: List[Dict[str, Any]]) -> Optional[str]:
    first = next((m for m in messages if m["role"] != "system"), None)
    if first is None:
        return None
    return hashlib.sha1(json.dumps(first, sort_keys=True).encode("utf-8")).hexdigest()


def _drop_oldest_exchange(messages: List[Dict[str, Any]]) -> bool:
    """Drop the oldest user turn and the replies to it.

    System messages (prompt, voice reference) and the latest user turn are kept.
    Returns False when nothing is left to drop.
    """
    first = next((i for i, m in enumerate(messages) if m["role"] != "system"), None)
    last_user = max((i for i, m in enumerate(messages) if m["role"] == "user"), default=None)
    if first is None or last_user is None or first >= last_user:
        return False
    end = first + 1
    while end < last_user and messages[end]["role"] != "user":
        end += 1
    del messages[first:end]
    return True


def _extract_audio_base64(payload: Dict[str, Any]) -> Optional[str]:
    for key in ("audio_base64", "audio_data", "audio"):
        value = payload.get(key)
        if isinstance(value, dict):
            value = value.get("data")
        if isinstance(value, str) and value:
            return value
    return None


def _extract_frames(payload: Dict[str, Any]) -> Optional[List[str]]:
    for key in ("frame_base64_list", "video_frames", "frames"):
        value = payload.get(key)
        if isinstance(value, list) and value:
            return [str(v).split(",", 1)[1] if str(v).startswith("data:") else str(v) for v in value]
    return None


# ---------------------------------------------------------------------------
# session state
# ---------------------------------------------------------------------------

class BridgeState:
    def __init__(self) -> None:
        self.sessions: Dict[str, "BridgeSession"] = {}
        self.active_session_id: Optional[str] = None
        self.lock = asyncio.Lock()

    async def register(self, session: "BridgeSession") -> None:
        async with self.lock:
            if self.active_session_id is not None:
                raise RuntimeError("backend already has an active session")
            self.active_session_id = session.session_id
            self.sessions[session.session_id] = session

    async def forget(self, session_id: str) -> None:
        async with self.lock:
            self.sessions.pop(session_id, None)
            if self.active_session_id == session_id:
                self.active_session_id = None


_state = BridgeState()


class BridgeSession:
    def __init__(self, *, session_id: str, mode: str, ws: WebSocket) -> None:
        self.session_id = session_id
        self.mode = mode
        self.ws = ws
        self.closed = False
        self.initialized = False
        self._send_lock = asyncio.Lock()
        # duplex
        self._up: Any = None
        self._up_reader: Optional[asyncio.Task[None]] = None
        self._response_id: Optional[str] = None
        self._response_text = ""
        self._last_append_t: Optional[float] = None
        self._audio_end_ms = 0.0
        self._chunks_sent = 0
        self._upstream_info: Dict[str, Any] = {}
        # turn-based
        self._http: Optional[httpx.AsyncClient] = None
        self._op_lock = asyncio.Lock()

    # -- wire ---------------------------------------------------------------

    async def send(self, event_type: str, **fields: Any) -> None:
        data = {"type": event_type, **{k: v for k, v in fields.items() if v is not None}}
        data["server_send_ts"] = time.time()
        async with self._send_lock:
            await self.ws.send_json(data)

    async def send_output_delta(self, kind: str, **fields: Any) -> None:
        await self.send("response.output.delta", kind=kind, session_id=self.session_id, **fields)

    # -- lifecycle ----------------------------------------------------------

    async def init(self, params: Dict[str, Any]) -> None:
        if self.mode == "full_duplex":
            await self._init_duplex(params)
        elif self.mode == "turn_based":
            self._http = httpx.AsyncClient(base_url=BRIDGE_CONFIG["upstream"], timeout=httpx.Timeout(600.0))
        else:
            raise RuntimeError(f"unsupported mode: {self.mode}")
        self.initialized = True
        await self.send(
            "session.created",
            session_id=self.session_id,
            mode=self.mode,
            metrics={"backend": "vllm-omni", **self._upstream_info},
            usage={},
        )

    async def push(self, message: Dict[str, Any]) -> None:
        if self.closed:
            raise RuntimeError("session is closed")
        payload = _first_dict(message.get("input"), message.get("payload"))
        if self.mode == "full_duplex":
            await self._push_duplex(payload)
        else:
            async with self._op_lock:
                await self._push_turn_based(payload)

    async def close(self, *, reason: str = "client_closed", emit_event: bool = True) -> None:
        if self.closed:
            return
        self.closed = True
        await self._teardown_upstream()
        if emit_event:
            with suppress(Exception):
                await self.send("session.closed", session_id=self.session_id, reason=reason, usage={})
        with suppress(Exception):
            await self.ws.close(code=1000, reason=reason)
        await _state.forget(self.session_id)

    async def fatal(self, reason: str, *, message: Optional[str] = None) -> None:
        if self.closed:
            return
        logger.error("fatal session termination: session=%s reason=%s message=%s", self.session_id, reason, message)
        self.closed = True
        with suppress(Exception):
            await self.send(
                "session.closed",
                session_id=self.session_id,
                reason=reason,
                diagnostic={"message": message} if message else None,
                usage={},
            )
        with suppress(Exception):
            await self.ws.close(code=1011, reason=reason)
        await self._teardown_upstream()
        await _state.forget(self.session_id)

    async def _teardown_upstream(self) -> None:
        if self._up is not None:
            with suppress(Exception):
                await self._up.send(json.dumps({"type": "session.close"}))
            if self._up_reader is not None and self._up_reader is not asyncio.current_task():
                with suppress(Exception, asyncio.CancelledError):
                    await asyncio.wait_for(asyncio.shield(self._up_reader), timeout=2.0)
                self._up_reader.cancel()
            with suppress(Exception):
                await self._up.close()
            self._up = None
        if self._http is not None:
            with suppress(Exception):
                await self._http.aclose()
            self._http = None

    # -- full duplex --------------------------------------------------------

    async def _init_duplex(self, params: Dict[str, Any]) -> None:
        config = _first_dict(params.get("config"), params.get("duplex"))
        voice = _first_dict(params.get("voice"), params.get("defaults"))
        ref_b64 = _coalesce(
            params.get("tts_ref_audio_base64"), voice.get("tts_ref_audio_base64"),
            params.get("ref_audio_base64"), voice.get("ref_audio_base64"), voice.get("ref_audio"),
        )
        ref_path = _coalesce(params.get("ref_audio_path"), voice.get("ref_audio_path"), BRIDGE_CONFIG["ref_audio_path"])
        ref_audio = None
        if ref_b64:
            ref_audio = _f32_b64_to_wav_data_url(ref_b64)
        elif ref_path:
            ref_audio = _file_to_wav_data_url(ref_path)

        session: Dict[str, Any] = {
            "model": BRIDGE_CONFIG["model"],
            "modalities": ["audio", "text"],
            "input_audio_format": "pcm_f32le",
            "output_audio_format": "pcm16",
            "sample_rate_hz": INPUT_SR,
            "audio": {"input": {"sample_rate_hz": INPUT_SR}, "output": {"sample_rate_hz": OUTPUT_SR}},
            "overlap_policy": "listen_only",
            "extra_body": {"auto_response": True},
        }
        instructions = _coalesce(params.get("system_prompt"), params.get("instructions"))
        if instructions:
            session["instructions"] = instructions
        if ref_audio:
            session["ref_audio"] = ref_audio
        if config.get("temperature") is not None:
            session["temperature"] = float(config["temperature"])

        url = _ws_url(BRIDGE_CONFIG["upstream"], "/v1/realtime?duplex=1")
        self._up = await websockets.connect(url, max_size=64 * 1024 * 1024, ping_interval=None)
        await self._up.send(json.dumps({"type": "session.update", "session": session}))

        # Wait for the server to accept the configuration.
        deadline = time.monotonic() + 60.0
        while True:
            raw = await asyncio.wait_for(self._up.recv(), timeout=max(0.1, deadline - time.monotonic()))
            ev = json.loads(raw)
            etype = ev.get("type")
            if etype == "error":
                raise RuntimeError(f"vllm-omni rejected session: {(ev.get('error') or {}).get('message')}")
            if etype == "session.updated":
                up = ev.get("session") or {}
                self._upstream_info = {"upstream_session_id": up.get("id")}
                break
        logger.info("duplex session %s -> upstream %s (ref_audio=%s)",
                    self.session_id, self._upstream_info.get("upstream_session_id"), bool(ref_audio))
        self._up_reader = asyncio.create_task(self._read_upstream())

    async def _push_duplex(self, payload: Dict[str, Any]) -> None:
        audio_b64 = _extract_audio_base64(payload)
        if not audio_b64:
            raise RuntimeError("full_duplex input requires audio")
        n_samples = len(base64.b64decode(audio_b64)) // 4
        duration_ms = n_samples * 1000.0 / INPUT_SR
        self._audio_end_ms += duration_ms
        event: Dict[str, Any] = {
            "type": "input_audio_buffer.append",
            "audio": audio_b64,
            "format": "pcm_f32le",
            "duration_ms": duration_ms,
            "audio_end_ms": self._audio_end_ms,
        }
        frames = _extract_frames(payload)
        if frames:
            event["video_frames"] = frames
        self._last_append_t = time.perf_counter()
        self._chunks_sent += 1
        if self._up is None:
            raise RuntimeError("upstream session is not connected")
        await self._up.send(json.dumps(event))

    def _metrics(self) -> Dict[str, Any]:
        metrics: Dict[str, Any] = {"chunks_sent": self._chunks_sent}
        if self._last_append_t is not None:
            metrics["wall_clock_ms"] = round((time.perf_counter() - self._last_append_t) * 1000, 1)
        return metrics

    async def _end_response(self) -> None:
        await self.send_output_delta("listen", response_id=self._response_id, end_of_turn=True, metrics=self._metrics())
        self._response_id = None
        self._response_text = ""

    async def _read_upstream(self) -> None:
        try:
            async for raw in self._up:
                ev = json.loads(raw)
                etype = ev.get("type")
                if etype == "response.listen":
                    if self._response_id is not None:
                        await self._end_response()
                    else:
                        await self.send_output_delta("listen", metrics=self._metrics())
                elif etype == "response.created":
                    self._response_id = (ev.get("response") or {}).get("id") or f"resp_{uuid.uuid4().hex[:12]}"
                elif etype in ("response.output_audio_transcript.delta", "response.output_text.delta"):
                    text = _join_segment(self._response_text, ev.get("delta") or "")
                    if text:
                        self._response_text += text
                        self._response_id = self._response_id or ev.get("response_id")
                        await self.send_output_delta("text", response_id=self._response_id,
                                                     text=text, metrics=self._metrics())
                elif etype == "response.output_audio.delta":
                    if ev.get("delta"):
                        self._response_id = self._response_id or ev.get("response_id")
                        audio = _pcm16_b64_to_f32_b64(ev["delta"], int(ev.get("sample_rate_hz") or OUTPUT_SR))
                        await self.send_output_delta("audio", response_id=self._response_id,
                                                     audio=audio, metrics=self._metrics())
                elif etype == "response.done":
                    if self._response_id is not None:
                        await self._end_response()
                elif etype == "error":
                    err = ev.get("error") or {}
                    logger.warning("upstream error: %s %s", err.get("code"), err.get("message"))
                elif etype == "session.closed":
                    if not self.closed:
                        await self.fatal("backend_error", message=f"upstream session closed: {ev.get('reason')}")
                    return
        except websockets.ConnectionClosed as exc:
            if not self.closed:
                await self.fatal("backend_error", message=f"upstream connection closed: {exc}")
        except Exception as exc:
            if not self.closed:
                logger.exception("upstream reader failed")
                await self.fatal("backend_error", message=str(exc))
        else:
            if not self.closed:
                await self.fatal("backend_error", message="upstream connection ended")

    # -- turn based ---------------------------------------------------------

    @staticmethod
    def _convert_messages(raw_messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for msg in raw_messages:
            role = msg.get("role", "user")
            content = msg.get("content")
            if not isinstance(content, list):
                out.append({"role": role, "content": content or ""})
                continue
            if role == "assistant":
                text = "".join(it.get("text", "") for it in content if isinstance(it, dict) and it.get("type") == "text")
                out.append({"role": role, "content": text})
                continue
            parts: List[Dict[str, Any]] = []
            for it in content:
                if not isinstance(it, dict):
                    continue
                itype = it.get("type")
                if itype == "text" and it.get("text"):
                    parts.append({"type": "text", "text": it["text"]})
                elif itype == "audio" and it.get("data"):
                    parts.append({"type": "audio_url", "audio_url": {"url": _f32_b64_to_wav_data_url(it["data"])}})
                elif itype == "image" and it.get("data"):
                    mime = _image_mime(base64.b64decode(it["data"][:64]))
                    parts.append({"type": "image_url", "image_url": {"url": f"data:{mime};base64,{it['data']}"}})
                elif itype == "video" and it.get("data"):
                    parts.append({"type": "video_url", "video_url": {"url": f"data:video/mp4;base64,{it['data']}"}})
            if parts:
                out.append({"role": role, "content": parts})
        return out

    async def _push_turn_based(self, payload: Dict[str, Any]) -> None:
        response_id = str(payload.get("response_id") or f"resp_{uuid.uuid4().hex[:12]}")
        input_id = payload.get("input_id")
        generation = _first_dict(payload.get("generation"))
        tts = _first_dict(payload.get("tts"))
        generate_audio = bool(tts.get("enabled", False))
        streaming = bool(payload.get("streaming", True))

        messages = self._convert_messages(payload.get("messages") or [])
        has_sys_audio = any(
            m["role"] == "system" and isinstance(m["content"], list)
            and any(p.get("type") == "audio_url" for p in m["content"])
            for m in messages
        )
        if generate_audio and tts.get("ref_audio_data") and not has_sys_audio:
            # Voice cloning in MiniCPM-o goes through a reference clip in the system prompt.
            ref = {"type": "audio_url", "audio_url": {"url": _f32_b64_to_wav_data_url(tts["ref_audio_data"])}}
            if messages and messages[0]["role"] == "system":
                sys_content = messages[0]["content"]
                if isinstance(sys_content, str):
                    sys_content = [{"type": "text", "text": sys_content}]
                messages[0]["content"] = [ref, *sys_content]
            else:
                messages.insert(0, {"role": "system", "content": [ref]})

        body: Dict[str, Any] = {
            "model": BRIDGE_CONFIG["model"],
            "messages": messages,
            "stream": True,
            "modalities": ["text", "audio"] if generate_audio else ["text"],
            "max_tokens": int(generation.get("max_new_tokens", 256)),
            "chat_template_kwargs": {
                "use_tts_template": bool(payload.get("use_tts_template", False) or generate_audio),
                "enable_thinking": bool(payload.get("enable_thinking", False)),
            },
        }
        if generation.get("temperature") is not None:
            body["temperature"] = float(generation["temperature"])

        assert self._http is not None
        t0 = time.perf_counter()
        first_token_ms: Optional[float] = None
        full_text = ""
        audio_parts: List[np.ndarray] = []

        conv_key = _conversation_key(messages)
        dropped = 0
        for _ in range(_TRIM_HINTS.get(conv_key, 0) if conv_key else 0):
            if not _drop_oldest_exchange(messages):
                break
            dropped += 1
        while True:
            request = self._http.build_request("POST", "/v1/chat/completions", json=body)
            resp = await self._http.send(request, stream=True)
            if resp.status_code == 200:
                break
            detail = (await resp.aread()).decode("utf-8", "replace")
            await resp.aclose()
            # vllm-omni reports this as a 500, plain vllm as a 400.
            if (resp.status_code in (400, 500) and any(m in detail for m in _CONTEXT_ERROR_MARKERS)
                    and _drop_oldest_exchange(messages)):
                dropped += 1
                continue
            raise RuntimeError(f"vllm-omni chat/completions HTTP {resp.status_code}: {detail[:500]}")
        if dropped:
            logger.info("session %s: dropped %d oldest exchange(s) to fit the context", self.session_id, dropped)
            if conv_key:
                if len(_TRIM_HINTS) > 256:
                    _TRIM_HINTS.clear()
                _TRIM_HINTS[conv_key] = dropped

        try:
            async for line in resp.aiter_lines():
                if not line.startswith("data: "):
                    continue
                data = line[6:]
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                if chunk.get("error"):
                    raise RuntimeError(f"vllm-omni error: {chunk['error']}")
                choices = chunk.get("choices") or []
                content = ((choices[0].get("delta") or {}).get("content")) if choices else None
                if not content:
                    continue
                if first_token_ms is None:
                    first_token_ms = round((time.perf_counter() - t0) * 1000, 1)
                if chunk.get("modality") == "audio":
                    wave = _wav_b64_to_f32(content)
                    if streaming:
                        await self.send_output_delta(
                            "audio", response_id=response_id, input_id=input_id,
                            audio=base64.b64encode(wave.tobytes()).decode("ascii"),
                        )
                    else:
                        audio_parts.append(wave)
                else:
                    full_text += content
                    if streaming:
                        await self.send_output_delta("text", response_id=response_id, input_id=input_id, text=content)
        finally:
            await resp.aclose()

        audio_b64 = None
        if not streaming and audio_parts:
            audio_b64 = base64.b64encode(np.concatenate(audio_parts).tobytes()).decode("ascii")
        await self.send(
            "response.done",
            session_id=self.session_id,
            response_id=response_id,
            input_id=input_id,
            text=full_text,
            audio=audio_b64,
            reason="turn_end",
            metrics={
                "backend": "vllm-omni",
                "generate_ms": round((time.perf_counter() - t0) * 1000, 1),
                "first_chunk_ms": first_token_ms,
                "dropped_exchanges": dropped,
            },
        )


# ---------------------------------------------------------------------------
# app
# ---------------------------------------------------------------------------

app = FastAPI(title="MiniCPM-o vllm-omni bridge")


@app.get("/health")
async def health() -> Dict[str, Any]:
    status = "loading"
    with suppress(Exception):
        async with httpx.AsyncClient(timeout=3.0) as client:
            resp = await client.get(BRIDGE_CONFIG["upstream"].rstrip("/") + "/health")
            if resp.status_code == 200:
                status = "ready"
    return {
        "status": status,
        "backend": "vllm-omni",
        "worker_status": None,
        "active_session_id": _state.active_session_id,
    }


@app.websocket("/backend")
async def backend_ws(ws: WebSocket) -> None:
    await ws.accept()
    session: Optional[BridgeSession] = None
    try:
        first = json.loads(await ws.receive_text())
        if first.get("type") != "session.init":
            raise RuntimeError("first message must be session.init")
        params = _first_dict(first.get("payload"))
        mode = str(params.get("mode") or "full_duplex")
        session = BridgeSession(session_id=f"sess_{uuid.uuid4().hex[:12]}", mode=mode, ws=ws)
        await _state.register(session)
        await session.init(params)

        while not session.closed:
            message = json.loads(await ws.receive_text())
            if message.get("type") == "input.append":
                await session.push(message)
                continue
            raise RuntimeError(f"unsupported message type: {message.get('type')}")
    except WebSocketDisconnect:
        if session is not None:
            await session.close(reason="client_disconnected", emit_event=False)
    except Exception as exc:
        if session is not None and session.closed:
            logger.info("session %s closed during in-flight work: %s", session.session_id, exc)
            return
        logger.exception("bridge session error")
        if session is not None:
            await session.fatal("backend_error", message=str(exc))
        else:
            with suppress(Exception):
                await ws.close(code=1011, reason="backend_error")


@app.post("/sessions/{session_id}/close")
async def close_session(session_id: str, request: Request) -> Dict[str, Any]:
    session = _state.sessions.get(session_id)
    if session is None:
        raise HTTPException(status_code=404, detail="session not found")
    reason = "client_closed"
    with suppress(Exception):
        body = await request.json()
        if isinstance(body, dict) and body.get("reason"):
            reason = str(body["reason"])
    await session.close(reason=reason)
    return {"ok": True, "session_id": session_id, "closed": True}


def main() -> None:
    import uvicorn

    parser = argparse.ArgumentParser(description="MiniCPM-o backend protocol bridge to vllm-omni")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=22500)
    parser.add_argument("--upstream", default=BRIDGE_CONFIG["upstream"], help="vllm-omni base URL")
    parser.add_argument("--model", default=BRIDGE_CONFIG["model"], help="served model name")
    parser.add_argument("--ref-audio-path", default=None, help="default duplex voice reference WAV")
    args = parser.parse_args()

    ref_audio_path = args.ref_audio_path
    if ref_audio_path is None:
        with suppress(Exception):
            from config import get_config
            ref_audio_path = get_config().ref_audio_path
    BRIDGE_CONFIG.update({"upstream": args.upstream, "model": args.model, "ref_audio_path": ref_audio_path})

    logging.basicConfig(level=logging.INFO)
    uvicorn.run(app, host=args.host, port=args.port, ws_max_size=128 * 1024 * 1024)


if __name__ == "__main__":
    main()
