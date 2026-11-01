"""LiveKit TTS for Voxist TTS: streaming synthesis over the text-streaming WebSocket.

Protocol: voxist-tts ``GET /v1/tts/{voice_id}/stream/text`` (voxist-tts
``voxist-server/src/routes/ws.rs`` and ``docs/tts-websocket-contract.md``), the
socket Voxlingo's translate gateway streams speech from
(voxist-translate ``src/serve/tts/voxist.py``, ``TTSSession``):

- client: one config message ``{"format": "pcm_s16le", "language", "version"}``,
  then ``{"text": ...}`` chunks as they arrive, then ``{"end": true}``;
- server: binary frames of raw PCM16 mono at 24 kHz (80 ms each), JSON
  ``{"error", "code", "severity", "request_id"}`` (``severity`` ``fatal`` ends the
  session, ``advisory`` does not), and ``{"done": true, ...}`` once the audio for
  every accepted chunk has been sent.

Kyutai voices start speaking about 0.3 s after the first text arrives, where the
REST ``POST /v1/tts/{voice_id}`` waits for the whole clip.

Behaviour:

- One socket per segment. :meth:`VoxistTTS.stream` dials as soon as the stream
  starts, so the handshake overlaps the LLM's first token; text is forwarded as it
  is pushed (the server joins word pieces), and ``flush()`` / ``end_input()``
  send ``end`` and wait for ``done``.
- Barge-in: closing the stream closes the socket, which frees the server's slot.
- Deadlines: ``connect_timeout`` per dial (local retries follow
  ``conn_options.max_retry``: the first immediately, later ones after
  ``retry_interval``), ``first_audio_timeout`` from the first text sent to the
  first audio frame, and ``idle_timeout`` between server messages once ``end``
  is sent. Every error raised to LiveKit is non-retryable, so LiveKit never
  replays text the server already spoke.
- Credentials: ``headers`` (for example a Cloudflare Access service token) go on
  the handshake only, are never logged and never appear in an error message. A
  redirect is never followed, since it would carry them to another URL.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
import weakref
from collections.abc import AsyncIterator, Mapping
from dataclasses import dataclass, field
from typing import Any
from urllib.parse import quote, urlsplit

import aiohttp
from livekit.agents import (
    APIConnectionError,
    APIError,
    APIStatusError,
    APITimeoutError,
    tts,
)
from livekit.agents.types import DEFAULT_API_CONNECT_OPTIONS, APIConnectOptions
from livekit.agents.utils import shortuuid

logger = logging.getLogger("livekit.plugins.voxist")

#: voxist-tts's fixed WebSocket output rate (``ws.rs``: ``done``'s ``duration_s``
#: divides by this literal).
SAMPLE_RATE = 24000
#: The contract generation this client follows (``tts-websocket-contract.md``,
#: "Protocol versioning"): strict config handling, error envelope, cancel.
PROTOCOL_VERSION = "2026-08-01"
#: Server frames are 80 ms; emitting at the same size adds no buffering delay.
FRAME_SIZE_MS = 80

CONNECT_TIMEOUT_DEFAULT = 5.0
FIRST_AUDIO_TIMEOUT_DEFAULT = 6.0
IDLE_TIMEOUT_DEFAULT = 6.0
CLOSE_TIMEOUT = 1.0
_WATCHDOG_TICK_S = 0.05

#: Voice ids that collide with other server routes (``/v1/tts/jobs`` ...).
RESERVED_VOICE_IDS = frozenset({"jobs", "languages", "voices"})

#: HTTP-equivalent status per fatal error code (``ws.rs`` ``WsErrorCode``).
_ERROR_STATUS = {
    "capacity_exceeded": 503,
    "language_unservable": 503,
    "session_timeout": 504,
    "backend_error": 502,
    "internal_error": 502,
    "voice_not_found": 404,
    "voice_no_embedding": 422,
    "voice_unservable": 422,
    "unsupported_language": 422,
    "config_rejected": 400,
    "unsupported_version": 400,
    "invalid_json": 400,
    "invalid_message_type": 400,
    "text_empty": 400,
    "text_too_long": 413,
}


class _RedirectRefused(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


async def _refuse_redirect(
    _session: aiohttp.ClientSession, _ctx: Any, params: aiohttp.TraceRequestRedirectParams
) -> None:
    raise _RedirectRefused(params.response.status)


def _ws_base(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    scheme = {"https": "wss", "http": "ws", "wss": "wss", "ws": "ws"}.get(parts.scheme)
    if scheme is None or not parts.netloc:
        raise ValueError("base_url must be an http(s) or ws(s) URL with a host")
    if parts.query or parts.fragment:
        raise ValueError("base_url must not carry a query or a fragment")
    return f"{scheme}://{parts.netloc}{parts.path}"


@dataclass
class _TTSOptions:
    ws_base: str
    voice_id: str
    language: str
    headers: dict[str, str] = field(repr=False)
    connect_timeout: float
    first_audio_timeout: float
    idle_timeout: float
    protocol_version: str

    @property
    def url(self) -> str:
        return f"{self.ws_base}/v1/tts/{quote(self.voice_id, safe='')}/stream/text"

    def config_message(self) -> str:
        return json.dumps(
            {"format": "pcm_s16le", "language": self.language, "version": self.protocol_version}
        )


class VoxistTTS(tts.TTS):
    """Streaming Voxist TTS over ``/v1/tts/{voice_id}/stream/text``.

    Args:
        base_url: Service root, e.g. ``https://tts.voxist-api.com`` (``ws(s)://``
            accepted too).
        voice_id: A Voxist voice id (``ada``, ``103488-es`` ...).
        language: The language the voice speaks, as the server takes it (``fr``).
        headers: Extra handshake headers, e.g. ``CF-Access-Client-Id`` /
            ``CF-Access-Client-Secret``. Never logged.
        connect_timeout: Seconds for one handshake, config included.
        first_audio_timeout: Seconds from the first text sent to the first audio.
        idle_timeout: Seconds between server messages once ``end`` is sent.
        protocol_version: Contract generation stated on the config message.
    """

    def __init__(
        self,
        *,
        base_url: str,
        voice_id: str,
        language: str,
        headers: Mapping[str, str] | None = None,
        connect_timeout: float = CONNECT_TIMEOUT_DEFAULT,
        first_audio_timeout: float = FIRST_AUDIO_TIMEOUT_DEFAULT,
        idle_timeout: float = IDLE_TIMEOUT_DEFAULT,
        protocol_version: str = PROTOCOL_VERSION,
    ) -> None:
        super().__init__(
            capabilities=tts.TTSCapabilities(streaming=True),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
        )
        if not voice_id or voice_id.strip() != voice_id:
            raise ValueError("voice_id must be a non-empty id without surrounding spaces")
        if voice_id.lower() in RESERVED_VOICE_IDS:
            raise ValueError(f"voice_id {voice_id!r} collides with a server route")
        if not language or not language.strip():
            raise ValueError("language is required (the server refuses a session without one)")
        for name, value in (
            ("connect_timeout", connect_timeout),
            ("first_audio_timeout", first_audio_timeout),
            ("idle_timeout", idle_timeout),
        ):
            if not value > 0:
                raise ValueError(f"{name} must be positive")
        self._opts = _TTSOptions(
            ws_base=_ws_base(base_url),
            voice_id=voice_id,
            language=language,
            headers=dict(headers or {}),
            connect_timeout=connect_timeout,
            first_audio_timeout=first_audio_timeout,
            idle_timeout=idle_timeout,
            protocol_version=protocol_version,
        )
        self._session: aiohttp.ClientSession | None = None
        self._closed = False
        self._streams: weakref.WeakSet[VoxistTTSStream] = weakref.WeakSet()

    @property
    def model(self) -> str:
        return "voxist-tts"

    @property
    def provider(self) -> str:
        return "voxist"

    @property
    def voice_id(self) -> str:
        return self._opts.voice_id

    @property
    def language(self) -> str:
        return self._opts.language

    def synthesize(
        self, text: str, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> VoxistTTSChunkedStream:
        return VoxistTTSChunkedStream(tts=self, input_text=text, conn_options=conn_options)

    def stream(
        self, *, conn_options: APIConnectOptions = DEFAULT_API_CONNECT_OPTIONS
    ) -> VoxistTTSStream:
        stream = VoxistTTSStream(tts=self, conn_options=conn_options)
        self._streams.add(stream)
        return stream

    async def aclose(self) -> None:
        """Close every open stream and the HTTP session; later streams fail."""
        self._closed = True
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        if self._session is not None:
            await self._session.close()
            self._session = None

    def _get_session(self) -> aiohttp.ClientSession:
        # A private session, so the redirect refusal is guaranteed: aiohttp's
        # ws_connect follows redirects and keeps custom headers across origins.
        if self._closed:
            raise APIConnectionError("VoxistTTS is closed", retryable=False)
        if self._session is None or self._session.closed:
            trace = aiohttp.TraceConfig()
            trace.on_request_redirect.append(_refuse_redirect)
            self._session = aiohttp.ClientSession(trace_configs=[trace])
        return self._session

    async def _connect_once(self, *, final: bool) -> aiohttp.ClientWebSocketResponse:
        opts = self._opts
        retryable = not final
        try:
            ws = await asyncio.wait_for(
                self._get_session().ws_connect(
                    opts.url,
                    headers=opts.headers,
                    max_msg_size=0,
                    timeout=aiohttp.ClientWSTimeout(ws_receive=None, ws_close=CLOSE_TIMEOUT),
                ),
                opts.connect_timeout,
            )
        except _RedirectRefused as e:
            raise APIStatusError(
                f"Voxist TTS answered the handshake with a redirect (HTTP {e.status}); "
                "not followed: it would carry the credentials to another URL",
                status_code=e.status,
                retryable=False,
            ) from None
        except aiohttp.WSServerHandshakeError as e:
            status = e.status
            again = retryable and (status in (408, 429) or status >= 500)
            raise APIStatusError(
                f"Voxist TTS refused the WebSocket handshake (HTTP {status})",
                status_code=status,
                retryable=again,
            ) from None
        except asyncio.TimeoutError:
            raise APITimeoutError(
                f"Voxist TTS handshake timed out after {opts.connect_timeout}s",
                retryable=retryable,
            ) from None
        except aiohttp.ClientError as e:
            raise APIConnectionError(
                f"Voxist TTS connection failed ({type(e).__name__})", retryable=retryable
            ) from None
        try:
            await ws.send_str(opts.config_message())
        except (aiohttp.ClientError, ConnectionError, RuntimeError) as e:
            await _close(ws)
            raise APIConnectionError(
                f"Voxist TTS socket failed before the config was sent ({type(e).__name__})",
                retryable=retryable,
            ) from None
        return ws

    async def _connect(self, conn_options: APIConnectOptions) -> aiohttp.ClientWebSocketResponse:
        attempts = max(conn_options.max_retry, 0) + 1
        for i in range(attempts):
            final = i == attempts - 1
            try:
                return await self._connect_once(final=final)
            except APIError as e:
                if final or not e.retryable:
                    raise
                logger.warning(
                    "Voxist TTS connect failed, retrying",
                    extra={"attempt": i + 1, "error": str(e)},
                )
                if i > 0:
                    await asyncio.sleep(conn_options.retry_interval)
        raise AssertionError("unreachable")  # pragma: no cover


async def _close(ws: aiohttp.ClientWebSocketResponse) -> None:
    try:
        await ws.close()
    except Exception:  # noqa: BLE001 - best effort; the socket is discarded anyway
        pass


@dataclass
class _Utterance:
    """Progress of one socket, read by the watchdog."""

    first_text_at: float | None = None
    end_sent: bool = False
    audio: bool = False
    last_message_at: float = field(default_factory=time.monotonic)
    audio_bytes: int = 0
    done: dict[str, Any] | None = None

    def deadline(self, opts: _TTSOptions) -> tuple[float, str] | None:
        if self.done is not None:
            return None
        if self.first_text_at is not None and not self.audio:
            return self.first_text_at + opts.first_audio_timeout, "first audio"
        if self.end_sent:
            return self.last_message_at + opts.idle_timeout, "the next server message"
        return None


async def _watchdog(state: _Utterance, opts: _TTSOptions) -> None:
    while True:
        limit = state.deadline(opts)
        now = time.monotonic()
        if limit is not None and now >= limit[0]:
            raise APITimeoutError(f"Voxist TTS: no {limit[1]} in time", retryable=False)
        await asyncio.sleep(_WATCHDOG_TICK_S)


async def _receive(
    ws: aiohttp.ClientWebSocketResponse,
    emitter: tts.AudioEmitter,
    state: _Utterance,
) -> dict[str, Any]:
    """Push audio until ``done``; raise on a fatal error or a premature close."""
    while True:
        msg = await ws.receive()
        state.last_message_at = time.monotonic()
        if msg.type == aiohttp.WSMsgType.BINARY:
            state.audio = True
            state.audio_bytes += len(msg.data)
            emitter.push(msg.data)
            continue
        if msg.type == aiohttp.WSMsgType.TEXT:
            try:
                data = json.loads(msg.data)
            except ValueError:
                logger.warning("Voxist TTS sent a text frame that is not JSON; ignored")
                continue
            if not isinstance(data, dict):
                continue
            if data.get("done") is True:
                state.done = data
                return data
            if "error" in data:
                code = str(data.get("code") or "")
                request_id = data.get("request_id")
                if data.get("severity") == "advisory":
                    logger.warning(
                        "Voxist TTS advisory", extra={"code": code, "request_id": request_id}
                    )
                    continue
                raise APIStatusError(
                    f"Voxist TTS error {code or 'unknown'}",
                    status_code=_ERROR_STATUS.get(code, 500),
                    request_id=request_id if isinstance(request_id, str) else None,
                    body={"code": code},
                    retryable=False,
                )
            continue  # "cancelled" acks and future message types
        raise APIConnectionError(
            f"Voxist TTS socket closed before done (close code {ws.close_code})",
            retryable=False,
        )


def _check_done(done: dict[str, Any], state: _Utterance) -> None:
    duration = done.get("duration_s")
    if isinstance(duration, (int, float)) and duration > 0:
        received = state.audio_bytes / 2 / SAMPLE_RATE
        if abs(received - duration) > max(0.25, 0.05 * duration):
            logger.warning(
                "Voxist TTS audio length differs from done (rate mismatch?)",
                extra={"received_s": round(received, 3), "duration_s": duration},
            )


async def _run_tasks(
    sender: asyncio.Task[Any] | None,
    receiver: asyncio.Task[dict[str, Any]],
    watchdog: asyncio.Task[None],
) -> dict[str, Any]:
    pending: set[asyncio.Task[Any]] = {receiver, watchdog}
    if sender is not None:
        pending.add(sender)
    try:
        while not receiver.done():
            done, pending = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
            for task in done:
                if task.exception() is not None:
                    raise task.exception()  # type: ignore[misc]
        return receiver.result()
    finally:
        tasks = [t for t in (sender, receiver, watchdog) if t is not None]
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)


class VoxistTTSChunkedStream(tts.ChunkedStream):
    """One-shot synthesis of a whole text over the same socket."""

    def __init__(self, *, tts: VoxistTTS, input_text: str, conn_options: APIConnectOptions):
        super().__init__(tts=tts, input_text=input_text, conn_options=conn_options)
        self._voxist = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=FRAME_SIZE_MS,
        )
        if not self._input_text.strip():
            return
        ws = await self._voxist._connect(self._conn_options)
        state = _Utterance()
        try:
            await ws.send_str(json.dumps({"text": self._input_text}))
            state.first_text_at = time.monotonic()
            await ws.send_str(json.dumps({"end": True}))
            state.end_sent = True
            done = await _run_tasks(
                None,
                asyncio.create_task(_receive(ws, output_emitter, state)),
                asyncio.create_task(_watchdog(state, self._voxist._opts)),
            )
            _check_done(done, state)
        except (aiohttp.ClientError, ConnectionError) as e:
            raise APIConnectionError(
                f"Voxist TTS socket failed ({type(e).__name__})", retryable=False
            ) from None
        finally:
            await _close(ws)
        output_emitter.flush()


class VoxistTTSStream(tts.SynthesizeStream):
    """Streaming synthesis: text is forwarded as it is pushed, audio as it arrives."""

    def __init__(self, *, tts: VoxistTTS, conn_options: APIConnectOptions):
        super().__init__(tts=tts, conn_options=conn_options)
        self._voxist = tts

    async def _run(self, output_emitter: tts.AudioEmitter) -> None:
        output_emitter.initialize(
            request_id=shortuuid(),
            sample_rate=SAMPLE_RATE,
            num_channels=1,
            mime_type="audio/pcm",
            frame_size_ms=FRAME_SIZE_MS,
            stream=True,
        )
        # The first socket is dialed before any text, so the handshake overlaps the
        # LLM's first token. A later segment (livekit-agents before 1.8 allowed
        # several per stream) dials once its first text arrives.
        first: str | None = None
        while True:
            ws = await self._voxist._connect(self._conn_options)
            try:
                input_open = await self._segment(ws, output_emitter, first)
            except (aiohttp.ClientError, ConnectionError) as e:
                raise APIConnectionError(
                    f"Voxist TTS socket failed ({type(e).__name__})", retryable=False
                ) from None
            finally:
                await _close(ws)
            if not input_open:
                return
            first = None
            async for data in self._input_ch:
                if isinstance(data, str) and data:
                    first = data
                    break
            if first is None:
                return

    async def _segment(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
        first: str | None,
    ) -> bool:
        """Speak one segment on `ws`. Returns False once the input has ended."""
        state = _Utterance()
        input_open = True

        async def texts() -> AsyncIterator[str | tts.SynthesizeStream._FlushSentinel]:
            if first is not None:
                yield first
            async for data in self._input_ch:
                yield data

        async def send() -> None:
            nonlocal input_open
            started = False
            async for data in texts():
                if isinstance(data, self._FlushSentinel):
                    if started:
                        break
                    continue
                if not data:
                    continue
                if not started:
                    output_emitter.start_segment(segment_id=shortuuid())
                    started = True
                await ws.send_str(json.dumps({"text": data}))
                if state.first_text_at is None:
                    state.first_text_at = time.monotonic()
                    self._mark_started()
            else:
                input_open = False
            if started:
                await ws.send_str(json.dumps({"end": True}))
                state.last_message_at = time.monotonic()
                state.end_sent = True

        sender = asyncio.create_task(send())
        receiver = asyncio.create_task(_receive(ws, output_emitter, state))
        watchdog = asyncio.create_task(_watchdog(state, self._voxist._opts))
        try:
            # Nothing to say: the input ended (or only flushed) with no text.
            await asyncio.wait({sender, receiver, watchdog}, return_when=asyncio.FIRST_COMPLETED)
            if sender.done() and sender.exception() is None and state.first_text_at is None:
                return input_open
            done = await _run_tasks(sender, receiver, watchdog)
        finally:
            for task in (sender, receiver, watchdog):
                task.cancel()
            await asyncio.gather(sender, receiver, watchdog, return_exceptions=True)
        _check_done(done, state)
        output_emitter.end_segment()
        return input_open
