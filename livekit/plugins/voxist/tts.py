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
  ``retry_interval``; the last attempt's error is non-retryable),
  ``first_audio_timeout`` from the latest text sent to the first audio frame (so
  the time an LLM takes to finish a sentence is not counted), capped at
  ``FIRST_AUDIO_CEILING_FACTOR`` x ``first_audio_timeout`` from the first text;
  once audio has started, ``idle_timeout`` without a server message or a text
  sent (a stall), and after ``end``, ``idle_timeout`` between server messages.
- Retries by LiveKit: only a fatal ``backend_error`` (on this endpoint, the
  server's "No slots available, please retry" or a voice-encoding failure) or
  ``capacity_exceeded`` (the server's connection cap, shared with other
  services, or the OmniVoice worker being at capacity: one retry per stream at
  most) that arrives before any audio of the
  stream is retryable; livekit-agents (>= 1.8.8) then replays the buffered text
  on a new socket. Everything else is non-retryable, so text the caller already
  heard is never spoken twice. The server's message travels in
  ``APIStatusError.body["message"]``.
- Text limits: after a ``text_limit_reached`` advisory nothing more is sent (the
  contract's "stop sending"); the rest of that segment is dropped and ``done`` is
  awaited. A ``done`` reporting ``limit_reached`` or ``ignored_messages`` is
  logged as a warning.
- Credentials: ``headers`` (for example a Cloudflare Access service token) go on
  the handshake only, are never logged and never appear in an error message. A
  redirect is never followed, since it would carry them to another URL, and a
  plain ``http``/``ws`` base URL is refused unless the host is loopback.
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
#: However the text trickles in, first audio must come within this many
#: ``first_audio_timeout`` of the first text.
FIRST_AUDIO_CEILING_FACTOR = 3
IDLE_TIMEOUT_DEFAULT = 6.0
#: How long a close waits for the server's close frame before dropping the socket.
CLOSE_TIMEOUT = 1.0
#: How long :meth:`VoxistTTS.check_connection` listens for a refusal after the config.
CHECK_QUIET_S = 0.5
_WATCHDOG_TICK_S = 0.05

#: Voice ids that collide with other server routes (``/v1/tts/jobs`` ...) or that a
#: URL would resolve as a path step.
RESERVED_VOICE_IDS = frozenset({"jobs", "languages", "voices", ".", ".."})
_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1"})

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
#: Fatal codes worth a new socket, as long as nothing has been heard yet.
#: ``session_timeout`` is not one: voxist-tts sends it on the batch ``/stream``
#: socket only (on ``/stream/text`` the timed-out session is already gone).
RETRYABLE_CODES = frozenset({"capacity_exceeded", "backend_error"})
#: The connection cap is shared (Voxlingo uses the same service) and the
#: OmniVoice worker can be at capacity too: retry it once.
CAPACITY_RETRIES = 1
#: Longest server message carried into an error (the server bounds its own echoes).
_MAX_MESSAGE_CHARS = 200
_TEXT_LIMIT_CODE = "text_limit_reached"


class _RedirectRefused(Exception):
    def __init__(self, status: int) -> None:
        super().__init__(status)
        self.status = status


async def _refuse_redirect(
    _session: aiohttp.ClientSession, _ctx: Any, params: aiohttp.TraceRequestRedirectParams
) -> None:
    status = params.response.status
    params.response.close()  # release the connection: nothing of it is read or reused
    raise _RedirectRefused(status)


def _ws_base(base_url: str) -> str:
    parts = urlsplit(base_url.rstrip("/"))
    scheme = {"https": "wss", "http": "ws", "wss": "wss", "ws": "ws"}.get(parts.scheme)
    if scheme is None or not parts.netloc:
        raise ValueError("base_url must be an http(s) or ws(s) URL with a host")
    if parts.query or parts.fragment:
        raise ValueError("base_url must not carry a query or a fragment")
    if scheme == "ws" and parts.hostname not in _LOOPBACK_HOSTS:
        raise ValueError(
            "base_url must be https:// or wss:// (plain http/ws only to a loopback host): "
            "the handshake carries the credentials"
        )
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
        base_url: Service root, e.g. ``https://tts.voxist-api.com`` (``wss://``
            accepted too; ``http``/``ws`` only to a loopback host).
        voice_id: A Voxist voice id (``ada``, ``103488-es`` ...).
        language: The language the voice speaks, as the server takes it (``fr``).
        headers: Extra handshake headers, e.g. ``CF-Access-Client-Id`` /
            ``CF-Access-Client-Secret``. Never logged.
        connect_timeout: Seconds for one handshake, config included.
        first_audio_timeout: Seconds from the latest text sent to the first audio,
            capped at 3x this value measured from the first text.
        idle_timeout: Seconds without a server message or a text once audio has
            started (a stall), and between server messages once ``end`` is sent.
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

    async def check_connection(self, *, quiet_s: float = CHECK_QUIET_S) -> None:
        """Open the socket, send the config, listen ``quiet_s`` for a refusal, close.

        A readiness probe for the streaming path itself (handshake, Access, voice
        and language as the server resolves them on this socket). Nothing is
        synthesised. One dial, no retry.

        Raises:
            APIStatusError: the handshake was refused (``status_code`` is the HTTP
                status) or the server sent a fatal error frame (``body["code"]``).
            APIConnectionError / APITimeoutError: the service could not be reached,
                or the socket closed before ``quiet_s`` ran out.
        """
        ws = await self._connect_once(final=True)
        try:
            deadline = time.monotonic() + quiet_s
            while (left := deadline - time.monotonic()) > 0:
                try:
                    msg = await asyncio.wait_for(ws.receive(), left)
                except asyncio.TimeoutError:
                    return
                if msg.type == aiohttp.WSMsgType.TEXT:
                    data = _json_object(msg.data)
                    if data is not None and "error" in data:
                        if data.get("severity") != "advisory":
                            raise _fatal_error(data, heard=True, attempts=_Attempts())
                        _log_advisory(data)
                    continue
                if msg.type == aiohttp.WSMsgType.BINARY:
                    continue
                raise APIConnectionError(
                    f"Voxist TTS socket closed during the check (close code {ws.close_code})",
                    retryable=False,
                )
        finally:
            await _close(ws)

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
        await ws.close()  # bounded by CLOSE_TIMEOUT (ClientWSTimeout.ws_close)
    except Exception:  # noqa: BLE001 - best effort; the socket is discarded anyway
        pass


def _json_object(raw: str) -> dict[str, Any] | None:
    try:
        data = json.loads(raw)
    except ValueError:
        logger.warning("Voxist TTS sent a text frame that is not JSON; ignored")
        return None
    return data if isinstance(data, dict) else None


def _log_advisory(data: dict[str, Any]) -> None:
    logger.warning(
        "Voxist TTS advisory",
        extra={"code": str(data.get("code") or ""), "request_id": data.get("request_id")},
    )


@dataclass
class _Attempts:
    """Per-stream retry budget, kept across LiveKit's attempts (same stream object)."""

    capacity_left: int = CAPACITY_RETRIES


def _fatal_error(data: dict[str, Any], *, heard: bool, attempts: _Attempts) -> APIStatusError:
    code = str(data.get("code") or "")
    request_id = data.get("request_id")
    prose = data.get("error")
    message = prose[:_MAX_MESSAGE_CHARS] if isinstance(prose, str) else ""
    retryable = code in RETRYABLE_CODES and not heard
    if retryable and code == "capacity_exceeded":
        retryable = attempts.capacity_left > 0
        attempts.capacity_left -= 1
    # backend_error's text is what tells "busy" (No slots available) from a failure.
    text = f"Voxist TTS error {code or 'unknown'}"
    if code == "backend_error" and message:
        text += f": {message}"
    return APIStatusError(
        text,
        status_code=_ERROR_STATUS.get(code, 500),
        request_id=request_id if isinstance(request_id, str) else None,
        body={"code": code, "message": message},
        retryable=retryable,
    )


@dataclass
class _Utterance:
    """Progress of one socket, read by the watchdog."""

    heard_before: bool = False  # audio of an earlier socket of the same stream
    attempts: _Attempts = field(default_factory=_Attempts)
    first_text_at: float | None = None
    last_text_at: float | None = None
    end_sent: bool = False
    limit_reached: bool = False
    audio: bool = False
    last_message_at: float = field(default_factory=time.monotonic)
    audio_bytes: int = 0
    done: dict[str, Any] | None = None

    def deadline(self, opts: _TTSOptions) -> tuple[float, str] | None:
        if self.done is not None:
            return None
        if self.last_text_at is not None and not self.audio:
            ceiling = (self.first_text_at or self.last_text_at) + (
                FIRST_AUDIO_CEILING_FACTOR * opts.first_audio_timeout
            )
            return min(self.last_text_at + opts.first_audio_timeout, ceiling), "first audio"
        if self.end_sent or self.limit_reached:
            return self.last_message_at + opts.idle_timeout, "the next server message"
        if self.audio and self.last_text_at is not None:
            # Speaking, more text may come: a stall is neither a frame nor a text.
            last = max(self.last_message_at, self.last_text_at)
            return last + opts.idle_timeout, "audio after the last frame"
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
            data = _json_object(msg.data)
            if data is None:
                continue
            if data.get("done") is True:
                state.done = data
                return data
            if "error" in data:
                if data.get("severity") == "advisory":
                    _log_advisory(data)
                    if data.get("code") == _TEXT_LIMIT_CODE:
                        state.limit_reached = True
                    continue
                raise _fatal_error(
                    data, heard=state.audio or state.heard_before, attempts=state.attempts
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
    ignored = done.get("ignored_messages")
    if done.get("limit_reached") is True or (isinstance(ignored, int) and ignored > 0):
        logger.warning(
            "Voxist TTS did not speak all the text it was sent",
            extra={
                "limit_reached": done.get("limit_reached") is True,
                "ignored_messages": ignored if isinstance(ignored, int) else 0,
                "stop_reason": done.get("stop_reason"),
                "request_id": done.get("request_id"),
            },
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
        self._attempts = _Attempts()

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
        state = _Utterance(attempts=self._attempts)
        try:
            await ws.send_str(json.dumps({"text": self._input_text}))
            state.first_text_at = state.last_text_at = time.monotonic()
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
        self._heard = False
        self._attempts = _Attempts()

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

    async def _drain_segment(self) -> bool:
        """Drop the rest of the current segment. False once the input has ended."""
        async for data in self._input_ch:
            if isinstance(data, self._FlushSentinel):
                return True
        return False

    async def _segment(
        self,
        ws: aiohttp.ClientWebSocketResponse,
        output_emitter: tts.AudioEmitter,
        first: str | None,
    ) -> bool:
        """Speak one segment on `ws`. Returns False once the input has ended."""
        state = _Utterance(heard_before=self._heard, attempts=self._attempts)
        input_open = True
        segment_over = False

        async def texts() -> AsyncIterator[str | tts.SynthesizeStream._FlushSentinel]:
            if first is not None:
                yield first
            async for data in self._input_ch:
                yield data

        async def send() -> None:
            nonlocal input_open, segment_over
            started = False
            async for data in texts():
                if isinstance(data, self._FlushSentinel):
                    if started:
                        segment_over = True
                        break
                    continue
                if not data:
                    continue
                if state.limit_reached:
                    continue  # the server stopped reading: send nothing more
                if not started:
                    output_emitter.start_segment(segment_id=shortuuid())
                    started = True
                await ws.send_str(json.dumps({"text": data}))
                if state.first_text_at is None:
                    self._mark_started()
                    state.first_text_at = time.monotonic()
                state.last_text_at = time.monotonic()
            else:
                input_open = False
                segment_over = True
            if started and not state.limit_reached:
                await ws.send_str(json.dumps({"end": True}))
                state.last_message_at = time.monotonic()
                state.end_sent = True

        sender = asyncio.create_task(send())
        receiver = asyncio.create_task(_receive(ws, output_emitter, state))
        watchdog = asyncio.create_task(_watchdog(state, self._voxist._opts))
        try:
            # Nothing to say: the input ended (or only flushed) with no text.
            await asyncio.wait({sender, receiver, watchdog}, return_when=asyncio.FIRST_COMPLETED)
            if sender.done() and sender.exception() is None and state.last_text_at is None:
                return input_open
            done = await _run_tasks(sender, receiver, watchdog)
        finally:
            self._heard = self._heard or state.audio
            for task in (sender, receiver, watchdog):
                task.cancel()
            await asyncio.gather(sender, receiver, watchdog, return_exceptions=True)
        _check_done(done, state)
        output_emitter.end_segment()
        if not segment_over:  # done came first (text limit): drop the segment's rest
            input_open = await self._drain_segment()
        return input_open
