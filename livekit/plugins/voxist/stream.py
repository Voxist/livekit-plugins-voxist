"""VoxistSTTStream - Streaming recognition interface.

One WebSocket per stream, matching the gateway's protocol:

- The socket is dialed by this stream, for this stream, with the language in
  the URL. There is nothing to renegotiate and no window where audio can meet
  the wrong engine.
- flush() marks the end of a SEGMENT. The engine does its own endpointing and
  emits a final per silence-delimited segment on the same socket (the gateway
  threads context across finals - they are the normal flow, not a special
  case). No "Done" is sent at segment boundaries.
- end_input() ends the SESSION: "Done" is written once, the engine flushes,
  and the gateway closes the client socket
  (simple-websocket-proxy.gateway.ts:899). The server close after "Done" is
  the expected terminator, not a failure.
- Retrying is owned by livekit: RecognizeStream._main_task already retries
  _run() up to conn_options.max_retry with proper error events. _run therefore
  performs exactly ONE attempt and raises APIConnectionError on interruption.
  An earlier design wrapped its own reconnect loop inside the framework's,
  and every retry-budget bug in this plugin's history lived in that
  duplication.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import time
from collections.abc import Awaitable
from typing import TYPE_CHECKING

import aiohttp
import numpy as np

# Required, not optional: a fallback shim here once silently disabled BCP-47
# normalization on older livekit-agents. The dependency floor guarantees the
# import, and failing loudly beats diverging quietly.
from livekit.agents import APIConnectionError, LanguageCode, utils
from livekit.agents.stt import (
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
)

from livekit import rtc  # type: ignore[attr-defined]

from .audio_processor import AudioProcessor
from .exceptions import ConnectionError as VoxistConnectionError
from .log import logger

if TYPE_CHECKING:
    from .stt import VoxistSTT


class VoxistSTTStream(RecognizeStream):
    """
    Streaming interface for Voxist ASR.

    Event Flow:
        1. START_OF_SPEECH (when first text detected)
        2. INTERIM_TRANSCRIPT (partial results, if enabled)
        3. FINAL_TRANSCRIPT (one per silence-delimited segment)
        4. END_OF_SPEECH (when the session ends)

    Example:
        stream = stt.stream(language="fr-medical")

        for frame in audio_frames:
            stream.push_frame(frame)
        stream.end_input()

        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                print(event.alternatives[0].text)
    """

    # Backpressure is owned entirely by aiohttp: `await ws.send_bytes()` drains
    # while the transport is paused, and the transport decides when to pause
    # using its own limits. Those limits differ per transport - 64KB for a
    # plain socket, 512KB for asyncio's SSL transport - so the plugin does NOT
    # compare the buffer against thresholds of its own. An earlier version did,
    # with marks calibrated for a plain socket; over wss:// the low mark sat
    # below the level the SSL layer relieves to, making the release condition
    # unreachable and throttling the stream to one chunk per timeout.
    #
    # What the plugin guarantees instead, independent of transport type:
    #   1. no send blocks longer than SEND_TIMEOUT_SECONDS,
    #   2. the input backlog is bounded by MAX_INPUT_BACKLOG_FRAMES.

    # A send that cannot complete in this long means the uplink is stalled.
    SEND_TIMEOUT_SECONDS = 5.0

    # How long to keep receiving after "Done" has been written. The gateway
    # closes the socket once the engine has flushed - normally within
    # milliseconds - so this is a generous watchdog, not an expected wait.
    # It is the ONLY receive-side timeout: during streaming, transport death
    # is detected by aiohttp's heartbeat (missed pong closes the socket), so
    # long user silences cannot false-trigger a watchdog.
    SESSION_DRAIN_TIMEOUT_SECONDS = 30.0

    # Rate limit for the audio-drop warning. The drop condition persists for
    # the whole overload, and the send loop runs per 10ms frame, so an
    # unlimited warning would emit ~100 lines/second per stream.
    DROP_LOG_INTERVAL_SECONDS = 5.0

    # Cap on unsent audio frames held in the input channel. livekit's channel
    # is unbounded and push_frame() never blocks, so without this a slow
    # uplink grows the backlog until the process is OOM-killed. At 10ms frames
    # this is ~10s of audio; beyond that, transcripts would arrive too late to
    # be useful anyway, so the oldest frames are dropped rather than queued.
    MAX_INPUT_BACKLOG_FRAMES = 1000

    def __init__(
        self,
        *,
        stt: VoxistSTT,
        config: dict,
        language: str,
        conn_options,
        enable_metrics: bool = True,
    ):
        """
        Initialize streaming recognition session.

        Args:
            stt: Parent VoxistSTT instance (owns the dialer and HTTP session)
            config: Configuration dictionary
            language: Language code for this stream
            conn_options: LiveKit connection options
            enable_metrics: Whether to emit metrics events
        """
        super().__init__(
            stt=stt,
            conn_options=conn_options,
            sample_rate=config["sample_rate"],
        )

        self._voxist_stt = stt
        self._config = config
        self._language = language
        # Language reported on emitted SpeechData. livekit normalizes this to
        # BCP-47, which uppercases the subtag: "fr-medical" is emitted as
        # "fr-MEDICAL". The engine language matches by construction: this
        # stream's socket is dialed with self._language in the URL.
        self._speech_language = LanguageCode(language)
        self._enable_metrics = enable_metrics

        self._session_id = utils.shortuuid()
        self._speaking = False
        self._ws: aiohttp.ClientWebSocketResponse | None = None

        # Session outcome. "Done" written once marks the input as fully
        # delivered; session_complete marks the whole exchange as finished so
        # a framework retry after a late failure does not redial pointlessly
        # (the audio of a dead session is unrecoverable - streaming ASR cannot
        # replay what was already consumed).
        self._done_sent = False
        self._session_complete = False

        # Latch so an unreachable transport is reported once, not per chunk
        self._transport_lookup_failed = False
        # Count of frames dropped to keep the input backlog bounded, and when
        # that was last reported. None means "not yet" - monotonic() has an
        # arbitrary epoch, so 0.0 is not a usable sentinel on a fresh host.
        self._dropped_frames = 0
        self._last_drop_log: float | None = None

        # Audio processor for format conversion and chunking
        self._audio_processor = AudioProcessor(
            sample_rate=config["sample_rate"],
            chunk_duration_ms=config["chunk_duration_ms"],
            stride_overlap_ms=config["stride_overlap_ms"],
            target_sample_rate=16000,  # Voxist expects 16kHz audio
        )

        logger.debug(
            f"Stream {self._session_id} created: "
            f"language={language}, sample_rate={config['sample_rate']}, "
            f"chunk_duration_ms={config['chunk_duration_ms']}"
        )

    async def _run(self) -> None:
        """
        One attempt at the session. Retries belong to the framework.

        Raises:
            APIConnectionError: The session was interrupted (dial failure,
                stalled send, server close before end of input). livekit's
                _main_task catches this, emits a recoverable error event, and
                calls _run() again up to conn_options.max_retry.
            AuthenticationError: The key was rejected. Deliberately NOT an
                APIError: retrying cannot fix a revoked key, so it propagates
                immediately as the true cause.
        """
        if self._session_complete:
            # A previous attempt already finished the exchange; a late error
            # (e.g. during drain) triggered a retry with nothing to recover.
            return

        try:
            ws = await self._voxist_stt._dial(self._language)
        except VoxistConnectionError as e:
            raise APIConnectionError(str(e)) from e

        self._ws = ws
        send_task = asyncio.create_task(
            self._send_audio_task(), name=f"send-{self._session_id}"
        )
        recv_task = asyncio.create_task(
            self._recv_results_task(), name=f"recv-{self._session_id}"
        )

        try:
            done, pending = await asyncio.wait(
                {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if send_task in done:
                exc = send_task.exception()
                if exc is not None:
                    raise exc

                # Input delivered and "Done" written. The trailing finals are
                # still being computed; the gateway closes the socket once the
                # engine has flushed, which ends the receive task naturally.
                if recv_task in pending:
                    try:
                        await asyncio.wait_for(
                            recv_task, timeout=self.SESSION_DRAIN_TIMEOUT_SECONDS
                        )
                    except asyncio.TimeoutError:
                        logger.warning(
                            f"Stream {self._session_id} server neither closed "
                            f"nor answered within "
                            f"{self.SESSION_DRAIN_TIMEOUT_SECONDS}s of Done - "
                            "a trailing transcript may have been lost"
                        )

            if recv_task.done() and not recv_task.cancelled():
                exc = recv_task.exception()
                if exc is not None:
                    raise exc
                if not self._done_sent:
                    # The server ended the session while input was still
                    # flowing: an interruption, not a completion. Before this
                    # was classified explicitly, a mid-call close truncated
                    # the transcript silently and reported success.
                    raise APIConnectionError(
                        "server closed the connection before end of input"
                    )

            self._session_complete = True

            if self._speaking:
                self._speaking = False
                logger.debug(
                    f"Stream {self._session_id} emitting END_OF_SPEECH"
                )
                self._event_ch.send_nowait(
                    SpeechEvent(
                        type=SpeechEventType.END_OF_SPEECH,
                        request_id=self._session_id,
                    )
                )

            logger.debug(f"Stream {self._session_id} session complete")

        finally:
            for task in (send_task, recv_task):
                if not task.done():
                    task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await task
            if not ws.closed:
                with contextlib.suppress(Exception):
                    await ws.close()
            self._ws = None

    async def _send_audio_task(self) -> None:
        """
        Deliver audio to the socket; write "Done" when the input ends.

        The backlog bound is applied on consumption: when the queue behind the
        frame in hand exceeds the cap, the frame is discarded instead of sent,
        so the loop drains the excess at full speed and keeps the most recent
        audio. Nothing is ever removed from the channel out of order, so flush
        sentinels are always honoured in sequence.
        """
        frame_count = 0

        async for data in self._input_ch:
            frame_count += 1
            if frame_count % 100 == 0:
                logger.debug(
                    f"Stream {self._session_id} processing frame {frame_count}"
                )

            if isinstance(data, self._FlushSentinel):
                # End of SEGMENT, not of session. Ship whatever the processor
                # is holding so the engine has the full segment to finalize.
                for chunk in self._audio_processor.flush():
                    await self._send_audio_chunk(chunk)
                await self._on_segment_end()
                continue

            if isinstance(data, rtc.AudioFrame):
                if self._input_ch.qsize() > self.MAX_INPUT_BACKLOG_FRAMES:
                    self._note_dropped_frame()
                    continue

                frame_bytes = bytes(data.data)
                for chunk in self._audio_processor.process_audio_frame(frame_bytes):
                    await self._send_audio_chunk(chunk)
            else:
                logger.warning(
                    f"Stream {self._session_id} unexpected data type: {type(data)}"
                )

        # Input channel closed: the session is over. end_input() pushed a
        # final sentinel before closing, so the processor is already flushed;
        # a second flush() returning [] is the expected no-op.
        for chunk in self._audio_processor.flush():
            await self._send_audio_chunk(chunk)

        if self._ws is not None and not self._ws.closed:
            await self._send_with_timeout(
                self._ws.send_str("Done"), "Done signal"
            )
            self._done_sent = True
            logger.debug(f"Stream {self._session_id} sent Done signal")
        elif not self._done_sent:
            # No socket to finalize on - the session cannot complete cleanly.
            raise APIConnectionError(
                "connection lost before the Done signal could be sent"
            )

    async def _on_segment_end(self) -> None:
        """
        Hook for the flush policy. Currently: nothing.

        The engine emits finals per silence-delimited segment on its own (the
        gateway threads precedingContext across finals - multi-final sessions
        are its normal operation), so a segment boundary needs no signal. If
        the live probe (scratchpad/probe_finals.py) ever shows an engine that
        only finalizes on "Done", this hook is where the policy changes:
        send "Done", drain, and rotate the socket - without touching the rest
        of the loop.
        """

    @property
    def dropped_frames(self) -> int:
        """
        Audio frames discarded to bound the input backlog.

        Non-zero means the transcript for this stream has gaps: the uplink
        could not keep up and audio was deliberately dropped. Exposed so a
        caller can tell a lossy transcript from a complete one - the loss is
        otherwise only visible in the logs, and the emitted events look
        normal.
        """
        return self._dropped_frames

    def _note_dropped_frame(self) -> None:
        """
        Count a dropped frame and report it, rate-limited.

        The drop condition persists for as long as the uplink is behind, and
        the send loop iterates per 10ms frame, so logging every drop would
        emit ~100 lines/second per stream during exactly the overload being
        reported.
        """
        self._dropped_frames += 1

        now = time.monotonic()
        if (
            self._last_drop_log is not None
            and now - self._last_drop_log < self.DROP_LOG_INTERVAL_SECONDS
        ):
            return
        self._last_drop_log = now

        logger.warning(
            f"Stream {self._session_id} dropping audio to bound the input "
            f"backlog (cap {self.MAX_INPUT_BACKLOG_FRAMES} frames, "
            f"{self._dropped_frames} dropped so far on this stream) - the "
            "uplink is not keeping up with real time, so the transcript will "
            "have gaps"
        )

    async def _send_with_timeout(self, coro: Awaitable[None], what: str) -> None:
        """
        Await a WebSocket send, bounded by SEND_TIMEOUT_SECONDS.

        Every send on this stream goes through here. aiohttp's drain has no
        timeout of its own, so an unbounded send would block the send loop
        indefinitely while the input backlog grows. On a stall the transport
        is aborted - a cancelled send may have left a frame mid-flight, so
        the socket is no longer safe to write to - and the failure surfaces
        as APIConnectionError for the framework to retry.

        Args:
            coro: The send coroutine to await
            what: Short description for the error path (e.g. "audio chunk")

        Raises:
            APIConnectionError: If the send stalls past SEND_TIMEOUT_SECONDS.
        """
        try:
            await asyncio.wait_for(coro, timeout=self.SEND_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as e:
            buffer_size = self._get_write_buffer_size()
            logger.warning(
                f"Stream {self._session_id} send stalled for "
                f"{self.SEND_TIMEOUT_SECONDS}s sending {what} "
                f"(write buffer={buffer_size}B); abandoning the connection"
            )
            transport = self._get_transport()
            if transport is not None:
                transport.abort()  # immediate, does not attempt to flush
            raise APIConnectionError(
                "WebSocket send timeout - connection stalled"
            ) from e

    def _get_transport(self) -> asyncio.Transport | None:
        """
        Reach the asyncio transport underlying the WebSocket, for diagnostics.

        Prefers a public accessor if the installed aiohttp grows one;
        otherwise walks ws._response.connection, which is not public API.
        Used only for observability and the stall-abort - a None here degrades
        logging, never behaviour.
        """
        ws = self._ws
        if ws is None:
            return None

        transport = None
        getter = getattr(ws, "get_transport", None)
        if callable(getter):
            try:
                transport = getter()
            except (AttributeError, RuntimeError):
                transport = None

        if transport is None:
            try:
                connection = ws._response.connection  # type: ignore[attr-defined]
                transport = connection.transport if connection is not None else None
            except (AttributeError, RuntimeError):
                transport = None

        if transport is not None:
            try:
                if transport.is_closing():
                    return None
                return transport  # type: ignore[no-any-return]
            except (AttributeError, RuntimeError):
                transport = None

        if not self._transport_lookup_failed:
            self._transport_lookup_failed = True
            logger.debug(
                f"Stream {self._session_id} cannot reach the WebSocket "
                "transport; write-buffer metrics unavailable for this stream "
                "(aiohttp internals may have changed)"
            )
        return None

    def _get_write_buffer_size(self) -> int:
        """
        Measure the transport write buffer size, for observability only.

        Deliberately NOT used for flow control: backpressure belongs to
        aiohttp, which drains inside send_bytes() using the transport's own
        pause state rather than any absolute byte count.

        Returns:
            Buffer size in bytes, or 0 if it cannot be measured.
        """
        transport = self._get_transport()
        if transport is None:
            return 0
        try:
            return int(transport.get_write_buffer_size())
        except (AttributeError, RuntimeError):
            return 0

    async def _send_audio_chunk(self, audio_int16: np.ndarray) -> None:
        """
        Send an Int16 PCM audio chunk to the WebSocket.

        Raises:
            APIConnectionError: The socket is gone or the send stalled. Not
                skippable: silently dropping chunks on a closed socket once
                made a mid-call close look like a clean end of input, so the
                stream "completed" with a truncated transcript.
        """
        if self._ws is None or self._ws.closed:
            raise APIConnectionError(
                "WebSocket closed while sending audio - connection lost"
            )

        audio_bytes = audio_int16.tobytes()
        await self._send_with_timeout(
            self._ws.send_bytes(audio_bytes), "audio chunk"
        )

    async def _recv_results_task(self) -> None:
        """
        Receive transcription results until the server closes the socket.

        No receive watchdog during streaming: the engine is silent while the
        user is silent, and aiohttp's heartbeat already closes the socket on a
        missed pong, which ends this loop. The only bounded wait is the
        post-Done drain in _run.
        """
        assert self._ws is not None
        async for msg in self._ws:
            if msg.type == aiohttp.WSMsgType.TEXT:
                try:
                    data = json.loads(msg.data)
                except json.JSONDecodeError:
                    logger.error(
                        f"Stream {self._session_id} invalid JSON received"
                    )
                    logger.debug(
                        f"Stream {self._session_id} invalid JSON: {msg.data[:80]}"
                    )
                    continue
                await self._process_result(data)
            elif msg.type == aiohttp.WSMsgType.ERROR:
                raise APIConnectionError("WebSocket connection error occurred")

        logger.debug(
            f"Stream {self._session_id} server closed the socket"
        )

    async def _process_result(self, data: dict):
        """
        Process a transcription result from Voxist and emit events.

        Voxist Message Format:
            {"type": "partial"|"final", "text": "...", "confidence": 0.95, ...}

        Args:
            data: Parsed JSON message from Voxist
        """
        msg_type = data.get("type")

        # Detect start of speech
        text = data.get("text", "").strip()
        if not self._speaking and text:
            self._speaking = True
            logger.debug(f"Stream {self._session_id} speech started")

            self._event_ch.send_nowait(
                SpeechEvent(
                    type=SpeechEventType.START_OF_SPEECH,
                    request_id=self._session_id,
                )
            )

        # Interim results (partial transcription)
        if msg_type == "partial":
            if text and self._config["interim_results"]:
                logger.debug(f"Stream {self._session_id} interim: {text[:50]}")

                event = SpeechEvent(
                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._session_id,
                    alternatives=[
                        SpeechData(
                            language=self._speech_language,
                            text=text,
                            confidence=data.get("confidence", 1.0),
                        )
                    ],
                )
                self._event_ch.send_nowait(event)

        # Final results (confirmed transcription)
        elif msg_type == "final":
            if text:
                logger.info(f"Stream {self._session_id} final: {text[:100]}")

                event = SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._session_id,
                    alternatives=[
                        SpeechData(
                            language=self._speech_language,
                            text=text,
                            confidence=data.get("confidence", 1.0),
                        )
                    ],
                )
                self._event_ch.send_nowait(event)
                # END_OF_SPEECH is NOT emitted here: the engine sends one
                # final per segment during continuous speech. It is emitted
                # when the session completes (see _run).

        # Server-reported error
        elif msg_type == "error":
            error_msg = data.get("message", "Unknown error")
            logger.error(f"Stream {self._session_id} Voxist error: {error_msg}")
            raise APIConnectionError(f"Voxist error: {error_msg}")

        # Unknown message type
        elif msg_type:
            logger.warning(
                f"Stream {self._session_id} unknown message type: {msg_type}"
            )
