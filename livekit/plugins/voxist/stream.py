"""VoxistSTTStream - Streaming recognition interface."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Awaitable
from typing import TYPE_CHECKING

import aiohttp
import numpy as np
from livekit.agents import utils
from livekit.agents.stt import (
    RecognizeStream,
    SpeechData,
    SpeechEvent,
    SpeechEventType,
)

try:  # livekit-agents >= 1.x
    from livekit.agents import LanguageCode
except ImportError:  # pragma: no cover - older livekit-agents has no such type
    LanguageCode = str  # type: ignore[assignment, misc]

from livekit import rtc  # type: ignore[attr-defined]

from .audio_processor import AudioProcessor
from .exceptions import OwnershipViolationError
from .log import logger
from .models import Connection

if TYPE_CHECKING:
    from .connection_pool import ConnectionPool
    from .stt import VoxistSTT


class VoxistSTTStream(RecognizeStream):
    """
    Streaming interface for Voxist ASR.

    Implements concurrent send/receive pattern for optimal latency:
    - Send task: Converts and streams audio to WebSocket
    - Receive task: Processes transcription results and emits events

    Event Flow:
        1. START_OF_SPEECH (when first text detected)
        2. INTERIM_TRANSCRIPT (partial results, if enabled)
        3. FINAL_TRANSCRIPT (confirmed transcription)
        4. END_OF_SPEECH (after final result)

    Example:
        stream = stt.stream(language="fr-medical")

        # Push audio frames
        for frame in audio_frames:
            stream.push_frame(frame)

        stream.end_input()

        # Consume events
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
    # Must stay well below RECEIVE_TIMEOUT_SECONDS so a stalled send surfaces
    # as a send error and reconnect, rather than tripping the receive watchdog
    # and looking like a server-side stall.
    SEND_TIMEOUT_SECONDS = 5.0

    # Receive watchdog: no message from Voxist for this long means the
    # connection is stalled. Class-level so the invariant against
    # SEND_TIMEOUT_SECONDS is visible and testable.
    RECEIVE_TIMEOUT_SECONDS = 30.0

    # How long to keep receiving after "Done" has been written. The trailing
    # final transcript is produced after end of input, so the receive side must
    # outlive the send side; the wait normally ends early when Voxist closes the
    # socket. Kept short because it delays END_OF_SPEECH when a server neither
    # answers nor closes, and well under RECEIVE_TIMEOUT_SECONDS so a genuine
    # stall is still classified by the receive watchdog.
    RESULT_DRAIN_TIMEOUT_SECONDS = 2.0

    # Rate limit for the audio-drop warning. The drop condition persists for
    # the whole overload, and the send loop runs per 10ms frame, so an unlimited
    # warning would emit ~100 lines/second per stream.
    DROP_LOG_INTERVAL_SECONDS = 5.0

    # Cap on unsent audio frames held in the input channel. livekit's channel is
    # unbounded and push_frame() never blocks, so without this a slow uplink
    # grows the backlog until the process is OOM-killed. At 10ms frames this is
    # ~10s of audio; beyond that, transcripts would arrive too late to be
    # useful anyway, so the oldest frames are dropped rather than queued.
    MAX_INPUT_BACKLOG_FRAMES = 1000

    def __init__(
        self,
        *,
        stt: VoxistSTT,
        pool: ConnectionPool,
        config: dict,
        language: str,
        conn_options,
        enable_metrics: bool = True,
    ):
        """
        Initialize streaming recognition session.

        Args:
            stt: Parent VoxistSTT instance
            pool: ConnectionPool for WebSocket management
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

        self._stt = stt
        self._pool = pool
        self._config = config
        self._language = language
        # Language reported on emitted SpeechData. livekit normalizes this to
        # BCP-47, which uppercases the subtag: "fr-medical" is emitted as
        # "fr-MEDICAL". Normalizing once here makes that visible instead of
        # leaving it an implicit side effect inside SpeechData.__post_init__.
        #
        # WARNING: self._language is NOT sent to Voxist. The socket was opened
        # by the pool with the pool-level language in the URL query, and this
        # stream reuses a pooled socket without renegotiating. So a per-stream
        # override - stt.stream(language="en") on a pool built with "fr" - is
        # transcribed by the pool's engine while being labelled with the
        # override here. Fixing that needs a config message on acquire; until
        # then, do not treat this value as the language Voxist actually used.
        self._speech_language = LanguageCode(language)
        self._enable_metrics = enable_metrics

        self._session_id = utils.shortuuid()
        self._speaking = False
        self._conn: Connection | None = None
        # Track if we own exclusive access to connection (VUL-003 mitigation)
        self._owns_connection = False
        # Latch so an unreachable transport is reported once, not per chunk
        self._transport_lookup_failed = False
        # Count of frames dropped to keep the input backlog bounded, and when
        # that was last reported. None means "not yet" - monotonic() has an
        # arbitrary epoch (uptime on Linux), so 0.0 is not a usable sentinel:
        # on a freshly booted host the elapsed check would suppress the first
        # report for as long as the interval.
        self._dropped_frames = 0
        self._last_drop_log: float | None = None

        # Audio processor for format conversion and chunking
        # Resample from input rate (e.g., 48kHz from LiveKit) to 16kHz for Voxist
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

    async def _acquire_connection(self) -> None:
        """
        Acquire connection from pool and mark ownership (HIGH-002 refactor).

        Security: VUL-003 mitigation - marks exclusive ownership to prevent
        race conditions on buffered_amount access.

        Raises:
            ConnectionError: If no connection available
        """
        self._conn = await self._pool.get_connection()
        # SECURITY: Mark exclusive ownership for VUL-003 mitigation
        # Connection is IN_USE state - only this stream should access buffered_amount
        self._owns_connection = True

        logger.debug(
            f"Stream {self._session_id} acquired connection {self._conn.id}"
        )

    async def _release_connection(self) -> None:
        """
        Release connection back to pool (HIGH-002 refactor).

        Safe to call even if no connection is held.
        """
        if self._conn:
            self._owns_connection = False  # Release exclusive ownership
            await self._pool.release_connection(self._conn)
            self._conn = None

    async def _run_stream_tasks(self) -> None:
        """
        Run concurrent send/receive tasks (HIGH-002 refactor).

        Creates and orchestrates the send and receive tasks,
        handling cancellation and exception propagation.

        Raises:
            Exception: Propagates exceptions from failed tasks
        """
        send_task = asyncio.create_task(
            self._send_audio_task(),
            name=f"send-{self._session_id}"
        )
        recv_task = asyncio.create_task(
            self._recv_results_task(),
            name=f"recv-{self._session_id}"
        )

        try:
            # Wait for either task to complete or fail
            done, pending = await asyncio.wait(
                [send_task, recv_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # The send task finishing is not the end of the exchange: it returns
            # as soon as "Done" is written, while the transcript for that audio
            # is still being computed. Cancelling the receive task here would
            # discard it - the final result of every utterance - so give it a
            # bounded window to finish. Voxist closes the socket once it has
            # flushed, which ends the wait immediately in the normal case; the
            # timeout only covers a server that leaves the socket open.
            if (
                recv_task in pending
                and send_task in done
                and not send_task.cancelled()
                and send_task.exception() is None
            ):
                logger.debug(
                    f"Stream {self._session_id} input ended, draining results"
                )
                _, pending = await asyncio.wait(
                    [recv_task], timeout=self.RESULT_DRAIN_TIMEOUT_SECONDS
                )
                if pending:
                    logger.warning(
                        f"Stream {self._session_id} gave up draining results "
                        f"{self.RESULT_DRAIN_TIMEOUT_SECONDS}s after end of input "
                        "- a trailing transcript may have been lost"
                    )
                done = {t for t in (send_task, recv_task) if t not in pending}

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check for exceptions in completed tasks
            for task in done:
                if task.cancelled():
                    continue
                exc = task.exception()
                if exc is not None:
                    raise exc
        except asyncio.CancelledError:
            # Clean up both tasks on cancellation
            for task in [send_task, recv_task]:
                if not task.done():
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass
            raise

    def _calculate_backoff(self, attempt: int) -> float:
        """
        Calculate exponential backoff delay (HIGH-002 refactor).

        Args:
            attempt: Current reconnection attempt number (1-based)

        Returns:
            Backoff delay in seconds (capped at 10s)
        """
        return min(1.0 * (1.5 ** attempt), 10.0)

    async def _run(self) -> None:
        """
        Main processing loop with reconnection logic (HIGH-002 refactored).

        Orchestrates concurrent send/receive tasks and handles connection failures
        with automatic reconnection. Complexity reduced by extracting:
        - _acquire_connection(): Connection pool acquisition
        - _release_connection(): Connection pool release
        - _run_stream_tasks(): Task orchestration
        - _calculate_backoff(): Backoff calculation
        """
        reconnect_attempts = 0
        max_attempts = self._conn_options.max_retry

        logger.debug(f"Stream {self._session_id} starting")

        while reconnect_attempts <= max_attempts:
            try:
                await self._acquire_connection()

                # Reset reconnection counter on successful connection
                reconnect_attempts = 0

                # Run concurrent send/receive tasks
                await self._run_stream_tasks()

                # Normal completion - emit END_OF_SPEECH if we were speaking
                # (Voxist doesn't signal end of utterance, client closes connection)
                if self._speaking:
                    self._speaking = False
                    logger.debug(f"Stream {self._session_id} emitting END_OF_SPEECH on close")
                    self._event_ch.send_nowait(
                        SpeechEvent(
                            type=SpeechEventType.END_OF_SPEECH,
                            request_id=self._session_id,
                        )
                    )

                logger.info(f"Stream {self._session_id} completed normally")
                break

            except Exception as e:
                logger.error(f"Stream {self._session_id} error: {e}")

                if reconnect_attempts >= max_attempts:
                    logger.error(
                        f"Stream {self._session_id} exceeded max reconnect attempts"
                    )
                    raise

                reconnect_attempts += 1
                backoff = self._calculate_backoff(reconnect_attempts)
                logger.info(
                    f"Stream {self._session_id} reconnecting in {backoff:.1f}s "
                    f"(attempt {reconnect_attempts}/{max_attempts})"
                )
                # Hand the connection back before waiting, not in the trailing
                # finally: holding it across the backoff leaves it visible to
                # the pool as reclaimable while this stream still owns it, so
                # another stream can acquire the same socket and our release
                # would then flip it to READY underneath that owner.
                await self._release_connection()
                await asyncio.sleep(backoff)

            finally:
                await self._release_connection()

        logger.info(f"Stream {self._session_id} finished")

    async def _send_audio_task(self) -> None:
        """
        Task for sending audio frames to WebSocket.

        Processes frames through AudioProcessor and sends as binary Float32 data.
        """
        try:
            logger.debug(f"Stream {self._session_id} send task started, waiting for frames...")
            frame_count = 0

            async for data in self._input_ch:
                frame_count += 1
                if frame_count % 100 == 0:
                    logger.debug(f"Stream {self._session_id} processing frame {frame_count}")

                # Check for flush sentinel
                if isinstance(data, self._FlushSentinel):
                    logger.debug(f"Stream {self._session_id} flushing audio")

                    # Flush remaining audio from processor
                    final_chunks = self._audio_processor.flush()
                    for chunk in final_chunks:
                        await self._send_audio_chunk(chunk)

                    # Signal end of stream to Voxist
                    if self._conn and self._conn.ws and not self._conn.ws.closed:
                        await self._send_with_timeout(
                            self._conn.ws.send_str("Done"), "Done signal"
                        )
                        logger.debug(f"Stream {self._session_id} sent Done signal")

                    continue

                # Process audio frame
                if isinstance(data, rtc.AudioFrame):
                    # CRIT-001: bound the unsent backlog. livekit's input channel
                    # is unbounded and push_frame() never blocks, so a slow uplink
                    # would otherwise grow it for the life of the call until the
                    # process is OOM-killed.
                    #
                    # The bound is applied here, on consumption: when the backlog
                    # is over the cap this frame is discarded instead of sent, so
                    # the loop drains the excess at full speed and keeps the most
                    # recent MAX_INPUT_BACKLOG_FRAMES. Discarding the frame in
                    # hand - rather than reaching into the channel to trim it -
                    # is what makes this safe: nothing is removed out of order,
                    # markers below always take the branch above and are honoured
                    # in sequence, and the loop's termination does not depend on
                    # what it finds. An earlier version trimmed the channel
                    # directly and had to re-queue markers, which reordered
                    # utterance boundaries, silently destroyed markers once the
                    # channel was closed, and could spin forever.
                    #
                    # Dropping audio is the right policy for live transcription:
                    # frames this far behind the speaker would produce transcripts
                    # too late to act on.
                    if self._input_ch.qsize() > self.MAX_INPUT_BACKLOG_FRAMES:
                        self._note_dropped_frame()
                        continue

                    # Convert and chunk audio
                    frame_bytes = bytes(data.data)
                    chunks = self._audio_processor.process_audio_frame(frame_bytes)

                    if chunks:
                        logger.debug(
                            f"Stream {self._session_id} got {len(chunks)} chunks"
                        )

                    # Send all chunks
                    for chunk in chunks:
                        await self._send_audio_chunk(chunk)
                else:
                    logger.warning(f"Stream {self._session_id} unexpected data type: {type(data)}")

            logger.debug(f"Stream {self._session_id} send task completed")

        except Exception as e:
            logger.error(f"Stream {self._session_id} send task error: {e}")
            raise

    @property
    def dropped_frames(self) -> int:
        """
        Audio frames discarded to bound the input backlog.

        Non-zero means the transcript for this stream has gaps: the uplink could
        not keep up and audio was deliberately dropped. Exposed so a caller can
        tell a lossy transcript from a complete one - the loss is otherwise only
        visible in the logs, and the emitted events look normal.
        """
        return self._dropped_frames

    def _note_dropped_frame(self) -> None:
        """
        Count a dropped frame and report it, rate-limited.

        The drop condition persists for as long as the uplink is behind, and the
        send loop iterates per 10ms frame, so logging every drop would emit ~100
        lines/second per stream during exactly the overload being reported.
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
        timeout of its own, so an unbounded send blocks the send loop until the
        receive watchdog fires 30s later - during which nothing is consumed from
        the input channel and the backlog grows unchecked. Routing all sends
        through one helper keeps the bound from being applied only to some of
        them.

        Args:
            coro: The send coroutine to await
            what: Short description for the error path (e.g. "audio chunk")

        Raises:
            ConnectionError: If the send stalls past SEND_TIMEOUT_SECONDS.
        """
        try:
            await asyncio.wait_for(coro, timeout=self.SEND_TIMEOUT_SECONDS)
        except asyncio.TimeoutError as e:
            buffer_size = self._get_write_buffer_size()
            logger.warning(
                f"Stream {self._session_id} send stalled for "
                f"{self.SEND_TIMEOUT_SECONDS}s sending {what} "
                f"(write buffer={buffer_size}B); treating the connection as broken"
            )
            await self._abandon_connection()
            raise ConnectionError(
                "WebSocket send timeout - connection stalled"
            ) from e

    async def _abandon_connection(self) -> None:
        """
        Give up on the current connection so the pool recycles it.

        A stalled socket is not a closed socket, so without this the pool would
        hand it straight to the next stream, which would stall in turn. The pool
        owns the state transition and the reconnect - the stream must not write
        ConnectionState itself, or release_connection() would find the
        connection no longer IN_USE and skip its own bookkeeping entirely.

        The transport is aborted because a cancelled send may have left a frame
        mid-flight, so this connection is no longer safe to write to.
        """
        if not self._conn:
            return

        transport = self._get_transport()
        if transport is not None:
            transport.abort()  # immediate, does not attempt to flush

        await self._pool.mark_broken(self._conn)

    def _get_transport(self) -> asyncio.Transport | None:
        """
        Reach the asyncio transport underlying the WebSocket, for diagnostics.

        Prefers a public accessor if the installed aiohttp grows one; otherwise
        walks ws._response.connection, which is not public API. Used only for
        observability - nothing in the send path depends on the result, so a
        None here degrades logging, never behaviour.

        Returns:
            The transport, or None if it cannot be reached or is closing.
        """
        if not self._conn or not self._conn.ws:
            return None

        ws = self._conn.ws
        transport = None

        # Public accessor, if this aiohttp has one.
        getter = getattr(ws, "get_transport", None)
        if callable(getter):
            try:
                transport = getter()
            except (AttributeError, RuntimeError):
                transport = None

        if transport is None:
            # Private fallback. get_extra_info("transport") is not an option:
            # asyncio transports carry no "transport" extra-info key, so it
            # always returns None.
            try:
                connection = ws._response.connection  # type: ignore[attr-defined]
                transport = connection.transport if connection is not None else None
            except (AttributeError, RuntimeError):
                transport = None

        if transport is not None:
            # Guarded: `getter` is any callable named get_transport, so the
            # result is not guaranteed to be a transport. This method is
            # documented as observability-only and must never raise into the
            # send path.
            try:
                if transport.is_closing():
                    return None
                return transport  # type: ignore[no-any-return]
            except (AttributeError, RuntimeError):
                transport = None

        # Log once per stream: a silent permanent failure would hide the loss of
        # every buffer metric behind an aiohttp upgrade.
        if not self._transport_lookup_failed:
            self._transport_lookup_failed = True
            logger.debug(
                f"Stream {self._session_id} cannot reach the WebSocket "
                "transport; write-buffer metrics unavailable for this "
                "stream (aiohttp internals may have changed)"
            )
        return None

    def _get_write_buffer_size(self) -> int:
        """
        Measure the transport write buffer size, for observability only.

        Deliberately NOT used for flow control. Two earlier versions got that
        wrong: first by accumulating half of every chunk ever sent into a
        counter that never decayed, then by comparing the real size against
        thresholds calibrated for a plain socket while production runs TLS.
        Backpressure belongs to aiohttp, which drains inside send_bytes() using
        the transport's own pause state rather than any absolute byte count.

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

        Backpressure is aiohttp's: this await drains while the transport is
        paused, which is correct for both plain and TLS transports because it
        follows the transport's own pause state instead of a fixed threshold.

        CRIT-001: the send is bounded by SEND_TIMEOUT_SECONDS so a stalled
        uplink cannot block the send loop indefinitely; memory is bounded
        separately by the input backlog cap in _send_audio_task.

        Args:
            audio_int16: Int16 NumPy array to send (Voxist expects raw Int16 PCM)

        Raises:
            ConnectionError: If the send stalls past SEND_TIMEOUT_SECONDS.
            OwnershipViolationError: If the stream does not own the connection.
        """
        if not self._conn or not self._conn.ws:
            logger.warning(f"Stream {self._session_id} no connection, skipping chunk")
            return

        if self._conn.ws.closed:
            logger.warning(f"Stream {self._session_id} WebSocket closed, skipping chunk")
            return

        # SECURITY: Validate exclusive ownership before touching shared
        # connection state (VUL-003). Checked before the send so a violation is
        # reported even when the send itself fails.
        if not self._owns_connection:
            raise OwnershipViolationError(
                f"Stream {self._session_id} updating buffered_amount without ownership - "
                "potential race condition. This indicates a bug in stream lifecycle."
            )

        audio_bytes = audio_int16.tobytes()
        await self._send_with_timeout(
            self._conn.ws.send_bytes(audio_bytes), "audio chunk"
        )

        logger.debug(f"Stream {self._session_id} sent {len(audio_bytes)} bytes to WebSocket")

        # Diagnostic snapshot only. The pool no longer selects on this value;
        # it round-robins, because a post-send measurement is ~0 for every
        # healthy connection and cannot distinguish between them.
        self._conn.buffered_amount = self._get_write_buffer_size()

    async def _recv_results_task(self) -> None:
        """
        Task for receiving transcription results from WebSocket.

        Processes JSON messages and emits LiveKit SpeechEvent objects.
        Uses RECEIVE_TIMEOUT_SECONDS to detect stalled connections.
        """
        RECEIVE_TIMEOUT_SECONDS = self.RECEIVE_TIMEOUT_SECONDS

        try:
            logger.debug(f"Stream {self._session_id} receive task started")

            if not self._conn or not self._conn.ws:
                raise ConnectionError("No active connection")

            # Use explicit receive loop with timeout instead of async for
            # This allows us to detect stalled connections
            while not self._conn.ws.closed:
                try:
                    msg = await asyncio.wait_for(
                        self._conn.ws.receive(),
                        timeout=RECEIVE_TIMEOUT_SECONDS
                    )
                except asyncio.TimeoutError as e:
                    logger.warning(
                        f"Stream {self._session_id} receive timeout after "
                        f"{RECEIVE_TIMEOUT_SECONDS}s - connection may be stalled"
                    )
                    # Same treatment as a stalled send: a socket that stops
                    # answering is not a closed socket, so without this the pool
                    # returns it to READY and the next stream inherits the stall.
                    # ws.receive() was also cancelled mid-await, which leaves
                    # aiohttp's reader in an undefined state.
                    await self._abandon_connection()
                    raise ConnectionError(
                        "WebSocket receive timeout - connection stalled"
                    ) from e

                # Any message is proof the socket is alive. The pool's heartbeat
                # loop only refreshes last_heartbeat for READY connections, so
                # without this a long call would hand back a connection with a
                # timestamp older than the staleness threshold and the pool
                # would tear down a perfectly healthy socket.
                if self._conn:
                    self._conn.last_heartbeat = time.time()

                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Parse JSON message
                    logger.debug(
                        f"Stream {self._session_id} received from Voxist: {msg.data[:200]}"
                    )
                    try:
                        data = json.loads(msg.data)
                        await self._process_result(data)
                    except json.JSONDecodeError:
                        # Log details internally (truncated to avoid log bloat)
                        logger.error(
                            f"Stream {self._session_id} invalid JSON received"
                        )
                        logger.debug(
                            f"Stream {self._session_id} invalid JSON: {msg.data[:80]}"
                        )
                        continue

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    # Log full details internally, but sanitize user-facing error
                    logger.error(
                        f"Stream {self._session_id} WebSocket error: {msg.data}"
                    )
                    raise ConnectionError("WebSocket connection error occurred")

                elif msg.type == aiohttp.WSMsgType.CLOSED:
                    logger.debug(f"Stream {self._session_id} WebSocket closed by server")
                    break

                elif msg.type == aiohttp.WSMsgType.CLOSING:
                    logger.debug(f"Stream {self._session_id} WebSocket closing")
                    break

            logger.debug(f"Stream {self._session_id} receive task completed")

        except Exception as e:
            logger.error(f"Stream {self._session_id} receive task error: {e}")
            raise

    async def _process_result(self, data: dict):
        """
        Process transcription result from Voxist and emit appropriate events.

        Voxist Message Format:
            {"type": "partial"|"final", "text": "...", "confidence": 0.95, ...}
            {"status": "connected"}

        Args:
            data: Parsed JSON message from Voxist
        """
        msg_type = data.get("type")
        msg_status = data.get("status")

        # Connection confirmation
        if msg_status == "connected":
            logger.debug(f"Stream {self._session_id} connection confirmed")
            return

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
                # Note: END_OF_SPEECH is NOT emitted here because Voxist sends
                # multiple segments during continuous speech. END_OF_SPEECH is
                # only emitted when the stream is explicitly closed (see _run)

        # Error handling
        elif msg_type == "error":
            error_msg = data.get("message", "Unknown error")
            logger.error(f"Stream {self._session_id} Voxist error: {error_msg}")
            raise Exception(f"Voxist error: {error_msg}")

        # Unknown message type
        elif msg_type:
            logger.warning(f"Stream {self._session_id} unknown message type: {msg_type}")
