"""VoxistSTTStream - Streaming recognition interface."""

from __future__ import annotations

import asyncio
import json
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

    # Backpressure thresholds, measured against the real asyncio transport
    # write buffer (safety net only). asyncio transports pause writing at their
    # own high-water mark - 64KB by default - and aiohttp drains inside
    # send_bytes() while paused, so that is the primary backpressure mechanism.
    # These marks are a secondary brake for a transport backed up well past it.
    HIGH_WATER_MARK = 256 * 1024  # 256KB - 4x the transport's own pause point
    LOW_WATER_MARK = 64 * 1024    # 64KB - resume once back to normal range
    BACKPRESSURE_CHECK_INTERVAL = 0.05  # 50ms between buffer checks
    # Hard cap on a single backpressure wait. Bounded on purpose: an unbounded
    # wait starves the stream of audio and times out the Voxist WebSocket.
    BACKPRESSURE_MAX_WAIT = 5.0  # seconds

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
        # self._language stays raw - it is what Voxist receives, and the
        # backend routes engines on the exact code.
        self._speech_language = LanguageCode(language)
        self._enable_metrics = enable_metrics

        self._session_id = utils.shortuuid()
        self._speaking = False
        self._conn: Connection | None = None
        # Track if we own exclusive access to connection (VUL-003 mitigation)
        self._owns_connection = False

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

            # Cancel pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check for exceptions in completed tasks
            for task in done:
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
                await asyncio.sleep(backoff)

            finally:
                await self._release_connection()

        logger.info(f"Stream {self._session_id} finished")

    async def _send_config(self) -> None:
        """
        Send initial configuration message to WebSocket.

        Protocol:
            {"config": {"lang": "fr", "sample_rate": 16000}}
        """
        if not self._conn or not self._conn.ws:
            raise ConnectionError("No active connection")

        config_message = {
            "config": {
                "lang": self._language,
                "sample_rate": self._config["sample_rate"],
            }
        }

        await self._conn.ws.send_json(config_message)

        logger.debug(
            f"Stream {self._session_id} sent config: "
            f"lang={self._language}, sample_rate={self._config['sample_rate']}"
        )

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
                        await self._conn.ws.send_str("Done")
                        logger.debug(f"Stream {self._session_id} sent Done signal")

                    continue

                # Process audio frame
                if isinstance(data, rtc.AudioFrame):
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

    def _get_transport(self) -> asyncio.Transport | None:
        """
        Reach the asyncio transport underlying the WebSocket.

        aiohttp exposes no public accessor for this: ClientWebSocketResponse
        has no get_transport(), and get_extra_info("transport") returns None
        because asyncio transports carry no "transport" extra-info key. The
        transport is only reachable through the response's connection, so
        every hop is guarded - this is not public API and may change.

        Returns:
            The transport, or None if it cannot be reached or is closing.
        """
        if not self._conn or not self._conn.ws:
            return None

        try:
            connection = self._conn.ws._response.connection  # type: ignore[attr-defined]
            if connection is None:
                return None
            transport = connection.transport
        except (AttributeError, RuntimeError):
            return None

        if transport is None or transport.is_closing():
            return None
        return transport  # type: ignore[no-any-return]

    def _get_write_buffer_size(self) -> int:
        """
        Measure the real transport write buffer size.

        Returns 0 when the size cannot be measured. Returning 0 rather than a
        guess is deliberate: `await ws.send_bytes()` already applies genuine
        backpressure by draining when the transport pauses, so an unmeasurable
        buffer must never throttle the stream on its own.

        A previous heuristic here accumulated half of every chunk ever sent and
        never decayed, so it crossed HIGH_WATER_MARK after ~100s of audio and
        paused the stream permanently, timing out the Voxist WebSocket. Any
        change here must keep this a *measurement*, never an accumulator.

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

    async def _await_backpressure_relief(self) -> bool:
        """
        Wait for a badly backed-up transport to drain, with a hard time cap.

        This is a secondary emergency brake. The primary backpressure is
        aiohttp's own: it drains inside send_bytes() once the transport pauses
        at its high-water mark (64KB by default). This brake only engages when
        the transport is backed up far beyond that.

        The wait is bounded by BACKPRESSURE_MAX_WAIT so a stream can never be
        paused indefinitely - an unbounded wait here is what silently killed
        streams by starving the WebSocket of audio until it timed out.

        Returns:
            True if sending should proceed, False if the connection was lost.
        """
        buffer_size = self._get_write_buffer_size()
        if buffer_size <= self.HIGH_WATER_MARK:
            return True

        logger.warning(
            f"Stream {self._session_id} backpressure triggered: "
            f"buffer={buffer_size}B > HIGH_WATER_MARK={self.HIGH_WATER_MARK}B"
        )

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.BACKPRESSURE_MAX_WAIT

        while buffer_size > self.LOW_WATER_MARK:
            if loop.time() >= deadline:
                # Give up waiting rather than starve the stream. aiohttp's
                # transport-level flow control remains in force during send.
                logger.warning(
                    f"Stream {self._session_id} backpressure wait exceeded "
                    f"{self.BACKPRESSURE_MAX_WAIT}s (buffer={buffer_size}B); "
                    "resuming send and deferring to transport flow control"
                )
                return True

            await asyncio.sleep(self.BACKPRESSURE_CHECK_INTERVAL)

            # Re-check connection state during wait
            if not self._conn or not self._conn.ws or self._conn.ws.closed:
                logger.warning(
                    f"Stream {self._session_id} connection lost during wait"
                )
                return False

            buffer_size = self._get_write_buffer_size()

        logger.debug(
            f"Stream {self._session_id} backpressure released: "
            f"buffer={buffer_size}B < LOW_WATER_MARK={self.LOW_WATER_MARK}B"
        )
        return True

    async def _send_audio_chunk(self, audio_int16: np.ndarray) -> None:
        """
        Send Int16 PCM audio chunk to WebSocket with backpressure handling.

        CRIT-001 FIX: Implements high/low water mark backpressure pattern to prevent
        buffer overflow and memory exhaustion during audio streaming.

        Flow Control:
            1. Check transport write buffer size before sending
            2. If buffer > HIGH_WATER_MARK, wait until it drains to LOW_WATER_MARK
            3. Send audio chunk as binary frame
            4. Pace at real-time rate to match audio duration

        Args:
            audio_int16: Int16 NumPy array to send (Voxist expects raw Int16 PCM)
        """
        if not self._conn or not self._conn.ws:
            logger.warning(f"Stream {self._session_id} no connection, skipping chunk")
            return

        if self._conn.ws.closed:
            logger.warning(f"Stream {self._session_id} WebSocket closed, skipping chunk")
            return

        # CRIT-001: Backpressure handling using transport write buffer.
        # Time-bounded: never pause a stream indefinitely.
        if not await self._await_backpressure_relief():
            return

        # Send as binary frame (raw Int16 PCM bytes).
        # This await is the primary backpressure point: aiohttp drains here
        # when the transport is paused above its own high-water mark.
        audio_bytes = audio_int16.tobytes()
        await self._conn.ws.send_bytes(audio_bytes)

        logger.debug(f"Stream {self._session_id} sent {len(audio_bytes)} bytes to WebSocket")

        # Record the current buffer level for pool load-balancing, which picks
        # the connection with the smallest buffered_amount. This is a snapshot
        # measurement, not a running total: accumulating here made the value
        # grow without bound and permanently trip backpressure.
        # SECURITY: Validate exclusive ownership to prevent race condition (VUL-003)
        if not self._owns_connection:
            raise OwnershipViolationError(
                f"Stream {self._session_id} updating buffered_amount without ownership - "
                "potential race condition. This indicates a bug in stream lifecycle."
            )
        self._conn.buffered_amount = self._get_write_buffer_size()

    async def _recv_results_task(self) -> None:
        """
        Task for receiving transcription results from WebSocket.

        Processes JSON messages and emits LiveKit SpeechEvent objects.
        Includes 30-second receive timeout to detect stalled connections.
        """
        # Receive timeout to detect stalled connections (network issues, server hang)
        RECEIVE_TIMEOUT_SECONDS = 30.0

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
                    raise ConnectionError(
                        "WebSocket receive timeout - connection stalled"
                    ) from e

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
