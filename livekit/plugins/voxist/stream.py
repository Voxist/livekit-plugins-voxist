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
import json
import sys
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
from .exceptions import TranscriptLostError
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
    # milliseconds - so this is a watchdog, not an expected wait, and it is
    # the ONLY receive-side timeout: during streaming, transport death is
    # detected by aiohttp's heartbeat, so long user silences cannot
    # false-trigger it. 5s because: (a) it must be >= SEND_TIMEOUT_SECONDS -
    # a server that was still ACKing our sends deserves at least as long to
    # flush its finals as a single send was given; (b) the old 2s bound was
    # arguably too tight for a slow final on a loaded engine; (c) every
    # second here is dead air at end of turn when the server has wedged, so
    # a 30s bound (briefly shipped) meant half a minute of silence reported
    # as success.
    SESSION_DRAIN_TIMEOUT_SECONDS = 5.0

    # Silence synthesized at a segment boundary (flush()) to force engine
    # endpointing; see _on_segment_end. Slightly above the engine's ~300ms
    # endpointing threshold.
    SEGMENT_SILENCE_SECONDS = 0.4

    # The wire format is always 16kHz Int16 mono; the AudioProcessor
    # resamples caller audio to this rate.
    WIRE_SAMPLE_RATE = 16000

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

    # Mid-session liveness bound. aiohttp's heartbeat only detects transport
    # death; the gateway's WS layer keeps answering pings even when the
    # engine behind it has wedged, so a mute-but-connected server used to
    # mean silent zero-transcripts until end_input. If real (non-silent)
    # caller audio has been flowing for this long with ZERO WebSocket
    # messages received in that window, the server is declared stalled and
    # the attempt fails as APIConnectionError so the framework redials.
    # Checked inline in the send loop per frame - no watchdog task - and the
    # receive loop resets the window on every message, so a silent USER
    # (no frames pushed, or silence-only frames) can never trip it: the old
    # 30s receive watchdog false-fired on exactly that.
    STALL_DETECTION_SECONDS = 30.0

    # Peak |Int16 amplitude| at or above which a caller frame counts as real
    # audio for stall detection. Pure and near-pure silence (VAD comfort
    # noise, zero-fill) stays below it; speech peaks are orders of magnitude
    # above it. Callers pushing continuous low-level room noise during a
    # long user silence are indistinguishable from speakers to anything but
    # a real VAD, so the threshold is deliberately conservative: missing a
    # whisper only delays detection, while a false positive would sever a
    # healthy session.
    NON_SILENCE_AMPLITUDE = 500

    # Bound on ws.close() during teardown. aiohttp waits up to its ws_close
    # default of 10s for the peer's close ACK - dead air a wedged server
    # does not deserve, right after the 5s drain bound was tightened for the
    # same reason. Failed attempts abort the transport outright; even the
    # polite close on the clean path is bounded by this.
    CLOSE_TIMEOUT_SECONDS = 1.0

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

        # ------------------------------------------------------------------
        # State scope matters here, and getting it wrong has shipped bugs:
        # _run() may execute several times on one stream (livekit's
        # _main_task retries it), so every flag is explicitly either
        #
        # per-SESSION - describes the stream's whole lifetime and is NEVER
        # reset by a retry:
        #   _session_complete  the exchange finished; retries are no-ops
        #   _audio_consumed    real audio left the input channel (on ANY
        #                      attempt) - it can never be replayed
        #   _final_received    at least one FINAL_TRANSCRIPT was emitted
        #   _speaking          START_OF_SPEECH was emitted without its
        #                      matching END_OF_SPEECH yet
        #   _dropped_frames / _last_drop_log   loss accounting for the caller
        #
        # per-ATTEMPT - describes one _run() and is reset at the top of each
        # attempt (a stale True from a failed attempt once disabled both the
        # "server closed before end of input" and the "connection lost
        # before Done" guards on the next attempt):
        #   _done_sent               "Done" was written on THIS socket
        #   _transport_lookup_failed log-once latch for THIS socket
        #   _final_received_this_attempt / _interim_received_this_attempt
        #                            what THIS attempt delivered. The
        #                            post-Done drain taxonomy keys on these:
        #                            a final from attempt 1 must not let
        #                            attempt 2 - whose entire audio produced
        #                            nothing before the server wedged -
        #                            complete as success.
        #   _audio_flowing_since     monotonic time of the first non-silent
        #                            caller frame sent since the LAST
        #                            WebSocket message was received on this
        #                            socket; None while the server is
        #                            responsive (stall detection, see
        #                            STALL_DETECTION_SECONDS)
        # ------------------------------------------------------------------
        self._done_sent = False
        self._session_complete = False
        self._audio_consumed = False
        self._final_received = False
        self._final_received_this_attempt = False
        self._interim_received_this_attempt = False
        self._audio_flowing_since: float | None = None

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
            target_sample_rate=self.WIRE_SAMPLE_RATE,  # Voxist expects 16kHz
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
                stalled send, mid-session server stall, server close before
                end of input). livekit's _main_task catches this, emits a
                recoverable error event, and calls _run() again up to
                conn_options.max_retry.
            TranscriptLostError: The session failed in a way no retry can
                fix - its audio was already consumed and produced no
                transcript. Raised instead of fabricating a clean, empty
                completion. Deliberately NOT an APIError (see the class
                docstring in exceptions.py): _main_task then emits exactly
                one error event (recoverable=False) and terminates the
                stream immediately instead of burning max_retry no-op
                attempts while telling the caller "recoverable" each time.
            AuthenticationError: The key was rejected even with a fresh
                token. Deliberately NOT an APIError: retrying cannot fix a
                revoked key, so it propagates immediately as the true cause.
        """
        if self._session_complete:
            # A previous attempt already finished the exchange; a late error
            # (e.g. during drain) triggered a retry with nothing to recover.
            return

        # Per-ATTEMPT reset (see the scope comment in __init__). A _done_sent
        # left True by a failed attempt would disable both the "server closed
        # before end of input" guard and the "connection lost before Done"
        # guard for this whole attempt.
        self._done_sent = False
        self._transport_lookup_failed = False
        self._final_received_this_attempt = False
        self._interim_received_this_attempt = False
        self._audio_flowing_since = None

        # end_input() is flush() + close(): the LAST channel item is always
        # a _FlushSentinel, so "input fully consumed" must treat a closed
        # channel whose only remaining items are sentinels as consumed. The
        # earlier guard required qsize()==0, and the trailing sentinel
        # bypassed it: after attempt 1 died on the last audio frame, attempt
        # 2 found qsize()==1, dialed, consumed only the sentinel, sent a
        # bare Done and completed as SUCCESS - total transcript loss.
        input_exhausted = self._input_ch.closed and self._pending_input_only_sentinels()

        if input_exhausted and not self._audio_consumed:
            # No real audio ever entered this session (end_input() with zero
            # frames) - there is nothing to transcribe, so there is nothing
            # to dial for. Dialing anyway would ship a bare Done and force
            # the engine to invent a result for zero audio. This only
            # short-circuits when end_input() lands before the attempt
            # starts (or on a retry); a zero-frame session whose end_input()
            # races in after the dial completes normally over the wire.
            logger.debug(
                f"Stream {self._session_id} ended with no audio pushed; "
                "completing without dialing"
            )
            self._finish_session()
            return

        # A retry cannot replay streamed audio. If a previous attempt already
        # consumed the input then dialing a fresh socket would send a bare
        # "Done" and "complete" with whatever the engine makes of zero audio
        # - total transcript loss presented as success. Against _main_task's
        # loop this plays out as:
        #   - no finals ever emitted: the session produced NOTHING, and no
        #     retry can change that -> raise TranscriptLostError, which
        #     _main_task does not retry: one honest error event, immediate
        #     termination. (The guard stays idempotent regardless - any
        #     framework that did call _run() again would land here without
        #     dialing.)
        #   - finals WERE emitted (by an earlier attempt): the data that
        #     could be delivered has been delivered; only audio past the
        #     last final (if any existed) is unrecoverable. Completing
        #     without a pointless re-dial matches the _session_complete
        #     path, with the possible tail loss reported in the log rather
        #     than silently absorbed.
        if input_exhausted:
            if not self._final_received:
                raise TranscriptLostError(
                    "session audio was consumed by a failed attempt and "
                    "cannot be replayed; no transcript was produced"
                )
            logger.warning(
                f"Stream {self._session_id} retry found the input already "
                "consumed by a previous attempt; completing with the finals "
                "already emitted - audio past the last final (if any) was "
                "lost with the failed connection"
            )
            self._finish_session()
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
                        # The taxonomy keys on what THIS attempt delivered
                        # for the audio THIS attempt sent (the flags are
                        # per-attempt): a final from attempt 1 must not let
                        # attempt 2 - whose entire audio produced nothing
                        # before the server wedged - complete as success.
                        if self._final_received_this_attempt:
                            # Finals made it out for this attempt's audio;
                            # only a trailing one can be missing. Completing
                            # with a warning beats failing a session whose
                            # data was delivered.
                            logger.warning(
                                f"Stream {self._session_id} server neither "
                                "closed nor answered within "
                                f"{self.SESSION_DRAIN_TIMEOUT_SECONDS}s of "
                                "Done - a trailing transcript may have been "
                                "lost"
                            )
                        elif self._interim_received_this_attempt:
                            # Interims were delivered but the final never
                            # arrived before the server wedged. Complete
                            # rather than error: the text already reached
                            # the caller as INTERIM_TRANSCRIPT events, the
                            # audio cannot be replayed so no retry can
                            # improve the outcome, and erroring would
                            # vaporize a session whose content was
                            # substantially delivered. Trade-off, stated
                            # plainly: agent code that consumes ONLY
                            # FINAL_TRANSCRIPT events still experiences this
                            # as transcript loss - hence the prominent
                            # warning instead of a silent success.
                            logger.warning(
                                f"Stream {self._session_id} server wedged "
                                "after Done with only interim transcripts "
                                "delivered - completing because the text "
                                "reached the caller as interims, but "
                                "consumers that read only FINAL_TRANSCRIPT "
                                "events will see this session's tail as "
                                "lost"
                            )
                        elif self._audio_consumed:
                            # Real audio was consumed (this or an earlier
                            # attempt), nothing was delivered for it, and it
                            # cannot be replayed: honest, non-retryable
                            # failure (see TranscriptLostError - one error
                            # event, immediate termination, no misleading
                            # "recoverable" retries).
                            raise TranscriptLostError(
                                "server produced no transcript and did not "
                                "close within "
                                f"{self.SESSION_DRAIN_TIMEOUT_SECONDS}s of "
                                "Done; the audio cannot be replayed"
                            ) from None
                        else:
                            # Zero audio ever consumed: nothing was lost. A
                            # fresh dial could legitimately succeed, so let
                            # the framework retry (the retry lands in the
                            # zero-audio short-circuit and completes empty).
                            raise APIConnectionError(
                                "server produced no transcript and did not "
                                "close within "
                                f"{self.SESSION_DRAIN_TIMEOUT_SECONDS}s of "
                                "Done"
                            ) from None

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

            self._finish_session()

        finally:
            try:
                # cancel_and_wait (livekit's own helper) never awaits the
                # children directly - it waits on done-callbacks - so an
                # OUTER cancellation delivered while we sit here propagates
                # out of _run as it must. The previous pattern,
                # `with suppress(CancelledError): await task`, swallowed the
                # outer task's own cancellation whenever aclose() cancelled
                # _main_task while this finally was awaiting a child.
                await utils.aio.cancel_and_wait(send_task, recv_task)
            finally:
                # Retrieve every completed task's exception. When send and
                # recv both fail in the same FIRST_COMPLETED wake, only one
                # is raised; the other would surface at GC as "Task
                # exception was never retrieved", once per retry. The one
                # currently propagating out of the try block is skipped so
                # the primary failure is not double-logged as "secondary".
                in_flight = sys.exc_info()[1]
                for task in (send_task, recv_task):
                    if task.done() and not task.cancelled():
                        exc2 = task.exception()
                        if exc2 is not None and exc2 is not in_flight:
                            logger.debug(
                                f"Stream {self._session_id} secondary task "
                                f"failure in {task.get_name()}: {exc2!r}"
                            )
                if not ws.closed:
                    # ws.close() waits up to aiohttp's ws_close default of
                    # 10s for the peer's close ACK - dead air a wedged or
                    # failing peer must not be granted right after the drain
                    # was bounded to 5s for the same reason. On a failed (or
                    # cancelled) attempt the transport is aborted first, so
                    # the close below returns immediately; the clean path
                    # keeps the polite close but bounds it, aborting as the
                    # fallback.
                    if in_flight is not None:
                        transport = self._get_transport()
                        if transport is not None:
                            transport.abort()
                    try:
                        await asyncio.wait_for(
                            ws.close(), timeout=self.CLOSE_TIMEOUT_SECONDS
                        )
                    except asyncio.CancelledError:
                        raise
                    except Exception:
                        transport = self._get_transport()
                        if transport is not None:
                            transport.abort()
                self._ws = None

    def _finish_session(self) -> None:
        """Mark the exchange finished and emit END_OF_SPEECH if it is owed."""
        self._session_complete = True

        if self._speaking:
            self._speaking = False
            logger.debug(f"Stream {self._session_id} emitting END_OF_SPEECH")
            self._event_ch.send_nowait(
                SpeechEvent(
                    type=SpeechEventType.END_OF_SPEECH,
                    request_id=self._session_id,
                )
            )

        logger.debug(f"Stream {self._session_id} session complete")

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
                # Ship whatever the processor is holding so the engine has
                # the full segment to finalize.
                for chunk in self._audio_processor.flush():
                    await self._send_audio_chunk(chunk)

                # end_input() is flush() + close(), so the LAST item of every
                # session is a sentinel. A sentinel pulled with the channel
                # already closed and nothing behind it is therefore END OF
                # SESSION, not a segment boundary: "Done" (written right
                # after this loop) forces the engine flush on its own, so the
                # endpointing silence would be pure waste - 400ms of extra
                # latency per turn, up to 20s of it on a stalling uplink.
                # A sentinel with the channel still open (or with more items
                # behind it) is a genuine mid-session flush() and keeps the
                # silence injection.
                if self._input_ch.closed and self._input_ch.qsize() == 0:
                    continue
                await self._on_segment_end()
                continue

            if isinstance(data, rtc.AudioFrame):
                # The frame has irrevocably left the channel - whether it is
                # sent or dropped below, a later retry can never replay it.
                # This is what the exhausted-input guard in _run keys on.
                self._audio_consumed = True
                if self._input_ch.qsize() > self.MAX_INPUT_BACKLOG_FRAMES:
                    self._note_dropped_frame()
                    continue

                frame_bytes = bytes(data.data)
                for chunk in self._audio_processor.process_audio_frame(frame_bytes):
                    await self._send_audio_chunk(chunk)
                self._check_server_liveness(frame_bytes)
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
        Flush policy hook: synthesize silence to force engine endpointing.

        The engine finalizes per silence-delimited segment, but it needs
        ~300ms of silence ON THE WIRE to endpoint. A caller that only pushes
        speech frames (VAD-gated capture that stops pushing during silence)
        never puts that silence on the wire, so its flush() would otherwise
        produce no final at all. Sending SEGMENT_SILENCE_SECONDS of zeros
        provides the endpointing trigger without "Done" - the session (and
        socket) stay open for the next segment.

        The zeros are built directly at the 16kHz wire rate and go through
        _send_audio_chunk like any other audio; they deliberately BYPASS the
        AudioProcessor, which would resample them as if they were caller-rate
        input. The whole segment-boundary policy lives in this one hook.

        NOTE: pending live-probe confirmation against the real engine. If the
        probe shows the engine does not finalize on injected silence, this
        hook is where the policy changes - nothing else in the loop assumes
        it.
        """
        # 100ms chunks, matching the normal streaming cadence
        chunk = np.zeros(self.WIRE_SAMPLE_RATE // 10, dtype=np.int16)
        for _ in range(int(self.SEGMENT_SILENCE_SECONDS * 10)):
            await self._send_audio_chunk(chunk)

    def _pending_input_only_sentinels(self) -> bool:
        """
        True if nothing but flush sentinels remains in the input channel.

        Sentinels are segment markers, not data: a closed channel holding
        only sentinels is semantically CONSUMED - draining it can produce no
        audio, only boundaries with nothing between them. end_input() always
        leaves exactly this state behind (flush() then close()), which is why
        the exhausted-input guard cannot key on qsize()==0.

        Reads Chan._queue, the deque backing qsize() - livekit exposes no
        peek. If that private attribute ever moves, fall back to qsize()==0:
        strictly conservative (never claims exhaustion falsely, may miss the
        trailing-sentinel case the tests would then catch).
        """
        queue = getattr(self._input_ch, "_queue", None)
        if queue is None:
            return self._input_ch.qsize() == 0
        return all(isinstance(item, self._FlushSentinel) for item in tuple(queue))

    def _check_server_liveness(self, frame_bytes: bytes) -> None:
        """
        Detect a mute-but-connected server; called per caller frame sent.

        aiohttp's heartbeat only catches transport death - the gateway's WS
        layer answers pings even when the engine behind it is wedged. The
        bound here is send-aware so long user silences cannot false-trigger
        it (the old unconditional 30s receive watchdog did): the clock only
        starts when a NON-SILENT caller frame is sent, and the receive loop
        resets it on every message. Only if real audio has been flowing for
        STALL_DETECTION_SECONDS with zero messages received in that window is
        the server declared stalled.

        Evaluated inline in the send loop - no watchdog task to create,
        cancel, or leak. The worst-case detection delay is one frame period
        past the bound, which is noise against 30s.

        Non-silence is approximated by peak amplitude (NON_SILENCE_AMPLITUDE)
        rather than a real VAD: synthesized zeros and near-zero comfort noise
        do not start the clock, while any plausible speech does. A caller
        pushing continuous above-threshold noise during a genuinely silent
        half-minute is indistinguishable from a speaker here; the engine
        normally answers real audio with partials well inside the bound, so
        the threshold errs toward never severing a healthy session.

        Raises:
            APIConnectionError: No server message for STALL_DETECTION_SECONDS
                of flowing audio; the framework redials.
        """
        samples = np.frombuffer(frame_bytes, dtype=np.int16)
        if samples.size == 0:
            return
        # astype(int32) before abs(): abs(-32768) overflows in int16
        if int(np.abs(samples.astype(np.int32)).max()) < self.NON_SILENCE_AMPLITUDE:
            return

        now = time.monotonic()
        if self._audio_flowing_since is None:
            self._audio_flowing_since = now
            return
        if now - self._audio_flowing_since >= self.STALL_DETECTION_SECONDS:
            logger.warning(
                f"Stream {self._session_id} sent real audio for "
                f"{self.STALL_DETECTION_SECONDS}s without receiving a single "
                "message - the server is connected but not responding; "
                "abandoning the attempt"
            )
            raise APIConnectionError(
                "no response from server while streaming audio"
            )

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
        # TimeoutError MUST be handled before OSError: on Python 3.11+
        # asyncio.TimeoutError is builtins.TimeoutError, an OSError subclass,
        # so the ordering decides which branch a stall takes.
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
        except (aiohttp.ClientError, ConnectionResetError, OSError) as e:
            # A send can also fail outright (socket reset mid-write, transport
            # torn down between the closed-check and the write). Wrap it:
            # livekit retries only APIError, so a raw ConnectionResetError
            # escaping _run would kill the stream with no retry and no error
            # event.
            raise APIConnectionError(f"WebSocket send failed: {e}") from e

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
            # Any message proves the server is alive: reset the stall clock
            # the send loop runs (see _check_server_liveness).
            self._audio_flowing_since = None
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

    @staticmethod
    def _extract_confidence(data: dict) -> float:
        """
        Server-supplied confidence, hardened like "text" already is.

        `data.get("confidence", 1.0)` only covers a missing key; the gateway
        can send `"confidence": null` (or any non-numeric junk), which would
        flow straight into SpeechData. Anything that is not a real number -
        including bool, which is an int subclass - defaults to 1.0. No string
        parsing, no clamping: boring by design.
        """
        value = data.get("confidence")
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return float(value)
        return 1.0

    async def _process_result(self, data: object):
        """
        Process a transcription result from Voxist and emit events.

        Voxist Message Format:
            {"type": "partial"|"final", "text": "...", "confidence": 0.95, ...}

        Defensive by design: the gateway can send frames outside that shape
        (it emits pub/sub redirect frames like {"type": "redirect", ...}, and
        valid JSON need not be an object at all). An unexpected frame must
        never crash the receive loop - non-dict frames are ignored at debug,
        a non-string "text" (e.g. {"text": null}) is treated as absent, and
        unknown "type" values take the warn path below.

        Args:
            data: Parsed JSON message from Voxist (any JSON value)
        """
        if not isinstance(data, dict):
            logger.debug(
                f"Stream {self._session_id} ignoring non-object frame: "
                f"{str(data)[:80]}"
            )
            return

        msg_type = data.get("type")

        # Detect start of speech. "text" is server-supplied: anything that
        # is not a string (absent, null, a number) counts as no text.
        raw_text = data.get("text")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
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

                # Set only when the event actually reaches the caller: the
                # drain taxonomy uses this flag to decide that "the text was
                # delivered as interims", which is false with interim_results
                # disabled.
                self._interim_received_this_attempt = True
                event = SpeechEvent(
                    type=SpeechEventType.INTERIM_TRANSCRIPT,
                    request_id=self._session_id,
                    alternatives=[
                        SpeechData(
                            language=self._speech_language,
                            text=text,
                            confidence=self._extract_confidence(data),
                        )
                    ],
                )
                self._event_ch.send_nowait(event)

        # Final results (confirmed transcription)
        elif msg_type == "final":
            if text:
                logger.info(f"Stream {self._session_id} final: {text[:100]}")

                self._final_received = True
                self._final_received_this_attempt = True
                event = SpeechEvent(
                    type=SpeechEventType.FINAL_TRANSCRIPT,
                    request_id=self._session_id,
                    alternatives=[
                        SpeechData(
                            language=self._speech_language,
                            text=text,
                            confidence=self._extract_confidence(data),
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
