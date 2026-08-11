"""VoxistSTTStream - Streaming recognition interface.

One WebSocket per stream, matching the gateway's protocol:

- The socket is dialed by this stream, for this stream, with the language in
  the URL. There is nothing to renegotiate and no window where audio can meet
  the wrong engine.
- flush() marks the end of a SEGMENT. The engine does its own endpointing and
  emits a final per silence-delimited segment on the same socket (the gateway
  threads context across finals - they are the normal flow, not a special
  case). No "Done" is sent at segment boundaries.
- end_input() ends the SESSION: "Done" is written once and the engine
  flushes. What the SERVER does next is engine-specific, and assuming
  otherwise cost 5s of dead air at the end of every turn on the engine the
  platform is migrating to:
    * the gateway only forwards "Done" to the engine and then waits for the
      ENGINE to close ("Forward Done to Banafo and let it close when ready",
      simple-websocket-proxy.gateway.ts:1360). It closes the CLIENT socket
      from its upstream-close handler (gateway.ts:948-973), so the client
      sees a close only if the engine closes.
    * legacy Banafo closes once it has flushed, so the client close does
      arrive and is an expected terminator, not a failure.
    * Kroko does NOT close: it sends its final and holds the socket open
      ("kroko engine keeps the socket open; prod banafo closes it",
      asr-all/kroko/bench/asr_bench.py:79). Staging routes the primary
      languages to Kroko and production Swedish is already Kroko, so this is
      a live path, not a hypothesis.
  The plugin therefore ends the exchange on the ENGINE's own post-"Done"
  transcript plus a short idle period (_await_engine_finalization), and
  treats a socket close as ONE of two acceptable terminators rather than as
  the definition of a healthy ending. An earlier version of this docstring
  asserted the close was THE terminator; every Kroko turn then paid the full
  drain timeout and was reported as an ending "imposed on us".
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
import time
from collections.abc import Awaitable, Iterator
from dataclasses import dataclass
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

from .audio_processor import MAX_FRAME_SIZE_BYTES, AudioProcessor
from .exceptions import ConnectionError as VoxistConnectionError
from .exceptions import TranscriptLostError
from .log import logger

if TYPE_CHECKING:
    from .stt import VoxistSTT


@dataclass(frozen=True)
class _SessionOutcome:
    """
    The complete set of facts the completion gate decides on.

    Three review rounds found the same bug - a clean, successful session
    reported after the whole transcript was lost - because the
    success/failure decision lived in the CALLERS of the completion path,
    one guard per entry point, and every round found another entry point
    with no guard. This record exists so the decision has exactly one home:
    _run states what happened, _finish_session decides what it means.

    Attributes:
        delivered_final: A FINAL_TRANSCRIPT reached the caller for the audio
            this verdict is accountable for.
        delivered_interim: An INTERIM_TRANSCRIPT reached the caller (only
            true when interim_results is enabled - the event must have
            actually been emitted, not merely received).
        unrecoverable_audio: Real caller audio was consumed for this
            verdict's scope. Streamed audio can never be replayed, so if
            nothing was delivered for it, no retry can change the outcome.
        concluded: The exchange ended the way the protocol says it should -
            the engine answered after "Done" and went idle, or the server
            closed the socket after "Done" (which of the two happens is
            engine-specific; see the module docstring) - or there was never
            anything to exchange. False means the ending was imposed on us:
            a drain that timed out with the engine mute, or an earlier
            attempt that died.
        engine_reported_empty: The ASR engine answered for the audio THIS
            verdict is accountable for, and everything it returned was
            EMPTY - at least one transcript frame arrived and not one of them
            carried text. That is the engine saying "I processed this audio
            and found nothing to transcribe", which is a legitimately empty
            transcript rather than a lost one: the difference between a
            participant on an open mic saying nothing recognizable and an
            engine that crashed without answering.

            Deliberately narrower than "a transcript frame arrived". An
            engine that produced TEXT which never reached the caller - a
            partial with interim_results disabled, and then no final - has
            LOST a transcript, not reported an empty one, and crediting it
            here turned that loss into a clean success.

            Defaults to False so any caller that does not state the fact gets
            the conservative reading: no acknowledgement, so an empty result
            is treated as loss.
        detail: Human-readable description of how the attempt ended, used
            verbatim in the log or error message the verdict produces.
    """

    delivered_final: bool
    delivered_interim: bool
    unrecoverable_audio: bool
    concluded: bool
    detail: str
    engine_reported_empty: bool = False


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

    # BACKSTOP bound on the post-"Done" drain: how long to keep receiving
    # from a server that says NOTHING AT ALL after Done. It is not the normal
    # end-of-turn wait - the normal wait ends on the engine's own post-Done
    # transcript (POST_FINAL_IDLE_SECONDS) or on a socket close, whichever
    # the engine does. It is also the ONLY receive-side timeout: during
    # streaming, transport death is detected by aiohttp's heartbeat, so long
    # user silences cannot false-trigger it. 5s because: (a) it must be >=
    # SEND_TIMEOUT_SECONDS - a server that was still ACKing our sends
    # deserves at least as long to flush its finals as a single send was
    # given; (b) the old 2s bound was arguably too tight for a slow final on
    # a loaded engine; (c) every second here is dead air at end of turn when
    # the server has wedged, so a 30s bound (briefly shipped) meant half a
    # minute of silence reported as success.
    SESSION_DRAIN_TIMEOUT_SECONDS = 5.0

    # How long the drain waits after the engine's LAST post-"Done" transcript
    # before calling the exchange finished. This exists because Kroko never
    # closes the socket (see the module docstring): waiting for a close that
    # will never come made every Kroko turn pay SESSION_DRAIN_TIMEOUT_SECONDS
    # of dead air and be classified as an ending imposed on us.
    #
    # Why an idle period at all, rather than "the first post-Done transcript
    # ends it": the engine emits one final per silence-delimited segment, so a
    # single transcript is not proof it has finished flushing - a tail with
    # two segments in it produces two finals back to back, and returning on
    # the first would silently drop the second.
    #
    # Why 0.5s: this is a DECODE-TIME margin, not an endpointing margin. Post
    # "Done" there is no more audio arriving, so the engine's trailing-silence
    # rules cannot fire again and successive finals of one flush are separated
    # only by how long the engine takes to decode them - back-to-back in
    # practice. Do NOT raise this to match an endpointing threshold (the
    # engine is configured with rule1 0.9s / rule2 0.3s trailing silence):
    # those govern when a segment ENDS while audio is still flowing, which is
    # what SEGMENT_SILENCE_SECONDS addresses, and copying 0.9s here would put
    # nearly a second of dead air at the end of every turn for nothing.
    #
    # The idle timer only STARTS once a first post-Done transcript has
    # arrived. A slow first final is therefore governed by
    # SESSION_DRAIN_TIMEOUT_SECONDS, not cut off after 0.5s - which is why
    # this can be tight without truncating a loaded engine.
    #
    # The trade-off is stated plainly because both directions are
    # user-visible: too short drops a second final out of a two-segment flush
    # (a lost sentence), too long is dead air at the end of every single turn
    # in a voice conversation.
    POST_FINAL_IDLE_SECONDS = 0.5

    # Poll period of the drain loop. The drain has to notice two things: the
    # receive task ending (a close) and a transcript arriving. Only the first
    # is awaitable from here - the receive path is deliberately free of drain
    # bookkeeping - so the second is polled off the timestamp
    # _note_engine_progress already maintains. 50ms adds at most 50ms to the
    # end of a turn and costs ~10 wakeups per session.
    DRAIN_POLL_INTERVAL_SECONDS = 0.05

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
    # be useful anyway, so frames are dropped rather than queued.
    #
    # Exceeding the cap is necessary but NOT sufficient to drop: the backlog
    # must also be failing to drain. See _uplink_is_falling_behind for why
    # depth alone silently discarded most of a batch caller's audio.
    MAX_INPUT_BACKLOG_FRAMES = 1000

    # Mid-session liveness bound, measured in AUDIO DELIVERED, not wall
    # time. aiohttp's heartbeat only detects transport death; the gateway's
    # WS layer keeps answering pings even when the engine behind it has
    # wedged, so a mute-but-connected server used to mean silent
    # zero-transcripts until end_input. Two conditions must BOTH hold before
    # the attempt is abandoned (see _check_server_liveness): this many
    # seconds' worth of SPEECH-BEARING audio has been written since the
    # engine last returned a transcript, AND this much wall time has passed
    # since then. Either alone produced false positives that cost far more
    # than the detector saves.
    STALL_DETECTION_SECONDS = 30.0

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
        #   _audio_consumed    audio CARRYING SIGNAL left the input channel
        #                      (on ANY attempt) - it can never be replayed.
        #                      Pure silence does not count: see
        #                      _carries_signal.
        #   _final_received    at least one FINAL_TRANSCRIPT was emitted
        #                      (read through _final_delivered_in_session, so
        #                      the scope is legible where it is USED)
        #   _interim_delivered_before_this_attempt
        #                      at least one INTERIM_TRANSCRIPT was emitted by
        #                      an EARLIER attempt; folded in at the top of
        #                      _run and combined with this attempt's flag by
        #                      _interim_delivered_in_session. It exists
        #                      because there was no session-scoped interim
        #                      flag at all: _outcome's session branch read the
        #                      session flag for finals but the PER-ATTEMPT one
        #                      for interims, and that flag is cleared at the
        #                      top of every _run - so "only interims were
        #                      delivered -> complete" was unreachable on any
        #                      verdict rendered by a retry, and a session
        #                      whose text had already reached the caller as
        #                      interims died with a terminal
        #                      TranscriptLostError instead.
        #   _speaking          START_OF_SPEECH was emitted without its
        #                      matching END_OF_SPEECH yet
        #   _sentinel_probe_failed  log-once latch for a livekit-version
        #                      incompatibility (see
        #                      _pending_input_only_sentinels)
        #   _dropped_frames / _last_drop_log   loss accounting for the caller
        #
        # per-ATTEMPT - describes one _run() and is reset at the top of each
        # attempt (a stale True from a failed attempt once disabled both the
        # "server closed before end of input" and the "connection lost
        # before Done" guards on the next attempt):
        #   _done_sent               "Done" was written on THIS socket
        #   _transport_lookup_failed log-once latch for THIS socket
        #   _final_received_this_attempt / _interim_received_this_attempt
        #                            what THIS attempt delivered.
        #   _real_audio_this_attempt audio carrying signal was consumed by
        #                            THIS attempt. Together with the two
        #                            flags above this decides whether the
        #                            attempt is judged on its own delivery
        #                            or on the session's (see _outcome):
        #                            an attempt that shipped only silence
        #                            owes nothing and can lose nothing.
        #   _bytes_sent_since_progress caller-audio bytes written to THIS
        #                            socket since the ASR engine last
        #                            returned a transcript frame (stall
        #                            detection, see STALL_DETECTION_SECONDS).
        #                            "Progress" is deliberately narrower than
        #                            "any frame": see _note_engine_progress.
        #   _last_progress_at        monotonic time of the last transcript
        #                            frame on THIS socket, for the stall
        #                            report; None until one arrives
        # ------------------------------------------------------------------
        self._done_sent = False
        self._session_complete = False
        self._audio_consumed = False
        self._final_received = False
        self._interim_delivered_before_this_attempt = False
        self._final_received_this_attempt = False
        self._interim_received_this_attempt = False
        self._real_audio_this_attempt = False
        self._bytes_sent_since_progress = 0
        self._last_progress_at: float | None = None
        # Restricted to finals; the post-Done drain's terminator. Separate
        # from _last_progress_at because liveness counts any transcript but
        # finalization only counts a final.
        self._last_final_at: float | None = None
        # Fallback origin for the stall detector's wall-clock floor while no
        # transcript has arrived yet. Reset per attempt, like the pair above.
        self._attempt_started_at = time.monotonic()
        # Last over-cap backlog depth, for the drain-direction test in
        # _uplink_is_falling_behind. None means "not currently over the cap".
        self._backlog_high_water: int | None = None

        # Latch so an unreachable transport is reported once, not per chunk
        self._transport_lookup_failed = False
        # Latch so a livekit-version incompatibility in the input-channel
        # probe is reported once per stream, not once per flush sentinel
        self._sentinel_probe_failed = False
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

        Every exit from an attempt - clean or not - passes through exactly
        one of two structures: the completion gate (_finish_session, which
        owns the success/failure verdict) or an exception. Nothing outside
        the gate decides "this session succeeded". Three review rounds each
        found a NEW branch that reached the completion path without checking
        whether anything had actually been transcribed, so the check moved
        into the path itself.

        Raises:
            APIConnectionError: The session was interrupted (dial failure,
                stalled send, mid-session server stall, server close before
                end of input), or the exchange never concluded on a session
                that lost nothing (a retry may still succeed). livekit's
                _main_task catches this, emits a recoverable error event, and
                calls _run() again up to conn_options.max_retry.
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

        # Fold what the attempt that just ended DELIVERED into the session
        # ledger before its per-attempt flag is cleared below. Finals need no
        # fold - _process_result sets a session flag for them directly - but
        # interims had no session-scoped flag at all, so every retry erased
        # the fact that text had already reached the caller as interims.
        self._interim_delivered_before_this_attempt = (
            self._interim_delivered_in_session
        )

        # Per-ATTEMPT reset (see the scope comment in __init__). A _done_sent
        # left True by a failed attempt would disable both the "server closed
        # before end of input" guard and the "connection lost before Done"
        # guard for this whole attempt.
        self._done_sent = False
        self._transport_lookup_failed = False
        self._final_received_this_attempt = False
        self._interim_received_this_attempt = False
        self._real_audio_this_attempt = False
        self._bytes_sent_since_progress = 0
        self._last_progress_at = None
        self._last_final_at = None
        self._attempt_started_at = time.monotonic()

        try:
            await self._run_attempt()
        finally:
            # Structural backstop for END_OF_SPEECH. The gate emits it on
            # every verdict it renders, but an attempt can also end by
            # raising before it ever reaches the gate (dial failure on a
            # retry, a stalled send, a mid-input server close, an outer
            # cancellation). If the stream then dies - retries exhausted, or
            # a non-retryable error - a START_OF_SPEECH already delivered to
            # the caller would never be matched, and downstream turn logic
            # would hold that turn open forever. Idempotent, so the gate
            # having already emitted it costs nothing.
            self._terminate_speaking()

    async def _run_attempt(self) -> None:
        """
        One dial-and-exchange, with the verdict delegated to the gate.

        Invariant, and the reason this method reads the way it does: every
        `return` here is immediately preceded by a _finish_session call, and
        the last statement of the try block is a _finish_session call. There
        is no route from here to "the session ended" that does not pass
        through the gate, so no branch - including one added later - can
        report success without the gate weighing what was actually
        delivered.
        """
        # end_input() is flush() + close(): the LAST channel item is always
        # a _FlushSentinel, so "input fully consumed" must treat a closed
        # channel whose only remaining items are sentinels as consumed. The
        # earlier guard required qsize()==0, and the trailing sentinel
        # bypassed it: after attempt 1 died on the last audio frame, attempt
        # 2 found qsize()==1, dialed, consumed only the sentinel, sent a
        # bare Done and completed as SUCCESS - total transcript loss.
        #
        # `is True` is load-bearing: the probe is TRI-STATE and None means "I
        # could not look" (livekit renamed Chan._queue). Both shortcuts below
        # skip the exchange entirely and decide the session's fate from
        # inference alone, so NEITHER may be taken on an unverified probe:
        #   - the empty-session shortcut declared a channel that still held
        #     every audio frame of the session "nothing to transcribe" and
        #     completed as a clean SUCCESS with zero events - the exact
        #     total-loss-reported-as-success this design exists to prevent;
        #   - the already-consumed shortcut fabricated a TranscriptLostError
        #     for a session it never attempted, abandoning any audio still
        #     queued behind the unreadable probe.
        # Dialing instead costs one socket (and possibly a zero-duration
        # billing event); it cannot lose audio and cannot invent an outcome,
        # because whatever happens on that socket is judged by the gate.
        input_exhausted = (
            self._input_ch.closed
            and self._probe_pending_input_only_sentinels() is True
        )

        if input_exhausted and not self._audio_consumed:
            # No audio carrying signal ever entered this session (end_input()
            # with zero frames, or with nothing but silence) - there is
            # nothing to transcribe, so there is nothing to dial for. Dialing
            # anyway would ship a bare Done and force the engine to invent a
            # result for zero audio. This only short-circuits when end_input()
            # lands before the attempt starts (or on a retry); a zero-frame
            # session whose end_input() races in after the dial completes
            # normally over the wire.
            logger.debug(
                f"Stream {self._session_id} ended with no audio to "
                "transcribe; completing without dialing"
            )
            self._finish_session(
                self._outcome(
                    concluded=True,
                    detail="the session carried no audio to transcribe",
                )
            )
            return

        # A retry cannot replay streamed audio. If a previous attempt already
        # consumed the input then dialing a fresh socket would send a bare
        # "Done" and "complete" with whatever the engine makes of zero audio
        # - total transcript loss presented as success. Whether that is a
        # completion (the finals already made it out, and only audio past the
        # last one is unrecoverable) or an honest failure (nothing made it
        # out at all) is NOT decided here: the gate decides it from the same
        # facts every other exit is judged on.
        if input_exhausted:
            logger.debug(
                f"Stream {self._session_id} retry found the input already "
                "consumed by a previous attempt; completing without dialing"
            )
            self._finish_session(
                self._outcome(
                    concluded=False,
                    detail=(
                        "the session's audio was consumed by an attempt that "
                        "failed before delivering it, and streamed audio "
                        "cannot be replayed"
                    ),
                )
            )
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

        # Captured in the except clause below rather than read back out of
        # sys.exc_info() in the finally: exc_info() is frame-local exception
        # state and the teardown awaits, so relying on it to identify "the
        # exception currently propagating" is fragile by construction even
        # where it happens to work.
        in_flight: BaseException | None = None

        try:
            done, pending = await asyncio.wait(
                {send_task, recv_task}, return_when=asyncio.FIRST_COMPLETED
            )

            if send_task in done:
                exc = send_task.exception()
                if exc is not None:
                    raise exc

                # Input delivered and "Done" written. The trailing finals are
                # still being computed. How the exchange ends depends on the
                # engine - Banafo closes the socket, Kroko answers and holds
                # it open - so the wait is delegated to the drain, which
                # accepts either terminator (see _await_engine_finalization).
                if recv_task in pending:
                    drained = await self._await_engine_finalization(recv_task)
                    if drained is not None:
                        # The engine had its say and went idle, or it stayed
                        # mute past the backstop. Either way the drain
                        # reports FACTS, not a verdict: for one round the
                        # timeout was the only route into the outcome
                        # taxonomy, so a server that closed PROMPTLY after
                        # Done without ever sending a transcript skipped the
                        # whole decision and the session was reported as a
                        # clean success with zero transcripts.
                        self._finish_session(drained)
                        return
                    # The server closed the socket: fall through to the
                    # shared end-of-exchange handling below, which still
                    # inspects the receive task's exception first.

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

            # The socket closed after "Done": one of the two acceptable
            # terminators (the Banafo-style one - Kroko never closes, and its
            # sessions end in the drain above). That still says NOTHING about
            # whether a transcript was produced - a gateway whose engine
            # crashed closes just as promptly, and an aiohttp heartbeat death
            # is indistinguishable from here (the receive iterator simply
            # ends, with no exception). The gate decides, weighing whether the
            # engine ever answered at all.
            self._finish_session(
                self._outcome(
                    concluded=True,
                    detail=(
                        "the server closed the socket after Done without "
                        "sending a transcript"
                    ),
                )
            )

        except BaseException as exc:
            in_flight = exc
            raise

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

    async def _await_engine_finalization(
        self, recv_task: asyncio.Task
    ) -> _SessionOutcome | None:
        """
        Wait out the post-"Done" exchange, accepting either terminator.

        Returns None when the SERVER closed the socket - the caller falls
        through to its shared end-of-exchange handling, which must still
        inspect the receive task's exception. Otherwise returns the outcome
        for the gate to judge.

        Two terminators, because the engines differ and only one of them ever
        closes (see the module docstring):

        1. the socket closes - legacy Banafo, and the only ending the plugin
           used to recognise;
        2. the engine returns a transcript after "Done" and then goes quiet
           for POST_FINAL_IDLE_SECONDS - Kroko, which never closes. Treating
           this as an interruption (which is what waiting for a close that
           never comes amounts to) put SESSION_DRAIN_TIMEOUT_SECONDS of dead
           air at the end of EVERY Kroko turn and then reported the healthy
           session as "the ending was imposed on us", warning about a
           trailing transcript that was never missing. The API's own
           reference client converges the two engines the same way
           (asr-all/kroko/bench/asr_bench.py:79).

        The idle period, not the first frame, is what ends case 2: the engine
        emits one final per silence-delimited segment, so a tail holding two
        segments produces two finals and returning on the first would drop the
        second silently.

        A transcript that arrived BEFORE "Done" does not end the drain: it
        says the engine was working earlier, not that it has finished
        flushing. That is why the pre-Done state is snapshotted here rather
        than tested for non-emptiness. Ending the drain on a pre-Done
        transcript would also break the mute-server case that matters: a
        server which wedges the moment it is asked to finalize would pass as a
        clean ending, and a session whose last segment never came would be
        reported without the "trailing transcript may be missing" warning.

        The snapshot cannot miss the engine's answer even though it is taken
        after the write rather than at it: the caller resumes from
        asyncio.wait the moment the send task completes, a same-loop callback,
        while a response CAUSED by "Done" has to traverse the socket and be
        dispatched by the selector - never in the same batch of ready
        callbacks. And if it were ever missed, the drain would fall back to
        the backstop, which is the behaviour that preceded this method.

        Both facts the loop needs are read, not awaited: the receive task's
        completion is awaitable, but a transcript's arrival is not - the
        receive path carries no drain bookkeeping - so the monotonic
        timestamp _note_engine_progress maintains is polled at
        DRAIN_POLL_INTERVAL_SECONDS. That timestamp is the ONE fact this
        method borrows from the liveness detector, and it is the same fact:
        "the ASR engine returned a transcript frame at time T, on this
        socket".
        """
        # A FINAL, not any transcript: a post-Done partial proves the engine
        # is alive but not that it has finished, and returning on one would
        # truncate the final still in flight.
        transcript_before_done = self._last_final_at
        deadline = time.monotonic() + self.SESSION_DRAIN_TIMEOUT_SECONDS

        while True:
            now = time.monotonic()
            # Read once per pass: the receive task can update it between the
            # two tests below, and None means "no transcript on this socket
            # yet" (the per-attempt reset in _run).
            last_transcript_at = self._last_final_at
            answered_after_done = (
                last_transcript_at is not None
                and last_transcript_at != transcript_before_done
            )
            if answered_after_done:
                assert last_transcript_at is not None  # implied; for the type
                if now - last_transcript_at >= self.POST_FINAL_IDLE_SECONDS:
                    return self._outcome(
                        concluded=True,
                        detail=(
                            "the engine returned its result after Done and "
                            "then stayed quiet for "
                            f"{self.POST_FINAL_IDLE_SECONDS}s without closing "
                            "the socket"
                        ),
                    )
            if now >= deadline:
                return self._outcome(
                    concluded=False,
                    detail=(
                        "the server sent no transcript after Done and did "
                        "not close the socket within "
                        f"{self.SESSION_DRAIN_TIMEOUT_SECONDS}s"
                    ),
                )

            done, _ = await asyncio.wait(
                {recv_task},
                timeout=min(self.DRAIN_POLL_INTERVAL_SECONDS, deadline - now),
            )
            if done:
                return None

    @property
    def _final_delivered_in_session(self) -> bool:
        """
        Session-scoped: a FINAL_TRANSCRIPT reached the caller on ANY attempt.

        An alias, deliberately: the flag itself is set in _process_result,
        whose name (_final_received) does not state its scope. Every read that
        MEANS "the session" goes through this name instead, because the defect
        this pair of accessors exists to prevent was a use site that read one
        scope while looking like it read the other.
        """
        return self._final_received

    @property
    def _interim_delivered_in_session(self) -> bool:
        """
        Session-scoped: an INTERIM_TRANSCRIPT reached the caller on ANY
        attempt, this one included.

        _process_result only maintains a per-attempt flag, and _run clears it
        at the top of every attempt, so the session-scoped fact is the fold of
        the earlier attempts (_interim_delivered_before_this_attempt, updated
        in _run before the reset) with the current one.
        """
        return (
            self._interim_delivered_before_this_attempt
            or self._interim_received_this_attempt
        )

    @property
    def _engine_reported_empty_this_attempt(self) -> bool:
        """
        Whether the engine answered on THIS socket and every transcript frame
        it sent was empty.

        An engine that emits {"type": "final", "text": ""} has told us it
        processed the audio and found nothing to transcribe. That is the
        difference between a participant sitting on an open mic (room noise,
        coughing, another language - all non-zero samples, so all "real audio"
        by _carries_signal) and an engine that crashed or a heartbeat death,
        which return NOTHING at all. Both used to end in the same terminal
        TranscriptLostError, which made a fatal error the DEFAULT outcome for
        a participant who simply never said anything.

        Two facts, neither of which this lane may record itself, so both are
        read off state other paths already maintain:

        - "the engine answered": _note_engine_progress's timestamp, set for
          exactly the frames this asks about ("partial"/"final", any text)
          and cleared per attempt in _run.
        - "nothing it sent carried text": _speaking, which _process_result
          latches on the first transcript frame carrying text and which _run's
          finally clears at the end of every attempt - so at verdict time
          (the gate reads the outcome before its own _terminate_speaking runs)
          it is exactly "this attempt's engine produced text". Text the engine
          produced but the plugin did not deliver is transcript LOSS, so it
          must not be credited as an empty report.
        """
        return self._last_progress_at is not None and not self._speaking

    def _outcome(self, *, concluded: bool, detail: str) -> _SessionOutcome:
        """
        Snapshot the facts the completion gate is allowed to decide on.

        Scoping is the whole subtlety here, and getting it wrong has shipped
        bugs in BOTH directions:

        - An attempt that consumed audio carrying signal is accountable for
          THAT audio and must be judged on what IT delivered. Keying on the
          session-scoped final flag once let a retry which lost an entire
          turn complete as success, because attempt 1 had delivered a final
          for an earlier turn.
        - An attempt that consumed no such audio - only silence, or only the
          trailing flush sentinel end_input() leaves behind - owes nothing
          and can lose nothing, so it is judged on the SESSION. Judging it on
          its own (necessarily empty) per-attempt flags killed a healthy
          session with a terminal TranscriptLostError when a connection blip
          made the retry ship nothing but the caller's trailing captured
          silence, after every final had already been delivered.

        Whichever branch applies, both delivery flags are read at the SAME
        scope. They were not: the session branch read the session flag for
        finals but the per-attempt flag for interims, and since that flag is
        cleared at the top of every _run, "only interims were delivered ->
        complete" could not be reached by any verdict a retry rendered. A
        session that had already shipped its text as interims, then hit a
        connection reset during the post-Done drain, came back on attempt 2
        with both delivery flags reading False and died with a terminal
        TranscriptLostError.

        Args:
            concluded: Whether the exchange ended on the protocol's terms.
            detail: How the attempt ended, for the resulting log or error.
        """
        if self._real_audio_this_attempt:
            return _SessionOutcome(
                delivered_final=self._final_received_this_attempt,
                delivered_interim=self._interim_received_this_attempt,
                unrecoverable_audio=True,
                concluded=concluded,
                detail=detail,
                engine_reported_empty=self._engine_reported_empty_this_attempt,
            )
        return _SessionOutcome(
            delivered_final=self._final_delivered_in_session,
            delivered_interim=self._interim_delivered_in_session,
            unrecoverable_audio=self._audio_consumed,
            concluded=concluded,
            detail=detail,
            # Deliberately NOT this attempt's engine acknowledgement. In this
            # branch the attempt shipped no audio carrying signal, so any
            # unrecoverable audio belongs to an EARLIER attempt whose socket is
            # gone - and this engine, which only ever saw silence, cannot
            # vouch for audio it never received. Claiming otherwise would let
            # an empty final for three frames of captured silence certify the
            # turn a dead attempt swallowed as "legitimately empty".
            engine_reported_empty=False,
        )

    def _finish_session(self, outcome: _SessionOutcome) -> None:
        """
        The single exit gate: decide what the outcome MEANS, then record it.

        This is the only place in the plugin that may conclude a session
        succeeded, and the only place that raises the terminal
        TranscriptLostError. It refuses to report success when the audio it
        is accountable for produced nothing, whatever route reached it - a
        drain that timed out, a server that closed instantly, or an input
        already consumed by a dead attempt. The taxonomy is expressed once,
        keyed only on the facts in ``outcome``:

          1. a final was delivered              -> success (warn if the
                                                   ending was not the
                                                   protocol's, so a trailing
                                                   final may be missing)
          2. only interims were delivered       -> success, loudly
          3. nothing delivered, the exchange
             concluded AND the engine answered
             for this audio with nothing but
             empty transcripts                  -> success, empty: the engine
                                                   processed the audio and
                                                   found nothing to
                                                   transcribe
          4. nothing delivered, real audio
             consumed and unreplayable          -> TranscriptLostError
          5. nothing delivered, nothing lost,
             exchange concluded                 -> success, empty
          6. nothing delivered, nothing lost,
             exchange never concluded           -> APIConnectionError: a
                                                   fresh dial may still work

        Case 3 is the one that took the longest to get right, and it is
        ordered before case 4 on purpose. "The engine produced no non-empty
        transcript" was treated as "the transcript was lost", so a session
        that behaved EXACTLY as the protocol prescribes - participant joins
        on an open mic and says nothing recognizable (room noise, coughing,
        a language the model does not serve), engine finalizes with
        {"type": "final", "text": ""}, exchange ends normally - died on a
        terminal TranscriptLostError. A real microphone never produces pure
        zeros, so _carries_signal counts that noise as real audio, which made
        it the DEFAULT outcome for a silent participant rather than an edge
        case. What separates it from genuine loss is not the audio and not
        the socket, but what the ENGINE said: no transcript frame at all, on
        an exchange that consumed real audio, is still case 4 - which is what
        keeps a crashed engine and a heartbeat death honest - and so is a
        frame that carried TEXT the caller never received.

        KNOWN LIMITATION, stated rather than half-fixed: case 1 asks whether
        ANY final was delivered, not whether the LAST one was. A session that
        delivered segment 1's final and lost segment 4's is a success here, and
        no fact available to this gate can tell the two apart - the protocol
        carries no segment count, no sequence number and no end-of-stream
        marker, and the engine the platform is moving to does not even close
        the socket. What IS detectable is covered: an exchange that ended
        without the engine answering "Done" has concluded=False, so case 1
        warns that a trailing transcript may be missing. Closing the gap for
        real needs new information on the wire (a final count, or an explicit
        end-of-stream frame), not a cleverer reading of what we have.

        Raises:
            TranscriptLostError: Case 4. Non-retryable by construction.
            APIConnectionError: Case 6. Retried by the framework.
        """
        try:
            if outcome.delivered_final:
                if not outcome.concluded:
                    logger.warning(
                        f"Stream {self._session_id} completing on the finals "
                        f"already delivered: {outcome.detail} - a trailing "
                        "transcript may have been lost"
                    )
            elif outcome.delivered_interim:
                # Complete rather than error: the text already reached the
                # caller as INTERIM_TRANSCRIPT events, the audio cannot be
                # replayed so no retry can improve the outcome, and erroring
                # would vaporize a session whose content was substantially
                # delivered. Trade-off, stated plainly: agent code that
                # consumes ONLY FINAL_TRANSCRIPT events still experiences
                # this as transcript loss - hence the prominent warning
                # instead of a silent success.
                logger.warning(
                    f"Stream {self._session_id} delivered only interim "
                    f"transcripts: {outcome.detail} - completing because the "
                    "text reached the caller as interims, but consumers that "
                    "read only FINAL_TRANSCRIPT events will see this "
                    "session's tail as lost"
                )
            elif outcome.concluded and outcome.engine_reported_empty:
                # The engine answered for this audio and its answer was
                # "nothing". An empty result is a legitimate result: it is
                # what a participant who never said anything recognizable
                # produces, and the plugin must not turn that into a fatal
                # error. INFO rather than debug because a caller staring at a
                # session with no transcript deserves to find out from the
                # logs that the engine, not the plugin, decided it was empty.
                logger.info(
                    f"Stream {self._session_id} produced no transcript: "
                    f"{outcome.detail} - the engine processed the audio and "
                    "returned an empty result, so this is an empty session, "
                    "not a lost one"
                )
            elif outcome.unrecoverable_audio:
                # Real audio was consumed, nothing was delivered for it, and
                # it cannot be replayed: honest, non-retryable failure (see
                # TranscriptLostError - one error event, immediate
                # termination, no misleading "recoverable" retries).
                #
                # Reached only when the exchange was interrupted, or the engine
                # never answered at all, or it produced text that never
                # reached the caller - the case above has already taken the
                # sessions whose engine answered "nothing to transcribe".
                # That is what keeps this raise meaningful instead of firing
                # on every silent participant: a crashed engine and a
                # heartbeat death both return NOTHING, and both still land
                # here.
                raise TranscriptLostError(
                    "no transcript was produced for audio that cannot be "
                    f"replayed: {outcome.detail}"
                )
            elif not outcome.concluded:
                # Nothing was lost, but the exchange never finished on the
                # protocol's terms. A fresh dial could legitimately succeed,
                # so let the framework retry (a retry with nothing left to
                # send lands in the no-audio short-circuit and completes
                # empty).
                raise APIConnectionError(outcome.detail)
            else:
                logger.debug(
                    f"Stream {self._session_id} completed with nothing to "
                    f"transcribe: {outcome.detail}"
                )

            self._session_complete = True
            logger.debug(f"Stream {self._session_id} session complete")
        finally:
            self._terminate_speaking()

    def _terminate_speaking(self) -> None:
        """
        Emit the END_OF_SPEECH a START_OF_SPEECH is owed, if any.

        Called from the gate's ``finally`` and again from _run's, so no exit
        - success, warn-complete, terminal loss, retryable interruption or
        cancellation - can leave the caller holding an unmatched
        START_OF_SPEECH. Downstream turn logic pairs the two events, so an
        unmatched START_OF_SPEECH strands a turn open indefinitely; for one
        round the TranscriptLostError raise sites bypassed the completion
        path entirely and did exactly that.

        Emitting it BEFORE an exception propagates is deliberate: the error
        event livekit emits afterwards then arrives with the speech state
        already closed. Idempotent - _speaking is the latch.
        """
        if not self._speaking:
            return

        self._speaking = False
        logger.debug(f"Stream {self._session_id} emitting END_OF_SPEECH")
        self._event_ch.send_nowait(
            SpeechEvent(
                type=SpeechEventType.END_OF_SPEECH,
                request_id=self._session_id,
            )
        )

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
                #
                # The predicate is _pending_input_only_sentinels, the SAME one
                # the exhausted-input guard in _run uses - not a second
                # expression that means roughly the same thing. This check was
                # `qsize() == 0` for one round, which made the commonest VAD
                # pattern of all - flush() immediately followed by
                # end_input(), i.e. two ADJACENT sentinels - see qsize()==1
                # on the first sentinel and inject 400ms of endpointing
                # silence before a "Done" that forces the engine flush
                # anyway: pure dead air at the end of every turn.
                if self._input_ch.closed and self._pending_input_only_sentinels():
                    continue
                await self._on_segment_end()
                continue

            if isinstance(data, rtc.AudioFrame):
                # The frame has irrevocably left the channel - whether it is
                # sent or dropped below, a later retry can never replay it -
                # so both consumption flags are set BEFORE the drop check.
                # Only frames carrying signal count: a frame of pure silence
                # could not have produced a transcript, so its loss is not
                # transcript loss. Session flag for the top-of-run guard,
                # per-attempt flag for the outcome scoping in _outcome.
                if self._carries_signal(data.data):
                    self._audio_consumed = True
                    self._real_audio_this_attempt = True
                if self._uplink_is_falling_behind():
                    self._note_dropped_frame()
                    continue

                frame_bytes = bytes(data.data)
                # Sliced, not rejected. The processor raises a bare ValueError
                # above MAX_FRAME_SIZE_BYTES (a resource-exhaustion guard), and
                # that ValueError is not an APIError, so it escaped the whole
                # error taxonomy: it unwound _run without the completion gate
                # ever running, leaving _session_complete False and killing the
                # stream with a bare ValueError at the call site. Legitimate
                # callers hit it - a batch caller pushing ~11s of 48kHz mono,
                # or livekit's own AudioResampler emitting a large frame.
                #
                # Slicing satisfies what the guard is actually for (bounded
                # allocation per call) without discarding audio: the processor
                # buffers across calls, so N sequential slices yield the same
                # chunk stream as one oversized call would have. Slices are
                # sample-aligned because the limit is even and the processor
                # reads mono Int16.
                for piece in self._iter_frame_slices(frame_bytes):
                    for chunk in self._audio_processor.process_audio_frame(piece):
                        await self._send_audio_chunk(chunk)
                self._check_server_liveness()
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

        They are also excluded from the liveness byte count: this is OUR
        audio, not the caller's, and injecting it must never move a detector
        whose whole question is "how much of the caller's audio has the
        server swallowed in silence".

        NOTE: pending live-probe confirmation against the real engine. If the
        probe shows the engine does not finalize on injected silence, this
        hook is where the policy changes - nothing else in the loop assumes
        it.
        """
        # 100ms chunks, matching the normal streaming cadence
        chunk = np.zeros(self.WIRE_SAMPLE_RATE // 10, dtype=np.int16)
        for _ in range(int(self.SEGMENT_SILENCE_SECONDS * 10)):
            await self._send_audio_chunk(chunk, caller_audio=False)

    def _probe_pending_input_only_sentinels(self) -> bool | None:
        """
        Whether nothing but flush sentinels remains in the input channel:
        True, False, or None for "could not look".

        THE single probe for "no audio remains in the input channel", used
        (directly or through _pending_input_only_sentinels) by both places
        that need that fact:

        - the exhausted-input guards in _run_attempt: sentinels are segment
          markers, not data, so a closed channel holding only sentinels is
          semantically CONSUMED - draining it can produce no audio, only
          boundaries with nothing between them. end_input() always leaves
          exactly this state behind (flush() then close()), which is why the
          guards cannot key on qsize()==0;
        - the send loop's end-of-SESSION detection: a sentinel with only
          sentinels behind it on a closed channel is the one end_input()
          queued, not a segment boundary.

        These were two different expressions for one round - the guard used
        this helper while the send loop still tested qsize()==0 - and the
        divergence cost 400ms of dead air on every flush()+end_input() turn.
        One probe, one meaning, both callers.

        Reads Chan._queue, the deque backing qsize() - livekit exposes no
        peek. If that private attribute ever moves, this reports None rather
        than guessing, because the safe answer is NOT the same for both
        callers and collapsing it to one bool got that wrong in both
        directions at once:

        - answering True let the empty-session shortcut declare a channel
          still holding every frame of the session "nothing to transcribe"
          and complete as a clean SUCCESS with zero events - the total
          transcript loss the shortcut exists to prevent, delivered by the
          fail-safe itself. It also fabricated a TranscriptLostError for a
          consumed-input retry that was never attempted.
        - answering False would restore exactly the pre-fix qsize()==0
          behaviour in the SEND LOOP: a boundary that never happened, 400ms
          of dead air per turn.

        So the callers decide: the guards refuse to shortcut on None and dial
        instead (one socket, no inference), while
        _pending_input_only_sentinels keeps answering True for the send loop,
        where the cost of being wrong is a merged segment.
        """
        queue = getattr(self._input_ch, "_queue", None)
        if queue is None:
            if not self._sentinel_probe_failed:
                self._sentinel_probe_failed = True
                logger.warning(
                    f"Stream {self._session_id} cannot inspect the input "
                    "channel's backing queue: livekit's Chan._queue has "
                    "moved, so this livekit-agents version is not compatible "
                    "with the plugin's input-exhaustion check. The session "
                    "will be run over the wire instead of short-circuited, "
                    "which can cost a segment boundary or a socket but can "
                    "never lose audio the channel still holds."
                )
            return None
        return all(isinstance(item, self._FlushSentinel) for item in tuple(queue))

    @staticmethod
    def _iter_frame_slices(frame_bytes: bytes) -> Iterator[bytes]:
        """
        Yield frame_bytes in pieces the audio processor will accept.

        Yields the frame unchanged in the overwhelmingly common case, so the
        hot path pays one comparison. See the call site for why an oversized
        frame is sliced rather than rejected.
        """
        limit = MAX_FRAME_SIZE_BYTES
        if len(frame_bytes) <= limit:
            yield frame_bytes
            return
        logger.warning(
            f"Frame of {len(frame_bytes)}B exceeds the {limit}B processor "
            "limit; sending it in slices rather than dropping it"
        )
        for start in range(0, len(frame_bytes), limit):
            yield frame_bytes[start : start + limit]

    def _uplink_is_falling_behind(self) -> bool:
        """
        True when the backlog is over the cap AND is not draining.

        The cap exists to stop a slow uplink growing the backlog until the
        process is OOM-killed. It used to fire on backlog DEPTH alone, which
        confused a bursty PRODUCER with a slow CONSUMER and made the repo's
        own documented usage lose most of its audio: push_frame() is
        synchronous and never yields, so `for f in frames: s.push_frame(f)`
        puts the entire file in the channel before the send loop runs once. On
        its first iteration the loop saw 6000 queued frames, discarded down to
        1000, and silently dropped ~83% of the session - while the uplink was
        never behind at all and every send completed instantly.

        Dropping on depth also did not achieve what it was for: push_frame has
        already allocated those frames, so discarding them reclaims the queue
        slot but not the memory spike that supposedly justified it.

        The discriminator is whether the backlog SHRINKS. Each iteration
        removes exactly one frame, so a burst from a producer that has
        stopped (or that pushes slower than we drain) shrinks monotonically
        and nothing is dropped - a file transcribes whole. A producer pushing
        faster than the uplink drains grows the backlog, and once it is over
        the cap and still not shrinking, frames are dropped as before. A
        producer exactly matching the drain rate holds it steady, which also
        drops: at the cap that is ~10s of accumulated latency, so the audio is
        already too stale to be worth sending.

        Returns:
            True if this frame should be dropped to bound the backlog.
        """
        backlog = self._input_ch.qsize()
        if backlog <= self.MAX_INPUT_BACKLOG_FRAMES:
            # Under the cap: reset the reference so a later burst is measured
            # from its own start rather than from a stale high-water mark.
            self._backlog_high_water = None
            return False

        previous = self._backlog_high_water
        self._backlog_high_water = backlog
        # First observation over the cap proves nothing about direction yet;
        # give the loop one iteration to show whether it is draining.
        return previous is not None and backlog >= previous
    def _pending_input_only_sentinels(self) -> bool:
        """
        The send loop's view of the probe: True also when it could not look.

        End-of-SESSION detection has to answer yes or no, and the two
        directions are not symmetric here. A wrong True skips one boundary's
        endpointing silence, merging two segments into one final. A wrong
        False injects 400ms of silence before a "Done" that forces the engine
        flush anyway - dead air at the end of every turn. So an unreadable
        queue takes the cheaper mistake.

        Kept as a separate method from the probe because the guards in
        _run_attempt must NOT see that collapse: for them an unverified probe
        means "run the session and find out", not "the input is consumed".
        """
        return self._probe_pending_input_only_sentinels() is not False

    def _note_engine_progress(self, *, is_final: bool) -> None:
        """
        Record that the ASR engine produced a transcript; clears the budget
        _check_server_liveness measures.

        Called ONLY for "partial"/"final" frames, and that narrowness is the
        whole point. The detector exists to catch a server whose WS layer is
        healthy while the engine behind it has wedged, so it must credit only
        evidence the ENGINE is consuming audio. An earlier version reset the
        budget at the top of the receive loop, before any classification, so
        every frame counted - including the gateway's own pub/sub control
        frames ({"type": "redirect", ...}), unknown types, non-object JSON,
        and frames that failed to parse at all. A gateway emitting any of
        those on a timer would hold the budget at zero forever and the
        detector could never fire, which is exactly the failure it was added
        to catch.

        A transcript with empty text still counts: a segment the engine
        finalized as silence is proof it processed the audio.

        Frames NOT credited are still handled normally by the caller - they
        are logged, and {"type": "error"} still raises. They simply do not buy
        the server another STALL_DETECTION_SECONDS of muteness.

        Setting _last_progress_at also permanently disarms the detector for
        this socket; see _check_server_liveness for why proving itself once is
        enough, and what that deliberately gives up.

        Args:
            is_final: Whether the frame was a "final" rather than a "partial".
                Both prove the engine is alive, so both clear the budget, but
                only a final is a FINALIZATION - which is what the post-Done
                drain waits for. Ending a turn on a post-Done partial would
                truncate the final that was still coming.
        """
        now = time.monotonic()
        self._bytes_sent_since_progress = 0
        self._last_progress_at = now
        if is_final:
            self._last_final_at = now

    def _check_server_liveness(self) -> None:
        """
        Detect a mute-but-connected server; called per caller frame sent.

        aiohttp's heartbeat only catches transport death - the gateway's WS
        layer answers pings even when the engine behind it is wedged. The
        question that matters is therefore "how much of the caller's audio
        has the server swallowed without transcribing any of it", and that is
        what is measured: caller-audio BYTES written to this socket since the
        engine last returned a transcript (_bytes_sent_since_progress, reset
        by _note_engine_progress), against the byte-equivalent of
        STALL_DETECTION_SECONDS of audio at the 16kHz wire rate. Non-transcript
        frames do not clear it - a chatty gateway is not a working engine.

        Counting delivered audio rather than wall time is what makes this
        self-normalising, and each property fixes a defect the previous
        amplitude-gated wall clock had:

        - A silent caller sends few or no bytes, so it cannot trip the bound
          however long it stays silent. The old unconditional 30s receive
          watchdog false-fired on exactly that.
        - The count resets on every message and is a measure of audio in
          flight, not "time since the first loud frame". The latched clock it
          replaces was never cleared by subsequent silence, so a cough
          followed by 35s of quiet killed a healthy session at the start of
          the next utterance.
        - A quiet or under-gained speaker delivers the same byte volume as a
          loud one, so a wedged server is detected for them too. The
          amplitude gate never armed at all below its threshold, and those
          sessions only discovered the wedge as zero transcripts at
          end_input - terminally, instead of retrying mid-call.
        - There is no per-frame numpy copy left on the hot path: the old gate
          built an int32 copy of every frame to compute a peak.

        Two consequences worth stating, since the unit is audio and not
        seconds:

        - A caller that pushes CONTINUOUS audio (a non-VAD capture streaming
          comfort noise) through a 30-second user pause can trip the bound
          against a healthy-but-quiet server, costing a redial. That is a
          retryable APIConnectionError which loses only the silence in
          flight, and it is the direction to err in - the alternative is a
          wedged engine going undetected for every speaker the amplitude gate
          could not hear.
        - A caller that pushes FASTER than real time (batch or file
          transcription) reaches the bound sooner in wall-clock terms, which
          is correct: the server has been handed 30s of audio either way.

        Evaluated inline in the send loop - no watchdog task to create,
        cancel, or leak. Worst-case detection delay is one frame period past
        the bound, noise against 30s.

        Raises:
            APIConnectionError: The server was handed STALL_DETECTION_SECONDS
                of audio and sent nothing; the framework redials.
        """
        # Derived at call time, not as a class constant, so the one source of
        # truth stays STALL_DETECTION_SECONDS.
        budget = int(
            self.STALL_DETECTION_SECONDS * self.WIRE_SAMPLE_RATE * 2  # Int16
        )
        # The bound applies ONLY until the engine has proven itself on this
        # socket. Once any transcript has arrived, mid-session policing stops.
        #
        # This is what separates "the engine is wedged" from "the user is
        # silent" WITHOUT trying to classify the audio - which is the part the
        # plugin cannot do. Counting every byte meant livekit's continuously
        # pushing pipeline accumulated a full 30s budget during any ordinary
        # user pause (someone listening to a long agent answer) and killed a
        # healthy attempt. livekit resets RecognizeStream._num_retries only on
        # FINAL_TRANSCRIPT (livekit/agents/stt/stt.py:531-533), so retries
        # bound CONSECUTIVE failures with no transcript in between - and a
        # single sustained silence is exactly that: each reconnect finds the
        # user still quiet, trips again, and around two minutes in max_retry
        # is gone and the agent is deaf for the rest of the call.
        #
        # An amplitude gate is NOT the answer and must not be reintroduced.
        # One existed (peak 500) and was deliberately removed because a
        # quiet or under-gained speaker never armed it, so a genuine wedge
        # went undetected for exactly the sessions that most needed it - see
        # tests/test_stream.py::...::test_quiet_speaker_wedge_is_detected,
        # whose audio peaks at 60. Comfort noise and quiet speech are not
        # separable by peak level, so the discriminator has to come from the
        # SERVER's behaviour rather than the caller's signal.
        #
        # What this gives up, stated plainly: an engine that works and then
        # wedges MID-session is no longer caught at 30s, only at end of
        # session where the completion gate raises for consumed audio that
        # produced no transcript. That is the narrow case, it still surfaces
        # as an error, and it is worth trading for never falsely accusing a
        # healthy engine - a false accusation is permanent, a late true one
        # costs one turn.
        if self._last_progress_at is not None:
            return

        if self._bytes_sent_since_progress <= budget:
            return

        # Wall-clock floor. The byte budget alone is not enough: a sender
        # faster than real time reaches 30s worth of audio in seconds, so a
        # batch/file caller - or a live caller's catch-up burst after a
        # network hiccup - tripped the detector on a perfectly healthy engine
        # that simply had not been given 30s of wall time to respond. The
        # retry then found the input already consumed and escalated to a
        # terminal TranscriptLostError, so a false positive here destroyed
        # the session outright.
        #
        # Measured from the start of the attempt: the guard above means no
        # transcript has ever arrived on this socket, so that IS the last
        # point at which the engine could have been believed healthy. For a
        # real-time caller both conditions cross at the same moment, so this
        # costs nothing in the case the detector was designed for; it only
        # holds back the fast sender.
        mute_seconds = time.monotonic() - self._attempt_started_at
        if mute_seconds < self.STALL_DETECTION_SECONDS:
            return

        logger.warning(
            f"Stream {self._session_id} has sent "
            f"{self._bytes_sent_since_progress}B of audio "
            f"(>{self.STALL_DETECTION_SECONDS}s worth) over {mute_seconds:.1f}s "
            "without a single transcript - the server is connected but the "
            "engine has never responded; abandoning the attempt"
        )
        raise APIConnectionError(
            "no response from server while streaming audio"
        )

    @staticmethod
    def _carries_signal(frame_data: object) -> bool:
        """
        True if a caller frame carries any signal at all.

        Decides whether consuming the frame counts as consuming
        unreplayable audio, which is what the completion gate's loss verdict
        keys on. An all-zeros frame (VAD zero-fill, synthesized padding,
        muted capture) could not have produced a transcript, so losing it is
        not transcript loss - treating every frame as real audio made a
        retry that shipped nothing but trailing silence look like a session
        whose transcript had vanished.

        Deliberately an all-zeros test rather than an amplitude threshold,
        and the asymmetry is the reason: mistaking silence for speech only
        costs a spurious honest error, while mistaking quiet speech for
        silence would let a lost transcript be reported as a clean, empty
        success - the exact failure class this design exists to make
        impossible. Near-silent-but-nonzero comfort noise therefore counts
        as real audio.

        Cheap by construction: a zero-copy int16 view and one reduction, no
        int32 promotion, on the per-frame hot path.
        """
        samples = np.frombuffer(frame_data, dtype=np.int16)  # type: ignore[call-overload]
        return bool(samples.any())

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

    async def _send_audio_chunk(
        self, audio_int16: np.ndarray, *, caller_audio: bool = True
    ) -> None:
        """
        Send an Int16 PCM audio chunk to the WebSocket.

        Args:
            audio_int16: The chunk to write.
            caller_audio: Whether this is the caller's audio and therefore
                counts toward the liveness bound (see
                _check_server_liveness). Only _on_segment_end's synthesized
                endpointing silence sets this False: it is our own audio, so
                injecting it must not move the detector. Every other path -
                frames, processor flushes at boundaries, the tail flush - is
                caller audio by default, so a new send site counts unless it
                deliberately opts out.

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
        if caller_audio:
            # Counted only once the send has actually completed: the bound
            # is about audio the server has received, not audio we queued.
            self._bytes_sent_since_progress += len(audio_bytes)

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
            # NOTE: the stall budget is deliberately NOT reset here. Arriving
            # frames are classified first, and only an engine transcript
            # counts as progress - see _note_engine_progress.
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
        is_transcript = msg_type in ("partial", "final")

        # Liveness accounting happens here, after classification, because only
        # a transcript frame proves the thing the detector is watching for.
        if is_transcript:
            self._note_engine_progress(is_final=msg_type == "final")

        # Detect start of speech. "text" is server-supplied: anything that
        # is not a string (absent, null, a number) counts as no text.
        #
        # Gated on is_transcript for the same reason the budget above is: a
        # "text" key is a SHAPE, not a meaning. The gateway's redirect frame
        # carries a target that can land in "text", and an error frame can
        # too; latching on either opened a speech turn no transcript would
        # ever close, leaving the caller waiting for an END_OF_SPEECH that
        # only arrives at session teardown.
        raw_text = data.get("text")
        text = raw_text.strip() if isinstance(raw_text, str) else ""
        if is_transcript and not self._speaking and text:
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
