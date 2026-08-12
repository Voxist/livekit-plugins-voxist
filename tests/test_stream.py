"""Unit tests for VoxistSTTStream: one socket per stream, framework-owned retries."""

import asyncio
import contextlib
import json
import logging
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import numpy as np
import pytest
from livekit.agents import APIConnectionError
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import APIConnectOptions

from livekit import rtc
from livekit.plugins.voxist.audio_processor import (
    MAX_FRAME_SIZE_BYTES,
    MIN_FRAME_SIZE_BYTES,
    AudioProcessor,
)
from livekit.plugins.voxist.exceptions import ConnectionError as VoxistConnectionError
from livekit.plugins.voxist.exceptions import TranscriptLostError
from livekit.plugins.voxist.stream import VoxistSTTStream

DEFAULT_CONFIG = {
    "sample_rate": 16000,
    "chunk_duration_ms": 100,
    "stride_overlap_ms": 20,
    "interim_results": True,
}


async def make_stream(language="fr", dial=None, config=None):
    """Build a stream whose auto-started task is cancelled, ready to drive."""
    stt = Mock()
    stt._config = dict(config or DEFAULT_CONFIG)
    stt._dial = dial if dial is not None else AsyncMock()

    stream = VoxistSTTStream(
        stt=stt,
        config=stt._config,
        language=language,
        conn_options=APIConnectOptions(max_retry=3, retry_interval=1.0, timeout=10.0),
    )
    stream._task.cancel()
    try:
        await stream._task
    except asyncio.CancelledError:
        pass
    return stream


def mock_event_ch(stream):
    """
    Replace the event channel with a Mock that still admits the REAL
    flush()/end_input() paths: those call _check_not_closed(), and a bare
    Mock's .closed attribute is truthy, which would make end_input() raise
    and silently push tests back to hand-building channel state.
    """
    stream._event_ch = Mock()
    stream._event_ch.closed = False
    return stream._event_ch


def attach_mock_ws(stream):
    """
    A mock socket faithful to aiohttp: no get_transport() method, and the
    private transport chain unreachable unless a test installs one. Mocking
    an API the library does not have is how the original backpressure bug
    shipped with a green suite.
    """
    ws = AsyncMock()
    ws.closed = False
    ws._response = None
    del ws.get_transport
    stream._ws = ws
    return ws


def install_fake_transport(stream, buffer_size):
    """Wire a fake transport into the real aiohttp attribute chain."""
    transport = Mock()
    transport.is_closing = Mock(return_value=False)
    if callable(buffer_size):
        transport.get_write_buffer_size = Mock(side_effect=buffer_size)
    else:
        transport.get_write_buffer_size = Mock(return_value=buffer_size)

    response = Mock()
    response.connection = Mock()
    response.connection.transport = transport
    stream._ws._response = response
    return transport


class FakeWS:
    """
    A scriptable WebSocket for _run orchestration tests.

    Messages put on `incoming` are yielded to the receive loop; `end()` makes
    the iterator stop, which is exactly how aiohttp surfaces a server close.
    """

    _END = object()

    def __init__(self):
        self.incoming: asyncio.Queue = asyncio.Queue()
        self.sent_bytes: list[bytes] = []
        self.sent_text: list[str] = []
        self.closed = False
        self.fail_on_text = False  # scripted failure: text sends reset
        self.fail_on_bytes = False  # scripted failure: audio sends reset
        self.hang_on_close = False  # scripted wedge: close() never returns
        self._response = None

    def feed_json(self, obj):
        self.incoming.put_nowait(
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data=json.dumps(obj))
        )

    def end(self):
        """Server closes the socket."""
        self.incoming.put_nowait(self._END)

    def __aiter__(self):
        return self

    async def __anext__(self):
        item = await self.incoming.get()
        if item is self._END:
            self.closed = True
            raise StopAsyncIteration
        return item

    async def send_bytes(self, data):
        if self.closed or self.fail_on_bytes:
            raise ConnectionResetError("closed")
        self.sent_bytes.append(data)

    async def send_str(self, data):
        if self.closed or self.fail_on_text:
            raise ConnectionResetError("closed")
        self.sent_text.append(data)

    async def close(self):
        if self.hang_on_close:
            # A wedged peer never ACKs the close handshake; aiohttp would
            # sit in ws.close() for its 10s default.
            await asyncio.Event().wait()
        self.closed = True
        self.end()


def frame(samples=160):
    return rtc.AudioFrame(
        data=np.zeros(samples, dtype=np.int16).tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=samples,
    )


def speech_frame(samples=1600, amplitude=20000):
    """A frame carrying real signal (sine wave, loud by default)."""
    t = np.arange(samples, dtype=np.float64)
    data = (np.sin(2 * np.pi * 440 * t / 16000.0) * amplitude).astype(np.int16)
    return rtc.AudioFrame(
        data=data.tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=samples,
    )


def quiet_speech_frame(samples=1600):
    """
    A quiet / under-gained speaker: real speech whose peak sits far below
    the amplitude gate the stall detector used to arm on (500). Sessions
    like this were invisible to that gate, so a wedged server went
    undetected for them until end_input killed the session terminally.
    """
    return speech_frame(samples, amplitude=60)


class TestSendPath:
    """Send-side invariants: bounded sends, no thresholds, honest failures."""

    @pytest.mark.asyncio
    async def test_timeout_constants_are_ordered(self):
        """
        The post-Done drain must be short (it is dead air at end of turn
        when the server wedges - a 30s bound once shipped meant half a
        minute of silence reported as success) yet no shorter than a single
        send's allowance: a server still ACKing sends deserves at least as
        long to flush its finals.
        """
        assert VoxistSTTStream.SEND_TIMEOUT_SECONDS > 0
        assert (
            VoxistSTTStream.SESSION_DRAIN_TIMEOUT_SECONDS
            >= VoxistSTTStream.SEND_TIMEOUT_SECONDS
        )
        assert VoxistSTTStream.SESSION_DRAIN_TIMEOUT_SECONDS <= 10.0, (
            "the drain bound is end-of-turn dead air on a wedged server; "
            "keep it well under conversational patience"
        )

    @pytest.mark.asyncio
    async def test_no_water_mark_thresholds(self):
        """
        The plugin must not compare the write buffer against its own marks.

        Absolute byte thresholds cannot be calibrated once: a plain socket
        pauses at 64KB while asyncio's SSL transport pauses at 512KB and
        relieves to 128KB. Marks chosen for one deadlocked the other.
        Backpressure belongs to aiohttp's own drain.
        """
        for removed in (
            "HIGH_WATER_MARK",
            "LOW_WATER_MARK",
            "BACKPRESSURE_MAX_WAIT",
            "BACKPRESSURE_CHECK_INTERVAL",
            "RECEIVE_TIMEOUT_SECONDS",
        ):
            assert not hasattr(VoxistSTTStream, removed), (
                f"{removed} reintroduces threshold- or watchdog-based flow "
                "control the transport already owns"
            )

    @pytest.mark.asyncio
    async def test_large_measured_buffer_never_throttles(self):
        """The buffer measurement is diagnostic; it must not gate the send."""
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        install_fake_transport(stream, 400 * 1024)  # far above any old mark

        loop = asyncio.get_running_loop()
        start = loop.time()
        await stream._send_audio_chunk(np.zeros(160, dtype=np.int16))
        assert loop.time() - start < 0.2
        ws.send_bytes.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_unmeasurable_buffer_never_throttles(self):
        """The production case: aiohttp exposes no accessor, measurement is 0."""
        stream = await make_stream()
        ws = attach_mock_ws(stream)

        assert stream._get_transport() is None
        assert stream._get_write_buffer_size() == 0

        await stream._send_audio_chunk(np.zeros(160, dtype=np.int16))
        ws.send_bytes.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_stalled_send_aborts_and_raises_api_error(self, monkeypatch):
        """
        A send that never completes must fail fast as a framework-visible
        error. aiohttp's drain has no timeout, and the transport is aborted
        because a cancelled send can leave a frame mid-flight.
        """
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        transport = install_fake_transport(stream, 1024)
        transport.abort = Mock()
        monkeypatch.setattr(VoxistSTTStream, "SEND_TIMEOUT_SECONDS", 0.1)

        async def never_completes(_data):
            await asyncio.Event().wait()

        ws.send_bytes = AsyncMock(side_effect=never_completes)

        with pytest.raises(APIConnectionError, match="send timeout"):
            await asyncio.wait_for(
                stream._send_audio_chunk(np.zeros(160, dtype=np.int16)), timeout=5.0
            )
        transport.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_closed_ws_raises_rather_than_skipping(self):
        """
        A closed socket must raise, not skip the chunk.

        Skipping made a mid-call close look like a clean end of input, so the
        stream "completed" with a truncated transcript and never retried.
        """
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        ws.closed = True

        with pytest.raises(APIConnectionError, match="closed while sending"):
            await stream._send_audio_chunk(np.zeros(160, dtype=np.int16))
        ws.send_bytes.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_transport_errors_are_contained(self):
        """A transport that raises must degrade to 0, not into the send path."""
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        transport = install_fake_transport(stream, 0)
        transport.get_write_buffer_size = Mock(side_effect=RuntimeError("detached"))

        assert stream._get_write_buffer_size() == 0
        await stream._send_audio_chunk(np.zeros(160, dtype=np.int16))
        ws.send_bytes.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_closing_transport_is_unmeasurable(self):
        stream = await make_stream()
        attach_mock_ws(stream)
        transport = install_fake_transport(stream, 123456)
        transport.is_closing = Mock(return_value=True)

        assert stream._get_transport() is None
        assert stream._get_write_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_unreachable_transport_reported_once(self, caplog):
        """Losing the accessor must leave one trace, not a per-chunk flood."""
        stream = await make_stream()
        attach_mock_ws(stream)

        with caplog.at_level(logging.DEBUG, logger="livekit.plugins.voxist"):
            for _ in range(50):
                stream._get_write_buffer_size()

        matching = [
            r for r in caplog.records if "cannot reach the WebSocket" in r.message
        ]
        assert len(matching) == 1

    @pytest.mark.asyncio
    async def test_sustained_sending_never_stalls(self):
        """
        Regression guard for the original outage: ~3x the old 2MB trip point
        of audio must flow without a single send blocking.
        """
        stream = await make_stream()
        ws = attach_mock_ws(stream)

        chunk = np.zeros(1600, dtype=np.int16)  # 3200B per send
        sends = (3 * 2 * 1024 * 1024) // 3200
        for _ in range(sends):
            await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=1.0)
        assert ws.send_bytes.await_count == sends

    @pytest.mark.asyncio
    async def test_buffer_parked_in_ssl_band_never_stalls(self):
        """
        A buffer resting between the SSL transport's low and high water marks
        (128KB-512KB) is normal TLS operation. The previous fix's thresholds
        made this exact band unescapable and throttled every chunk.
        """
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        install_fake_transport(stream, 320 * 1024)

        loop = asyncio.get_running_loop()
        start = loop.time()
        chunk = np.zeros(1600, dtype=np.int16)
        for _ in range(20):
            await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=1.0)
        assert loop.time() - start < 1.0
        assert ws.send_bytes.await_count == 20


class TestInputBacklogBound:
    """
    The unsent-audio backlog must be bounded without touching the channel.

    livekit's input channel is unbounded and push_frame() never blocks. The
    bound is applied on consumption - a frame over the cap is discarded
    instead of sent - so nothing is removed out of order, flush sentinels are
    honoured in sequence, and termination never depends on queue contents.
    """

    @pytest.mark.asyncio
    async def test_cap_is_sane(self):
        # Generous enough that a live conversation never reaches it (a stalled
        # uplink raises via SEND_TIMEOUT_SECONDS first), small enough to be an
        # actual OOM ceiling.
        assert 30.0 <= VoxistSTTStream.MAX_INPUT_BACKLOG_SECONDS <= 600.0

    @pytest.mark.asyncio
    async def test_frames_over_the_cap_are_dropped(self, monkeypatch):
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        monkeypatch.setattr(
            VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 0.1
        )  # 10 frames of 10ms

        total = 40
        for _ in range(total):
            stream._input_ch.send_nowait(frame())
        stream._input_ch.close()

        # A closed channel's real depth falls as the loop drains it; pin it so
        # the bound is actually exceeded while the loop runs.
        stream._input_ch = _OverloadedChannel(stream._input_ch, 999)

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        # With the depth pinned at 999, the pre-drain snapshot is 999 and
        # never fully drains, so the growth subject to the bound is exactly
        # the number of pops so far: the first 10 frames (0.1s bound / 10ms
        # frames) are sent, the remaining 30 dropped.
        assert stream.dropped_frames == total - 10
        assert stream._done_sent, "the session must still end with Done"
        assert ws.send_str.await_args.args[0] == "Done"

    @pytest.mark.asyncio
    async def test_no_drops_below_the_cap(self, monkeypatch):
        stream = await make_stream()
        attach_mock_ws(stream)
        monkeypatch.setattr(
            VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 1.0
        )

        for _ in range(20):
            stream._input_ch.send_nowait(frame())
        stream._input_ch.close()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)
        assert stream.dropped_frames == 0

    @pytest.mark.asyncio
    async def test_sentinels_do_not_end_the_session(self, monkeypatch):
        """
        flush() is a segment boundary, not "Done".

        The engine finalizes segments on its own; sending "Done" per sentinel
        was the original architecture error - the gateway closes the socket
        after it, so everything past the first turn was lost.
        """
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        stream._audio_processor = Mock()
        stream._audio_processor.flush = Mock(return_value=[])
        stream._audio_processor.process_audio_frame = Mock(return_value=[])

        for _ in range(3):
            stream._input_ch.send_nowait(frame())
            stream._input_ch.send_nowait(VoxistSTTStream._FlushSentinel())
        stream._input_ch.close()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        # One Done for the whole session, at the end - regardless of sentinels
        assert ws.send_str.await_count == 1
        assert stream._audio_processor.flush.call_count >= 3

    @pytest.mark.asyncio
    async def test_sentinel_order_is_preserved(self, monkeypatch):
        """A sentinel is honoured at its own position, never displaced."""
        stream = await make_stream()
        attach_mock_ws(stream)
        monkeypatch.setattr(
            VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 10.0
        )

        events = []
        stream._audio_processor = Mock()
        stream._audio_processor.process_audio_frame = Mock(
            side_effect=lambda _b: events.append("audio") or []
        )
        stream._audio_processor.flush = Mock(
            side_effect=lambda: events.append("flush") or []
        )

        for _ in range(3):
            stream._input_ch.send_nowait(frame())
        stream._input_ch.send_nowait(VoxistSTTStream._FlushSentinel())
        for _ in range(3):
            stream._input_ch.send_nowait(frame())
        stream._input_ch.close()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        # trailing "flush" comes from the end-of-channel tail flush
        assert events == ["audio"] * 3 + ["flush"] + ["audio"] * 3 + ["flush"]

    @pytest.mark.asyncio
    async def test_terminates_with_sentinel_only_backlog(self, monkeypatch):
        """A backlog of sentinels must never wedge the loop (old regression)."""
        stream = await make_stream()
        attach_mock_ws(stream)
        monkeypatch.setattr(
            VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 0.05
        )

        for _ in range(20):
            stream._input_ch.send_nowait(VoxistSTTStream._FlushSentinel())
        stream._input_ch.close()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

    @pytest.mark.asyncio
    async def test_drop_warning_is_rate_limited(self, monkeypatch, caplog):
        stream = await make_stream()
        attach_mock_ws(stream)
        monkeypatch.setattr(
            VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 0.05
        )
        monkeypatch.setattr(VoxistSTTStream, "DROP_LOG_INTERVAL_SECONDS", 3600.0)

        # Patch the *name* `time` inside the plugin's own module namespace -
        # NOT `time.monotonic` on the real stdlib module. `stream.py` does
        # `import time` and calls `time.monotonic()`, so rebinding the
        # module-level `time` reference that production code sees leaves the
        # actual `time` module - and therefore the asyncio event loop's
        # clock, which is `BaseEventLoop.time() == time.monotonic()` on that
        # very same real module - completely untouched. A previous version
        # of this test patched `livekit.plugins.voxist.stream.time.monotonic`
        # by dotted string, which resolves attribute-by-attribute to the
        # real stdlib module and hijacked the event loop's own clock along
        # with it, making the whole test's pass/fail depend on how many
        # times the loop happened to read the clock.
        import time as real_time_module

        from livekit.plugins.voxist import stream as stream_module

        real_monotonic = real_time_module.monotonic

        # Inexhaustible by construction: a mutable counter held in a
        # closure, never a finite iterator that could raise StopIteration
        # into whatever calls it.
        clock_state = {"value": 1.0}

        def fake_monotonic() -> float:
            # Fresh-host clock: small monotonic values must not suppress
            # the first report (0.0 was once used as a sentinel and did
            # exactly that).
            value = clock_state["value"]
            clock_state["value"] += 0.001
            return value

        monkeypatch.setattr(
            stream_module, "time", SimpleNamespace(monotonic=fake_monotonic)
        )

        for _ in range(200):
            stream._input_ch.send_nowait(frame())
        stream._input_ch.close()

        # Pin the depth so the bound stays exceeded while the loop runs.
        stream._input_ch = _OverloadedChannel(stream._input_ch, 999)

        loop = asyncio.get_running_loop()
        loop_time_before = loop.time()

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._send_audio_task(), timeout=10.0)

            drops = [r for r in caplog.records if "dropping audio" in r.message]
            # Pinned depth again: subject = pops, 0.05s bound / 10ms
            # frames = first 5 sent, 195 dropped.
            assert stream.dropped_frames == 200 - 5
            assert len(drops) == 1

            # The limiter genuinely tracks elapsed time rather than just
            # "log the first drop and never again": once
            # DROP_LOG_INTERVAL_SECONDS has passed on the fake clock, the
            # next drop must be reported too.
            clock_state["value"] += VoxistSTTStream.DROP_LOG_INTERVAL_SECONDS + 1
            stream._note_dropped_frame()

        drops_after = [r for r in caplog.records if "dropping audio" in r.message]
        assert len(drops_after) == 2

        # Prove the event loop's own clock kept advancing on the real
        # monotonic clock throughout, unaffected by the patched seam, and
        # that the real stdlib `time.monotonic` was never mutated.
        await asyncio.sleep(0.01)
        assert loop.time() > loop_time_before
        assert real_time_module.monotonic is real_monotonic


class TestRunOutcome:
    """
    _run performs exactly one attempt; the framework owns retries.

    An earlier design nested its own reconnect loop inside livekit's and
    every retry-budget bug lived in that duplication - including a loop whose
    exit condition was unreachable.
    """

    @pytest.mark.asyncio
    async def test_dial_failure_is_a_framework_error(self):
        """Transport-level dial failures become APIConnectionError (retried)."""
        stream = await make_stream(
            dial=AsyncMock(side_effect=VoxistConnectionError("refused"))
        )
        with pytest.raises(APIConnectionError):
            await stream._run()

    @pytest.mark.asyncio
    async def test_auth_failure_propagates_unwrapped(self):
        """
        AuthenticationError must NOT become an APIError.

        livekit retries APIErrors; retrying cannot fix a revoked key, so the
        true cause propagates immediately.
        """
        from livekit.plugins.voxist.exceptions import AuthenticationError

        stream = await make_stream(
            dial=AsyncMock(side_effect=AuthenticationError("revoked"))
        )
        with pytest.raises(AuthenticationError):
            await stream._run()

    @pytest.mark.asyncio
    async def test_normal_session_completes_with_end_of_speech(self):
        """Audio in, finals out, Done once, server closes, END_OF_SPEECH."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        stream._event_ch = Mock()

        async def scenario():
            stream._input_ch.send_nowait(frame(1600))
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "bonjour", "confidence": 0.9})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # end_input
            await asyncio.sleep(0.05)
            ws.end()  # gateway closes after Done

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert ws.sent_text == ["Done"]
        assert stream._done_sent and stream._session_complete

        emitted = [c.args[0].type for c in stream._event_ch.send_nowait.call_args_list]
        from livekit.agents.stt import SpeechEventType

        assert SpeechEventType.FINAL_TRANSCRIPT in emitted
        assert emitted[-1] == SpeechEventType.END_OF_SPEECH

    @pytest.mark.asyncio
    async def test_server_close_before_done_is_an_interruption(self):
        """A close mid-input must raise for retry, never 'complete normally'."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        stream._event_ch = Mock()

        stream._input_ch.send_nowait(frame(1600))  # input NOT ended
        ws.end()  # server drops the socket

        # Depending on who observes the close first (receive loop ending, the
        # closed-socket check, or the send itself resetting), the message
        # differs - the invariant is that it is an APIConnectionError so the
        # framework retries, and never a "normal" completion.
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        assert not stream._session_complete
        assert not stream._done_sent

    @pytest.mark.asyncio
    async def test_completed_session_is_not_redialed(self):
        """A framework retry after completion must be a no-op."""
        dial = AsyncMock(side_effect=AssertionError("must not dial again"))
        stream = await make_stream(dial=dial)
        stream._session_complete = True

        await asyncio.wait_for(stream._run(), timeout=1.0)
        dial.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_drain_timeout_without_any_audio_stays_retryable(self, monkeypatch):
        """
        A server that ignores Done on a session that never carried audio is
        a FAILED attempt, not a silent success - and since nothing was
        consumed, the failure stays retryable (the retry completes empty via
        the zero-audio short-circuit). end_input() lands AFTER the dial here:
        landing before it would legitimately skip the dial entirely.
        """
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.2)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        loop = asyncio.get_running_loop()
        start = loop.time()
        run_task = asyncio.create_task(stream._run())
        await asyncio.sleep(0.05)  # dialed; the send loop is waiting on input
        stream.end_input()

        with pytest.raises(APIConnectionError, match="no transcript"):
            await asyncio.wait_for(run_task, timeout=5.0)
        elapsed = loop.time() - start

        assert elapsed < 3.0, "the drain must still be bounded"
        assert not stream._session_complete
        assert ws.sent_text == ["Done"]

    @pytest.mark.asyncio
    async def test_drain_timeout_after_consumed_audio_raises_transcript_lost(
        self, monkeypatch
    ):
        """
        Wedged server after Done, audio consumed, nothing delivered: a retry
        cannot replay the audio, and the installed _main_task retries EVERY
        APIError regardless of retryable - so the honest failure must be a
        non-APIError (TranscriptLostError): one terminal error event,
        immediate death, no misleading "recoverable" events.
        """
        from livekit.agents import APIError

        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.2)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()

        with pytest.raises(TranscriptLostError) as excinfo:
            await asyncio.wait_for(stream._run(), timeout=5.0)
        assert not isinstance(excinfo.value, APIError), (
            "TranscriptLostError must not be an APIError: the framework "
            "would burn max_retry no-op attempts on it"
        )
        assert not stream._session_complete

    @pytest.mark.asyncio
    async def test_drain_timeout_after_finals_completes_with_warning(
        self, monkeypatch, caplog
    ):
        """
        If finals already made it out for THIS attempt's audio, a wedged
        post-Done server costs at most a trailing transcript: complete (with
        a warning), do not fail a session whose data was delivered.
        """
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.2)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "bonjour", "confidence": 0.9})
            await asyncio.sleep(0.05)
            stream.end_input()
            # the server never closes: drain must time out

        task = asyncio.create_task(scenario())
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert stream._session_complete
        assert any("trailing transcript" in r.message for r in caplog.records)

    @pytest.mark.asyncio
    async def test_interim_only_drain_timeout_completes_with_loud_warning(
        self, monkeypatch, caplog
    ):
        """
        Interims were delivered (the user saw text) but the server wedged
        before the final: complete rather than error - the audio cannot be
        replayed, so no retry improves the outcome and erroring vaporizes a
        session whose content was substantially delivered. The warning must
        say that finals-only consumers still experience loss.
        """
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.2)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "partial", "text": "bonj", "confidence": 0.5})
            await asyncio.sleep(0.05)
            stream.end_input()
            # server wedges: no final, no close

        task = asyncio.create_task(scenario())
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert stream._session_complete
        wedge_warnings = [
            r.message for r in caplog.records if "interim" in r.message
        ]
        assert wedge_warnings, "the interim-only completion must be loud"
        assert any("FINAL_TRANSCRIPT" in m for m in wedge_warnings), (
            "the warning must state that finals-only consumers see loss"
        )

    @pytest.mark.asyncio
    async def test_earlier_attempts_final_does_not_mask_later_turn_loss(
        self, monkeypatch
    ):
        """
        Attempt 1 delivered a final for turn 1, then died mid-input. Attempt
        2 carried turn 2's audio into a wedged server and delivered NOTHING
        for it. Keying the drain taxonomy on the session-scoped final flag
        let attempt 2 complete as success - turn 2 silently vanished. It
        must fail honestly instead.
        """
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.2)
        ws1, ws2 = FakeWS(), FakeWS()
        dial = AsyncMock(side_effect=[ws1, ws2])
        stream = await make_stream(dial=dial)
        mock_event_ch(stream)

        # Attempt 1: turn 1 flows, a final arrives, then the server drops
        # the socket while input is still open - an interruption.
        async def attempt1():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws1.feed_json({"type": "final", "text": "tour un", "confidence": 0.9})
            await asyncio.sleep(0.05)
            ws1.end()

        task = asyncio.create_task(attempt1())
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task
        assert stream._final_received, "attempt 1's final was delivered"

        # Attempt 2 (the framework's retry): turn 2 flows into a wedged
        # server that never answers and never closes.
        async def attempt2():
            await asyncio.sleep(0.02)
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream.end_input()

        task = asyncio.create_task(attempt2())
        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task
        assert not stream._session_complete, (
            "turn 2 was wholly lost; completing would mask it behind "
            "attempt 1's final"
        )


class TestPerAttemptStateReset:
    """
    _run() executes once per framework retry; flags scoped to one attempt
    must not leak into the next. A stale _done_sent=True from a failed
    attempt once disabled the "server closed before end of input" guard,
    turning a mid-call interruption into a fabricated clean completion.
    """

    @pytest.mark.asyncio
    async def test_stale_done_sent_does_not_disable_interruption_guard(self):
        """Retry after a failed attempt: mid-input close must still raise."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        stream._event_ch = Mock()

        # State exactly as a failed previous attempt leaves it
        stream._done_sent = True
        stream._transport_lookup_failed = True

        stream._input_ch.send_nowait(frame(1600))  # input NOT ended
        ws.end()  # server drops the socket mid-input

        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        assert not stream._session_complete

    @pytest.mark.asyncio
    async def test_transport_latch_resets_per_attempt(self):
        """The log-once latch describes one socket, not the whole stream."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        stream._event_ch = Mock()
        stream._transport_lookup_failed = True  # left over from attempt N-1

        stream._input_ch.send_nowait(frame(1600))
        ws.end()
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)

        assert stream._transport_lookup_failed is False


class TestExhaustedInputRetry:
    """
    Streamed audio cannot be replayed. A framework retry that finds the
    input already consumed must not dial a fresh socket, send a bare Done,
    and present an empty session as success (total transcript loss).
    """

    @pytest.mark.asyncio
    async def test_retry_with_consumed_input_and_no_finals_raises(self):
        """The end_input() shape: attempt 1 dies writing Done, sentinel gone."""
        ws = FakeWS()
        ws.fail_on_text = True  # the attempt dies when it writes "Done"
        dial = AsyncMock(return_value=ws)
        stream = await make_stream(dial=dial)
        mock_event_ch(stream)

        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()

        # Attempt 1: consumes the frame and the sentinel, fails on Done.
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)

        assert stream._audio_consumed
        assert not stream._final_received
        assert dial.await_count == 1

        # Attempt 2 (the framework's retry): must fail honestly, and must
        # not dial - there is nothing left to send. TranscriptLostError, not
        # APIError: the installed _main_task retries every APIError, so an
        # APIError here meant max_retry misleading "recoverable" events and
        # ~4s of dead air before the same death.
        with pytest.raises(TranscriptLostError, match="cannot be replayed"):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        assert dial.await_count == 1, "a bare-Done redial fabricates success"
        assert not stream._session_complete

    @pytest.mark.asyncio
    async def test_trailing_sentinel_does_not_bypass_exhausted_guard(self):
        """
        THE root-fact test: end_input() is flush() + close(), so after an
        attempt dies on the LAST audio frame the channel still holds one
        trailing sentinel. The old guard required qsize()==0, so the retry
        dialed, consumed only the sentinel, sent a bare Done and completed
        as SUCCESS with zero finals - total transcript loss presented as a
        clean session. The retry must fail honestly without dialing.
        """
        ws = FakeWS()
        ws.fail_on_bytes = True  # the attempt dies sending the LAST frame
        dial = AsyncMock(return_value=ws)
        stream = await make_stream(dial=dial)
        mock_event_ch(stream)

        # Deterministic chunking: every frame yields exactly one chunk, so
        # the failure lands on the audio send, before the sentinel is pulled.
        stream._audio_processor = Mock()
        stream._audio_processor.process_audio_frame = Mock(
            return_value=[np.ones(1600, dtype=np.int16)]
        )
        stream._audio_processor.flush = Mock(return_value=[])

        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()  # REAL path: trailing sentinel + close

        # Attempt 1 dies on the frame; the sentinel is still queued.
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        assert stream._audio_consumed
        assert stream._input_ch.qsize() == 1, (
            "precondition: the trailing sentinel must still be queued - "
            "this is the exact state the old guard let through"
        )

        # Attempt 2: the sentinel must count as consumed input.
        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        assert dial.await_count == 1, (
            "the retry dialed for a sentinel-only channel: bare Done, "
            "empty success, transcript lost"
        )
        assert not stream._session_complete

    @pytest.mark.asyncio
    async def test_retry_with_consumed_input_but_finals_completes(self):
        """
        Finals already emitted: the deliverable data made it out. The retry
        completes without a pointless redial instead of erroring a session
        whose transcript was delivered. The channel state is built by the
        REAL end_input() (trailing sentinel included - the state the guard
        must classify as consumed); the delivery flags are set to the exact
        post-failure picture (their setting paths are covered by the
        no-finals test above and the normal-session test).
        """
        from livekit.agents.stt import SpeechEventType

        dial = AsyncMock(side_effect=AssertionError("must not dial again"))
        stream = await make_stream(dial=dial)
        events = mock_event_ch(stream)

        # As left by a failed attempt that had consumed every frame after
        # the caller ended input normally:
        stream.end_input()
        stream._audio_consumed = True
        stream._final_received = True
        stream._speaking = True

        await asyncio.wait_for(stream._run(), timeout=5.0)

        assert stream._session_complete
        dial.assert_not_awaited()
        emitted = [c.args[0].type for c in events.send_nowait.call_args_list]
        assert emitted == [SpeechEventType.END_OF_SPEECH]

    @pytest.mark.asyncio
    async def test_zero_audio_session_completes_without_dialing(self):
        """
        end_input() with no frames ever pushed: there is nothing to
        transcribe, so there is nothing to dial for - no socket, no bare
        Done, no synthesized silence shipped to an engine that would have
        to invent a result for it. A clean empty completion, offline.
        """
        dial = AsyncMock(side_effect=AssertionError("zero-audio must not dial"))
        stream = await make_stream(dial=dial)
        events = mock_event_ch(stream)

        stream.end_input()  # REAL path: sentinel + close, nothing else

        await asyncio.wait_for(stream._run(), timeout=5.0)

        assert stream._session_complete
        dial.assert_not_awaited()
        # No speech ever started, so no events are owed either
        events.send_nowait.assert_not_called()


class TestCompletionGate:
    """
    ONE gate decides whether a session succeeded.

    Three review rounds found the same bug - a clean success reported after
    the entire transcript was lost - at three DIFFERENT entry points into
    the completion path (an exhausted-input retry, a trailing FlushSentinel,
    a prompt server close). Each round guarded that one entry point. These
    tests pin the structural property instead: the verdict lives inside the
    exit, so a new caller cannot forget to check.
    """

    @pytest.mark.asyncio
    async def test_prompt_close_after_done_without_transcript_is_not_success(
        self,
    ):
        """
        THE round-7 finding. The gateway closes the socket promptly after
        "Done" without ever sending a transcript - its engine crashed, and an
        aiohttp heartbeat death looks identical from here: `async for msg in
        ws` simply ends, with no exception. The drain's wait_for then returns
        WITHOUT TimeoutError, so the whole outcome taxonomy was skipped and a
        session with real audio and zero transcripts was reported as a clean
        success.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            await asyncio.sleep(0.05)
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(0.05)
            ws.end()  # closes right after Done, having sent nothing

        task = asyncio.create_task(scenario())
        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert ws.sent_text == ["Done"], "precondition: Done really was sent"
        assert not stream._session_complete, (
            "a session that consumed real audio and delivered nothing is not "
            "a success, however politely the server closed"
        )

    @pytest.mark.asyncio
    async def test_prompt_close_still_closes_the_speech_state(self):
        """
        Same prompt close, but START_OF_SPEECH already reached the caller:
        the engine sent a partial (interims disabled, so nothing was
        delivered) and then died. The terminal raise must not skip
        END_OF_SPEECH - the raise sites used to bypass the completion path
        entirely, leaving _speaking True and a turn open forever downstream.
        """
        config = dict(DEFAULT_CONFIG, interim_results=False)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws), config=config)
        events = mock_event_ch(stream)

        async def scenario():
            await asyncio.sleep(0.05)
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            # Speech detected (START_OF_SPEECH) but, with interims disabled,
            # nothing is DELIVERED to the caller.
            ws.feed_json({"type": "partial", "text": "bonj"})
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        emitted = [c.args[0].type for c in events.send_nowait.call_args_list]
        assert emitted == [
            SpeechEventType.START_OF_SPEECH,
            SpeechEventType.END_OF_SPEECH,
        ], f"START_OF_SPEECH must be matched even when the session fails: {emitted}"
        assert not stream._speaking

    @pytest.mark.asyncio
    async def test_interrupted_attempt_closes_the_speech_state(self):
        """
        The backstop in _run's own finally: an attempt that raises before it
        ever reaches the gate (here a mid-input server close) must still not
        leave a START_OF_SPEECH unmatched. If the retry budget is exhausted,
        that raise is the end of the stream.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        events = mock_event_ch(stream)

        async def scenario():
            await asyncio.sleep(0.05)
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "bonjour"})
            await asyncio.sleep(0.05)
            ws.end()  # server drops the socket, input still open

        task = asyncio.create_task(scenario())
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        emitted = [c.args[0].type for c in events.send_nowait.call_args_list]
        assert emitted[-1] == SpeechEventType.END_OF_SPEECH
        assert not stream._speaking

    @pytest.mark.asyncio
    async def test_success_is_decided_in_exactly_one_place(self):
        """
        The structural property, asserted on the source: if a second place
        could mark a session complete, round 8 would find it. Likewise the
        terminal TranscriptLostError has exactly one raise site, inside the
        gate - it used to be raised from branches that never emitted
        END_OF_SPEECH.
        """
        import inspect

        module_src = inspect.getsource(
            __import__(
                "livekit.plugins.voxist.stream", fromlist=["stream"]
            )
        )
        gate_src = inspect.getsource(VoxistSTTStream._finish_session)

        assert module_src.count("_session_complete = True") == 1, (
            "only the completion gate may mark a session successful"
        )
        assert gate_src.count("_session_complete = True") == 1

        assert module_src.count("raise TranscriptLostError(") == 1, (
            "the terminal verdict must have a single raise site"
        )
        assert gate_src.count("raise TranscriptLostError(") == 1

    @pytest.mark.asyncio
    async def test_no_exit_from_an_attempt_bypasses_the_gate(self):
        """
        The invariant that makes the whole class of bug impossible rather
        than merely guarded, checked on the parse tree: inside
        _run_attempt, every `return` is immediately preceded by a
        _finish_session call, and the exchange's normal fall-through ends
        with one too. A branch that "completes" without passing through the
        gate cannot be added without failing here - which is exactly how
        rounds 5, 6 and 7 each found a fresh unguarded entry point.
        """
        import ast
        import inspect
        import textwrap

        fn = ast.parse(
            textwrap.dedent(inspect.getsource(VoxistSTTStream._run_attempt))
        ).body[0]

        def is_gate_call(stmt):
            return (
                isinstance(stmt, ast.Expr)
                and isinstance(stmt.value, ast.Call)
                and isinstance(stmt.value.func, ast.Attribute)
                and stmt.value.func.attr == "_finish_session"
            )

        def statement_lists(node):
            for field in ("body", "orelse", "finalbody"):
                seq = getattr(node, field, None)
                if isinstance(seq, list) and seq and isinstance(seq[0], ast.stmt):
                    yield seq

        returns = 0
        for node in ast.walk(fn):
            for block in statement_lists(node):
                for i, stmt in enumerate(block):
                    if not isinstance(stmt, ast.Return):
                        continue
                    returns += 1
                    assert i > 0 and is_gate_call(block[i - 1]), (
                        f"the return on line {stmt.lineno} of _run_attempt "
                        "leaves the session without a verdict from the "
                        "completion gate"
                    )

        assert returns >= 3, (
            "sanity: the early exits (no audio, exhausted input, drain "
            "timeout) must still be there"
        )

        exchange = fn.body[-1]
        assert isinstance(exchange, ast.Try), "the exchange must be the tail"
        assert is_gate_call(exchange.body[-1]), (
            "falling out of the exchange must render a verdict, not imply "
            "success - a server that closes promptly after Done having sent "
            "nothing lands exactly here"
        )

    @pytest.mark.asyncio
    async def test_gate_emits_end_of_speech_on_every_verdict(self):
        """
        Whatever the gate decides, a pending START_OF_SPEECH is closed
        first. Driven directly over the taxonomy so a new verdict cannot be
        added without an END_OF_SPEECH.
        """
        from livekit.plugins.voxist.stream import _SessionOutcome

        def outcome(*, final=False, interim=False, lost=False, concluded=False):
            return _SessionOutcome(
                delivered_final=final,
                delivered_interim=interim,
                unrecoverable_audio=lost,
                concluded=concluded,
                detail=(
                    f"final={final} interim={interim} lost={lost} "
                    f"concluded={concluded}"
                ),
            )

        verdicts = [
            # (outcome, expected exception type or None)
            (outcome(final=True, lost=True, concluded=True), None),
            (outcome(final=True, lost=True), None),
            (outcome(interim=True, lost=True), None),
            (outcome(lost=True, concluded=True), TranscriptLostError),
            (outcome(), APIConnectionError),
            (outcome(concluded=True), None),
        ]

        for outcome, expected in verdicts:
            stream = await make_stream()
            events = mock_event_ch(stream)
            stream._speaking = True

            if expected is None:
                stream._finish_session(outcome)
                assert stream._session_complete
            else:
                with pytest.raises(expected):
                    stream._finish_session(outcome)
                assert not stream._session_complete

            emitted = [c.args[0].type for c in events.send_nowait.call_args_list]
            assert emitted == [SpeechEventType.END_OF_SPEECH], (
                f"verdict {outcome.detail!r} left the speech state open"
            )

    @pytest.mark.asyncio
    async def test_prompt_close_with_no_audio_completes_empty(self):
        """
        The mirror image: nothing was consumed, so nothing was lost. A
        prompt close after a bare Done is a clean empty session, not an
        error - the gate must not overcorrect into failing those.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        run_task = asyncio.create_task(stream._run())
        await asyncio.sleep(0.05)  # dialed, send loop waiting on input
        stream.end_input()
        await asyncio.sleep(0.05)
        ws.end()

        await asyncio.wait_for(run_task, timeout=5.0)
        assert stream._session_complete


class TestConsumedAudioPredicate:
    """
    "Consumed real audio" must mean audio that could plausibly have produced
    a transcript. Counting every frame - including pure silence - made a
    retry that shipped nothing but the caller's trailing captured silence
    look like a session whose transcript had vanished, and killed it with a
    terminal error although every final had already been delivered.
    """

    @pytest.mark.asyncio
    async def test_pure_silence_is_not_consumed_audio(self):
        stream = await make_stream()
        attach_mock_ws(stream)
        mock_event_ch(stream)

        for _ in range(10):
            stream._input_ch.send_nowait(frame(1600))  # zeros
        stream.end_input()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        assert stream._audio_consumed is False
        assert stream._real_audio_this_attempt is False

    @pytest.mark.asyncio
    async def test_signal_bearing_audio_is_consumed_audio(self):
        stream = await make_stream()
        attach_mock_ws(stream)
        mock_event_ch(stream)

        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        assert stream._audio_consumed
        assert stream._real_audio_this_attempt

    @pytest.mark.asyncio
    async def test_quiet_speech_counts_as_real_audio(self):
        """
        The asymmetry, pinned: mistaking quiet speech for silence would let
        a lost transcript be reported as a clean empty success, so anything
        that is not pure zeros counts.
        """
        stream = await make_stream()
        attach_mock_ws(stream)
        mock_event_ch(stream)

        stream._input_ch.send_nowait(quiet_speech_frame())
        stream.end_input()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)
        assert stream._audio_consumed

    @pytest.mark.asyncio
    async def test_dropped_real_frame_still_counts_as_consumed(self, monkeypatch):
        """
        A frame discarded to bound the backlog has still irrevocably left
        the channel: no retry can replay it, so it is consumed.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 0.001
        )
        stream = await make_stream()
        attach_mock_ws(stream)
        mock_event_ch(stream)

        for _ in range(5):
            stream._input_ch.send_nowait(speech_frame())
        stream.end_input()

        # Pin the depth so the bound is exceeded while the loop runs.
        stream._input_ch = _OverloadedChannel(stream._input_ch, 5)

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        assert stream.dropped_frames > 0
        assert stream._audio_consumed

    @pytest.mark.asyncio
    async def test_silence_only_retry_after_finals_completes(self, monkeypatch):
        """
        Attempt 1 delivered every final, then the connection blipped. The
        retry ships nothing but the caller's trailing captured silence plus
        Done; the server has nothing to finalize, so it neither answers nor
        closes within the drain.

        Keying the loss verdict on the SESSION's consumption made this a
        terminal TranscriptLostError - the stream died although its whole
        transcript had been delivered. An attempt that shipped only silence
        owes nothing and can lose nothing: it is judged on the session.
        """
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.2)
        ws1, ws2 = FakeWS(), FakeWS()
        dial = AsyncMock(side_effect=[ws1, ws2])
        stream = await make_stream(dial=dial)
        mock_event_ch(stream)

        async def attempt1():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws1.feed_json({"type": "final", "text": "bonjour", "confidence": 0.9})
            await asyncio.sleep(0.05)
            ws1.end()  # blip: server drops the socket, input still open

        task = asyncio.create_task(attempt1())
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task
        assert stream._final_received, "attempt 1 delivered the transcript"

        async def attempt2():
            await asyncio.sleep(0.02)
            for _ in range(3):
                stream._input_ch.send_nowait(frame(1600))  # captured silence
            await asyncio.sleep(0.05)
            stream.end_input()
            # the server has nothing to finalize: no answer, no close

        task = asyncio.create_task(attempt2())
        await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert stream._session_complete, (
            "the session's transcript was fully delivered; a retry that "
            "shipped only silence cannot turn it into transcript loss"
        )


class TestSentinelPredicateFailsSafe:
    """
    The probe reads livekit's private Chan._queue. If that attribute ever
    moves, what the plugin does next decides what a rename costs - and the
    original fallback (qsize()==0) was the pre-fix buggy check itself, so a
    livekit rename would silently resurrect total-transcript-loss-as-success.

    The probe is tri-state for that reason: True/False when it could look,
    None when it could not, because the safe answer differs per caller. The
    send loop still collapses None to True (a merged segment is the cheap
    mistake), while the shortcuts in _run_attempt refuse to fire on None -
    both of them decide a session's fate WITHOUT attempting the exchange, and
    an unverified probe is not grounds for that.
    """

    @pytest.mark.asyncio
    async def test_unreadable_queue_reports_the_incompatibility_once(self, caplog):
        stream = await make_stream()
        mock_event_ch(stream)
        stream.end_input()
        # The rename: the deque backing qsize() is no longer where we look.
        del stream._input_ch._queue

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            for _ in range(20):
                assert stream._pending_input_only_sentinels() is True

        warnings = [
            r for r in caplog.records if "Chan._queue has moved" in r.message
        ]
        assert len(warnings) == 1, (
            "name the livekit-version incompatibility exactly once per stream"
        )

    @pytest.mark.asyncio
    async def test_unreadable_queue_runs_the_session_instead_of_guessing(self):
        """
        With the channel uninspectable, the attempt must be RUN, and the
        verdict must come from what happened on the socket.

        This test previously asserted the opposite - never dial, raise
        TranscriptLostError - under the name
        "prefers_an_honest_error_to_a_fake_success". That name encodes a false
        dichotomy: an error inferred from a probe that failed is not honest
        either. It abandons whatever audio the channel still holds (the
        unreadable probe is exactly the case where we do NOT know that it
        holds none) and it fabricates a terminal verdict for an exchange that
        was never attempted. Dialing costs one socket and possibly a
        zero-duration billing event; it cannot lose audio and cannot invent an
        outcome.

        Here the channel really does hold nothing but the trailing sentinel,
        so the session ships a bare Done, the engine answers nothing, and the
        gate raises - the same end state as before, but earned.
        """
        class RenamedQueueChan:
            """
            The rename, modelled faithfully: livekit's channel still WORKS -
            it iterates, closes and reports qsize() as always - but the deque
            our probe reaches for is no longer called _queue. Deleting the
            real attribute instead breaks livekit's own iteration, which tests
            a channel that could never exist.
            """

            def __init__(self, chan):
                self._chan = chan

            def __getattr__(self, name):
                if name == "_queue":
                    raise AttributeError(name)
                return getattr(self._chan, name)

            def __aiter__(self):
                return self._chan.__aiter__()

        ws = FakeWS()
        dial = AsyncMock(return_value=ws)
        stream = await make_stream(dial=dial)
        mock_event_ch(stream)

        # State a failed attempt leaves behind: audio consumed, no finals,
        # input closed - and now uninspectable.
        stream.end_input()
        stream._audio_consumed = True
        stream._input_ch = RenamedQueueChan(stream._input_ch)

        async def scenario():
            await asyncio.sleep(0.05)
            ws.end()  # the server closes after Done, having sent nothing

        task = asyncio.create_task(scenario())
        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        assert dial.await_count == 1, (
            "an unverified probe must not decide the session's fate without "
            "attempting the exchange"
        )
        assert ws.sent_text == ["Done"], "the session really was attempted"
        assert not stream._session_complete


class TestDefensiveResultProcessing:
    """
    The gateway can emit frames outside the transcript shape (e.g. a pub/sub
    redirect frame). No frame the server sends may crash the receive loop.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            "just a string",
            ["an", "array"],
            42,
            3.14,
            None,
            True,
        ],
    )
    async def test_non_dict_json_is_ignored(self, payload):
        stream = await make_stream()
        stream._event_ch = Mock()

        await stream._process_result(payload)  # must not raise

        stream._event_ch.send_nowait.assert_not_called()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "payload",
        [
            {"type": "final", "text": None},
            {"type": "partial", "text": None},
            {"type": "final", "text": 123},
            {"type": "final"},
            {"text": None},
        ],
    )
    async def test_non_string_text_is_treated_as_absent(self, payload):
        stream = await make_stream()
        stream._event_ch = Mock()

        await stream._process_result(payload)  # must not raise

        stream._event_ch.send_nowait.assert_not_called()
        assert not stream._final_received

    @pytest.mark.asyncio
    async def test_redirect_frame_takes_the_unknown_type_path(self, caplog):
        stream = await make_stream()
        stream._event_ch = Mock()

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await stream._process_result(
                {"type": "redirect", "target": "wss://other-node"}
            )

        assert any("unknown message type" in r.message for r in caplog.records)

    @pytest.mark.parametrize(
        "frame_obj",
        [
            # The gateway's redirect target can land in "text"
            {"type": "redirect", "target": "wss://node2", "text": "node2"},
            {"type": "some-future-frame", "text": "not a transcript"},
            {"text": "no type at all"},
        ],
    )
    @pytest.mark.asyncio
    async def test_non_transcript_frame_does_not_open_a_speech_turn(
        self, frame_obj
    ):
        """
        START_OF_SPEECH must follow frame MEANING, not the presence of "text".

        A "text" key is a shape. Latching _speaking on it opened a speech turn
        for frames that carry no transcript, and nothing closes that turn
        until session teardown - the caller waits for an END_OF_SPEECH that
        never comes mid-session.
        """
        stream = await make_stream()
        stream._event_ch = Mock()

        await stream._process_result(frame_obj)

        assert not stream._speaking, (
            f"{frame_obj!r} carries no transcript and must not start speech"
        )
        assert stream._event_ch.send_nowait.call_count == 0

    @pytest.mark.asyncio
    async def test_transcript_frame_does_open_a_speech_turn(self):
        """The other half: a real transcript still starts the turn."""
        stream = await make_stream()
        stream._event_ch = Mock()

        await stream._process_result({"type": "partial", "text": "bonjour"})

        assert stream._speaking
        emitted = [
            c.args[0].type for c in stream._event_ch.send_nowait.call_args_list
        ]
        assert SpeechEventType.START_OF_SPEECH in emitted

    @pytest.mark.asyncio
    async def test_error_frame_still_raises(self):
        stream = await make_stream()
        stream._event_ch = Mock()

        with pytest.raises(APIConnectionError, match="Voxist error"):
            await stream._process_result({"type": "error", "message": "boom"})


class TestSegmentEndSilence:
    """
    flush() from a VAD-gated caller (which pushes only speech frames) must
    still yield a final: the engine endpoints on ~300ms of wire silence that
    nobody else provides. _on_segment_end synthesizes it - but ONLY at
    genuine mid-session segment boundaries. end_input() is flush() + close(),
    so every session's last item is a sentinel: injecting the silence there
    shipped 400ms of zeros before every Done (Done itself forces the engine
    flush), pure dead air at the end of every turn.
    """

    WIRE_CHUNK_BYTES = VoxistSTTStream.WIRE_SAMPLE_RATE // 10 * 2  # 100ms Int16

    @classmethod
    def silence_payloads(cls, ws):
        return [
            c.args[0]
            for c in ws.send_bytes.await_args_list
            if len(c.args[0]) == cls.WIRE_CHUNK_BYTES
            and c.args[0] == b"\x00" * cls.WIRE_CHUNK_BYTES
        ]

    @pytest.mark.asyncio
    async def test_mid_session_flush_sends_silence_at_wire_rate(self):
        """A sentinel with the channel still OPEN is a segment boundary."""
        stream = await make_stream()
        ws = attach_mock_ws(stream)

        # Real mid-session state: the sentinel is consumed while the input
        # channel is still open (the session continues afterwards).
        stream._input_ch.send_nowait(VoxistSTTStream._FlushSentinel())
        task = asyncio.create_task(stream._send_audio_task())
        await asyncio.sleep(0.2)  # sentinel consumed, channel still open

        total = b"".join(c.args[0] for c in ws.send_bytes.await_args_list)
        expected_bytes = int(
            VoxistSTTStream.SEGMENT_SILENCE_SECONDS
            * VoxistSTTStream.WIRE_SAMPLE_RATE
            * 2  # Int16
        )
        assert len(total) == expected_bytes, (
            "a mid-session boundary must put the full endpointing silence "
            "on the wire"
        )
        assert total == b"\x00" * expected_bytes, "the filler must be silence"
        # and the silence must be >= the engine's ~300ms endpointing need
        assert VoxistSTTStream.SEGMENT_SILENCE_SECONDS >= 0.3

        stream._input_ch.close()
        await asyncio.wait_for(task, timeout=5.0)

    @pytest.mark.asyncio
    async def test_silence_bypasses_the_audio_processor(self):
        """Silence is wire-rate by construction; resampling it would change
        its duration. It must not pass through the AudioProcessor."""
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        stream._audio_processor = Mock()
        stream._audio_processor.flush = Mock(return_value=[])
        stream._audio_processor.process_audio_frame = Mock(return_value=[])

        stream._input_ch.send_nowait(VoxistSTTStream._FlushSentinel())
        task = asyncio.create_task(stream._send_audio_task())
        await asyncio.sleep(0.2)  # mid-session: channel still open

        assert ws.send_bytes.await_count > 0
        stream._audio_processor.process_audio_frame.assert_not_called()

        stream._input_ch.close()
        await asyncio.wait_for(task, timeout=5.0)

    @pytest.mark.asyncio
    async def test_end_of_session_sends_no_silence_before_done(self):
        """
        The trailing sentinel end_input() pushes is END OF SESSION, not a
        segment boundary: Done forces the engine flush on its own, so the
        400ms of zeros the old code shipped there was pure added latency on
        every single turn (and a spurious 4-chunk dial for empty sessions).
        """
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        mock_event_ch(stream)  # the cancelled _task closed the real one

        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()  # REAL path: flush() + close()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        assert stream._done_sent
        assert ws.send_str.await_args.args[0] == "Done"
        assert ws.send_bytes.await_count > 0, "the speech itself must ship"
        assert self.silence_payloads(ws) == [], (
            "end_input() must not inject endpointing silence before Done"
        )

    @pytest.mark.asyncio
    async def test_flush_immediately_followed_by_end_input_sends_no_silence(self):
        """
        THE commonest VAD pattern of all, and it was untested: turn
        detection calls flush() at end of speech and the caller ends the
        session in the same breath, so the channel holds TWO ADJACENT
        sentinels with no frames between them.

        The end-of-session check keyed on qsize()==0, so pulling the FIRST
        sentinel saw qsize()==1 and treated it as a mid-session boundary:
        400ms of endpointing silence in front of a "Done" that forces the
        engine flush by itself. Pure dead air on every single turn ending
        this way. One predicate for both callers is what makes it
        impossible: whatever remains is sentinels, so this is end of
        session.
        """
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        mock_event_ch(stream)

        stream._input_ch.send_nowait(speech_frame())
        stream.flush()       # REAL path: end of segment...
        stream.end_input()   # ...and end of session, back to back

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        assert self.silence_payloads(ws) == [], (
            "two adjacent sentinels are one session ending, not a segment "
            "boundary: injecting silence before Done is pure dead air"
        )
        assert ws.send_bytes.await_count > 0, "the speech itself must ship"
        assert ws.send_str.await_count == 1, "exactly one Done"
        assert ws.send_str.await_args.args[0] == "Done"

    @pytest.mark.asyncio
    async def test_end_of_session_detection_uses_the_shared_predicate(self):
        """
        The send loop and the exhausted-input guard must ask the SAME
        question. They diverged for a round - the guard used the predicate,
        the send loop still tested qsize() - and that divergence is the
        two-sentinel dead-air bug above.
        """
        import inspect

        src = inspect.getsource(VoxistSTTStream._send_audio_task)
        assert "_pending_input_only_sentinels" in src
        # Comments describe the old check on purpose; only code counts.
        code = "\n".join(
            line for line in src.splitlines() if not line.strip().startswith("#")
        )
        assert "qsize() == 0" not in code, (
            "a second expression for 'no audio remains' is how the two "
            "callers drifted apart"
        )

    @pytest.mark.asyncio
    async def test_mid_session_flush_before_end_input_still_injects(self):
        """
        The [11] fix must not overreach: a genuine flush() boundary inside
        the session keeps its silence even when end_input() follows later.
        """
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        mock_event_ch(stream)

        stream._input_ch.send_nowait(speech_frame())
        stream.flush()  # REAL mid-session boundary
        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()  # REAL session end

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        expected_chunks = int(VoxistSTTStream.SEGMENT_SILENCE_SECONDS * 10)
        assert len(self.silence_payloads(ws)) == expected_chunks, (
            "exactly one boundary's worth of silence: the flush() keeps "
            "its injection, the end_input() sentinel adds none"
        )
        assert stream._done_sent


class TestServerStallDetection:
    """
    A mute-but-connected server (wedged engine behind a live gateway that
    still answers pings) must be detected MID-session, not discovered as
    zero transcripts at end_input.

    The detector counts CALLER-AUDIO BYTES delivered since the last message
    from the server, not wall time since a loud frame. Three defects died
    with the wall clock: the latch was never cleared by later silence (a
    cough then 35s of quiet killed a healthy session), quiet speakers never
    armed it at all (their wedged servers went undetected), and it copied
    every frame to int32 on the per-frame hot path.
    """

    @pytest.mark.asyncio
    async def test_stall_bound_constants_are_sane(self):
        assert 10.0 <= VoxistSTTStream.STALL_DETECTION_SECONDS <= 60.0, (
            "short enough to save a live call, long enough that a loaded "
            "engine's slowest partial cannot false-trigger"
        )
        assert not hasattr(VoxistSTTStream, "NON_SILENCE_AMPLITUDE"), (
            "an amplitude gate cannot hear a quiet speaker, so it silently "
            "exempts them from stall detection; the bound is measured in "
            "delivered audio bytes instead"
        )

    @pytest.mark.asyncio
    async def test_bound_is_the_byte_equivalent_of_the_second_bound(self):
        """One source of truth: the byte budget derives from the seconds."""
        stream = await make_stream()
        attach_mock_ws(stream)

        expected = int(
            VoxistSTTStream.STALL_DETECTION_SECONDS
            * VoxistSTTStream.WIRE_SAMPLE_RATE
            * 2  # Int16
        )
        # The wall-clock floor is a SECOND necessary condition; age the
        # attempt origin past it so this test isolates the byte bound.
        stream._attempt_started_at -= (
            VoxistSTTStream.STALL_DETECTION_SECONDS + 1.0
        )

        stream._bytes_sent_since_progress = expected
        stream._check_server_liveness()  # exactly at the bound: not a stall

        stream._bytes_sent_since_progress = expected + 1
        with pytest.raises(APIConnectionError, match="no response"):
            stream._check_server_liveness()

    @pytest.mark.asyncio
    async def test_mute_server_with_flowing_audio_raises(self, monkeypatch):
        # raising=False: reverting the fix removes the constant, and this
        # test must then fail on the missing BEHAVIOUR, not on monkeypatch
        monkeypatch.setattr(
            VoxistSTTStream, "STALL_DETECTION_SECONDS", 0.2, raising=False
        )
        ws = FakeWS()  # never feeds a single message
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        stopped = asyncio.Event()

        async def pump():
            # Real audio keeps flowing, as from a live microphone
            for _ in range(200):
                if stopped.is_set():
                    return
                stream._input_ch.send_nowait(speech_frame(320))  # 20ms
                await asyncio.sleep(0.02)

        pump_task = asyncio.create_task(pump())
        try:
            with pytest.raises(APIConnectionError, match="no response"):
                await asyncio.wait_for(stream._run(), timeout=5.0)
        finally:
            stopped.set()
            await pump_task
        assert not stream._session_complete

    @pytest.mark.asyncio
    async def test_quiet_speaker_wedge_is_detected(self, monkeypatch):
        """
        THE quiet-speaker test. This audio's peak (60) sits far below the
        amplitude gate the detector used to arm on (500), so the old clock
        never started and the wedge was only discovered at end_input - by
        which point the audio was unreplayable and the session died
        terminally instead of retrying mid-call. Byte accounting does not
        care how loud the speaker is.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "STALL_DETECTION_SECONDS", 0.2, raising=False
        )
        ws = FakeWS()  # wedged engine: never answers, never closes
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        stopped = asyncio.Event()

        async def pump():
            for _ in range(200):
                if stopped.is_set():
                    return
                stream._input_ch.send_nowait(quiet_speech_frame(320))
                await asyncio.sleep(0.02)

        pump_task = asyncio.create_task(pump())
        try:
            with pytest.raises(APIConnectionError, match="no response"):
                await asyncio.wait_for(stream._run(), timeout=5.0)
        finally:
            stopped.set()
            await pump_task
        assert not stream._session_complete

    @pytest.mark.asyncio
    async def test_silent_user_never_trips_the_detector(self, monkeypatch):
        """
        A user who says nothing sends nothing: no bytes reach the server, so
        the bound cannot be reached however long the silence lasts. The old
        unconditional 30s receive watchdog false-fired on exactly this.
        """
        monkeypatch.setattr(VoxistSTTStream, "STALL_DETECTION_SECONDS", 0.0)
        ws = FakeWS()  # never answers
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        run_task = asyncio.create_task(stream._run())
        await asyncio.sleep(0.3)  # a "long" user silence at this scale

        assert not run_task.done(), (
            "a silent user must not be able to trip the liveness bound"
        )
        assert stream._bytes_sent_since_progress == 0

        run_task.cancel()
        with pytest.raises(asyncio.CancelledError):
            await run_task

    @pytest.mark.asyncio
    async def test_injected_endpointing_silence_is_not_counted(self, monkeypatch):
        """
        The endpointing silence is OUR audio, synthesized to make the engine
        finalize a segment. Counting it would let the plugin declare the
        server stalled because of bytes the caller never sent - and with a
        stalling uplink, 20s of injected zeros would do it.
        """
        monkeypatch.setattr(VoxistSTTStream, "STALL_DETECTION_SECONDS", 0.0)
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        # No caller bytes at all: every chunk sent below is injected silence.
        stream._audio_processor = Mock()
        stream._audio_processor.flush = Mock(return_value=[])
        stream._audio_processor.process_audio_frame = Mock(return_value=[])

        stream._input_ch.send_nowait(VoxistSTTStream._FlushSentinel())
        stream._input_ch.send_nowait(frame(1600))
        task = asyncio.create_task(stream._send_audio_task())
        await asyncio.sleep(0.2)  # mid-session boundary: silence is injected

        assert ws.send_bytes.await_count > 0, "the silence must reach the wire"
        assert stream._bytes_sent_since_progress == 0, (
            "injected endpointing silence must not move the liveness bound"
        )
        assert not task.done(), "and it must certainly not trip it"

        stream._input_ch.close()
        await asyncio.wait_for(task, timeout=5.0)

    @pytest.mark.asyncio
    async def test_caller_audio_is_counted(self):
        """The other half of the contract: the caller's audio does count."""
        stream = await make_stream()
        attach_mock_ws(stream)

        chunk = np.zeros(1600, dtype=np.int16)  # 3200B
        await stream._send_audio_chunk(chunk)
        assert stream._bytes_sent_since_progress == 3200
        await stream._send_audio_chunk(chunk)
        assert stream._bytes_sent_since_progress == 6400

    @pytest.mark.asyncio
    async def test_transcript_frame_resets_the_byte_budget(self):
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._ws = ws
        stream._bytes_sent_since_progress = 900_000  # nearly at the bound

        ws.feed_json({"type": "partial", "text": "bonjour"})
        ws.end()
        await asyncio.wait_for(stream._recv_results_task(), timeout=5.0)

        assert stream._bytes_sent_since_progress == 0, (
            "a transcript proves the engine is working and must reset the "
            "budget"
        )
        assert stream._last_progress_at is not None

    @pytest.mark.parametrize(
        "frame_obj,resets",
        [
            # Engine transcripts: the only proof the ASR is consuming audio.
            ({"type": "partial", "text": "bonjour"}, True),
            ({"type": "final", "text": "bonjour"}, True),
            # A segment the engine finalized as silence is still engine work.
            ({"type": "final", "text": ""}, True),
            # The gateway's own pub/sub control frame - says nothing about
            # the engine.
            ({"type": "redirect", "url": "wss://elsewhere"}, False),
            ({"type": "some-future-frame"}, False),
            ({"text": "no type at all"}, False),
            ("a bare JSON string", False),
            (None, False),
        ],
    )
    @pytest.mark.asyncio
    async def test_only_transcript_frames_count_as_progress(
        self, frame_obj, resets
    ):
        """
        The stall budget must be cleared by transcripts, not by traffic.

        Resetting on any frame (the shipped behaviour this replaced) let a
        gateway that emits control frames on a timer hold the budget at zero
        forever, disarming the detector precisely in the case it exists for:
        WS layer healthy, engine wedged.
        """
        stream = await make_stream()
        mock_event_ch(stream)
        stream._bytes_sent_since_progress = 900_000

        await stream._process_result(frame_obj)

        if resets:
            assert stream._bytes_sent_since_progress == 0
            assert stream._last_progress_at is not None
        else:
            assert stream._bytes_sent_since_progress == 900_000, (
                f"{frame_obj!r} is not evidence the engine is transcribing "
                "and must not buy the server another stall window"
            )
            assert stream._last_progress_at is None

    @pytest.mark.asyncio
    async def test_unparseable_frame_does_not_count_as_progress(self):
        """Garbage on the wire is not liveness either."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._ws = ws
        stream._bytes_sent_since_progress = 900_000

        ws.incoming.put_nowait(
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="{not json")
        )
        ws.end()
        await asyncio.wait_for(stream._recv_results_task(), timeout=5.0)

        assert stream._bytes_sent_since_progress == 900_000
        assert stream._last_progress_at is None

    @pytest.mark.asyncio
    async def test_chatty_gateway_does_not_disarm_the_stall_detector(
        self, monkeypatch
    ):
        """
        End-to-end: a server that talks but never transcribes still trips.

        This is the regression that matters. The detector was armed by
        transcripts but disarmed by ANY frame, so a gateway emitting redirect
        frames faster than the byte bound accumulates - a plausible pub/sub
        keepalive pattern - meant a wedged engine went undetected for the
        whole session, delivering silent zero-transcripts to the caller.
        """
        # raising=False so reverting the fix fails on BEHAVIOUR, not setattr
        monkeypatch.setattr(
            VoxistSTTStream, "STALL_DETECTION_SECONDS", 0.2, raising=False
        )
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        stopped = asyncio.Event()

        async def pump():
            """Real audio keeps flowing, as from a live microphone."""
            for _ in range(200):
                if stopped.is_set():
                    return
                stream._input_ch.send_nowait(speech_frame(320))  # 20ms
                await asyncio.sleep(0.02)

        async def chatter():
            """The gateway is talkative but the engine never transcribes."""
            while not stopped.is_set():
                ws.feed_json({"type": "redirect", "url": "wss://elsewhere"})
                await asyncio.sleep(0.05)

        pump_task = asyncio.create_task(pump())
        chatter_task = asyncio.create_task(chatter())
        try:
            with pytest.raises(APIConnectionError, match="no response"):
                await asyncio.wait_for(stream._run(), timeout=5.0)
        finally:
            stopped.set()
            await pump_task
            chatter_task.cancel()
        assert not stream._session_complete

    @pytest.mark.asyncio
    async def test_no_per_frame_int32_copy_on_the_hot_path(self):
        """
        The old detector built an int32 copy of every frame to compute a
        peak. Nothing on the per-frame path may promote dtypes again.
        """
        import inspect

        hot_path = "".join(
            inspect.getsource(fn)
            for fn in (
                VoxistSTTStream._send_audio_task,
                VoxistSTTStream._carries_signal,
                VoxistSTTStream._check_server_liveness,
                VoxistSTTStream._send_audio_chunk,
            )
        )
        assert "astype" not in hot_path

    @pytest.mark.asyncio
    async def test_no_watchdog_task_exists(self):
        """
        The check is evaluated inline in the send loop; a third background
        task would need the same cancellation care as send/recv and has no
        payoff. Guard against one creeping back in.
        """
        import inspect

        src = inspect.getsource(VoxistSTTStream._run) + inspect.getsource(
            VoxistSTTStream._run_attempt
        )
        assert src.count("create_task") == 2, (
            "an attempt must own exactly two children: send and recv"
        )


class TestBoundedClose:
    """
    aiohttp's ws.close() waits up to 10s for the peer's close ACK. A wedged
    server already cost the 5s drain bound; gifting it another 10s of dead
    air in the finally defeats that. Teardown must be bounded.
    """

    @pytest.mark.asyncio
    async def test_failed_attempt_teardown_is_bounded(self, monkeypatch):
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.2)
        monkeypatch.setattr(VoxistSTTStream, "CLOSE_TIMEOUT_SECONDS", 0.2)
        ws = FakeWS()
        ws.hang_on_close = True  # wedged peer never completes the handshake
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()

        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        elapsed = loop.time() - start

        assert elapsed < 2.0, (
            f"teardown took {elapsed:.1f}s: the close was not bounded and "
            "the wedged peer bought itself extra dead air"
        )

    @pytest.mark.asyncio
    async def test_failed_attempt_aborts_transport_before_close(self, monkeypatch):
        """On a failed attempt, the transport is aborted, not flushed."""
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.2)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        transport = Mock()
        transport.is_closing = Mock(return_value=False)
        transport.abort = Mock()
        response = Mock()
        response.connection = Mock()
        response.connection.transport = transport
        ws._response = response

        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()

        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        transport.abort.assert_called()

    @pytest.mark.asyncio
    async def test_clean_completion_still_closes_politely(self):
        """The clean path keeps the polite (but bounded) close."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "bonjour", "confidence": 0.9})
            await asyncio.sleep(0.05)
            stream.end_input()
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=5.0)
        await task
        assert stream._session_complete


class TestConfidenceHardening:
    """
    "confidence" is server-supplied; data.get(key, default) only covers a
    MISSING key. null or junk must default to 1.0, not flow into SpeechData.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("msg_type", ["partial", "final"])
    @pytest.mark.parametrize(
        "confidence",
        [None, "0.9", "high", [], {}, True, False],
    )
    async def test_non_numeric_confidence_defaults(self, msg_type, confidence):
        stream = await make_stream()
        events = mock_event_ch(stream)

        await stream._process_result(
            {"type": msg_type, "text": "bonjour", "confidence": confidence}
        )

        data = [
            c.args[0].alternatives[0]
            for c in events.send_nowait.call_args_list
            if c.args[0].alternatives
        ]
        assert data, "the transcript event must still be emitted"
        assert data[0].confidence == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("msg_type", ["partial", "final"])
    async def test_missing_confidence_defaults(self, msg_type):
        stream = await make_stream()
        events = mock_event_ch(stream)

        await stream._process_result({"type": msg_type, "text": "bonjour"})

        data = [
            c.args[0].alternatives[0]
            for c in events.send_nowait.call_args_list
            if c.args[0].alternatives
        ]
        assert data and data[0].confidence == 1.0

    @pytest.mark.asyncio
    @pytest.mark.parametrize("msg_type", ["partial", "final"])
    @pytest.mark.parametrize("confidence", [0.42, 1, 0])
    async def test_numeric_confidence_passes_through(self, msg_type, confidence):
        stream = await make_stream()
        events = mock_event_ch(stream)

        await stream._process_result(
            {"type": msg_type, "text": "bonjour", "confidence": confidence}
        )

        data = [
            c.args[0].alternatives[0]
            for c in events.send_nowait.call_args_list
            if c.args[0].alternatives
        ]
        assert data and data[0].confidence == float(confidence)


class TestFinallyCancellationSemantics:
    """Findings on _run's finally: outer cancellation must propagate, and
    every child exception must be retrieved."""

    @pytest.mark.asyncio
    async def test_outer_cancellation_not_swallowed_by_child_cleanup(self):
        """
        aclose() cancels _main_task while _run's finally awaits a slow
        child. The old `with suppress(CancelledError): await task` pattern
        ate the OUTER cancellation there, so _run returned normally from a
        cancelled task.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        stream._event_ch = Mock()

        cleanup_started = asyncio.Event()

        async def stubborn_send():
            try:
                await asyncio.Event().wait()
            except asyncio.CancelledError:
                cleanup_started.set()
                await asyncio.sleep(0.5)  # slow, cancellation-resistant cleanup
                raise

        stream._send_audio_task = stubborn_send

        stream._input_ch.send_nowait(frame(1600))
        ws.end()  # recv finishes -> _run raises -> finally cancels send

        run_task = asyncio.create_task(stream._run())
        await asyncio.wait_for(cleanup_started.wait(), timeout=5.0)
        run_task.cancel()  # what aclose() does, mid-finally

        with pytest.raises(asyncio.CancelledError):
            await asyncio.wait_for(run_task, timeout=5.0)
        assert run_task.cancelled(), (
            "the outer cancellation was swallowed by child cleanup"
        )
        # let the stubborn child finish its cleanup before the loop closes
        await asyncio.sleep(0.6)

    @pytest.mark.asyncio
    async def test_secondary_exception_is_retrieved(self, caplog):
        """
        When send and recv fail in the same wake, the unraised one must be
        retrieved (visible as the debug log) instead of surfacing at GC as
        'Task exception was never retrieved' on every retry.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        stream._event_ch = Mock()

        # Both fail before their first await, so both are already done in
        # the same FIRST_COMPLETED wake - the exact double-failure race.
        async def failing_send():
            raise ConnectionResetError("send died")

        async def failing_recv():
            raise ConnectionResetError("recv died")

        stream._send_audio_task = failing_send
        stream._recv_results_task = failing_recv

        with caplog.at_level(logging.DEBUG, logger="livekit.plugins.voxist"):
            with pytest.raises(ConnectionResetError):
                await asyncio.wait_for(stream._run(), timeout=5.0)

        secondary = [
            r for r in caplog.records if "secondary task failure" in r.message
        ]
        assert len(secondary) == 1, (
            "exactly one of the two simultaneous failures is the primary; "
            "the other must be retrieved and logged"
        )


class TestTransportAccessorAgainstRealAiohttp:
    """Guard the private attribute chain used to read the write buffer."""

    @pytest.mark.asyncio
    async def test_public_accessor_is_preferred_when_available(self):
        """If aiohttp ever ships get_transport(), it must win over the chain."""
        stream = await make_stream()

        sentinel = Mock()
        sentinel.is_closing = Mock(return_value=False)
        sentinel.get_write_buffer_size = Mock(return_value=4242)

        ws = Mock()
        ws.closed = False
        ws.get_transport = Mock(return_value=sentinel)
        other = Mock()
        other.is_closing = Mock(return_value=False)
        other.get_write_buffer_size = Mock(return_value=1)
        ws._response = Mock()
        ws._response.connection = Mock()
        ws._response.connection.transport = other
        stream._ws = ws

        assert stream._get_write_buffer_size() == 4242

    @pytest.mark.asyncio
    async def test_reads_real_transport_buffer_size(self):
        """The chain must resolve on a live aiohttp WebSocket."""
        from aiohttp import web

        async def handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for _ in ws:
                pass
            return ws

        app = web.Application()
        app.router.add_get("/ws", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    stream = await make_stream()
                    stream._ws = ws

                    assert stream._get_transport() is not None, (
                        "ws._response.connection.transport has moved in aiohttp"
                    )
                    assert isinstance(stream._get_write_buffer_size(), int)

                    await stream._send_audio_chunk(np.zeros(1600, dtype=np.int16))
        finally:
            await runner.cleanup()


class TestLanguageCodeHandling:
    """
    The dialed language is raw; the emitted language is livekit-normalized.

    With one socket per stream, the engine language is correct by
    construction - the raw code rides this stream's own dial URL.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", ["fr", "fr-medical", "fr-FR", "en-US", "nl"])
    async def test_raw_language_preserved_for_the_dial(self, language):
        stream = await make_stream(language=language)
        assert stream._language == language

    @pytest.mark.asyncio
    async def test_medical_language_normalized_for_emitted_events(self):
        stream = await make_stream(language="fr-medical")
        assert str(stream._speech_language) == "fr-MEDICAL"
        assert stream._language == "fr-medical"

    @pytest.mark.asyncio
    async def test_run_dials_with_the_raw_language(self):
        """What reaches the wire is the exact code the caller chose."""
        ws = FakeWS()
        dial = AsyncMock(return_value=ws)
        stream = await make_stream(language="fr-medical", dial=dial)
        mock_event_ch(stream)

        # A session with audio: a zero-audio end_input() would (correctly)
        # never dial at all. A final must arrive too - a server that closes
        # after Done having sent NOTHING for real audio is transcript loss,
        # not a completed session, and the gate now says so.
        stream._input_ch.send_nowait(speech_frame())
        stream.end_input()
        ws_task = asyncio.create_task(stream._run())
        await asyncio.sleep(0.05)
        ws.feed_json({"type": "final", "text": "bonjour"})
        await asyncio.sleep(0.05)
        ws.end()
        await asyncio.wait_for(ws_task, timeout=5.0)

        dial.assert_awaited_once_with("fr-medical")

    @pytest.mark.asyncio
    async def test_emitted_event_carries_normalized_language(self):
        stream = await make_stream(language="fr-medical")
        stream._event_ch = Mock()

        await stream._process_result({"type": "final", "text": "bonjour"})

        languages = [
            c.args[0].alternatives[0].language
            for c in stream._event_ch.send_nowait.call_args_list
            if c.args[0].alternatives
        ]
        assert languages and all(str(x).lower() == "fr-medical" for x in languages)


class TestStallDetectorFalsePositives:
    """
    The detector must never abort a HEALTHY attempt.

    Its accusation ("we gave the engine speech and it said nothing") costs
    more than it saves when wrong: RecognizeStream._num_retries is set once in
    __init__ and never reset after a successful stretch, so every spurious
    APIConnectionError is permanently deducted from the stream's lifetime
    budget and the fourth one leaves the agent deaf for the rest of the call.
    A missed accusation only defers the same error to the completion gate at
    session end. Hence two independent necessary conditions, each tested here.
    """

    @staticmethod
    def _budget() -> int:
        return int(
            VoxistSTTStream.STALL_DETECTION_SECONDS
            * VoxistSTTStream.WIRE_SAMPLE_RATE
            * 2  # Int16
        )

    @pytest.mark.asyncio
    async def test_a_long_pause_punctuated_by_the_engine_never_trips(self):
        """
        The mechanism that makes a resettable budget safe, exercised directly.

        Measured live: the engine emits a partial/final pair roughly every
        0.67s even on input carrying no speech - 37 frames across 24s. Each
        one clears the budget, so a healthy engine cannot accumulate it however
        long the user stays quiet.

        The previous version of this test aged `_attempt_started_at`, which
        `_check_server_liveness` stops reading the moment `_last_progress_at`
        is set, so it could not fail for its stated reason. This ages BOTH
        clocks so that only the budget reset can be what prevents the raise:
        with the reset removed the budget accumulates across iterations while
        the aged clock keeps the wall-time floor satisfied, and it fires.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._ws = ws

        budget = self._budget()
        loud = np.full(1600, 8000, dtype=np.int16)
        # 80% of a budget per round, so four rounds stream >3x the bound.
        chunks_per_round = int(budget * 0.8 / loud.nbytes)
        total = 0

        for _ in range(4):
            # A FIXED amount per iteration, not a loop on the budget: with
            # the reset removed, a budget-conditioned loop sends fewer bytes
            # each round and the test then fails on its own bookkeeping
            # instead of on the detector firing.
            for _ in range(chunks_per_round):
                await stream._send_audio_chunk(loud)
                total += loud.nbytes

            # The engine punctuates the silence, as it does every ~0.67s.
            await stream._process_result({"type": "final", "text": ""})

            # Age both clocks AFTER the transcript, not before. Ageing first
            # let the transcript refresh the wall-clock origin and it was then
            # the clock, not the budget reset, that prevented the raise - so
            # the test passed with the reset removed.
            stream._attempt_started_at -= 60.0
            assert stream._last_progress_at is not None
            stream._last_progress_at -= 60.0

            stream._check_server_liveness()  # must never raise

        assert total > budget * 3, "must stream well past the bound"

    @pytest.mark.asyncio
    async def test_quiet_audio_still_arms_it_before_any_transcript(self):
        """No amplitude gate: an under-gained speaker must still be protected."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._ws = ws

        quiet = np.full(1600, 60, dtype=np.int16)  # peak 60, as the fixture
        while stream._bytes_sent_since_progress <= self._budget():
            await stream._send_audio_chunk(quiet)
        stream._attempt_started_at -= (
            VoxistSTTStream.STALL_DETECTION_SECONDS + 1.0
        )

        with pytest.raises(APIConnectionError, match="no response"):
            stream._check_server_liveness()

    @pytest.mark.asyncio
    async def test_bytes_alone_do_not_trip_it_wall_time_is_also_required(self):
        """
        Finding: a faster-than-real-time sender reached 30s of audio in
        seconds.

        Batch/file callers and a live caller's catch-up burst after a network
        hiccup both do this. Tripping there aborted a healthy attempt whose
        retry then found the input consumed and escalated to a terminal
        TranscriptLostError, destroying the session outright.

        Both halves are asserted here so neither condition can be dropped
        without a failure.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._ws = ws

        loud = np.full(1600, 8000, dtype=np.int16)
        while stream._bytes_sent_since_progress <= self._budget():
            await stream._send_audio_chunk(loud)

        assert stream._bytes_sent_since_progress > self._budget()
        assert stream._last_progress_at is None, "detector must still be armed"
        # Byte budget blown, but no wall time has passed: healthy fast sender.
        stream._check_server_liveness()

        # Same byte state, but now the engine really has been mute that long.
        stream._attempt_started_at -= (
            VoxistSTTStream.STALL_DETECTION_SECONDS + 1.0
        )
        with pytest.raises(APIConnectionError, match="no response"):
            stream._check_server_liveness()


class _OverloadedChannel:
    """
    A real channel that reports a pinned depth.

    A test must close the channel for the send loop to terminate, but closing
    it also empties it, so `qsize()` would fall below any interesting bound
    before the loop ran. Pinning the depth keeps these tests on the drop path.

    It pins ONLY the depth. An earlier version also forced `closed` to False,
    which silently inverted the send loop's end-of-session sentinel branch -
    the double was steering a code path production never takes. The drop
    predicate no longer consults `closed` at all, so there is nothing to
    override.
    """

    def __init__(self, real, depth):
        self._real = real
        self._depth = depth

    def qsize(self):
        return self._depth

    def __aiter__(self):
        return self._real.__aiter__()

    def __getattr__(self, name):
        return getattr(self._real, name)


class TestBacklogBoundIsHard:
    """
    A generous absolute ceiling on backlog GROWTH, and nothing else.

    Three versions tried to tell "a burst that will drain" from "a producer
    outpacing the uplink" - by depth, by direction, by `closed` - and each
    was right for one case and wrong for the other, because that intent is
    not observable from this side of the channel. The current version infers
    nothing; its two refinements (running-mean estimate, pre-drain
    exclusion) each fix a measured false positive and are pinned here.
    """

    @staticmethod
    def _chan(depth):
        return SimpleNamespace(qsize=lambda: depth, closed=False)

    @staticmethod
    def _seed(stream, *, depth, mean=0.01, samples=100, pre_drain=0, popped=0):
        """Put the stream in a known accounting state, no send loop needed."""
        stream._input_ch = SimpleNamespace(qsize=lambda: depth, closed=False)
        stream._pre_drain_items = pre_drain
        stream._items_popped = popped
        stream._frame_durations.clear()
        for _ in range(samples):
            stream._note_frame_duration(mean)

    @pytest.mark.asyncio
    async def test_over_the_bound_drops(self, monkeypatch):
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 1.0)
        stream = await make_stream()
        self._seed(stream, depth=101)  # 1.01s of 10ms frames, all growth
        assert stream._backlog_exceeds_the_bound()

    @pytest.mark.asyncio
    async def test_at_or_under_the_bound_never_drops(self, monkeypatch):
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 1.0)
        stream = await make_stream()
        # 99, not 100: the windowed mean accumulates float error (fifty
        # 0.01 additions sum to 0.500000...02), so exactly-at-the-bound is
        # not a stable boundary and the production comparison never needs it
        # to be - the ceiling is 120s with minutes of headroom.
        for depth in (0, 1, 50, 99):
            self._seed(stream, depth=depth)
            assert not stream._backlog_exceeds_the_bound()

    @pytest.mark.asyncio
    async def test_one_large_frame_does_not_forge_a_huge_backlog(
        self, monkeypatch
    ):
        """
        THE mixed-size regression. Estimating from the frame IN HAND made the
        error unbounded in the lossy direction: one ~11s frame (the size
        _iter_frame_slices exists for) tripped the bound at depth 12 - a
        depth any live pipeline hits during a scheduling hiccup - and the
        frame was discarded when the true backlog was a couple of seconds.
        The running mean of a 10ms stream barely notices one large frame.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 120.0
        )
        stream = await make_stream()
        self._seed(stream, depth=12, mean=0.01, samples=100)
        stream._note_frame_duration(10.9)  # the big frame passes through

        assert not stream._backlog_exceeds_the_bound(), (
            "one 10.9s frame in the window must not make 12 queued frames "
            "read as 130s of backlog"
        )
        # Sanity on the arithmetic the docstring claims: the old estimator
        # would have seen 12 * 10.9 = 130.8s and dropped.
        assert 12 * 10.9 > 120.0

    @pytest.mark.asyncio
    async def test_pre_drain_backlog_is_not_growth(self, monkeypatch):
        """
        THE rate-limiter-park regression. Audio queued while nothing could
        drain - a dial parked in the limiter for up to a full window, twice
        on the stale-token path - is not evidence of overload, and counting
        it made the send loop discard the start of the caller's speech after
        a throttled dial.
        """
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 0.5)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        # 200 frames of 100ms = 20s of audio, 40x the bound, ALL queued
        # before the send loop exists - exactly what a 20s park produces.
        for _ in range(200):
            stream._input_ch.send_nowait(speech_frame())

        async def scenario():
            await asyncio.sleep(0.3)
            ws.feed_json({"type": "final", "text": "bonjour"})
            await asyncio.sleep(0.1)
            stream._input_ch.close()
            await asyncio.sleep(0.1)
            ws.feed_json({"type": "final", "text": "bonjour"})
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=15.0)
        await task

        assert stream.dropped_frames == 0, (
            f"{stream.dropped_frames} frames of park-accumulated audio were "
            "discarded as if the uplink were overloaded"
        )

    @pytest.mark.asyncio
    async def test_blind_until_a_duration_has_been_measured(self):
        """With nothing to estimate from, guessing would drop audio."""
        stream = await make_stream()
        self._seed(stream, depth=10**6, samples=0)
        assert not stream._backlog_exceeds_the_bound()

    @pytest.mark.asyncio
    async def test_invalid_durations_are_not_measured(self):
        stream = await make_stream()
        self._seed(stream, depth=10, samples=0)
        for bad in (0.0, -1.0):
            stream._note_frame_duration(bad)
        assert len(stream._frame_durations) == 0

    @pytest.mark.asyncio
    async def test_repeated_calls_agree(self, monkeypatch):
        """The predicate reads its state; it must not consume it."""
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 1.0)
        stream = await make_stream()
        self._seed(stream, depth=500)
        assert [stream._backlog_exceeds_the_bound() for _ in range(5)] == [
            True
        ] * 5

    @pytest.mark.asyncio
    async def test_depth_stays_bounded_against_a_producer_beating_the_uplink(
        self, monkeypatch
    ):
        """
        End-to-end with a real slow socket - the property the bound exists
        for. The direction version passed its own unit tests while the real
        send loop let the channel grow to 9928 frames, because those tests
        fed it a scripted depth sequence a live loop never produces.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 0.5
        )  # 50 frames of 10ms
        cap_frames = 50

        class SlowWS(FakeWS):
            async def send_bytes(self, data):
                await asyncio.sleep(0.01)  # uplink far slower than realtime
                await super().send_bytes(data)

        ws = SlowWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        peak = 0
        stop = asyncio.Event()

        async def producer():
            nonlocal peak
            # BATCHED pushes, one sleep per 30 frames. The first version
            # slept 0.5ms per frame, and under machine load the per-
            # iteration event-loop overhead dominated that sleep: the
            # producer quietly stopped outpacing the uplink, no drops
            # occurred, and the test failed on its own premise guard - a
            # load-sensitive flake. Batching keeps the push/drain ratio ~6x
            # under any load, because both sides' sleeps stretch together
            # while the per-frame overhead is paid 30x less often.
            for _ in range(40):
                if stop.is_set():
                    return
                for _ in range(30):
                    stream._input_ch.send_nowait(speech_frame(160))
                peak = max(peak, stream._input_ch.qsize())
                await asyncio.sleep(0.005)

        async def finisher():
            await asyncio.sleep(0.6)
            ws.feed_json({"type": "final", "text": "bonjour"})
            await asyncio.sleep(0.6)
            stream._input_ch.close()
            await asyncio.sleep(0.3)
            ws.feed_json({"type": "final", "text": "bonjour"})
            await asyncio.sleep(0.05)
            ws.end()

        prod = asyncio.create_task(producer())
        fin = asyncio.create_task(finisher())
        # The bound is the premise, not whether this scripted session also
        # ends cleanly. Narrowed to the plugin's own errors, NOT bare
        # Exception: suppressing everything also swallowed the wait_for
        # TimeoutError, so a DEADLOCKED send loop passed this test.
        try:
            with contextlib.suppress(APIConnectionError, TranscriptLostError):
                await asyncio.wait_for(stream._run(), timeout=20.0)
        finally:
            stop.set()
            for t in (prod, fin):
                t.cancel()

        # x4: batch granularity can admit two or three 30-frame batches
        # during one stretched send, but a bound that engages LATE (a
        # 250-300 peak regression) must still go red - x6 would have let it
        # pass. The drop-share witness below is the second jaw of the same
        # clamp: a bound that drops only every other frame keeps the peak
        # low but sheds far less.
        assert peak <= cap_frames * 4, (
            f"backlog reached {peak} frames against a bound of ~{cap_frames}: "
            "the ceiling is not holding, so a long call would grow the "
            "channel until OOM"
        )
        assert stream.dropped_frames >= 700, (
            f"only {stream.dropped_frames} of 1200 frames were dropped "
            "against a 50-frame cap with a ~6x-overloaded uplink: the bound "
            "is shedding a fraction of what it must, which a bare "
            "dropped>0 guard could not see"
        )


class TestOversizedFrameIsSlicedNotRejected:
    """
    The processor's 1MB guard raised a bare ValueError - not an APIError - so
    it unwound _run without the completion gate running, leaving
    _session_complete False and killing the stream with a ValueError at the
    call site. Legitimate callers hit it: ~11s of 48kHz mono, or livekit's own
    AudioResampler emitting a large frame.
    """

    @pytest.mark.parametrize(
        "extra",
        [
            # A tail the processor would SILENTLY DISCARD as "too small".
            # The first version of this test used exactly 100 and asserted
            # losslessness only over the joined slices - which is trivially
            # true of any partition - so it demonstrated the discarded case
            # while proving nothing about it.
            2,
            100,
            MIN_FRAME_SIZE_BYTES - 2,
            MIN_FRAME_SIZE_BYTES,
            MIN_FRAME_SIZE_BYTES + 2,
            MAX_FRAME_SIZE_BYTES // 2,
            0,
        ],
    )
    def test_every_slice_survives_the_processor(self, extra):
        original = bytes(MAX_FRAME_SIZE_BYTES * 2 + extra)
        pieces = list(VoxistSTTStream._iter_frame_slices(original))

        assert all(len(p) <= MAX_FRAME_SIZE_BYTES for p in pieces)
        assert b"".join(pieces) == original, "slicing must not lose a sample"
        assert all(len(p) % 2 == 0 for p in pieces), "Int16 alignment"
        # The real bar: the processor must ACCEPT every piece. It rejects
        # anything under MIN_FRAME_SIZE_BYTES outright, so a short tail is
        # discarded and the joined-bytes assertion above never notices.
        assert all(len(p) >= MIN_FRAME_SIZE_BYTES for p in pieces), (
            f"a slice below {MIN_FRAME_SIZE_BYTES}B is dropped by "
            "_validate_frame, so its audio never reaches the engine"
        )

    def test_a_sliced_frame_reaches_the_processor_whole(self):
        """End-to-end through the processor, not just over the partition."""
        samples = MAX_FRAME_SIZE_BYTES + 50  # tail of 100 bytes as Int16
        audio = np.random.default_rng(3).integers(
            -20000, 20000, samples, dtype=np.int16
        )
        raw = audio.tobytes()
        assert len(raw) % MAX_FRAME_SIZE_BYTES < MIN_FRAME_SIZE_BYTES

        p = AudioProcessor(sample_rate=16000)
        accepted = 0
        for piece in VoxistSTTStream._iter_frame_slices(raw):
            before = p._available_samples()
            chunks = p.process_audio_frame(piece)
            consumed = sum(c.size for c in chunks) or 0
            if chunks or p._available_samples() != before or consumed:
                accepted += len(piece)

        assert accepted == len(raw), (
            f"{len(raw) - accepted}B of the frame was rejected outright"
        )

    def test_ordinary_frame_is_yielded_untouched(self):
        frame = bytes(3200)
        pieces = list(VoxistSTTStream._iter_frame_slices(frame))
        assert pieces == [frame]

    @pytest.mark.asyncio
    async def test_oversized_frame_reaches_the_wire_without_raising(self):
        """End-to-end: the audio is sent, not lost, and no ValueError escapes."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        # Int16, so 600k samples is ~1.2MB of bytes: over the 1MB limit.
        big = speech_frame(samples=600_000)
        frame_bytes = len(bytes(big.data))
        assert frame_bytes > MAX_FRAME_SIZE_BYTES, (
            f"frame must exceed the limit to exercise slicing, got {frame_bytes}B"
        )

        async def scenario():
            stream._input_ch.send_nowait(big)
            await asyncio.sleep(0.2)
            ws.feed_json({"type": "final", "text": "bonjour"})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # end_input
            await asyncio.sleep(0.05)
            ws.end()  # gateway closes after Done

        task = asyncio.create_task(scenario())
        # No ValueError escapes, and the audio actually went out.
        await asyncio.wait_for(stream._run(), timeout=30.0)
        await task

        assert sum(len(b) for b in ws.sent_bytes) >= frame_bytes * 0.99, (
            "the whole oversized frame must reach the wire"
        )
        assert stream.dropped_frames == 0


class TestDrainWaitsForAFinalNotAnyTranscript:
    """
    Kroko never closes after Done, so the drain ends the turn on the engine
    going quiet after answering. The terminator must therefore be a FINAL: a
    post-Done partial proves the engine is alive but not that it has finished,
    and ending the turn on one truncates the final still in flight.
    """

    @pytest.mark.asyncio
    async def test_a_post_done_partial_does_not_end_the_turn(self, monkeypatch):
        monkeypatch.setattr(
            VoxistSTTStream, "POST_FINAL_IDLE_SECONDS", 0.05, raising=False
        )
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 1.0, raising=False
        )
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # end_input -> Done
            await asyncio.sleep(0.1)
            # The engine is alive and still working: partials only.
            for _ in range(6):
                ws.feed_json({"type": "partial", "text": "bonj"})
                await asyncio.sleep(0.05)
            # Only now does it finalize.
            ws.feed_json({"type": "final", "text": "bonjour"})

        task = asyncio.create_task(scenario())
        try:
            await asyncio.wait_for(stream._run(), timeout=5.0)
        finally:
            task.cancel()

        finals = [
            c.args[0]
            for c in stream._event_ch.send_nowait.call_args_list
            if c.args[0].type == SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals, (
            "the drain ended the turn on a partial and truncated the final "
            "that was still coming"
        )
        assert finals[-1].alternatives[0].text == "bonjour"


class TestRoundNineRegressions:
    """
    The six defects round 9 found in round 8's own fixes.

    Every one is a case where a guard added to prevent a false positive
    created a false NEGATIVE - a session that failed silently, or a wedge that
    was never policed. They are grouped here because they share that shape.
    """

    @pytest.mark.asyncio
    async def test_a_mid_session_wedge_after_a_transcript_is_still_caught(
        self, monkeypatch
    ):
        """
        Permanently disarming the detector on the first transcript left the
        agent deaf for whole calls AND reported success.

        The engine answers once, then wedges. The old code returned on the
        first line of _check_server_liveness forever after, and the gate then
        took case 1 (delivered_final was set by that early final) and logged a
        WARNING while completing the stream successfully.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "STALL_DETECTION_SECONDS", 0.2, raising=False
        )
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        stopped = asyncio.Event()

        async def pump():
            # The engine answers early, then goes mute while audio flows.
            ws.feed_json({"type": "final", "text": "bonjour"})
            for _ in range(400):
                if stopped.is_set():
                    return
                stream._input_ch.send_nowait(speech_frame(320))
                await asyncio.sleep(0.01)

        task = asyncio.create_task(pump())
        try:
            with pytest.raises(APIConnectionError, match="no response"):
                await asyncio.wait_for(stream._run(), timeout=5.0)
        finally:
            stopped.set()
            task.cancel()
        assert not stream._session_complete, (
            "a wedged engine must not yield a completed session"
        )

    @pytest.mark.asyncio
    async def test_an_empty_final_then_a_wedge_is_never_a_clean_session(self):
        """
        The engine endpoints silence, so the opening second of every call
        produces {"type":"final","text":""}. One such frame must not certify a
        session in which the engine then died and swallowed real speech.

        What refuses it is `concluded`: a wedged engine neither closes nor
        answers, so the drain times out, and the gate's empty-result exemption
        requires a concluded exchange. Deliberately NOT a comparison against
        when Done was written - that was too strict (an engine that finalized
        before end_input and then had nothing more to say made a quiet
        participant fatal) and unsound (the two stamps are unsynchronized
        monotonic reads in different tasks).
        """
        ws = FakeWS()  # wedges: never answers again, never closes
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            # Opening silence, finalized as empty.
            ws.feed_json({"type": "final", "text": ""})
            await asyncio.sleep(0.05)
            # Then real speech the engine never transcribes.
            for _ in range(20):
                stream._input_ch.send_nowait(speech_frame())
                await asyncio.sleep(0.005)
            stream._input_ch.close()

        task = asyncio.create_task(scenario())
        try:
            with pytest.raises(TranscriptLostError):
                await asyncio.wait_for(stream._run(), timeout=15.0)
        finally:
            task.cancel()

        assert not stream._session_complete, (
            "an empty final from before the wedge must not certify the speech "
            "that followed it"
        )

    @pytest.mark.asyncio
    async def test_an_empty_final_before_done_still_earns_the_exemption(self):
        """
        The other side: requiring a POST-Done final made a quiet participant
        fatal whenever the engine finalized the trailing silence before
        end_input and then had nothing further to send.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            # Engine finalizes the segment as empty BEFORE Done is written.
            ws.feed_json({"type": "final", "text": ""})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.05)
            ws.end()  # gateway closes after Done: the exchange concluded

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=10.0)
        await task

        assert stream._session_complete, (
            "a concluded exchange whose engine reported nothing is an empty "
            "session, not a fatal one"
        )

    @pytest.mark.asyncio
    async def test_dropped_audio_denies_the_empty_exemption(self, monkeypatch):
        """
        The engine can only report on audio it RECEIVED. With frames discarded
        to bound the backlog, its "nothing" is not a verdict on the session -
        and without this the plugin's own loss was logged as "an empty session,
        not a lost one".
        """
        monkeypatch.setattr(
            VoxistSTTStream, "MAX_INPUT_BACKLOG_SECONDS", 0.001
        )
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            # Pushed AFTER the send loop is running: audio queued before the
            # loop starts is pre-drain and correctly exempt from the bound,
            # so a pre-filled channel would (rightly) drop nothing and this
            # test would lose its premise.
            await asyncio.sleep(0.05)
            for _ in range(40):
                stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream._input_ch.close()
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": ""})
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        try:
            with pytest.raises(TranscriptLostError):
                await asyncio.wait_for(stream._run(), timeout=10.0)
        finally:
            task.cancel()

        assert stream.dropped_frames > 0, "the premise needs a real drop"

    @pytest.mark.asyncio
    async def test_the_wall_clock_floor_does_not_count_dial_latency(self):
        """
        _attempt_started_at was stamped before the dial, so a rate-limiter
        park of up to a full window counted as engine muteness and the
        detector fired on the first burst of audio after a throttled dial.
        """
        ws = FakeWS()

        async def slow_dial(_language):
            await asyncio.sleep(0.3)  # stands in for a limiter park
            return ws

        stream = await make_stream(dial=AsyncMock(side_effect=slow_dial))
        mock_event_ch(stream)

        before = time.monotonic()

        async def scenario():
            await asyncio.sleep(0.4)
            ws.feed_json({"type": "final", "text": "bonjour"})
            await asyncio.sleep(0.05)
            stream._input_ch.close()
            await asyncio.sleep(0.05)
            ws.end()

        stream._input_ch.send_nowait(speech_frame())
        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=5.0)
        await task

        # The origin must post-date the dial, not the call to _run.
        assert stream._attempt_started_at >= before + 0.25, (
            "the stall clock started before the socket existed, so dial "
            "latency is charged to the engine"
        )

    @pytest.mark.asyncio
    async def test_a_late_post_done_final_concludes_instead_of_raising(
        self, monkeypatch
    ):
        """
        A final arriving inside the last POST_FINAL_IDLE_SECONDS of the drain
        bound was reported as "sent no transcript" - factually false - and the
        gate then raised the fatal TranscriptLostError on a session the engine
        had actually finalized.
        """
        # Deterministic by construction rather than by timing: an idle margin
        # LARGER than the drain bound makes the ordinary idle path
        # unreachable, so whichever moment the final lands, the deadline
        # branch is the one under test. The previous version slept 0.45s
        # against a 0.5s bound with a 0.4s margin - a ~50ms window that a
        # loaded CI box would miss in either direction, passing without ever
        # touching the branch.
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 1.0, raising=False
        )
        monkeypatch.setattr(
            VoxistSTTStream, "POST_FINAL_IDLE_SECONDS", 30.0, raising=False
        )
        ws = FakeWS()  # Kroko: never closes
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # Done
            # Answer inside the drain bound. The idle margin can never
            # elapse before the deadline, so this always exercises the
            # deadline branch.
            await asyncio.sleep(0.2)
            ws.feed_json({"type": "final", "text": ""})

        task = asyncio.create_task(scenario())
        try:
            await asyncio.wait_for(stream._run(), timeout=5.0)
        finally:
            task.cancel()

        assert stream._session_complete, (
            "the engine finalized the session; the drain must not call that "
            "a lost transcript"
        )


class TestEngineDoneAck:
    """
    The engine acks "Done" with a bare non-JSON "Done!" text frame.

    Verified live against api-asr.voxist.com (lang=fr): it arrives ~0.12s
    after Done, immediately behind the last final, and the socket then stays
    OPEN indefinitely (12s later it was still open). The API's own reference
    client skips the same frame (kroko/bench/asr_bench.py:103).
    """

    @pytest.mark.asyncio
    async def test_the_ack_is_not_logged_as_malformed(self, caplog):
        """It is protocol, not garbage - and it arrives once per session."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._ws = ws

        stream._done_sent = True  # the ack only counts once Done went out
        ws.incoming.put_nowait(
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
        )
        ws.end()
        with caplog.at_level(logging.DEBUG, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._recv_results_task(), timeout=5.0)

        assert not [
            r for r in caplog.records if "invalid JSON" in r.message
        ], "the engine's own ack must not be reported as malformed input"
        assert stream._engine_acked_done

    @pytest.mark.asyncio
    async def test_the_ack_is_not_credited_as_engine_progress(self):
        """It proves the engine finished, not that it transcribed anything."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._ws = ws
        stream._bytes_sent_since_progress = 900_000

        ws.incoming.put_nowait(
            SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
        )
        ws.end()
        await asyncio.wait_for(stream._recv_results_task(), timeout=5.0)

        assert stream._bytes_sent_since_progress == 900_000
        assert stream._last_final_at is None

    @pytest.mark.asyncio
    async def test_the_ack_does_not_shorten_the_idle_margin(self, monkeypatch):
        """
        The margin is UNIFORM. A shortened acked margin (0.2s) was tried and
        re-opened the two-final tail drop: the ack is a trivial echo that can
        land ahead of decode on a loaded pod, and 0.2s of quiet between two
        flush finals separated by decode time is not enough. The 0.5s
        calibration exists for exactly that inter-final gap, ack or no ack.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "POST_FINAL_IDLE_SECONDS", 1.0, raising=False
        )
        # The backstop is pushed far out so it cannot masquerade as the
        # margin path: round 14 showed that if the fed final slips in before
        # the drain snapshots its counter, the (then-5s) acked backstop
        # ended the drain at >= 0.8s too - and the assertion passed with the
        # margin code never executed. With an 8s backstop the upper bound
        # below separates the two exits unambiguously.
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 8.0, raising=False
        )
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        final_at = {}

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.1)
            ws.feed_json({"type": "final", "text": "bonjour", "segment": 0})
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )
            final_at["t"] = time.monotonic()

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=10.0)
        ended = time.monotonic()
        await task

        assert stream._session_complete
        elapsed_after_final = ended - final_at["t"]
        assert elapsed_after_final >= 0.8, (
            f"the drain ended {elapsed_after_final:.2f}s after the final: "
            "the ack shortened the idle margin again, and a second flush "
            "final separated by decode time would have been dropped"
        )
        assert elapsed_after_final < 4.0, (
            f"the drain took {elapsed_after_final:.2f}s after the final - "
            "that is the 8s backstop exit, not the idle margin, so this "
            "test never exercised the code it pins"
        )


class TestSegmentLifecycle:
    """
    The engine numbers its segments and increments per utterance - verified
    live on api-asr.voxist.com, where 12s of French produced segments 0/1/2
    with a final for each. That makes a lost trailing utterance DETECTABLE,
    which this plugin previously documented as impossible.
    """

    @pytest.mark.asyncio
    async def test_a_finalized_segment_is_not_reported_as_loss(self):
        stream = await make_stream()
        mock_event_ch(stream)
        for seg in (0, 1, 2):
            await stream._process_result(
                {"type": "partial", "text": "x", "segment": seg}
            )
            await stream._process_result(
                {"type": "final", "text": "x", "segment": seg}
            )
        assert not stream._trailing_segment_unfinalized

    @pytest.mark.asyncio
    async def test_an_opened_but_unfinalized_segment_is_detected(self):
        """The case the gate could previously only guess at."""
        stream = await make_stream()
        mock_event_ch(stream)
        await stream._process_result(
            {"type": "partial", "text": "bonjour", "segment": 0}
        )
        await stream._process_result(
            {"type": "final", "text": "bonjour", "segment": 0}
        )
        # The engine starts segment 1 and then dies.
        await stream._process_result(
            {"type": "partial", "text": "au rev", "segment": 1}
        )
        assert stream._trailing_segment_unfinalized

    @pytest.mark.asyncio
    async def test_an_engine_that_omits_the_field_stays_blind(self):
        """Absent numbering degrades to the old behaviour, never to fake loss."""
        stream = await make_stream()
        mock_event_ch(stream)
        await stream._process_result({"type": "partial", "text": "x"})
        assert not stream._trailing_segment_unfinalized
        for bad in (None, "1", 1.5, True):
            stream._open_segment = None
            stream._finalized_segment = None
            await stream._process_result(
                {"type": "partial", "text": "x", "segment": bad}
            )
            assert not stream._trailing_segment_unfinalized, f"segment={bad!r}"

    @pytest.mark.asyncio
    async def test_the_warning_says_IS_missing_not_MAY_be(self, caplog):
        """
        A concluded exchange can still have left a segment open - and the ack
        fast path makes concluding early the normal case, so this had been
        reported as a clean success.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "partial", "text": "bonjour", "segment": 0})
            ws.feed_json({"type": "final", "text": "bonjour", "segment": 0})
            await asyncio.sleep(0.05)
            # A new utterance opens and is never finalized.
            ws.feed_json({"type": "partial", "text": "au rev", "segment": 1})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.05)
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )

        task = asyncio.create_task(scenario())
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=10.0)
        await task

        assert stream._session_complete, "the delivered final still counts"
        assert any(
            "unfinalized engine segment" in r.message for r in caplog.records
        ), (
            "an exchange that concluded with a segment still open must say so; "
            f"got {[r.message for r in caplog.records]}"
        )


class TestRoundElevenRegressions:
    """
    The round-11 findings, each pinned by the scenario that was broken.

    The shared theme: the completion question ("did the engine finish, and
    deliver everything?") was answered by single weak signals - a timestamp
    ordering, `concluded` alone, an unconditional ack. These pin the two
    measured facts that replaced them: the engine's 0.67s punctuation
    cadence (via the byte counter) and text-gated segment accounting.
    """

    # ---- Move 2: the empty verdict needs the engine answering to the end

    @pytest.mark.asyncio
    async def test_a_wedge_after_the_leading_silence_is_not_an_empty_session(
        self, caplog
    ):
        """
        Round-11 finding 0, the reopened empty-success hole. The engine
        finalizes the leading silence (setting _last_final_at within the
        first second of EVERY session), then wedges while the user speaks.
        The gateway still closes after Done, so `concluded` alone certified
        the loss as a clean empty session.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            await asyncio.sleep(0.05)
            # The engine punctuates the leading silence...
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            # ...then wedges. The user speaks well past the unanswered
            # allowance (1s at the wire rate = 32000B; each frame is 3200B).
            for _ in range(40):
                stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.3)
            # Model REAL-TIME capture: in production those 4s of speech take
            # 4s of wall time, so the engine's last transcript is 4s stale at
            # verdict. A test burst compresses that to milliseconds, which
            # the wall-gap rescue would (correctly, for a burst) read as
            # innocent - age the timestamp so the test carries the wedge's
            # actual signature: large counter AND stale transcript.
            if stream._last_progress_at is not None:
                stream._last_progress_at -= 4.0
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.1)
            ws.end()  # gateway closes regardless: concluded=True

        task = asyncio.create_task(scenario())
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=15.0)
        await task

        # 4s of unanswered tail sits in the AMBIGUOUS band: a loaded pod's
        # stretched cadence produces the identical signature, so a terminal
        # error here killed healthy quiet sessions (round 13). The session
        # completes - but it must NOT be certified as a clean empty, and the
        # warning must name the unanswered window.
        assert stream._session_complete
        assert any(
            "had not answered the last" in r.message for r in caplog.records
        ), "the possible-loss warning is the whole point of the middle verdict"
        assert not any(
            "empty session, not a lost one" in r.message
            for r in caplog.records
        ), "an unanswered 4s tail must never be CERTIFIED empty"

    @pytest.mark.asyncio
    async def test_a_wedge_past_the_fatal_band_still_raises(self):
        """Above WEDGE_FATAL_UNANSWERED_SECONDS no pod load explains it."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            for _ in range(120):  # 12s of speech bytes: past the band
                stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.3)
            if stream._last_progress_at is not None:
                stream._last_progress_at -= 15.0  # real-time capture
            stream._input_ch.close()
            await asyncio.sleep(0.1)
            ws.end()

        task = asyncio.create_task(scenario())
        try:
            with pytest.raises(TranscriptLostError):
                await asyncio.wait_for(stream._run(), timeout=15.0)
        finally:
            task.cancel()
        assert not stream._session_complete

    @pytest.mark.asyncio
    async def test_a_quiet_participant_still_earns_the_empty_verdict(self):
        """
        The counter must not reintroduce the round-9 failure: a healthy
        engine keeps punctuating (measured every ~0.67s), so the counter
        stays near zero however long the user is quiet.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            # Quiet audio flows; the engine punctuates it continuously.
            for i in range(12):
                stream._input_ch.send_nowait(speech_frame())
                if i % 3 == 2:
                    await asyncio.sleep(0.03)
                    ws.feed_json({"type": "final", "text": ""})
            await asyncio.sleep(0.05)
            stream._input_ch.close()
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": ""})
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=15.0)
        await task
        assert stream._session_complete

    # ---- Move 1: only TEXT opens a segment

    @pytest.mark.asyncio
    async def test_the_punctuation_cadence_never_claims_a_lost_utterance(
        self, caplog
    ):
        """
        Round-11 finding 7. The engine emits an EMPTY partial/final pair
        every ~0.67s on silence, reusing the segment number - and a session
        concluding between such a partial and its final claimed a trailing
        utterance IS missing when nothing was ever there.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "partial", "text": "bonjour", "segment": 0})
            ws.feed_json({"type": "final", "text": "bonjour", "segment": 0})
            await asyncio.sleep(0.05)
            # The cadence opens the NEXT tick with an empty partial...
            ws.feed_json({"type": "partial", "text": "", "segment": 1})
            await asyncio.sleep(0.05)
            # ...and the session concludes before its (empty) final.
            stream._input_ch.close()
            await asyncio.sleep(0.05)
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )

        task = asyncio.create_task(scenario())
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=10.0)
        await task

        assert stream._session_complete
        assert not any(
            "unfinalized engine segment" in r.message for r in caplog.records
        ), (
            "an empty punctuation partial is not an utterance; warning on it "
            "cries wolf on every session that concludes mid-cadence"
        )

    # ---- Move 3: the ack is gated and subordinate to segment accounting

    @pytest.mark.asyncio
    async def test_a_pre_done_done_frame_does_not_disarm_the_drain(self):
        """
        Round-11 finding 2. A Done-prefixed frame arriving MID-session used
        to latch the ack permanently, so the real Done later found the drain
        pre-disarmed and every trailing final was dropped.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            # Mid-session control frame, long before Done.
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done-ish")
            )
            await asyncio.sleep(0.05)
            assert not stream._engine_acked_done, (
                "a Done-prefixed frame before Done was written must not latch"
            )
            stream._input_ch.close()  # NOW Done goes out
            await asyncio.sleep(0.2)
            # The trailing final that the stale latch used to drop:
            ws.feed_json({"type": "final", "text": "bonjour", "segment": 0})
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=10.0)
        await task

        finals = [
            c.args[0]
            for c in stream._event_ch.send_nowait.call_args_list
            if c.args[0].type == SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals and finals[-1].alternatives[0].text == "bonjour", (
            "the trailing final was dropped: the pre-Done frame disarmed "
            "the drain"
        )

    @pytest.mark.asyncio
    async def test_the_ack_waits_for_an_open_text_segment(self, monkeypatch):
        """
        Round-11 finding 1. The ack is a receipt for the Done command, not
        proof decoding finished; if it lands ahead of the trailing final the
        fast path used to end the turn and drop that final. With text-gated
        segments the fast path holds whenever transcribed text is still
        unfinalized, whatever order the ack arrives in.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 3.0, raising=False
        )
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            # Text is in flight when the session ends...
            ws.feed_json({"type": "partial", "text": "au revoir", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.1)
            # ...the ack arrives AHEAD of the final...
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )
            await asyncio.sleep(0.3)
            # ...and the final lands after it.
            ws.feed_json({"type": "final", "text": "au revoir", "segment": 0})

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=10.0)
        await task

        finals = [
            c.args[0]
            for c in stream._event_ch.send_nowait.call_args_list
            if c.args[0].type == SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals and finals[-1].alternatives[0].text == "au revoir", (
            "the ack ended the turn ahead of the final it acknowledges"
        )

    # ---- Findings 3 and 9: the per-attempt resets, behaviourally

    @pytest.mark.asyncio
    async def test_a_dead_attempts_numbers_do_not_corrupt_the_next_attempts(
        self, caplog
    ):
        """
        Re-premised in round 16. This test originally asserted that a dead
        attempt's segment maxima must produce NO warning ("invent a loss") -
        but those maxima are EVIDENCE, not garbage: segments open only on
        text partials and close only on text finals, so open > finalized at
        attempt death means text genuinely went unfinalized, and round 16
        made that fact sticky precisely because forgetting it let the
        A-delivered-B-lost shape certify clean. The warning is now CORRECT.

        What round 11 validly established survives as the assertion below:
        the dead attempt's numbering must not corrupt the NEW attempt's own
        per-attempt accounting - attempt 2 finalizes its segment 0 and its
        per-attempt trailing check must read clean, whatever the session
        remembers.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        # A dead attempt's evidence: text was partial'd into segment 5 and
        # never finalized (numbers only move on text - this is not garbage).
        stream._open_segment = 5
        stream._finalized_segment = 2
        stream._text_seen_in_session = True

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "partial", "text": "bonjour", "segment": 0})
            ws.feed_json({"type": "final", "text": "bonjour", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()
            await asyncio.sleep(0.05)
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )

        task = asyncio.create_task(scenario())
        try:
            with caplog.at_level(
                logging.WARNING, logger="livekit.plugins.voxist"
            ):
                await asyncio.wait_for(stream._run(), timeout=10.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert stream._session_complete, "attempt 2 delivered its final"
        # The isolation property round 11 was actually about: the new
        # attempt's own accounting is clean (its segment 0 was finalized) -
        # the dead attempt's 5-vs-2 did not leak into the per-attempt facts.
        assert not stream._trailing_segment_unfinalized, (
            "attempt 2 finalized everything it saw; a dead attempt's "
            "numbers corrupted the per-attempt comparison"
        )
        # And the session-level warning for the dead attempt's genuinely
        # unfinalized text is PRESENT - it is evidence, not an invention.
        assert any(
            "unfinalized engine segment" in r.message for r in caplog.records
        ), (
            "text the dead attempt saw and never saw finalized must be "
            "surfaced, or A-delivered-B-lost certifies clean (round 16)"
        )

    @pytest.mark.asyncio
    async def test_stale_segment_maxima_do_not_swallow_a_loss(self, caplog):
        """The other direction: stale _finalized_segment hides a real one."""
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._open_segment = 7
        stream._finalized_segment = 7  # dead attempt finalized everything

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            # This attempt delivers one final, then transcribes text it
            # never finalizes.
            ws.feed_json({"type": "final", "text": "bonjour", "segment": 0})
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "partial", "text": "au rev", "segment": 1})
            await asyncio.sleep(0.05)
            stream._input_ch.close()
            await asyncio.sleep(0.05)
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )

        task = asyncio.create_task(scenario())
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._run(), timeout=15.0)
        await task

        assert any(
            "unfinalized engine segment" in r.message for r in caplog.records
        ), (
            "segment 1's text was never finalized, but the dead attempt's "
            "stale maximum (7 >= 7) swallowed the loss"
        )

    @pytest.mark.asyncio
    async def test_a_stale_ack_does_not_conclude_the_next_attempt(
        self, monkeypatch
    ):
        """
        Round-11 finding 9, mutation-verified there: deleting the ack's
        per-attempt reset left 580/580 green. This fails without it - the
        stale latch concludes the retry's drain on its first pass, before
        the new socket's engine has answered, and its final is dropped.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "POST_FINAL_IDLE_SECONDS", 0.2, raising=False
        )
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 3.0, raising=False
        )
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._engine_acked_done = True  # what a dead attempt leaves behind

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            # The engine answers AFTER a delay: a fresh drain waits for it,
            # a stale-latched one concluded at time zero and dropped it.
            await asyncio.sleep(0.4)
            ws.feed_json({"type": "final", "text": "bonjour", "segment": 0})

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=10.0)
        await task

        finals = [
            c.args[0]
            for c in stream._event_ch.send_nowait.call_args_list
            if c.args[0].type == SpeechEventType.FINAL_TRANSCRIPT
        ]
        assert finals, (
            "the previous attempt's ack concluded this attempt's drain "
            "before the engine answered"
        )


class TestRoundTwelveRegressions:
    """
    Round 12's findings: the ack fast path was unsalvageable as an instant
    terminator, and the empty-verdict byte counter confused bytes with wall
    time in both directions. The fixes demote the ack to an idle-margin
    shortener and give the verdict the same bytes-OR-wall-clock shape the
    stall detector already needed.
    """

    # ---- the empty verdict's two-clause test, pinned at its boundaries

    @staticmethod
    async def _verdict_stream(*, counter, final_seen=True):
        """
        The property reads exactly three facts: a final was seen, no text
        was seen, and the unanswered byte counter. No clock - a gap
        parameter used to live here and its meaningful-looking values
        implied the verdict still distinguished recent from stale
        transcripts, the premise the rescue-clause deletion killed.
        """
        stream = await make_stream()
        mock_event_ch(stream)
        stream._last_final_at = time.monotonic() if final_seen else None
        stream._last_progress_at = stream._last_final_at
        stream._speaking = False
        stream._bytes_sent_since_progress = counter
        return stream

    @pytest.mark.asyncio
    async def test_a_nonzero_counter_under_the_limit_is_granted(self):
        """
        Pins the threshold FROM BELOW - round 12 showed every green suite
        survived a 10x-too-strict threshold because no test exercised a
        nonzero counter under the limit. 16000B is half the 1s allowance:
        if the limit shrinks tenfold (3200B) or degrades to ==0, this fails.
        """
        stream = await self._verdict_stream(counter=16_000)
        assert stream._engine_reported_empty_this_attempt

    @pytest.mark.asyncio
    async def test_over_the_limit_is_denied(self):
        """Bytes past the allowance: never certified, whatever the clock."""
        stream = await self._verdict_stream(counter=64_000)
        assert not stream._engine_reported_empty_this_attempt, (
            "2s of speech the engine never answered, sent 4s ago, is a "
            "wedge - certifying it as empty is the loss this exists to stop"
        )

    @pytest.mark.asyncio
    async def test_no_rescue_clause_certifies_a_large_tail(self):
        """
        Round-14 finding 1 falsified the last rescue premise. A recency test
        rescued a wedge whose server closed promptly; a rate test rescued a
        wedge during BATCH input, whose bytes beat real time by
        construction - unbounded loss certified clean. There is no send-side
        signature separating "not answered YET" from "never will be", so the
        clean-empty certificate is byte-bounded and nothing else: large
        tails belong to the gate's warn/fatal tiers, which never say clean.
        """
        for counter in (640_000, 256_000, 64_000):
            stream = await self._verdict_stream(counter=counter)
            assert not stream._engine_reported_empty_this_attempt, (
                f"counter={counter}: a tail past the byte allowance must "
                "never be CERTIFIED clean, whatever shape it arrived in - "
                "the warn/fatal tiers own it"
            )

    @pytest.mark.asyncio
    async def test_no_final_seen_is_always_denied(self):
        """An engine that never finalized anything has rendered no verdict."""
        stream = await self._verdict_stream(counter=0, final_seen=False)
        assert not stream._engine_reported_empty_this_attempt

    # ---- the ack never concludes on its own

    @pytest.mark.asyncio
    async def test_the_ack_alone_concludes_nothing(self, monkeypatch):
        """
        Round-12 findings 0 and 5, one root cause: any instant-ack exit
        races frames the receive loop has not seen - whether held open by
        segment accounting (released early by an empty cadence final reusing
        the open number) or not (an utterance still in the decode buffer
        opens no segment at all). The ack now only shortens the idle margin
        AFTER a post-Done final; with no such final it concludes nothing and
        the drain runs to its backstop.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 1.5, raising=False
        )
        ws = FakeWS()  # never closes, never sends a post-Done final
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "bonjour", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.1)
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )

        task = asyncio.create_task(scenario())
        started = time.monotonic()
        await asyncio.wait_for(stream._run(), timeout=10.0)
        elapsed = time.monotonic() - started
        await task

        # A full second of margin against the 1.5s backstop: round 13
        # showed a 0.1s margin lets a reverted instant-ack exit slip past on
        # a loaded CI box (its ~0.3s nominal exit plus scheduling delays).
        assert elapsed >= 1.0, (
            f"the drain ended after {elapsed:.2f}s: the ack concluded on its "
            "own instead of waiting for the backstop - with no post-Done "
            "final, an ack proves nothing about frames still in flight"
        )
        # The acked backstop CLASSIFIES the ending as concluded (round-13
        # finding 0), so the session completes - what the ack may never do
        # is end the drain early.
        assert stream._session_complete

    @pytest.mark.asyncio
    async def test_an_acked_backstop_concludes_the_exchange(self, monkeypatch):
        """
        Round-13 finding 0. An engine that finalized the trailing silence
        BEFORE end_input has nothing to send after Done - it just acks. The
        backstop used to report that as concluded=False, and the gate then
        raised a terminal TranscriptLostError on a healthy quiet participant
        whose socket never closes. An acked backstop is not the instant-ack
        exit two earlier findings killed: nothing is in flight after a full
        drain window of quiet.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.5, raising=False
        )
        ws = FakeWS()  # Kroko: never closes
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            # The engine finalizes the trailing silence BEFORE Done...
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.1)
            # ...and afterwards sends ONLY the ack. No post-Done final.
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )

        task = asyncio.create_task(scenario())
        await asyncio.wait_for(stream._run(), timeout=10.0)
        await task

        assert stream._session_complete, (
            "a quiet participant on a never-closing socket whose engine "
            "finalized before Done and acked must complete, not die with a "
            "terminal TranscriptLostError"
        )

    # ---- segment accounting under the cadence's number reuse

    @pytest.mark.asyncio
    async def test_an_empty_final_does_not_close_the_open_text_segment(self):
        """
        Round-12 finding 0's root: cadence pairs REUSE the current segment
        number, so an empty tick carrying seg N can arrive while seg N's
        text final is in flight. Closing N on it vouched for text nobody had
        seen finalized.
        """
        stream = await make_stream()
        mock_event_ch(stream)
        await stream._process_result(
            {"type": "partial", "text": "au revoir", "segment": 0}
        )
        # The cadence tick, same number, no text:
        await stream._process_result(
            {"type": "final", "text": "", "segment": 0}
        )
        assert stream._trailing_segment_unfinalized, (
            "an empty final reusing the open text segment's number is not a "
            "verdict on that text"
        )
        # The real final closes it.
        await stream._process_result(
            {"type": "final", "text": "au revoir", "segment": 0}
        )
        assert not stream._trailing_segment_unfinalized

    # ---- the backlog estimator's two new properties

    @pytest.mark.asyncio
    async def test_the_mean_recovers_after_a_frame_regime_change(self):
        """
        Round-12 finding: an attempt-global mean held ~1.0s long after a
        caller switched from 1s prompt frames to live 10ms capture, so an
        ordinary 130-frame live hiccup read as 130s and live speech was
        dropped. The 50-frame window converges within half a second of live
        audio.
        """
        stream = await make_stream()
        stream._input_ch = SimpleNamespace(qsize=lambda: 130, closed=False)
        stream._pre_drain_items = 0
        stream._items_popped = 0
        # The prompt regime: enough 1s frames that a GLOBAL mean stays near
        # 1.0 long after the switch (1000 prompt + 50 live pops leaves a
        # global mean of ~0.95, and 130 x 0.95 = 124s > 120s - the revert
        # signature). The first version used 120 prompt frames, which
        # diluted the global mean to 0.7 and made the test pass either way.
        for _ in range(1000):
            stream._note_frame_duration(1.0)
        # The live regime takes over the 50-frame window completely.
        for _ in range(50):
            stream._note_frame_duration(0.01)

        assert not stream._backlog_exceeds_the_bound(), (
            "130 live frames (1.3s) read as over 120s: the prompt regime is "
            "still poisoning the estimate after the window should have "
            "flushed it"
        )

    @pytest.mark.asyncio
    async def test_a_retry_does_not_grandfather_the_previous_backlog(self):
        """
        Round-12 finding: a per-attempt snapshot let every retry exempt the
        backlog the previous attempt accumulated - up to N x 120s across
        max_retry attempts, unbounding the ceiling. The snapshot is taken
        once per session; the pop counter persists.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._ws = ws

        # Full-chunk frames (1600 samples = one 100ms chunk each) so every
        # frame produces a send: 160-sample frames buffer 10:1 in the
        # processor and the flaky send below would never fire.
        for _ in range(30):
            stream._input_ch.send_nowait(speech_frame(1600))
        ws.fail_on_bytes = False

        # Attempt 1: consume a few frames, then die mid-stream.
        sent = 0
        original = ws.send_bytes

        async def flaky(data):
            nonlocal sent
            sent += 1
            if sent > 3:
                raise ConnectionResetError("mid-stream death")
            await original(data)

        ws.send_bytes = flaky
        with pytest.raises(APIConnectionError):
            await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        first_snapshot = stream._pre_drain_items
        popped_after_one = stream._items_popped
        assert stream._pre_drain_snapshot_taken
        assert first_snapshot == 30

        # More audio arrives before the retry - growth, not pre-drain.
        for _ in range(20):
            stream._input_ch.send_nowait(speech_frame(1600))

        ws.send_bytes = original
        stream._input_ch.close()
        await asyncio.wait_for(stream._send_audio_task(), timeout=10.0)

        assert stream._pre_drain_items == first_snapshot, (
            "the retry re-snapshotted the queue, grandfathering the growth "
            "the first attempt left behind - the ceiling bounds nothing if "
            "every attempt starts a fresh exemption"
        )
        assert stream._items_popped > popped_after_one, (
            "the pop counter must persist across attempts, or the snapshot "
            "is never consumed"
        )


class TestRoundThirteenRegressions:
    """
    Round 13's findings, each pinned by the mechanism that replaced the
    defective one: the burst rescue is rate-shaped, post-Done finals are
    counted rather than timestamp-compared, and empty finals touch no
    segment state at all.
    """

    @pytest.mark.asyncio
    async def test_the_verdict_reads_no_clock(self):
        """
        Round-14 findings 1 and 4, one stone: the property consults bytes
        only. Freezing the module clock proves no gap is computed (the old
        rate clause raced wall time with 0.3s of headroom in this very
        test), and the byte boundary is pinned on both sides.
        """
        from livekit.plugins.voxist import stream as stream_module

        stream = await TestRoundTwelveRegressions._verdict_stream(counter=32_000)
        assert stream._engine_reported_empty_this_attempt, (
            "exactly the 1s allowance is within the certificate"
        )
        stream._bytes_sent_since_progress = 32_001
        assert not stream._engine_reported_empty_this_attempt

        # No clock reads: a poisoned monotonic would explode if consulted.
        def boom():
            raise AssertionError("the verdict must not consult the clock")

        import pytest as _pytest

        with _pytest.MonkeyPatch.context() as mp:
            mp.setattr(
                stream_module, "time", SimpleNamespace(monotonic=boom)
            )
            stream._bytes_sent_since_progress = 16_000
            assert stream._engine_reported_empty_this_attempt

    @pytest.mark.asyncio
    async def test_post_done_finals_are_counted_not_timestamp_compared(
        self, monkeypatch
    ):
        """
        Round-13 finding 5. On a coarse monotonic clock (~15.6ms ticks on
        Windows before 3.13) a pre-Done and post-Done final stamped in the
        same tick compare equal, so timestamp-inequality detection never saw
        the post-Done final and a healthy session ran the full backstop.
        A counter cannot alias, whatever the clock does.
        """
        from livekit.plugins.voxist import stream as stream_module

        stream = await make_stream()
        mock_event_ch(stream)

        frozen = time.monotonic()
        monkeypatch.setattr(
            stream_module,
            "time",
            SimpleNamespace(monotonic=lambda: frozen),
        )

        # Two finals inside the SAME clock tick.
        stream._note_engine_progress(is_final=True)
        before = stream._finals_count_this_attempt
        timestamp_before = stream._last_final_at
        stream._note_engine_progress(is_final=True)

        assert stream._last_final_at == timestamp_before, (
            "the premise needs both finals stamped identically"
        )
        assert stream._finals_count_this_attempt == before + 1, (
            "the counter must distinguish what the timestamps cannot"
        )

    @pytest.mark.asyncio
    async def test_a_higher_numbered_empty_final_closes_nothing(self):
        """
        Round-13 finding 2. The same-number-only guard was defeated by
        max(): a HIGHER-numbered empty cadence final (the engine having
        moved on) raised _finalized_segment past the open text segment and
        silenced the trailing-loss warning for text nobody saw finalized.
        Empty finals now touch neither maximum.
        """
        stream = await make_stream()
        mock_event_ch(stream)
        # Text opens segment 0; its (malformed-text) final reads as empty
        # and is rightly refused...
        await stream._process_result(
            {"type": "partial", "text": "au revoir", "segment": 0}
        )
        await stream._process_result(
            {"type": "final", "text": "", "segment": 0}
        )
        # ...and the engine's NEXT cadence pair carries the incremented
        # number. This must not close segment 0 either.
        await stream._process_result(
            {"type": "final", "text": "", "segment": 1}
        )
        assert stream._trailing_segment_unfinalized, (
            "a higher-numbered empty final vouched for text the plugin "
            "never saw finalized"
        )


class TestRoundFourteenRegressions:
    """
    Round 14's findings: the tier structure held; its edges did not.
    """

    @pytest.mark.asyncio
    async def test_a_talking_engine_is_not_a_quiet_backstop(
        self, monkeypatch, caplog
    ):
        """
        Round-14 finding 0. The acked backstop concluded on the ack alone,
        classifying an engine still streaming text partials (alive,
        mid-decode, no final yet) as a concluded-and-quiet exchange - a
        clean success over silently truncated text when a pre-Done final
        was on record. Quiet is now checked, not assumed: a talking engine
        falls through to concluded=False and the gate says a trailing
        transcript may be missing.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.8, raising=False
        )
        ws = FakeWS()  # never closes
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "bonjour", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.05)
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )
            # The engine keeps DECODING: text partials right through the
            # backstop, never a final. No segment field - the documented
            # blind-spot degradation the finding exploited. The loop is
            # always inside an await, so cancellation alone stops it.
            while True:
                ws.feed_json({"type": "partial", "text": "au rev"})
                await asyncio.sleep(0.1)

        task = asyncio.create_task(scenario())
        try:
            with caplog.at_level(
                logging.WARNING, logger="livekit.plugins.voxist"
            ):
                await asyncio.wait_for(stream._run(), timeout=10.0)
        finally:
            # In a finally, awaited, or a regression in _run leaks the 0.1s
            # feed loop through the unwind and buries the real diagnostics
            # under "Task was destroyed but it is pending!".
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert stream._session_complete, "the pre-Done final was delivered"
        assert any(
            "may have been lost" in r.message for r in caplog.records
        ), (
            "an engine still talking at the backstop is an ending imposed on "
            "it - reporting it as a clean, quiet conclusion silently "
            "truncates the text it was decoding"
        )

    @pytest.mark.asyncio
    async def test_text_seen_by_a_dead_attempt_still_bars_the_middle_tier(
        self,
    ):
        """
        Round-14 finding 2. engine_produced_text was per-attempt state, so a
        retry after an attempt that SAW text (interims off: seen, never
        delivered) treated a known loss as ambiguous and completed with only
        the middle-tier warning. The fact is session-scoped now.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        # What a dead attempt leaves behind: its finally cleared _speaking,
        # but the session remembers the text.
        stream._text_seen_in_session = True

        async def scenario():
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            for _ in range(20):  # ~2s tail: inside the ambiguous band
                stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.2)
            if stream._last_progress_at is not None:
                stream._last_progress_at -= 3.0
            stream._input_ch.close()
            await asyncio.sleep(0.1)
            ws.end()

        task = asyncio.create_task(scenario())
        try:
            with pytest.raises(TranscriptLostError):
                await asyncio.wait_for(stream._run(), timeout=15.0)
        finally:
            task.cancel()

        assert not stream._session_complete, (
            "text the session is KNOWN to have lost is not ambiguous; the "
            "middle tier may not complete it"
        )


class TestRoundFifteenRegressions:
    """
    Round 15's findings: the round-14 latch conflated "text seen" with
    "text lost", failing in both directions at once, and the quiet check
    keyed on any transcript instead of text.
    """

    @pytest.mark.asyncio
    async def test_delivered_text_does_not_bar_a_retry_from_the_middle_tier(
        self, caplog
    ):
        """
        Round-15 finding 0. Attempt 1 DELIVERED the user's speech; a blip
        forced a retry whose trailing room-noise ended with a 1-10s tail.
        The seen-only latch barred the middle tier and a fully-delivered
        session's retry died with a terminal error. Seen-minus-delivered is
        the loss; delivered text is not.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        # What a successful attempt 1 leaves behind:
        stream._text_seen_in_session = True
        stream._final_received = True  # session-scoped delivered-final fact

        async def scenario():
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            for _ in range(20):  # ~2s tail: the ambiguous band
                stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.2)
            if stream._last_progress_at is not None:
                stream._last_progress_at -= 3.0
            stream._input_ch.close()
            await asyncio.sleep(0.1)
            ws.end()

        task = asyncio.create_task(scenario())
        try:
            with caplog.at_level(
                logging.WARNING, logger="livekit.plugins.voxist"
            ):
                await asyncio.wait_for(stream._run(), timeout=15.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert stream._session_complete, (
            "a retry after fully DELIVERED text is the ambiguous middle, "
            "not a known loss - fatal here destroys a healthy session"
        )
        assert any(
            "had not answered the last" in r.message for r in caplog.records
        )

    @pytest.mark.asyncio
    async def test_undelivered_text_bars_the_clean_certificate_too(self):
        """
        Round-15 finding 1, the other face. Text seen-and-lost by a dead
        attempt barred only the middle tier; a quiet retry with a sub-1s
        tail was still CERTIFIED clean one tier up. A known loss may take
        neither quiet tier.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        stream._text_seen_in_session = True  # seen by a dead attempt...
        # ...and nothing was ever delivered (both session flags False).

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # tiny tail: inside the clean allowance
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        try:
            with pytest.raises(TranscriptLostError):
                await asyncio.wait_for(stream._run(), timeout=10.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert not stream._session_complete, (
            "text the session KNOWS it lost was certified as a clean empty "
            "session because only the middle tier consulted the latch"
        )

    @pytest.mark.asyncio
    async def test_empty_cadence_through_the_drain_is_still_quiet(
        self, monkeypatch, caplog
    ):
        """
        Round-15 finding 2. Keying the backstop's quiet check on ANY
        transcript meant an engine whose empty cadence continued post-Done
        never read as quiet: concluded=False, and a healthy no-text session
        fell through both quiet tiers to the terminal raise. Cadence noise
        is not speech; only TEXT denies the quiet.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.8, raising=False
        )
        ws = FakeWS()  # never closes
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.05)
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )
            # The cadence keeps ticking: EMPTY partials, finals delayed past
            # the drain window. Healthy, quiet, no text anywhere.
            while True:
                ws.feed_json({"type": "partial", "text": ""})
                await asyncio.sleep(0.1)

        task = asyncio.create_task(scenario())
        try:
            with caplog.at_level(logging.INFO, logger="livekit.plugins.voxist"):
                await asyncio.wait_for(stream._run(), timeout=10.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert stream._session_complete, (
            "an empty cadence through the drain is a healthy quiet engine; "
            "reading it as 'still talking' sent a silent participant to the "
            "terminal raise"
        )
        # WHICH tier matters as much as completing: this is the clean-empty
        # certificate, and landing in the warn-complete middle tier instead
        # would log a possible-loss warning on every silent participant's
        # turn - the misdiagnosis the INFO/WARNING split exists to prevent.
        assert any(
            "empty session, not a lost one" in r.message
            for r in caplog.records
        ), "the healthy quiet session must be CERTIFIED clean, not warned"
        assert not any(
            "had not answered the last" in r.message for r in caplog.records
        )


class TestRoundSixteenRegressions:
    """
    Round 16's findings: delivery anywhere in the session unbarred every
    later loss, and protocol-drifted text manufactured a clean conclusion.
    """

    @pytest.mark.asyncio
    async def test_a_delivered_then_lost_session_is_never_certified_clean(
        self, caplog
    ):
        """
        Round-16 finding 0, the A-delivered-B-lost shape. Attempt 1
        delivered A's final, saw B's text partial, and died before B's
        final; the retry's room noise ended cleanly. A's delivery zeroed
        the undelivered-text bar and the per-attempt reset destroyed B's
        segment evidence, so B's KNOWN loss took the clean-empty INFO.
        The segment evidence is sticky across attempts now.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        # Attempt 1's end-state, exactly as it dies: A delivered, B's text
        # partial seen, B's segment open and unfinalized.
        stream._final_received = True
        stream._text_seen_in_session = True
        stream._open_segment = 1
        stream._finalized_segment = 0

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        try:
            with caplog.at_level(
                logging.INFO, logger="livekit.plugins.voxist"
            ):
                with pytest.raises(TranscriptLostError):
                    await asyncio.wait_for(stream._run(), timeout=10.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert not any(
            "empty session, not a lost one" in r.message
            for r in caplog.records
        ), (
            "B's loss was certified clean because A's delivery zeroed the "
            "text bar and the reset destroyed the segment evidence"
        )

    @pytest.mark.asyncio
    async def test_malformed_text_partials_deny_the_quiet(self, monkeypatch):
        """
        Round-16 finding 1. Partials whose "text" is present but non-string
        did not stamp the text timestamp, so a drifted engine's traffic read
        as 'quiet by definition' and the acked backstop certified a clean
        empty session where the previous behaviour failed loudly. Unreadable
        text is text; only genuine emptiness is cadence.
        """
        monkeypatch.setattr(
            VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.8, raising=False
        )
        ws = FakeWS()  # never closes
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.05)
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )
            # Protocol drift: text present but non-string, forever.
            while True:
                ws.feed_json({"type": "partial", "text": 123})
                await asyncio.sleep(0.1)

        task = asyncio.create_task(scenario())
        try:
            with pytest.raises(TranscriptLostError):
                await asyncio.wait_for(stream._run(), timeout=10.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert not stream._session_complete, (
            "unreadable text through the drain was read as quiet cadence "
            "and certified clean - protocol drift must fail loudly"
        )


class TestRoundSeventeenRegressions:
    """
    Round 17's findings: the drift guard covered one drain exit out of
    three, and drift evidence died with the attempt that saw it.
    """

    @pytest.mark.asyncio
    async def test_drifted_finals_are_never_certified_clean(self):
        """
        Round-17 finding 0. A drifted engine's non-string-text FINALS
        conclude the drain through the answered paths - untouched by the
        quiet check - and every verdict fact read clean: certified empty
        over undeciphered speech, on every session once the drift ships.
        Unreadable text now latches a session known-loss fact that bars
        both quiet tiers.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            stream._input_ch.close()  # -> Done
            await asyncio.sleep(0.1)
            # The engine "answers" - with text nobody can read.
            ws.feed_json({"type": "final", "text": 123, "segment": 0})
            ws.incoming.put_nowait(
                SimpleNamespace(type=aiohttp.WSMsgType.TEXT, data="Done!")
            )

        task = asyncio.create_task(scenario())
        try:
            with pytest.raises(TranscriptLostError):
                await asyncio.wait_for(stream._run(), timeout=10.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert not stream._session_complete, (
            "a drifted final concluded the drain and was certified as a "
            "clean empty session - undeciphered speech lost silently"
        )

    @pytest.mark.asyncio
    async def test_drift_seen_by_a_dead_attempt_still_bars_the_verdict(self):
        """
        Round-17 finding 1. Drift partials seen mid-decode by attempt 1
        latched nothing session-scoped, so the retry's clean ending
        certified the loss. The unreadable-text fact is sticky now.
        """
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        mock_event_ch(stream)
        # What attempt 1 leaves behind after seeing {"text": 123} partials:
        stream._unreadable_text_seen_in_session = True

        async def scenario():
            stream._input_ch.send_nowait(speech_frame())
            await asyncio.sleep(0.05)
            ws.feed_json({"type": "final", "text": "", "segment": 0})
            await asyncio.sleep(0.05)
            stream._input_ch.close()
            await asyncio.sleep(0.05)
            ws.end()

        task = asyncio.create_task(scenario())
        try:
            with pytest.raises(TranscriptLostError):
                await asyncio.wait_for(stream._run(), timeout=10.0)
        finally:
            task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await task

        assert not stream._session_complete
