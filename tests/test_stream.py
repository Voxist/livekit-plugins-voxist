"""Unit tests for VoxistSTTStream: one socket per stream, framework-owned retries."""

import asyncio
import json
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import aiohttp
import numpy as np
import pytest
from livekit.agents import APIConnectionError
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import APIConnectOptions

from livekit import rtc
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
        assert 100 <= VoxistSTTStream.MAX_INPUT_BACKLOG_FRAMES <= 6000

    @pytest.mark.asyncio
    async def test_frames_over_the_cap_are_dropped(self, monkeypatch):
        stream = await make_stream()
        ws = attach_mock_ws(stream)
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 10)

        total = 40
        for _ in range(total):
            stream._input_ch.send_nowait(frame())
        stream._input_ch.close()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

        # The frame in hand has already left the channel when the check runs,
        # so cap + 1 frames survive.
        assert stream.dropped_frames == total - 10 - 1
        assert stream._done_sent, "the session must still end with Done"
        assert ws.send_str.await_args.args[0] == "Done"

    @pytest.mark.asyncio
    async def test_no_drops_below_the_cap(self, monkeypatch):
        stream = await make_stream()
        attach_mock_ws(stream)
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 100)

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
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 1000)

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
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 5)

        for _ in range(20):
            stream._input_ch.send_nowait(VoxistSTTStream._FlushSentinel())
        stream._input_ch.close()

        await asyncio.wait_for(stream._send_audio_task(), timeout=5.0)

    @pytest.mark.asyncio
    async def test_drop_warning_is_rate_limited(self, monkeypatch, caplog):
        stream = await make_stream()
        attach_mock_ws(stream)
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 5)
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

        loop = asyncio.get_running_loop()
        loop_time_before = loop.time()

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._send_audio_task(), timeout=10.0)

            drops = [r for r in caplog.records if "dropping audio" in r.message]
            assert stream.dropped_frames == 200 - 5 - 1
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
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 0)
        stream = await make_stream()
        attach_mock_ws(stream)
        mock_event_ch(stream)

        for _ in range(5):
            stream._input_ch.send_nowait(speech_frame())
        stream.end_input()

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
    The predicate reads livekit's private Chan._queue. If that attribute
    ever moves, the fallback decides what a rename costs - and the previous
    fallback (qsize()==0) was the pre-fix buggy check itself, so a livekit
    rename would silently resurrect total-transcript-loss-as-success.
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
    async def test_unreadable_queue_prefers_an_honest_error_to_a_fake_success(
        self,
    ):
        """
        With the channel uninspectable, a retry whose audio is gone must
        reach the honest TranscriptLostError - never dial, ship a bare Done
        and report success. The safe direction is stated in the helper and
        pinned here.
        """
        dial = AsyncMock(side_effect=AssertionError("must not dial"))
        stream = await make_stream(dial=dial)
        mock_event_ch(stream)

        # State a failed attempt leaves behind: audio consumed, no finals,
        # input closed - and now uninspectable.
        stream.end_input()
        stream._audio_consumed = True
        del stream._input_ch._queue

        with pytest.raises(TranscriptLostError):
            await asyncio.wait_for(stream._run(), timeout=5.0)
        dial.assert_not_awaited()
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
