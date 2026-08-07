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
from livekit.agents.types import APIConnectOptions

from livekit import rtc
from livekit.plugins.voxist.exceptions import ConnectionError as VoxistConnectionError
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
        if self.closed:
            raise ConnectionResetError("closed")
        self.sent_bytes.append(data)

    async def send_str(self, data):
        if self.closed:
            raise ConnectionResetError("closed")
        self.sent_text.append(data)

    async def close(self):
        self.closed = True
        self.end()


def frame(samples=160):
    return rtc.AudioFrame(
        data=np.zeros(samples, dtype=np.int16).tobytes(),
        sample_rate=16000,
        num_channels=1,
        samples_per_channel=samples,
    )


class TestSendPath:
    """Send-side invariants: bounded sends, no thresholds, honest failures."""

    @pytest.mark.asyncio
    async def test_timeout_constants_are_ordered(self):
        """A stalled send must surface long before the post-Done drain cap."""
        assert VoxistSTTStream.SEND_TIMEOUT_SECONDS > 0
        assert (
            VoxistSTTStream.SEND_TIMEOUT_SECONDS
            <= VoxistSTTStream.SESSION_DRAIN_TIMEOUT_SECONDS / 2
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

        # Fresh-host clock: small monotonic values must not suppress the
        # first report (0.0 was once used as a sentinel and did exactly that)
        clock = iter([1.0 + i * 0.001 for i in range(5000)])
        monkeypatch.setattr(
            "livekit.plugins.voxist.stream.time.monotonic", lambda: next(clock)
        )

        for _ in range(200):
            stream._input_ch.send_nowait(frame())
        stream._input_ch.close()

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            await asyncio.wait_for(stream._send_audio_task(), timeout=10.0)

        drops = [r for r in caplog.records if "dropping audio" in r.message]
        assert stream.dropped_frames == 200 - 5 - 1
        assert len(drops) == 1


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
    async def test_drain_is_bounded_when_server_never_closes(self, monkeypatch):
        """A server that ignores Done cannot hang the stream forever."""
        monkeypatch.setattr(VoxistSTTStream, "SESSION_DRAIN_TIMEOUT_SECONDS", 0.2)
        ws = FakeWS()
        stream = await make_stream(dial=AsyncMock(return_value=ws))
        stream._event_ch = Mock()

        stream._input_ch.close()  # empty session, immediate Done

        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.wait_for(stream._run(), timeout=5.0)
        elapsed = loop.time() - start

        assert 0.2 <= elapsed < 3.0
        assert stream._session_complete
        assert ws.sent_text == ["Done"]


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
        stream._event_ch = Mock()

        stream._input_ch.close()
        ws_task = asyncio.create_task(stream._run())
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
