"""Integration tests for complete LiveKit plugin pipeline."""

import asyncio
import time

import numpy as np
import pytest
from livekit.agents.stt import SpeechEventType
from livekit.agents.types import APIConnectOptions

from livekit import rtc
from livekit.plugins.voxist import VoxistSTT

from .fixtures.mock_server import MockVoxistServer


@pytest.mark.integration
class TestBasicStreaming:
    """Test basic audio streaming and transcription."""

    @pytest.mark.asyncio
    async def test_end_to_end_streaming(self, mock_voxist_server, generate_test_audio):
        """Test complete audio streaming pipeline."""
        # Create STT instance
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
            language="fr",
        )


        # Create stream
        stream = stt.stream()

        # Generate test audio (1 second at 16kHz)
        test_audio = generate_test_audio(duration_ms=1000, sample_rate=16000)

        # Send audio in chunks
        chunk_size = 1600  # 100ms at 16kHz
        for i in range(0, len(test_audio), chunk_size):
            chunk = test_audio[i:i+chunk_size]

            # Create AudioFrame
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(chunk),
            )

            stream.push_frame(frame)

        # Signal end of input
        stream.end_input()

        # Collect events
        events = []
        async for event in stream:
            events.append(event)

            # Break after END_OF_SPEECH
            if event.type == SpeechEventType.END_OF_SPEECH:
                break

        # Cleanup
        await stt.aclose()

        # Verify we got events
        assert len(events) > 0

        # Verify event types
        event_types = [e.type for e in events]
        assert SpeechEventType.START_OF_SPEECH in event_types
        assert SpeechEventType.FINAL_TRANSCRIPT in event_types

    @pytest.mark.asyncio
    async def test_event_sequence_correct_order(self, mock_voxist_server, generate_test_audio):
        """Test events are emitted in correct sequence."""
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
            language="fr",
            interim_results=True,
        )

        stream = stt.stream()

        # Send audio
        test_audio = generate_test_audio(duration_ms=1000)
        chunk_size = 1600

        for i in range(0, len(test_audio), chunk_size):
            chunk = test_audio[i:i+chunk_size]
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            stream.push_frame(frame)
            await asyncio.sleep(0.05)  # Small delay between frames

        stream.end_input()

        # Collect events
        events = []
        async for event in stream:
            events.append(event)
            if event.type == SpeechEventType.END_OF_SPEECH:
                break

        await stt.aclose()

        # Verify sequence
        assert len(events) >= 3  # At least START, INTERIM/FINAL, END

        # First event should be START_OF_SPEECH
        assert events[0].type == SpeechEventType.START_OF_SPEECH

        # Last event should be END_OF_SPEECH
        assert events[-1].type == SpeechEventType.END_OF_SPEECH

        # Should have at least one transcript (interim or final)
        transcript_events = [
            e for e in events
            if e.type in (SpeechEventType.INTERIM_TRANSCRIPT, SpeechEventType.FINAL_TRANSCRIPT)
        ]
        assert len(transcript_events) > 0

    @pytest.mark.asyncio
    async def test_transcription_content(self, mock_voxist_server, generate_test_audio):
        """Test transcription content is received correctly."""
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
        )

        stream = stt.stream()

        # Send audio
        test_audio = generate_test_audio(duration_ms=500)
        chunk_size = 1600

        for i in range(0, len(test_audio), chunk_size):
            chunk = test_audio[i:i+chunk_size]
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            stream.push_frame(frame)

        stream.end_input()

        # Find final transcript
        final_text = None
        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                final_text = event.alternatives[0].text
                break

        await stt.aclose()

        # Verify transcription
        assert final_text is not None
        assert "bonjour monde" == final_text
        assert len(event.alternatives) > 0
        assert event.alternatives[0].confidence > 0.8


@pytest.mark.integration
class TestMultiLanguage:
    """Test multi-language support."""

    @pytest.mark.asyncio
    async def test_french_language(self, generate_test_audio):
        """Test French language transcription."""
        server = MockVoxistServer(
            valid_api_key="test",
            transcription_text="bonjour le monde",
        )
        await server.start()

        stt = VoxistSTT(
            api_key="test",
            base_url=f"ws://{server.host}:{server.port}/ws",
            language="fr",
        )

        stream = stt.stream()

        # Send minimal audio
        test_audio = generate_test_audio(duration_ms=500)
        frame = rtc.AudioFrame(
            data=test_audio.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(test_audio),
        )
        stream.push_frame(frame)
        stream.end_input()

        # Get result
        final_text = None
        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                final_text = event.alternatives[0].text
                assert event.alternatives[0].language == "fr"
                break

        await stt.aclose()
        await server.stop()

        assert final_text == "bonjour le monde"

    @pytest.mark.asyncio
    async def test_medical_french_language(self, generate_test_audio):
        """Test French medical language configuration."""
        server = MockVoxistServer(
            valid_api_key="test",
            transcription_text="20 milligrammes",  # Simulated text2num output
        )
        await server.start()

        stt = VoxistSTT(
            api_key="test",
            base_url=f"ws://{server.host}:{server.port}/ws",
            language="fr-medical",
        )

        stream = stt.stream()

        # Send audio
        test_audio = generate_test_audio(duration_ms=500)
        frame = rtc.AudioFrame(
            data=test_audio.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(test_audio),
        )
        stream.push_frame(frame)
        stream.end_input()

        # Get result
        final_text = None
        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                final_text = event.alternatives[0].text
                # livekit normalizes the code to BCP-47 on SpeechData, which
                # uppercases the subtag: "fr-medical" is emitted "fr-MEDICAL".
                # Compare case-insensitively - the exact casing is livekit's.
                assert event.alternatives[0].language.lower() == "fr-medical"
                break

        await stt.aclose()
        await server.stop()

        assert final_text == "20 milligrammes"


@pytest.mark.integration
class TestSessionDialing:
    """Per-stream dialing: token reuse across sessions, isolation between them."""

    @pytest.mark.asyncio
    async def test_token_cached_across_streams(self, generate_test_audio):
        """The HTTPS token exchange happens once, not once per stream."""
        server = MockVoxistServer(valid_api_key="test")
        await server.start()

        stt = VoxistSTT(
            api_key="test", base_url=f"ws://{server.host}:{server.port}/ws"
        )

        async def run_stream():
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
            test_audio = generate_test_audio(duration_ms=300)
            stream.push_frame(
                rtc.AudioFrame(
                    data=test_audio.tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=len(test_audio),
                )
            )
            stream.end_input()
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    return event.alternatives[0].text

        first = await run_stream()
        second = await run_stream()

        await stt.aclose()
        await server.stop()

        assert first == second == "bonjour monde"
        assert server.connections_count == 2, "each stream dials its own socket"
        assert server.token_requests_count == 1, (
            "the token must be exchanged once and cached, not per stream"
        )

    @pytest.mark.asyncio
    async def test_warm_up_prefetches_the_token(self):
        """wait_for_initialization() caches the token before the first stream."""
        server = MockVoxistServer(valid_api_key="test")
        await server.start()

        stt = VoxistSTT(
            api_key="test", base_url=f"ws://{server.host}:{server.port}/ws"
        )
        ready = await stt.wait_for_initialization(timeout=5.0)
        # Read BEFORE aclose(). is_ready reports the readiness of a LIVE
        # plugin, and a closed one is never ready (stream() raises on it), so
        # asserting it after the teardown below asserted the opposite of what
        # this test is about.
        ready_property = stt.is_ready

        await stt.aclose()
        await server.stop()

        assert ready is True
        assert ready_property is True
        assert server.token_requests_count == 1

    @pytest.mark.asyncio
    async def test_concurrent_streams_are_isolated(self, generate_test_audio):
        """Concurrent streams run on separate sockets and both complete."""
        server = MockVoxistServer(valid_api_key="test")
        await server.start()

        stt = VoxistSTT(
            api_key="test", base_url=f"ws://{server.host}:{server.port}/ws"
        )

        test_audio = generate_test_audio(duration_ms=300)

        async def run_stream():
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
            stream.push_frame(
                rtc.AudioFrame(
                    data=test_audio.tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=len(test_audio),
                )
            )
            stream.end_input()

            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    return event.alternatives[0].text

        results = await asyncio.gather(run_stream(), run_stream())

        await stt.aclose()
        await server.stop()

        assert list(results) == ["bonjour monde", "bonjour monde"]
        assert server.connections_count == 2


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_authentication_failure(self):
        """A rejected key surfaces as AuthenticationError, not a retry loop."""
        server = MockVoxistServer(error_mode="auth_failure")
        await server.start()

        stt = VoxistSTT(
            api_key="any_key", base_url=f"ws://{server.host}:{server.port}/ws"
        )

        from livekit.plugins.voxist.exceptions import AuthenticationError

        # Warm-up reports it...
        ready = await stt.wait_for_initialization(timeout=5.0)
        assert ready is False
        assert isinstance(stt.initialization_error, AuthenticationError)

        # ...and a stream fails with the true cause rather than a generic
        # connection error. Deliberately not an APIError: livekit would retry
        # those, and retrying cannot fix a revoked key. The stream carries
        # audio: a zero-audio end_input() correctly never dials at all, so
        # it would never even reach authentication.
        stream = stt.stream()
        stream.push_frame(
            rtc.AudioFrame(
                data=np.zeros(1600, dtype=np.int16).tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=1600,
            )
        )
        stream.end_input()
        with pytest.raises(AuthenticationError):
            async for _event in stream:
                pass

        await stt.aclose()
        await server.stop()

    async def test_unreachable_server_fails_after_bounded_retries(
        self, generate_test_audio
    ):
        """
        A dead endpoint fails the stream after livekit's retry budget.

        Retries are owned by RecognizeStream._main_task (conn_options), not by
        the plugin: an earlier design kept its own reconnect loop inside the
        framework's, and its budget could be made unreachable. This pins the
        bounded behaviour end to end.
        """
        from livekit.agents import APIConnectionError
        from livekit.agents.types import APIConnectOptions

        stt = VoxistSTT(
            api_key="test",
            base_url="ws://127.0.0.1:9/ws",  # discard port: refused
        )

        stream = stt.stream(
            conn_options=APIConnectOptions(
                max_retry=1, retry_interval=0.1, timeout=2.0
            )
        )
        test_audio = generate_test_audio(duration_ms=100)
        stream.push_frame(
            rtc.AudioFrame(
                data=test_audio.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(test_audio),
            )
        )
        stream.end_input()

        with pytest.raises(APIConnectionError):
            async for _event in stream:
                pass

        await stt.aclose()


@pytest.mark.integration
class TestServerStallDetection:
    """A wedged engine behind a live gateway must fail the attempt, fast."""

    @pytest.mark.asyncio
    async def test_wedged_engine_mid_session_is_detected(
        self, monkeypatch, generate_test_audio
    ):
        """
        End to end against the mock's wedge mode: the WS layer stays up and
        keeps accepting audio, but the engine never answers. The send-aware
        liveness bound must abandon the attempt instead of streaming into
        the void until end_input.
        """
        from livekit.agents import APIConnectionError

        from livekit.plugins.voxist.stream import VoxistSTTStream

        monkeypatch.setattr(VoxistSTTStream, "STALL_DETECTION_SECONDS", 1.0)

        server = MockVoxistServer(valid_api_key="test", error_mode="wedge")
        await server.start()
        try:
            stt = VoxistSTT(
                api_key="test", base_url=f"ws://{server.host}:{server.port}/ws"
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

            speech = generate_test_audio(duration_ms=100)
            stopped = asyncio.Event()

            async def pump():
                # Live microphone: real audio keeps flowing
                while not stopped.is_set():
                    try:
                        stream.push_frame(
                            rtc.AudioFrame(
                                data=speech.tobytes(),
                                sample_rate=16000,
                                num_channels=1,
                                samples_per_channel=len(speech),
                            )
                        )
                    except RuntimeError:
                        return  # stream already dead
                    await asyncio.sleep(0.05)

            async def drain():
                async for _event in stream:
                    pass

            pump_task = asyncio.create_task(pump())
            # monotonic, not time(): an NTP step mid-test must not decide
            # whether CI passes.
            start = time.monotonic()
            try:
                # Bounded: a detector that never fires leaves this stream
                # open forever (the wedge never answers and never closes),
                # and a regression must FAIL, not hang CI.
                with pytest.raises(APIConnectionError, match="no response"):
                    await asyncio.wait_for(drain(), timeout=10.0)
            finally:
                stopped.set()
                await pump_task
            elapsed = time.monotonic() - start

            assert elapsed < 10.0, (
                "the stall must be detected around STALL_DETECTION_SECONDS, "
                "not discovered at end of input"
            )
            assert server.audio_frames_received > 0, (
                "precondition: audio really was flowing into the wedge"
            )
            await stt.aclose()
        finally:
            await server.stop()


    @pytest.mark.asyncio
    async def test_quiet_speaker_wedge_is_detected(
        self, monkeypatch, generate_test_audio
    ):
        """
        Same wedge, but the speaker is quiet or under-gained: this audio's
        peak sits far below the amplitude gate the detector used to arm on,
        so the clock never started and the wedge was only discovered as zero
        transcripts at end_input - terminally, instead of retrying mid-call.
        Counting delivered bytes does not care how loud the speaker is.
        """
        from livekit.agents import APIConnectionError

        from livekit.plugins.voxist.stream import VoxistSTTStream

        monkeypatch.setattr(VoxistSTTStream, "STALL_DETECTION_SECONDS", 1.0)

        server = MockVoxistServer(valid_api_key="test", error_mode="wedge")
        await server.start()
        try:
            stt = VoxistSTT(
                api_key="test", base_url=f"ws://{server.host}:{server.port}/ws"
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

            # Real speech, 60 LSB peak: inaudible to a 500-amplitude gate.
            quiet = (generate_test_audio(duration_ms=100) // 546).astype(np.int16)
            assert 0 < int(np.abs(quiet.astype(np.int32)).max()) < 500, (
                "precondition: this speaker must be below the old gate"
            )
            stopped = asyncio.Event()

            async def pump():
                while not stopped.is_set():
                    try:
                        stream.push_frame(
                            rtc.AudioFrame(
                                data=quiet.tobytes(),
                                sample_rate=16000,
                                num_channels=1,
                                samples_per_channel=len(quiet),
                            )
                        )
                    except RuntimeError:
                        return  # stream already dead
                    await asyncio.sleep(0.05)

            async def drain():
                async for _event in stream:
                    pass

            pump_task = asyncio.create_task(pump())
            start = time.monotonic()
            try:
                # Bounded on purpose: the failure mode being tested is a
                # detector that never fires, and that must not hang CI.
                with pytest.raises(APIConnectionError, match="no response"):
                    await asyncio.wait_for(drain(), timeout=10.0)
            finally:
                stopped.set()
                await pump_task
            elapsed = time.monotonic() - start

            assert elapsed < 10.0, (
                "a quiet speaker's wedged server must be detected around "
                "STALL_DETECTION_SECONDS too, not at end of input"
            )
            assert server.audio_frames_received > 0
            await stt.aclose()
        finally:
            await server.stop()


@pytest.mark.integration
class TestSessionEndingWithoutTranscript:
    """
    A gateway whose engine crashed closes the socket promptly after "Done"
    without ever sending a transcript. From the plugin's side that is
    indistinguishable from a heartbeat death: the receive iterator simply
    ends. It must never be reported as a clean, successful session.
    """

    @pytest.mark.asyncio
    async def test_prompt_close_with_no_transcript_is_not_success(
        self, generate_test_audio
    ):
        from livekit.plugins.voxist.exceptions import TranscriptLostError

        # transcription_text="" models the crashed engine: the gateway
        # forwards Done, nothing comes back, and it closes the client socket.
        server = MockVoxistServer(
            valid_api_key="test", transcription_text="", send_interim=False
        )
        await server.start()
        try:
            stt = VoxistSTT(
                api_key="test", base_url=f"ws://{server.host}:{server.port}/ws"
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

            speech = generate_test_audio(duration_ms=500)
            stream.push_frame(
                rtc.AudioFrame(
                    data=speech.tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=len(speech),
                )
            )
            stream.end_input()

            events = []
            with pytest.raises(TranscriptLostError):
                async for event in stream:
                    events.append(event)

            assert server.done_received_count == 1, (
                "precondition: the session really did reach end of input"
            )
            assert not [
                e for e in events
                if e.type == SpeechEventType.FINAL_TRANSCRIPT
            ], "precondition: nothing was delivered"
            await stt.aclose()
        finally:
            await server.stop()


@pytest.mark.integration
class TestTurnBoundarySilence:
    """
    flush() immediately followed by end_input() is the commonest VAD pattern
    there is, and it was untested: two adjacent sentinels with no frames
    between them. "Done" forces the engine flush by itself, so any
    endpointing silence shipped here is pure dead air at the end of a turn.
    """

    @pytest.mark.asyncio
    async def test_flush_then_end_input_ships_no_endpointing_silence(
        self, generate_test_audio
    ):
        received: list[bytes] = []

        server = MockVoxistServer(
            valid_api_key="test",
            on_audio_received=lambda data, _samples: received.append(bytes(data)),
        )
        await server.start()
        try:
            stt = VoxistSTT(
                api_key="test", base_url=f"ws://{server.host}:{server.port}/ws"
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

            speech = generate_test_audio(duration_ms=500)
            stream.push_frame(
                rtc.AudioFrame(
                    data=speech.tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=len(speech),
                )
            )
            stream.flush()      # end of segment...
            stream.end_input()  # ...and end of session, back to back

            async def drain():
                async for _ in stream:
                    pass
            await asyncio.wait_for(drain(), timeout=15.0)
            await stt.aclose()

            silent_chunks = [c for c in received if not any(c)]
            assert silent_chunks == [], (
                f"{len(silent_chunks)} chunks of endpointing silence were "
                "shipped before Done: 400ms of dead air on the commonest "
                "turn ending there is"
            )
            assert server.done_received_count == 1, "exactly one Done"
            assert server.connections_count == 1, "one socket for the session"
        finally:
            await server.stop()


@pytest.mark.integration
class TestZeroAudioSession:
    """end_input() with no frames: clean empty completion, zero network."""

    @pytest.mark.asyncio
    async def test_zero_audio_session_never_dials(self):
        server = MockVoxistServer(valid_api_key="test")
        await server.start()
        try:
            stt = VoxistSTT(
                api_key="test", base_url=f"ws://{server.host}:{server.port}/ws"
            )
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))
            # No await between stream() and end_input(): _run has not begun,
            # so the zero-audio short-circuit decides deterministically.
            stream.end_input()

            events = [event async for event in stream]

            await stt.aclose()

            assert events == [], "an empty session owes no events"
            assert server.connections_count == 0, (
                "a session with no audio must not dial: there is nothing "
                "to transcribe"
            )
        finally:
            await server.stop()


@pytest.mark.integration
class TestPerformance:
    """Test performance characteristics."""

    @pytest.mark.asyncio
    async def test_latency_measurement(self, mock_voxist_server, generate_test_audio):
        """Measure end-to-end latency."""
        # Set mock server to fast mode (10ms processing)
        mock_voxist_server.processing_delay_ms = 10
        mock_voxist_server.interim_delay_ms = 5

        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
        )


        # Measure time to first final transcript
        test_audio = generate_test_audio(duration_ms=500)

        start_time = time.time()

        stream = stt.stream()

        # Send audio quickly
        chunk_size = 1600
        for i in range(0, len(test_audio), chunk_size):
            chunk = test_audio[i:i+chunk_size]
            frame = rtc.AudioFrame(
                data=chunk.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(chunk),
            )
            stream.push_frame(frame)

        stream.end_input()

        # Wait for final transcript
        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                latency = (time.time() - start_time) * 1000  # ms
                break

        await stt.aclose()

        # Latency should be reasonable (< 1000ms for this test)
        # In production with real API, should be < 300ms
        assert latency < 1000

        print(f"\nMeasured latency: {latency:.1f}ms")

    @pytest.mark.asyncio
    async def test_warm_token_keeps_first_stream_fast(self, generate_test_audio):
        """With the token prefetched, the first stream avoids the HTTPS trip."""
        server = MockVoxistServer(valid_api_key="test")
        await server.start()

        stt_pooled = VoxistSTT(
            api_key="test",
            base_url=f"ws://{server.host}:{server.port}/ws",
        )

        assert await stt_pooled.wait_for_initialization(timeout=5.0)

        test_audio = generate_test_audio(duration_ms=300)

        # Measure pooled connection acquisition
        start = time.time()
        stream = stt_pooled.stream()
        frame = rtc.AudioFrame(
            data=test_audio.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(test_audio),
        )
        stream.push_frame(frame)
        stream.end_input()

        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                break

        pooled_time = time.time() - start

        await stt_pooled.aclose()
        await server.stop()

        # Dial + stream + final, with no token round-trip in the path
        assert pooled_time < 1.0
        assert server.token_requests_count == 1


@pytest.mark.integration
class TestStreamLifecycle:
    """Test stream lifecycle and cleanup."""

    @pytest.mark.asyncio
    async def test_multiple_sequential_streams(self, mock_voxist_server, generate_test_audio):
        """Test multiple streams can be created sequentially."""
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
        )


        test_audio = generate_test_audio(duration_ms=300)

        # Run 3 streams sequentially. max_retry=0: a silent re-dial would
        # inflate connections_count and mask a mid-session failure.
        for i in range(3):
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

            frame = rtc.AudioFrame(
                data=test_audio.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(test_audio),
            )
            stream.push_frame(frame)
            stream.end_input()

            got_final = False
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    got_final = True
                    break

            assert got_final, f"Stream {i} did not get final transcript"

        await stt.aclose()

        # One socket per session, one Done per session, none left open
        assert mock_voxist_server.connections_count == 3
        assert mock_voxist_server.done_received_count == 3

    @pytest.mark.asyncio
    async def test_stream_cleanup_closes_its_socket(self, mock_voxist_server, generate_test_audio):
        """A completed stream leaves no socket behind."""
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
        )

        # Create and run stream; no retries, so a socket-lifecycle failure
        # surfaces instead of being recovered into a false pass.
        stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

        test_audio = generate_test_audio(duration_ms=300)
        frame = rtc.AudioFrame(
            data=test_audio.tobytes(),
            sample_rate=16000,
            num_channels=1,
            samples_per_channel=len(test_audio),
        )
        stream.push_frame(frame)
        stream.end_input()

        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                break

        # Give time for cleanup
        await asyncio.sleep(0.1)

        # The session ended with exactly one Done and the stream's own socket;
        # aclose() then has nothing to tear down but the HTTP session.
        assert mock_voxist_server.done_received_count == 1
        assert mock_voxist_server.connections_count == 1

        await stt.aclose()


@pytest.mark.integration
class TestMultiTurnConversation:
    """
    The test that gates the architecture: a VAD-driven multi-turn stream.

    livekit's turn detection calls flush() at each speech end WITHOUT closing
    the input (only end_input() closes it), and the engine emits a final per
    silence-delimited segment on ONE socket. The pooled architecture conflated
    flush() with end-of-session ("Done"), after which the gateway closes the
    socket - so everything after the first turn was lost, misclassified, or
    fed through reconnect machinery. This test drives two turns end to end and
    accepts nothing less than both transcripts and a clean completion.
    """

    @pytest.mark.asyncio
    async def test_two_turns_produce_two_finals(self, generate_test_audio):
        server = MockVoxistServer(
            valid_api_key="test",
            transcription_text="bonjour le monde",
        )
        await server.start()

        try:
            stt = VoxistSTT(
                api_key="test",
                base_url=f"ws://{server.host}:{server.port}/ws",
                language="fr",
            )

            # max_retry=0 keeps this test honest: with livekit's default
            # retry budget, a flush() that regressed to sending "Done" (the
            # server then closes the socket) would be silently re-dialed and
            # turn 2 would arrive on a SECOND socket - >=2 finals plus
            # END_OF_SPEECH would still be observed and the regression would
            # pass. With no retries, any mid-conversation failure surfaces,
            # and the socket/Done invariants below pin the architecture.
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

            def push(samples):
                stream.push_frame(
                    rtc.AudioFrame(
                        data=samples.tobytes(),
                        sample_rate=16000,
                        num_channels=1,
                        samples_per_channel=len(samples),
                    )
                )

            speech = generate_test_audio(duration_ms=600)
            silence = np.zeros(16000 // 2, dtype=np.int16)  # 500ms

            # Turn 1: speech, trailing silence, VAD end-of-turn
            push(speech)
            push(silence)
            stream.flush()

            # Turn 2 on the same stream
            await asyncio.sleep(0.3)
            push(speech)
            push(silence)
            stream.flush()

            # End of session
            stream.end_input()

            events = []
            async def collect():
                async for event in stream:
                    events.append(event)
            await asyncio.wait_for(collect(), timeout=15.0)

            await stt.aclose()

            finals = [
                e for e in events
                if e.type == SpeechEventType.FINAL_TRANSCRIPT
            ]
            assert len(finals) >= 2, (
                f"expected a final per turn, got {len(finals)}: the second "
                "turn was lost - flush() must not end the session"
            )
            for f in finals:
                assert f.alternatives[0].text == "bonjour le monde"

            end_events = [
                e for e in events if e.type == SpeechEventType.END_OF_SPEECH
            ]
            assert end_events, "stream ended without END_OF_SPEECH"

            # The architectural invariants, asserted directly: the whole
            # conversation rides ONE socket, and "Done" is the end-of-SESSION
            # signal, written exactly once at end_input(). If flush() ever
            # ends the session again, these fail even if event counts look
            # healthy.
            assert server.connections_count == 1, (
                f"{server.connections_count} sockets for one conversation: "
                "the session was ended and re-dialed mid-stream - flush() "
                "must not end the session"
            )
            assert server.done_received_count == 1, (
                f"Done received {server.done_received_count} times; it is "
                "the end-of-session signal and must be sent exactly once, "
                "at end_input() - never at a flush() turn boundary"
            )
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_session_ends_cleanly_after_done(self, generate_test_audio):
        """After end_input, Done is sent once and the server closes the socket."""
        server = MockVoxistServer(
            valid_api_key="test",
            transcription_text="fin de session",
        )
        await server.start()

        try:
            stt = VoxistSTT(
                api_key="test",
                base_url=f"ws://{server.host}:{server.port}/ws",
                language="fr",
            )
            # No retries: a re-dial would hide a lifecycle failure behind a
            # second socket and a second Done.
            stream = stt.stream(conn_options=APIConnectOptions(max_retry=0))

            speech = generate_test_audio(duration_ms=500)
            stream.push_frame(
                rtc.AudioFrame(
                    data=speech.tobytes(),
                    sample_rate=16000,
                    num_channels=1,
                    samples_per_channel=len(speech),
                )
            )
            stream.end_input()

            async def drain():
                async for _ in stream:
                    pass
            await asyncio.wait_for(drain(), timeout=15.0)
            await stt.aclose()

            assert server.done_received_count == 1, (
                f"Done sent {server.done_received_count} times; it is the "
                "end-of-session signal and must be sent exactly once"
            )
            assert server.connections_count == 1, (
                f"{server.connections_count} sockets for one session: the "
                "stream must live and die on a single connection"
            )
        finally:
            await server.stop()
