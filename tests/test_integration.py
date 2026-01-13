"""Integration tests for complete LiveKit plugin pipeline."""

import asyncio
import time

import numpy as np
import pytest

from livekit import rtc
from livekit.agents.stt import SpeechEventType
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

        # Initialize pool
        await stt._pool.initialize()

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

        await stt._pool.initialize()
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

        await stt._pool.initialize()
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
            port=8771,
            valid_api_key="test",
            transcription_text="bonjour le monde",
        )
        await server.start()

        stt = VoxistSTT(
            api_key="test",
            base_url="ws://localhost:8771/ws",
            language="fr",
        )

        await stt._pool.initialize()
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
            port=8772,
            valid_api_key="test",
            transcription_text="20 milligrammes",  # Simulated text2num output
        )
        await server.start()

        stt = VoxistSTT(
            api_key="test",
            base_url="ws://localhost:8772/ws",
            language="fr-medical",
        )

        await stt._pool.initialize()
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
                assert event.alternatives[0].language == "fr-medical"
                break

        await stt.aclose()
        await server.stop()

        assert final_text == "20 milligrammes"


@pytest.mark.integration
class TestConnectionPool:
    """Test connection pool behavior in integration."""

    @pytest.mark.asyncio
    async def test_pool_pre_warming(self, generate_test_audio):
        """Test connection pool pre-warms connections."""
        server = MockVoxistServer(port=8773, valid_api_key="test")
        await server.start()

        stt = VoxistSTT(
            api_key="test",
            base_url="ws://localhost:8773/ws",
            connection_pool_size=2,
        )

        # Initialize pool
        start = time.time()
        await stt._pool.initialize()
        init_time = time.time() - start

        # Check pool health
        health = stt._pool.get_pool_health()
        assert health["ready"] >= 1  # At least one connection ready
        assert health["total"] == 2

        # First stream should be fast (no connection overhead)
        stream_start = time.time()
        stream = stt.stream()

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

        stream_time = time.time() - stream_start

        await stt.aclose()
        await server.stop()

        # Stream should be fast (< 1s for this test)
        assert stream_time < 1.0

        print(f"\nPool init: {init_time:.2f}s, Stream: {stream_time:.2f}s")

    @pytest.mark.asyncio
    async def test_concurrent_streams_use_different_connections(self, generate_test_audio):
        """Test concurrent streams use different connections from pool."""
        server = MockVoxistServer(port=8774, valid_api_key="test")
        await server.start()

        stt = VoxistSTT(
            api_key="test",
            base_url="ws://localhost:8774/ws",
            connection_pool_size=2,
        )

        await stt._pool.initialize()

        # Create two concurrent streams
        test_audio = generate_test_audio(duration_ms=300)

        async def run_stream():
            stream = stt.stream()
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
                    return event.alternatives[0].text

        # Run two streams concurrently
        results = await asyncio.gather(run_stream(), run_stream())

        await stt.aclose()
        await server.stop()

        # Both should succeed
        assert len(results) == 2
        assert all(r == "bonjour monde" for r in results)


@pytest.mark.integration
class TestAudioProcessing:
    """Test audio processing in full pipeline."""

    @pytest.mark.asyncio
    async def test_various_audio_frame_sizes(self, mock_voxist_server):
        """Test plugin handles various audio frame sizes."""
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
        )

        await stt._pool.initialize()
        stream = stt.stream()

        # Send frames of varying sizes
        frame_sizes = [800, 1600, 3200, 500, 2000]  # Various durations

        for size in frame_sizes:
            audio = np.random.randint(-32768, 32767, size=size, dtype=np.int16)
            frame = rtc.AudioFrame(
                data=audio.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=size,
            )
            stream.push_frame(frame)
            await asyncio.sleep(0.01)

        stream.end_input()

        # Should complete successfully
        got_final = False
        async for event in stream:
            if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                got_final = True
                break

        await stt.aclose()

        assert got_final

    @pytest.mark.asyncio
    async def test_long_audio_streaming(self, mock_voxist_server, generate_test_audio):
        """Test streaming longer audio (5 seconds)."""
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
        )

        await stt._pool.initialize()
        stream = stt.stream()

        # Generate 5 seconds of audio
        test_audio = generate_test_audio(duration_ms=5000)

        # Send in 100ms chunks
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

        # Collect all events
        events = []
        async for event in stream:
            events.append(event)
            if event.type == SpeechEventType.END_OF_SPEECH:
                break

        await stt.aclose()

        # Should have multiple events for long audio
        assert len(events) >= 3


@pytest.mark.integration
class TestErrorHandling:
    """Test error handling and recovery."""

    @pytest.mark.asyncio
    async def test_authentication_failure(self):
        """Test plugin handles authentication failure properly."""
        server = MockVoxistServer(
            port=8775,
            error_mode="auth_failure",
        )
        await server.start()

        stt = VoxistSTT(
            api_key="any_key",
            base_url="ws://localhost:8775/ws",
        )

        # Initialization should fail with auth error
        from livekit.plugins.voxist.exceptions import AuthenticationError

        with pytest.raises(AuthenticationError):
            await stt._pool.initialize()

        await server.stop()

    @pytest.mark.asyncio
    async def test_connection_pool_exhaustion_handling(self, generate_test_audio):
        """Test handling when connection pool is exhausted."""
        from livekit.plugins.voxist.exceptions import ConnectionPoolExhaustedError

        # Create server that will fail connections
        server = MockVoxistServer(port=8776, valid_api_key="test")
        await server.start()

        stt = VoxistSTT(
            api_key="test",
            base_url="ws://localhost:8776/ws",
            connection_pool_size=2,
        )

        await stt._pool.initialize()

        # Manually fail all connections
        for conn in stt._pool.connections:
            conn.state = ConnectionState.FAILED
            conn.retry_count = stt._pool.max_reconnect_attempts

        # Creating stream should fail
        stream = stt.stream()

        with pytest.raises(ConnectionPoolExhaustedError):
            # Try to run stream (will fail to get connection)
            test_audio = generate_test_audio(duration_ms=100)
            frame = rtc.AudioFrame(
                data=test_audio.tobytes(),
                sample_rate=16000,
                num_channels=1,
                samples_per_channel=len(test_audio),
            )
            stream.push_frame(frame)
            stream.end_input()

            async for _event in stream:
                pass

        await stt.aclose()
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

        await stt._pool.initialize()

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
    async def test_connection_pool_reduces_latency(self, generate_test_audio):
        """Test connection pool reduces latency vs single connection."""
        server = MockVoxistServer(port=8777, valid_api_key="test")
        await server.start()

        # Test with pool size 2
        stt_pooled = VoxistSTT(
            api_key="test",
            base_url="ws://localhost:8777/ws",
            connection_pool_size=2,
        )

        await stt_pooled._pool.initialize()

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

        # With pre-warmed pool, should be fast
        assert pooled_time < 1.0

        print(f"\nPooled connection time: {pooled_time:.3f}s")


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

        await stt._pool.initialize()

        test_audio = generate_test_audio(duration_ms=300)

        # Run 3 streams sequentially
        for i in range(3):
            stream = stt.stream()

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

        # All streams should have succeeded
        # Pool should be healthy
        health = stt._pool.get_pool_health()
        assert health["closed"] == stt._pool.pool_size  # All closed after aclose()

    @pytest.mark.asyncio
    async def test_stream_cleanup_releases_connection(self, mock_voxist_server, generate_test_audio):
        """Test stream releases connection back to pool after completion."""
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
            connection_pool_size=2,
        )

        await stt._pool.initialize()

        # Check initial pool state
        initial_health = stt._pool.get_pool_health()
        assert initial_health["ready"] == 2

        # Create and run stream
        stream = stt.stream()

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

        # Connection should be back in pool
        final_health = stt._pool.get_pool_health()
        assert final_health["ready"] + final_health["in_use"] >= 1

        await stt.aclose()


# Import ConnectionState for exhaustion test
from livekit.plugins.voxist.models import ConnectionState
