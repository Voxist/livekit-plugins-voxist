"""Tests for MockVoxistServer."""

import asyncio

import aiohttp
import numpy as np
import pytest

from .fixtures.mock_server import MockVoxistServer


class TestMockServerBasics:
    """Test basic mock server functionality."""

    @pytest.mark.asyncio
    async def test_server_starts_and_stops(self):
        """Test server can start and stop."""
        server = MockVoxistServer(port=8766)

        await server.start()
        assert server.runner is not None
        assert server.site is not None

        await server.stop()

    @pytest.mark.asyncio
    async def test_server_accepts_connection(self, mock_voxist_server):
        """Test server accepts WebSocket connections."""
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key"

            async with session.ws_connect(url) as ws:
                # Should receive connection confirmation
                msg = await ws.receive()
                assert msg.type == aiohttp.WSMsgType.TEXT

                data = msg.json()
                assert data["status"] == "connected"

    @pytest.mark.asyncio
    async def test_server_rejects_invalid_api_key(self, mock_voxist_server):
        """Test server rejects connections with invalid API key."""
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=invalid"

            async with session.ws_connect(url) as ws:
                # Server should close connection
                msg = await ws.receive()
                assert msg.type == aiohttp.WSMsgType.CLOSE

    @pytest.mark.asyncio
    async def test_server_handles_config_message(self, mock_voxist_server):
        """Test server accepts config message."""
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key"

            async with session.ws_connect(url) as ws:
                # Receive connection confirmation
                await ws.receive()

                # Send config
                config = {"config": {"lang": "fr", "sample_rate": 16000}}
                await ws.send_json(config)

                # Server should process it (no response expected for config)
                # Just verify connection stays open
                await asyncio.sleep(0.1)
                assert not ws.closed

    @pytest.mark.asyncio
    async def test_server_processes_audio_and_returns_transcription(self, mock_voxist_server):
        """Test server receives audio and sends transcription."""
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key"

            async with session.ws_connect(url) as ws:
                # Receive connection confirmation
                await ws.receive()

                # Send config
                await ws.send_json({"config": {"lang": "fr", "sample_rate": 16000}})

                # Send audio frames (Float32 format, 1600 samples = 100ms at 16kHz)
                for _ in range(3):
                    audio_float32 = np.random.rand(1600).astype(np.float32)
                    await ws.send_bytes(audio_float32.tobytes())
                    await asyncio.sleep(0.01)

                # Should receive interim result (after 1st frame)
                msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                assert msg.type == aiohttp.WSMsgType.TEXT
                interim = msg.json()
                assert interim["type"] == "partial"
                assert "bonjour" in interim["text"].lower()

                # Should receive final result (after 3 frames)
                msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                assert msg.type == aiohttp.WSMsgType.TEXT
                final = msg.json()
                assert final["type"] == "final"
                assert final["text"] == "bonjour monde"
                assert final["confidence"] == 0.95

    @pytest.mark.asyncio
    async def test_server_handles_done_signal(self, mock_voxist_server):
        """Test server handles Done signal properly."""
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key"

            async with session.ws_connect(url) as ws:
                # Receive connection confirmation
                await ws.receive()

                # Send Done signal
                await ws.send_str("Done")

                # Server should break from message loop
                # Wait for any final messages or close
                try:
                    msg = await asyncio.wait_for(ws.receive(), timeout=0.5)
                    # Server might send final result or close
                    assert msg.type in (aiohttp.WSMsgType.TEXT, aiohttp.WSMsgType.CLOSE)
                except asyncio.TimeoutError:
                    # Or connection stays open (that's fine)
                    pass

    @pytest.mark.asyncio
    async def test_server_tracks_statistics(self):
        """Test server tracks connection and audio statistics."""
        server = MockVoxistServer(port=8767, valid_api_key="test")

        await server.start()

        async with aiohttp.ClientSession() as session:
            url = "ws://localhost:8767/ws?api_key=test"

            async with session.ws_connect(url) as ws:
                await ws.receive()  # Connection confirmation

                # Send 5 audio frames
                for _ in range(5):
                    audio = np.random.rand(1600).astype(np.float32)
                    await ws.send_bytes(audio.tobytes())

                await asyncio.sleep(0.1)

        # Check stats
        stats = server.get_stats()
        assert stats["connections_count"] == 1
        assert stats["audio_frames_received"] == 5
        assert stats["total_audio_bytes"] == 5 * 1600 * 4  # 5 frames * 1600 samples * 4 bytes

        await server.stop()

    @pytest.mark.asyncio
    async def test_server_handles_multiple_concurrent_connections(self):
        """Test server handles multiple connections simultaneously."""
        server = MockVoxistServer(port=8768, valid_api_key="test")

        await server.start()

        async def connect_and_send():
            async with aiohttp.ClientSession() as session:
                url = "ws://localhost:8768/ws?api_key=test"
                async with session.ws_connect(url) as ws:
                    await ws.receive()  # Confirmation
                    audio = np.random.rand(1600).astype(np.float32)
                    await ws.send_bytes(audio.tobytes())
                    await asyncio.sleep(0.1)

        # Connect 3 clients concurrently
        await asyncio.gather(
            connect_and_send(),
            connect_and_send(),
            connect_and_send(),
        )

        stats = server.get_stats()
        assert stats["connections_count"] == 3

        await server.stop()


class TestMockServerErrorSimulation:
    """Test mock server error simulation capabilities."""

    @pytest.mark.asyncio
    async def test_server_auth_failure_mode(self):
        """Test server can simulate authentication failures."""
        server = MockVoxistServer(
            port=8769,
            error_mode="auth_failure"
        )

        await server.start()

        async with aiohttp.ClientSession() as session:
            url = "ws://localhost:8769/ws?api_key=any_key"

            async with session.ws_connect(url) as ws:
                # Should close with auth error
                msg = await ws.receive()
                assert msg.type == aiohttp.WSMsgType.CLOSE
                # Close code is in msg.data, message is in msg.extra
                assert msg.data == 1008  # Insufficient balance / auth failure code

        await server.stop()


class TestMockServerProtocolCompliance:
    """Test mock server implements Voxist protocol correctly."""

    @pytest.mark.asyncio
    async def test_protocol_sequence(self, mock_voxist_server):
        """Test complete protocol sequence."""
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key"

            async with session.ws_connect(url) as ws:
                # Step 1: Receive connection confirmation
                msg = await ws.receive()
                assert msg.type == aiohttp.WSMsgType.TEXT
                assert msg.json()["status"] == "connected"

                # Step 2: Send config
                await ws.send_json({"config": {"lang": "fr", "sample_rate": 16000}})

                # Step 3: Send binary audio (Float32)
                for _i in range(5):
                    audio = np.random.rand(1600).astype(np.float32)
                    await ws.send_bytes(audio.tobytes())
                    await asyncio.sleep(0.02)

                # Step 4: Receive interim result
                msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                assert msg.type == aiohttp.WSMsgType.TEXT
                partial = msg.json()
                assert partial["type"] == "partial"

                # Step 5: Receive final result
                msg = await asyncio.wait_for(ws.receive(), timeout=1.0)
                assert msg.type == aiohttp.WSMsgType.TEXT
                final = msg.json()
                assert final["type"] == "final"
                assert "confidence" in final

                # Step 6: Send Done signal
                await ws.send_str("Done")

    @pytest.mark.asyncio
    async def test_audio_format_validation(self, mock_voxist_server):
        """Test server correctly receives Int16 PCM audio (matches plugin format)."""
        received_frames = []

        def audio_callback(data: bytes, num_samples: int):
            # Verify Int16 format (2 bytes per sample)
            assert len(data) == num_samples * 2
            received_frames.append((data, num_samples))

        server = MockVoxistServer(
            port=8770,
            valid_api_key="test",
            on_audio_received=audio_callback,
        )

        await server.start()

        async with aiohttp.ClientSession() as session:
            url = "ws://localhost:8770/ws?api_key=test"

            async with session.ws_connect(url) as ws:
                await ws.receive()  # Confirmation

                # Send known Int16 PCM data (plugin sends Int16)
                test_audio = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
                await ws.send_bytes(test_audio.tobytes())

                await asyncio.sleep(0.1)

        # Verify callback was called
        assert len(received_frames) == 1
        data, num_samples = received_frames[0]
        assert num_samples == 5
        assert len(data) == 10  # 5 samples * 2 bytes

        # Verify we can parse it back
        received_audio = np.frombuffer(data, dtype=np.int16)
        np.testing.assert_array_equal(received_audio, test_audio)

        await server.stop()
