"""Tests for MockVoxistServer against the gateway-faithful contract.

The mock mirrors verified gateway behaviour (simple-websocket-proxy.gateway.ts):
no greeting frame on connect, a final per silence-delimited segment without
"Done", and "Done" as the end-of-SESSION signal after which the socket closes.
"""

import asyncio

import aiohttp
import numpy as np
import pytest

from .fixtures.mock_server import MockVoxistServer


def speech_frame(ms=100, rate=16000):
    """Non-silent Int16 audio (sine), as the plugin would send."""
    n = rate * ms // 1000
    t = np.linspace(0, ms / 1000.0, n, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16).tobytes()


def silence_frame(ms=100, rate=16000):
    return b"\x00" * (rate * ms // 1000 * 2)


async def collect_json(ws, timeout=2.0):
    """Drain currently-available JSON messages."""
    out = []
    try:
        while True:
            msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
            if msg.type != aiohttp.WSMsgType.TEXT:
                break
            out.append(msg.json())
    except asyncio.TimeoutError:
        pass
    return out


class TestMockServerBasics:
    @pytest.mark.asyncio
    async def test_server_starts_and_stops(self):
        server = MockVoxistServer(port=8766)
        await server.start()
        assert server.runner is not None
        assert server.site is not None
        await server.stop()

    @pytest.mark.asyncio
    async def test_no_greeting_on_connect(self, mock_voxist_server):
        """
        The gateway sends nothing on connect.

        The mock once sent {"status": "connected"}, which made it more
        talkative than production and kept an unreachable plugin branch
        looking alive. Connecting must yield silence until audio flows.
        """
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key&lang=fr"
            async with session.ws_connect(url) as ws:
                with pytest.raises(asyncio.TimeoutError):
                    await asyncio.wait_for(ws.receive(), timeout=0.3)

    @pytest.mark.asyncio
    async def test_server_rejects_invalid_api_key(self, mock_voxist_server):
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=invalid"
            async with session.ws_connect(url) as ws:
                msg = await ws.receive()
                assert msg.type == aiohttp.WSMsgType.CLOSE

    @pytest.mark.asyncio
    async def test_final_per_silence_delimited_segment(self, mock_voxist_server):
        """Speech followed by enough silence finalizes WITHOUT Done."""
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key&lang=fr"
            async with session.ws_connect(url) as ws:
                for _ in range(3):
                    await ws.send_bytes(speech_frame())
                for _ in range(5):  # 500ms of silence > silence_ms_to_finalize
                    await ws.send_bytes(silence_frame())

                messages = await collect_json(ws)

        types = [m["type"] for m in messages]
        assert "final" in types, f"no final without Done, got {types}"
        assert mock_voxist_server.done_received_count == 0

    @pytest.mark.asyncio
    async def test_two_segments_two_finals_one_socket(self, mock_voxist_server):
        """Multi-final sessions are the engine's normal operation."""
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key&lang=fr"
            async with session.ws_connect(url) as ws:
                for _segment in range(2):
                    for _ in range(3):
                        await ws.send_bytes(speech_frame())
                    for _ in range(5):
                        await ws.send_bytes(silence_frame())
                messages = await collect_json(ws)

        finals = [m for m in messages if m["type"] == "final"]
        assert len(finals) == 2
        assert len(mock_voxist_server.segments_finalized) == 2

    @pytest.mark.asyncio
    async def test_done_finalizes_and_closes_the_socket(self, mock_voxist_server):
        """
        Done is end-of-session: pending speech is finalized, then the server
        closes the client socket - mirroring gateway.ts:899.
        """
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key&lang=fr"
            async with session.ws_connect(url) as ws:
                for _ in range(3):
                    await ws.send_bytes(speech_frame())
                await ws.send_str("Done")

                got_final = False
                closed = False
                while True:
                    msg = await asyncio.wait_for(ws.receive(), timeout=3.0)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        if msg.json().get("type") == "final":
                            got_final = True
                    elif msg.type in (
                        aiohttp.WSMsgType.CLOSE,
                        aiohttp.WSMsgType.CLOSED,
                        aiohttp.WSMsgType.CLOSING,
                    ):
                        closed = True
                        break

        assert got_final, "pending speech must be finalized on Done"
        assert closed, "the server must close the socket after Done"
        assert mock_voxist_server.done_received_count == 1

    @pytest.mark.asyncio
    async def test_config_message_is_tolerated(self, mock_voxist_server):
        """A config frame is accepted without ending the session."""
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key&lang=fr"
            async with session.ws_connect(url) as ws:
                await ws.send_json({"config": {"lang": "fr"}})
                for _ in range(3):
                    await ws.send_bytes(speech_frame())
                for _ in range(5):
                    await ws.send_bytes(silence_frame())
                messages = await collect_json(ws)

        assert any(m["type"] == "final" for m in messages)

    @pytest.mark.asyncio
    async def test_server_tracks_statistics(self, mock_voxist_server):
        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key&lang=fr"
            async with session.ws_connect(url) as ws:
                payload = speech_frame()
                await ws.send_bytes(payload)
                await ws.send_str("Done")
                await collect_json(ws, timeout=1.0)

        assert mock_voxist_server.connections_count >= 1
        assert mock_voxist_server.audio_frames_received >= 1
        assert mock_voxist_server.total_audio_bytes >= len(payload)
        assert mock_voxist_server.connected_languages[-1] == "fr"

    @pytest.mark.asyncio
    async def test_concurrent_connections(self, mock_voxist_server):
        async def one_session():
            async with aiohttp.ClientSession() as session:
                url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key&lang=fr"
                async with session.ws_connect(url) as ws:
                    for _ in range(3):
                        await ws.send_bytes(speech_frame())
                    await ws.send_str("Done")
                    messages = await collect_json(ws, timeout=2.0)
                    return any(m.get("type") == "final" for m in messages)

        results = await asyncio.gather(one_session(), one_session())
        assert all(results)


class TestMockServerErrorSimulation:
    @pytest.mark.asyncio
    async def test_server_auth_failure_mode(self):
        server = MockVoxistServer(port=8767, error_mode="auth_failure")
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                url = "ws://localhost:8767/ws?api_key=anything"
                async with session.ws_connect(url) as ws:
                    msg = await ws.receive()
                    assert msg.type == aiohttp.WSMsgType.CLOSE
        finally:
            await server.stop()


class TestFinalsWithoutDoneToggle:
    """
    finals_without_done=False models an engine that only finalizes on Done.

    This is the fallback the live probe (scratchpad/probe_finals.py) would
    select; tests can flip the flag to develop against that behaviour.
    """

    @pytest.mark.asyncio
    async def test_silence_does_not_finalize_when_disabled(self):
        server = MockVoxistServer(port=8768, valid_api_key="test_key")
        server.finals_without_done = False
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                url = "ws://localhost:8768/ws?api_key=test_key&lang=fr"
                async with session.ws_connect(url) as ws:
                    for _ in range(3):
                        await ws.send_bytes(speech_frame())
                    for _ in range(8):
                        await ws.send_bytes(silence_frame())
                    interim_only = await collect_json(ws, timeout=0.5)
                    assert not any(
                        m["type"] == "final" for m in interim_only
                    ), "finalized on silence despite finals_without_done=False"

                    await ws.send_str("Done")
                    rest = await collect_json(ws, timeout=2.0)
                    assert any(m["type"] == "final" for m in rest)
        finally:
            await server.stop()
