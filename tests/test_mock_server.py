"""Tests for MockVoxistServer against the gateway-faithful contract.

The mock mirrors verified gateway behaviour (simple-websocket-proxy.gateway.ts):
no greeting frame on connect, a final per silence-delimited segment without
"Done", and "Done" as the end-of-SESSION signal after which the socket closes.
"""

import asyncio

import aiohttp
import numpy as np
import pytest

from .fixtures.mock_server import ConfigurableMockServer, MockVoxistServer


def speech_frame(ms=100, rate=16000):
    """Non-silent Int16 audio (sine), as the plugin would send."""
    n = rate * ms // 1000
    t = np.linspace(0, ms / 1000.0, n, endpoint=False)
    return (np.sin(2 * np.pi * 440 * t) * 20000).astype(np.int16).tobytes()


def silence_frame(ms=100, rate=16000):
    return b"\x00" * (rate * ms // 1000 * 2)


async def collect_json(ws, timeout=2.0):
    """
    Drain currently-available JSON messages.

    Skips the engine's bare "Done!" acknowledgement, which is a real protocol
    frame and not JSON - the API's own reference client skips it the same way
    (kroko/bench/asr_bench.py:103). Parsing it raised JSONDecodeError here the
    moment the mock became faithful to it.
    """
    out = []
    try:
        while True:
            msg = await asyncio.wait_for(ws.receive(), timeout=timeout)
            if msg.type != aiohttp.WSMsgType.TEXT:
                break
            if msg.data.startswith("Done"):
                continue
            out.append(msg.json())
    except asyncio.TimeoutError:
        pass
    return out


class TestMockServerBasics:
    @pytest.mark.asyncio
    async def test_server_starts_and_stops(self):
        server = MockVoxistServer()
        await server.start()
        assert server.runner is not None
        assert server.site is not None
        assert server.port != 0, "start() must publish the real bound port"
        await server.stop()

    @pytest.mark.asyncio
    async def test_start_binds_exactly_one_socket(self):
        """
        [A] self.port must honestly describe the server.

        The default host is the 127.0.0.1 literal, not "localhost", precisely
        so ONE socket is bound. "localhost" resolves to both 127.0.0.1 and
        ::1; TCPSite then opens a listener per family, and with port=0 the OS
        assigns each a DIFFERENT ephemeral port. Publishing sockets[0]'s port
        would leave a client that resolved to the other family dialing a port
        nothing advertised - an intermittent failure that reads as a flake.
        """
        server = MockVoxistServer()
        await server.start()
        try:
            assert server.site is not None
            assert server.site._server is not None
            sockets = server.site._server.sockets
            assert len(sockets) == 1, (
                f"expected a single listening socket, got "
                f"{[s.getsockname() for s in sockets]}"
            )
            assert sockets[0].getsockname()[1] == server.port
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_multi_family_host_with_ephemeral_port_is_refused(self):
        """
        A host that fans out across families cannot publish one ephemeral
        port, so start() must refuse rather than advertise a half-working URL.

        Where "localhost" happens to resolve to a single family the hazard
        does not exist, and start() must succeed with one honest socket - so
        both outcomes are asserted rather than one being skipped.
        """
        server = MockVoxistServer(host="localhost")
        try:
            await server.start()
        except RuntimeError as e:
            assert "listening sockets" in str(e)
            assert server.runner is None, "a refused start must not leak a runner"
            assert server.site is None
            return

        try:
            assert server.site is not None and server.site._server is not None
            assert len(server.site._server.sockets) == 1, (
                "start() accepted a multi-socket bind"
            )
            assert server.site._server.sockets[0].getsockname()[1] == server.port
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_two_default_servers_run_concurrently(self):
        """
        Two servers with default args coexist - impossible with a fixed
        default port. Proves the ephemeral-port default (port=0) end to end:
        distinct real ports after start(), and both actually serving (the
        token endpoint answers on each, advertising its own port).
        """
        server_a = MockVoxistServer()
        server_b = MockVoxistServer()
        await server_a.start()
        try:
            await server_b.start()
            try:
                assert server_a.port != 0 and server_b.port != 0
                assert server_a.port != server_b.port

                async with aiohttp.ClientSession() as session:
                    for srv in (server_a, server_b):
                        async with session.get(
                            f"http://{srv.host}:{srv.port}/websocket",
                            headers={srv.api_key_header: srv.valid_api_key},
                        ) as resp:
                            assert resp.status == 200
                            body = await resp.json()
                            assert f":{srv.port}/ws" in body["url"], (
                                "token endpoint must advertise the port the "
                                "server actually bound"
                            )
            finally:
                await server_b.stop()
        finally:
            await server_a.stop()

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
                acked = False
                closed = False
                while True:
                    msg = await asyncio.wait_for(ws.receive(), timeout=3.0)
                    if msg.type == aiohttp.WSMsgType.TEXT:
                        # The engine's bare "Done!" ack precedes the close and
                        # is not JSON; parsing it here raised JSONDecodeError
                        # once the mock became faithful to it.
                        if msg.data.startswith("Done"):
                            acked = True
                            continue
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
        assert acked, (
            "the engine acks Done with a bare 'Done!' frame before the close; "
            "verified live on api-asr.voxist.com"
        )
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


    @pytest.mark.asyncio
    async def test_stats_cover_every_counter(self, mock_voxist_server):
        """
        [13] get_stats()/reset_stats() must describe the WHOLE observation
        surface, not the four counters that existed first. A counter missing
        from reset_stats() leaks state across a test that reuses the server.
        """
        tracked = {
            "connections_count",
            "token_requests_count",
            "audio_frames_received",
            "total_audio_bytes",
            "ws_upgrade_refusals",
            "connected_languages",
            "done_received_count",
            "finals_sent",
            "segments_finalized",
        }
        assert set(mock_voxist_server.get_stats()) == tracked

        async with aiohttp.ClientSession() as session:
            url = f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws?api_key=test_key&lang=fr"
            async with session.ws_connect(url) as ws:
                for _ in range(3):
                    await ws.send_bytes(speech_frame())
                for _ in range(5):
                    await ws.send_bytes(silence_frame())
                await collect_json(ws, timeout=1.0)
                await ws.send_str("Done")
                await collect_json(ws, timeout=1.0)

        dirty = mock_voxist_server.get_stats()
        assert dirty["finals_sent"] >= 1
        assert dirty["segments_finalized"]
        assert dirty["connected_languages"] == ["fr"]
        assert dirty["done_received_count"] == 1

        # A snapshot must not mutate under the caller.
        snapshot = mock_voxist_server.get_stats()
        mock_voxist_server.reset_stats()
        assert snapshot["connected_languages"] == ["fr"]
        assert snapshot["segments_finalized"]

        for name, value in mock_voxist_server.get_stats().items():
            assert not value, f"reset_stats() left {name}={value!r}"


class TestMockServerErrorSimulation:
    @pytest.mark.asyncio
    async def test_server_auth_failure_mode(self):
        server = MockVoxistServer(error_mode="auth_failure")
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"ws://{server.host}:{server.port}/ws?api_key=anything"
                async with session.ws_connect(url) as ws:
                    msg = await ws.receive()
                    assert msg.type == aiohttp.WSMsgType.CLOSE
        finally:
            await server.stop()


class TestWsBlockedMode:
    """
    [J] error_mode="ws_blocked" models a deployment whose HTTPS token endpoint
    is healthy but whose WebSocket path is broken (a proxy stripping the
    Upgrade header) - the case the token-only warm-up cannot see.
    """

    @pytest.mark.asyncio
    async def test_token_endpoint_healthy_but_ws_never_upgrades(self):
        server = MockVoxistServer(valid_api_key="test_key", error_mode="ws_blocked")
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(
                    f"http://{server.host}:{server.port}/websocket",
                    headers={server.api_key_header: "test_key"},
                ) as resp:
                    assert resp.status == 200, "the token endpoint stays healthy"
                    assert f":{server.port}/ws" in (await resp.json())["url"]

                with pytest.raises(aiohttp.WSServerHandshakeError):
                    await session.ws_connect(
                        f"ws://{server.host}:{server.port}/ws?api_key=test_key"
                    )

                # The blocked route is a plain HTTP endpoint, not an upgrade.
                async with session.get(
                    f"http://{server.host}:{server.port}/ws?api_key=test_key"
                ) as resp:
                    assert resp.status == 200
                    assert "Upgrade header" in await resp.text()

            assert server.ws_upgrade_refusals == 2
            assert server.connections_count == 0, (
                "a refused upgrade is not a WebSocket session"
            )
        finally:
            await server.stop()


class TestConfigurableMockServerParity:
    """
    [13] ConfigurableMockServer used to carry a SECOND websocket_handler that
    drifted from the contract the base class enforces: it sent a
    {"status": "connected"} greeting the gateway never sends, had no
    silence-delimited segmentation, and ignored error_mode. It now inherits the
    base handler, so it must exhibit the base contract exactly.
    """

    def test_it_adds_no_handler_of_its_own(self):
        assert (
            ConfigurableMockServer.websocket_handler
            is MockVoxistServer.websocket_handler
        ), "the duplicated handler is back - it will drift again"

    @pytest.mark.asyncio
    async def test_no_greeting_and_finals_without_done(self):
        server = ConfigurableMockServer(valid_api_key="test_key")
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"ws://{server.host}:{server.port}/ws?api_key=test_key&lang=fr"
                async with session.ws_connect(url) as ws:
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(ws.receive(), timeout=0.3)

                    for _ in range(3):
                        await ws.send_bytes(speech_frame())
                    for _ in range(5):
                        await ws.send_bytes(silence_frame())
                    messages = await collect_json(ws)

            assert any(m["type"] == "final" for m in messages), (
                f"no final without Done, got {[m['type'] for m in messages]}"
            )
            assert server.done_received_count == 0
            assert server.finals_sent == 1
            assert server.connected_languages == ["fr"]
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_wedge_mode_reaches_the_subclass(self):
        """error_mode was silently ignored by the old override."""
        server = ConfigurableMockServer(valid_api_key="test_key", error_mode="wedge")
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"ws://{server.host}:{server.port}/ws?api_key=test_key&lang=fr"
                async with session.ws_connect(url) as ws:
                    for _ in range(3):
                        await ws.send_bytes(speech_frame())
                    for _ in range(5):
                        await ws.send_bytes(silence_frame())
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(ws.receive(), timeout=0.5)
            assert server.finals_sent == 0
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_scripted_responses_replace_engine_output(self):
        """`responses` gives the test full control of the message sequence."""
        server = ConfigurableMockServer(
            valid_api_key="test_key",
            responses=[
                {"message": {"type": "partial", "text": "un"}, "delay": 0},
                {"message": {"type": "final", "text": "un deux"}, "delay": 0},
            ],
        )
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"ws://{server.host}:{server.port}/ws?api_key=test_key&lang=fr"
                async with session.ws_connect(url) as ws:
                    for _ in range(4):  # one more frame than scripted replies
                        await ws.send_bytes(speech_frame())
                    messages = await collect_json(ws, timeout=0.5)

            assert [m["text"] for m in messages] == ["un", "un deux"], (
                "the script must be the only thing sent"
            )
            assert server.finals_sent == 0, (
                "engine-side finalization must stay suspended in scripted mode"
            )
            assert server.audio_frames_received == 4
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_disconnect_after_drops_the_socket_midsession(self):
        server = ConfigurableMockServer(valid_api_key="test_key", disconnect_after=2)
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"ws://{server.host}:{server.port}/ws?api_key=test_key&lang=fr"
                async with session.ws_connect(url) as ws:
                    for _ in range(2):
                        await ws.send_bytes(speech_frame())
                    while True:
                        msg = await asyncio.wait_for(ws.receive(), timeout=2.0)
                        if msg.type in (
                            aiohttp.WSMsgType.CLOSE,
                            aiohttp.WSMsgType.CLOSED,
                            aiohttp.WSMsgType.CLOSING,
                        ):
                            break
            assert server.audio_frames_received == 2
        finally:
            await server.stop()


class TestWedgeMode:
    """
    error_mode="wedge" models the mute-but-connected failure: the engine
    behind the gateway is dead, but the gateway's WS layer stays up and
    answers protocol pings. The socket accepts everything - audio, silence,
    even Done - and never sends a byte back, never closes.
    """

    @pytest.mark.asyncio
    async def test_wedge_accepts_audio_but_never_responds(self):
        server = MockVoxistServer(valid_api_key="test_key", error_mode="wedge")
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                url = (
                    f"ws://{server.host}:{server.port}/ws"
                    "?api_key=test_key&lang=fr"
                )
                async with session.ws_connect(url) as ws:
                    for _ in range(3):
                        await ws.send_bytes(speech_frame())
                    for _ in range(5):  # enough silence to finalize normally
                        await ws.send_bytes(silence_frame())
                    await ws.send_str("Done")

                    # No final on silence, no finalize-and-close on Done:
                    # the socket just sits there, connected and mute.
                    with pytest.raises(asyncio.TimeoutError):
                        await asyncio.wait_for(ws.receive(), timeout=0.5)

            assert server.audio_frames_received == 8
            assert server.done_received_count == 1, (
                "wedge must still observe Done for test bookkeeping"
            )
            assert server.finals_sent == 0
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
        server = MockVoxistServer(valid_api_key="test_key")
        server.finals_without_done = False
        await server.start()
        try:
            async with aiohttp.ClientSession() as session:
                url = f"ws://{server.host}:{server.port}/ws?api_key=test_key&lang=fr"
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
