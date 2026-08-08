"""Mock WebSocket server for integration testing."""

import asyncio
import json
from collections.abc import Callable

import aiohttp
from aiohttp import web


class MockVoxistServer:
    """
    Mock WebSocket server simulating Voxist API protocol.

    Simulates the complete Voxist protocol:
    1. HTTP token exchange: API key -> short-lived WebSocket token (SEC-001)
    2. WebSocket connection authenticated with that token
    3. Connection confirmation message
    4. Binary Int16 audio reception
    5. Partial and final transcription results
    6. Done signal handling

    Both the token endpoint and the WebSocket route are served on the same
    host/port, because ConnectionPool._get_http_base_url() derives the token
    URL from base_url by swapping the scheme and dropping the "/ws" suffix.

    By default the server binds port 0: the OS picks a free ephemeral port,
    so any number of servers (parallel pytest workers, concurrent suites,
    leaked processes from a previous run) coexist without EADDRINUSE. After
    start(), ``self.port`` holds the real bound port - always build URLs
    from ``server.port`` AFTER ``await server.start()``.

    Example:
        server = MockVoxistServer()
        await server.start()

        # Token exchange at http://{server.host}:{server.port}/websocket
        # WebSocket at ws://{server.host}:{server.port}/ws

        await server.stop()
    """

    def __init__(
        self,
        port: int = 0,
        host: str = "localhost",
        *,
        valid_api_key: str = "test_key",
        processing_delay_ms: int = 50,
        transcription_text: str = "bonjour monde",
        transcription_confidence: float = 0.95,
        send_interim: bool = True,
        interim_delay_ms: int = 25,
        error_mode: str | None = None,
        on_audio_received: Callable | None = None,
        api_key_header: str = "X-LVL-KEY",
        ws_token: str = "mock_jwt_token",
    ):
        """
        Initialize mock Voxist server.

        Args:
            port: Server port; 0 (the default) binds an OS-assigned ephemeral
                  port, published on self.port once start() returns
            host: Server host
            valid_api_key: Expected API key for authentication
            processing_delay_ms: Delay before sending final result (simulates processing)
            transcription_text: Text to return in transcription
            transcription_confidence: Confidence score (0.0-1.0)
            send_interim: Whether to send interim results
            interim_delay_ms: Delay before sending interim result
            error_mode: Error simulation mode (None, "auth_failure",
                        "disconnect", "wedge"). "wedge" models a wedged
                        ENGINE behind a healthy gateway: the WebSocket layer
                        stays connected (aiohttp answers pings at protocol
                        level) and keeps accepting audio and Done, but never
                        sends a single message and never closes - the exact
                        mute-but-connected failure a transport heartbeat
                        cannot see.
            on_audio_received: Callback when audio is received (for testing)
            api_key_header: Header carrying the API key on the token exchange
                            (must match ConnectionPool.api_key_header)
            ws_token: Token handed out by the token endpoint and accepted by
                      the WebSocket route
        """
        self.port = port
        self.host = host
        self.valid_api_key = valid_api_key
        self.api_key_header = api_key_header
        self.ws_token = ws_token
        self.processing_delay_ms = processing_delay_ms
        self.transcription_text = transcription_text
        self.transcription_confidence = transcription_confidence
        self.send_interim = send_interim
        # Gateway realism: the engine emits a final per silence-delimited
        # segment (banafo does its own endpointing; the gateway threads
        # precedingContext across finals), and the gateway closes the client
        # socket when the engine closes after "Done"
        # (simple-websocket-proxy.gateway.ts:899). Tests may flip
        # finals_without_done to model an engine that only finalizes on Done.
        self.finals_without_done = True
        self.silence_ms_to_finalize = 300
        # Observability for tests: what the server actually saw
        self.connected_languages: list[str | None] = []
        self.done_received_count = 0
        self.finals_sent = 0
        self.segments_finalized: list[int] = []  # speech bytes per segment
        self.interim_delay_ms = interim_delay_ms
        self.error_mode = error_mode
        self.on_audio_received = on_audio_received

        self.app = web.Application()
        self.app.router.add_get("/websocket", self.token_handler)
        self.app.router.add_get("/ws", self.websocket_handler)
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        self.connections_count = 0
        self.token_requests_count = 0
        self.audio_frames_received = 0
        self.total_audio_bytes = 0

    async def token_handler(self, request: web.Request) -> web.Response:
        """
        Handle the SEC-001 token exchange: API key -> WebSocket URL with token.

        Contract, from ConnectionPool._get_ws_token():
        - GET {http_base}/websocket with the API key in the api_key_header
          header and engine=voxist-rt as a query param
        - 401/403 means the key is invalid and must not be retried
        - any other non-200 is a transport failure
        - a 200 body must carry a "url" field; the pool appends lang and
          sample_rate to it and connects there

        The pool reads nothing else from the body: it caches the URL for a
        hard-coded hour rather than honouring a server-supplied expiry, so
        expires_in below is returned for fidelity but has no effect.
        """
        self.token_requests_count += 1

        api_key = request.headers.get(self.api_key_header)

        if self.error_mode == "auth_failure" or api_key != self.valid_api_key:
            return web.json_response({"error": "Invalid API key"}, status=401)

        return web.json_response({
            "url": f"ws://{self.host}:{self.port}/ws?token={self.ws_token}",
            "expires_in": 3600,
        })

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """
        Handle WebSocket connection.

        Implements Voxist protocol:
        1. Authenticate via query parameter
        2. Send connection confirmation
        3. Process audio frames
        4. Send transcription results
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.connections_count += 1

        try:
            # Authenticate via query parameter. The plugin arrives with the
            # token issued by /websocket; api_key is still accepted so tests
            # can drive the WebSocket route directly.
            credential = request.query.get("api_key") or request.query.get("token")

            if self.error_mode == "auth_failure":
                await ws.close(code=1008, message=b"Invalid API key")
                return ws

            is_valid = credential in (self.valid_api_key, self.ws_token)
            if not is_valid:
                await ws.close(code=1008, message=b"Invalid API key")
                return ws

            # The real gateway sends NO confirmation frame on connect: the only
            # things a client ever receives are transcription results (and a
            # pub/sub redirect). The {"status": "connected"} this mock used to
            # send was a fiction that made the mock more talkative than
            # production and kept an unreachable plugin branch looking alive.

            self.connected_languages.append(request.query.get("lang"))

            # Engine-side segmentation state, mirroring banafo endpointing:
            # a run of silence after speech finalizes the segment.
            speech_bytes = 0
            silence_run_ms = 0
            sample_rate = int(request.query.get("sample_rate", "16000"))
            bytes_per_ms = sample_rate * 2 // 1000
            interim_sent_for_segment = False

            async def finalize_segment() -> None:
                nonlocal speech_bytes, silence_run_ms, interim_sent_for_segment
                if speech_bytes == 0:
                    return
                if self.processing_delay_ms:
                    await asyncio.sleep(self.processing_delay_ms / 1000.0)
                self.segments_finalized.append(speech_bytes)
                self.finals_sent += 1
                await ws.send_json({
                    "type": "final",
                    "text": self.transcription_text,
                    "confidence": self.transcription_confidence,
                })
                speech_bytes = 0
                silence_run_ms = 0
                interim_sent_for_segment = False

            async for msg in ws:
                if self.error_mode == "wedge":
                    # Wedged engine, healthy gateway: bookkeeping only.
                    # Nothing is ever sent back, Done neither finalizes nor
                    # closes, and the loop only ends when the CLIENT closes.
                    if msg.type == aiohttp.WSMsgType.BINARY:
                        self.audio_frames_received += 1
                        self.total_audio_bytes += len(msg.data)
                        if self.on_audio_received:
                            self.on_audio_received(msg.data, len(msg.data) // 2)
                    elif (
                        msg.type == aiohttp.WSMsgType.TEXT
                        and "Done" in msg.data
                    ):
                        self.done_received_count += 1
                    continue

                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        json.loads(msg.data)
                        # Config messages are accepted (the gateway supports
                        # them) but nothing here depends on one: lang and
                        # sample_rate arrive on the URL.
                        continue
                    except json.JSONDecodeError:
                        pass

                    # Done = end of SESSION, not end of utterance. The gateway
                    # forwards it to the engine, the engine flushes and closes,
                    # and the gateway then closes the client socket.
                    if "Done" in msg.data:
                        self.done_received_count += 1
                        await finalize_segment()
                        break

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self.audio_frames_received += 1
                    self.total_audio_bytes += len(msg.data)

                    if self.on_audio_received:
                        self.on_audio_received(msg.data, len(msg.data) // 2)

                    is_silence = not any(msg.data)
                    if is_silence:
                        if speech_bytes and self.finals_without_done:
                            silence_run_ms += len(msg.data) // bytes_per_ms
                            if silence_run_ms >= self.silence_ms_to_finalize:
                                await finalize_segment()
                    else:
                        silence_run_ms = 0
                        speech_bytes += len(msg.data)
                        if self.send_interim and not interim_sent_for_segment:
                            interim_sent_for_segment = True
                            if self.interim_delay_ms:
                                await asyncio.sleep(self.interim_delay_ms / 1000.0)
                            await ws.send_json({
                                "type": "partial",
                                "text": self.transcription_text.split()[0],
                                "confidence": self.transcription_confidence - 0.1,
                            })

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

            # Close like the gateway: client.close() once the engine is done.
            await ws.close()

        except Exception as e:
            # Log error but don't crash server
            print(f"MockVoxistServer error: {e}")

        finally:
            if not ws.closed:
                await ws.close()

        return ws

    async def start(self):
        """Start the mock WebSocket server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        # With port=0 the OS assigned an ephemeral port; publish the real one
        # so URL construction (tests AND token_handler) uses it. The listening
        # socket lives on the underlying asyncio Server held by the TCPSite.
        assert self.site._server is not None
        self.port = self.site._server.sockets[0].getsockname()[1]

        print(
            f"Mock Voxist server started at ws://{self.host}:{self.port}/ws "
            f"(token exchange at http://{self.host}:{self.port}/websocket)"
        )

    async def stop(self):
        """Stop the mock WebSocket server."""
        if self.site:
            await self.site.stop()

        if self.runner:
            await self.runner.cleanup()

        print("Mock Voxist server stopped")

    def get_stats(self) -> dict:
        """
        Get server statistics.

        Returns:
            Dictionary with connection and audio stats
        """
        return {
            "connections_count": self.connections_count,
            "token_requests_count": self.token_requests_count,
            "audio_frames_received": self.audio_frames_received,
            "total_audio_bytes": self.total_audio_bytes,
        }

    def reset_stats(self):
        """Reset server statistics."""
        self.connections_count = 0
        self.token_requests_count = 0
        self.audio_frames_received = 0
        self.total_audio_bytes = 0


class ConfigurableMockServer(MockVoxistServer):
    """
    Extended mock server with configurable behaviors for advanced testing.

    Supports:
    - Multi-utterance handling
    - Custom response sequences
    - Connection drops
    - Latency variations
    """

    def __init__(
        self,
        port: int = 0,
        *,
        responses: list[dict] | None = None,
        disconnect_after: int | None = None,
        variable_latency: bool = False,
        **kwargs
    ):
        """
        Initialize configurable mock server.

        Args:
            port: Server port; 0 (the default) binds an OS-assigned ephemeral
                  port, published on self.port once start() returns
            responses: List of response dicts to send in sequence
            disconnect_after: Disconnect after N audio frames (for reconnection testing)
            variable_latency: Vary processing delay randomly (20-100ms)
            **kwargs: Additional arguments for MockVoxistServer
        """
        super().__init__(port=port, **kwargs)

        self.responses = responses or []
        self.disconnect_after = disconnect_after
        self.variable_latency = variable_latency
        self._response_index = 0

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Extended handler with configurable behaviors."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.connections_count += 1

        try:
            # Authenticate (token issued by /websocket, or a raw key)
            credential = request.query.get("api_key") or request.query.get("token")
            if credential not in (self.valid_api_key, self.ws_token):
                await ws.close(code=1008, message=b"Invalid API key")
                return ws

            # Send connection confirmation
            await ws.send_json({"status": "connected"})

            frame_count = 0

            # Process messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle JSON or "Done"
                    try:
                        json.loads(msg.data)
                        # Config received
                    except json.JSONDecodeError:
                        if "Done" in msg.data:
                            break

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self.audio_frames_received += 1
                    self.total_audio_bytes += len(msg.data)
                    frame_count += 1

                    # Check disconnect condition
                    if self.disconnect_after and frame_count >= self.disconnect_after:
                        await ws.close(code=1001, message=b"Test disconnect")
                        return ws

                    # Send configured responses
                    if self.responses and self._response_index < len(self.responses):
                        response = self.responses[self._response_index]

                        # Apply variable latency if enabled
                        if self.variable_latency:
                            import random
                            delay = random.uniform(0.02, 0.1)  # 20-100ms
                        else:
                            delay = response.get("delay", 0.05)

                        await asyncio.sleep(delay)
                        await ws.send_json(response["message"])
                        self._response_index += 1

        except Exception as e:
            print(f"ConfigurableMockServer error: {e}")

        finally:
            if not ws.closed:
                await ws.close()

        return ws
