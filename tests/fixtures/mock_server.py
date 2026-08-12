"""Mock WebSocket server for integration testing."""

import asyncio
import json
import random
from collections.abc import Callable

import aiohttp
from aiohttp import web


class MockVoxistServer:
    """
    Mock Voxist gateway: the SEC-001 token endpoint plus the WebSocket route.

    What the plugin actually does against it, verified in connection.py:

    1. ``VoxistDialer._http_base_url()`` turns ``base_url`` into an HTTP base
       by swapping the scheme (``wss://`` -> ``https://``, ``ws://`` ->
       ``http://``) and dropping a trailing ``/ws``.
    2. ``VoxistDialer._get_token_url()`` GETs ``{http_base}/websocket`` with
       the API key in the ``api_key_header`` header and ``engine=voxist-rt``
       as a query param, then reads the ``url`` field out of the JSON body.
    3. ``VoxistDialer.dial()`` appends ``lang`` and ``sample_rate`` to that
       URL and opens the WebSocket.
    4. The socket carries binary Int16 audio up and transcription JSON down.
       There is NO greeting frame (see websocket_handler).
    5. ``"Done"`` is the end-of-SESSION signal: the engine flushes a last
       final and the gateway then closes the client socket.

    Both routes are therefore served on the SAME host/port, so a test can
    point ``VoxistSTT(base_url=f"ws://{server.host}:{server.port}/ws")`` at
    this server and exercise the real token exchange.

    By default the server binds port 0: the OS picks a free ephemeral port,
    so any number of servers (parallel pytest workers, concurrent suites,
    leaked processes from a previous run) coexist without EADDRINUSE. After
    start(), ``self.port`` holds the real bound port - always build URLs
    from ``server.port`` AFTER ``await server.start()``. The default host is
    the ``127.0.0.1`` LITERAL, not ``"localhost"``, so exactly one socket is
    bound and ``self.port`` can honestly describe it (see start()).

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
        host: str = "127.0.0.1",
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
        responses: list[dict] | None = None,
        disconnect_after: int | None = None,
        variable_latency: bool = False,
    ):
        """
        Initialize mock Voxist server.

        Args:
            port: Server port; 0 (the default) binds an OS-assigned ephemeral
                  port, published on self.port once start() returns
            host: Server host; must resolve to a SINGLE address family (the
                  default is the 127.0.0.1 literal - see start())
            valid_api_key: Expected API key for authentication
            processing_delay_ms: Delay before sending final result (simulates processing)
            transcription_text: Text to return in transcription
            transcription_confidence: Confidence score (0.0-1.0)
            send_interim: Whether to send interim results
            interim_delay_ms: Delay before sending interim result
            error_mode: Error simulation mode (None, "auth_failure", "wedge",
                        "ws_blocked", "ws_upgraded_then_rejected").
                        "wedge" models a wedged ENGINE behind a healthy
                        gateway: the WebSocket layer stays connected (aiohttp
                        answers pings at protocol level) and keeps accepting
                        audio and Done, but never sends a single message and
                        never closes - the exact mute-but-connected failure a
                        transport heartbeat cannot see.
                        "ws_blocked" models a deployment whose HTTPS token
                        endpoint is healthy but whose WebSocket path is broken
                        (a proxy stripping the Upgrade header): /websocket
                        hands out a valid-looking token URL while /ws answers
                        a plain HTTP 200 instead of upgrading. Refusals are
                        counted in self.ws_upgrade_refusals.
                        "ws_upgraded_then_rejected" models a gateway that
                        refuses at the APPLICATION layer: the token endpoint is
                        healthy, the WebSocket upgrade SUCCEEDS (a real 101,
                        counted in self.connections_count), and only then is
                        the socket closed with 1008 - the gateway's documented
                        answer for an invalid or expired app-level credential
                        and for an exhausted wallet balance. Distinct from
                        "auth_failure", which fails the token exchange with a
                        401 and so never upgrades at all, and from
                        "ws_blocked", which never upgrades either. This is the
                        only mode where a client that observes nothing but the
                        101 concludes the deployment is healthy.
            on_audio_received: Callback when audio is received (for testing)
            api_key_header: Header carrying the API key on the token exchange
                            (must match VoxistDialer's api_key_header, whose
                            default is also X-LVL-KEY)
            ws_token: Token handed out by the token endpoint and accepted by
                      the WebSocket route
            responses: Scripted reply sequence - one dict per received audio
                       frame, {"message": {...}, "delay": seconds}. While set,
                       the script is the ONLY thing sent: the engine's own
                       interim/segment-final generation is suspended so a test
                       controls the exact message sequence.
            disconnect_after: Close the socket with code 1001 once this many
                              audio frames have arrived (reconnection testing)
            variable_latency: Randomise the scripted delay to 20-100ms instead
                              of honouring each response's "delay"
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
        # Faithful by default: the real engine always acks. Settable to False
        # to model an engine variant that does not, since the API's own
        # reference client treats the ack as skippable rather than required.
        self.send_done_ack = True
        self.finals_sent = 0
        self.segments_finalized: list[int] = []  # speech bytes per segment
        self.interim_delay_ms = interim_delay_ms
        self.error_mode = error_mode
        self.on_audio_received = on_audio_received

        # Configurable behaviours (formerly a second, divergent handler in
        # ConfigurableMockServer - see that class).
        self.responses = responses or []
        self.disconnect_after = disconnect_after
        self.variable_latency = variable_latency
        self._response_index = 0

        self.app = web.Application()
        self.app.router.add_get("/websocket", self.token_handler)
        self.app.router.add_get("/ws", self.websocket_handler)
        self.runner: web.AppRunner | None = None
        self.site: web.TCPSite | None = None

        self.connections_count = 0
        self.token_requests_count = 0
        self.audio_frames_received = 0
        self.total_audio_bytes = 0
        self.ws_upgrade_refusals = 0

    async def token_handler(self, request: web.Request) -> web.Response:
        """
        Handle the SEC-001 token exchange: API key -> WebSocket URL with token.

        Contract, as implemented by VoxistDialer._get_token_url():
        - GET {http_base}/websocket with the API key in the api_key_header
          header and engine=voxist-rt as a query param, where {http_base} is
          _http_base_url()'s scheme-swapped, "/ws"-stripped base_url
        - 401/403 raises AuthenticationError: the key is invalid, and no retry
          can fix it
        - any other non-200 raises ConnectionError (a transport failure, which
          livekit's conn_options.max_retry may retry)
        - a 200 body must be a JSON OBJECT carrying a "url" string that starts
          with ws:// or wss://; anything else raises ConnectionError. dial()
          then appends lang and sample_rate to that url and connects there

        The dialer reads nothing else from the body. In particular it does NOT
        read expires_in: the refresh deadline comes from the token's own JWT
        exp claim (_token_expiry_from_url), sanity-clamped, falling back to the
        gateway's known 1h lifetime when the claim is unreadable. self.ws_token
        defaults to the non-JWT literal "mock_jwt_token", so that fallback is
        the path tests normally take; expires_in below is returned only for
        shape fidelity and has no effect.
        """
        self.token_requests_count += 1

        api_key = request.headers.get(self.api_key_header)

        if self.error_mode == "auth_failure" or api_key != self.valid_api_key:
            return web.json_response({"error": "Invalid API key"}, status=401)

        return web.json_response({
            "url": f"ws://{self.host}:{self.port}/ws?token={self.ws_token}",
            "expires_in": 3600,
        })

    async def websocket_handler(self, request: web.Request) -> web.StreamResponse:
        """
        Handle the WebSocket route, as the gateway does.

        1. Authenticate the token (or a raw api_key) from the query string
        2. Send NOTHING on connect - the gateway has no greeting frame
        3. Receive binary Int16 audio; emit one partial per segment and one
           final per silence-delimited segment
        4. Treat "Done" as end-of-SESSION: flush a final, then close

        error_mode shortcuts this: "ws_blocked" never upgrades at all,
        "auth_failure" closes with 1008, "ws_upgraded_then_rejected" upgrades
        for a VALID credential and then closes with 1008, "wedge" accepts
        everything and answers nothing.
        """
        if self.error_mode == "ws_blocked":
            # Broken WebSocket path behind a healthy token endpoint: answer
            # the plain HTTP request instead of upgrading, exactly as a proxy
            # that stripped the Upgrade header would.
            self.ws_upgrade_refusals += 1
            return web.Response(text="the proxy ate your Upgrade header")

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

            if self.error_mode == "ws_upgraded_then_rejected":
                # The credential is fine at the token endpoint and the upgrade
                # already succeeded; the refusal happens at the application
                # layer, after the 101. A client that treats the handshake as
                # proof of a working deployment cannot tell this apart from a
                # healthy connect.
                await ws.close(
                    code=1008, message=b"Application layer refused the session"
                )
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
            frame_count = 0

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
                        # The engine acks with a bare non-JSON "Done!" text
                        # frame, which the gateway forwards verbatim. Verified
                        # live against api-asr.voxist.com (lang=fr): it lands
                        # ~0.12s after Done, right behind the last final.
                        # Modelled here so the plugin's handling of it is
                        # exercised by the integration suite and not only by
                        # unit tests.
                        if self.send_done_ack:
                            await ws.send_str("Done!")
                        break

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self.audio_frames_received += 1
                    self.total_audio_bytes += len(msg.data)
                    frame_count += 1

                    if self.on_audio_received:
                        self.on_audio_received(msg.data, len(msg.data) // 2)

                    # Optional: model a gateway that drops the socket
                    # mid-session (reconnection testing).
                    if (
                        self.disconnect_after is not None
                        and frame_count >= self.disconnect_after
                    ):
                        await ws.close(code=1001, message=b"Test disconnect")
                        return ws

                    if self.responses:
                        # Scripted mode: the test owns the message sequence, so
                        # the engine's own interim/final generation is
                        # suspended (speech_bytes stays 0, so a later "Done"
                        # finalizes nothing and just closes).
                        if self._response_index < len(self.responses):
                            response = self.responses[self._response_index]
                            self._response_index += 1
                            if self.variable_latency:
                                delay = random.uniform(0.02, 0.1)  # 20-100ms
                            else:
                                delay = response.get("delay", 0.05)
                            if delay:
                                await asyncio.sleep(delay)
                            await ws.send_json(response["message"])
                        continue

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
        """
        Start the mock server on exactly ONE listening socket.

        The single-socket requirement is not pedantry. A hostname like
        "localhost" resolves to both 127.0.0.1 and ::1, so aiohttp's TCPSite
        opens one listener per family - and with port=0 the OS assigns each a
        DIFFERENT ephemeral port. self.port can only publish one of them, so a
        client that resolved the host to the other family would dial a port
        nothing ever advertised: an intermittent connection failure that reads
        as a random flake. Binding a single-family literal (the 127.0.0.1
        default) makes self.port an honest description of the server.
        """
        requested_port = self.port
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        # With port=0 the OS assigned an ephemeral port; publish the real one
        # so URL construction (tests AND token_handler) uses it. The listening
        # socket lives on the underlying asyncio Server held by the TCPSite.
        assert self.site._server is not None
        sockets = self.site._server.sockets
        if requested_port == 0 and len(sockets) != 1:
            # Multi-family bind with an OS-assigned port: the ports differ per
            # family and only one can be published. Fail loudly at start()
            # rather than hand out a URL that works for some resolutions only.
            await self.site.stop()
            await self.runner.cleanup()
            self.site = None
            self.runner = None
            raise RuntimeError(
                f"host={self.host!r} bound {len(sockets)} listening sockets; "
                "with port=0 each address family gets a DIFFERENT ephemeral "
                "port and only one can be advertised. Use a single-family "
                "literal host such as '127.0.0.1' (the default)."
            )
        self.port = sockets[0].getsockname()[1]

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
        Snapshot of everything the server observed.

        Lists are copied so a snapshot never mutates underneath a test that
        holds it while the server keeps running.
        """
        return {
            "connections_count": self.connections_count,
            "token_requests_count": self.token_requests_count,
            "audio_frames_received": self.audio_frames_received,
            "total_audio_bytes": self.total_audio_bytes,
            "ws_upgrade_refusals": self.ws_upgrade_refusals,
            "connected_languages": list(self.connected_languages),
            "done_received_count": self.done_received_count,
            "finals_sent": self.finals_sent,
            "segments_finalized": list(self.segments_finalized),
        }

    def reset_stats(self):
        """Reset every counter get_stats() reports."""
        self.connections_count = 0
        self.token_requests_count = 0
        self.audio_frames_received = 0
        self.total_audio_bytes = 0
        self.ws_upgrade_refusals = 0
        self.connected_languages = []
        self.done_received_count = 0
        # Faithful by default: the real engine always acks. Settable to False
        # to model an engine variant that does not, since the API's own
        # reference client treats the ack as skippable rather than required.
        self.send_done_ack = True
        self.finals_sent = 0
        self.segments_finalized = []


class ConfigurableMockServer(MockVoxistServer):
    """
    Backwards-compatible name for behaviours that now live on the base class.

    This used to override websocket_handler with a SECOND, independent
    implementation, and it drifted from the contract the base class enforces:
    it still sent a {"status": "connected"} greeting the real gateway never
    sends, had no silence-delimited segmentation, and ignored error_mode
    entirely. Its extra behaviours are plain MockVoxistServer options now
    (``responses``, ``disconnect_after``, ``variable_latency``), so it adds
    nothing and inherits the gateway-faithful handler, the ws_blocked/wedge
    error modes, and the full stats surface.

    Prefer MockVoxistServer(...) directly in new tests.
    """
