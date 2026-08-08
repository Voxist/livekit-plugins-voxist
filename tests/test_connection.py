"""Unit tests for VoxistDialer: token exchange, expiry, and dialing."""

import asyncio
import base64
import json
import ssl as ssl_module
import time
from unittest.mock import Mock

import aiohttp
import pytest

from livekit.plugins.voxist.connection import VoxistDialer
from livekit.plugins.voxist.exceptions import AuthenticationError, ConnectionError


def handshake_error(status: int) -> aiohttp.WSServerHandshakeError:
    return aiohttp.WSServerHandshakeError(
        Mock(), (), status=status, message="handshake rejected"
    )


class FakeResponse:
    """Async-context-manager response for FakeSession.get."""

    def __init__(self, *, status=200, payload=None, json_exc=None, enter_exc=None):
        self.status = status
        self._payload = payload
        self._json_exc = json_exc
        self._enter_exc = enter_exc

    async def __aenter__(self):
        if self._enter_exc is not None:
            raise self._enter_exc
        return self

    async def __aexit__(self, *args):
        return False

    async def json(self):
        if self._json_exc is not None:
            raise self._json_exc
        return self._payload

    async def text(self):
        return "error body"


class FakeSession:
    """
    Just enough of aiohttp.ClientSession for the dialer: get() hands back a
    scripted response per call, ws_connect() a scripted result per call.
    """

    def __init__(self, responses=(), ws_results=()):
        self.closed = False
        self._responses = list(responses)
        self._ws_results = list(ws_results)
        self.get_calls: list = []
        self.ws_calls: list = []

    def get(self, url, **kwargs):
        self.get_calls.append((url, kwargs))
        return self._responses.pop(0)

    async def ws_connect(self, url, **kwargs):
        self.ws_calls.append((url, kwargs))
        result = self._ws_results.pop(0)
        if isinstance(result, BaseException):
            raise result
        if result == "hang":
            await asyncio.Event().wait()
        return result


def token_response(token: str) -> FakeResponse:
    return FakeResponse(payload={"url": f"wss://host/ws?token={token}"})


def make_dialer(session, **kw) -> VoxistDialer:
    return VoxistDialer(session=session, base_url="wss://host/ws", api_key="k", **kw)


def prime_cache(dialer: VoxistDialer, token: str) -> str:
    """Install a cached token that reads as valid for hours."""
    url = f"wss://host/ws?token={token}"
    dialer._token_url = url
    dialer._token_expires_at = time.time() + 7200.0
    return url


@pytest.mark.no_auto_mock_token
class TestStaleCachedTokenRedial:
    """
    A 401 handshake on a CACHED token must not be treated as a revoked API
    key: the cache is invalidated, the key is exchanged for a fresh token
    once, and the dial is retried. Only a fresh token's rejection is fatal.
    """

    @pytest.mark.asyncio
    async def test_stale_cached_token_is_refetched_and_redialed(self):
        ws = object()
        session = FakeSession(
            responses=[token_response("fresh")],
            ws_results=[handshake_error(401), ws],
        )
        dialer = make_dialer(session)
        prime_cache(dialer, "stale")

        result = await dialer.dial("fr", 16000)

        assert result is ws
        assert len(session.ws_calls) == 2
        assert "token=stale" in session.ws_calls[0][0]
        assert "token=fresh" in session.ws_calls[1][0]
        assert len(session.get_calls) == 1, "exactly one refetch"
        assert dialer._token_url is not None and "fresh" in dialer._token_url

    @pytest.mark.asyncio
    async def test_fresh_token_rejection_is_fatal(self):
        """When even the freshly exchanged token 401s, the key is the
        problem - AuthenticationError, and no endless redial loop."""
        session = FakeSession(
            responses=[token_response("fresh")],
            ws_results=[handshake_error(401), handshake_error(401)],
        )
        dialer = make_dialer(session)
        prime_cache(dialer, "stale")

        with pytest.raises(AuthenticationError):
            await dialer.dial("fr", 16000)

        assert len(session.ws_calls) == 2, "exactly one redial, never a loop"
        assert len(session.get_calls) == 1

    @pytest.mark.asyncio
    async def test_byte_identical_refetched_token_still_gets_one_redial(self):
        """
        [3] JWTs have one-second iat/exp granularity: a transient 401
        (token-store propagation lag, gateway restart) followed by a refetch
        in the same second yields a byte-identical token - and a server
        reusing tokens within validity always returns the same URL. Neither
        is proof of a bad key: the redial must happen (bounded by the
        attempt counter, never by token comparison) and the stream survives.
        """
        ws = object()
        session = FakeSession(
            responses=[token_response("same")],
            ws_results=[handshake_error(401), ws],
        )
        dialer = make_dialer(session)
        prime_cache(dialer, "same")

        result = await dialer.dial("fr", 16000)

        assert result is ws, "the stream must survive a transient 401"
        assert len(session.ws_calls) == 2, "one redial with the refetched token"
        assert "token=same" in session.ws_calls[1][0]
        assert len(session.get_calls) == 1, "exactly one refetch"

    @pytest.mark.asyncio
    async def test_revoked_key_fails_fast_at_the_refetch_exchange(self):
        """
        [3] A genuinely revoked key still fails fast without the
        identical-token heuristic: the token EXCHANGE itself 401s during
        the refetch and raises AuthenticationError directly - no redial.
        """
        session = FakeSession(
            responses=[FakeResponse(status=401)],
            ws_results=[handshake_error(401)],
        )
        dialer = make_dialer(session)
        prime_cache(dialer, "cached")

        with pytest.raises(AuthenticationError):
            await dialer.dial("fr", 16000)

        assert len(session.ws_calls) == 1, "no redial with a rejected key"
        assert len(session.get_calls) == 1

    @pytest.mark.asyncio
    async def test_non_auth_handshake_failure_does_not_refetch(self):
        session = FakeSession(ws_results=[handshake_error(500)])
        dialer = make_dialer(session)
        prime_cache(dialer, "any")

        with pytest.raises(ConnectionError):
            await dialer.dial("fr", 16000)
        assert session.get_calls == []

    @pytest.mark.asyncio
    async def test_invalidate_preserves_a_concurrent_refetch(self):
        """
        Race safety: if another stream already replaced the cached token by
        the time our 401 comes back, invalidating with OUR stale URL must
        not clobber the newer token.
        """
        dialer = make_dialer(FakeSession())
        newer = prime_cache(dialer, "newer")

        await dialer._invalidate_token("wss://host/ws?token=stale")
        assert dialer._token_url == newer, "the concurrent refetch survives"

        await dialer._invalidate_token(newer)
        assert dialer._token_url is None, "our own stale token is dropped"


@pytest.mark.no_auto_mock_token
class TestTokenExchangeErrorMapping:
    """
    Every transport-shaped failure of the exchange must surface as our
    ConnectionError (which the stream maps to a retryable APIError) - never
    as a raw TimeoutError/JSONDecodeError/RuntimeError that kills the
    stream with zero retries.
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "enter_exc",
        [
            asyncio.TimeoutError(),  # the 10s ClientTimeout firing
            RuntimeError("Session is closed"),  # shutdown race
        ],
        ids=["timeout", "closed-session-race"],
    )
    async def test_request_failures_map_to_connection_error(self, enter_exc):
        session = FakeSession(responses=[FakeResponse(enter_exc=enter_exc)])
        dialer = make_dialer(session)

        with pytest.raises(ConnectionError):
            await dialer._get_token_url()

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "json_exc",
        [
            json.JSONDecodeError("bad", "doc", 0),
            aiohttp.ContentTypeError(Mock(), ()),
        ],
        ids=["invalid-json-body", "non-json-content-type"],
    )
    async def test_body_decode_failures_map_to_connection_error(self, json_exc):
        session = FakeSession(responses=[FakeResponse(json_exc=json_exc)])
        dialer = make_dialer(session)

        with pytest.raises(ConnectionError):
            await dialer._get_token_url()

    @pytest.mark.asyncio
    async def test_non_object_json_maps_to_connection_error(self):
        session = FakeSession(responses=[FakeResponse(payload=["not", "a", "dict"])])
        dialer = make_dialer(session)

        with pytest.raises(ConnectionError):
            await dialer._get_token_url()

    @pytest.mark.asyncio
    async def test_unenumerated_body_failure_maps_to_connection_error(self):
        """[5] UnicodeDecodeError is outside the enumerated tuple; the
        catch-all must map it to our retryable ConnectionError instead of
        letting it kill the stream with zero retries."""
        session = FakeSession(
            responses=[
                FakeResponse(
                    json_exc=UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")
                )
            ]
        )
        dialer = make_dialer(session)

        with pytest.raises(ConnectionError, match="unexpectedly"):
            await dialer._get_token_url()

    @pytest.mark.asyncio
    async def test_dial_unenumerated_failure_maps_to_connection_error(self):
        """[5] Same audit for dial(): an exotic exception from the
        handshake must not escape unmapped."""
        session = FakeSession(
            ws_results=[UnicodeDecodeError("utf-8", b"\xff", 0, 1, "bad")]
        )
        dialer = make_dialer(session)
        prime_cache(dialer, "cached")

        with pytest.raises(ConnectionError, match="unexpectedly"):
            await dialer.dial("fr", 16000)

    @pytest.mark.asyncio
    async def test_auth_error_is_never_double_wrapped(self):
        """[5] Our own AuthenticationError must pass through the catch-all
        untouched - wrapping it into ConnectionError would turn a fatal
        credential failure into an endless retry loop."""
        session = FakeSession(responses=[FakeResponse(status=401)])
        dialer = make_dialer(session)

        with pytest.raises(AuthenticationError):
            await dialer._get_token_url()

    @pytest.mark.asyncio
    async def test_status_connection_error_is_never_double_wrapped(self):
        """[5] Our own ConnectionError from the status check must surface
        as-is, not re-wrapped by the catch-all."""
        session = FakeSession(responses=[FakeResponse(status=503)])
        dialer = make_dialer(session)

        with pytest.raises(ConnectionError) as exc_info:
            await dialer._get_token_url()
        assert str(exc_info.value) == "Token exchange failed with status 503"

    @pytest.mark.asyncio
    async def test_cancellation_is_never_swallowed_by_the_exchange(self):
        """[5] CancelledError derives from BaseException (3.10+), so no
        `except Exception` clause may absorb it."""
        session = FakeSession(
            responses=[FakeResponse(enter_exc=asyncio.CancelledError())]
        )
        dialer = make_dialer(session)

        with pytest.raises(asyncio.CancelledError):
            await dialer._get_token_url()

    @pytest.mark.asyncio
    async def test_cancellation_is_never_swallowed_by_the_dial(self):
        session = FakeSession(ws_results=[asyncio.CancelledError()])
        dialer = make_dialer(session)
        prime_cache(dialer, "cached")

        with pytest.raises(asyncio.CancelledError):
            await dialer.dial("fr", 16000)


@pytest.mark.no_auto_mock_token
class TestMalformedTokenUrl:
    """
    [4] A malformed 'url' in the token response must be rejected AT THE
    EXCHANGE as our ConnectionError - not crash dial() later with an
    unmapped TypeError, which livekit treats as instant death (no retries,
    no error event).
    """

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "url",
        [
            123,
            ["wss://host/ws"],
            {"nested": "wss://host/ws"},
            True,
            "https://host/ws?token=t",
            "not-a-url",
        ],
        ids=["int", "list", "dict", "bool", "http-scheme", "garbage"],
    )
    async def test_malformed_url_maps_to_connection_error(self, url):
        session = FakeSession(responses=[FakeResponse(payload={"url": url})])
        dialer = make_dialer(session)

        with pytest.raises(ConnectionError, match="malformed url"):
            await dialer._get_token_url()

        assert dialer._token_url is None, "a malformed url must not be cached"

    @pytest.mark.asyncio
    async def test_empty_url_maps_to_connection_error(self):
        session = FakeSession(responses=[FakeResponse(payload={"url": ""})])
        dialer = make_dialer(session)

        with pytest.raises(ConnectionError):
            await dialer._get_token_url()

    @pytest.mark.asyncio
    async def test_closed_session_raises_before_the_request(self):
        session = FakeSession()
        session.closed = True
        dialer = make_dialer(session)

        with pytest.raises(ConnectionError, match="session is closed"):
            await dialer._get_token_url()

    @pytest.mark.asyncio
    async def test_dial_on_closed_session_maps_to_connection_error(self):
        session = FakeSession()
        session.closed = True
        dialer = make_dialer(session)
        prime_cache(dialer, "cached")  # skip the exchange, reach the dial

        with pytest.raises(ConnectionError, match="session is closed"):
            await dialer.dial("fr", 16000)

    @pytest.mark.asyncio
    async def test_dial_runtime_error_maps_to_connection_error(self):
        """aiohttp raises RuntimeError('Session is closed') when the session
        closes between our check and the connect."""
        session = FakeSession(ws_results=[RuntimeError("Session is closed")])
        dialer = make_dialer(session)
        prime_cache(dialer, "cached")

        with pytest.raises(ConnectionError):
            await dialer.dial("fr", 16000)


class TestConnectionTimeoutIsHonoured:
    """connection_timeout was documented but silently ignored (dial and
    token exchange both hardcoded 10s)."""

    @pytest.mark.asyncio
    @pytest.mark.no_auto_mock_token
    async def test_token_exchange_uses_the_configured_timeout(self):
        session = FakeSession(responses=[token_response("t")])
        dialer = make_dialer(session, connection_timeout=3.5)

        await dialer._get_token_url()

        (_, kwargs) = session.get_calls[0]
        assert kwargs["timeout"].total == 3.5

    @pytest.mark.asyncio
    @pytest.mark.no_auto_mock_token
    async def test_dial_uses_the_configured_timeout(self):
        session = FakeSession(ws_results=["hang"])
        dialer = make_dialer(session, connection_timeout=0.1)
        prime_cache(dialer, "cached")

        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(ConnectionError, match="timed out after 0.1s"):
            # bounded well below the old hardcoded 10s: if the parameter
            # were still ignored, this would time out the test instead
            await asyncio.wait_for(dialer.dial("fr", 16000), timeout=2.0)
        assert loop.time() - start < 2.0


def _url_with_exp(exp):
    def seg(obj):
        raw = json.dumps(obj).encode()
        return base64.urlsafe_b64encode(raw).decode().rstrip("=")

    token = f"{seg({'alg': 'HS256'})}.{seg({'exp': exp})}.sig"
    return f"wss://host/ws?token={token}"


class TestTokenExpiry:
    """
    Token lifetime must come from the JWT, sanity-clamped.

    The exchange response carries only a `url`, so expiry is read from the
    JWT inside it. The claim is server-supplied and compared against the
    local wall clock, so it cannot be trusted blindly in either direction.
    """

    def test_expiry_read_from_jwt(self):
        """A plausible exp claim wins over the one-hour default."""
        now = 1_800_000_000.0
        expected = now + 7200  # two hours: longer than the default, still sane
        assert VoxistDialer._token_expiry_from_url(_url_with_exp(expected), now) == expected

    def test_expired_claim_means_immediately_stale(self):
        """
        A token that reads as already expired is never cached as valid.

        Caching it for the assumed lifetime (an earlier behaviour) meant a
        clock ahead of the server made every dial reuse a dead token for ~55
        minutes; treating it as stale costs one refetch per dial until the
        skew clears, and always uses a fresh token.
        """
        now = 1_800_000_000.0
        got = VoxistDialer._token_expiry_from_url(_url_with_exp(now - 600), now)
        assert got <= now, "an expired claim must not yield a future expiry"

    def test_implausible_expiry_is_clamped(self):
        """Seconds-vs-milliseconds mix-ups fall back to the known lifetime."""
        now = 1_800_000_000.0
        got = VoxistDialer._token_expiry_from_url(
            _url_with_exp((now + 3600) * 1000), now
        )
        assert got == now + 3600.0

    def test_falls_back_when_unreadable(self):
        """Malformed or absent tokens assume the server's current lifetime."""
        now = 500.0
        for url in (
            "wss://host/ws",  # no token at all
            "wss://host/ws?token=not-a-jwt",  # no segments
            "wss://host/ws?token=a.!!!!.c",  # undecodable payload
            "wss://host/ws?token=a.e30.c",  # valid JSON, no exp
        ):
            assert VoxistDialer._token_expiry_from_url(url, now) == now + 3600.0


class TestUrlAndSslPolicy:
    def _dialer(self, base_url, **kw):
        return VoxistDialer(
            session=None,  # type: ignore[arg-type]  # not used by these tests
            base_url=base_url,
            api_key="k",
            **kw,
        )

    def test_http_base_url_derivation(self):
        assert (
            self._dialer("wss://api-asr.voxist.com/ws")._http_base_url()
            == "https://api-asr.voxist.com"
        )
        assert (
            self._dialer("ws://localhost:3000/ws")._http_base_url()
            == "http://localhost:3000"
        )

    def test_ssl_never_disabled(self):
        """
        Without an explicit context the fallback must still verify.

        The token exchange can return a wss:// URL even for a ws:// base_url,
        and ssl=False on it would expose the JWT and all audio to a
        man-in-the-middle. There is no configuration that yields False.
        """
        assert self._dialer("ws://localhost/ws")._ssl_param() is True

        ctx = ssl_module.create_default_context()
        assert self._dialer("wss://x/ws", ssl_context=ctx)._ssl_param() is ctx


@pytest.mark.no_auto_mock_token
class TestTokenExchangeAgainstServer:
    """The real exchange path, against the mock server's token endpoint."""

    @pytest.mark.asyncio
    async def test_token_fetched_and_cached(self, mock_voxist_server):
        async with aiohttp.ClientSession() as session:
            dialer = VoxistDialer(
                session=session,
                base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
                api_key="test_key",
            )
            url1 = await dialer._get_token_url()
            url2 = await dialer._get_token_url()

        assert "token=" in url1 and url1 == url2
        assert mock_voxist_server.token_requests_count == 1, (
            "the token must be cached across dials, not refetched"
        )

    @pytest.mark.asyncio
    async def test_bad_key_raises_authentication_error(self, mock_voxist_server):
        async with aiohttp.ClientSession() as session:
            dialer = VoxistDialer(
                session=session,
                base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
                api_key="wrong_key",
            )
            with pytest.raises(AuthenticationError):
                await dialer._get_token_url()

    @pytest.mark.asyncio
    async def test_unreachable_endpoint_is_a_connection_error(self):
        """Transport failures are retryable ConnectionError, never auth."""
        async with aiohttp.ClientSession() as session:
            dialer = VoxistDialer(
                session=session,
                base_url="ws://127.0.0.1:9/ws",  # discard port: refused
                api_key="k",
            )
            with pytest.raises(ConnectionError):
                await dialer._get_token_url()

    @pytest.mark.asyncio
    async def test_dial_appends_language_and_rate(self, mock_voxist_server):
        """The per-stream language rides each dial's URL - nothing else."""
        async with aiohttp.ClientSession() as session:
            dialer = VoxistDialer(
                session=session,
                base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
                api_key="test_key",
            )
            ws = await dialer.dial("fr-medical", 16000)
            await ws.close()

        assert mock_voxist_server.connected_languages == ["fr-medical"]
