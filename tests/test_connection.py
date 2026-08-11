"""Unit tests for VoxistDialer: token exchange, expiry, and dialing."""

import asyncio
import base64
import gc
import hashlib
import json
import ssl as ssl_module
import time
from unittest.mock import Mock

import aiohttp
import pytest

from livekit.plugins.voxist import connection
from livekit.plugins.voxist.connection import (
    DIAL_RATE_LIMIT_WINDOW_SECONDS,
    MAX_DIALS_PER_WINDOW,
    VoxistDialer,
    _DialRateLimiter,
    reset_dial_rate_limits,
)
from livekit.plugins.voxist.exceptions import AuthenticationError, ConnectionError


@pytest.fixture(autouse=True)
def _isolated_dial_rate_limits():
    """The dial limiter is shared process-wide per gateway credential, so
    every test in this module starts from an empty registry (dialers built
    afterwards get fresh windows) and leaves none behind."""
    reset_dial_rate_limits()
    yield
    reset_dial_rate_limits()


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


def _dialer_capturing_url(monkeypatch):
    """A primed dialer whose ws_connect URL is captured, no network."""
    session = FakeSession(ws_results=[object()])
    dialer = make_dialer(session)
    prime_cache(dialer, "tok")
    captured: list[str] = []

    original = session.ws_connect

    async def spy(url, **kwargs):
        captured.append(url)
        return await original(url, **kwargs)

    monkeypatch.setattr(session, "ws_connect", spy)
    return dialer, captured


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


class FakeClock:
    """Deterministic stand-in for the `time` module inside connection.py.

    Injected with monkeypatch.setattr(connection, "time", clock) so the
    global clock - which asyncio itself depends on - is never touched.
    """

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def time(self) -> float:
        return self.now

    def monotonic(self) -> float:
        return self.now


@pytest.mark.no_auto_mock_token
class TestTokenClockReadAfterLock:
    """
    [B] The clock must be read INSIDE _token_lock. Reading it before the
    await meant that, after lock contention, both the freshness check and the
    recorded cache expiry used a timestamp that was already seconds old - so
    a token could be judged fresh against a time that had passed, or have its
    expiry recorded against a moment before the round-trip it describes.
    """

    @pytest.mark.asyncio
    async def test_freshness_check_uses_the_post_lock_clock(self, monkeypatch):
        clock = FakeClock(0.0)
        monkeypatch.setattr(connection, "time", clock)

        session = FakeSession(responses=[token_response("fresh")])
        dialer = make_dialer(session)
        # A cached token that is fresh at t=0 but inside the 300s refresh
        # buffer from t=10 onwards.
        dialer._token_url = "wss://host/ws?token=cached"
        dialer._token_expires_at = 310.0

        # Hold the lock so the caller must wait, exactly as a concurrent
        # exchange would make it wait.
        await dialer._token_lock.acquire()
        task = asyncio.create_task(dialer._get_token_url())
        for _ in range(5):
            await asyncio.sleep(0)  # let the task reach the lock

        clock.now = 20.0  # the wait for the lock cost 20 seconds
        dialer._token_lock.release()
        url = await task

        assert "token=fresh" in url, (
            "the stale cached token was judged fresh against a pre-lock clock"
        )
        assert len(session.get_calls) == 1

    @pytest.mark.asyncio
    async def test_expiry_is_recorded_from_the_arrival_clock(self, monkeypatch):
        clock = FakeClock(0.0)
        monkeypatch.setattr(connection, "time", clock)

        class SlowResponse(FakeResponse):
            async def json(self):
                clock.now += 20.0  # the HTTPS round-trip took 20s
                return self._payload

        session = FakeSession(
            responses=[SlowResponse(payload={"url": "wss://host/ws?token=nojwt"})]
        )
        dialer = make_dialer(session)

        await dialer._get_token_url()

        # exp unreadable -> assume the 1h lifetime, counted from ARRIVAL
        assert dialer._token_expires_at == pytest.approx(3620.0), (
            "the expiry was recorded against the clock read before the "
            "round-trip"
        )


class TestDialRateLimitWindow:
    """[12] The sliding window itself, exercised with a synthetic clock."""

    def test_window_slides(self):
        limiter = _DialRateLimiter(max_dials=2, window=60.0)

        assert limiter.reserve(100.0) == 0.0
        assert limiter.reserve(100.0) == 0.0
        # full at t=120: the earliest attempt (t=100) frees at t=160
        assert limiter.reserve(120.0) == pytest.approx(40.0)
        # the first two attempts age out once t - 60 passes them
        assert limiter.reserve(161.0) == 0.0
        assert limiter.reserve(161.0) == 0.0
        assert limiter.reserve(161.0) > 0.0

    def test_the_reported_wait_is_the_earliest_possible_slot(self):
        """The wait handed to a parked caller must be the moment the EARLIEST
        charged attempt ages out - never a fixed backoff, which would either
        overshoot (idle streams) or undershoot (a spin)."""
        limiter = _DialRateLimiter(max_dials=2, window=10.0)
        limiter.reserve(0.0)
        limiter.reserve(5.0)

        assert limiter.reserve(6.0) == pytest.approx(4.0)  # t=0 frees at t=10
        limiter.reserve(10.0)  # the t=0 charge has aged out, so this succeeds
        assert limiter.reserve(11.0) == pytest.approx(4.0)  # t=5 frees at t=15

    def test_defaults_match_the_retired_pool_budget(self):
        assert MAX_DIALS_PER_WINDOW == 30
        assert DIAL_RATE_LIMIT_WINDOW_SECONDS == 60.0


@pytest.mark.no_auto_mock_token
class TestDialRateLimit:
    """
    [12] With the pool gone, every stream dials independently under livekit's
    per-stream retry with no coordination: a gateway outage with N streams
    produced a thundering herd of dials (plus one HTTPS token exchange per
    401-rejected cached token) against a proxy pod that admits a fixed number
    of concurrent sockets and closes the surplus with 1013 "retry in a few
    seconds". Dial ATTEMPTS are therefore capped per gateway credential, and
    an exhausted window makes the caller WAIT for the window to slide (see
    TestDialRateLimitWaits for why refusing was a defect).
    """

    @pytest.mark.asyncio
    async def test_the_ordinary_dial_is_charged_exactly_once(self):
        """One dial(), one charge. (Characterisation: the reviewer asked
        whether charging before the token exchange AND again before the
        post-401 redial could double-charge one logical dial and halve the
        effective ceiling. It cannot - the second charge only happens on the
        stale-token path, where a second exchange and a second socket really
        do go on the wire.)"""
        ws = object()
        session = FakeSession(responses=[token_response("t")], ws_results=[ws])
        dialer = make_dialer(session, max_dials_per_window=5)

        assert await dialer.dial("fr", 16000) is ws

        assert dialer._rate_limiter.charged_attempts == 1
        assert len(session.get_calls) == 1
        assert len(session.ws_calls) == 1

    @pytest.mark.asyncio
    async def test_budget_is_shared_across_dialer_instances(self):
        """Process-wide, not per dialer: one pod's admission budget for one
        key does not grow because a process hosts several plugins."""
        ws = object()
        first = FakeSession(responses=[token_response("t")], ws_results=[ws])
        second = FakeSession(responses=[token_response("t")], ws_results=[ws])
        d1 = make_dialer(first, max_dials_per_window=1)
        # A wait bound short enough to assert on: the shared window is the
        # 60s default, so the default bound would park d2 for a full minute.
        d2 = make_dialer(
            second, max_dials_per_window=1, max_dial_rate_limit_wait=0.02
        )

        assert d1._rate_limiter is d2._rate_limiter

        assert await d1.dial("fr", 16000) is ws
        with pytest.raises(ConnectionError, match="Dial rate limit"):
            await d2.dial("fr", 16000)
        assert second.ws_calls == []

    @pytest.mark.asyncio
    async def test_a_different_credential_has_its_own_budget(self):
        ws = object()
        mine = FakeSession(responses=[token_response("t")], ws_results=[ws])
        theirs = FakeSession(responses=[token_response("t")], ws_results=[ws])
        d1 = make_dialer(mine, max_dials_per_window=1)
        d2 = VoxistDialer(
            session=theirs,
            base_url="wss://host/ws",
            api_key="a-different-key",
            max_dials_per_window=1,
        )

        assert d1._rate_limiter is not d2._rate_limiter
        assert await d1.dial("fr", 16000) is ws
        assert await d2.dial("fr", 16000) is ws

    @pytest.mark.asyncio
    async def test_the_post_401_redial_is_charged_too(self):
        """The extra token exchange a rejected cached token triggers is part
        of the herd, so the redial attempt draws from the same budget - and
        the resulting error is surfaced verbatim, not double-wrapped by
        dial()'s catch-all ([E])."""
        session = FakeSession(
            responses=[token_response("fresh")],
            ws_results=[handshake_error(401)],
        )
        dialer = make_dialer(
            session, max_dials_per_window=1, max_dial_rate_limit_wait=0.02
        )
        prime_cache(dialer, "stale")

        with pytest.raises(ConnectionError) as excinfo:
            await dialer.dial("fr", 16000)

        assert "Dial rate limit" in str(excinfo.value)
        assert "unexpectedly" not in str(excinfo.value), (
            "our own mapped error must not be re-wrapped by the catch-all"
        )
        assert len(session.ws_calls) == 1, "no second socket"
        assert session.get_calls == [], "and no extra token exchange either"


@pytest.mark.no_auto_mock_token
class TestDialRateLimitWaits:
    """
    [B] An exhausted window must DELAY a dial, not refuse it.

    The defect: the limiter used to fail the dial with a retryable
    ConnectionError, so our own throttle spent livekit's finite retry budget.
    RecognizeStream._main_task grants max_retry (default 3) + 1 attempts at
    ~retry_interval (default 2s) apart and only resets _num_retries when a
    FINAL_TRANSCRIPT arrives - which during an outage never happens. All four
    attempts therefore landed inside the same 60s window, every one refused by
    us rather than by the gateway, and the stream died permanently with
    "failed to recognize speech after 3 attempts" even though the gateway
    recovered seconds later. The deleted pool slept out the window instead.

    These tests use real (small) windows and the real clock: the wait loop
    reads the clock it sleeps against, so a frozen FakeClock cannot drive it.
    Only lower bounds are asserted on elapsed time, so a loaded machine can
    never make them flaky.
    """

    @pytest.mark.asyncio
    async def test_an_exhausted_window_waits_and_then_dials(self):
        ws = object()
        session = FakeSession(
            responses=[token_response("t")], ws_results=[ws, ws]
        )
        dialer = make_dialer(
            session, max_dials_per_window=1, dial_rate_limit_window=0.2
        )

        assert await dialer.dial("fr", 16000) is ws

        started = time.perf_counter()
        assert await dialer.dial("fr", 16000) is ws, (
            "the second dial was refused instead of waiting for the window"
        )
        elapsed = time.perf_counter() - started

        assert elapsed >= 0.1, (
            f"the second dial returned in {elapsed:.3f}s, so it cannot have "
            "waited for the 0.2s window to slide"
        )
        assert len(session.ws_calls) == 2

    @pytest.mark.asyncio
    async def test_a_slot_beyond_the_wait_bound_fails_retryably_at_once(self):
        """Waiting must never park a caller indefinitely. reserve() reports
        the exact moment the earliest slot frees, so when that is past the
        caller's bound there is nothing to wait for - it fails immediately,
        and the message names the bound so the throttle is not mistaken for
        the gateway being down."""
        ws = object()
        session = FakeSession(responses=[token_response("t")], ws_results=[ws])
        dialer = make_dialer(
            session,
            max_dials_per_window=1,
            dial_rate_limit_window=30.0,
            max_dial_rate_limit_wait=0.05,
        )

        assert await dialer.dial("fr", 16000) is ws

        with pytest.raises(ConnectionError) as excinfo:
            await dialer.dial("fr", 16000)

        message = str(excinfo.value)
        assert "Dial rate limit" in message
        assert "1 dial attempts per 30s" in message, message
        assert "of the 0.05s bound" in message, (
            "the refusal must be attributed to the caller's wait bound, not "
            f"presented as a flat refusal to dial; got: {message}"
        )
        assert len(session.ws_calls) == 1, "the refused dial never reached the wire"

    @pytest.mark.asyncio
    async def test_a_starved_waiter_gives_up_at_its_bound(self):
        """The bound is what makes waiting safe: two callers contend for one
        slot, the winner takes it, and the loser - whose next slot is a whole
        window away - gives up at its bound instead of waiting forever."""
        ws = object()
        session = FakeSession(
            responses=[token_response("t")], ws_results=[ws, ws]
        )
        dialer = make_dialer(
            session, max_dials_per_window=1, dial_rate_limit_window=0.3
        )
        assert await dialer.dial("fr", 16000) is ws  # fills the window

        started = time.perf_counter()
        first, second = await asyncio.gather(
            dialer.dial("fr", 16000),
            dialer.dial("fr", 16000),
            return_exceptions=True,
        )
        elapsed = time.perf_counter() - started

        outcomes = [first, second]
        assert sum(o is ws for o in outcomes) == 1, outcomes
        losers = [o for o in outcomes if o is not ws]
        assert len(losers) == 1 and isinstance(losers[0], ConnectionError), outcomes
        assert "Dial rate limit" in str(losers[0])
        assert elapsed >= 0.2, (
            f"both dials resolved in {elapsed:.3f}s, so neither waited for the "
            "0.3s window to slide"
        )
        assert len(session.ws_calls) == 2, "one waiter got its socket"

    def test_the_default_wait_bound_is_the_window(self):
        """The window is the largest wait the limiter can ever ask for, so a
        bound of one window means a caller that is merely EARLY always gets its
        slot instead of dying - the refusal path is reachable only under
        contention. A shorter default would reinstate the defect above."""
        dialer = make_dialer(FakeSession(), dial_rate_limit_window=17.0)
        assert dialer._max_dial_rate_limit_wait == 17.0

    @pytest.mark.asyncio
    async def test_cancelling_a_parked_dial_leaves_no_charge_or_held_lock(self):
        """
        [B] Cancellation correctness for the wait, and the two properties the
        reviewer asked to confirm:

        - the threading.Lock is genuinely not held across the await (a parked
          waiter must not block a dialer on another event loop, and a
          cancellation cannot release a threading.Lock)
        - a caller cancelled while parked charges nothing: the attempt is
          appended only by the iteration that actually proceeds
        """
        ws = object()
        session = FakeSession(
            responses=[token_response("t")], ws_results=[ws, ws]
        )
        dialer = make_dialer(
            session, max_dials_per_window=1, dial_rate_limit_window=5.0
        )
        assert await dialer.dial("fr", 16000) is ws

        parked = asyncio.create_task(dialer.dial("fr", 16000))
        await asyncio.sleep(0.05)  # let it reach the wait
        assert not parked.done(), (
            "the second dial failed instead of parking for a free slot"
        )

        limiter = dialer._rate_limiter
        acquired = limiter._lock.acquire(blocking=False)
        if acquired:
            limiter._lock.release()
        assert acquired, "the threading.Lock is held across the await"
        assert limiter.charged_attempts == 1, "the parked waiter charged a slot"

        parked.cancel()
        with pytest.raises(asyncio.CancelledError):
            await parked

        assert len(session.ws_calls) == 1, "the cancelled dial reached the wire"
        assert limiter.charged_attempts == 1, (
            "cancellation left a phantom charge in the window"
        )
        assert limiter._lock.acquire(blocking=False), (
            "the cancelled waiter stranded the lock"
        )
        limiter._lock.release()


@pytest.mark.no_auto_mock_token
class TestDialLimiterRegistryLifetime:
    """
    [B] The process-wide budget must outlive any individual dialer.

    The defect: _dial_limiters was a weakref.WeakValueDictionary whose only
    strong reference was VoxistDialer._rate_limiter, so the "process-wide"
    budget silently reset to a full window as soon as the last dialer for a
    credential was collected. The common LiveKit agent shape builds one
    VoxistSTT per job and awaits aclose() in a finally, so during an outage
    each job burned its dials, its teardown let the limiter be collected, and
    the next job started with an empty deque - handing the gateway exactly the
    unbounded herd the limiter exists to prevent.
    """

    @pytest.mark.asyncio
    async def test_the_budget_survives_dialer_teardown_and_gc(self):
        ws = object()
        first = FakeSession(responses=[token_response("t")], ws_results=[ws])
        # A generous window: gc.collect() on a loaded interpreter is not free,
        # and the assertion below is a LOWER bound on the successor's wait, so
        # the window only has to outlast the teardown.
        dialer = make_dialer(
            first, max_dials_per_window=1, dial_rate_limit_window=0.6
        )
        assert await dialer.dial("fr", 16000) is ws

        # Job over: the only dialer for this credential goes away.
        del dialer
        gc.collect()

        # Next job, same credential: it must inherit the spent window.
        second = FakeSession(responses=[token_response("t")], ws_results=[ws])
        successor = make_dialer(
            second, max_dials_per_window=1, dial_rate_limit_window=0.6
        )

        started = time.perf_counter()
        assert await successor.dial("fr", 16000) is ws
        elapsed = time.perf_counter() - started

        assert elapsed >= 0.15, (
            f"the successor dialed in {elapsed:.3f}s, so teardown reset the "
            "process-wide window to a full budget"
        )

    def test_a_charged_limiter_outlives_gc_but_an_idle_one_is_evicted(
        self, monkeypatch
    ):
        """Strong references alone would leak one entry per credential
        forever, so growth is bounded by idle eviction instead: an entry is
        reclaimed only once its window has been empty for well over the window
        duration, by which point dropping it cannot restore a budget its owner
        had not already earned back."""
        clock = FakeClock(1000.0)
        monkeypatch.setattr(connection, "time", clock)
        fingerprint = hashlib.sha256(b"k").hexdigest()[:16]
        mine = f"wss://host/ws|{fingerprint}|2/10.0"

        limiter = connection._limiter_for(
            base_url="wss://host/ws", api_key="k", max_dials=2, window=10.0
        )
        limiter.reserve(clock.now)
        del limiter
        gc.collect()

        assert list(connection._dial_limiters) == [mine], (
            "a charged limiter was collected the moment nobody held it"
        )
        survivor = connection._limiter_for(
            base_url="wss://host/ws", api_key="k", max_dials=2, window=10.0
        )
        assert survivor.charged_attempts == 1

        # Still inside window + grace: not yet reclaimable. (The sweep runs on
        # every _limiter_for call, so another credential's lookup drives it.)
        idle_at = 1000.0 + 10.0 + connection._DIAL_LIMITER_IDLE_GRACE_SECONDS
        clock.now = idle_at - 1
        connection._limiter_for(
            base_url="wss://other/ws", api_key="k", max_dials=2, window=10.0
        )
        assert mine in connection._dial_limiters

        # Past window + grace with an empty window: reclaimed.
        clock.now = idle_at + 1
        connection._limiter_for(
            base_url="wss://third/ws", api_key="k", max_dials=2, window=10.0
        )
        assert mine not in connection._dial_limiters, connection._dial_limiters

    def test_reset_dial_rate_limits_clears_the_registry(self):
        """The test seam must fully reset whatever structure backs the
        registry, or state leaks between tests."""
        connection._limiter_for(
            base_url="wss://host/ws", api_key="k", max_dials=1, window=5.0
        ).reserve(0.0)
        assert connection._dial_limiters

        reset_dial_rate_limits()

        assert connection._dial_limiters == {}


class TestPunctuationModeOnTheDialUrl:
    """
    punctuation_mode rides the connect URL, verified live: with 'Dictated' the
    same audio returns without automatic commas or sentence periods.

    Deliberately NOT a {"config": {...}} message. That message can also carry
    sample_rate and the gateway bills on the last rate it was told, so a config
    echoing the caller's 48000 while the wire carries 16kHz would under-report
    usage threefold.
    """

    @pytest.mark.asyncio
    async def test_it_is_appended_when_set(self, monkeypatch):
        dialer, captured = _dialer_capturing_url(monkeypatch)
        await dialer.dial("fr", 16000, punctuation_mode="Dictated")
        assert "punctuation_mode=Dictated" in captured[0]

    @pytest.mark.asyncio
    async def test_it_is_absent_by_default(self, monkeypatch):
        dialer, captured = _dialer_capturing_url(monkeypatch)
        await dialer.dial("fr", 16000)
        assert "punctuation_mode" not in captured[0]

    @pytest.mark.asyncio
    async def test_sample_rate_is_always_the_wire_rate(self, monkeypatch):
        """The billing hazard: the URL must never carry the caller's rate."""
        dialer, captured = _dialer_capturing_url(monkeypatch)
        await dialer.dial("fr", 16000, punctuation_mode="Dictated")
        assert "sample_rate=16000" in captured[0]

    @pytest.mark.asyncio
    async def test_a_hostile_value_cannot_inject_query_parameters(
        self, monkeypatch
    ):
        dialer, captured = _dialer_capturing_url(monkeypatch)
        await dialer.dial(
            "fr", 16000, punctuation_mode="Dictated&sample_rate=48000"
        )
        assert "sample_rate=48000" not in captured[0], (
            "an unsanitized punctuation_mode could forge the billing rate"
        )
