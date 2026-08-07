"""Unit tests for VoxistDialer: token exchange, expiry, and dialing."""

import base64
import json
import ssl as ssl_module

import aiohttp
import pytest

from livekit.plugins.voxist.connection import VoxistDialer
from livekit.plugins.voxist.exceptions import AuthenticationError, ConnectionError


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
