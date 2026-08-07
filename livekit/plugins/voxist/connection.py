"""Token exchange and WebSocket dialing for the Voxist gateway.

This replaces the former connection *pool*. The gateway's protocol makes
long-lived reusable sockets impossible: "Done" is the end-of-session signal,
the ASR engine closes when it has flushed, and the gateway then closes the
client socket (simple-websocket-proxy.gateway.ts:899). A socket is therefore
spent at the end of every session, and each stream dials its own.

Everything that fought that fact - heartbeat staleness machinery, recycling
versus retirement, per-connection retry budgets, load balancing on buffer
snapshots - is gone with the pool. What remains is the part that was always
sound: the SEC-001 token exchange, its JWT-derived expiry, and a shared SSL
policy.
"""

from __future__ import annotations

import asyncio
import json
import ssl
import time
from base64 import urlsafe_b64decode
from binascii import Error as BinasciiError
from urllib.parse import parse_qs, urlsplit

import aiohttp

from .exceptions import AuthenticationError, ConnectionError
from .log import logger
from .models import sanitize_url_param

# Refresh the token this long before its expiry
TOKEN_REFRESH_BUFFER_SECONDS = 300.0

# A dial that cannot complete in this long means the endpoint is unreachable
DIAL_TIMEOUT_SECONDS = 10.0


class VoxistDialer:
    """
    Exchanges the API key for a short-lived WebSocket URL and dials it.

    One instance per VoxistSTT; every stream calls dial() for its own socket.
    The token URL is cached across dials (SEC-001: the API key travels only in
    an HTTPS header, never in a WebSocket URL), with expiry read from the
    JWT's own exp claim.
    """

    def __init__(
        self,
        *,
        session: aiohttp.ClientSession,
        base_url: str,
        api_key: str,
        api_key_header: str = "X-LVL-KEY",
        ssl_context: ssl.SSLContext | None = None,
        heartbeat_interval: float = 30.0,
    ) -> None:
        self._session = session
        self._base_url = base_url
        self._api_key = api_key
        self._api_key_header = api_key_header
        self._heartbeat_interval = heartbeat_interval

        # Certificate verification is never disabled. An explicit context is
        # the only way to trust a private CA; without one, aiohttp's default
        # verification applies (ssl=True) - even when base_url is ws://,
        # because the token exchange may hand back a wss:// URL.
        self._ssl_context = ssl_context

        self._token_url: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    def _ssl_param(self) -> ssl.SSLContext | bool:
        return self._ssl_context if self._ssl_context is not None else True

    def _http_base_url(self) -> str:
        url = self._base_url
        if url.startswith("wss://"):
            url = "https://" + url[6:]
        elif url.startswith("ws://"):
            url = "http://" + url[5:]
        if url.endswith("/ws"):
            url = url[:-3]
        return url

    @staticmethod
    def _token_expiry_from_url(token_url: str, now: float) -> float:
        """
        Read the token's expiry from its JWT exp claim.

        Only the payload is decoded, purely to schedule the refresh; the
        signature is the server's to verify. The claim is server-supplied and
        compared against the local wall clock, so it is sanity-clamped:

        - unreadable claim: assume the known 1h lifetime
          (websocket.controller.ts signs with expiresIn '1h')
        - already expired per the claim (clock skew): treat the token as
          immediately stale so every dial refetches, rather than caching a
          token the server may reject for the assumed lifetime
        - implausibly far ahead (skew or a seconds/milliseconds unit change):
          assume the known lifetime rather than trusting it
        """
        fallback = now + 3600.0
        try:
            query = urlsplit(token_url).query
            token = parse_qs(query).get("token", [""])[0]
            payload_segment = token.split(".")[1]
            padded = payload_segment + "=" * (-len(payload_segment) % 4)
            claims = json.loads(urlsafe_b64decode(padded))
            exp = float(claims["exp"])
        except (IndexError, KeyError, ValueError, TypeError, BinasciiError):
            logger.debug(
                "Could not read exp from the WebSocket token; assuming a "
                "1 hour lifetime"
            )
            return fallback

        if exp <= now:
            logger.warning(
                "WebSocket token reads as already expired (clock skew?); "
                "it will be refetched on every dial until this clears"
            )
            return now  # immediately stale: never cached as valid
        if exp - now > 24 * 3600.0:
            logger.warning(
                f"WebSocket token exp claim is implausibly far ahead "
                f"({exp - now:.0f}s); assuming the default lifetime instead"
            )
            return fallback
        return exp

    async def _get_token_url(self) -> str:
        """
        Return a valid token URL, exchanging the API key when needed.

        Raises:
            AuthenticationError: The key was rejected (401/403). Fatal - not
                something a retry can fix.
            ConnectionError: The exchange failed for transport reasons.
        """
        now = time.time()
        async with self._token_lock:
            if (
                self._token_url
                and now < self._token_expires_at - TOKEN_REFRESH_BUFFER_SECONDS
            ):
                return self._token_url

            http_url = f"{self._http_base_url()}/websocket"
            logger.debug(f"Exchanging API key for WebSocket token at {http_url}")

            try:
                async with self._session.get(
                    http_url,
                    headers={self._api_key_header: self._api_key},
                    params={"engine": "voxist-rt"},
                    ssl=self._ssl_param(),
                    timeout=aiohttp.ClientTimeout(total=10),
                ) as resp:
                    if resp.status in (401, 403):
                        raise AuthenticationError(
                            "Invalid API key. Check VOXIST_API_KEY environment "
                            "variable."
                        )
                    if resp.status != 200:
                        text = await resp.text()
                        logger.error(
                            f"Token exchange failed: {resp.status} - {text[:200]}"
                        )
                        raise ConnectionError(
                            f"Token exchange failed with status {resp.status}"
                        )
                    data = await resp.json()
            except aiohttp.ClientError as e:
                raise ConnectionError(f"Token exchange failed: {e}") from e

            token_url = data.get("url")
            if not token_url:
                raise ConnectionError("Token exchange response missing 'url' field")

            self._token_url = token_url
            self._token_expires_at = self._token_expiry_from_url(token_url, now)
            return token_url

    async def dial(
        self, language: str, sample_rate: int
    ) -> aiohttp.ClientWebSocketResponse:
        """
        Open a WebSocket configured for one session.

        The language rides the URL, so a per-stream language is simply the
        language of that stream's dial - nothing to renegotiate, no window
        where audio meets the wrong engine.

        Raises:
            AuthenticationError: The key (or token) was rejected.
            ConnectionError: The dial failed for transport reasons.
        """
        token_url = await self._get_token_url()

        safe_language = sanitize_url_param(language)
        safe_rate = sanitize_url_param(str(sample_rate))
        separator = "&" if "?" in token_url else "?"
        ws_url = f"{token_url}{separator}lang={safe_language}&sample_rate={safe_rate}"

        try:
            ws = await asyncio.wait_for(
                self._session.ws_connect(
                    ws_url,
                    heartbeat=self._heartbeat_interval,
                    autoping=True,
                    ssl=self._ssl_param(),
                ),
                timeout=DIAL_TIMEOUT_SECONDS,
            )
        except aiohttp.WSServerHandshakeError as e:
            if e.status in (401, 403):
                # An invalid token with a valid key means our cache went
                # stale server-side; drop it so the next attempt refetches.
                self._token_url = None
                raise AuthenticationError(
                    "WebSocket authentication failed"
                ) from e
            raise ConnectionError(f"WebSocket handshake failed: {e}") from e
        except asyncio.TimeoutError as e:
            raise ConnectionError(
                f"WebSocket dial timed out after {DIAL_TIMEOUT_SECONDS}s"
            ) from e
        except aiohttp.ClientError as e:
            raise ConnectionError(f"WebSocket dial failed: {e}") from e

        logger.debug(f"Dialed Voxist WebSocket (lang={language})")
        return ws
