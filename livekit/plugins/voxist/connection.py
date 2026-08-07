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

# Default bound for the token exchange and the WebSocket dial. Overridable
# per plugin instance via VoxistSTT(connection_timeout=...), which threads
# through to VoxistDialer(connection_timeout=...).
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
        connection_timeout: float = DIAL_TIMEOUT_SECONDS,
    ) -> None:
        self._session = session
        self._base_url = base_url
        self._api_key = api_key
        self._api_key_header = api_key_header
        self._heartbeat_interval = heartbeat_interval
        self._connection_timeout = connection_timeout

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

            if self._session.closed:
                # A retry racing shutdown must die as a mapped, retryable
                # error, not as aiohttp's raw RuntimeError('Session is
                # closed') - livekit only retries APIError subclasses, and
                # the stream maps ConnectionError into one.
                raise ConnectionError(
                    "Token exchange impossible: the HTTP session is closed"
                )

            http_url = f"{self._http_base_url()}/websocket"
            logger.debug(f"Exchanging API key for WebSocket token at {http_url}")

            # Every transport-shaped failure must map to our ConnectionError:
            # - aiohttp.ClientError: connection refused/reset, bad status
            #   handling, ContentTypeError from resp.json() on a non-JSON body
            # - asyncio.TimeoutError / TimeoutError: the ClientTimeout firing
            #   (distinct classes on 3.10, unified on 3.11+)
            # - json.JSONDecodeError: a 200 with a JSON content type but an
            #   unparseable body
            # - RuntimeError: the session closing between the check above and
            #   the request (shutdown race)
            # Anything escaping unmapped kills the stream with zero retries.
            try:
                async with self._session.get(
                    http_url,
                    headers={self._api_key_header: self._api_key},
                    params={"engine": "voxist-rt"},
                    ssl=self._ssl_param(),
                    timeout=aiohttp.ClientTimeout(total=self._connection_timeout),
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
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as e:
                raise ConnectionError(f"Token exchange failed: {e!r}") from e

            if not isinstance(data, dict):
                raise ConnectionError(
                    "Token exchange returned a non-object JSON response"
                )
            token_url = data.get("url")
            if not token_url:
                raise ConnectionError("Token exchange response missing 'url' field")

            self._token_url = token_url
            self._token_expires_at = self._token_expiry_from_url(token_url, now)
            return token_url

    async def _invalidate_token(self, rejected_url: str) -> None:
        """
        Drop the cached token, but only if it is still the rejected one.

        Taken under _token_lock so a concurrent dial that already refetched a
        fresh token is not clobbered: if the cache no longer holds the URL we
        were rejected with, another stream beat us to the refetch and its
        token must be preserved.
        """
        async with self._token_lock:
            if self._token_url == rejected_url:
                self._token_url = None
                self._token_expires_at = 0.0

    async def dial(
        self, language: str, sample_rate: int
    ) -> aiohttp.ClientWebSocketResponse:
        """
        Open a WebSocket configured for one session.

        The language rides the URL, so a per-stream language is simply the
        language of that stream's dial - nothing to renegotiate, no window
        where audio meets the wrong engine.

        Raises:
            AuthenticationError: The key (or token) was rejected even with a
                freshly exchanged token.
            ConnectionError: The dial failed for transport reasons.
        """
        token_url = await self._get_token_url()
        refetched_token = False

        while True:
            safe_language = sanitize_url_param(language)
            safe_rate = sanitize_url_param(str(sample_rate))
            separator = "&" if "?" in token_url else "?"
            ws_url = (
                f"{token_url}{separator}lang={safe_language}"
                f"&sample_rate={safe_rate}"
            )

            if self._session.closed:
                # Same shutdown race as in the token exchange: surface a
                # mapped, retryable error instead of a raw RuntimeError.
                raise ConnectionError(
                    "WebSocket dial impossible: the HTTP session is closed"
                )

            try:
                ws = await asyncio.wait_for(
                    self._session.ws_connect(
                        ws_url,
                        heartbeat=self._heartbeat_interval,
                        autoping=True,
                        ssl=self._ssl_param(),
                    ),
                    timeout=self._connection_timeout,
                )
            except aiohttp.WSServerHandshakeError as e:
                if e.status not in (401, 403):
                    raise ConnectionError(
                        f"WebSocket handshake failed: {e}"
                    ) from e

                # A 401/403 here does NOT prove the API key is bad: this
                # token may have been cached and invalidated server-side
                # (expiry edge, gateway restart). Killing the stream with a
                # fatal AuthenticationError on a stale CACHED token would
                # punish a perfectly valid key. So: invalidate the cache
                # (race-safely - see _invalidate_token), exchange the key for
                # a fresh token ONCE, and redial. Only when the fresh token
                # is also rejected is the credential itself the problem.
                if not refetched_token:
                    refetched_token = True
                    await self._invalidate_token(token_url)
                    fresh_url = await self._get_token_url()
                    if fresh_url != token_url:
                        logger.info(
                            "WebSocket handshake rejected the cached token; "
                            "redialing once with a freshly exchanged token"
                        )
                        token_url = fresh_url
                        continue
                    # The exchange handed back the identical token; redialing
                    # with the same credential cannot end differently.
                raise AuthenticationError(
                    "WebSocket authentication failed with a freshly "
                    "exchanged token"
                ) from e
            except asyncio.TimeoutError as e:
                raise ConnectionError(
                    f"WebSocket dial timed out after {self._connection_timeout}s"
                ) from e
            except aiohttp.ClientError as e:
                raise ConnectionError(f"WebSocket dial failed: {e}") from e
            except RuntimeError as e:
                # aiohttp raises RuntimeError('Session is closed') when a
                # dial races aclose(); map it so livekit can retry cleanly.
                raise ConnectionError(f"WebSocket dial failed: {e}") from e

            logger.debug(f"Dialed Voxist WebSocket (lang={language})")
            return ws
