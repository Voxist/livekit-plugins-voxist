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
import hashlib
import json
import ssl
import threading
import time
import weakref
from base64 import urlsafe_b64decode
from binascii import Error as BinasciiError
from collections import deque
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

# Ceiling on dial ATTEMPTS against one gateway, restoring the bound the
# deleted connection pool used to provide (30 reconnects/minute). Overridable
# per dialer via VoxistDialer(max_dials_per_window=...).
MAX_DIALS_PER_WINDOW = 30
DIAL_RATE_LIMIT_WINDOW_SECONDS = 60.0


class _DialRateLimiter:
    """
    Sliding-window cap on dial attempts against one gateway credential.

    Why a limiter exists at all: without the pool, every stream dials
    independently under livekit's per-stream retry, with no coordination. A
    gateway outage with N concurrent streams therefore produces N x
    max_retry dials (plus one HTTPS token exchange per 401-rejected cached
    token) in a few seconds - against a gateway that may rate-limit or ban
    the key, turning a transient outage into a much longer one.

    Why PROCESS-WIDE (see _limiter_for) rather than per dialer instance: the
    resource being protected is the gateway's own rate-limit/ban state for
    one API key, and that budget does not grow just because a process
    happens to host several VoxistSTT instances. A per-dialer limit would
    silently multiply by the instance count, which is exactly the herd this
    is meant to prevent. The cost, stated plainly: a process running many
    plugins against the same gateway shares one budget, so a mass restart
    can hit the limit - and then fails as a retryable ConnectionError that
    livekit re-attempts later, which is the intended behaviour.

    Thread-safe (a plain threading.Lock, never held across an await) because
    one gateway's limiter can legitimately be shared by dialers living on
    different event loops in different threads.
    """

    def __init__(self, *, max_dials: int, window: float) -> None:
        self._max_dials = max_dials
        self._window = window
        self._attempts: deque[float] = deque()
        self._lock = threading.Lock()
        self._last_warned_at: float | None = None

    @property
    def max_dials(self) -> int:
        return self._max_dials

    @property
    def window(self) -> float:
        return self._window

    def try_acquire(self, now: float) -> bool:
        """Charge one dial attempt, or return False when the window is full."""
        with self._lock:
            cutoff = now - self._window
            while self._attempts and self._attempts[0] <= cutoff:
                self._attempts.popleft()

            if len(self._attempts) >= self._max_dials:
                # Throttled to one line per window: a herd would otherwise
                # log once per rejected dial.
                if (
                    self._last_warned_at is None
                    or now - self._last_warned_at >= self._window
                ):
                    self._last_warned_at = now
                    logger.warning(
                        f"Dial rate limit reached: {self._max_dials} dial "
                        f"attempts in {self._window:.0f}s; further dials fail "
                        "fast (retryable) until the window clears"
                    )
                return False

            self._attempts.append(now)
            return True


# Process-wide registry, keyed by gateway URL + a fingerprint of the API key
# (never the key itself). Values are weak: the limiter lives exactly as long
# as some VoxistDialer holds it, so the registry cannot grow without bound
# and a fully torn-down deployment leaves no state behind.
_dial_limiters: weakref.WeakValueDictionary[str, _DialRateLimiter] = (
    weakref.WeakValueDictionary()
)
_dial_limiters_lock = threading.Lock()


def _limiter_for(
    *, base_url: str, api_key: str, max_dials: int, window: float
) -> _DialRateLimiter:
    key = (
        f"{base_url}|{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        f"|{max_dials}/{window}"
    )
    with _dial_limiters_lock:
        limiter = _dial_limiters.get(key)
        if limiter is None:
            limiter = _DialRateLimiter(max_dials=max_dials, window=window)
            _dial_limiters[key] = limiter
        return limiter


def reset_dial_rate_limits() -> None:
    """Forget every shared dial limiter (test seam; not part of the plugin
    lifecycle - dialers built afterwards get fresh windows)."""
    with _dial_limiters_lock:
        _dial_limiters.clear()


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
        max_dials_per_window: int = MAX_DIALS_PER_WINDOW,
        dial_rate_limit_window: float = DIAL_RATE_LIMIT_WINDOW_SECONDS,
    ) -> None:
        self._session = session
        self._base_url = base_url
        self._api_key = api_key
        self._api_key_header = api_key_header
        self._heartbeat_interval = heartbeat_interval
        self._connection_timeout = connection_timeout

        # Shared with every other dialer aimed at the same gateway with the
        # same credential (see _DialRateLimiter for the scope rationale).
        # Strong reference: the limiter's lifetime is this dialer's.
        self._rate_limiter = _limiter_for(
            base_url=base_url,
            api_key=api_key,
            max_dials=max_dials_per_window,
            window=dial_rate_limit_window,
        )

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
        async with self._token_lock:
            # The clock is read INSIDE the lock: waiting for a contended lock
            # (another stream mid-exchange) can take seconds, and a timestamp
            # captured before the wait would judge freshness - and record the
            # cache expiry - against a time that is already stale.
            now = time.time()
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
            # Anything escaping unmapped kills the stream with zero retries,
            # so a final catch-all maps whatever the enumeration missed (a
            # UnicodeDecodeError from body decoding, say). Our own
            # AuthenticationError/ConnectionError are re-raised first so they
            # are never double-wrapped, and asyncio.CancelledError derives
            # from BaseException (Python 3.8+), so no `except Exception`
            # clause here can ever swallow a cancellation.
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
            except (AuthenticationError, ConnectionError):
                raise
            except (
                aiohttp.ClientError,
                asyncio.TimeoutError,
                TimeoutError,
                json.JSONDecodeError,
                RuntimeError,
            ) as e:
                raise ConnectionError(f"Token exchange failed: {e!r}") from e
            except Exception as e:
                raise ConnectionError(
                    f"Token exchange failed unexpectedly: {e!r}"
                ) from e

            if not isinstance(data, dict):
                raise ConnectionError(
                    "Token exchange returned a non-object JSON response"
                )
            token_url = data.get("url")
            if not token_url:
                raise ConnectionError("Token exchange response missing 'url' field")
            # Validate at the exchange, where the failure is attributable:
            # a non-string (or non-WebSocket) url would otherwise crash
            # dial() later with an unmapped TypeError and zero retries.
            if not isinstance(token_url, str) or not token_url.startswith(
                ("ws://", "wss://")
            ):
                raise ConnectionError(
                    "Token exchange returned a malformed url: expected a "
                    f"ws:// or wss:// string, got {token_url!r:.100}"
                )

            self._token_url = token_url
            # Re-read the clock: `now` predates the HTTPS round-trip, which
            # can itself take seconds (up to connection_timeout). The expiry
            # sanity-clamps in _token_expiry_from_url compare the server's exp
            # claim against local time, so they must use the time the token
            # actually arrived.
            self._token_expires_at = self._token_expiry_from_url(
                token_url, time.time()
            )
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

    def _charge_dial_attempt(self) -> None:
        """
        Charge one dial attempt against the shared window, or fail retryably.

        Called once per gateway dial ATTEMPT - before the token exchange of
        the first attempt, and again before the one post-401 redial - so the
        extra HTTPS token exchange a rejected cached token triggers is
        bounded by the same budget as the sockets themselves.

        Raises:
            ConnectionError: The window is full. Deliberately our retryable
                mapping: the stream turns it into APIConnectionError, so
                livekit re-attempts later on its own backoff instead of the
                caller blocking here (no sleeps on this path).
        """
        if not self._rate_limiter.try_acquire(time.monotonic()):
            raise ConnectionError(
                "Dial rate limit reached for this gateway: "
                f"{self._rate_limiter.max_dials} dial attempts per "
                f"{self._rate_limiter.window:.0f}s (shared process-wide per "
                "gateway credential). Refusing to dial so the gateway is not "
                "hammered; this is retryable and will clear as the window "
                "slides."
            )

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
            ConnectionError: The dial failed for transport reasons, or the
                shared dial rate limit is exhausted (see
                _charge_dial_attempt).
        """
        self._charge_dial_attempt()
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
                # (expiry edge, gateway restart, token-store propagation
                # lag). Killing the stream with a fatal AuthenticationError
                # on a rejected token would punish a perfectly valid key.
                # So: invalidate the cache (race-safely - see
                # _invalidate_token), exchange the key for a fresh token
                # ONCE, and redial - bounded by this attempt counter, never
                # by comparing tokens. JWTs have one-second iat/exp
                # granularity, so a refetch in the same second (or a server
                # reusing tokens within validity) legitimately hands back a
                # byte-identical token that is still worth the one redial.
                # A genuinely revoked key still fails fast: the token
                # EXCHANGE itself 401s inside _get_token_url below and
                # raises AuthenticationError directly.
                if not refetched_token:
                    refetched_token = True
                    self._charge_dial_attempt()
                    await self._invalidate_token(token_url)
                    token_url = await self._get_token_url()
                    logger.info(
                        "WebSocket handshake rejected the token; redialing "
                        "once with a freshly exchanged token"
                    )
                    continue
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
            except Exception as e:
                # Catch-all for anything the enumeration above missed
                # (e.g. a UnicodeDecodeError surfacing from the handshake):
                # an unmapped exception kills the stream with zero retries.
                # asyncio.CancelledError derives from BaseException
                # (Python 3.8+), so cancellation is never swallowed here.
                #
                # No `except (AuthenticationError, ConnectionError): raise`
                # double-wrap guard: this try covers ONLY ws_connect, which
                # raises neither. The token exchange and the rate-limit
                # charge - the two things that DO raise our own types - are
                # outside it (the refetch inside the 401 handler raises out
                # of the handler, which sibling except clauses never see).
                # Widening this try would require reinstating that guard
                # ABOVE this clause.
                raise ConnectionError(
                    f"WebSocket dial failed unexpectedly: {e!r}"
                ) from e

            logger.debug(f"Dialed Voxist WebSocket (lang={language})")
            return ws
