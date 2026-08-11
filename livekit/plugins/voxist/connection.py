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

# Floor on how long a throttled caller sleeps between re-checks. The computed
# wait is the exact moment the earliest charged attempt ages out, so a caller
# that wakes a hair early (monotonic granularity) would otherwise recompute a
# microsecond-long wait and spin. Doubles as the tolerance on the caller's
# wait bound, so a slot that is microseconds beyond it is still taken.
_SLOT_POLL_FLOOR_SECONDS = 0.005

# How long a limiter must sit with an empty window before the process-wide
# registry reclaims its entry (on top of the window itself). Purely a
# memory bound: by the time it applies, the limiter's window is already empty,
# so dropping it cannot hand anyone a budget they had not already earned back.
_DIAL_LIMITER_IDLE_GRACE_SECONDS = 60.0


class _DialRateLimitExhausted(Exception):
    """
    Internal signal: no dial slot came free inside the caller's wait bound.

    Never escapes VoxistDialer - _charge_dial_attempt maps it to our
    retryable ConnectionError. It exists only so the limiter can report how
    long the caller was parked without importing the plugin's exceptions.
    """

    def __init__(self, waited: float) -> None:
        super().__init__(f"no dial slot came free within {waited:.1f}s")
        self.waited = waited


class _DialRateLimiter:
    """
    Sliding-window cap on dial attempts against one gateway credential.

    Why a limiter exists at all: without the pool, every stream dials
    independently under livekit's per-stream retry, with no coordination. A
    gateway outage with N concurrent streams therefore produces N x
    max_retry dials in a few seconds, each one an HTTPS token exchange (a JWT
    signing plus a user lookup, on an endpoint the server marks
    @SkipThrottle() so nothing upstream bounds it) followed by a WebSocket
    upgrade against a proxy pod that admits WS_HARD_LIMIT (default 40)
    concurrent sockets and closes the surplus with 1013 "retry in a few
    seconds" (simple-websocket-proxy.gateway.ts:1038). The gateway implements
    no per-key rate limit or ban of its own, so nothing but this class stops a
    herd from piling onto a pod that is already telling us to back off.

    Why PROCESS-WIDE (see _limiter_for) rather than per dialer instance: the
    resource being protected is one pod's admission budget for one API key,
    and that budget does not grow just because a process happens to host
    several VoxistSTT instances. A per-dialer limit would silently multiply by
    the instance count, which is exactly the herd this is meant to prevent.

    Why an exhausted window WAITS rather than refusing - the defect this
    replaced: acquire() used to be a synchronous try_acquire() that returned
    False, and _charge_dial_attempt turned that into a ConnectionError. That
    made our own throttle spend livekit's finite retry budget. livekit's
    RecognizeStream._main_task allows conn_options.max_retry + 1 attempts
    (4 by default) at ~retry_interval (default 2s) apart, and only ever resets
    _num_retries when a FINAL_TRANSCRIPT arrives (stt.py:531-533) - which,
    during an outage, never happens. So all four attempts landed inside the
    same 60s window, every one of them refused by us rather than by the
    gateway, and the stream died permanently with "failed to recognize speech
    after 3 attempts" even though the gateway recovered seconds later. The
    deleted connection pool got this right: it slept out the window
    (max 5 concurrent reconnects) and reconnected. Waiting restores that.

    The wait is bounded by the CALLER (acquire(max_wait=...)), not by the
    limiter, because the tolerance for being parked belongs to the caller;
    VoxistDialer defaults it to the window, the largest value the wait can
    ever need (see VoxistDialer.__init__).

    Thread-safe, and the threading.Lock is taken only for bookkeeping inside
    the synchronous helpers - never held across the await in acquire(), which
    would stall every event loop sharing this limiter. One gateway's limiter
    can legitimately be shared by dialers living on different event loops in
    different threads.
    """

    def __init__(self, *, max_dials: int, window: float) -> None:
        self._max_dials = max_dials
        self._window = window
        self._attempts: deque[float] = deque()
        self._lock = threading.Lock()
        # Most recent charge - or construction, for a limiter nobody has
        # dialled through yet - used by is_idle() for registry eviction.
        # Seeded from the clock rather than left as a "never charged" sentinel
        # so a freshly minted limiter is never swept out from under the dialer
        # that just resolved it: two dialers must not end up charging two
        # different limiters for one credential.
        self._last_activity_at = time.monotonic()
        # Per-kind timestamps for the once-per-window log throttle; a herd
        # would otherwise log once per throttled dial.
        self._warned_at: dict[str, float] = {}

    @property
    def max_dials(self) -> int:
        return self._max_dials

    @property
    def window(self) -> float:
        return self._window

    @property
    def charged_attempts(self) -> int:
        """Charges currently on record (without pruning). Diagnostics only."""
        with self._lock:
            return len(self._attempts)

    def reserve(self, now: float) -> float:
        """
        Charge one dial attempt and return 0.0, or return the seconds until
        the earliest charged attempt ages out of the window.

        Synchronous and non-blocking on purpose: the lock is taken for the
        bookkeeping alone and is released before acquire() awaits.
        """
        with self._lock:
            cutoff = now - self._window
            while self._attempts and self._attempts[0] <= cutoff:
                self._attempts.popleft()

            if len(self._attempts) < self._max_dials:
                self._attempts.append(now)
                self._last_activity_at = now
                return 0.0

            # The earliest recorded attempt is the earliest moment a slot can
            # possibly free, so this is a floor on the wait - never a guess.
            return max(self._attempts[0] + self._window - now, 0.0)

    def is_idle(self, now: float, grace: float) -> bool:
        """
        True when the window holds no live attempt AND the last activity
        (charge, or construction) is older than window + grace, so the entry
        is safe to reclaim.
        """
        with self._lock:
            cutoff = now - self._window
            while self._attempts and self._attempts[0] <= cutoff:
                self._attempts.popleft()
            if self._attempts:
                return False
            return now - self._last_activity_at >= self._window + grace

    def _claim_warning_slot(self, now: float, kind: str) -> bool:
        with self._lock:
            last = self._warned_at.get(kind)
            if last is not None and now - last < self._window:
                return False
            self._warned_at[kind] = now
            return True

    async def acquire(self, *, max_wait: float) -> None:
        """
        Charge one dial attempt, waiting for a slot when the window is full.

        Cancellation-correct: the charge is appended only on the iteration
        that returns, so a caller cancelled while parked leaves no phantom
        attempt behind, and the lock is always released by reserve() before
        the sleep, so a cancellation can never strand it.

        Note for tests: this reads the clock it sleeps against, so a frozen
        clock (a FakeClock injected as connection.time) would make the wait
        loop unable to make progress. Drive this with the real clock and a
        small window instead.

        Raises:
            _DialRateLimitExhausted: no slot came free within max_wait.
        """
        started_at = time.monotonic()
        while True:
            now = time.monotonic()
            wait_for = self.reserve(now)
            if wait_for == 0.0:
                return

            waited = now - started_at
            remaining = max_wait - waited
            # The bound is honoured to within the poll floor: an event loop
            # timer may fire a hair early (loop clock resolution), and giving
            # up on a slot that is microseconds away would be absurd.
            if wait_for > remaining + _SLOT_POLL_FLOOR_SECONDS:
                # The earliest slot is beyond what this caller will tolerate,
                # and reserve() reports the exact moment it frees - so no
                # amount of further sleeping changes that. A caller whose
                # bound is the full window (VoxistDialer's default) can never
                # reach this branch on the first pass, because a slot always
                # frees strictly within one window; it is reached only when
                # competing waiters keep taking the slot first.
                if self._claim_warning_slot(now, "exhausted"):
                    logger.warning(
                        f"Dial rate limit: {self._max_dials} dial attempts in "
                        f"{self._window:.3g}s and no slot came free within the "
                        f"{max_wait:.3g}s wait bound; dials now fail "
                        "(retryable) until the window clears"
                    )
                raise _DialRateLimitExhausted(waited)

            if self._claim_warning_slot(now, "waiting"):
                logger.warning(
                    f"Dial rate limit reached: {self._max_dials} dial attempts "
                    f"in {self._window:.3g}s; dials now WAIT up to "
                    f"{max_wait:.3g}s for the window to slide rather than "
                    "failing and spending livekit's retry budget"
                )

            # The lock is NOT held here (reserve() released it): sleeping
            # under a threading.Lock would block every event loop that shares
            # this limiter, and a threading.Lock held across an await cannot
            # be released by the cancellation that interrupts it.
            await asyncio.sleep(max(wait_for, _SLOT_POLL_FLOOR_SECONDS))


# Process-wide registry, keyed by gateway URL + a fingerprint of the API key
# (never the key itself).
#
# Strong references, and the defect that forced them: this used to be a
# weakref.WeakValueDictionary whose only strong reference was
# VoxistDialer._rate_limiter. The "process-wide" budget therefore reset to a
# full window the moment the last dialer for a credential was collected -
# defeating the anti-herd protection in precisely the pattern it exists for.
# The common LiveKit agent shape builds one VoxistSTT per job and awaits
# aclose() in a finally, so during a gateway outage each job burned its dials,
# its teardown let the limiter be collected, and the next job started with an
# empty deque; the gateway saw the unbounded herd the class was written to
# prevent.
#
# Registry growth is bounded instead by idle eviction (see _limiter_for): an
# entry whose window has been empty for window + _DIAL_LIMITER_IDLE_GRACE
# seconds is dropped, so a process cycling through many distinct credentials
# retains only the recently-active ones, while a credential that dialled
# seconds ago keeps its charges no matter how many dialers came and went.
_dial_limiters: dict[str, _DialRateLimiter] = {}
_dial_limiters_lock = threading.Lock()


def _limiter_for(
    *, base_url: str, api_key: str, max_dials: int, window: float
) -> _DialRateLimiter:
    """Resolve the one canonical limiter for a credential, sweeping idle ones.

    Called on every charge (VoxistDialer._rate_limiter is a property, not a
    cached attribute) so there is exactly one live limiter per credential at
    any moment: a cached reference could outlive an eviction and charge a
    limiter nobody else can see, splitting one credential's budget in two.

    Lock ordering is registry lock -> limiter lock (is_idle takes the
    limiter's), and nothing ever goes the other way: a limiter never touches
    the registry, and acquire()'s wait happens after this function has
    returned and released the registry lock.
    """
    key = (
        f"{base_url}|{hashlib.sha256(api_key.encode()).hexdigest()[:16]}"
        f"|{max_dials}/{window}"
    )
    now = time.monotonic()
    with _dial_limiters_lock:
        stale = [
            k
            for k, limiter in _dial_limiters.items()
            if limiter.is_idle(now, _DIAL_LIMITER_IDLE_GRACE_SECONDS)
        ]
        for k in stale:
            del _dial_limiters[k]

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
        max_dial_rate_limit_wait: float | None = None,
    ) -> None:
        self._session = session
        self._base_url = base_url
        self._api_key = api_key
        self._api_key_header = api_key_header
        self._heartbeat_interval = heartbeat_interval
        self._connection_timeout = connection_timeout

        # Parameters of the limiter shared with every other dialer aimed at the
        # same gateway with the same credential (see _DialRateLimiter for the
        # scope rationale). Deliberately NOT a cached limiter reference - see
        # the _rate_limiter property.
        self._max_dials_per_window = max_dials_per_window
        self._dial_rate_limit_window = dial_rate_limit_window

        # How long a dial will wait for a free slot before failing retryably.
        # Defaults to the window because that is the largest wait the limiter
        # can ever ask for (a slot frees at most `window` after it was
        # charged), so it is the smallest bound that still guarantees a caller
        # which is merely EARLY - rather than starved by other waiters - gets
        # its slot instead of dying. Anything shorter reintroduces the defect
        # documented on _DialRateLimiter: our own throttle killing a stream
        # that the gateway would have served moments later.
        self._max_dial_rate_limit_wait = (
            dial_rate_limit_window
            if max_dial_rate_limit_wait is None
            else max_dial_rate_limit_wait
        )

        # Certificate verification is never disabled. An explicit context is
        # the only way to trust a private CA; without one, aiohttp's default
        # verification applies (ssl=True) - even when base_url is ws://,
        # because the token exchange may hand back a wss:// URL.
        self._ssl_context = ssl_context

        self._token_url: str | None = None
        self._token_expires_at = 0.0
        self._token_lock = asyncio.Lock()

    @property
    def _rate_limiter(self) -> _DialRateLimiter:
        """
        The canonical limiter for this dialer's credential, resolved on every
        use rather than cached at construction.

        Why not a cached attribute: the registry evicts limiters whose window
        has been idle (see _limiter_for), and a cached strong reference would
        let this dialer go on charging an EVICTED limiter while a freshly built
        dialer for the same credential charges its replacement - two live
        budgets for one credential, which is the very doubling the
        process-wide scope exists to prevent. Resolving per use costs one dict
        lookup per dial, which is nothing next to an HTTPS round-trip.
        """
        return _limiter_for(
            base_url=self._base_url,
            api_key=self._api_key,
            max_dials=self._max_dials_per_window,
            window=self._dial_rate_limit_window,
        )

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

    async def _charge_dial_attempt(self) -> None:
        """
        Charge one dial attempt against the shared window, WAITING for a free
        slot when it is full and failing retryably only if none comes.

        Called once per gateway dial ATTEMPT - before the token exchange of
        the first attempt, and again before the one post-401 redial - so the
        extra HTTPS token exchange a rejected cached token triggers is
        bounded by the same budget as the sockets themselves. That is two
        charges for one dial() call on the stale-token path, and it is not a
        double-charge: each covers a distinct token exchange plus a distinct
        WebSocket upgrade actually put on the wire. The ordinary path charges
        exactly once, so the effective ceiling is the configured one.

        Why waiting rather than the refusal this replaced: see the defect
        recorded on _DialRateLimiter. Delaying a dial costs the stream time;
        refusing one costs it an irreplaceable livekit retry.

        Raises:
            ConnectionError: No slot came free inside the wait bound. Our
                retryable mapping: the stream turns it into
                APIConnectionError, so livekit re-attempts later on its own
                backoff.
        """
        limiter = self._rate_limiter
        try:
            await limiter.acquire(max_wait=self._max_dial_rate_limit_wait)
        except _DialRateLimitExhausted as e:
            raise ConnectionError(
                "Dial rate limit reached for this gateway: "
                f"{limiter.max_dials} dial attempts per "
                f"{limiter.window:.3g}s (shared process-wide per gateway "
                f"credential). Waited {e.waited:.2f}s of the "
                f"{self._max_dial_rate_limit_wait:.3g}s bound for a free slot "
                "and gave up rather than hammering the gateway; this is "
                "retryable and will clear as the window slides."
            ) from e

    async def dial(
        self,
        language: str,
        sample_rate: int,
        punctuation_mode: str | None = None,
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
        await self._charge_dial_attempt()
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
            # The gateway reads punctuation_mode from the connect URL
            # alongside lang and sample_rate. Verified live against
            # api-asr.voxist.com: with 'Dictated' the same audio comes back
            # without automatic commas or sentence periods, which is the point
            # - a dictation user speaks their punctuation.
            #
            # Sent via the URL rather than a {"config": {...}} message on
            # purpose. That message can also carry sample_rate, and the
            # gateway BILLS on the last rate it was told
            # (durationSecs = bytes / (rate * 2)), so a config message that
            # echoed the caller's 48000 while the wire carries 16kHz would
            # under-report usage threefold. The URL path cannot make that
            # mistake.
            if punctuation_mode:
                ws_url += (
                    f"&punctuation_mode={sanitize_url_param(punctuation_mode)}"
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
                    await self._charge_dial_attempt()
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
