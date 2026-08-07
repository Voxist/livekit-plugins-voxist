"""ConnectionPool - Persistent WebSocket connection management."""

from __future__ import annotations

import asyncio
import json
import random
import ssl
import time
from base64 import urlsafe_b64decode
from binascii import Error as BinasciiError
from typing import cast
from urllib.parse import parse_qs, urlsplit

import aiohttp

from .exceptions import (
    AuthenticationError,
    ConnectionError,
    ConnectionPoolExhaustedError,
    LanguageNotSupportedError,
)
from .log import logger
from .models import Connection, ConnectionState, sanitize_url_param, validate_language_format

# Token refresh buffer - refresh 5 minutes before expiry
TOKEN_REFRESH_BUFFER_SECONDS = 300


class ConnectionPool:
    """
    Manages a pool of persistent WebSocket connections to Voxist API.

    Features:
    - Pre-warming for zero cold-start latency
    - Health monitoring with heartbeat (30s ping/pong)
    - Automatic reconnection with exponential backoff
    - Round-robin load balancing with backpressure awareness
    - Connection state tracking and management

    Performance:
    - Connection acquisition: < 5ms (from ready pool)
    - Recovery time: < 2s (exponential backoff)
    - Zero cold-start: Pre-warmed on initialization

    Example:
        pool = ConnectionPool(
            base_url="wss://api-asr.voxist.com/ws",
            api_key="voxist_...",
            pool_size=2,
        )

        await pool.initialize()

        # Get connection
        conn = await pool.get_connection()

        # Use connection
        await conn.ws.send_json({"config": {"lang": "fr"}})

        # Release back to pool
        await pool.release_connection(conn)

        # Cleanup
        await pool.close()
    """

    def __init__(
        self,
        base_url: str,
        api_key: str,
        pool_size: int = 2,
        connection_timeout: float = 10.0,
        heartbeat_interval: float = 30.0,
        max_reconnect_attempts: int = 10,
        ssl_context: ssl.SSLContext | None = None,
        language: str = "fr",
        sample_rate: int = 16000,
        api_key_header: str = "X-LVL-KEY",
    ):
        """
        Initialize connection pool.

        Args:
            base_url: WebSocket URL (e.g., "wss://api-asr.voxist.com/ws")
            api_key: Voxist API key for authentication
            pool_size: Number of connections to maintain (2-3 recommended)
            connection_timeout: Timeout for initial connection in seconds
            heartbeat_interval: Interval between heartbeat pings in seconds
            max_reconnect_attempts: Max reconnection attempts per connection
            ssl_context: Optional SSL context for TLS configuration.
                         If None, a secure default context with certificate
                         validation is created automatically for wss:// URLs.
            language: Language code for ASR (e.g., "fr", "en", "fr-medical")
            sample_rate: Audio sample rate in Hz (default: 16000)
            api_key_header: HTTP header name for API key (default: X-LVL-KEY)
        """
        self.base_url = base_url
        self.api_key = api_key
        self.pool_size = pool_size
        self.connection_timeout = connection_timeout
        self.heartbeat_interval = heartbeat_interval
        self.max_reconnect_attempts = max_reconnect_attempts
        self.language = language
        self.sample_rate = sample_rate
        self.api_key_header = api_key_header

        # Create secure SSL context for wss:// connections
        self._ssl_context: ssl.SSLContext | None = None
        if ssl_context is not None:
            self._ssl_context = ssl_context
        elif base_url.startswith("wss://"):
            self._ssl_context = ssl.create_default_context()
            self._ssl_context.check_hostname = True
            self._ssl_context.verify_mode = ssl.CERT_REQUIRED

        self.connections: list[Connection] = []
        self.current_index = 0
        self._lock = asyncio.Lock()
        # Strong references to in-flight reconnect tasks (see _spawn_reconnect)
        self._reconnect_tasks: set[asyncio.Task] = set()
        self._initialized = False
        self._heartbeat_task: asyncio.Task | None = None
        self._session: aiohttp.ClientSession | None = None
        self._closing = False

        # Token exchange caching (SEC-001 fix)
        # API key is exchanged for short-lived JWT via HTTPS, not sent in WebSocket URL
        self._ws_token_url: str | None = None
        self._token_expires_at: float = 0
        self._token_lock = asyncio.Lock()

        # SEC-012 FIX: Rate limiting for reconnection attempts
        # Prevents resource exhaustion attacks and API bans from rapid reconnection cycles
        # CWE-770: Allocation of Resources Without Limits or Throttling
        self._reconnect_semaphore = asyncio.Semaphore(5)  # Max 5 concurrent reconnects
        self._reconnect_times: list[float] = []  # Timestamps of recent reconnects
        self._max_reconnects_per_minute = 30  # Global rate limit

        logger.debug(
            f"ConnectionPool created: pool_size={pool_size}, "
            f"base_url={base_url}"
        )

    @property
    def is_initialized(self) -> bool:
        """Check if the connection pool has been initialized."""
        return self._initialized

    async def mark_broken(self, conn: Connection) -> None:
        """
        Retire a connection a stream found unusable, and reconnect it.

        Called when a send or receive stalled: the socket is not closed, so
        nothing else can tell it apart from a healthy one, and returning it to
        the pool would hand the stall to the next stream.

        The stream must not set ConnectionState itself. Doing so made
        release_connection() skip its IN_USE branch entirely, so no reconnect
        was ever scheduled and recovery waited on the next heartbeat tick.

        Args:
            conn: The connection to retire
        """
        async with self._lock:
            if conn.state == ConnectionState.CLOSED:
                return  # already retired
            conn.state = ConnectionState.CLOSED
            conn.buffered_amount = 0
            logger.warning(f"Connection {conn.id} marked broken, will reconnect")

        self._spawn_reconnect(conn)

    def _spawn_reconnect(
        self, conn: Connection, state_already_set: bool = False
    ) -> None:
        """
        Schedule a reconnect, unless the pool is shutting down.

        Guarded on _closing: without it, a connection released during aclose()
        starts a reconnect chain against an already-closed ClientSession, which
        retries with backoff for minutes after shutdown returned. The task
        reference is held so it is not garbage collected mid-flight.
        """
        if self._closing:
            logger.debug(
                f"Connection {conn.id} not reconnecting - pool is closing"
            )
            return

        task = asyncio.create_task(
            self._reconnect(conn, state_already_set=state_already_set)
        )
        self._reconnect_tasks.add(task)
        task.add_done_callback(self._reconnect_tasks.discard)

    def _usable_ready(self, reconnect_dead: bool) -> list[Connection]:
        """
        READY connections whose WebSocket is actually open.

        Filtering on state alone was not enough: a pooled socket closed by the
        server (or by aiohttp after a missed pong) stays READY until the
        heartbeat loop notices, and handing one out produced a session that sent
        every chunk into a closed socket, received nothing, and completed with no
        transcripts, no exception and no reconnect.

        Caller must hold self._lock.

        Args:
            reconnect_dead: Retire and reconnect the dead ones found. False when
                called from a path that must not mutate state.

        Returns:
            The usable READY connections, in pool order.
        """
        usable = []
        for conn in self.connections:
            if conn.state != ConnectionState.READY:
                continue
            if conn.ws is None or conn.ws.closed:
                if reconnect_dead:
                    logger.warning(
                        f"Connection {conn.id} was READY with a closed socket, "
                        "retiring it instead of handing it out"
                    )
                    conn.state = ConnectionState.CLOSED
                    self._spawn_reconnect(conn)
                continue
            usable.append(conn)
        return usable

    def _select_round_robin(self, ready_conns: list[Connection]) -> Connection:
        """
        Pick the next ready connection, rotating across the pool.

        Selection used to be min(buffered_amount), which only rotated because
        that field accumulated every byte ever sent - it ranked connections by
        lifetime usage, not current load. Now that buffered_amount is a
        post-send measurement (~0 on every healthy connection, and 0 whenever
        the transport cannot be measured), min() would tie on every call and
        always return connections[0], so `current_index` - present but never
        read - does the rotation explicitly.

        Caller must hold self._lock.

        Args:
            ready_conns: Non-empty list of READY connections

        Returns:
            The selected connection.
        """
        ready = {id(c) for c in ready_conns}
        total = len(self.connections)

        # Walk the pool from just past the cursor and take the first ready
        # connection. Indexing the filtered list by a persistent counter would
        # be unfair when the ready subset changes size between calls.
        for offset in range(1, total + 1):
            index = (self.current_index + offset) % total
            candidate = self.connections[index]
            if id(candidate) in ready:
                self.current_index = index
                return candidate

        return ready_conns[0]  # not reachable while ready_conns is non-empty

    def _ssl_param(self) -> ssl.SSLContext | bool:
        """
        SSL argument for aiohttp calls, shared by every request this pool makes.

        Single source of truth on purpose: the token exchange and the WebSocket
        connect previously computed this separately and disagreed, so one
        verified certificates while the other silently skipped verification for
        the same configuration.

        `True` means aiohttp's default verification. It is the right fallback
        even when no context was built (non-wss:// base_url), because
        _get_ws_token() returns a server-supplied URL that may itself be
        wss:// - and for a genuinely plaintext URL the argument is ignored.
        """
        if self._ssl_context is not None:
            return self._ssl_context
        return True

    async def initialize(self) -> None:
        """
        Pre-warm all connections in the pool.

        Establishes connections in parallel and waits for at least one
        successful connection before returning.

        Raises:
            ConnectionError: If no connections could be established
            AuthenticationError: If API key is invalid
        """
        if self._initialized:
            logger.debug("Pool already initialized")
            return

        logger.info(f"Initializing connection pool with {self.pool_size} connections")

        # Create shared aiohttp session
        self._session = aiohttp.ClientSession()

        # Create all connection objects
        for i in range(self.pool_size):
            conn = Connection(id=i)
            self.connections.append(conn)

        # Connect all in parallel
        connect_tasks = [
            asyncio.create_task(self._connect(conn, self.language, self.sample_rate))
            for conn in self.connections
        ]

        # Wait for at least one success, but give all a chance
        done, pending = await asyncio.wait(
            connect_tasks,
            return_when=asyncio.FIRST_COMPLETED,
            timeout=self.connection_timeout
        )

        # Let remaining connections finish in background
        for _task in pending:
            # Don't cancel, let them complete
            pass

        # Check if at least one succeeded
        successful = sum(
            1 for task in done
            if not task.exception() and task.result()
        )

        if successful == 0:
            # Check for auth errors in completed tasks
            for task in done:
                if task.exception():
                    exc = task.exception()
                    if isinstance(exc, AuthenticationError):
                        raise exc

            raise ConnectionError(
                f"Failed to establish any connections. "
                f"Completed: {len(done)}/{self.pool_size}"
            )

        logger.debug(
            f"Connection pool initialized: {successful}/{self.pool_size} successful"
        )

        self._initialized = True

        # Start heartbeat monitoring in background
        self._heartbeat_task = asyncio.create_task(self._heartbeat_loop())

    def _get_http_base_url(self) -> str:
        """
        Convert WebSocket URL to HTTP URL for token exchange.

        Examples:
            wss://api-asr.voxist.com/ws -> https://api-asr.voxist.com
            ws://localhost:3000/ws -> http://localhost:3000
        """
        url = self.base_url
        # Convert protocol
        if url.startswith("wss://"):
            url = "https://" + url[6:]
        elif url.startswith("ws://"):
            url = "http://" + url[5:]

        # Remove /ws path suffix
        if url.endswith("/ws"):
            url = url[:-3]

        return url

    @staticmethod
    def _token_expiry_from_url(token_url: str, now: float) -> float:
        """
        Read the token's expiry from its JWT `exp` claim.

        The token-exchange response carries only a `url`, so the expiry has to
        come from the JWT embedded in it. Only the payload is decoded, purely to
        schedule the refresh - this is not a security check and the signature is
        deliberately not verified (the server enforces that).

        Falls back to one hour when the claim cannot be read, which is the
        current server-side lifetime.

        Args:
            token_url: WebSocket URL containing a `token` query parameter
            now: Current timestamp to base the fallback on

        Returns:
            Absolute expiry timestamp.
        """
        fallback = now + 3600.0
        try:
            query = urlsplit(token_url).query
            token = parse_qs(query).get("token", [""])[0]
            payload_segment = token.split(".")[1]
            # JWT uses base64url without padding
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
                "WebSocket token is already expired according to its exp claim"
            )
        return exp

    async def _get_ws_token(self, language: str, sample_rate: int) -> str:
        """
        Exchange API key for short-lived WebSocket token via HTTPS.

        This implements secure token exchange (SEC-001 fix):
        1. API key sent via HTTPS header (not in URL)
        2. Server returns short-lived JWT (1h expiry)
        3. JWT used in WebSocket URL instead of raw API key

        Args:
            language: Language code for ASR
            sample_rate: Audio sample rate in Hz

        Returns:
            WebSocket URL with token (e.g., wss://host/ws?token=jwt&lang=fr&sample_rate=16000)

        Raises:
            AuthenticationError: If API key is invalid
            ConnectionError: If token exchange fails
            LanguageNotSupportedError: If language format is invalid (SEC-002)
        """
        # SEC-002 FIX: Validate language format before URL construction
        if not validate_language_format(language):
            raise LanguageNotSupportedError(
                f"Language '{language}' has invalid format. "
                f"Expected: 'xx' or 'xx-YY' (e.g., 'fr', 'en-US', 'fr-medical')"
            )

        current_time = time.time()

        # SEC-002 FIX: Sanitize parameters for URL construction
        safe_language = sanitize_url_param(language)
        safe_sample_rate = sanitize_url_param(str(sample_rate))

        # Check if cached token is still valid (with buffer for safety)
        async with self._token_lock:
            if (
                self._ws_token_url
                and current_time < self._token_expires_at - TOKEN_REFRESH_BUFFER_SECONDS
            ):
                # Add language and sample_rate to cached URL (SEC-002: use sanitized params)
                cached_url = self._ws_token_url
                if "?" in cached_url:
                    return f"{cached_url}&lang={safe_language}&sample_rate={safe_sample_rate}"
                return f"{cached_url}?lang={safe_language}&sample_rate={safe_sample_rate}"

        # Token expired or not yet obtained - fetch new one
        http_url = f"{self._get_http_base_url()}/websocket"

        logger.debug(f"Exchanging API key for WebSocket token at {http_url}")

        try:
            headers = {self.api_key_header: self.api_key}
            params = {"engine": "voxist-rt"}

            assert self._session is not None  # Initialized in initialize()
            async with self._session.get(
                http_url,
                headers=headers,
                params=params,
                ssl=self._ssl_param(),
                timeout=aiohttp.ClientTimeout(total=10),
            ) as resp:
                if resp.status == 401 or resp.status == 403:
                    raise AuthenticationError(
                        "Invalid API key. Check VOXIST_API_KEY environment variable."
                    )

                if resp.status != 200:
                    error_text = await resp.text()
                    logger.error(f"Token exchange failed: {resp.status} - {error_text[:200]}")
                    raise ConnectionError(
                        f"Token exchange failed with status {resp.status}"
                    )

                data = await resp.json()
                token_url = data.get("url")

                if not token_url:
                    raise ConnectionError("Token exchange response missing 'url' field")

                # Cache the token URL (valid for 1 hour, we refresh 5 min early)
                async with self._token_lock:
                    self._ws_token_url = token_url
                    # Use the expiry the server actually issued. The JWT's own
                    # `exp` claim is authoritative; hard-coding an hour
                    # duplicates a constant that lives in the backend, so a
                    # shorter-lived token would be reused until it failed.
                    self._token_expires_at = self._token_expiry_from_url(
                        token_url, current_time
                    )

                logger.debug("WebSocket token obtained successfully")

                # Add language and sample_rate params (SEC-002: use sanitized params)
                if "?" in token_url:
                    return f"{token_url}&lang={safe_language}&sample_rate={safe_sample_rate}"
                return f"{token_url}?lang={safe_language}&sample_rate={safe_sample_rate}"

        except AuthenticationError:
            raise
        except aiohttp.ClientError as e:
            logger.error(f"Token exchange network error: {e}")
            raise ConnectionError(f"Token exchange failed: {e}") from e
        except Exception as e:
            logger.error(f"Token exchange unexpected error: {e}")
            raise ConnectionError(f"Token exchange failed: {e}") from e

    async def _connect(
        self, conn: Connection, language: str = "fr", sample_rate: int = 16000
    ) -> bool:
        """
        Establish WebSocket connection for a single Connection object.

        Args:
            conn: Connection object to establish
            language: Language code for ASR (default: fr)
            sample_rate: Audio sample rate in Hz (default: 16000)

        Returns:
            True if successful, False otherwise

        Raises:
            AuthenticationError: If authentication fails (API key invalid)
        """
        try:
            conn.state = ConnectionState.CONNECTING
            logger.debug(f"Connecting connection {conn.id}")

            # SEC-001 FIX: Use token exchange instead of raw API key in URL
            # API key is sent via HTTPS to /websocket endpoint, which returns
            # a short-lived JWT token. This prevents API key exposure in logs.
            ws_url = await self._get_ws_token(language, sample_rate)

            # Connect with timeout and SSL context
            assert self._session is not None  # Initialized in initialize()
            # Same SSL policy as every other call this pool makes. Never False:
            # _get_ws_token() may hand back a wss:// URL even when base_url was
            # not wss://, and disabling verification on it would expose the JWT
            # and all audio to a man-in-the-middle.
            ssl_param = self._ssl_param()
            # cast: aiohttp >= 3.14 parameterizes ClientWebSocketResponse over
            # its text-decoding flag, so ws_connect() returns [bool] while a
            # bare annotation resolves to the narrower [Literal[True]]. Writing
            # the parameter into Connection.ws instead would not type-check at
            # all on aiohttp 3.10-3.13, which the dependency floor allows.
            conn.ws = cast(
                "aiohttp.ClientWebSocketResponse",
                await asyncio.wait_for(
                    self._session.ws_connect(
                        ws_url,
                        heartbeat=self.heartbeat_interval,
                        autoping=True,  # Automatic ping/pong
                        ssl=ssl_param,  # Explicit SSL/TLS validation
                    ),
                    timeout=self.connection_timeout
                ),
            )

            conn.state = ConnectionState.READY
            conn.last_heartbeat = time.time()
            conn.retry_count = 0
            # Fresh WebSocket means a fresh transport, so any buffer reading
            # carried over from the previous socket is meaningless.
            conn.buffered_amount = 0

            logger.debug(f"Connection {conn.id} established successfully")
            return True

        except asyncio.TimeoutError:
            logger.error(f"Connection {conn.id} timeout after {self.connection_timeout}s")
            conn.state = ConnectionState.FAILED
            return False

        except aiohttp.WSServerHandshakeError as e:
            # WebSocket handshake failed - likely auth error
            # Log full details internally, sanitize user-facing message
            if e.status == 401 or e.status == 403:
                logger.error(f"Authentication failed for connection {conn.id}: {e}")
                conn.state = ConnectionState.FAILED
                raise AuthenticationError(
                    "Invalid API key. Check VOXIST_API_KEY environment variable."
                ) from e
            else:
                logger.error(f"WebSocket handshake failed for connection {conn.id}: {e}")
                conn.state = ConnectionState.FAILED
                return False

        except aiohttp.ClientError as e:
            logger.error(f"Connection {conn.id} client error: {e}")
            conn.state = ConnectionState.FAILED
            return False

        except Exception as e:
            logger.error(f"Connection {conn.id} unexpected error: {e}")
            conn.state = ConnectionState.FAILED
            return False

    async def get_connection(self) -> Connection:
        """
        Get a healthy connection from the pool.

        Uses round-robin selection with health checks and backpressure awareness.
        Prioritizes connections with lowest buffered data.

        Returns:
            Connection object ready for use (state will be IN_USE)

        Raises:
            ConnectionPoolExhaustedError: No healthy connections available

        Performance:
            < 5ms when connections are pre-warmed and ready.
            Uses fine-grained locking to avoid blocking during async waits.
        """
        # Phase 1: Quick check for ready connection (hold lock briefly)
        async with self._lock:
            ready_conns = self._usable_ready(reconnect_dead=True)

            if ready_conns:
                conn = self._select_round_robin(ready_conns)
                conn.state = ConnectionState.IN_USE
                logger.debug(
                    f"Acquired connection {conn.id} "
                    f"(buffered: {conn.buffered_amount} bytes)"
                )
                return conn

            # Capture connecting connection to wait for (outside lock)
            connecting = [
                c for c in self.connections
                if c.state == ConnectionState.CONNECTING
            ]
            wait_conn = connecting[0] if connecting else None

        # Phase 2: Wait for connecting connection WITHOUT holding lock
        # This allows other callers to proceed with any ready connections
        if wait_conn:
            logger.debug(
                f"No ready connections, waiting for connection {wait_conn.id}"
            )
            try:
                # Wait up to 2 seconds for connection to be ready
                await asyncio.wait_for(
                    self._wait_for_ready(wait_conn),
                    timeout=2.0
                )
                # Re-acquire lock to atomically check and claim
                # SECURITY: Re-validate state after wait to prevent TOCTOU race (VUL-001)
                # Connection state may have changed while we waited without the lock
                async with self._lock:
                    if wait_conn.state == ConnectionState.READY:
                        wait_conn.state = ConnectionState.IN_USE
                        logger.debug(f"Acquired connecting connection {wait_conn.id}")
                        return wait_conn
                    else:
                        # State changed during wait - check for any other ready connections
                        logger.debug(
                            f"Connection {wait_conn.id} state changed to {wait_conn.state.value} "
                            f"during wait, checking for alternatives"
                        )
                        ready_conns = self._usable_ready(reconnect_dead=False)
                        if ready_conns:
                            conn = self._select_round_robin(ready_conns)
                            conn.state = ConnectionState.IN_USE
                            logger.debug(f"Fallback to connection {conn.id}")
                            return conn
            except asyncio.TimeoutError:
                logger.warning(f"Connection {wait_conn.id} took too long")

        # Phase 3: Trigger reconnection for failed connections (hold lock briefly)
        # SECURITY: Atomically check state and set RECONNECTING to prevent race (VUL-002)
        async with self._lock:
            failed = [
                c for c in self.connections
                if c.state in (ConnectionState.FAILED, ConnectionState.CLOSED)
            ]

            if failed:
                conn_to_reconnect = failed[0]
                # Atomically set RECONNECTING state BEFORE spawning task
                # This prevents multiple callers from spawning duplicate reconnection tasks
                if conn_to_reconnect.state in (ConnectionState.FAILED, ConnectionState.CLOSED):
                    conn_to_reconnect.state = ConnectionState.RECONNECTING
                    logger.info(f"Triggering reconnection for connection {conn_to_reconnect.id}")
                    self._spawn_reconnect(conn_to_reconnect, state_already_set=True)

                # Don't wait for reconnection, fail fast
                # Application can retry get_connection() in a moment

            # No connections available
            pool_status = self._get_pool_status()
            logger.error(f"Connection pool exhausted: {pool_status}")

            # Log detailed status internally, but sanitize user-facing message
            raise ConnectionPoolExhaustedError(
                "No healthy connections available. "
                "Check network connectivity and Voxist API status."
            )

    async def release_connection(self, conn: Connection) -> None:
        """
        Release connection back to pool.

        Args:
            conn: Connection to release
        """
        needs_reconnect = False

        async with self._lock:
            if conn.state == ConnectionState.IN_USE:
                # Diagnostic snapshot from the departing stream - it describes a
                # transport that is no longer in use.
                conn.buffered_amount = 0

                if conn.ws is None or conn.ws.closed:
                    # Never hand a dead socket back to the pool. A stream whose
                    # connection died mid-session would otherwise return it to
                    # READY, and the next stream would send every chunk into a
                    # closed WebSocket and finish with zero transcripts, no
                    # error, and no reconnect.
                    conn.state = ConnectionState.CLOSED
                    needs_reconnect = True
                    logger.debug(
                        f"Connection {conn.id} released with a closed WebSocket, "
                        "scheduling reconnect instead of returning it to the pool"
                    )
                else:
                    conn.state = ConnectionState.READY
                    # An active stream is proof of life, but only the stream
                    # refreshes last_heartbeat. Stamp it here too so a long call
                    # cannot hand back a connection that the heartbeat loop then
                    # judges stale and tears down.
                    conn.last_heartbeat = time.time()
                    logger.debug(f"Released connection {conn.id}")

        if needs_reconnect:
            self._spawn_reconnect(conn)

    async def _wait_for_ready(self, conn: Connection) -> None:
        """
        Wait for connection to reach READY state.

        Args:
            conn: Connection to wait for

        Raises:
            ConnectionError: If connection fails
        """
        max_wait = 100  # 10 seconds (100 * 0.1s)
        waited = 0

        while conn.state != ConnectionState.READY and waited < max_wait:
            if conn.state == ConnectionState.FAILED:
                raise ConnectionError(f"Connection {conn.id} failed while waiting")

            await asyncio.sleep(0.1)
            waited += 1

        if conn.state != ConnectionState.READY:
            raise ConnectionError(f"Connection {conn.id} did not become ready in time")

    async def _check_reconnect_rate_limit(self) -> bool:
        """
        Check and enforce global reconnection rate limit.

        SEC-012 FIX: Prevents resource exhaustion from rapid reconnection cycles.

        Returns:
            True if reconnection is allowed, False if rate limit exceeded
            and caller should back off.
        """
        now = time.time()

        # Clean up timestamps older than 60 seconds
        self._reconnect_times = [
            t for t in self._reconnect_times
            if now - t < 60
        ]

        # Check if rate limit exceeded
        if len(self._reconnect_times) >= self._max_reconnects_per_minute:
            oldest = min(self._reconnect_times) if self._reconnect_times else now
            wait_time = 60 - (now - oldest)
            if wait_time > 0:
                logger.warning(
                    f"SEC-012: Reconnection rate limit exceeded "
                    f"({len(self._reconnect_times)}/{self._max_reconnects_per_minute} per minute), "
                    f"waiting {wait_time:.1f}s"
                )
                await asyncio.sleep(wait_time)
                # Clean again after waiting
                now = time.time()
                self._reconnect_times = [
                    t for t in self._reconnect_times
                    if now - t < 60
                ]

        # Record this reconnection attempt
        self._reconnect_times.append(now)
        return True

    async def _reconnect(self, conn: Connection, state_already_set: bool = False) -> None:
        """
        Reconnect a failed connection with exponential backoff.

        Args:
            conn: Connection to reconnect
            state_already_set: If True, assumes caller already set RECONNECTING
                             state atomically under lock (VUL-002 fix).
                             If False, will transition state safely.

        Implements exponential backoff: 1s → 1.5s → 2.25s → ... → 30s max
        with jitter to prevent thundering herd.

        SEC-012 FIX: Uses semaphore to limit concurrent reconnections
        and tracks global reconnection rate to prevent resource exhaustion.
        """
        # Handle state transition based on caller context
        if not state_already_set:
            if conn.state == ConnectionState.RECONNECTING:
                logger.debug(f"Connection {conn.id} already reconnecting")
                return  # Already reconnecting

            if conn.retry_count >= self.max_reconnect_attempts:
                logger.error(
                    f"Connection {conn.id} exceeded max reconnect attempts "
                    f"({self.max_reconnect_attempts})"
                )
                conn.state = ConnectionState.FAILED
                return

            conn.state = ConnectionState.RECONNECTING

        # SEC-012 FIX: Check global rate limit BEFORE proceeding
        await self._check_reconnect_rate_limit()

        # SEC-012 FIX: Use semaphore to limit concurrent reconnections
        async with self._reconnect_semaphore:
            # Increment retry counter
            conn.retry_count += 1

            # Enforce max attempts (unified check)
            if conn.retry_count > self.max_reconnect_attempts:
                logger.error(
                    f"Connection {conn.id} exceeded max reconnect attempts "
                    f"({self.max_reconnect_attempts})"
                )
                conn.state = ConnectionState.FAILED
                return

            # Exponential backoff with jitter
            backoff = min(1.0 * (1.5 ** conn.retry_count), 30.0)
            jitter = random.uniform(0, 0.1 * backoff)  # 0-10% jitter
            total_wait = backoff + jitter

            logger.debug(
                f"Reconnecting connection {conn.id} in {total_wait:.1f}s "
                f"(attempt {conn.retry_count}/{self.max_reconnect_attempts})"
            )

            await asyncio.sleep(total_wait)

            # Close existing WebSocket if any
            if conn.ws and not conn.ws.closed:
                try:
                    await conn.ws.close()
                except Exception as e:
                    logger.debug(f"Error closing old WebSocket: {e}")

            # Attempt reconnection
            try:
                success = await self._connect(conn, self.language, self.sample_rate)

                if success:
                    logger.info(f"Connection {conn.id} reconnected successfully")
                    conn.retry_count = 0  # Reset on success
                else:
                    # Retry again if not at max attempts
                    if conn.retry_count < self.max_reconnect_attempts:
                        self._spawn_reconnect(conn)
                    else:
                        logger.error(f"Connection {conn.id} failed all reconnect attempts")
                        conn.state = ConnectionState.FAILED

            except AuthenticationError:
                # Don't retry auth errors
                logger.error(f"Connection {conn.id} authentication failed, stopping retries")
                conn.state = ConnectionState.FAILED
                raise

            except Exception as e:
                logger.error(f"Connection {conn.id} reconnect error: {e}")
                # Retry again if not at max attempts
                if conn.retry_count < self.max_reconnect_attempts:
                    self._spawn_reconnect(conn)
                else:
                    conn.state = ConnectionState.FAILED

    async def _heartbeat_loop(self) -> None:
        """
        Background task for heartbeat monitoring and stale connection detection.

        Runs every heartbeat_interval seconds to:
        - Send ping to active connections (handled by aiohttp autoping)
        - Detect stale connections (no heartbeat in 90s)
        - Trigger reconnection for failed connections
        """
        logger.debug("Heartbeat monitoring started")

        while not self._closing:
            try:
                await asyncio.sleep(self.heartbeat_interval)

                current_time = time.time()

                for conn in self.connections:
                    # Reconnect idle connections whose socket has died.
                    #
                    # Liveness is judged on ws.closed, not on an elapsed-time
                    # threshold. aiohttp already does the probing: `heartbeat`
                    # sends pings and closes the WebSocket when a pong does not
                    # arrive, so a dead peer surfaces as ws.closed. The previous
                    # elapsed-time check could never fire - the branch that
                    # refreshed last_heartbeat ran on every tick for any
                    # connection whose ws was open, so the interval it compared
                    # against never grew past one tick. It also compared
                    # wall-clock stamps, so an NTP correction read as idle time
                    # and tore down every connection at once.
                    #
                    # IN_USE connections are left alone: they belong to a live
                    # stream with its own RECEIVE_TIMEOUT_SECONDS watchdog, which
                    # retires the connection through mark_broken().
                    if conn.state == ConnectionState.READY:
                        if conn.ws is None or conn.ws.closed:
                            logger.warning(
                                f"Connection {conn.id} socket closed while idle, "
                                "reconnecting"
                            )
                            conn.state = ConnectionState.CLOSED
                            self._spawn_reconnect(conn)
                        else:
                            # Diagnostic only - nothing gates on this value.
                            conn.last_heartbeat = current_time

                    # Opportunistically reconnect failed connections
                    elif conn.state in (ConnectionState.FAILED, ConnectionState.CLOSED):
                        if conn.retry_count < self.max_reconnect_attempts:
                            logger.debug(f"Heartbeat triggering reconnect for connection {conn.id}")
                            self._spawn_reconnect(conn)

                # Log pool health periodically
                if logger.isEnabledFor(10):  # DEBUG level
                    logger.debug(f"Pool health: {self._get_pool_status()}")

            except asyncio.CancelledError:
                logger.debug("Heartbeat monitoring cancelled")
                break

            except Exception as e:
                logger.error(f"Heartbeat loop error: {e}")
                # Continue running despite errors

        logger.debug("Heartbeat monitoring stopped")

    def _get_pool_status(self) -> str:
        """
        Get human-readable pool status.

        Returns:
            Status string like "2 ready, 1 connecting"
        """
        status: dict[str, int] = {}
        for conn in self.connections:
            state = conn.state.value
            status[state] = status.get(state, 0) + 1

        return ", ".join(
            f"{count} {state}" for state, count in sorted(status.items())
        )

    async def close(self) -> None:
        """
        Gracefully close all connections in the pool.

        Stops heartbeat monitoring and closes all WebSocket connections.
        Safe to call multiple times.
        """
        if self._closing:
            return

        self._closing = True
        logger.info("Closing connection pool")

        # Stop heartbeat monitoring
        if self._heartbeat_task and not self._heartbeat_task.done():
            self._heartbeat_task.cancel()
            try:
                await self._heartbeat_task
            except asyncio.CancelledError:
                pass

        # Close all connections
        close_tasks = []
        for conn in self.connections:
            if conn.ws and not conn.ws.closed:
                logger.debug(f"Closing connection {conn.id}")
                close_tasks.append(conn.ws.close())
                conn.state = ConnectionState.CLOSED

        if close_tasks:
            # Close all in parallel, ignore errors
            await asyncio.gather(*close_tasks, return_exceptions=True)

        # Close aiohttp session
        if self._session and not self._session.closed:
            await self._session.close()

        logger.info(f"Connection pool closed: {len(self.connections)} connections")

    def get_pool_health(self) -> dict[str, int | float]:
        """
        Get detailed pool health statistics.

        Returns:
            Dictionary with connection counts by state and health metrics

        Example:
            {
                "total": 2,
                "ready": 1,
                "in_use": 0,
                "failed": 1,
                "connecting": 0,
                "health_percentage": 50.0
            }
        """
        stats: dict[str, int | float] = {
            "total": len(self.connections),
            "ready": 0,
            "in_use": 0,
            "failed": 0,
            "connecting": 0,
            "reconnecting": 0,
            "closed": 0,
        }

        for conn in self.connections:
            state_key = conn.state.value.replace("-", "_")
            stats[state_key] = stats.get(state_key, 0) + 1

        # Calculate health percentage
        healthy = stats["ready"] + stats["in_use"]
        stats["health_percentage"] = (healthy / stats["total"] * 100) if stats["total"] > 0 else 0

        return stats
