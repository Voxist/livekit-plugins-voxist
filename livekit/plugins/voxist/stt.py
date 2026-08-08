"""Main VoxistSTT plugin class."""

from __future__ import annotations

import asyncio
import contextlib
import os
import ssl
import weakref
from enum import Enum

import aiohttp
from livekit.agents.stt import STT, STTCapabilities
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr

from .connection import VoxistDialer
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    InitializationError,
    LanguageNotSupportedError,
)
from .exceptions import (
    ConnectionError as VoxistConnectionError,
)
from .log import logger
from .models import SUPPORTED_LANGUAGES, validate_language_format
from .stream import VoxistSTTStream


class InitializationState(Enum):
    """
    Tracks the state of background initialization task.

    State transitions:
        PENDING -> RUNNING -> COMPLETED (success)
        PENDING -> RUNNING -> FAILED (error)
        PENDING -> NOT_STARTED (no event loop)

    Use VoxistSTT.initialization_state property to check current state.
    """
    NOT_STARTED = "not_started"   # No event loop available, init on demand
    PENDING = "pending"           # Task created but not yet started
    RUNNING = "running"           # Initialization in progress
    COMPLETED = "completed"       # Successfully initialized
    FAILED = "failed"             # Initialization failed with error


class VoxistSTT(STT):
    """
    Voxist ASR Speech-to-Text plugin for LiveKit.

    Features:
    - Connection pooling for ultra-low latency (< 300ms end-to-end)
    - Support for 8+ languages including French medical
    - Automatic text2num and medical units processing (fr-medical)
    - Interim and final transcription results
    - Automatic reconnection and error recovery

    Task Lifecycle (QUAL-002):
        The plugin performs background initialization to pre-warm connections.
        Use these properties and methods to manage the initialization lifecycle:

        - initialization_state: Current state (NOT_STARTED, PENDING, RUNNING,
          COMPLETED, FAILED)
        - initialization_error: Exception if initialization failed
        - is_ready: True if initialization completed successfully
        - wait_for_initialization(): Await initialization completion with timeout
        - check_initialization(): Raise InitializationError if failed

        State transitions:
            PENDING -> RUNNING -> COMPLETED (success path)
            PENDING -> RUNNING -> FAILED (error path)
            NOT_STARTED (no event loop, init on demand)

    Example:
        # Minimal usage
        stt = VoxistSTT()  # Uses VOXIST_API_KEY env var

        # With configuration
        stt = VoxistSTT(
            api_key="voxist_...",
            language="fr-medical",
            connection_pool_size=3,
        )

        # Use in LiveKit agent
        agent = agents.VoicePipelineAgent(stt=stt, llm=..., tts=...)
        await agent.start(ctx.room)

        # Explicit initialization check (optional)
        await stt.wait_for_initialization()
        if not stt.is_ready:
            raise stt.initialization_error
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str = "fr",
        sample_rate: int = 16000,
        base_url: str = "wss://api-asr.voxist.com/ws",
        interim_results: bool = True,
        connection_pool_size: int = 2,
        connection_timeout: float = 10.0,
        heartbeat_interval: float = 30.0,
        chunk_duration_ms: int = 100,
        stride_overlap_ms: int = 20,
        max_reconnect_attempts: int = 10,
        enable_metrics: bool = True,
        http_session: aiohttp.ClientSession | None = None,
        api_key_header: str = "X-LVL-KEY",
        ssl_context: ssl.SSLContext | None = None,
        validate_websocket: bool = True,
    ):
        """
        Initialize Voxist STT plugin.

        Args:
            api_key: Voxist API key (or use VOXIST_API_KEY env var)
            language: Language code (fr, en, de, it, es, nl, pt, sv, fr-medical)
            sample_rate: Audio sample rate in Hz (default: 16000)
            base_url: WebSocket endpoint URL
            interim_results: Enable partial transcription results
            connection_pool_size: Accepted for backwards compatibility and
                IGNORED. There is no pool: the gateway closes every socket at
                end of session, so each stream dials its own. Passing a
                non-default value logs a deprecation warning.
            connection_timeout: Bound, in seconds, on both the HTTPS token
                exchange and the WebSocket dial (default: 10.0).
            heartbeat_interval: WebSocket ping/pong interval in seconds.
                Still honoured: it is passed to aiohttp's heartbeat, which is
                how a dead transport is detected during long silences.
            chunk_duration_ms: Audio chunk size in milliseconds
            stride_overlap_ms: Chunk overlap for boundary accuracy
            max_reconnect_attempts: Accepted for backwards compatibility and
                IGNORED. Retries are owned by livekit's own machinery
                (conn_options.max_retry on stream()). Passing a non-default
                value logs a deprecation warning.
            enable_metrics: Emit LiveKit metrics events
            http_session: Optional aiohttp session (for advanced use)
            api_key_header: HTTP header name for API key (default: X-LVL-KEY)
            ssl_context: Optional SSL context for TLS. Needed to reach a
                deployment whose certificate is signed by a private CA:
                certificate verification is always enabled, so without a
                context trusting that CA the connection is refused.
            validate_websocket: When True (the default), the explicit
                readiness paths - wait_for_initialization() and therefore
                __aenter__ - prove WebSocket reachability by dialing one
                short-lived WS after the token pre-fetch (see
                _validate_websocket_path for the deployment-health story
                and the cost). Set False to make readiness token-only, as
                it was before this validation existed.

        Raises:
            ConfigurationError: If API key missing or invalid config
            LanguageNotSupportedError: If language not supported
        """
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=interim_results
            )
        )

        # API Configuration
        self._api_key = api_key or os.environ.get("VOXIST_API_KEY")
        if not self._api_key:
            raise ConfigurationError(
                "Voxist API key required. Set VOXIST_API_KEY environment "
                "variable or pass api_key parameter.\n\n"
                "Get your API key at: https://asr-demo.voxist.com"
            )

        # Validate language (SEC-002 FIX: both allowlist and format validation)
        if language not in SUPPORTED_LANGUAGES:
            raise LanguageNotSupportedError(
                f"Language '{language}' not supported.\n"
                f"Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}\n"
                f"See documentation for language codes."
            )

        # SEC-002 FIX: Defense-in-depth format validation
        if not validate_language_format(language):
            raise LanguageNotSupportedError(
                f"Language '{language}' has invalid format.\n"
                f"Expected format: 'xx' or 'xx-YY' (e.g., 'fr', 'en-US', 'fr-medical')"
            )

        # Validate configuration
        if sample_rate not in [8000, 16000, 44100, 48000]:
            logger.warning(
                f"Unusual sample rate: {sample_rate}. "
                f"Recommended: 16000 Hz for optimal quality."
            )

        if connection_pool_size < 1 or connection_pool_size > 5:
            raise ConfigurationError(
                f"connection_pool_size must be 1-5, got {connection_pool_size}"
            )

        if chunk_duration_ms < 50 or chunk_duration_ms > 500:
            raise ConfigurationError(
                f"chunk_duration_ms must be 50-500ms, got {chunk_duration_ms}"
            )

        # No silent lies about dead parameters: both are accepted only for
        # backwards compatibility, and a caller passing a non-default value
        # is told so once, here, instead of wondering why it has no effect.
        if connection_pool_size != 2:
            logger.warning(
                f"connection_pool_size={connection_pool_size} is deprecated "
                "and ignored: there is no connection pool (one socket per "
                "stream, dialed on demand)"
            )
        if max_reconnect_attempts != 10:
            logger.warning(
                f"max_reconnect_attempts={max_reconnect_attempts} is "
                "deprecated and ignored: retries are owned by livekit "
                "(conn_options.max_retry on stream())"
            )

        # Store configuration
        self._config = {
            "language": language,
            "sample_rate": sample_rate,
            "interim_results": interim_results,
            "chunk_duration_ms": chunk_duration_ms,
            "stride_overlap_ms": stride_overlap_ms,
        }

        self._base_url = base_url
        self._session = http_session
        self._enable_metrics = enable_metrics

        # One dialer per plugin; one socket per stream. There is no pool:
        # the gateway ends every session by closing the socket after "Done",
        # so connections cannot be reused. connection_pool_size,
        # max_reconnect_attempts and heartbeat_interval are accepted for
        # backwards compatibility; reconnection is owned by livekit's own
        # retry (conn_options.max_retry) and liveness by aiohttp's heartbeat.
        self._ssl_context = ssl_context
        self._api_key_header = api_key_header
        self._heartbeat_interval = heartbeat_interval
        self._connection_timeout = connection_timeout
        self._owns_session = http_session is None
        self._dialer: VoxistDialer | None = None
        self._dialer_lock = asyncio.Lock()
        self._closed = False
        self._validate_websocket = validate_websocket
        self._ws_validated = False

        # Live streams, tracked weakly so a stream that ends normally
        # disappears on its own. aclose() closes these BEFORE the shared
        # HTTP session: closing the session first left live streams to
        # redial on a closed session and crash with an unmapped
        # RuntimeError('Session is closed').
        self._live_streams: weakref.WeakSet[VoxistSTTStream] = weakref.WeakSet()

        # Task lifecycle tracking (QUAL-002: asr-all-dxe)
        self._init_task: asyncio.Task | None = None
        self._init_error: Exception | None = None
        self._init_state = InitializationState.PENDING

        logger.info(
            f"VoxistSTT initialized: language={language}, "
            f"sample_rate={sample_rate}, connection_timeout={connection_timeout}"
        )

        # Pre-warm connections asynchronously (non-blocking)
        # Only if event loop is running (avoid issues in tests)
        try:
            loop = asyncio.get_running_loop()
            self._init_task = loop.create_task(self._initialize_pool())
            # The task re-raises AuthenticationError so an awaiter sees the
            # true cause, but nobody is REQUIRED to await it - without this
            # callback, an unawaited failed init surfaces as a GC-time
            # "Task exception was never retrieved" even though the failure
            # is already recorded in _init_error/_init_state.
            self._init_task.add_done_callback(self._retrieve_init_exception)
            logger.debug("Background initialization task created")
        except RuntimeError:
            # No running event loop (e.g., in tests)
            # Pool will be initialized on first stream() call
            self._init_state = InitializationState.NOT_STARTED
            logger.debug("No running event loop, pool will initialize on demand")

    @staticmethod
    def _retrieve_init_exception(task: asyncio.Task) -> None:
        """Retrieve (and debug-log) the init task's exception so GC never
        reports it as unretrieved. The failure itself is already surfaced
        through initialization_state / initialization_error."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.debug(
                f"Background initialization failed: {exc!r} (already "
                "recorded in initialization_state/initialization_error)"
            )

    @staticmethod
    def _session_loop(
        session: aiohttp.ClientSession,
    ) -> asyncio.AbstractEventLoop | None:
        """The loop an aiohttp session is bound to, or None if undeterminable."""
        loop = getattr(session, "_loop", None)
        return loop if isinstance(loop, asyncio.AbstractEventLoop) else None

    async def _ensure_dialer(self) -> VoxistDialer:
        """
        Create the HTTP session and dialer on first use, in a running loop.

        aiohttp binds a ClientSession to whichever loop first uses it; a
        stream running in a DIFFERENT loop would otherwise die with an
        unmapped RuntimeError('Event loop is closed'). Two situations are
        distinguished:

        - The bound loop is CLOSED (sequential asyncio.run calls, test
          loops): the session is unusable and nothing can race us on it.
          A plugin-owned session is rebuilt transparently for the current
          loop; a caller-supplied session is theirs, so a clear mapped
          error is raised instead of replacing it.
        - The bound loop is ALIVE but not the running one (two live loops
          sharing one VoxistSTT): multi-loop sharing is unsupported, not
          silently accommodated. Destroying the session here would leak
          connectors, thrash the token cache and sabotage the healthy
          loop's streams, so the session is left untouched and a mapped
          (retryable-but-informative) error is raised: create one
          VoxistSTT instance per event loop.
        """
        async with self._dialer_lock:
            if self._closed:
                # A stream retry racing aclose() must fail as a mapped,
                # retryable error - NOT resurrect a fresh session after
                # shutdown, and not crash with a raw RuntimeError.
                raise VoxistConnectionError(
                    "VoxistSTT is closed; cannot dial"
                )

            running = asyncio.get_running_loop()
            if self._session is not None:
                bound_loop = self._session_loop(self._session)
                defunct = self._session.closed or (
                    bound_loop is not None and bound_loop.is_closed()
                )
                if defunct:
                    if not self._owns_session:
                        raise VoxistConnectionError(
                            "the caller-supplied http_session is closed or "
                            "bound to a closed event loop; it is not "
                            "plugin-owned, so it will not be replaced"
                        )
                    old, old_loop = self._session, bound_loop
                    self._session = None
                    self._dialer = None
                    if not old.closed:
                        if old_loop is running:
                            await old.close()
                        else:
                            # Bound to a closed loop: it cannot be
                            # awaited-closed from here. Dropping it leaks at
                            # most its idle connector, once, and is logged
                            # rather than hidden.
                            logger.warning(
                                "Dropping the plugin-owned HTTP session "
                                "bound to a defunct event loop; it cannot "
                                "be closed from the current loop"
                            )
                elif bound_loop is not None and bound_loop is not running:
                    # Alive-but-different loop: do NOT destroy the healthy
                    # loop's session. The stream wraps this into a retryable
                    # APIConnectionError, so the caller gets an informative
                    # failure while the other loop keeps working.
                    raise VoxistConnectionError(
                        "VoxistSTT is bound to a different running event "
                        "loop; sharing one instance across live loops is "
                        "unsupported - create one VoxistSTT per event loop"
                    )

            if self._dialer is None:
                assert self._api_key is not None  # validated in __init__
                if self._session is None:
                    self._session = aiohttp.ClientSession()
                    self._owns_session = True
                self._dialer = VoxistDialer(
                    session=self._session,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    api_key_header=self._api_key_header,
                    ssl_context=self._ssl_context,
                    heartbeat_interval=self._heartbeat_interval,
                    connection_timeout=self._connection_timeout,
                )
            return self._dialer

    async def _dial(self, language: str):
        """
        Open a WebSocket for one stream session.

        Raises:
            AuthenticationError: The key was rejected (fatal, not retried).
            ConnectionError: Transport failure (the stream converts this to
                APIConnectionError so livekit retries it).
        """
        dialer = await self._ensure_dialer()
        # Always 16kHz on the wire; the stream resamples its input.
        return await dialer.dial(language, 16000)

    async def _initialize_pool(self) -> None:
        """
        Warm up: pre-fetch the WebSocket token (called asynchronously).

        The token exchange is the one slow step of the first dial (an HTTPS
        round-trip). Pre-fetching it keeps the InitializationState API
        meaningful: COMPLETED means the first stream dials without it, and
        FAILED surfaces a bad key at startup instead of on first use.

        Deliberately token-only: this task runs fire-and-forget on EVERY
        construction, and each WebSocket dial opens a real ASR engine
        session server-side (the gateway dials the engine on connect). The
        end-to-end WebSocket reachability proof - which the HTTPS exchange
        alone cannot give - lives in _validate_websocket_path and runs once,
        on the explicit readiness paths (wait_for_initialization /
        __aenter__), where a caller has actually asked for the guarantee.

        State transitions (QUAL-002):
            PENDING -> RUNNING (start)
            RUNNING -> COMPLETED (success)
            RUNNING -> FAILED (error)
        """
        self._init_state = InitializationState.RUNNING
        logger.debug("Initialization state: RUNNING")

        try:
            dialer = await self._ensure_dialer()
            await dialer._get_token_url()
            self._init_state = InitializationState.COMPLETED
            logger.debug("Token pre-fetch complete (state: COMPLETED)")
        except AuthenticationError as e:
            # Store and re-raise critical errors - never swallow auth failures
            self._init_error = e
            self._init_state = InitializationState.FAILED
            logger.error(f"Authentication failed during pool initialization: {e} (state: FAILED)")
            raise
        except Exception as e:
            # Store error for later access, mark as failed
            self._init_error = e
            self._init_state = InitializationState.FAILED
            logger.error(f"Failed to pre-fetch WebSocket token: {e} (state: FAILED)")
            # Don't re-raise - allow stream() to attempt on-demand initialization

    async def _validate_websocket_path(self, timeout: float) -> None:
        """
        Prove end-to-end WebSocket reachability with one short-lived dial.

        Deployment-health story: the token pre-fetch is plain HTTPS, so it
        succeeds on deployments where the WebSocket path is broken (a proxy
        stripping the Upgrade header, a firewall blocking WS) - "healthy"
        startup, then every real call fails. Dialing one WS and closing it
        immediately restores the old initialize() guarantee: COMPLETED
        readiness means a stream can actually connect.

        Cost, weighed deliberately: the gateway opens a real ASR engine
        session server-side on connect (lang rides the URL). One extra
        short-lived engine session per PLUGIN STARTUP is acceptable; per
        stream it would not be - hence the _ws_validated cache (at most one
        probe per plugin) and hence this living on the explicit readiness
        paths, not in the fire-and-forget background warm-up.

        Raises whatever the dial raises (mapped ConnectionError /
        AuthenticationError, or asyncio.TimeoutError from the bound); the
        caller records it as an initialization failure.
        """
        if self._ws_validated or not self._validate_websocket:
            return
        dialer = await self._ensure_dialer()
        language = self._config["language"]
        assert isinstance(language, str)  # validated in __init__
        ws = await asyncio.wait_for(
            dialer.dial(language, 16000), timeout=timeout
        )
        self._ws_validated = True
        try:
            await ws.close()
        except Exception as e:
            # Reachability is proven by the successful dial; a hiccup while
            # closing the probe socket must not fail readiness.
            logger.debug(f"Closing the warm-up WebSocket failed: {e!r}")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions | None = None,
    ) -> VoxistSTTStream:
        """
        Create a new streaming recognition session.

        Args:
            language: Override default language for this stream
            conn_options: LiveKit connection options

        Returns:
            VoxistSTTStream instance ready for audio streaming

        Raises:
            RuntimeError: The plugin has been closed with aclose(). Creating
                a stream on a closed plugin is a programming error; failing
                here, at the call site, beats burning livekit's whole retry
                budget against a plugin that can never dial again.
            LanguageNotSupportedError: The language override is invalid.

        Example:
            stream = stt.stream(language="fr-medical")
            stream.push_frame(audio_frame)
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    print(event.alternatives[0].text)
        """
        if self._closed:
            raise RuntimeError(
                "stream() called on a closed VoxistSTT; create a new "
                "instance (a closed plugin cannot dial, and retrying "
                "against it can never succeed)"
            )

        stream_language = language if language is not NOT_GIVEN else self._config["language"]

        # Validate language if overridden (SEC-002 FIX: both allowlist and format)
        if language is not NOT_GIVEN:
            assert isinstance(language, str)  # Type narrowing for mypy
            if language not in SUPPORTED_LANGUAGES:
                raise LanguageNotSupportedError(
                    f"Language '{language}' not supported. "
                    f"Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
                )
            # SEC-002 FIX: Defense-in-depth format validation
            if not validate_language_format(language):
                raise LanguageNotSupportedError(
                    f"Language '{language}' has invalid format. "
                    f"Expected: 'xx' or 'xx-YY' (e.g., 'fr', 'en-US')"
                )

        # Ensure stream_language is str (from language or config)
        assert isinstance(stream_language, str)

        stream = VoxistSTTStream(
            stt=self,
            config=self._config,
            language=stream_language,
            conn_options=conn_options if conn_options is not None else APIConnectOptions(),
            enable_metrics=self._enable_metrics,
        )
        # Tracked weakly so aclose() can shut live streams down BEFORE the
        # shared HTTP session goes away (see aclose); streams that finish on
        # their own drop out of the set automatically.
        self._live_streams.add(stream)
        return stream

    async def _recognize_impl(  # type: ignore[override]
        self,
        buffer,
        *,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        """
        Batch recognition not implemented.

        Voxist plugin only supports streaming recognition for real-time use cases.
        Use stream() method instead.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            "Batch recognition not supported by Voxist plugin. "
            "Use stream() method for real-time transcription."
        )

    async def aclose(self) -> None:
        """
        Cleanup plugin resources.

        Order matters: live streams are closed BEFORE the shared HTTP
        session. Closing the session first left running streams to retry
        their dial on a closed session and crash with an unmapped
        RuntimeError('Session is closed') instead of ending cleanly.

        This teardown is best-effort-COMPLETE: every step runs even if an
        earlier one fails, and failures are logged rather than re-raised.
        aclose() is called from finally blocks and __aexit__; an exception
        escaping mid-teardown would both mask the caller's original error
        and abort the remaining cleanup (leaking streams or the HTTP
        session). Only cancellation of aclose() itself propagates.
        """
        logger.info("Closing VoxistSTT plugin")
        # From here on, _ensure_dialer refuses to build a new session, so a
        # stream retry racing this shutdown dies as a mapped ConnectionError.
        self._closed = True

        # Cancel pending initialization task if running (QUAL-HIGH:
        # asr-all-cga). The task may live on a dead or foreign loop (the
        # plugin was built under one asyncio.run and closed under another);
        # cancelling/awaiting it from here would then raise RuntimeError and,
        # unguarded, abort the rest of this teardown before any stream or
        # the session was closed.
        init_task = self._init_task
        if init_task is not None and not init_task.done():
            if init_task.get_loop().is_closed():
                # The task can never run again, and nothing on this loop can
                # cancel or await it. Drop the reference and move on.
                logger.debug(
                    "Dropping the init task stranded on a closed event loop"
                )
                self._init_task = None
            else:
                try:
                    init_task.cancel()
                    await init_task
                except asyncio.CancelledError:
                    logger.debug("Initialization task cancelled")
                except RuntimeError as e:
                    # Task bound to a live loop that is not this one:
                    # unawaitable from here, but the teardown must go on.
                    logger.warning(
                        f"Could not await the init task during close: {e!r}"
                    )
                except Exception as e:
                    # Lost race: the task completed with an error between
                    # the done() check and the cancel. Already recorded in
                    # initialization_error; not this shutdown's problem.
                    logger.debug(f"Init task finished with {e!r} during close")

        # Close live streams first (RecognizeStream.aclose closes the input
        # channel and cancels _main_task, which tears down the socket).
        # A stream may have been closed concurrently by its owner; that is
        # not this shutdown's failure to report.
        for stream in list(self._live_streams):
            with contextlib.suppress(Exception):
                await stream.aclose()

        try:
            await super().aclose()
        except Exception as e:
            logger.warning(f"Base STT close failed during shutdown: {e!r}")

        # Sockets are per-stream and close with their stream; the only shared
        # resource is the HTTP session, and only if this plugin created it.
        try:
            if (
                self._owns_session
                and self._session is not None
                and not self._session.closed
            ):
                await self._session.close()
        except Exception as e:
            logger.warning(f"HTTP session close failed during shutdown: {e!r}")

    @property
    def initialization_error(self) -> Exception | None:
        """Return any error that occurred during initialization."""
        return self._init_error

    @property
    def initialization_state(self) -> InitializationState:
        """
        Return current initialization state (QUAL-002).

        Returns:
            InitializationState enum value:
            - NOT_STARTED: No event loop was available, init on demand
            - PENDING: Task created but not yet started
            - RUNNING: Initialization in progress
            - COMPLETED: Successfully initialized
            - FAILED: Initialization failed with error
        """
        return self._init_state

    @property
    def is_ready(self) -> bool:
        """
        Check if plugin is ready for use (QUAL-002).

        Returns True if:
        - Background initialization completed successfully, OR
        - Pool is initialized (via on-demand or context manager)

        Returns False whenever initialization FAILED - including a failed
        WebSocket reachability probe (see _validate_websocket_path), even
        though the token itself was fetched: a deployment whose WS path is
        blocked is not ready, whatever its HTTPS endpoint says.

        Returns:
            True if ready, False otherwise
        """
        if self._init_state == InitializationState.COMPLETED:
            return True
        if self._init_state == InitializationState.FAILED:
            return False
        # Also ready if a token was fetched on demand by a stream
        return self._dialer is not None and self._dialer._token_url is not None

    async def wait_for_initialization(self, timeout: float = 30.0) -> bool:
        """
        Wait for background initialization to complete (QUAL-002).

        Beyond awaiting the token warm-up, this is the path that proves the
        deployment end to end: unless validate_websocket=False, the first
        successful call also dials one short-lived WebSocket (see
        _validate_websocket_path) so True means "a stream can actually
        connect", not merely "the HTTPS token endpoint answered".

        Args:
            timeout: Maximum time to wait in seconds (default: 30.0).
                Bounds the token warm-up and the WS probe separately.

        Returns:
            True if initialization completed successfully, False otherwise

        Example:
            stt = VoxistSTT(api_key="...")
            if await stt.wait_for_initialization(timeout=10.0):
                stream = stt.stream()
            else:
                logger.error(f"Init failed: {stt.initialization_error}")
        """
        if self._init_state == InitializationState.FAILED:
            return False

        if self._init_state == InitializationState.NOT_STARTED:
            # No background task, initialize on demand
            try:
                await asyncio.wait_for(self._initialize_pool(), timeout=timeout)
                # _initialize_pool records its own outcome and swallows
                # non-auth errors (streams may still succeed on demand), so
                # the state - not the absence of an exception - is the result.
            except asyncio.TimeoutError:
                self._init_error = asyncio.TimeoutError(
                    f"Initialization timed out after {timeout}s"
                )
                self._init_state = InitializationState.FAILED
                return False
            except Exception as e:
                self._init_error = e
                self._init_state = InitializationState.FAILED
                return False
        elif (
            self._init_state != InitializationState.COMPLETED
            and self._init_task is not None
        ):
            # Wait for background task
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._init_task),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                self._init_error = asyncio.TimeoutError(
                    f"Initialization timed out after {timeout}s"
                )
                self._init_state = InitializationState.FAILED
                return False
            except Exception:
                # Error already stored in _init_error by _initialize_pool
                pass

        if self._init_state != InitializationState.COMPLETED:
            return False

        # Token warm-up succeeded; now prove the WebSocket path (once).
        try:
            await self._validate_websocket_path(timeout)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._init_error = e
            self._init_state = InitializationState.FAILED
            logger.error(
                f"WebSocket reachability validation failed: {e!r} "
                "(state: FAILED - the token endpoint works but the "
                "WebSocket path does not)"
            )
            return False
        return True

    def check_initialization(self) -> None:
        """
        Raise InitializationError if initialization failed (QUAL-002).

        Use this before operations that require successful initialization.

        Raises:
            InitializationError: If initialization failed

        Example:
            stt.check_initialization()  # Raises if failed
            stream = stt.stream()
        """
        if self._init_state == InitializationState.FAILED:
            raise InitializationError(
                f"Plugin initialization failed: {self._init_error}"
            ) from self._init_error

    async def __aenter__(self) -> VoxistSTT:
        """
        Enter async context manager.

        Ensures the plugin is initialized before use - including, unless
        validate_websocket=False, proof that a WebSocket can actually be
        dialed (not just that the HTTPS token endpoint answers). Raises
        InitializationError if either fails.

        Example:
            async with VoxistSTT(api_key="...") as stt:
                stream = stt.stream()
                # Use stream...

        Returns:
            Self for use in context

        Raises:
            InitializationError: If initialization fails
        """
        # Wait for initialization using lifecycle-aware method (QUAL-002)
        if not await self.wait_for_initialization():
            # Raise with context from the original error
            raise InitializationError(
                f"Plugin initialization failed: {self._init_error}"
            ) from self._init_error

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit async context manager.

        Cleans up resources regardless of exception.

        Args:
            exc_type: Exception type if raised in context
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        await self.aclose()
