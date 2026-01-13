"""Main VoxistSTT plugin class."""

from __future__ import annotations

import asyncio
import os
from enum import Enum

import aiohttp
from livekit.agents.stt import STT, STTCapabilities
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr

from .connection_pool import ConnectionPool
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    InitializationError,
    LanguageNotSupportedError,
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
    ):
        """
        Initialize Voxist STT plugin.

        Args:
            api_key: Voxist API key (or use VOXIST_API_KEY env var)
            language: Language code (fr, en, de, it, es, nl, pt, sv, fr-medical)
            sample_rate: Audio sample rate in Hz (default: 16000)
            base_url: WebSocket endpoint URL
            interim_results: Enable partial transcription results
            connection_pool_size: Number of persistent connections (2-3 recommended)
            connection_timeout: Connection timeout in seconds
            heartbeat_interval: Ping/pong interval in seconds
            chunk_duration_ms: Audio chunk size in milliseconds
            stride_overlap_ms: Chunk overlap for boundary accuracy
            max_reconnect_attempts: Max reconnection attempts per connection
            enable_metrics: Emit LiveKit metrics events
            http_session: Optional aiohttp session (for advanced use)
            api_key_header: HTTP header name for API key (default: X-LVL-KEY)

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

        # Initialize connection pool
        # Always use 16kHz for Voxist API (we resample internally)
        self._pool = ConnectionPool(
            base_url=base_url,
            api_key=self._api_key,
            pool_size=connection_pool_size,
            connection_timeout=connection_timeout,
            heartbeat_interval=heartbeat_interval,
            max_reconnect_attempts=max_reconnect_attempts,
            language=language,
            sample_rate=16000,  # Voxist expects 16kHz (we resample from input rate)
            api_key_header=api_key_header,
        )

        # Task lifecycle tracking (QUAL-002: asr-all-dxe)
        self._init_task: asyncio.Task | None = None
        self._init_error: Exception | None = None
        self._init_state = InitializationState.PENDING

        logger.info(
            f"VoxistSTT initialized: language={language}, "
            f"pool_size={connection_pool_size}, sample_rate={sample_rate}"
        )

        # Pre-warm connections asynchronously (non-blocking)
        # Only if event loop is running (avoid issues in tests)
        try:
            loop = asyncio.get_running_loop()
            self._init_task = loop.create_task(self._initialize_pool())
            logger.debug("Background initialization task created")
        except RuntimeError:
            # No running event loop (e.g., in tests)
            # Pool will be initialized on first stream() call
            self._init_state = InitializationState.NOT_STARTED
            logger.debug("No running event loop, pool will initialize on demand")

    async def _initialize_pool(self) -> None:
        """
        Initialize connection pool (called asynchronously).

        State transitions (QUAL-002):
            PENDING -> RUNNING (start)
            RUNNING -> COMPLETED (success)
            RUNNING -> FAILED (error)
        """
        self._init_state = InitializationState.RUNNING
        logger.debug("Initialization state: RUNNING")

        try:
            await self._pool.initialize()
            self._init_state = InitializationState.COMPLETED
            logger.debug("Connection pool pre-warming complete (state: COMPLETED)")
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
            logger.error(f"Failed to pre-warm connection pool: {e} (state: FAILED)")
            # Don't re-raise - allow stream() to attempt on-demand initialization

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

        Example:
            stream = stt.stream(language="fr-medical")
            stream.push_frame(audio_frame)
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    print(event.alternatives[0].text)
        """
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

        return VoxistSTTStream(
            stt=self,
            pool=self._pool,
            config=self._config,
            language=stream_language,
            conn_options=conn_options if conn_options is not None else APIConnectOptions(),
            enable_metrics=self._enable_metrics,
        )

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

        Cancels pending initialization task and closes all WebSocket
        connections in the pool gracefully.
        """
        logger.info("Closing VoxistSTT plugin")

        # Cancel pending initialization task if running (QUAL-HIGH: asr-all-cga)
        if self._init_task is not None and not self._init_task.done():
            self._init_task.cancel()
            try:
                await self._init_task
            except asyncio.CancelledError:
                logger.debug("Initialization task cancelled")

        await self._pool.close()
        await super().aclose()

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

        Returns:
            True if ready, False otherwise
        """
        if self._init_state == InitializationState.COMPLETED:
            return True
        # Also check pool directly for on-demand initialization
        return self._pool.is_initialized

    async def wait_for_initialization(self, timeout: float = 30.0) -> bool:
        """
        Wait for background initialization to complete (QUAL-002).

        Args:
            timeout: Maximum time to wait in seconds (default: 30.0)

        Returns:
            True if initialization completed successfully, False otherwise

        Example:
            stt = VoxistSTT(api_key="...")
            if await stt.wait_for_initialization(timeout=10.0):
                stream = stt.stream()
            else:
                logger.error(f"Init failed: {stt.initialization_error}")
        """
        if self._init_state == InitializationState.COMPLETED:
            return True

        if self._init_state == InitializationState.FAILED:
            return False

        if self._init_state == InitializationState.NOT_STARTED:
            # No background task, initialize on demand
            try:
                await asyncio.wait_for(self._pool.initialize(), timeout=timeout)
                self._init_state = InitializationState.COMPLETED
                return True
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

        # Wait for background task
        if self._init_task is not None:
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

        return self._init_state == InitializationState.COMPLETED

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

        Ensures the connection pool is initialized before use.
        Raises InitializationError if initialization fails.

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
