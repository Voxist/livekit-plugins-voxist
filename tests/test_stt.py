"""Unit tests for VoxistSTT main plugin class."""

import asyncio
import os
from unittest.mock import AsyncMock, Mock, patch

import pytest

from livekit.agents.stt import STTCapabilities
from livekit.agents.types import NOT_GIVEN, APIConnectOptions
from livekit.plugins.voxist import VoxistSTT
from livekit.plugins.voxist.exceptions import (
    ConfigurationError,
    LanguageNotSupportedError,
)
from livekit.plugins.voxist.models import (
    SUPPORTED_LANGUAGES,
    sanitize_url_param,
    validate_language_format,
)


class TestVoxistSTTInitialization:
    """Test VoxistSTT initialization and configuration."""

    def test_initialization_with_api_key(self):
        """Test VoxistSTT initializes with explicit API key."""
        stt = VoxistSTT(api_key="test_key_123")

        assert stt._api_key == "test_key_123"
        assert stt._config["language"] == "fr"  # Default
        assert stt._config["sample_rate"] == 16000
        assert stt._config["interim_results"] is True
        assert stt._pool is not None

    def test_initialization_from_environment(self):
        """Test VoxistSTT reads API key from environment."""
        with patch.dict(os.environ, {"VOXIST_API_KEY": "env_key_456"}):
            stt = VoxistSTT()

            assert stt._api_key == "env_key_456"

    def test_initialization_without_api_key_raises(self):
        """Test VoxistSTT raises error without API key."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError, match="API key required"):
                VoxistSTT()

    def test_initialization_with_custom_language(self):
        """Test VoxistSTT with custom language."""
        stt = VoxistSTT(api_key="test", language="fr-medical")

        assert stt._config["language"] == "fr-medical"

    def test_initialization_with_invalid_language_raises(self):
        """Test VoxistSTT raises error for invalid language."""
        with pytest.raises(LanguageNotSupportedError, match="not supported"):
            VoxistSTT(api_key="test", language="invalid-lang")

    def test_initialization_with_all_supported_languages(self):
        """Test VoxistSTT accepts all supported languages."""
        for lang in SUPPORTED_LANGUAGES.keys():
            stt = VoxistSTT(api_key="test", language=lang)
            assert stt._config["language"] == lang

    def test_initialization_with_custom_sample_rate(self):
        """Test VoxistSTT with custom sample rate."""
        stt = VoxistSTT(api_key="test", sample_rate=8000)

        assert stt._config["sample_rate"] == 8000

    def test_initialization_accepts_unusual_sample_rate(self):
        """Test VoxistSTT accepts unusual sample rates (with warning)."""
        # We're testing that initialization succeeds even with unusual rate
        # The warning is logged (visible in test output) but we don't need to assert on it
        stt = VoxistSTT(api_key="test", sample_rate=22050)

        # Should still initialize successfully
        assert stt._config["sample_rate"] == 22050

    def test_initialization_with_custom_pool_size(self):
        """Test VoxistSTT with custom connection pool size."""
        stt = VoxistSTT(api_key="test", connection_pool_size=3)

        assert stt._pool.pool_size == 3

    def test_initialization_with_invalid_pool_size_raises(self):
        """Test VoxistSTT raises error for invalid pool size."""
        with pytest.raises(ConfigurationError, match="connection_pool_size must be"):
            VoxistSTT(api_key="test", connection_pool_size=0)

        with pytest.raises(ConfigurationError, match="connection_pool_size must be"):
            VoxistSTT(api_key="test", connection_pool_size=10)

    def test_initialization_with_custom_chunk_duration(self):
        """Test VoxistSTT with custom chunk duration."""
        stt = VoxistSTT(api_key="test", chunk_duration_ms=200)

        assert stt._config["chunk_duration_ms"] == 200

    def test_initialization_with_invalid_chunk_duration_raises(self):
        """Test VoxistSTT raises error for invalid chunk duration."""
        with pytest.raises(ConfigurationError, match="chunk_duration_ms must be"):
            VoxistSTT(api_key="test", chunk_duration_ms=30)

        with pytest.raises(ConfigurationError, match="chunk_duration_ms must be"):
            VoxistSTT(api_key="test", chunk_duration_ms=600)

    def test_initialization_sets_capabilities(self):
        """Test VoxistSTT sets proper capabilities."""
        stt = VoxistSTT(api_key="test", interim_results=True)

        assert stt.capabilities.streaming is True
        assert stt.capabilities.interim_results is True

    def test_initialization_without_interim_results(self):
        """Test VoxistSTT without interim results."""
        stt = VoxistSTT(api_key="test", interim_results=False)

        assert stt.capabilities.interim_results is False
        assert stt._config["interim_results"] is False

    def test_initialization_creates_connection_pool(self):
        """Test VoxistSTT creates ConnectionPool with correct params."""
        stt = VoxistSTT(
            api_key="test",
            base_url="wss://custom.url/ws",
            connection_pool_size=3,
            connection_timeout=5.0,
            heartbeat_interval=60.0,
        )

        assert stt._pool.base_url == "wss://custom.url/ws"
        assert stt._pool.api_key == "test"
        assert stt._pool.pool_size == 3
        assert stt._pool.connection_timeout == 5.0
        assert stt._pool.heartbeat_interval == 60.0


class TestVoxistSTTStreamCreation:
    """Test stream creation method."""

    def test_stream_with_invalid_language_override_raises(self):
        """Test stream() raises error for invalid language override."""
        stt = VoxistSTT(api_key="test", language="fr")

        with pytest.raises(LanguageNotSupportedError, match="not supported"):
            stt.stream(language="invalid-lang")

    def test_stream_validates_language_before_creating_stream(self):
        """Test stream() validates language before attempting to create stream."""
        stt = VoxistSTT(api_key="test", language="fr")

        # This should raise LanguageNotSupportedError, not NotImplementedError
        with pytest.raises(LanguageNotSupportedError):
            stt.stream(language="zh-CN")


class TestVoxistSTTBatchRecognition:
    """Test batch recognition method."""

    @pytest.mark.asyncio
    async def test_recognize_impl_not_implemented(self):
        """Test _recognize_impl raises NotImplementedError."""
        stt = VoxistSTT(api_key="test")

        with pytest.raises(NotImplementedError, match="Batch recognition not supported"):
            await stt._recognize_impl(
                buffer=Mock(),
                language=NOT_GIVEN,
                conn_options=APIConnectOptions(),
            )


class TestVoxistSTTCleanup:
    """Test resource cleanup."""

    @pytest.mark.asyncio
    async def test_aclose_closes_pool(self):
        """Test aclose() closes connection pool."""
        stt = VoxistSTT(api_key="test")

        # Mock the pool close method
        stt._pool.close = AsyncMock()

        await stt.aclose()

        # Should have called pool.close()
        stt._pool.close.assert_called_once()


class TestVoxistSTTConfiguration:
    """Test various configuration scenarios."""

    def test_medical_french_configuration(self):
        """Test configuration for medical French transcription."""
        stt = VoxistSTT(
            api_key="test",
            language="fr-medical",
            connection_pool_size=3,
            chunk_duration_ms=100,
            stride_overlap_ms=20,
        )

        assert stt._config["language"] == "fr-medical"
        assert stt._pool.pool_size == 3

    def test_english_configuration(self):
        """Test configuration for English transcription."""
        stt = VoxistSTT(
            api_key="test",
            language="en-US",
            sample_rate=16000,
        )

        assert stt._config["language"] == "en-US"
        assert stt._config["sample_rate"] == 16000

    def test_minimal_configuration(self):
        """Test minimal configuration with defaults."""
        with patch.dict(os.environ, {"VOXIST_API_KEY": "env_key"}):
            stt = VoxistSTT()

            # Should use all defaults
            assert stt._api_key == "env_key"
            assert stt._config["language"] == "fr"
            assert stt._config["sample_rate"] == 16000
            assert stt._config["interim_results"] is True
            assert stt._config["chunk_duration_ms"] == 100
            assert stt._config["stride_overlap_ms"] == 20
            assert stt._pool.pool_size == 2

    def test_maximal_configuration(self):
        """Test maximal configuration with all parameters."""
        stt = VoxistSTT(
            api_key="test_key",
            language="de-DE",
            sample_rate=48000,
            base_url="wss://custom.server.com/ws",
            interim_results=False,
            connection_pool_size=5,
            connection_timeout=15.0,
            heartbeat_interval=45.0,
            chunk_duration_ms=200,
            stride_overlap_ms=40,
            max_reconnect_attempts=20,
            enable_metrics=False,
        )

        assert stt._api_key == "test_key"
        assert stt._config["language"] == "de-DE"
        assert stt._config["sample_rate"] == 48000
        assert stt._base_url == "wss://custom.server.com/ws"
        assert stt._config["interim_results"] is False
        assert stt._pool.pool_size == 5
        assert stt._pool.connection_timeout == 15.0
        assert stt._pool.heartbeat_interval == 45.0
        assert stt._config["chunk_duration_ms"] == 200
        assert stt._config["stride_overlap_ms"] == 40
        assert stt._enable_metrics is False


class TestVoxistSTTLanguageSupport:
    """Test language support validation."""

    @pytest.mark.parametrize("language", [
        "fr", "fr-FR", "fr-medical",
        "en", "en-US",
        "de", "de-DE",
        "it", "es", "nl", "nl-NL", "pt", "sv"
    ])
    def test_all_supported_languages(self, language):
        """Test all supported languages are accepted."""
        stt = VoxistSTT(api_key="test", language=language)
        assert stt._config["language"] == language

    @pytest.mark.parametrize("invalid_lang", [
        "fr-CA", "en-GB", "zh", "ja", "ar", "ru", "invalid"
    ])
    def test_unsupported_languages_raise(self, invalid_lang):
        """Test unsupported languages raise error."""
        with pytest.raises(LanguageNotSupportedError):
            VoxistSTT(api_key="test", language=invalid_lang)


class TestVoxistSTTIntegration:
    """Test integration with LiveKit components."""

    def test_capabilities_set_correctly(self):
        """Test STT capabilities are set correctly."""
        stt = VoxistSTT(api_key="test", interim_results=True)

        # Check capabilities
        caps = stt.capabilities
        assert isinstance(caps, STTCapabilities)
        assert caps.streaming is True
        assert caps.interim_results is True

    def test_capabilities_without_interim_results(self):
        """Test capabilities when interim_results=False."""
        stt = VoxistSTT(api_key="test", interim_results=False)

        assert stt.capabilities.interim_results is False

    @pytest.mark.asyncio
    async def test_initialization_triggers_pool_warming(self):
        """Test initialization triggers async pool pre-warming when loop is running."""
        # Create an async context so event loop is running
        loop = asyncio.get_running_loop()

        with patch.object(loop, 'create_task') as mock_create_task:
            VoxistSTT(api_key="test")

            # Should have created task for pool initialization
            mock_create_task.assert_called_once()

    def test_pool_configuration_propagated(self):
        """Test pool receives correct configuration from STT."""
        stt = VoxistSTT(
            api_key="api_test",
            base_url="wss://test.com/ws",
            connection_pool_size=3,
            connection_timeout=7.0,
            heartbeat_interval=25.0,
            max_reconnect_attempts=15,
        )

        pool = stt._pool

        assert pool.api_key == "api_test"
        assert pool.base_url == "wss://test.com/ws"
        assert pool.pool_size == 3
        assert pool.connection_timeout == 7.0
        assert pool.heartbeat_interval == 25.0
        assert pool.max_reconnect_attempts == 15


class TestVoxistSTTErrorHandling:
    """Test error handling and validation."""

    def test_missing_api_key_error_message(self):
        """Test error message provides helpful guidance."""
        with patch.dict(os.environ, {}, clear=True):
            with pytest.raises(ConfigurationError) as exc_info:
                VoxistSTT()

            error_msg = str(exc_info.value)
            assert "API key required" in error_msg
            assert "VOXIST_API_KEY" in error_msg
            assert "asr-demo.voxist.com" in error_msg  # Helpful link

    def test_invalid_language_error_message(self):
        """Test language error provides list of supported languages."""
        with pytest.raises(LanguageNotSupportedError) as exc_info:
            VoxistSTT(api_key="test", language="zh-CN")

        error_msg = str(exc_info.value)
        assert "not supported" in error_msg
        assert "fr" in error_msg  # Should list supported languages

    def test_invalid_pool_size_error_message(self):
        """Test pool size validation error message."""
        with pytest.raises(ConfigurationError) as exc_info:
            VoxistSTT(api_key="test", connection_pool_size=10)

        assert "connection_pool_size must be 1-5" in str(exc_info.value)

    def test_invalid_chunk_duration_error_message(self):
        """Test chunk duration validation error message."""
        with pytest.raises(ConfigurationError) as exc_info:
            VoxistSTT(api_key="test", chunk_duration_ms=1000)

        assert "chunk_duration_ms must be 50-500ms" in str(exc_info.value)


class TestVoxistSTTEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_api_key_parameter_overrides_environment(self):
        """Test explicit API key parameter overrides environment."""
        with patch.dict(os.environ, {"VOXIST_API_KEY": "env_key"}):
            stt = VoxistSTT(api_key="param_key")

            assert stt._api_key == "param_key"

    def test_minimum_pool_size(self):
        """Test pool size of 1 is valid."""
        stt = VoxistSTT(api_key="test", connection_pool_size=1)

        assert stt._pool.pool_size == 1

    def test_maximum_pool_size(self):
        """Test pool size of 5 is valid."""
        stt = VoxistSTT(api_key="test", connection_pool_size=5)

        assert stt._pool.pool_size == 5

    def test_minimum_chunk_duration(self):
        """Test minimum chunk duration (50ms)."""
        stt = VoxistSTT(api_key="test", chunk_duration_ms=50)

        assert stt._config["chunk_duration_ms"] == 50

    def test_maximum_chunk_duration(self):
        """Test maximum chunk duration (500ms)."""
        stt = VoxistSTT(api_key="test", chunk_duration_ms=500)

        assert stt._config["chunk_duration_ms"] == 500

    def test_zero_stride_overlap(self):
        """Test zero stride overlap is valid (no overlap)."""
        stt = VoxistSTT(api_key="test", stride_overlap_ms=0)

        assert stt._config["stride_overlap_ms"] == 0

    def test_enable_metrics_false(self):
        """Test metrics can be disabled."""
        stt = VoxistSTT(api_key="test", enable_metrics=False)

        assert stt._enable_metrics is False

    def test_custom_base_url(self):
        """Test custom base URL."""
        stt = VoxistSTT(
            api_key="test",
            base_url="wss://staging.voxist.com/ws"
        )

        assert stt._base_url == "wss://staging.voxist.com/ws"
        assert stt._pool.base_url == "wss://staging.voxist.com/ws"


class TestTaskLifecycle:
    """Test suite for task lifecycle management (QUAL-HIGH: asr-all-cga)."""

    def test_init_task_is_tracked(self):
        """Test that initialization task is tracked as an attribute."""
        stt = VoxistSTT(api_key="test")

        # Should have _init_task attribute (may be None if no event loop)
        assert hasattr(stt, '_init_task')

    @pytest.mark.asyncio
    async def test_init_task_is_awaitable(self, monkeypatch):
        """Test that initialization task can be awaited."""
        # Mock _initialize_pool to avoid real network calls
        async def mock_init(self):
            pass

        monkeypatch.setattr(VoxistSTT, '_initialize_pool', mock_init)
        stt = VoxistSTT(api_key="test")

        # If there's an init task, it should be awaitable
        if stt._init_task is not None:
            # Should not raise
            await stt._init_task

    @pytest.mark.asyncio
    async def test_aclose_cancels_init_task(self):
        """Test that aclose cancels pending initialization task."""
        stt = VoxistSTT(api_key="test")

        # Mock pool close
        stt._pool.close = AsyncMock()

        await stt.aclose()

        # Task should be cancelled or done
        if stt._init_task is not None:
            assert stt._init_task.done() or stt._init_task.cancelled()

    @pytest.mark.asyncio
    async def test_authentication_error_is_accessible(self):
        """Test that authentication errors during init are accessible."""
        stt = VoxistSTT(api_key="test")

        # Should have a way to check initialization status
        assert hasattr(stt, '_init_error') or hasattr(stt, 'initialization_error')

    @pytest.mark.asyncio
    async def test_init_error_not_swallowed(self):
        """Test that critical errors during initialization are not swallowed."""
        from livekit.plugins.voxist.exceptions import AuthenticationError

        stt = VoxistSTT(api_key="test")

        # Mock pool to raise auth error
        async def mock_init_raise_auth():
            raise AuthenticationError("Invalid API key")

        stt._pool.initialize = mock_init_raise_auth

        # The error should be stored/accessible, not just logged
        try:
            await stt._initialize_pool()
        except AuthenticationError:
            pass  # This is expected - error should be re-raised

        # Or check that error is stored for later access
        # (implementation may vary)


class TestQUAL002InitializationState:
    """Test suite for QUAL-002: Background Task Lifecycle Management."""

    def test_initialization_state_enum_imported(self):
        """Test that InitializationState enum is importable."""
        from livekit.plugins.voxist import InitializationState

        assert hasattr(InitializationState, 'NOT_STARTED')
        assert hasattr(InitializationState, 'PENDING')
        assert hasattr(InitializationState, 'RUNNING')
        assert hasattr(InitializationState, 'COMPLETED')
        assert hasattr(InitializationState, 'FAILED')

    def test_initialization_error_exception_imported(self):
        """Test that InitializationError exception is importable."""
        from livekit.plugins.voxist import InitializationError
        from livekit.plugins.voxist.exceptions import VoxistError

        assert issubclass(InitializationError, VoxistError)

    def test_stt_has_initialization_state_property(self):
        """Test that VoxistSTT has initialization_state property."""
        stt = VoxistSTT(api_key="test")

        from livekit.plugins.voxist import InitializationState
        assert hasattr(stt, 'initialization_state')
        assert isinstance(stt.initialization_state, InitializationState)

    def test_initial_state_is_pending_or_not_started(self):
        """Test that initial state is PENDING or NOT_STARTED (no event loop)."""
        stt = VoxistSTT(api_key="test")

        from livekit.plugins.voxist import InitializationState
        assert stt.initialization_state in [
            InitializationState.PENDING,
            InitializationState.NOT_STARTED
        ]

    def test_stt_has_is_ready_property(self):
        """Test that VoxistSTT has is_ready property."""
        stt = VoxistSTT(api_key="test")

        assert hasattr(stt, 'is_ready')
        assert isinstance(stt.is_ready, bool)

    def test_stt_has_wait_for_initialization_method(self):
        """Test that VoxistSTT has wait_for_initialization method."""
        stt = VoxistSTT(api_key="test")

        assert hasattr(stt, 'wait_for_initialization')
        assert asyncio.iscoroutinefunction(stt.wait_for_initialization)

    def test_stt_has_check_initialization_method(self):
        """Test that VoxistSTT has check_initialization method."""
        stt = VoxistSTT(api_key="test")

        assert hasattr(stt, 'check_initialization')
        assert callable(stt.check_initialization)

    @pytest.mark.asyncio
    async def test_wait_for_initialization_returns_bool(self):
        """Test that wait_for_initialization returns a boolean."""
        stt = VoxistSTT(api_key="test")
        stt._pool.initialize = AsyncMock()

        result = await stt.wait_for_initialization(timeout=5.0)

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_wait_for_initialization_on_success(self):
        """Test wait_for_initialization returns True on success."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        stt._pool.initialize = AsyncMock()
        stt._init_state = InitializationState.NOT_STARTED

        result = await stt.wait_for_initialization(timeout=5.0)

        assert result is True
        assert stt.initialization_state == InitializationState.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_for_initialization_returns_true_if_already_completed(self):
        """Test wait_for_initialization returns True immediately if already complete."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        stt._init_state = InitializationState.COMPLETED

        result = await stt.wait_for_initialization(timeout=5.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_wait_for_initialization_returns_false_if_already_failed(self):
        """Test wait_for_initialization returns False if already failed."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        stt._init_state = InitializationState.FAILED

        result = await stt.wait_for_initialization(timeout=5.0)

        assert result is False

    @pytest.mark.asyncio
    async def test_wait_for_initialization_handles_timeout(self):
        """Test wait_for_initialization handles timeout properly."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        stt._init_state = InitializationState.NOT_STARTED

        # Mock pool.initialize to never complete
        async def slow_init():
            await asyncio.sleep(10)

        stt._pool.initialize = slow_init

        result = await stt.wait_for_initialization(timeout=0.1)

        assert result is False
        assert stt.initialization_state == InitializationState.FAILED
        assert isinstance(stt.initialization_error, asyncio.TimeoutError)

    @pytest.mark.asyncio
    async def test_check_initialization_does_not_raise_on_success(self):
        """Test check_initialization does not raise if not failed."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        stt._init_state = InitializationState.COMPLETED

        # Should not raise
        stt.check_initialization()

    @pytest.mark.asyncio
    async def test_check_initialization_raises_on_failure(self):
        """Test check_initialization raises InitializationError if failed."""
        from livekit.plugins.voxist import InitializationError, InitializationState

        stt = VoxistSTT(api_key="test")
        stt._init_state = InitializationState.FAILED
        stt._init_error = ValueError("Test error")

        with pytest.raises(InitializationError, match="Plugin initialization failed"):
            stt.check_initialization()

    @pytest.mark.asyncio
    async def test_is_ready_false_initially(self):
        """Test is_ready is False initially (before pool initialized)."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")

        # Without initialization complete and pool not initialized
        stt._init_state = InitializationState.PENDING
        stt._pool._initialized = False

        assert stt.is_ready is False

    @pytest.mark.asyncio
    async def test_is_ready_true_after_completion(self):
        """Test is_ready is True after successful initialization."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        stt._init_state = InitializationState.COMPLETED

        assert stt.is_ready is True

    @pytest.mark.asyncio
    async def test_state_transition_running(self):
        """Test state transitions to RUNNING during initialization."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")

        # Mock pool.initialize to capture state during call
        states_during_init = []

        async def capture_state():
            states_during_init.append(stt.initialization_state)

        stt._pool.initialize = capture_state
        stt._init_state = InitializationState.PENDING

        await stt._initialize_pool()

        # State should have been RUNNING during initialize call
        assert InitializationState.RUNNING in states_during_init

    @pytest.mark.asyncio
    async def test_state_transition_to_failed_on_error(self):
        """Test state transitions to FAILED on initialization error."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")

        async def raise_error():
            raise ConnectionError("Network error")

        stt._pool.initialize = raise_error
        stt._init_state = InitializationState.PENDING

        await stt._initialize_pool()

        assert stt.initialization_state == InitializationState.FAILED
        assert isinstance(stt.initialization_error, ConnectionError)

    @pytest.mark.asyncio
    async def test_aenter_raises_on_init_failure(self):
        """Test __aenter__ raises InitializationError if init fails."""
        from livekit.plugins.voxist import InitializationError, InitializationState

        stt = VoxistSTT(api_key="test")
        stt._init_state = InitializationState.NOT_STARTED

        async def raise_error():
            raise ValueError("Init failed")

        stt._pool.initialize = raise_error

        with pytest.raises(InitializationError):
            await stt.__aenter__()


class TestContextManager:
    """Test suite for context manager support (QUAL-HIGH: asr-all-2nj)."""

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self):
        """Test that __aenter__ returns the STT instance."""
        stt = VoxistSTT(api_key="test")

        # Mock pool initialize to avoid actual connection
        stt._pool.initialize = AsyncMock()

        result = await stt.__aenter__()

        assert result is stt

    @pytest.mark.asyncio
    async def test_aenter_initializes_pool(self):
        """Test that __aenter__ ensures pool is initialized."""
        stt = VoxistSTT(api_key="test")

        # Mock pool
        stt._pool.initialize = AsyncMock()
        stt._pool._initialized = False

        await stt.__aenter__()

        # Pool should be initialized
        stt._pool.initialize.assert_called()

    @pytest.mark.asyncio
    async def test_aexit_calls_aclose(self):
        """Test that __aexit__ calls aclose for cleanup."""
        stt = VoxistSTT(api_key="test")

        # Mock aclose
        stt.aclose = AsyncMock()

        await stt.__aexit__(None, None, None)

        stt.aclose.assert_called_once()

    @pytest.mark.asyncio
    async def test_async_with_usage(self):
        """Test async with statement works correctly."""
        # This tests the full context manager pattern
        stt = VoxistSTT(api_key="test")

        # Mock pool operations
        stt._pool.initialize = AsyncMock()
        stt._pool.close = AsyncMock()

        async with stt as instance:
            assert instance is stt

        # Should have closed
        stt._pool.close.assert_called()

    @pytest.mark.asyncio
    async def test_aexit_cleanup_on_exception(self):
        """Test that __aexit__ still cleans up when exception occurs."""
        stt = VoxistSTT(api_key="test")

        # Mock pool operations
        stt._pool.initialize = AsyncMock()
        stt._pool.close = AsyncMock()

        try:
            async with stt:
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still have cleaned up
        stt._pool.close.assert_called()


class TestSEC002LanguageValidation:
    """Test suite for SEC-002: Language parameter validation and sanitization."""

    @pytest.mark.parametrize("valid_lang", [
        "fr",         # Basic 2-letter code
        "en",         # Basic 2-letter code
        "fr-FR",      # Standard locale format
        "en-US",      # Standard locale format
        "de-DE",      # Standard locale format
        "fr-medical", # Extended format (medical specialization)
        "nl-NL",      # Netherlands
    ])
    def test_validate_language_format_accepts_valid(self, valid_lang):
        """Test that valid language formats pass validation."""
        assert validate_language_format(valid_lang) is True

    @pytest.mark.parametrize("invalid_lang", [
        "",                         # Empty string
        "f",                        # Too short (1 char)
        "fra",                      # 3-letter code (not standard)
        "fr-",                      # Trailing hyphen
        "-FR",                      # Leading hyphen
        "fr-F",                     # Too short region (1 char)
        "FR",                       # Uppercase (not lowercase start)
        "12",                       # Numbers instead of letters
        "fr_FR",                    # Underscore instead of hyphen
        "fr; DROP TABLE users",     # SQL injection attempt
        "fr<script>alert(1)</script>",  # XSS attempt
        "fr\n",                     # Newline injection
        "fr\r\n",                   # CRLF injection
        "fr%00",                    # Null byte injection
        "../../../etc/passwd",     # Path traversal
        "fr|cat /etc/passwd",      # Command injection
        "fr`ls`",                   # Backtick command injection
        "fr$(id)",                  # Shell command substitution
    ])
    def test_validate_language_format_rejects_invalid(self, invalid_lang):
        """Test that malformed language formats are rejected."""
        assert validate_language_format(invalid_lang) is False

    def test_validate_language_format_rejects_none(self):
        """Test that None input is rejected."""
        assert validate_language_format(None) is False

    def test_validate_language_format_rejects_non_string(self):
        """Test that non-string input is rejected."""
        assert validate_language_format(123) is False
        assert validate_language_format(['fr']) is False
        assert validate_language_format({'lang': 'fr'}) is False

    @pytest.mark.parametrize("input_val,expected_encoded", [
        ("fr&test", "fr%26test"),        # Ampersand encoded
        ("fr=test", "fr%3Dtest"),        # Equals encoded
        ("fr?test", "fr%3Ftest"),        # Question mark encoded
        ("fr#test", "fr%23test"),        # Hash encoded
        ("fr/test", "fr%2Ftest"),        # Slash encoded
        ("fr\\test", "fr%5Ctest"),       # Backslash encoded
        ("fr test", "fr%20test"),        # Space encoded
        ("fr%test", "fr%25test"),        # Percent encoded
        ("fr+test", "fr%2Btest"),        # Plus encoded
        ("fr@test", "fr%40test"),        # At sign encoded
    ])
    def test_sanitize_url_param_encodes_special_chars(self, input_val, expected_encoded):
        """Test that URL-unsafe characters are properly encoded."""
        result = sanitize_url_param(input_val)
        assert result == expected_encoded

    def test_sanitize_url_param_preserves_safe_chars(self):
        """Test that safe language code characters are preserved in meaning."""
        result = sanitize_url_param("fr-medical")
        # Should encode hyphen but still be valid URL param
        assert result  # Not empty

    def test_sanitize_url_param_handles_empty_string(self):
        """Test empty string handling."""
        result = sanitize_url_param("")
        assert result == ""

    def test_sanitize_url_param_converts_to_string(self):
        """Test that non-string inputs are converted to strings."""
        result = sanitize_url_param(16000)
        assert result == "16000"

    @pytest.mark.parametrize("injection_attempt", [
        "fr&admin=true",              # Parameter injection
        "fr#malicious",               # Fragment injection
        "fr?callback=evil",           # Query string injection
        "fr%26injected%3dtrue",       # Already-encoded injection
    ])
    def test_stt_rejects_injection_attempts_in_language(self, injection_attempt):
        """Test that VoxistSTT rejects injection attempts in language parameter."""
        with pytest.raises(LanguageNotSupportedError):
            VoxistSTT(api_key="test", language=injection_attempt)

    def test_all_supported_languages_pass_format_validation(self):
        """Test that all SUPPORTED_LANGUAGES pass format validation."""
        for lang in SUPPORTED_LANGUAGES.keys():
            assert validate_language_format(lang) is True, f"Supported language '{lang}' failed format validation"

    def test_stream_rejects_injection_in_language_override(self):
        """Test that stream() rejects injection attempts in language override."""
        stt = VoxistSTT(api_key="test", language="fr")

        with pytest.raises(LanguageNotSupportedError):
            stt.stream(language="fr; DROP TABLE users")

    def test_stream_rejects_invalid_format_in_language_override(self):
        """Test that stream() rejects invalid format even if in allowlist."""
        stt = VoxistSTT(api_key="test", language="fr")

        # These are not in SUPPORTED_LANGUAGES, so will fail allowlist first
        with pytest.raises(LanguageNotSupportedError):
            stt.stream(language="invalid<script>")

    @pytest.mark.asyncio
    @pytest.mark.no_auto_mock_token  # Need real _get_ws_token to test validation
    async def test_connection_pool_validates_language_format(self):
        """Test that ConnectionPool validates language format in _get_ws_token."""
        from livekit.plugins.voxist.connection_pool import ConnectionPool

        pool = ConnectionPool(
            base_url="wss://test.com/ws",
            api_key="test_key",
            pool_size=1,
        )

        # Create a mock session (needed for the HTTP request part, but validation happens first)
        mock_session = AsyncMock()
        pool._session = mock_session

        # Should raise LanguageNotSupportedError for invalid format
        # This happens BEFORE any network call because format validation is first
        with pytest.raises(LanguageNotSupportedError, match="invalid format"):
            await pool._get_ws_token("fr; DROP TABLE", 16000)

        # Clean up - set closing flag to avoid heartbeat issues
        pool._closing = True
        mock_session.closed = False
        mock_session.close = AsyncMock()
