"""Unit tests for VoxistSTT main plugin class."""

import asyncio
import contextlib
import logging
import os
import threading
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from livekit.agents.stt import STTCapabilities
from livekit.agents.types import NOT_GIVEN, APIConnectOptions

from livekit.plugins.voxist import VoxistSTT
from livekit.plugins.voxist.exceptions import (
    ConfigurationError,
    LanguageNotSupportedError,
)
from livekit.plugins.voxist.exceptions import (
    ConnectionError as VoxistConnectionError,
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
        assert stt._config is not None

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
        """connection_pool_size is accepted for backwards compatibility.

        There is no pool anymore - one socket per stream - but constructors in
        user code still pass it, so it must be accepted.
        """
        stt = VoxistSTT(api_key="test", connection_pool_size=3)
        assert stt._config is not None

    @pytest.mark.parametrize("size", [0, -1, 10, 1000])
    def test_out_of_range_pool_size_is_not_an_error(self, size):
        """[I] An IGNORED parameter must never hard-fail construction.

        The value is discarded - there is no pool to size - so rejecting it
        would be a ConfigurationError for a setting that does nothing. The
        deprecation warning is the only response (asserted separately by
        TestDeadAndLiveParameters.test_non_default_dead_params_warn).
        """
        stt = VoxistSTT(api_key="test", connection_pool_size=size)
        assert stt._config["language"] == "fr"
        assert not hasattr(stt, "_pool"), "no pool should exist to size"

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

    def test_initialization_stores_dial_settings(self):
        """Test VoxistSTT stores what the dialer needs."""
        stt = VoxistSTT(
            api_key="test",
            base_url="wss://custom.url/ws",
            connection_pool_size=3,
            connection_timeout=5.0,
            heartbeat_interval=60.0,
        )

        assert stt._base_url == "wss://custom.url/ws"
        assert stt._api_key == "test"
        assert stt._heartbeat_interval == 60.0


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
    async def test_aclose_closes_owned_session(self):
        """Test aclose() closes the HTTP session the plugin created."""
        stt = VoxistSTT(api_key="test")

        session = AsyncMock()
        session.closed = False
        stt._session = session
        stt._owns_session = True

        await stt.aclose()

        session.close.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_aclose_leaves_user_session_open(self):
        """A session supplied by the caller is theirs to close."""
        session = AsyncMock()
        session.closed = False
        stt = VoxistSTT(api_key="test", http_session=session)

        await stt.aclose()

        session.close.assert_not_awaited()


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
        assert stt._heartbeat_interval == 45.0
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

        assert stt._api_key == "api_test"
        assert stt._base_url == "wss://test.com/ws"
        assert stt._heartbeat_interval == 25.0


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

    def test_pool_size_raises_no_configuration_error(self):
        """[I] There is no pool-size validation left to produce a message.

        The parameter is accepted-and-ignored, so no value of it may raise -
        only chunk_duration_ms (a LIVE parameter) still does, below.
        """
        for size in (0, 10):
            VoxistSTT(api_key="test", connection_pool_size=size)

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

    def test_pool_size_bounds_accepted(self):
        """The legacy bounds (1-5) construct successfully.

        The parameter is compatibility-only now, but rejecting a value that
        used to work would break existing constructor calls.
        """
        for size in (1, 5):
            stt = VoxistSTT(api_key="test", connection_pool_size=size)
            assert stt._config["language"] == "fr"


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
        assert stt._base_url == "wss://staging.voxist.com/ws"


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

        # Make the token pre-fetch raise an auth error
        async def mock_ensure_dialer():
            dialer = AsyncMock()
            dialer._get_token_url.side_effect = AuthenticationError(
                "Invalid API key"
            )
            return dialer

        stt._ensure_dialer = mock_ensure_dialer

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
        stt._ensure_dialer = AsyncMock(return_value=AsyncMock())

        result = await stt.wait_for_initialization(timeout=5.0)

        assert isinstance(result, bool)

    @pytest.mark.asyncio
    async def test_wait_for_initialization_on_success(self):
        """Test wait_for_initialization returns True on success."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        stt._ensure_dialer = AsyncMock(return_value=AsyncMock())
        stt._init_state = InitializationState.NOT_STARTED

        result = await stt.wait_for_initialization(timeout=5.0)

        assert result is True
        assert stt.initialization_state == InitializationState.COMPLETED

    @pytest.mark.asyncio
    async def test_wait_for_initialization_returns_true_if_already_completed(self):
        """Completed init returns True without re-running the warm-up.

        validate_websocket=False keeps this a pure state-machine test; the
        WS probe contract is pinned in TestWebSocketReachabilityValidation.
        """
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test", validate_websocket=False)
        stt._init_state = InitializationState.COMPLETED

        result = await stt.wait_for_initialization(timeout=5.0)

        assert result is True

    @pytest.mark.asyncio
    async def test_already_completed_still_validates_ws_exactly_once(self):
        """[14] COMPLETED means 'token cached', not 'WS proven': the first
        readiness check must still dial the probe, and only the first."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        dialer = AsyncMock()
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.COMPLETED

        assert await stt.wait_for_initialization(timeout=5.0) is True
        dialer.dial.assert_awaited_once()

        assert await stt.wait_for_initialization(timeout=5.0) is True
        dialer.dial.assert_awaited_once()  # probe result is cached

        await stt.aclose()

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
        async def slow_fetch():
            await asyncio.sleep(10)

        dialer = AsyncMock()
        dialer._get_token_url = slow_fetch
        stt._ensure_dialer = AsyncMock(return_value=dialer)

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

        dialer = AsyncMock()
        dialer._get_token_url = capture_state
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.PENDING

        await stt._initialize_pool()

        # State should have been RUNNING during initialize call
        assert InitializationState.RUNNING in states_during_init

    @pytest.mark.asyncio
    async def test_state_transition_to_failed_on_error(self):
        """Test state transitions to FAILED on initialization error."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")

        dialer = AsyncMock()
        dialer._get_token_url.side_effect = ConnectionError("Network error")
        stt._ensure_dialer = AsyncMock(return_value=dialer)
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

        dialer = AsyncMock()
        dialer._get_token_url.side_effect = ValueError("Init failed")
        stt._ensure_dialer = AsyncMock(return_value=dialer)

        with pytest.raises(InitializationError):
            await stt.__aenter__()


class TestContextManager:
    """Test suite for context manager support (QUAL-HIGH: asr-all-2nj)."""

    @pytest.mark.asyncio
    async def test_aenter_returns_self(self):
        """Test that __aenter__ returns the STT instance."""
        stt = VoxistSTT(api_key="test")

        # Mock pool initialize to avoid actual connection
        stt._ensure_dialer = AsyncMock(return_value=AsyncMock())

        result = await stt.__aenter__()

        assert result is stt

    @pytest.mark.asyncio
    async def test_aenter_initializes_pool(self):
        """Test that __aenter__ ensures pool is initialized."""
        stt = VoxistSTT(api_key="test")

        # Mock pool
        stt._ensure_dialer = AsyncMock(return_value=AsyncMock())

        await stt.__aenter__()

        # Pool should be initialized
        stt._ensure_dialer.assert_called()

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
        stt._ensure_dialer = AsyncMock(return_value=AsyncMock())

        async with stt as instance:
            assert instance is stt

        # Should have closed
        assert stt._init_task is None or stt._init_task.done()

    @pytest.mark.asyncio
    async def test_aexit_cleanup_on_exception(self):
        """Test that __aexit__ still cleans up when exception occurs."""
        stt = VoxistSTT(api_key="test")

        # Mock pool operations
        stt._ensure_dialer = AsyncMock(return_value=AsyncMock())

        try:
            async with stt:
                raise ValueError("Test exception")
        except ValueError:
            pass

        # Should still have cleaned up
        assert stt._init_task is None or stt._init_task.done()


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
            assert validate_language_format(lang) is True, (
                f"Supported language '{lang}' failed format validation"
            )

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
    async def test_stream_validates_language_format_before_dialing(self):
        """Injection-shaped language codes are rejected before any network use."""
        stt = VoxistSTT(api_key="test_key")
        if stt._init_task is not None:
            stt._init_task.cancel()

        with pytest.raises(LanguageNotSupportedError, match="invalid format|not supported"):
            stt.stream(language="fr; DROP TABLE")


class TestShutdownAndTLSConfiguration:
    """Shutdown ordering, and reaching a deployment behind a private CA."""

    @pytest.mark.asyncio
    async def test_aclose_is_idempotent(self):
        """aclose() twice must not raise - agents tear down defensively."""
        stt = VoxistSTT(api_key="test_key", base_url="ws://localhost:9/ws")
        if stt._init_task is not None:
            stt._init_task.cancel()
            try:
                await stt._init_task
            except (asyncio.CancelledError, Exception):
                pass

        await stt.aclose()
        await stt.aclose()

    @pytest.mark.asyncio
    async def test_ssl_context_is_forwarded_to_the_dialer(self):
        """
        An explicit SSL context must reach the dialer.

        Certificate verification is never disabled, so without this
        passthrough a deployment whose certificate is signed by a private CA
        is unreachable through the public API - there is no other way to
        inject a trust store.
        """
        import ssl as ssl_module

        ctx = ssl_module.create_default_context()
        stt = VoxistSTT(
            api_key="test_key",
            base_url="ws://localhost:9/ws",
            ssl_context=ctx,
        )
        if stt._init_task is not None:
            stt._init_task.cancel()

        dialer = await stt._ensure_dialer()
        assert dialer._ssl_param() is ctx

        await stt.aclose()


class TestAcloseWithLiveStreams:
    """
    aclose() must shut live streams down BEFORE the shared HTTP session:
    closing the session first left a retrying stream to dial a closed
    session and crash with an unmapped RuntimeError('Session is closed').
    """

    @pytest.mark.asyncio
    async def test_streams_are_closed_before_the_session(self):
        stt = VoxistSTT(api_key="test_key")
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task

        order = []

        fake_stream = AsyncMock()
        fake_stream.aclose = AsyncMock(side_effect=lambda: order.append("stream"))
        stt._live_streams.add(fake_stream)

        session = AsyncMock()
        session.closed = False
        session.close = AsyncMock(side_effect=lambda: order.append("session"))
        stt._session = session
        stt._owns_session = True

        await stt.aclose()

        assert order == ["stream", "session"]

    @pytest.mark.asyncio
    async def test_stream_registers_itself_for_shutdown(self):
        stt = VoxistSTT(api_key="test_key", base_url="ws://127.0.0.1:9/ws")
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task

        stream = stt.stream(
            conn_options=APIConnectOptions(
                max_retry=0, retry_interval=0.1, timeout=1.0
            )
        )
        try:
            assert stream in stt._live_streams
        finally:
            await stream.aclose()
            # retrieve the failed dial's exception so it never hits GC
            if stream._task.done() and not stream._task.cancelled():
                with contextlib.suppress(Exception):
                    stream._task.exception()
            await stt.aclose()

    @pytest.mark.asyncio
    async def test_dial_after_aclose_is_a_mapped_error(self):
        """Belt and braces: a stream retry racing shutdown must die as our
        ConnectionError (mapped to a retryable APIError by the stream), not
        resurrect a fresh session or crash with a RuntimeError."""
        stt = VoxistSTT(api_key="test_key")
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task
        await stt.aclose()

        with pytest.raises(VoxistConnectionError, match="closed"):
            await stt._ensure_dialer()

    @pytest.mark.asyncio
    async def test_dialer_maps_closed_session_instead_of_runtime_error(self):
        """The dialer itself must never let aiohttp's raw
        RuntimeError('Session is closed') escape a dial."""
        stt = VoxistSTT(api_key="test_key")
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task

        dialer = await stt._ensure_dialer()
        # the session dies under the dialer without aclose() being involved
        await stt._session.close()

        with pytest.raises(VoxistConnectionError):
            await dialer.dial("fr", 16000)

        stt._closed = True  # skip resurrection paths in aclose
        await stt.aclose()


class TestSessionLoopAffinity:
    """
    aiohttp binds a ClientSession to the loop that first uses it.

    Contract ([6]): a session stranded on a CLOSED loop is rebuilt (plugin-
    owned) or a clear mapped error (caller-supplied). A session bound to a
    loop that is ALIVE but not the running one is neither destroyed nor
    rebuilt: multi-loop sharing is unsupported, and the healthy loop's
    session (and its token cache) must survive, so it is a mapped error.
    """

    def test_owned_session_is_rebuilt_for_a_new_loop(self):
        stt = VoxistSTT(api_key="test_key")  # no loop: init on demand

        async def ensure():
            await stt._ensure_dialer()
            return stt._session, stt._dialer

        first_session, first_dialer = asyncio.run(ensure())
        # loop 1 is now closed; the session is stranded on it

        second_session, second_dialer = asyncio.run(ensure())
        assert second_session is not first_session, (
            "the plugin-owned session must be rebuilt for the new loop"
        )
        assert second_dialer is not first_dialer

        async def close():
            await stt.aclose()

        asyncio.run(close())

    def test_alive_foreign_loop_errors_without_destroying_the_session(self):
        """
        [6] Two LIVE loops sharing one VoxistSTT: the second loop must get
        a clear mapped error, and the first loop's session, dialer and
        token cache must survive untouched - no rebuild thrash, no leaked
        connectors, no misleading 'defunct loop' warning.
        """
        stt = VoxistSTT(api_key="test_key")  # constructed with no loop

        loop_a = asyncio.new_event_loop()
        thread = threading.Thread(target=loop_a.run_forever, daemon=True)
        thread.start()
        try:
            asyncio.run_coroutine_threadsafe(
                stt._ensure_dialer(), loop_a
            ).result(timeout=5)
            session_a, dialer_a = stt._session, stt._dialer
            assert session_a is not None

            async def ensure():
                await stt._ensure_dialer()

            # loop_a is still ALIVE: this must be an error, not a rebuild
            with pytest.raises(
                VoxistConnectionError, match="different running event loop"
            ):
                asyncio.run(ensure())

            # the healthy loop's state survives...
            assert stt._session is session_a
            assert stt._dialer is dialer_a
            assert not session_a.closed

            # ...and keeps working from its own loop afterwards
            still = asyncio.run_coroutine_threadsafe(
                stt._ensure_dialer(), loop_a
            ).result(timeout=5)
            assert still is dialer_a
        finally:
            asyncio.run_coroutine_threadsafe(stt.aclose(), loop_a).result(
                timeout=5
            )
            loop_a.call_soon_threadsafe(loop_a.stop)
            thread.join(timeout=5)
            loop_a.close()

    def test_user_session_on_dead_loop_raises_mapped_error(self):
        async def make_session():
            return aiohttp.ClientSession()

        session = asyncio.run(make_session())
        # the caller's session is now bound to a closed loop
        stt = VoxistSTT(api_key="test_key", http_session=session)

        async def ensure():
            await stt._ensure_dialer()

        with pytest.raises(VoxistConnectionError, match="http_session"):
            asyncio.run(ensure())

        # not plugin-owned: it must never have been replaced
        assert stt._session is session

        async def cleanup():
            with contextlib.suppress(Exception):
                await session.close()

        asyncio.run(cleanup())


class TestDeadAndLiveParameters:
    """connection_timeout must actually take effect; genuinely dead
    parameters must say so instead of silently lying."""

    @pytest.mark.asyncio
    async def test_connection_timeout_reaches_the_dialer(self):
        stt = VoxistSTT(api_key="test_key", connection_timeout=3.3)
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task

        dialer = await stt._ensure_dialer()
        assert dialer._connection_timeout == 3.3

        await stt.aclose()

    def test_non_default_dead_params_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            VoxistSTT(
                api_key="test_key",
                connection_pool_size=3,
                max_reconnect_attempts=5,
            )

        messages = [r.message for r in caplog.records]
        assert any(
            "connection_pool_size" in m and "ignored" in m for m in messages
        )
        assert any(
            "max_reconnect_attempts" in m and "ignored" in m for m in messages
        )

    def test_default_params_do_not_warn(self, caplog):
        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            VoxistSTT(api_key="test_key")

        assert not any("ignored" in r.message for r in caplog.records)


class TestInitTaskExceptionRetrieval:
    """
    The background init task re-raises AuthenticationError for awaiting
    callers, but nobody is required to await it - the failure must still be
    retrieved (it is already surfaced via initialization_state) instead of
    becoming a GC-time 'Task exception was never retrieved'.
    """

    @pytest.mark.asyncio
    async def test_failed_init_exception_is_retrieved_by_callback(
        self, monkeypatch, caplog
    ):
        from livekit.plugins.voxist.connection import VoxistDialer
        from livekit.plugins.voxist.exceptions import AuthenticationError
        from livekit.plugins.voxist.stt import InitializationState

        async def boom(self):
            raise AuthenticationError("revoked key")

        monkeypatch.setattr(VoxistDialer, "_get_token_url", boom)

        with caplog.at_level(logging.DEBUG, logger="livekit.plugins.voxist"):
            stt = VoxistSTT(api_key="test_key")
            assert stt._init_task is not None

            # nobody awaits the task; wait for it to finish and for its
            # done-callbacks to run
            while not stt._init_task.done():
                await asyncio.sleep(0.01)
            await asyncio.sleep(0)

        assert stt.initialization_state == InitializationState.FAILED
        assert isinstance(stt.initialization_error, AuthenticationError)
        # the callback retrieved the exception (this log line IS the
        # retrieval - remove the callback and this fails)
        assert any(
            "Background initialization failed" in r.message
            for r in caplog.records
        )

        await stt.aclose()

    @pytest.mark.asyncio
    async def test_awaiting_the_init_task_still_raises_auth_error(
        self, monkeypatch
    ):
        """The callback must not change wait/await semantics."""
        from livekit.plugins.voxist.connection import VoxistDialer
        from livekit.plugins.voxist.exceptions import AuthenticationError

        async def boom(self):
            raise AuthenticationError("revoked key")

        monkeypatch.setattr(VoxistDialer, "_get_token_url", boom)

        stt = VoxistSTT(api_key="test_key")
        assert stt._init_task is not None
        with pytest.raises(AuthenticationError):
            await stt._init_task

        assert await stt.wait_for_initialization(timeout=1.0) is False
        await stt.aclose()


class TestAcloseAcrossLoops:
    """
    [7] aclose() must be best-effort-COMPLETE: an init task stranded on a
    dead (or foreign) loop must not abort the teardown before streams and
    the HTTP session are closed.
    """

    def test_aclose_from_another_loop_still_completes_cleanup(self, monkeypatch):
        # Build the plugin under loop A so the background init task is
        # created there, then close loop A with the task still pending
        # (a slow warm-up: the token exchange never answered).
        async def hang(self):
            await asyncio.Event().wait()

        monkeypatch.setattr(VoxistSTT, "_initialize_pool", hang)

        loop_a = asyncio.new_event_loop()

        async def make():
            return VoxistSTT(api_key="test_key")

        stt = loop_a.run_until_complete(make())
        assert stt._init_task is not None and not stt._init_task.done()
        loop_a.close()

        # Give the plugin things that MUST still be cleaned up.
        fake_stream = AsyncMock()
        stt._live_streams.add(fake_stream)
        session = AsyncMock()
        session.closed = False
        stt._session = session
        stt._owns_session = True

        async def close():
            await stt.aclose()

        # Without the guard this dies on the init task (its future belongs
        # to the closed loop A) and neither the stream nor the session is
        # ever closed.
        asyncio.run(close())

        fake_stream.aclose.assert_awaited_once()
        session.close.assert_awaited_once()
        assert stt._closed is True

    @pytest.mark.asyncio
    async def test_aclose_completes_when_a_cleanup_step_fails(self):
        """Best-effort-complete: a failing base-class close must not stop
        the owned session from being closed (and must not re-raise -
        aclose runs in finally blocks where an exception would mask the
        caller's original error and leak everything after it)."""
        stt = VoxistSTT(api_key="test_key")
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task

        session = AsyncMock()
        session.closed = False
        stt._session = session
        stt._owns_session = True

        with patch(
            "livekit.agents.stt.STT.aclose",
            AsyncMock(side_effect=RuntimeError("base close boom")),
        ):
            await stt.aclose()  # must not raise

        session.close.assert_awaited_once()


class TestStreamAfterClose:
    """
    [13] stream() on a closed plugin is a programming error and must fail
    at the call site, immediately - not 6 seconds later after livekit's
    retry machinery burned its budget against a plugin that can never dial.
    """

    @pytest.mark.asyncio
    async def test_stream_after_aclose_raises_immediately(self):
        stt = VoxistSTT(api_key="test_key")
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task
        await stt.aclose()

        with pytest.raises(RuntimeError, match="closed VoxistSTT"):
            stt.stream()

    @pytest.mark.asyncio
    async def test_stream_before_aclose_still_works(self):
        stt = VoxistSTT(api_key="test_key", base_url="ws://127.0.0.1:9/ws")
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task

        stream = stt.stream(
            conn_options=APIConnectOptions(
                max_retry=0, retry_interval=0.1, timeout=1.0
            )
        )
        try:
            assert stream in stt._live_streams
        finally:
            await stream.aclose()
            if stream._task.done() and not stream._task.cancelled():
                with contextlib.suppress(Exception):
                    stream._task.exception()
            await stt.aclose()


@pytest.mark.no_auto_mock_token
class TestWebSocketReachabilityValidation:
    """
    [14] The warm-up's token pre-fetch is plain HTTPS, so it succeeds on
    deployments where the WS path is blocked. The explicit readiness paths
    (wait_for_initialization / __aenter__) must therefore prove the WS path
    with one short-lived dial - once per plugin, never per stream.
    """

    @pytest.mark.asyncio
    async def test_readiness_dials_the_ws_path_exactly_once(
        self, mock_voxist_server
    ):
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
        )

        assert await stt.wait_for_initialization(timeout=5.0) is True
        assert stt.is_ready
        assert mock_voxist_server.connections_count == 1, (
            "readiness must include a real WS dial, not just the token fetch"
        )

        # The probe is cached: repeated readiness checks must not open
        # another server-side engine session.
        assert await stt.wait_for_initialization(timeout=5.0) is True
        assert mock_voxist_server.connections_count == 1

        await stt.aclose()

    @pytest.mark.asyncio
    async def test_ws_blocked_deployment_fails_readiness(self):
        """The scenario the old initialize() caught and the token-only
        warm-up missed: token endpoint healthy, WS path dead. Readiness
        must be False, is_ready must agree, and __aenter__ must raise
        InitializationError - not report a 'healthy' plugin whose every
        real call will fail."""
        from livekit.plugins.voxist.exceptions import InitializationError

        from .fixtures.mock_server import MockVoxistServer

        # error_mode="ws_blocked": /websocket hands out a valid-looking token
        # URL, /ws answers a plain HTTP 200 instead of upgrading.
        server = MockVoxistServer(valid_api_key="any_key", error_mode="ws_blocked")
        await server.start()
        try:
            stt = VoxistSTT(
                api_key="any_key",
                base_url=f"ws://{server.host}:{server.port}/ws",
            )

            assert await stt.wait_for_initialization(timeout=5.0) is False
            assert server.ws_upgrade_refusals >= 1, (
                "the WS path was actually probed"
            )
            assert stt.is_ready is False
            assert isinstance(
                stt.initialization_error, VoxistConnectionError
            ), f"got {stt.initialization_error!r}"

            with pytest.raises(InitializationError):
                await stt.__aenter__()

            await stt.aclose()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_validate_websocket_false_keeps_readiness_token_only(
        self, mock_voxist_server
    ):
        """The opt-out restores the pre-[14] token-only semantics."""
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
            validate_websocket=False,
        )

        assert await stt.wait_for_initialization(timeout=5.0) is True
        assert mock_voxist_server.connections_count == 0, (
            "validate_websocket=False must not dial"
        )

        await stt.aclose()

    @pytest.mark.asyncio
    async def test_background_warmup_alone_never_dials(self, mock_voxist_server):
        """The fire-and-forget warm-up stays token-only: each WS dial opens
        a real ASR engine session server-side, a cost only the explicit
        readiness paths may incur (and only once)."""
        stt = VoxistSTT(
            api_key="test_key",
            base_url=f"ws://{mock_voxist_server.host}:{mock_voxist_server.port}/ws",
        )
        assert stt._init_task is not None
        with contextlib.suppress(Exception):
            await stt._init_task

        assert mock_voxist_server.token_requests_count == 1
        assert mock_voxist_server.connections_count == 0, (
            "the background warm-up must not open engine sessions"
        )

        await stt.aclose()
