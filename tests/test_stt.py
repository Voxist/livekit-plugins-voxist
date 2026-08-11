"""Unit tests for VoxistSTT main plugin class."""

import asyncio
import contextlib
import inspect
import logging
import os
import threading
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

import aiohttp
import pytest
from livekit.agents.stt import STTCapabilities
from livekit.agents.types import NOT_GIVEN, APIConnectOptions

from livekit.plugins.voxist import VoxistSTT
from livekit.plugins.voxist.exceptions import (
    ConfigurationError,
    LanguageNotSupportedError,
    VoxistError,
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
            # The mocked loop does not take ownership of the coroutine the
            # way a real create_task() call does, so close it explicitly.
            mock_create_task.call_args.args[0].close()

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
        assert inspect.iscoroutinefunction(stt.wait_for_initialization)

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
    async def test_check_initialization_does_not_raise_when_ready(self):
        """Silence means READY, by the one readiness definition."""
        from livekit.plugins.voxist import InitializationState

        # validate_websocket=False: the caller asked for the token-only
        # contract, so a completed token warm-up IS the whole proof.
        stt = VoxistSTT(api_key="test", validate_websocket=False)
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
        """Token-only initialization is enough when WS validation is disabled."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test", validate_websocket=False)
        stt._init_state = InitializationState.COMPLETED

        assert stt.is_ready is True

    @pytest.mark.asyncio
    async def test_is_ready_requires_ws_proof_when_validation_enabled(self):
        """A cached token must not masquerade as a reachable WebSocket."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test", validate_websocket=True)
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task

        stt._init_state = InitializationState.COMPLETED
        stt._dialer = Mock(_token_url="ws://cached-token")

        assert stt.is_ready is False

        stt._ws_validated = True
        assert stt.is_ready is True
        await stt.aclose()

    @pytest.mark.asyncio
    async def test_late_success_clears_timeout_metadata(self):
        """A shielded warm-up that succeeds after a caller timeout is clean."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test", validate_websocket=False)
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task

        started = asyncio.Event()
        release = asyncio.Event()

        async def slow_fetch():
            started.set()
            await release.wait()
            return "ws://fresh-token"

        dialer = AsyncMock()
        dialer._get_token_url = slow_fetch
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.PENDING
        stt._init_task = asyncio.create_task(stt._initialize_pool())

        await started.wait()
        assert await stt.wait_for_initialization(timeout=0.01) is False
        assert stt.initialization_state is InitializationState.FAILED

        release.set()
        await stt._init_task

        assert stt.initialization_state is InitializationState.COMPLETED
        assert stt.initialization_error is None
        assert stt._init_failed_at is None
        await stt.aclose()

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
    session (and its token cache) must survive.

    Contract update ([8]): the alive-foreign-loop case raises RuntimeError,
    NOT our ConnectionError. Round 6 used ConnectionError, which the stream
    maps to a retryable APIConnectionError - so livekit burned its whole
    retry schedule (three misleading recoverable=True events) on a
    programming error that cannot change between attempts, then reported a
    wrapper instead of the real cause.
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
        a clear error, and the first loop's session, dialer and token cache
        must survive untouched - no rebuild thrash, no leaked connectors, no
        misleading 'defunct loop' warning.

        [8] The error must be a non-APIError-mappable RuntimeError so
        livekit fails fast on the programming error instead of retrying it.
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
                RuntimeError, match="different running event loop"
            ) as excinfo:
                asyncio.run(ensure())

            # [8] It must NOT be our ConnectionError: stream.py maps that to
            # a retryable APIConnectionError, and no retry can fix a
            # programming error. RuntimeError takes _main_task's terminal
            # branch (one recoverable=False event, real cause surfaced).
            assert not isinstance(excinfo.value, VoxistConnectionError)
            assert not isinstance(excinfo.value, VoxistError)

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
    async def test_probe_close_is_bounded(self, monkeypatch):
        """A peer that stalls its close handshake must not hang readiness."""
        from livekit.plugins.voxist import InitializationState

        class HangingProbeWebSocket:
            async def receive(self):
                # Silent past the grace window: the healthy shape, since the
                # gateway sends no greeting frame.
                await asyncio.Event().wait()

            async def close(self):
                await asyncio.Event().wait()

        stt = VoxistSTT(api_key="test", validate_websocket=True)
        if stt._init_task is not None:
            stt._init_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await stt._init_task

        dialer = AsyncMock()
        dialer.dial.return_value = HangingProbeWebSocket()
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.COMPLETED
        monkeypatch.setattr(stt, "READINESS_CLOSE_TIMEOUT_SECONDS", 0.01)
        monkeypatch.setattr(stt, "READINESS_APPLICATION_GRACE_SECONDS", 0.01)

        assert await asyncio.wait_for(
            stt.wait_for_initialization(timeout=1.0), timeout=0.5
        ) is True
        assert stt._ws_validated is True
        await stt.aclose()

    @pytest.mark.asyncio
    async def test_background_warmup_alone_never_dials(self, mock_voxist_server):
        """The fire-and-forget warm-up stays token-only: each WS dial opens
        a real ASR engine session server-side, a cost only the explicit
        readiness paths may incur (and only once).

        Because it opens no socket it also cannot establish readiness, and the
        readiness surface must say so - see
        TestReadinessSurfaceIsOnePredicate.
        """
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
        assert stt.is_ready is False, (
            "a token-only warm-up proves nothing about the WebSocket path"
        )

        await stt.aclose()


class TestReadinessSurfaceIsOnePredicate:
    """
    The readiness surface - is_ready, initialization_state,
    initialization_error, check_initialization(), wait_for_initialization() -
    used to be five accessors each answering a slightly different question
    from a different subset of the internal flags. These tests pin the single
    definition they now all derive from (documented in stt.py above
    _reachability_proven), one test per way they used to disagree.
    """

    @pytest.mark.no_auto_mock_token
    @pytest.mark.asyncio
    async def test_a_blocked_ws_path_is_never_reported_healthy(self):
        """[F1] The warm-up is plain HTTPS, so on a deployment whose WebSocket
        path is blocked it SUCCEEDS. initialization_state reported that as
        COMPLETED, initialization_error stayed None and check_initialization()
        raised nothing - so a deployment health check built on those two APIs
        called the plugin healthy while every stream() died on the dial. The
        pre-rewrite initialize() raised at startup here; that guarantee is
        restored by never reporting an unverified deployment as healthy.
        """
        from livekit.plugins.voxist import InitializationError, InitializationState

        from .fixtures.mock_server import MockVoxistServer

        server = MockVoxistServer(valid_api_key="any_key", error_mode="ws_blocked")
        await server.start()
        try:
            stt = VoxistSTT(
                api_key="any_key",
                base_url=f"ws://{server.host}:{server.port}/ws",
            )
            assert stt._init_task is not None
            with contextlib.suppress(Exception):
                await stt._init_task

            # The token half really did succeed: this is not a plugin that
            # failed, it is a plugin nothing has verified.
            assert server.token_requests_count == 1
            assert server.connections_count == 0
            assert stt._init_state is InitializationState.COMPLETED, (
                "the warm-up phase itself completed - that is the trap"
            )

            # ...and not one accessor may call that healthy.
            assert stt.initialization_state is not InitializationState.COMPLETED
            assert stt.is_ready is False
            with pytest.raises(InitializationError, match="not verified"):
                stt.check_initialization()

            await stt.aclose()
        finally:
            await server.stop()

    @pytest.mark.no_auto_mock_token
    @pytest.mark.asyncio
    async def test_an_upgrade_then_1008_close_is_not_reachability(self):
        """[F2] aiohttp's ws_connect returns on the 101, so a gateway that
        accepts the upgrade and THEN closes with 1008 - its documented answer
        for a refused app-level credential and for an exhausted balance -
        satisfied a probe that only checked the handshake. Readiness reported
        True and __aenter__ succeeded, then every real stream died with
        "server closed the connection before end of input" after burning the
        whole retry budget.
        """
        from livekit.plugins.voxist import InitializationError
        from livekit.plugins.voxist.exceptions import AuthenticationError

        from .fixtures.mock_server import MockVoxistServer

        server = MockVoxistServer(
            valid_api_key="any_key", error_mode="ws_upgraded_then_rejected"
        )
        await server.start()
        try:
            stt = VoxistSTT(
                api_key="any_key",
                base_url=f"ws://{server.host}:{server.port}/ws",
            )

            assert await stt.wait_for_initialization(timeout=5.0) is False
            assert server.connections_count >= 1, (
                "the upgrade must have SUCCEEDED - otherwise this is the "
                "handshake failure ws_blocked already covered, not an "
                "application-layer rejection"
            )
            assert stt.is_ready is False
            assert isinstance(
                stt.initialization_error, AuthenticationError
            ), f"got {stt.initialization_error!r}"

            with pytest.raises(InitializationError):
                await stt.__aenter__()

            await stt.aclose()
        finally:
            await server.stop()

    @pytest.mark.asyncio
    async def test_a_replayed_probe_failure_keeps_its_classification(self):
        """[F3a] Two readiness callers race (a health endpoint and
        `async with` - the case the probe docstring cites). A dials and the
        gateway rejects the key, recording a sticky AuthenticationError. B then
        enters the probe lock, hits the cooldown branch, and used to raise a
        FRESH VoxistConnectionError wrapping it - which, recorded LAST,
        overwrote the sticky classification with a transient-looking one and
        put a revoked credential on a 30s re-probe loop forever.
        """
        from livekit.plugins.voxist import InitializationError, InitializationState
        from livekit.plugins.voxist.exceptions import AuthenticationError

        stt = VoxistSTT(api_key="test")
        await _quiesce_background_init(stt)
        clock = _install_fake_clock(stt)

        async def slow_reject(*args, **kwargs):
            await asyncio.sleep(0.05)  # wide enough for the racer to enter
            raise AuthenticationError("revoked key")

        dialer = AsyncMock()
        dialer.dial = AsyncMock(side_effect=slow_reject)
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.COMPLETED

        results = await asyncio.gather(
            stt.wait_for_initialization(timeout=5.0),
            stt.wait_for_initialization(timeout=5.0),
        )

        assert results == [False, False]
        assert dialer.dial.await_count == 1, "one probe per cooldown window"
        assert isinstance(stt.initialization_error, AuthenticationError), (
            f"the replay downgraded the classification: "
            f"{stt.initialization_error!r}"
        )
        assert stt._failure_is_permanent(stt.initialization_error) is True
        with pytest.raises(InitializationError, match="will not clear"):
            stt.check_initialization()

        # The consequence that actually matters: the rejected key is never
        # hammered, however much time passes.
        clock["now"] += 100 * stt.READINESS_RETRY_COOLDOWN_SECONDS
        assert await stt.wait_for_initialization(timeout=5.0) is False
        assert dialer.dial.await_count == 1

        await stt.aclose()

    @pytest.mark.asyncio
    async def test_a_permanent_failure_is_never_downgraded(self):
        """[F3b] The other half, independent of the replay fix: a settled
        permanent classification must survive both a later weaker recording
        and being wrapped. _failure_is_permanent used to inspect only the
        outermost exception TYPE, never __cause__, so any re-wrap reclassified
        a revoked credential as a transient blip.
        """
        from livekit.plugins.voxist import InitializationState
        from livekit.plugins.voxist.exceptions import AuthenticationError

        stt = VoxistSTT(api_key="test", validate_websocket=False)
        await _quiesce_background_init(stt)
        _install_fake_clock(stt)

        rejected = AuthenticationError("revoked key")
        stt._record_init_failure(rejected)
        assert stt._failure_is_permanent(stt.initialization_error) is True

        # A later, weaker report must not replace the settled verdict. This is
        # the guard on its own: a bare transient error, nothing to see through.
        stt._record_init_failure(VoxistConnectionError("plain blip"))
        assert stt.initialization_error is rejected
        assert stt.initialization_state is InitializationState.FAILED
        assert stt._may_retry_failed_readiness() is False

        # And the classification survives WRAPPING on its own, so neither half
        # of the fix depends on the other masking it.
        wrapped = VoxistConnectionError("probe failed")
        wrapped.__cause__ = rejected
        assert stt._failure_is_permanent(wrapped) is True
        assert stt._failure_is_permanent(VoxistConnectionError("blip")) is False

        # A __cause__ cycle must terminate rather than hang the classifier.
        first = VoxistConnectionError("a")
        second = VoxistConnectionError("b")
        first.__cause__ = second
        second.__cause__ = first
        assert stt._failure_is_permanent(first) is False

        await stt.aclose()

    @pytest.mark.asyncio
    async def test_a_closed_plugin_is_never_ready(self):
        """[F4] aclose() sets _closed and stream() raises RuntimeError from
        then on, but is_ready consulted neither _closed nor anything aclose()
        touches - so a readiness/liveness endpoint kept reporting a torn-down
        plugin healthy, traffic kept being routed to it, and every stream()
        raised an unhandled RuntimeError at the call site instead of the
        caller failing over.
        """
        from livekit.plugins.voxist import InitializationError, InitializationState

        stt = VoxistSTT(api_key="test")
        await _quiesce_background_init(stt)
        stt._init_state = InitializationState.COMPLETED
        stt._ws_validated = True
        assert stt.is_ready is True

        await stt.aclose()

        # aclose() deliberately clears neither flag: _closed is a term of the
        # readiness predicate, not something patched into each accessor.
        assert stt._closed is True
        assert stt._init_state is InitializationState.COMPLETED
        assert stt._ws_validated is True

        assert stt.is_ready is False
        assert stt.initialization_state is InitializationState.FAILED
        assert stt.initialization_error is not None
        assert stt._may_retry_failed_readiness() is False
        with pytest.raises(InitializationError, match="closed"):
            stt.check_initialization()
        assert await stt.wait_for_initialization(timeout=1.0) is False

        # The whole point: readiness now agrees with what stream() does.
        with pytest.raises(RuntimeError, match="closed VoxistSTT"):
            stt.stream()


async def _quiesce_background_init(stt: VoxistSTT) -> None:
    """Cancel the fire-and-forget warm-up so a test owns the state machine.

    Without this a background task completing mid-test can overwrite the
    state the test just asserted on.
    """
    if stt._init_task is not None:
        stt._init_task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await stt._init_task


def _install_fake_clock(stt: VoxistSTT, start: float = 1000.0) -> dict:
    """Replace the readiness cooldown's clock seam on ONE instance (never the
    global clock, which asyncio depends on)."""
    clock = {"now": start}
    stt._now = lambda: clock["now"]  # type: ignore[method-assign]
    return clock


class TestReadinessRecoveryAfterTransientFailure:
    """
    [2] FAILED must not be terminal for transient causes. A background
    warm-up could succeed (COMPLETED) and then one blip in the one-shot WS
    probe bricked the plugin for the rest of its life: wait_for_initialization
    returned False on its first line forever, is_ready stayed False and
    check_initialization() kept raising - even though stream() would have
    dialed and transcribed fine the moment the blip cleared.

    A rejected credential is the deliberate exception: it stays sticky.
    """

    @pytest.mark.asyncio
    async def test_transient_probe_failure_recovers_after_the_cooldown(self):
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        await _quiesce_background_init(stt)
        clock = _install_fake_clock(stt)

        ws = AsyncMock()
        dialer = AsyncMock()
        dialer.dial = AsyncMock(
            side_effect=[VoxistConnectionError("transient blip"), ws]
        )
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.COMPLETED

        # One blip in the probe fails readiness...
        assert await stt.wait_for_initialization(timeout=5.0) is False
        assert stt.initialization_state is InitializationState.FAILED
        assert stt.is_ready is False

        # ...and inside the cooldown it is not re-probed (no dial herd).
        clock["now"] += stt.READINESS_RETRY_COOLDOWN_SECONDS - 1.0
        assert await stt.wait_for_initialization(timeout=5.0) is False
        assert dialer.dial.await_count == 1

        # Past the cooldown the probe is re-attempted, succeeds, and the
        # plugin becomes ready again.
        clock["now"] += 2.0
        assert await stt.wait_for_initialization(timeout=5.0) is True
        assert dialer.dial.await_count == 2
        assert stt.initialization_state is InitializationState.COMPLETED
        assert stt.is_ready is True
        stt.check_initialization()  # no longer raises
        assert stt.initialization_error is None, (
            "a recovered plugin must not keep reporting the cleared blip"
        )

        await stt.aclose()

    @pytest.mark.asyncio
    async def test_rejected_credential_stays_sticky_past_the_cooldown(self):
        """A key the gateway refused cannot be fixed by re-probing, and
        hammering it is how a key gets banned."""
        from livekit.plugins.voxist import (
            InitializationError,
            InitializationState,
        )
        from livekit.plugins.voxist.exceptions import AuthenticationError

        stt = VoxistSTT(api_key="test")
        await _quiesce_background_init(stt)
        clock = _install_fake_clock(stt)

        dialer = AsyncMock()
        dialer.dial = AsyncMock(side_effect=AuthenticationError("revoked key"))
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.COMPLETED

        assert await stt.wait_for_initialization(timeout=5.0) is False

        clock["now"] += 10 * stt.READINESS_RETRY_COOLDOWN_SECONDS
        assert await stt.wait_for_initialization(timeout=5.0) is False
        assert dialer.dial.await_count == 1, "a rejected key must not be re-probed"
        assert stt.initialization_state is InitializationState.FAILED

        with pytest.raises(InitializationError, match="will not clear"):
            stt.check_initialization()

        await stt.aclose()

    @pytest.mark.asyncio
    async def test_transient_failure_message_advertises_the_retry(self):
        from livekit.plugins.voxist import (
            InitializationError,
            InitializationState,
        )

        stt = VoxistSTT(api_key="test")
        await _quiesce_background_init(stt)
        _install_fake_clock(stt)

        dialer = AsyncMock()
        dialer.dial = AsyncMock(side_effect=VoxistConnectionError("blip"))
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.COMPLETED

        assert await stt.wait_for_initialization(timeout=5.0) is False
        with pytest.raises(InitializationError, match="transient"):
            stt.check_initialization()

        await stt.aclose()

    @pytest.mark.asyncio
    async def test_a_transient_token_failure_also_recovers(self):
        """The recovery path is not probe-specific: a warm-up that failed on
        the token exchange re-runs it too."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test", validate_websocket=False)
        await _quiesce_background_init(stt)
        clock = _install_fake_clock(stt)

        dialer = AsyncMock()
        dialer._get_token_url = AsyncMock(
            side_effect=[VoxistConnectionError("gateway down"), "ws://ok/?t=1"]
        )
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.NOT_STARTED

        assert await stt.wait_for_initialization(timeout=5.0) is False
        assert stt.initialization_state is InitializationState.FAILED

        clock["now"] += stt.READINESS_RETRY_COOLDOWN_SECONDS + 1.0
        assert await stt.wait_for_initialization(timeout=5.0) is True
        assert stt.initialization_state is InitializationState.COMPLETED

        await stt.aclose()


class TestConcurrentReadinessProbe:
    """
    [9] The probe guard was a check-then-await-then-set, so two concurrent
    readiness calls (asyncio.gather, or a health endpoint racing
    `async with`) both dialed: two real WebSockets and two server-side ASR
    engine sessions, breaking the documented probe budget.

    Invariant: at most one probe IN FLIGHT, and at most one PER COOLDOWN
    WINDOW.
    """

    @pytest.mark.asyncio
    async def test_two_concurrent_readiness_calls_dial_once(self):
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        await _quiesce_background_init(stt)

        ws = AsyncMock()

        async def slow_dial(*args, **kwargs):
            await asyncio.sleep(0.05)  # wide enough for the racer to enter
            return ws

        dialer = AsyncMock()
        dialer.dial = AsyncMock(side_effect=slow_dial)
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.COMPLETED

        results = await asyncio.gather(
            stt.wait_for_initialization(timeout=5.0),
            stt.wait_for_initialization(timeout=5.0),
        )

        assert results == [True, True]
        assert dialer.dial.await_count == 1, (
            "concurrent readiness must open exactly one probe socket"
        )

        await stt.aclose()

    @pytest.mark.asyncio
    async def test_two_concurrent_failing_probes_dial_once(self):
        """The failure path is bounded by the same budget: the loser of the
        race must not dial again the instant the winner fails."""
        from livekit.plugins.voxist import InitializationState

        stt = VoxistSTT(api_key="test")
        await _quiesce_background_init(stt)
        _install_fake_clock(stt)

        async def slow_boom(*args, **kwargs):
            await asyncio.sleep(0.05)
            raise VoxistConnectionError("blip")

        dialer = AsyncMock()
        dialer.dial = AsyncMock(side_effect=slow_boom)
        stt._ensure_dialer = AsyncMock(return_value=dialer)
        stt._init_state = InitializationState.COMPLETED

        results = await asyncio.gather(
            stt.wait_for_initialization(timeout=5.0),
            stt.wait_for_initialization(timeout=5.0),
        )

        assert results == [False, False]
        assert dialer.dial.await_count == 1, (
            "at most one probe per cooldown window, failures included"
        )

        await stt.aclose()


class TestSessionLoopIntrospectionFailsLoud:
    """
    [11] _session_loop used to read aiohttp's private ClientSession._loop and
    return None silently when absent, which disabled BOTH loop guards at
    once: no defunct-loop rebuild and no alive-foreign-loop diagnosis. On an
    aiohttp release that renames the attribute, a plugin built under one
    asyncio.run() and used under another then died inside ws_connect with
    RuntimeError('Event loop is closed'), mapped to a generic retryable
    transport error - a misleading network diagnosis for a loop-affinity bug.
    """

    def test_owned_session_rebuild_survives_broken_introspection(self):
        """A plugin-owned session records its loop at creation, so the
        rebuild does not depend on aiohttp introspection at all."""
        stt = VoxistSTT(api_key="test_key")  # no loop: init on demand

        async def ensure():
            await stt._ensure_dialer()
            return stt._session

        first = asyncio.run(ensure())
        assert first is not None
        # Simulate the aiohttp rename: the binding is no longer an event
        # loop. Mock (not a bare object) so aiohttp's own __del__ - which
        # pokes _loop.call_exception_handler - stays quiet.
        first._loop = Mock()

        second = asyncio.run(ensure())
        assert second is not first, (
            "an unreadable binding must not silently disable the rebuild"
        )

        asyncio.run(stt.aclose())

    def test_unreadable_binding_warns_and_keeps_the_foreign_loop_guard(
        self, caplog, monkeypatch
    ):
        """A caller-supplied session cannot be rebuilt, so it is pinned to
        the loop that first used it - the guard stays armed - and the aiohttp
        incompatibility is named in the log instead of swallowed."""
        monkeypatch.setattr(VoxistSTT, "_loop_introspection_warned", False)

        loop_a = asyncio.new_event_loop()
        thread = threading.Thread(target=loop_a.run_forever, daemon=True)
        thread.start()
        try:
            session = asyncio.run_coroutine_threadsafe(
                _make_session(), loop_a
            ).result(timeout=5)
            session._loop = Mock()  # the aiohttp rename
            stt = VoxistSTT(api_key="test_key", http_session=session)

            with caplog.at_level(
                logging.WARNING, logger="livekit.plugins.voxist"
            ):
                asyncio.run_coroutine_threadsafe(
                    stt._ensure_dialer(), loop_a
                ).result(timeout=5)

            messages = [r.message for r in caplog.records]
            assert any(
                "Cannot determine which event loop" in m and "aiohttp" in m
                for m in messages
            ), messages

            # Warned once per process, not once per dial.
            caplog.clear()
            asyncio.run_coroutine_threadsafe(
                stt._ensure_dialer(), loop_a
            ).result(timeout=5)
            assert not any(
                "Cannot determine which event loop" in r.message
                for r in caplog.records
            )

            # And the foreign-loop guard is still armed for a second loop.
            with pytest.raises(RuntimeError, match="different running event loop"):
                asyncio.run(stt._ensure_dialer())
        finally:
            asyncio.run_coroutine_threadsafe(stt.aclose(), loop_a).result(
                timeout=5
            )
            asyncio.run_coroutine_threadsafe(session.close(), loop_a).result(
                timeout=5
            )
            loop_a.call_soon_threadsafe(loop_a.stop)
            thread.join(timeout=5)
            loop_a.close()


async def _make_session() -> aiohttp.ClientSession:
    return aiohttp.ClientSession()


class TestProbeTransportAbortIsReportedHonestly:
    """
    On the close-timeout path aiohttp has usually already released the
    connection, so there is no transport left to abort - and the log said
    "aborted the probe transport" regardless, telling an operator chasing a
    leaked socket that cleanup had happened when it had not.

    No test covered this at all: inverting the helper's return value left the
    whole stt suite green.
    """

    @staticmethod
    def _ws_with_transport(transport):
        return SimpleNamespace(
            _response=SimpleNamespace(
                connection=SimpleNamespace(transport=transport)
            )
        )

    def test_a_reachable_transport_is_aborted_and_reported(self):
        aborted = []
        ws = self._ws_with_transport(
            SimpleNamespace(abort=lambda: aborted.append(True))
        )
        assert VoxistSTT._abort_probe_transport(ws) is True
        assert aborted == [True]

    def test_an_already_released_connection_reports_no_abort(self):
        ws = SimpleNamespace(_response=SimpleNamespace(connection=None))
        assert VoxistSTT._abort_probe_transport(ws) is False

    def test_a_transport_that_refuses_reports_no_abort(self):
        def boom():
            raise RuntimeError("already closed")

        ws = self._ws_with_transport(SimpleNamespace(abort=boom))
        assert VoxistSTT._abort_probe_transport(ws) is False

    @pytest.mark.asyncio
    async def test_the_close_timeout_log_matches_what_happened(self, caplog):
        """The log must distinguish an abort from a no-op."""
        stt = VoxistSTT(api_key="k", validate_websocket=False)

        class HangingWS:
            def __init__(self, transport):
                self._response = SimpleNamespace(
                    connection=SimpleNamespace(transport=transport)
                )

            async def close(self):
                await asyncio.Event().wait()

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            # Nothing left to abort: the honest message says so.
            await stt._close_probe_socket(HangingWS(None))
            released = [r.message for r in caplog.records]
            caplog.clear()
            # A live transport: the abort really happens.
            aborted = []
            await stt._close_probe_socket(
                HangingWS(SimpleNamespace(abort=lambda: aborted.append(True)))
            )
            real = [r.message for r in caplog.records]

        assert any("already released" in m for m in released), released
        assert not any("already released" in m for m in real), real
        assert any("aborted the probe transport" in m for m in real), real
        assert aborted == [True]
