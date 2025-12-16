"""Pytest configuration and shared fixtures."""

import pytest
import asyncio
import numpy as np
from unittest.mock import patch
from .fixtures.mock_server import MockVoxistServer


@pytest.fixture(autouse=True)
def auto_mock_token_exchange(request):
    """
    Automatically mock _get_ws_token for all tests to bypass HTTP token exchange.

    This is needed because the SEC-001 fix adds token exchange via HTTPS before
    WebSocket connection. Tests don't have a real HTTP server, so we mock this.

    To skip this fixture, mark test with:
    - @pytest.mark.no_auto_mock_token - for tests that need to test token exchange directly
    - @pytest.mark.integration - for integration tests that use mock_voxist_server fixture
    """
    # Skip if test is marked with no_auto_mock_token or integration (which has its own server)
    if (request.node.get_closest_marker('no_auto_mock_token') or
        request.node.get_closest_marker('integration')):
        yield
        return

    from livekit.plugins.voxist.connection_pool import ConnectionPool

    async def _mock_get_ws_token(self, language: str, sample_rate: int) -> str:
        return f"ws://localhost:8765/ws?token=mock_jwt_token&lang={language}&sample_rate={sample_rate}"

    with patch.object(ConnectionPool, '_get_ws_token', _mock_get_ws_token):
        yield


@pytest.fixture(scope="session")
def event_loop():
    """Create event loop for async tests."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_token_exchange_for_server():
    """
    Factory fixture to mock token exchange for a specific mock server.

    Use this in integration tests that need to test with a mock server:

        @pytest.mark.no_auto_mock_token
        async def test_something(self, mock_voxist_server, mock_token_exchange_for_server):
            with mock_token_exchange_for_server(mock_voxist_server):
                # Test code here
    """
    from contextlib import contextmanager

    @contextmanager
    def _mock_for_server(server):
        from livekit.plugins.voxist.connection_pool import ConnectionPool

        async def _mock_get_ws_token(self, language: str, sample_rate: int) -> str:
            return f"ws://{server.host}:{server.port}/ws?token=mock_jwt_token&lang={language}&sample_rate={sample_rate}"

        with patch.object(ConnectionPool, '_get_ws_token', _mock_get_ws_token):
            yield

    return _mock_for_server


@pytest.fixture
def sample_rate():
    """Default sample rate for tests."""
    return 16000


@pytest.fixture
def test_api_key():
    """Test API key (from environment or mock)."""
    return "voxist_test_key_for_testing"


@pytest.fixture
async def mock_voxist_server(request):
    """
    Create and start mock Voxist WebSocket server for testing.

    Also sets up the token exchange mock to return this server's URL.

    Yields:
        MockVoxistServer instance running on localhost:8765
    """
    from livekit.plugins.voxist.connection_pool import ConnectionPool

    server = MockVoxistServer(
        port=8765,
        valid_api_key="test_key",
        transcription_text="bonjour monde",
        transcription_confidence=0.95,
        send_interim=True,
    )

    await server.start()

    # Mock token exchange to return this server's URL (for integration tests)
    async def _mock_get_ws_token(self, language: str, sample_rate: int) -> str:
        return f"ws://{server.host}:{server.port}/ws?token=mock_jwt_token&lang={language}&sample_rate={sample_rate}"

    with patch.object(ConnectionPool, '_get_ws_token', _mock_get_ws_token):
        yield server

    await server.stop()


@pytest.fixture
def generate_test_audio():
    """
    Factory fixture for generating test audio.

    Returns:
        Function that generates audio arrays
    """
    def _generate(
        duration_ms: int = 1000,
        sample_rate: int = 16000,
        frequency: int = 440,
    ) -> np.ndarray:
        """
        Generate sine wave test audio.

        Args:
            duration_ms: Duration in milliseconds
            sample_rate: Sample rate in Hz
            frequency: Sine wave frequency in Hz

        Returns:
            Int16 NumPy array with audio samples
        """
        duration_s = duration_ms / 1000.0
        num_samples = int(sample_rate * duration_s)

        # Generate time array
        t = np.linspace(0, duration_s, num_samples, endpoint=False)

        # Generate sine wave
        audio = np.sin(2 * np.pi * frequency * t)

        # Convert to Int16 range
        audio_int16 = (audio * 32767).astype(np.int16)

        return audio_int16

    return _generate
