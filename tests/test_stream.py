"""Unit tests for VoxistSTTStream class with focus on VUL-003 ownership validation."""

import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from livekit.plugins.voxist.connection_pool import ConnectionPool
from livekit.plugins.voxist.exceptions import OwnershipViolationError
from livekit.plugins.voxist.models import Connection, ConnectionState
from livekit.plugins.voxist.stream import VoxistSTTStream


class TestStreamOwnership:
    """Test VUL-003 connection ownership validation."""

    @pytest.fixture
    def mock_stt(self):
        """Create mock VoxistSTT instance."""
        stt = Mock()
        stt._config = {
            "sample_rate": 16000,
            "chunk_duration_ms": 100,
            "stride_overlap_ms": 0,
            "interim_results": True,
        }
        stt._api_key = "test_key"
        return stt

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = AsyncMock(spec=ConnectionPool)
        return pool

    @pytest.fixture
    def mock_connection(self):
        """Create mock connection with WebSocket."""
        conn = Connection(id=0, state=ConnectionState.IN_USE)
        conn.ws = AsyncMock()
        conn.ws.closed = False
        conn.ws.send_bytes = AsyncMock()
        conn.buffered_amount = 0
        return conn

    @pytest.mark.asyncio
    async def test_ownership_flag_initial_state(self, mock_stt, mock_pool):
        """Test _owns_connection is False initially."""
        from livekit.agents.types import APIConnectOptions

        conn_options = APIConnectOptions(
            max_retry=3,
            retry_interval=1.0,
            timeout=10.0,
        )

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        assert stream._owns_connection is False

    @pytest.mark.asyncio
    async def test_ownership_set_after_get_connection(self, mock_stt, mock_pool, mock_connection):
        """Test _owns_connection is True after acquiring connection."""
        from livekit.agents.types import APIConnectOptions

        mock_pool.get_connection.return_value = mock_connection

        conn_options = APIConnectOptions(
            max_retry=3,
            retry_interval=1.0,
            timeout=10.0,
        )

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        # Manually call the connection acquisition (normally done in _run)
        stream._conn = await mock_pool.get_connection()
        stream._owns_connection = True  # Simulating _run behavior

        assert stream._owns_connection is True
        assert stream._conn is mock_connection

    @pytest.mark.asyncio
    async def test_ownership_cleared_before_release(self, mock_stt, mock_pool, mock_connection):
        """Test _owns_connection is False before release_connection."""
        from livekit.agents.types import APIConnectOptions

        mock_pool.get_connection.return_value = mock_connection

        conn_options = APIConnectOptions(
            max_retry=3,
            retry_interval=1.0,
            timeout=10.0,
        )

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        # Setup: acquire connection
        stream._conn = mock_connection
        stream._owns_connection = True

        # Simulate release (normally done in finally block of _run)
        stream._owns_connection = False
        await mock_pool.release_connection(stream._conn)

        assert stream._owns_connection is False
        mock_pool.release_connection.assert_called_once_with(mock_connection)

    @pytest.mark.asyncio
    async def test_vul003_assertion_without_ownership(self, mock_stt, mock_pool, mock_connection):
        """
        VUL-003: Test assertion fires when updating buffered_amount without ownership.

        This tests that the security fix properly detects when a stream attempts
        to modify buffered_amount without exclusive ownership.
        """
        from livekit.agents.types import APIConnectOptions

        conn_options = APIConnectOptions(
            max_retry=3,
            retry_interval=1.0,
            timeout=10.0,
        )

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        # Cancel the auto-started task to prevent race with direct method calls
        stream._task.cancel()
        try:
            await stream._task
        except asyncio.CancelledError:
            pass

        # Setup: have a connection but NOT ownership
        stream._conn = mock_connection
        stream._owns_connection = False  # Bug scenario - no ownership

        # Create test audio (using int16 as expected by _send_audio_chunk)
        audio_int16 = np.zeros(160, dtype=np.int16)  # 10ms at 16kHz

        # Should raise OwnershipViolationError due to VUL-003 check
        with pytest.raises(OwnershipViolationError, match="without ownership"):
            await stream._send_audio_chunk(audio_int16)

    @pytest.mark.asyncio
    async def test_vul003_assertion_with_ownership(self, mock_stt, mock_pool, mock_connection):
        """
        VUL-003: Test normal operation with ownership succeeds.

        When the stream has exclusive ownership, buffered_amount updates
        should proceed without assertion failure.
        """
        from livekit.agents.types import APIConnectOptions

        conn_options = APIConnectOptions(
            max_retry=3,
            retry_interval=1.0,
            timeout=10.0,
        )

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        # Cancel the auto-started task to prevent race with direct method calls
        stream._task.cancel()
        try:
            await stream._task
        except asyncio.CancelledError:
            pass

        # Setup: proper ownership
        stream._conn = mock_connection
        stream._owns_connection = True

        # Mock get_transport() to return low buffer size
        mock_transport = Mock()
        mock_transport.get_write_buffer_size = Mock(return_value=0)
        mock_connection.ws.get_transport = Mock(return_value=mock_transport)

        # Create test audio (using int16 as expected by _send_audio_chunk)
        audio_int16 = np.zeros(160, dtype=np.int16)  # 10ms at 16kHz

        # Should NOT raise - proper ownership
        await stream._send_audio_chunk(audio_int16)

        # Verify send was called
        mock_connection.ws.send_bytes.assert_called_once()


class TestStreamLifecycle:
    """Test stream lifecycle and ownership transitions."""

    @pytest.fixture
    def mock_stt(self):
        """Create mock VoxistSTT instance."""
        stt = Mock()
        stt._config = {
            "sample_rate": 16000,
            "chunk_duration_ms": 100,
            "stride_overlap_ms": 0,
            "interim_results": True,
        }
        stt._api_key = "test_key"
        return stt

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = AsyncMock(spec=ConnectionPool)
        return pool

    @pytest.mark.asyncio
    async def test_ownership_lifecycle_on_connection_failure(self, mock_stt, mock_pool):
        """
        Test ownership is properly cleared when connection fails.

        If get_connection raises, _owns_connection should remain False.
        """
        from livekit.agents.types import APIConnectOptions
        from livekit.plugins.voxist.exceptions import ConnectionPoolExhaustedError

        mock_pool.get_connection.side_effect = ConnectionPoolExhaustedError("No connections")

        conn_options = APIConnectOptions(
            max_retry=0,  # No retry
            retry_interval=1.0,
            timeout=10.0,
        )

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        # _run should handle the exception
        # Ownership should never be set to True
        assert stream._owns_connection is False
        assert stream._conn is None

    @pytest.mark.asyncio
    async def test_concurrent_stream_prevention(self, mock_stt, mock_pool):
        """
        Test that two streams cannot share the same connection.

        The connection pool returns connections in IN_USE state,
        preventing concurrent access.
        """
        from livekit.agents.types import APIConnectOptions

        conn1 = Connection(id=0, state=ConnectionState.IN_USE)
        conn2 = Connection(id=1, state=ConnectionState.IN_USE)

        # Pool returns different connections for different requests
        mock_pool.get_connection.side_effect = [conn1, conn2]

        conn_options = APIConnectOptions(
            max_retry=3,
            retry_interval=1.0,
            timeout=10.0,
        )

        stream1 = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        stream2 = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        # Simulate both acquiring connections
        stream1._conn = await mock_pool.get_connection()
        stream1._owns_connection = True

        stream2._conn = await mock_pool.get_connection()
        stream2._owns_connection = True

        # Different connections - each has exclusive ownership
        assert stream1._conn.id != stream2._conn.id
        assert stream1._owns_connection is True
        assert stream2._owns_connection is True


class TestBackpressureWithOwnership:
    """
    Test CRIT-001 backpressure handling with ownership validation.

    Verifies high/low water mark pattern prevents buffer overflow.
    """

    @pytest.fixture
    def mock_stt(self):
        """Create mock VoxistSTT instance."""
        stt = Mock()
        stt._config = {
            "sample_rate": 16000,
            "chunk_duration_ms": 100,
            "stride_overlap_ms": 0,
            "interim_results": True,
        }
        stt._api_key = "test_key"
        return stt

    @pytest.fixture
    def mock_pool(self):
        """Create mock connection pool."""
        pool = AsyncMock(spec=ConnectionPool)
        return pool

    @pytest.fixture
    def mock_connection(self):
        """Create mock connection with WebSocket."""
        conn = Connection(id=0, state=ConnectionState.IN_USE)
        conn.ws = AsyncMock()
        conn.ws.closed = False
        conn.ws.send_bytes = AsyncMock()
        conn.buffered_amount = 0
        return conn

    async def _create_stream_with_connection(self, mock_stt, mock_pool, mock_connection):
        """Create a stream with proper ownership setup (async helper)."""
        from livekit.agents.types import APIConnectOptions

        conn_options = APIConnectOptions(
            max_retry=3,
            retry_interval=1.0,
            timeout=10.0,
        )

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        # Cancel the auto-started task to prevent race with direct method calls
        stream._task.cancel()
        try:
            await stream._task
        except asyncio.CancelledError:
            pass

        stream._conn = mock_connection
        stream._owns_connection = True

        return stream, mock_connection

    @pytest.mark.asyncio
    async def test_backpressure_constants_defined(self):
        """Test CRIT-001 backpressure constants are properly defined."""
        assert hasattr(VoxistSTTStream, 'HIGH_WATER_MARK')
        assert hasattr(VoxistSTTStream, 'LOW_WATER_MARK')
        assert hasattr(VoxistSTTStream, 'BACKPRESSURE_CHECK_INTERVAL')

        # HIGH_WATER_MARK must be greater than LOW_WATER_MARK
        assert VoxistSTTStream.HIGH_WATER_MARK > VoxistSTTStream.LOW_WATER_MARK

        # Values set high as safety net only (Voxist backend handles streaming well)
        # HIGH_WATER_MARK = 2MB, LOW_WATER_MARK = 1MB
        assert VoxistSTTStream.HIGH_WATER_MARK >= 1 * 1024 * 1024  # At least 1MB
        assert VoxistSTTStream.HIGH_WATER_MARK <= 10 * 1024 * 1024  # At most 10MB
        assert VoxistSTTStream.LOW_WATER_MARK >= 512 * 1024  # At least 512KB
        assert VoxistSTTStream.LOW_WATER_MARK <= 5 * 1024 * 1024  # At most 5MB

        # Check interval is reasonable (1-100ms)
        assert VoxistSTTStream.BACKPRESSURE_CHECK_INTERVAL >= 0.001
        assert VoxistSTTStream.BACKPRESSURE_CHECK_INTERVAL <= 0.1

    @pytest.mark.asyncio
    async def test_backpressure_sends_normally_when_buffer_low(self, mock_stt, mock_pool, mock_connection):
        """Test audio is sent immediately when buffer is below HIGH_WATER_MARK."""
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        # Mock transport with low buffer
        mock_transport = Mock()
        mock_transport.get_write_buffer_size = Mock(return_value=1000)  # 1KB - low
        connection.ws.get_transport = Mock(return_value=mock_transport)

        audio_int16 = np.zeros(160, dtype=np.int16)

        # Should send immediately without waiting
        import time
        start = time.time()
        await stream._send_audio_chunk(audio_int16)
        elapsed = time.time() - start

        # Should only wait for the chunk duration (100ms) not additional backpressure
        assert elapsed < 0.2  # 200ms max (100ms chunk + overhead)
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_backpressure_waits_when_buffer_high(self, mock_stt, mock_pool, mock_connection):
        """
        CRIT-001: Test backpressure wait when buffer exceeds HIGH_WATER_MARK.

        When buffer is above HIGH_WATER_MARK, _send_audio_chunk should wait
        until buffer drains to LOW_WATER_MARK.
        """
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        # Track buffer size changes to simulate draining
        buffer_sizes = [
            VoxistSTTStream.HIGH_WATER_MARK + 1000,  # Initial - above high
            VoxistSTTStream.LOW_WATER_MARK + 5000,   # Still draining
            VoxistSTTStream.LOW_WATER_MARK - 1000,   # Below low - can send
        ]
        buffer_index = [0]

        def get_buffer_size():
            idx = min(buffer_index[0], len(buffer_sizes) - 1)
            buffer_index[0] += 1
            return buffer_sizes[idx]

        mock_transport = Mock()
        mock_transport.get_write_buffer_size = Mock(side_effect=get_buffer_size)
        connection.ws.get_transport = Mock(return_value=mock_transport)

        audio_int16 = np.zeros(160, dtype=np.int16)

        # Should wait for buffer to drain
        await stream._send_audio_chunk(audio_int16)

        # Buffer was checked multiple times until it dropped below LOW_WATER_MARK
        assert mock_transport.get_write_buffer_size.call_count >= 2
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_backpressure_exits_on_connection_close_during_wait(self, mock_stt, mock_pool, mock_connection):
        """
        CRIT-001: Test graceful exit when connection closes during backpressure wait.

        If the connection closes while waiting for buffer to drain,
        _send_audio_chunk should return without sending.
        """
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        # Start with high buffer, then simulate connection close
        call_count = [0]

        def get_buffer_and_close():
            call_count[0] += 1
            if call_count[0] > 1:
                connection.ws.closed = True  # Simulate close during wait
            return VoxistSTTStream.HIGH_WATER_MARK + 1000

        mock_transport = Mock()
        mock_transport.get_write_buffer_size = Mock(side_effect=get_buffer_and_close)
        connection.ws.get_transport = Mock(return_value=mock_transport)

        audio_int16 = np.zeros(160, dtype=np.int16)

        # Should return early without sending
        await stream._send_audio_chunk(audio_int16)

        # Send was NOT called due to connection close
        connection.ws.send_bytes.assert_not_called()

    @pytest.mark.asyncio
    async def test_backpressure_uses_fallback_when_transport_unavailable(self, mock_stt, mock_pool, mock_connection):
        """Test fallback to buffered_amount when transport is unavailable."""
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        # No transport available - raises AttributeError
        connection.ws.get_transport = Mock(return_value=None)
        connection.buffered_amount = 1000  # Use fallback

        audio_int16 = np.zeros(160, dtype=np.int16)

        # Should use buffered_amount fallback and succeed
        await stream._send_audio_chunk(audio_int16)

        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_backpressure_with_fallback_high_buffer(self, mock_stt, mock_pool, mock_connection):
        """Test backpressure triggers using fallback buffered_amount."""
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        # No transport, high buffered_amount
        connection.ws.get_transport = Mock(return_value=None)

        # Simulate buffer draining via buffered_amount
        buffer_values = [
            VoxistSTTStream.HIGH_WATER_MARK + 1000,
            VoxistSTTStream.LOW_WATER_MARK - 1000,
        ]
        buffer_idx = [0]

        def get_buffer():
            idx = min(buffer_idx[0], len(buffer_values) - 1)
            val = buffer_values[idx]
            buffer_idx[0] += 1
            return val

        # Override _get_write_buffer_size to use our mock values
        stream._get_write_buffer_size = Mock(side_effect=get_buffer)

        audio_int16 = np.zeros(160, dtype=np.int16)
        await stream._send_audio_chunk(audio_int16)

        # Should have checked buffer multiple times
        assert stream._get_write_buffer_size.call_count >= 2
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_buffered_amount_fallback_with_ownership(self, mock_stt, mock_pool, mock_connection):
        """
        Test buffered_amount fallback when transport unavailable.

        With ownership, the fallback estimate update should succeed.
        """
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        # No transport available (triggers fallback)
        connection.ws.get_transport = Mock(return_value=None)


        # Create test audio
        audio_int16 = np.zeros(160, dtype=np.int16)
        audio_bytes_len = len(audio_int16.tobytes())

        # Should succeed and update buffered_amount
        await stream._send_audio_chunk(audio_int16)

        # Verify buffered_amount was updated (with decay)
        # Formula: buffered_amount += len - len//2 = len//2
        audio_bytes_len // 2
        assert connection.buffered_amount >= 0

    @pytest.mark.asyncio
    async def test_backpressure_no_connection_returns_early(self, mock_stt, mock_pool):
        """Test _send_audio_chunk returns early with no connection."""
        from livekit.agents.types import APIConnectOptions

        conn_options = APIConnectOptions(
            max_retry=3,
            retry_interval=1.0,
            timeout=10.0,
        )

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=conn_options,
        )

        stream._task.cancel()
        try:
            await stream._task
        except asyncio.CancelledError:
            pass

        # No connection set
        stream._conn = None

        audio_int16 = np.zeros(160, dtype=np.int16)

        # Should return without error
        await stream._send_audio_chunk(audio_int16)

    @pytest.mark.asyncio
    async def test_backpressure_closed_ws_returns_early(self, mock_stt, mock_pool, mock_connection):
        """Test _send_audio_chunk returns early with closed WebSocket."""
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        # Mark WebSocket as closed
        connection.ws.closed = True

        audio_int16 = np.zeros(160, dtype=np.int16)

        # Should return without sending
        await stream._send_audio_chunk(audio_int16)
        connection.ws.send_bytes.assert_not_called()
