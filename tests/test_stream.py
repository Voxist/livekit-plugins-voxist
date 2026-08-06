"""Unit tests for VoxistSTTStream class with focus on VUL-003 ownership validation."""

import asyncio
from unittest.mock import AsyncMock, Mock

import numpy as np
import pytest

from livekit.plugins.voxist.connection_pool import ConnectionPool
from livekit.plugins.voxist.exceptions import OwnershipViolationError
from livekit.plugins.voxist.models import Connection, ConnectionState
from livekit.plugins.voxist.stream import VoxistSTTStream


def install_fake_transport(connection, buffer_size):
    """
    Wire a fake transport into the real aiohttp attribute chain.

    _get_write_buffer_size() reaches the transport via
    ws._response.connection.transport, because aiohttp exposes no public
    accessor. Tests must populate that same chain: mocking a ws.get_transport()
    method instead would assert against an API aiohttp does not have, which is
    precisely how the indefinite-backpressure bug shipped with a green suite.

    Args:
        connection: Connection whose ws should expose the transport
        buffer_size: int, or a callable returning successive sizes

    Returns:
        The fake transport mock.
    """
    transport = Mock()
    transport.is_closing = Mock(return_value=False)
    if callable(buffer_size):
        transport.get_write_buffer_size = Mock(side_effect=buffer_size)
    else:
        transport.get_write_buffer_size = Mock(return_value=buffer_size)

    response = Mock()
    response.connection = Mock()
    response.connection.transport = transport
    connection.ws._response = response
    return transport


def remove_transport(connection):
    """Make the transport unreachable, as with a real aiohttp WebSocket."""
    connection.ws._response = None


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
        # Default to an unreachable transport, matching a real aiohttp
        # WebSocket. Tests that need a size call install_fake_transport().
        conn.ws._response = None
        # Real aiohttp WebSockets have no get_transport(); make the mock raise
        # AttributeError like the real object so no future implementation can
        # quietly depend on a method that does not exist.
        del conn.ws.get_transport
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
        # Default to an unreachable transport, matching a real aiohttp
        # WebSocket. Tests that need a size call install_fake_transport().
        conn.ws._response = None
        # Real aiohttp WebSockets have no get_transport(); make the mock raise
        # AttributeError like the real object so no future implementation can
        # quietly depend on a method that does not exist.
        del conn.ws.get_transport
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

        # Marks are compared against the real asyncio transport write buffer,
        # which pauses at its own 64KB high-water mark. They must sit above
        # that (so the transport's own flow control acts first) but low enough
        # to still be reachable - marks in the megabytes would never trigger.
        transport_default_high_water = 64 * 1024
        assert VoxistSTTStream.HIGH_WATER_MARK > transport_default_high_water
        assert VoxistSTTStream.HIGH_WATER_MARK <= 1 * 1024 * 1024
        assert VoxistSTTStream.LOW_WATER_MARK >= 16 * 1024

        # Check interval is reasonable (1-100ms)
        assert VoxistSTTStream.BACKPRESSURE_CHECK_INTERVAL >= 0.001
        assert VoxistSTTStream.BACKPRESSURE_CHECK_INTERVAL <= 0.1

        # The backpressure wait must be bounded: an unbounded wait starves the
        # stream of audio until the Voxist WebSocket times out.
        assert VoxistSTTStream.BACKPRESSURE_MAX_WAIT > 0
        assert VoxistSTTStream.BACKPRESSURE_MAX_WAIT <= 30.0

    @pytest.mark.asyncio
    async def test_backpressure_sends_normally_when_buffer_low(self, mock_stt, mock_pool, mock_connection):
        """Test audio is sent immediately when buffer is below HIGH_WATER_MARK."""
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        # Transport reporting a low buffer
        install_fake_transport(connection, 1000)  # 1KB - low

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

        mock_transport = install_fake_transport(connection, get_buffer_size)

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

        install_fake_transport(connection, get_buffer_and_close)

        audio_int16 = np.zeros(160, dtype=np.int16)

        # Should return early without sending
        await stream._send_audio_chunk(audio_int16)

        # Send was NOT called due to connection close
        connection.ws.send_bytes.assert_not_called()

    @pytest.mark.asyncio
    async def test_sends_normally_when_buffer_unmeasurable(
        self, mock_stt, mock_pool, mock_connection
    ):
        """
        Test sending proceeds when the transport size cannot be measured.

        aiohttp exposes no public accessor for the write buffer, so this is the
        common production case. It must never throttle: `await send_bytes()`
        applies real backpressure by draining when the transport pauses.
        """
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        remove_transport(connection)

        assert stream._get_write_buffer_size() == 0

        audio_int16 = np.zeros(160, dtype=np.int16)
        await stream._send_audio_chunk(audio_int16)

        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_backpressure_waits_then_sends_when_measured_high(
        self, mock_stt, mock_pool, mock_connection
    ):
        """Test backpressure engages on a measured high buffer, then releases."""
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

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

        transport = install_fake_transport(connection, get_buffer)

        audio_int16 = np.zeros(160, dtype=np.int16)
        await stream._send_audio_chunk(audio_int16)

        # Should have checked buffer multiple times
        assert transport.get_write_buffer_size.call_count >= 2
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_buffered_amount_records_measurement_with_ownership(
        self, mock_stt, mock_pool, mock_connection
    ):
        """
        Test buffered_amount records the measured buffer, not a running total.

        The pool load-balances on this value, so it must be a snapshot of the
        current transport buffer. Accumulating here is what caused streams to
        pause permanently after ~100s of audio.
        """
        stream, connection = await self._create_stream_with_connection(mock_stt, mock_pool, mock_connection)

        install_fake_transport(connection, 8192)

        audio_int16 = np.zeros(160, dtype=np.int16)
        await stream._send_audio_chunk(audio_int16)

        assert connection.buffered_amount == 8192

        # A second send records the new measurement rather than adding to it
        await stream._send_audio_chunk(audio_int16)
        assert connection.buffered_amount == 8192

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


class TestBackpressureLiveness:
    """
    Regression tests for the indefinite-backpressure outage.

    Symptom: after roughly 90-110 seconds of audio the plugin stopped sending
    to Voxist entirely and the WebSocket timed out.

    Cause: _get_write_buffer_size() called a non-existent aiohttp method
    (ws.get_transport()), silently fell back to a counter that added half of
    every chunk ever sent and never decayed, and the drain loop that counter
    fed had no time bound. Once the counter passed HIGH_WATER_MARK the loop
    re-read the same frozen value forever.

    These tests pin the three properties that make that impossible:
      1. sending stays live over far more audio than the old trip point,
      2. buffered_amount is a measurement, never an accumulator,
      3. any backpressure wait is time-bounded.
    """

    @pytest.fixture
    def mock_stt(self):
        stt = Mock()
        stt._config = {
            "sample_rate": 16000,
            "chunk_duration_ms": 100,
            "stride_overlap_ms": 20,
            "interim_results": True,
        }
        stt._api_key = "test_key"
        return stt

    @pytest.fixture
    def mock_pool(self):
        return AsyncMock(spec=ConnectionPool)

    @pytest.fixture
    def mock_connection(self):
        conn = Connection(id=0, state=ConnectionState.IN_USE)
        conn.ws = AsyncMock()
        conn.ws.closed = False
        conn.ws.send_bytes = AsyncMock()
        conn.buffered_amount = 0
        conn.ws._response = None  # unreachable transport, as in production
        del conn.ws.get_transport  # real aiohttp has no such method
        return conn

    async def _stream(self, mock_stt, mock_pool, mock_connection):
        from livekit.agents.types import APIConnectOptions

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=APIConnectOptions(max_retry=3, retry_interval=1.0, timeout=10.0),
        )
        stream._task.cancel()
        try:
            await stream._task
        except asyncio.CancelledError:
            pass
        stream._conn = mock_connection
        stream._owns_connection = True
        return stream, mock_connection

    @pytest.mark.asyncio
    async def test_stream_stays_live_past_old_trip_point(
        self, mock_stt, mock_pool, mock_connection
    ):
        """
        Sending must not stall after a sustained run of audio.

        The old accounting tripped at 2MB, reached after ~106s of 16kHz audio.
        This drives more than three times that volume and requires every chunk
        to reach the WebSocket. Each send is individually timed out, so a
        reintroduced infinite wait fails fast instead of hanging the suite.
        """
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        chunk = np.zeros(1600, dtype=np.int16)  # 100ms @ 16kHz = 3200 bytes
        chunk_bytes = len(chunk.tobytes())
        old_trip_point = 2 * 1024 * 1024
        chunk_count = (3 * old_trip_point) // chunk_bytes

        for _ in range(chunk_count):
            await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=1.0)

        assert connection.ws.send_bytes.await_count == chunk_count
        total_sent = chunk_count * chunk_bytes
        assert total_sent > 3 * old_trip_point - chunk_bytes
        # Nothing accumulated along the way
        assert connection.buffered_amount == 0

    @pytest.mark.asyncio
    async def test_buffered_amount_never_accumulates(
        self, mock_stt, mock_pool, mock_connection
    ):
        """buffered_amount must track the measured buffer, not sum of sends."""
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        steady_buffer = 4096
        install_fake_transport(connection, steady_buffer)

        chunk = np.zeros(1600, dtype=np.int16)
        for _ in range(500):
            await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=1.0)

        # 500 sends of 3200B each: an accumulator would be ~800KB by now
        assert connection.buffered_amount == steady_buffer
        assert connection.ws.send_bytes.await_count == 500

    @pytest.mark.asyncio
    async def test_backpressure_wait_is_time_bounded(
        self, mock_stt, mock_pool, mock_connection, monkeypatch
    ):
        """
        A permanently full buffer must not pause the stream forever.

        This is the exact shape of the outage: the measured buffer never drops
        below LOW_WATER_MARK. The wait must expire and the chunk must still be
        sent, deferring to aiohttp's own transport flow control.
        """
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        monkeypatch.setattr(VoxistSTTStream, "BACKPRESSURE_MAX_WAIT", 0.3)
        install_fake_transport(connection, VoxistSTTStream.HIGH_WATER_MARK * 10)

        chunk = np.zeros(1600, dtype=np.int16)
        loop = asyncio.get_running_loop()
        start = loop.time()
        await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=5.0)
        elapsed = loop.time() - start

        # It waited, gave up at the cap, and still delivered the audio
        assert elapsed >= 0.3
        assert elapsed < 3.0
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_repeated_sends_under_permanent_backpressure_still_flow(
        self, mock_stt, mock_pool, mock_connection, monkeypatch
    ):
        """Even with the buffer stuck full, audio keeps flowing chunk after chunk."""
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        monkeypatch.setattr(VoxistSTTStream, "BACKPRESSURE_MAX_WAIT", 0.05)
        install_fake_transport(connection, VoxistSTTStream.HIGH_WATER_MARK * 10)

        chunk = np.zeros(1600, dtype=np.int16)
        for _ in range(10):
            await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=2.0)

        assert connection.ws.send_bytes.await_count == 10

    @pytest.mark.asyncio
    async def test_unmeasurable_buffer_never_throttles(
        self, mock_stt, mock_pool, mock_connection
    ):
        """
        An unmeasurable transport must report 0, not a guess.

        Guessing here is what broke production: any non-zero heuristic can
        drift above the water marks and stop the stream.
        """
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        remove_transport(connection)
        assert stream._get_transport() is None
        assert stream._get_write_buffer_size() == 0

        # A stale value on the connection must not be resurrected as an estimate
        connection.buffered_amount = 999_999_999
        assert stream._get_write_buffer_size() == 0

        chunk = np.zeros(1600, dtype=np.int16)
        await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=1.0)
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_closing_transport_treated_as_unmeasurable(
        self, mock_stt, mock_pool, mock_connection
    ):
        """A closing transport must not be read for a buffer size."""
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        transport = install_fake_transport(connection, 123456)
        transport.is_closing = Mock(return_value=True)

        assert stream._get_transport() is None
        assert stream._get_write_buffer_size() == 0

    @pytest.mark.asyncio
    async def test_transport_errors_are_contained(
        self, mock_stt, mock_pool, mock_connection
    ):
        """A transport that raises must degrade to 0, not propagate."""
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        transport = install_fake_transport(connection, 0)
        transport.get_write_buffer_size = Mock(side_effect=RuntimeError("detached"))

        assert stream._get_write_buffer_size() == 0

        chunk = np.zeros(1600, dtype=np.int16)
        await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=1.0)
        connection.ws.send_bytes.assert_called_once()


class TestTransportAccessorAgainstRealAiohttp:
    """
    Guard the private aiohttp attribute chain used to read the write buffer.

    aiohttp publishes no accessor for a WebSocket's transport, so
    _get_write_buffer_size() walks ws._response.connection.transport. These
    tests run against a real aiohttp client/server pair so that an aiohttp
    upgrade which moves that attribute fails loudly here, instead of silently
    degrading every measurement to 0 forever.

    The original bug was exactly this failure mode, undetected because the unit
    tests mocked a ws.get_transport() method that aiohttp never had.
    """

    @pytest.mark.asyncio
    async def test_ws_has_no_get_transport_method(self):
        """Document why the private chain is necessary at all."""
        import aiohttp

        assert not hasattr(aiohttp.ClientWebSocketResponse, "get_transport"), (
            "aiohttp now exposes get_transport(); prefer it over the private "
            "ws._response.connection.transport chain in _get_transport()"
        )

    @pytest.mark.asyncio
    async def test_reads_real_transport_buffer_size(self):
        """_get_write_buffer_size() must return a real size on a live socket."""
        import aiohttp
        from aiohttp import web

        async def handler(request):
            ws = web.WebSocketResponse()
            await ws.prepare(request)
            async for _ in ws:
                pass
            return ws

        app = web.Application()
        app.router.add_get("/ws", handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, "127.0.0.1", 0)
        await site.start()
        port = site._server.sockets[0].getsockname()[1]

        try:
            async with aiohttp.ClientSession() as session:
                async with session.ws_connect(f"http://127.0.0.1:{port}/ws") as ws:
                    stt = Mock()
                    stt._config = {
                        "sample_rate": 16000,
                        "chunk_duration_ms": 100,
                        "stride_overlap_ms": 20,
                        "interim_results": True,
                    }
                    stt._api_key = "k"
                    from livekit.agents.types import APIConnectOptions

                    stream = VoxistSTTStream(
                        stt=stt,
                        pool=AsyncMock(spec=ConnectionPool),
                        config=stt._config,
                        language="fr",
                        conn_options=APIConnectOptions(
                            max_retry=3, retry_interval=1.0, timeout=10.0
                        ),
                    )
                    stream._task.cancel()
                    try:
                        await stream._task
                    except asyncio.CancelledError:
                        pass

                    conn = Connection(id=0, state=ConnectionState.IN_USE)
                    conn.ws = ws
                    stream._conn = conn
                    stream._owns_connection = True

                    transport = stream._get_transport()
                    assert transport is not None, (
                        "could not reach the transport on a live aiohttp WebSocket - "
                        "the ws._response.connection.transport chain has moved"
                    )
                    assert isinstance(stream._get_write_buffer_size(), int)

                    # Sending real audio keeps the measurement sane and finite
                    await stream._send_audio_chunk(np.zeros(1600, dtype=np.int16))
                    assert 0 <= conn.buffered_amount < VoxistSTTStream.HIGH_WATER_MARK
        finally:
            await runner.cleanup()


class TestLanguageCodeHandling:
    """
    The code sent to Voxist stays raw; the code emitted to livekit is normalized.

    livekit's SpeechData declares `language: LanguageCode` and coerces a plain
    str in __post_init__. LanguageCode normalizes to BCP-47, which uppercases
    the subtag - so "fr-medical" is emitted as "fr-MEDICAL". Voxist routes
    engines on the exact code, so the value used for the connection must not be
    normalized. These tests pin both halves of that split.
    """

    async def _stream(self, language):
        from livekit.agents.types import APIConnectOptions

        stt = Mock()
        stt._config = {
            "sample_rate": 16000,
            "chunk_duration_ms": 100,
            "stride_overlap_ms": 20,
            "interim_results": True,
        }
        stt._api_key = "k"
        stream = VoxistSTTStream(
            stt=stt,
            pool=AsyncMock(spec=ConnectionPool),
            config=stt._config,
            language=language,
            conn_options=APIConnectOptions(max_retry=3, retry_interval=1.0, timeout=10.0),
        )
        stream._task.cancel()
        try:
            await stream._task
        except asyncio.CancelledError:
            pass
        return stream

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", ["fr", "fr-medical", "fr-FR", "en-US", "nl"])
    async def test_raw_language_preserved_for_voxist(self, language):
        """_language must stay byte-identical - it is sent to the backend."""
        stream = await self._stream(language)
        assert stream._language == language

    @pytest.mark.asyncio
    async def test_medical_language_normalized_for_emitted_events(self):
        """Document the normalization applied to the emitted language code."""
        stream = await self._stream("fr-medical")

        # Normalized form differs only in case
        assert str(stream._speech_language) == "fr-MEDICAL"
        assert str(stream._speech_language).lower() == "fr-medical"
        # ...while the code sent to Voxist is untouched
        assert stream._language == "fr-medical"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("language", ["fr", "fr-FR", "en-US", "nl", "sv"])
    async def test_plain_language_codes_round_trip(self, language):
        """Ordinary codes are unchanged by normalization."""
        stream = await self._stream(language)
        assert str(stream._speech_language) == language

    @pytest.mark.asyncio
    async def test_emitted_event_carries_normalized_language(self):
        """An end-to-end check that _process_result emits the normalized code."""
        stream = await self._stream("fr-medical")

        # Collect emitted events directly: the real channel is closed once the
        # stream task is cancelled, which would mask what _process_result sends.
        stream._event_ch = Mock()

        await stream._process_result({"type": "final", "text": "bonjour"})

        languages = [
            call.args[0].alternatives[0].language
            for call in stream._event_ch.send_nowait.call_args_list
            if call.args and call.args[0].alternatives
        ]
        assert languages, "no transcription event emitted"
        assert all(str(lang).lower() == "fr-medical" for lang in languages)
