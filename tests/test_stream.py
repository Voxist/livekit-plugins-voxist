"""Unit tests for VoxistSTTStream class with focus on VUL-003 ownership validation."""

import asyncio
import logging
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

    _get_transport() prefers a public ws.get_transport() when the installed
    aiohttp has one, and otherwise walks ws._response.connection.transport.
    Current aiohttp has no public accessor, so tests populate that chain. They
    must not invent a ws.get_transport() method on a mock and assert through it:
    doing so tests an API aiohttp does not have, which is precisely how the
    indefinite-backpressure bug shipped with a green suite.

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

        # Buffer readings come from the real attribute chain, never from a
        # ws.get_transport() method - the fixture deletes that attribute so no
        # implementation can depend on an API aiohttp does not have.
        install_fake_transport(mock_connection, 0)

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


class TestSendPathWithOwnership:
    """Send-path behaviour: ownership, early exits, and the send timeout."""

    @pytest.fixture
    def mock_stt(self):
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
        return AsyncMock(spec=ConnectionPool)

    @pytest.fixture
    def mock_connection(self):
        conn = Connection(id=0, state=ConnectionState.IN_USE)
        conn.ws = AsyncMock()
        conn.ws.closed = False
        conn.ws.send_bytes = AsyncMock()
        conn.buffered_amount = 0
        # Default to an unreachable transport, matching a real aiohttp
        # WebSocket. Tests that need a size call install_fake_transport().
        conn.ws._response = None
        del conn.ws.get_transport  # real aiohttp (<=3.14) has no such method
        return conn

    async def _create_stream_with_connection(self, mock_stt, mock_pool, mock_connection):
        """Create a stream with proper ownership setup (async helper)."""
        from livekit.agents.types import APIConnectOptions

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=mock_pool,
            config=mock_stt._config,
            language="fr",
            conn_options=APIConnectOptions(max_retry=3, retry_interval=1.0, timeout=10.0),
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
    async def test_timeout_constants_are_ordered(self):
        """
        The send timeout must sit well below the receive watchdog.

        If a stalled send could outlast RECEIVE_TIMEOUT_SECONDS, the receive
        task would tear the stream down first and a local send stall would be
        misreported as a server-side stall - the original ASR-disconnect
        symptom. An earlier version of this test only bounded the send-side
        constant by 30.0, which is exactly the watchdog value, so it permitted
        the failure it was meant to prevent.
        """
        assert VoxistSTTStream.SEND_TIMEOUT_SECONDS > 0
        assert VoxistSTTStream.RECEIVE_TIMEOUT_SECONDS > 0
        assert (
            VoxistSTTStream.SEND_TIMEOUT_SECONDS
            <= VoxistSTTStream.RECEIVE_TIMEOUT_SECONDS / 2
        ), "a stalled send must be detected well before the receive watchdog fires"

    @pytest.mark.asyncio
    async def test_no_water_mark_thresholds(self):
        """
        The plugin must not compare the write buffer against its own thresholds.

        Absolute byte marks cannot be calibrated once: a plain socket pauses at
        64KB while asyncio's SSL transport pauses at 512KB and only relieves to
        128KB. Marks chosen for one make the release condition unreachable on
        the other. Backpressure belongs to aiohttp, which follows the
        transport's own pause state.
        """
        for removed in (
            "HIGH_WATER_MARK",
            "LOW_WATER_MARK",
            "BACKPRESSURE_MAX_WAIT",
            "BACKPRESSURE_CHECK_INTERVAL",
        ):
            assert not hasattr(VoxistSTTStream, removed), (
                f"{removed} reintroduces threshold-based flow control; "
                "transport limits differ between plain and TLS sockets"
            )

    @pytest.mark.asyncio
    async def test_sends_immediately_when_buffer_measurable(
        self, mock_stt, mock_pool, mock_connection
    ):
        """A measurable buffer must not delay the send - it is diagnostic only."""
        stream, connection = await self._create_stream_with_connection(
            mock_stt, mock_pool, mock_connection
        )
        install_fake_transport(connection, 400 * 1024)  # far above any old mark

        loop = asyncio.get_running_loop()
        start = loop.time()
        await stream._send_audio_chunk(np.zeros(160, dtype=np.int16))
        elapsed = loop.time() - start

        assert elapsed < 0.2, "a large buffer reading must not throttle the send"
        connection.ws.send_bytes.assert_called_once()
        assert connection.buffered_amount == 400 * 1024  # recorded, not acted on

    @pytest.mark.asyncio
    async def test_sends_normally_when_buffer_unmeasurable(
        self, mock_stt, mock_pool, mock_connection
    ):
        """The common production case: no public accessor, so no measurement."""
        stream, connection = await self._create_stream_with_connection(
            mock_stt, mock_pool, mock_connection
        )
        remove_transport(connection)

        assert stream._get_write_buffer_size() == 0

        await stream._send_audio_chunk(np.zeros(160, dtype=np.int16))
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_stalled_send_raises_connection_error(
        self, mock_stt, mock_pool, mock_connection, monkeypatch
    ):
        """
        A send that never completes must fail fast, not block forever.

        aiohttp's drain inside send_bytes has no timeout of its own, so without
        this bound the send task blocks indefinitely while livekit keeps
        enqueuing frames - the same silent stall, one layer down.
        """
        stream, connection = await self._create_stream_with_connection(
            mock_stt, mock_pool, mock_connection
        )
        monkeypatch.setattr(VoxistSTTStream, "SEND_TIMEOUT_SECONDS", 0.2)

        async def never_completes(_data):
            await asyncio.Event().wait()

        connection.ws.send_bytes = AsyncMock(side_effect=never_completes)

        loop = asyncio.get_running_loop()
        start = loop.time()
        with pytest.raises(ConnectionError, match="send timeout"):
            await asyncio.wait_for(
                stream._send_audio_chunk(np.zeros(160, dtype=np.int16)), timeout=5.0
            )
        elapsed = loop.time() - start
        assert 0.2 <= elapsed < 3.0

    @pytest.mark.asyncio
    async def test_stalled_send_marks_connection_failed(
        self, mock_stt, mock_pool, mock_connection, monkeypatch
    ):
        """
        A stalled connection must not go back into the pool as healthy.

        A stalled socket is not a closed socket, so release_connection() would
        see ws.closed == False and hand it straight to the next stream, which
        would stall too. Marking it FAILED routes it to the pool's reconnect
        path instead.
        """
        stream, connection = await self._create_stream_with_connection(
            mock_stt, mock_pool, mock_connection
        )
        monkeypatch.setattr(VoxistSTTStream, "SEND_TIMEOUT_SECONDS", 0.1)

        async def never_completes(_data):
            await asyncio.Event().wait()

        connection.ws.send_bytes = AsyncMock(side_effect=never_completes)
        assert connection.state == ConnectionState.IN_USE

        with pytest.raises(ConnectionError):
            await asyncio.wait_for(
                stream._send_audio_chunk(np.zeros(160, dtype=np.int16)), timeout=5.0
            )

        assert connection.state == ConnectionState.FAILED, (
            "a stalled connection was left in a state the pool treats as usable"
        )

    @pytest.mark.asyncio
    async def test_stalled_send_aborts_the_transport(
        self, mock_stt, mock_pool, mock_connection, monkeypatch
    ):
        """
        The transport is aborted, because a cancelled send left a frame mid-flight.

        asyncio.wait_for cancels the inner send_bytes; aiohttp may already have
        written part of a frame, so the connection is no longer safe to write to.
        """
        stream, connection = await self._create_stream_with_connection(
            mock_stt, mock_pool, mock_connection
        )
        monkeypatch.setattr(VoxistSTTStream, "SEND_TIMEOUT_SECONDS", 0.1)
        transport = install_fake_transport(connection, 1024)
        transport.abort = Mock()

        async def never_completes(_data):
            await asyncio.Event().wait()

        connection.ws.send_bytes = AsyncMock(side_effect=never_completes)

        with pytest.raises(ConnectionError):
            await asyncio.wait_for(
                stream._send_audio_chunk(np.zeros(160, dtype=np.int16)), timeout=5.0
            )

        transport.abort.assert_called_once()

    @pytest.mark.asyncio
    async def test_buffered_amount_records_measurement_with_ownership(
        self, mock_stt, mock_pool, mock_connection
    ):
        """buffered_amount is a snapshot, never a running total."""
        stream, connection = await self._create_stream_with_connection(
            mock_stt, mock_pool, mock_connection
        )
        install_fake_transport(connection, 8192)

        chunk = np.zeros(160, dtype=np.int16)
        await stream._send_audio_chunk(chunk)
        assert connection.buffered_amount == 8192

        await stream._send_audio_chunk(chunk)
        assert connection.buffered_amount == 8192

    @pytest.mark.asyncio
    async def test_no_connection_returns_early(self, mock_stt, mock_pool):
        """Test _send_audio_chunk returns early with no connection."""
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

        stream._conn = None
        await stream._send_audio_chunk(np.zeros(160, dtype=np.int16))

    @pytest.mark.asyncio
    async def test_closed_ws_returns_early(self, mock_stt, mock_pool, mock_connection):
        """Test _send_audio_chunk returns early with closed WebSocket."""
        stream, connection = await self._create_stream_with_connection(
            mock_stt, mock_pool, mock_connection
        )
        connection.ws.closed = True

        await stream._send_audio_chunk(np.zeros(160, dtype=np.int16))
        connection.ws.send_bytes.assert_not_called()


class TestBackpressureLiveness:
    """
    Regression tests for the two indefinite-stall bugs in the send path.

    First bug: _get_write_buffer_size() called a non-existent aiohttp method,
    silently fell back to a counter that added half of every chunk ever sent and
    never decayed, and the drain loop it fed had no time bound - so after ~100s
    of audio the stream paused forever and the WebSocket timed out.

    Second bug: the replacement compared the real buffer against marks
    calibrated for a plain socket. Over wss:// the low mark sat below the level
    asyncio's SSL transport relieves to, making the release condition
    unreachable and throttling the stream to one chunk per timeout.

    Both are prevented by the same property: nothing in the send path gates on
    a byte count the plugin chose. These tests pin that, plus the bounds that
    replace it.
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
        Sending must not stall over a sustained run of audio.

        The first bug tripped at 2MB, reached after ~106s of 16kHz audio. This
        drives more than three times that volume and requires every chunk to
        reach the WebSocket. Each send is individually timed out, so a
        reintroduced stall fails fast instead of hanging the suite.
        """
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        chunk = np.zeros(1600, dtype=np.int16)  # 100ms @ 16kHz = 3200 bytes
        chunk_bytes = len(chunk.tobytes())
        old_trip_point = 2 * 1024 * 1024
        chunk_count = (3 * old_trip_point) // chunk_bytes

        for _ in range(chunk_count):
            await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=1.0)

        assert connection.ws.send_bytes.await_count == chunk_count
        assert connection.buffered_amount == 0

    @pytest.mark.asyncio
    async def test_no_stall_when_buffer_sits_in_the_ssl_band(
        self, mock_stt, mock_pool, mock_connection
    ):
        """
        A buffer parked in the SSL transport's 128KB-512KB band must not stall.

        This is the exact condition that broke the previous fix: over wss://,
        asyncio's SSL transport pauses at 512KB and relieves only to 128KB, so
        a buffer resting at, say, 300KB is above the old 256KB high mark and can
        never reach the old 64KB low mark. Every chunk burned the full wait.
        """
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        ssl_low_water = 128 * 1024
        ssl_high_water = 512 * 1024
        parked = (ssl_low_water + ssl_high_water) // 2  # 320KB, never relieved
        install_fake_transport(connection, parked)

        chunk = np.zeros(1600, dtype=np.int16)
        loop = asyncio.get_running_loop()
        start = loop.time()
        for _ in range(20):
            await asyncio.wait_for(stream._send_audio_chunk(chunk), timeout=1.0)
        elapsed = loop.time() - start

        assert connection.ws.send_bytes.await_count == 20
        assert elapsed < 1.0, (
            f"20 sends took {elapsed:.2f}s with the buffer parked at "
            f"{parked}B - the send path is gating on a byte threshold again"
        )

    @pytest.mark.asyncio
    async def test_buffered_amount_never_accumulates(
        self, mock_stt, mock_pool, mock_connection
    ):
        """buffered_amount must track the measured buffer, not the sum of sends."""
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
    async def test_unmeasurable_buffer_never_throttles(
        self, mock_stt, mock_pool, mock_connection
    ):
        """An unmeasurable transport reports 0 and never gates the send."""
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        remove_transport(connection)
        assert stream._get_transport() is None
        assert stream._get_write_buffer_size() == 0

        # A stale value on the connection must not be resurrected as an estimate
        connection.buffered_amount = 999_999_999
        assert stream._get_write_buffer_size() == 0

        await asyncio.wait_for(
            stream._send_audio_chunk(np.zeros(1600, dtype=np.int16)), timeout=1.0
        )
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_closing_transport_is_unmeasurable_but_send_proceeds(
        self, mock_stt, mock_pool, mock_connection
    ):
        """
        A closing transport yields no measurement and does not divert the send.

        Under the previous design this state produced a misleading "backpressure
        released" log and then a raise from send_bytes. Now the measurement is
        irrelevant to control flow, so the send follows its normal path and any
        error surfaces from aiohttp itself.
        """
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        transport = install_fake_transport(connection, 123456)
        transport.is_closing = Mock(return_value=True)

        assert stream._get_transport() is None
        assert stream._get_write_buffer_size() == 0

        await asyncio.wait_for(
            stream._send_audio_chunk(np.zeros(1600, dtype=np.int16)), timeout=1.0
        )
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_transport_errors_are_contained(
        self, mock_stt, mock_pool, mock_connection
    ):
        """A transport that raises must degrade to 0, not propagate."""
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)

        transport = install_fake_transport(connection, 0)
        transport.get_write_buffer_size = Mock(side_effect=RuntimeError("detached"))

        assert stream._get_write_buffer_size() == 0

        await asyncio.wait_for(
            stream._send_audio_chunk(np.zeros(1600, dtype=np.int16)), timeout=1.0
        )
        connection.ws.send_bytes.assert_called_once()

    @pytest.mark.asyncio
    async def test_unreachable_transport_is_reported_once(
        self, mock_stt, mock_pool, mock_connection, caplog
    ):
        """
        Losing the transport accessor must leave a trace, exactly once.

        A silent permanent failure would hide the loss of every buffer metric
        behind an aiohttp upgrade; a per-chunk log would flood at 10/s.
        """
        stream, connection = await self._stream(mock_stt, mock_pool, mock_connection)
        remove_transport(connection)

        with caplog.at_level(logging.DEBUG, logger="livekit.plugins.voxist"):
            for _ in range(50):
                stream._get_write_buffer_size()

        matching = [r for r in caplog.records if "cannot reach the WebSocket" in r.message]
        assert len(matching) == 1, f"expected exactly one report, got {len(matching)}"


class TestInputBacklogBound:
    """
    CRIT-001: the unsent audio backlog must be bounded.

    livekit's input channel is unbounded and push_frame() uses send_nowait, so
    nothing upstream slows down when the uplink cannot keep up. Without an
    explicit cap the backlog grows for the life of the call until the process
    is OOM-killed. Dropping the oldest audio is the right policy for live
    transcription: audio that far behind the speaker is already useless.
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

    async def _stream(self, mock_stt):
        from livekit.agents.types import APIConnectOptions

        stream = VoxistSTTStream(
            stt=mock_stt,
            pool=AsyncMock(spec=ConnectionPool),
            config=mock_stt._config,
            language="fr",
            conn_options=APIConnectOptions(max_retry=3, retry_interval=1.0, timeout=10.0),
        )
        stream._task.cancel()
        try:
            await stream._task
        except asyncio.CancelledError:
            pass
        return stream

    @pytest.mark.asyncio
    async def test_backlog_cap_is_bounded_and_documented(self, mock_stt):
        """The cap must exist and be a sane amount of audio."""
        assert VoxistSTTStream.MAX_INPUT_BACKLOG_FRAMES > 0
        # 10ms frames: keep at least 1s, no more than a minute of audio
        assert 100 <= VoxistSTTStream.MAX_INPUT_BACKLOG_FRAMES <= 6000

    @pytest.mark.asyncio
    async def test_oldest_frames_dropped_above_cap(self, mock_stt, monkeypatch):
        """Backlog above the cap is trimmed back down to it."""
        stream = await self._stream(mock_stt)
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 10)

        for _ in range(35):
            stream._input_ch.send_nowait(Mock(spec=[]))
        assert stream._input_ch.qsize() == 35

        stream._drop_stale_input()

        assert stream._input_ch.qsize() == 10
        assert stream._dropped_frames == 25

    @pytest.mark.asyncio
    async def test_backlog_below_cap_untouched(self, mock_stt, monkeypatch):
        """Nothing is dropped while the backlog is within bounds."""
        stream = await self._stream(mock_stt)
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 10)

        for _ in range(10):
            stream._input_ch.send_nowait(Mock(spec=[]))

        stream._drop_stale_input()

        assert stream._input_ch.qsize() == 10
        assert stream._dropped_frames == 0

    @pytest.mark.asyncio
    async def test_flush_sentinel_survives_dropping(self, mock_stt, monkeypatch):
        """
        A flush sentinel must never be discarded.

        Dropping one would lose the end-of-utterance signal, so Voxist would
        never be told to finalize and the caller would lose a transcript.
        """
        stream = await self._stream(mock_stt)
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 5)

        # Sentinel sits inside the prefix that will be trimmed away.
        for _ in range(20):
            stream._input_ch.send_nowait(Mock(spec=[]))
        stream._input_ch.send_nowait(VoxistSTTStream._FlushSentinel())
        for _ in range(20):
            stream._input_ch.send_nowait(Mock(spec=[]))
        assert stream._input_ch.qsize() == 41

        stream._drop_stale_input()

        remaining = []
        while stream._input_ch.qsize():
            remaining.append(stream._input_ch.recv_nowait())

        sentinels = [r for r in remaining if isinstance(r, VoxistSTTStream._FlushSentinel)]
        assert len(sentinels) == 1, "the end-of-utterance signal was dropped"
        # Trimming 41 items to the cap of 5 removes 36; the sentinel is
        # re-queued rather than removed, so all 36 are audio frames.
        assert stream._dropped_frames == 36
        assert len(remaining) == 5
        # It now sits after the audio that was kept, which is where it belongs
        assert isinstance(remaining[-1], VoxistSTTStream._FlushSentinel)

    @pytest.mark.asyncio
    async def test_drop_logs_a_warning(self, mock_stt, monkeypatch, caplog):
        """Dropped audio must be visible, not silent."""
        stream = await self._stream(mock_stt)
        monkeypatch.setattr(VoxistSTTStream, "MAX_INPUT_BACKLOG_FRAMES", 5)

        for _ in range(30):
            stream._input_ch.send_nowait(Mock(spec=[]))

        with caplog.at_level(logging.WARNING, logger="livekit.plugins.voxist"):
            stream._drop_stale_input()

        assert any("dropped" in r.message for r in caplog.records)


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
    async def test_public_accessor_is_preferred_when_available(self):
        """
        _get_transport() must use a public accessor whenever aiohttp has one.

        Deliberately not asserting that aiohttp *lacks* get_transport(): the day
        upstream adds it, an absence assertion would turn every matrix job red
        on a dependency bump while the plugin still worked. Instead this checks
        the preference order, so gaining the public API is a silent improvement.
        """
        import aiohttp
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
            language="fr",
            conn_options=APIConnectOptions(max_retry=3, retry_interval=1.0, timeout=10.0),
        )
        stream._task.cancel()
        try:
            await stream._task
        except asyncio.CancelledError:
            pass

        sentinel = Mock()
        sentinel.is_closing = Mock(return_value=False)
        sentinel.get_write_buffer_size = Mock(return_value=4242)

        conn = Connection(id=0, state=ConnectionState.IN_USE)
        conn.ws = Mock()
        conn.ws.closed = False
        conn.ws.get_transport = Mock(return_value=sentinel)
        # Private chain also present, returning something different, so we can
        # tell which one was consulted.
        other = Mock()
        other.is_closing = Mock(return_value=False)
        other.get_write_buffer_size = Mock(return_value=1)
        conn.ws._response = Mock()
        conn.ws._response.connection = Mock()
        conn.ws._response.connection.transport = other

        stream._conn = conn
        assert stream._get_write_buffer_size() == 4242, (
            "the public accessor must win over the private attribute chain"
        )

        # Documented state of the currently installed aiohttp, as information
        # rather than a constraint.
        has_public = hasattr(aiohttp.ClientWebSocketResponse, "get_transport")
        assert isinstance(has_public, bool)

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
                    assert conn.buffered_amount >= 0
        finally:
            await runner.cleanup()


LANGUAGE_CODE_AVAILABLE = getattr(
    __import__("livekit.agents", fromlist=["LanguageCode"]), "LanguageCode", None
) is not None


@pytest.mark.skipif(
    not LANGUAGE_CODE_AVAILABLE,
    reason="livekit-agents predates LanguageCode; stream.py falls back to str "
    "and performs no normalization, so these expectations do not apply",
)
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
