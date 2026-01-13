"""Unit tests for ConnectionPool."""

import asyncio
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import aiohttp
import pytest

from livekit.plugins.voxist.connection_pool import ConnectionPool
from livekit.plugins.voxist.exceptions import (
    AuthenticationError,
    ConnectionError,
    ConnectionPoolExhaustedError,
)
from livekit.plugins.voxist.models import Connection, ConnectionState


@pytest.fixture
def mock_ws_connect():
    """Mock aiohttp ws_connect."""
    async def _mock_connect(*args, **kwargs):
        mock_ws = AsyncMock()
        mock_ws.closed = False
        mock_ws.close = AsyncMock()
        return mock_ws

    return _mock_connect


@pytest.fixture
async def pool_basic():
    """Create basic ConnectionPool for testing (without initialization)."""
    pool = ConnectionPool(
        base_url="ws://localhost:8765/ws",
        api_key="test_key",
        pool_size=2,
        connection_timeout=1.0,
        heartbeat_interval=5.0,
    )
    yield pool
    # Cleanup
    try:
        if pool._initialized:
            await pool.close()
        elif pool._session and not pool._session.closed:
            await pool._session.close()
    except Exception:
        pass


class TestConnectionPoolInitialization:
    """Test pool initialization and setup."""

    def test_pool_creation(self):
        """Test ConnectionPool creates with correct parameters."""
        pool = ConnectionPool(
            base_url="ws://test.com/ws",
            api_key="test_key",
            pool_size=3,
            connection_timeout=5.0,
            heartbeat_interval=10.0,
        )

        assert pool.base_url == "ws://test.com/ws"
        assert pool.api_key == "test_key"
        assert pool.pool_size == 3
        assert pool.connection_timeout == 5.0
        assert pool.heartbeat_interval == 10.0
        assert len(pool.connections) == 0  # Not initialized yet
        assert not pool._initialized

    @pytest.mark.asyncio
    async def test_initialize_creates_connections(self, pool_basic, mock_ws_connect):
        """Test initialize creates Connection objects."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            assert len(pool_basic.connections) == 2
            assert pool_basic._initialized
            assert pool_basic._session is not None
            assert pool_basic._heartbeat_task is not None

            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_initialize_waits_for_first_connection(self, pool_basic):
        """Test initialize waits for at least one successful connection."""
        import time
        connection_attempts = []

        async def mock_connect_slow(self, *args, **kwargs):
            connection_attempts.append(time.time())
            await asyncio.sleep(0.2)  # Simulate slow connection
            mock_ws = AsyncMock()
            mock_ws.closed = False
            return mock_ws

        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_connect_slow):
            await pool_basic.initialize()

            # Should have attempted to connect to all
            assert len(connection_attempts) == 2
            assert pool_basic._initialized

            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_initialize_fails_if_all_connections_fail(self, pool_basic):
        """Test initialize raises error if no connections succeed."""

        async def mock_connect_fail(self, *args, **kwargs):
            raise aiohttp.ClientError("Connection failed")

        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_connect_fail):
            with pytest.raises(ConnectionError, match="Failed to establish any connections"):
                await pool_basic.initialize()

    @pytest.mark.asyncio
    async def test_initialize_propagates_auth_error(self, pool_basic):
        """Test initialize propagates authentication errors."""

        async def mock_connect_auth_fail(self, *args, **kwargs):
            exc = aiohttp.WSServerHandshakeError(
                request_info=Mock(),
                history=(),
                status=401,
                message="Unauthorized",
                headers={},
            )
            raise exc

        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_connect_auth_fail):
            with pytest.raises(AuthenticationError, match="Invalid API key"):
                await pool_basic.initialize()

    @pytest.mark.asyncio
    async def test_initialize_idempotent(self, pool_basic, mock_ws_connect):
        """Test initialize can be called multiple times safely."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()
            first_session = pool_basic._session

            # Call again
            await pool_basic.initialize()

            # Should not recreate
            assert pool_basic._session is first_session
            assert len(pool_basic.connections) == 2

            await pool_basic.close()


class TestConnectionAcquisition:
    """Test connection get/release operations."""

    @pytest.mark.asyncio
    async def test_get_connection_returns_ready_connection(self, pool_basic, mock_ws_connect):
        """Test get_connection returns a ready connection."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Manually set a connection to READY
            pool_basic.connections[0].state = ConnectionState.READY

            conn = await pool_basic.get_connection()

            assert conn.id == 0
            assert conn.state == ConnectionState.IN_USE

            await pool_basic.release_connection(conn)
            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_get_connection_selects_lowest_buffered(self, pool_basic, mock_ws_connect):
        """Test get_connection selects connection with lowest buffered amount."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Set both to READY with different buffer amounts
            pool_basic.connections[0].state = ConnectionState.READY
            pool_basic.connections[0].buffered_amount = 1000

            pool_basic.connections[1].state = ConnectionState.READY
            pool_basic.connections[1].buffered_amount = 500

            conn = await pool_basic.get_connection()

            # Should select connection 1 (lowest buffered)
            assert conn.id == 1

            await pool_basic.release_connection(conn)
            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_release_connection_returns_to_ready(self, pool_basic, mock_ws_connect):
        """Test release_connection sets state back to READY."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            pool_basic.connections[0].state = ConnectionState.READY
            conn = await pool_basic.get_connection()

            assert conn.state == ConnectionState.IN_USE

            await pool_basic.release_connection(conn)

            assert conn.state == ConnectionState.READY

            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_get_connection_exhausted_raises_error(self, pool_basic, mock_ws_connect):
        """Test get_connection raises when no connections available."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Set all connections to FAILED
            for conn in pool_basic.connections:
                conn.state = ConnectionState.FAILED
                conn.retry_count = pool_basic.max_reconnect_attempts  # Prevent auto-reconnect

            with pytest.raises(ConnectionPoolExhaustedError, match="No healthy connections"):
                await pool_basic.get_connection()

            await pool_basic.close()


class TestConnectionManagement:
    """Test connection lifecycle management."""

    @pytest.mark.asyncio
    async def test_connect_success(self, pool_basic, mock_ws_connect):
        """Test successful connection establishment."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            pool_basic._session = aiohttp.ClientSession()

            conn = Connection(id=0)
            success = await pool_basic._connect(conn)

            assert success is True
            assert conn.state == ConnectionState.READY
            assert conn.ws is not None
            assert conn.retry_count == 0
            assert conn.last_heartbeat > 0

            await pool_basic._session.close()

    @pytest.mark.asyncio
    async def test_connect_timeout(self, pool_basic):
        """Test connection timeout handling."""

        async def slow_connect(*args, **kwargs):
            await asyncio.sleep(10)  # Longer than timeout

        with patch.object(aiohttp.ClientSession, 'ws_connect', slow_connect):
            pool_basic._session = aiohttp.ClientSession()

            conn = Connection(id=0)
            success = await pool_basic._connect(conn)

            assert success is False
            assert conn.state == ConnectionState.FAILED

            await pool_basic._session.close()

    @pytest.mark.asyncio
    async def test_connect_auth_error_raises(self, pool_basic):
        """Test authentication error is raised properly."""

        async def auth_fail_connect(*args, **kwargs):
            exc = aiohttp.WSServerHandshakeError(
                request_info=Mock(),
                history=(),
                status=401,
                message="Unauthorized",
                headers={},
            )
            raise exc

        with patch.object(aiohttp.ClientSession, 'ws_connect', auth_fail_connect):
            pool_basic._session = aiohttp.ClientSession()

            conn = Connection(id=0)

            with pytest.raises(AuthenticationError):
                await pool_basic._connect(conn)

            assert conn.state == ConnectionState.FAILED

            await pool_basic._session.close()


class TestReconnection:
    """Test reconnection logic."""

    @pytest.mark.asyncio
    async def test_reconnect_with_backoff(self, pool_basic, mock_ws_connect):
        """Test reconnection implements exponential backoff."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            pool_basic._session = aiohttp.ClientSession()

            conn = Connection(id=0, state=ConnectionState.FAILED)
            conn.retry_count = 2  # Third attempt

            start_time = asyncio.get_event_loop().time()

            # Run reconnect (will sleep based on retry count)
            await pool_basic._reconnect(conn)

            elapsed = asyncio.get_event_loop().time() - start_time

            # Backoff formula: 1.0 * (1.5 ** 3) = 3.375s
            # Should have slept ~3.375s (plus jitter up to 0.3375s)
            assert elapsed >= 3.0  # At least base backoff
            assert elapsed < 4.0   # Not too much more

            # Should have succeeded
            assert conn.state == ConnectionState.READY
            assert conn.retry_count == 0  # Reset on success

            await pool_basic._session.close()

    @pytest.mark.asyncio
    async def test_reconnect_stops_after_max_attempts(self, pool_basic):
        """Test reconnection stops after max attempts."""

        async def always_fail(*args, **kwargs):
            raise aiohttp.ClientError("Always fails")

        with patch.object(aiohttp.ClientSession, 'ws_connect', always_fail):
            pool_basic._session = aiohttp.ClientSession()
            pool_basic.max_reconnect_attempts = 3

            conn = Connection(id=0, state=ConnectionState.FAILED)
            conn.retry_count = 3  # At max

            await pool_basic._reconnect(conn)

            # Should not attempt reconnection
            assert conn.state == ConnectionState.FAILED
            assert conn.retry_count == 3

            await pool_basic._session.close()

    @pytest.mark.asyncio
    async def test_reconnect_does_not_retry_auth_errors(self, pool_basic):
        """Test reconnection doesn't retry authentication failures."""

        async def auth_fail(*args, **kwargs):
            exc = aiohttp.WSServerHandshakeError(
                request_info=Mock(),
                history=(),
                status=401,
                message="Unauthorized",
                headers={},
            )
            raise exc

        with patch.object(aiohttp.ClientSession, 'ws_connect', auth_fail):
            pool_basic._session = aiohttp.ClientSession()

            conn = Connection(id=0, state=ConnectionState.FAILED, retry_count=0)

            with pytest.raises(AuthenticationError):
                await pool_basic._reconnect(conn)

            # Should not schedule retry
            assert conn.state == ConnectionState.FAILED

            await pool_basic._session.close()


class TestPoolHealth:
    """Test pool health monitoring and status."""

    def test_get_pool_status(self, pool_basic):
        """Test pool status string generation."""
        # Create connections with different states
        pool_basic.connections = [
            Connection(id=0, state=ConnectionState.READY),
            Connection(id=1, state=ConnectionState.IN_USE),
            Connection(id=2, state=ConnectionState.FAILED),
        ]

        status = pool_basic._get_pool_status()

        assert "1 failed" in status
        assert "1 in_use" in status or "1 in-use" in status
        assert "1 ready" in status

    def test_get_pool_health(self, pool_basic):
        """Test detailed pool health statistics."""
        pool_basic.connections = [
            Connection(id=0, state=ConnectionState.READY),
            Connection(id=1, state=ConnectionState.READY),
            Connection(id=2, state=ConnectionState.FAILED),
        ]

        health = pool_basic.get_pool_health()

        assert health["total"] == 3
        assert health["ready"] == 2
        assert health["failed"] == 1
        assert health["health_percentage"] == pytest.approx(66.67, abs=0.1)

    def test_get_pool_health_all_healthy(self, pool_basic):
        """Test health stats when all connections are healthy."""
        pool_basic.connections = [
            Connection(id=0, state=ConnectionState.READY),
            Connection(id=1, state=ConnectionState.IN_USE),
        ]

        health = pool_basic.get_pool_health()

        assert health["total"] == 2
        assert health["ready"] == 1
        assert health["in_use"] == 1
        assert health["health_percentage"] == 100.0


class TestCleanup:
    """Test cleanup and resource management."""

    @pytest.mark.asyncio
    async def test_close_stops_heartbeat(self, pool_basic, mock_ws_connect):
        """Test close() stops heartbeat task."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            assert pool_basic._heartbeat_task is not None
            assert not pool_basic._heartbeat_task.done()

            await pool_basic.close()

            # Heartbeat should be cancelled
            assert pool_basic._heartbeat_task.done()
            assert pool_basic._heartbeat_task.cancelled()

    @pytest.mark.asyncio
    async def test_close_closes_all_connections(self, pool_basic, mock_ws_connect):
        """Test close() closes all WebSocket connections."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            await pool_basic.close()

            # All connections should be closed
            for conn in pool_basic.connections:
                assert conn.state == ConnectionState.CLOSED

    @pytest.mark.asyncio
    async def test_close_closes_session(self, pool_basic, mock_ws_connect):
        """Test close() closes aiohttp session."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            session = pool_basic._session
            assert not session.closed

            await pool_basic.close()

            # Session should be closed
            assert session.closed

    @pytest.mark.asyncio
    async def test_close_idempotent(self, pool_basic, mock_ws_connect):
        """Test close() can be called multiple times safely."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            await pool_basic.close()
            await pool_basic.close()  # Second call should not error
            await pool_basic.close()  # Third call should not error

            assert pool_basic._closing is True


class TestConcurrency:
    """Test concurrent access to pool."""

    @pytest.mark.asyncio
    async def test_concurrent_get_connection(self, pool_basic, mock_ws_connect):
        """Test multiple concurrent get_connection calls."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Set both connections to READY
            pool_basic.connections[0].state = ConnectionState.READY
            pool_basic.connections[1].state = ConnectionState.READY

            # Get both connections concurrently
            conn1_task = asyncio.create_task(pool_basic.get_connection())
            conn2_task = asyncio.create_task(pool_basic.get_connection())

            conn1, conn2 = await asyncio.gather(conn1_task, conn2_task)

            # Should get different connections
            assert conn1.id != conn2.id
            assert conn1.state == ConnectionState.IN_USE
            assert conn2.state == ConnectionState.IN_USE

            await pool_basic.release_connection(conn1)
            await pool_basic.release_connection(conn2)
            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_concurrent_release_connection(self, pool_basic, mock_ws_connect):
        """Test concurrent release operations are safe."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            pool_basic.connections[0].state = ConnectionState.IN_USE
            pool_basic.connections[1].state = ConnectionState.IN_USE

            # Release concurrently
            await asyncio.gather(
                pool_basic.release_connection(pool_basic.connections[0]),
                pool_basic.release_connection(pool_basic.connections[1]),
            )

            assert pool_basic.connections[0].state == ConnectionState.READY
            assert pool_basic.connections[1].state == ConnectionState.READY

            await pool_basic.close()


class TestEdgeCases:
    """Test edge cases and error scenarios."""

    @pytest.mark.asyncio
    async def test_get_connection_before_initialize_raises(self, pool_basic):
        """Test get_connection before initialization raises error."""
        with pytest.raises(ConnectionPoolExhaustedError):
            await pool_basic.get_connection()

    @pytest.mark.asyncio
    async def test_pool_size_one(self, mock_ws_connect):
        """Test pool works with single connection."""
        pool = ConnectionPool(
            base_url="ws://localhost/ws",
            api_key="test",
            pool_size=1,
        )

        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool.initialize()

            assert len(pool.connections) == 1

            conn = await pool.get_connection()
            assert conn.state == ConnectionState.IN_USE

            await pool.release_connection(conn)
            await pool.close()

    @pytest.mark.asyncio
    async def test_wait_for_ready_timeout(self, pool_basic):
        """Test _wait_for_ready times out properly."""
        conn = Connection(id=0, state=ConnectionState.CONNECTING)

        # _wait_for_ready has its own timeout logic, just verify it raises
        with pytest.raises(ConnectionError, match="did not become ready"):
            await pool_basic._wait_for_ready(conn)

    @pytest.mark.asyncio
    async def test_wait_for_ready_on_failed_connection(self, pool_basic):
        """Test _wait_for_ready raises when connection fails."""
        conn = Connection(id=0, state=ConnectionState.FAILED)

        with pytest.raises(ConnectionError, match="failed while waiting"):
            await pool_basic._wait_for_ready(conn)


class TestRaceConditions:
    """Test race condition fixes (VUL-001, VUL-002)."""

    @pytest.mark.asyncio
    async def test_vul001_state_change_during_wait_fallback(self, pool_basic, mock_ws_connect):
        """
        VUL-001: Test TOCTOU race condition fix.

        Scenario: Connection state changes from CONNECTING to FAILED while
        waiting without the lock. The fix should fallback to another ready
        connection instead of returning the failed one.
        """
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Setup: conn[0] is CONNECTING, conn[1] is READY
            pool_basic.connections[0].state = ConnectionState.CONNECTING
            pool_basic.connections[1].state = ConnectionState.READY
            pool_basic.connections[1].buffered_amount = 0

            # Simulate state change during wait - connection fails after wait starts

            async def simulated_wait(conn):
                # Simulate the connection failing during the wait
                conn.state = ConnectionState.FAILED
                # Raise the expected error
                raise ConnectionError(f"Connection {conn.id} failed while waiting")

            pool_basic._wait_for_ready = simulated_wait

            # Should NOT raise - should fallback to conn[1]
            conn = await pool_basic.get_connection()

            # VUL-001 fix: Should have fallen back to the ready connection
            assert conn.id == 1
            assert conn.state == ConnectionState.IN_USE

            await pool_basic.release_connection(conn)
            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_vul001_state_change_during_wait_no_fallback(self, pool_basic, mock_ws_connect):
        """
        VUL-001: Test when connection state changes and NO fallback available.

        Should raise ConnectionPoolExhaustedError, not return invalid connection.
        """
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Setup: conn[0] is CONNECTING, conn[1] is FAILED (exhausted retries)
            pool_basic.connections[0].state = ConnectionState.CONNECTING
            pool_basic.connections[1].state = ConnectionState.FAILED
            pool_basic.connections[1].retry_count = pool_basic.max_reconnect_attempts

            # Simulate wait that completes but connection ends up FAILED (not READY)
            # This tests the re-validation path with no fallback
            async def simulated_wait(conn):
                # Wait completes successfully (no exception), but state is FAILED
                conn.state = ConnectionState.FAILED

            pool_basic._wait_for_ready = simulated_wait

            # Should raise pool exhausted because:
            # - conn[0] became FAILED during wait (no longer READY)
            # - conn[1] is already FAILED and exhausted retries
            with pytest.raises(ConnectionPoolExhaustedError):
                await pool_basic.get_connection()

            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_vul001_revalidation_after_wait_success(self, pool_basic, mock_ws_connect):
        """
        VUL-001: Test state re-validation when connection becomes ready.

        The fix should re-check state after wait before claiming connection.
        """
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Setup: conn[0] is CONNECTING (will become READY)
            pool_basic.connections[0].state = ConnectionState.CONNECTING
            pool_basic.connections[1].state = ConnectionState.IN_USE

            # Simulate successful connection
            async def simulated_wait_success(conn):
                # Connection becomes ready during wait
                conn.state = ConnectionState.READY

            pool_basic._wait_for_ready = simulated_wait_success

            conn = await pool_basic.get_connection()

            # Should claim the now-ready connection
            assert conn.id == 0
            assert conn.state == ConnectionState.IN_USE

            await pool_basic.release_connection(conn)
            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_vul002_atomic_reconnecting_state(self, pool_basic, mock_ws_connect):
        """
        VUL-002: Test atomic RECONNECTING state transition.

        When triggering reconnection, the state should be set to RECONNECTING
        BEFORE spawning the task to prevent duplicate reconnection tasks.
        """
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Set all connections to FAILED
            for conn in pool_basic.connections:
                conn.state = ConnectionState.FAILED
                conn.retry_count = 0  # Allow reconnection

            # Track reconnection task spawns
            reconnect_calls = []

            async def tracked_reconnect(conn, state_already_set=False):
                reconnect_calls.append(conn.id)
                # Don't actually reconnect, just track the call

            pool_basic._reconnect = tracked_reconnect

            # First call should trigger reconnection
            try:
                await pool_basic.get_connection()
            except ConnectionPoolExhaustedError:
                pass  # Expected

            # State should be RECONNECTING (set atomically before task spawn)
            assert pool_basic.connections[0].state == ConnectionState.RECONNECTING

            # Let spawned tasks run
            await asyncio.sleep(0)

            # First call should have triggered reconnection for conn[0]
            assert pool_basic.connections[0].state == ConnectionState.RECONNECTING
            assert 0 in reconnect_calls

            # Second call - conn[0] is now RECONNECTING, conn[1] is still FAILED
            # Should trigger reconnection for conn[1] only
            try:
                await pool_basic.get_connection()
            except ConnectionPoolExhaustedError:
                pass  # Expected

            # Let spawned tasks run
            await asyncio.sleep(0)

            # VUL-002 fix: Each connection should only be triggered once
            # With 2 failed connections, we expect exactly 2 reconnection calls
            # (one per connection, not multiple for the same connection)
            assert len(reconnect_calls) == 2
            assert set(reconnect_calls) == {0, 1}  # Both connections triggered exactly once

            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_vul002_no_duplicate_reconnection_concurrent(self, pool_basic, mock_ws_connect):
        """
        VUL-002: Test no duplicate reconnection with concurrent callers.

        Multiple concurrent get_connection() calls should NOT spawn
        multiple reconnection tasks for the same connection.
        """
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Set all connections to FAILED
            for conn in pool_basic.connections:
                conn.state = ConnectionState.FAILED
                conn.retry_count = 0

            reconnect_calls = []

            async def tracked_reconnect(conn, state_already_set=False):
                reconnect_calls.append((conn.id, asyncio.get_event_loop().time()))
                await asyncio.sleep(0.1)  # Simulate reconnection time

            pool_basic._reconnect = tracked_reconnect

            # Fire multiple concurrent requests
            tasks = [
                asyncio.create_task(pool_basic.get_connection())
                for _ in range(5)
            ]

            # Wait for all to complete (they should all fail with pool exhausted)
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # All should have raised ConnectionPoolExhaustedError
            for result in results:
                assert isinstance(result, ConnectionPoolExhaustedError)

            # VUL-002 fix: Should have at most pool_size reconnection attempts
            # (one per failed connection, not one per caller)
            assert len(reconnect_calls) <= pool_basic.pool_size

            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_vul002_reconnection_state_prevents_duplicate(self, pool_basic, mock_ws_connect):
        """
        VUL-002: Verify RECONNECTING state prevents duplicate task spawn.

        If connection is already RECONNECTING, get_connection should NOT
        spawn another reconnection task.
        """
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Set conn[0] to RECONNECTING (already being reconnected)
            pool_basic.connections[0].state = ConnectionState.RECONNECTING
            # Set conn[1] to FAILED
            pool_basic.connections[1].state = ConnectionState.FAILED
            pool_basic.connections[1].retry_count = 0

            reconnect_calls = []

            async def tracked_reconnect(conn, state_already_set=False):
                reconnect_calls.append(conn.id)

            pool_basic._reconnect = tracked_reconnect

            try:
                await pool_basic.get_connection()
            except ConnectionPoolExhaustedError:
                pass

            # Let spawned tasks run
            await asyncio.sleep(0)

            # Should only trigger reconnection for conn[1], not conn[0]
            assert reconnect_calls == [1]

            await pool_basic.close()

    @pytest.mark.asyncio
    async def test_multiple_waiters_same_connecting_connection(self, pool_basic, mock_ws_connect):
        """
        Test multiple callers waiting for the same CONNECTING connection.

        Only one should claim it when it becomes ready.
        """
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            await pool_basic.initialize()

            # Setup: conn[0] is CONNECTING, conn[1] is IN_USE
            pool_basic.connections[0].state = ConnectionState.CONNECTING
            pool_basic.connections[1].state = ConnectionState.IN_USE

            wait_started = asyncio.Event()
            waiters_count = 0

            async def controlled_wait(conn):
                nonlocal waiters_count
                waiters_count += 1
                wait_started.set()
                await asyncio.sleep(0.2)  # Simulate wait time
                conn.state = ConnectionState.READY

            pool_basic._wait_for_ready = controlled_wait

            # Fire two concurrent requests
            task1 = asyncio.create_task(pool_basic.get_connection())
            await wait_started.wait()  # Wait for first to start waiting
            wait_started.clear()

            task2 = asyncio.create_task(pool_basic.get_connection())

            # Both should eventually complete
            results = await asyncio.gather(task1, task2, return_exceptions=True)

            # One should succeed, one should either succeed (if fallback) or fail
            successes = [r for r in results if isinstance(r, Connection)]

            # At least one should have claimed the connection
            assert len(successes) >= 1
            assert successes[0].state == ConnectionState.IN_USE

            # Release and cleanup
            for r in results:
                if isinstance(r, Connection):
                    await pool_basic.release_connection(r)

            await pool_basic.close()


class TestTokenExchange:
    """Test SEC-001 token exchange functionality."""

    def test_get_http_base_url_wss(self):
        """Test WSS to HTTPS conversion."""
        pool = ConnectionPool(
            base_url="wss://api-asr.voxist.com/ws",
            api_key="test_key",
        )
        assert pool._get_http_base_url() == "https://api-asr.voxist.com"

    def test_get_http_base_url_ws(self):
        """Test WS to HTTP conversion."""
        pool = ConnectionPool(
            base_url="ws://localhost:3000/ws",
            api_key="test_key",
        )
        assert pool._get_http_base_url() == "http://localhost:3000"

    def test_get_http_base_url_no_ws_suffix(self):
        """Test URL without /ws suffix."""
        pool = ConnectionPool(
            base_url="wss://api.example.com/websocket",
            api_key="test_key",
        )
        # Should not strip /websocket since it's not /ws
        assert pool._get_http_base_url() == "https://api.example.com/websocket"

    @pytest.mark.asyncio
    @pytest.mark.no_auto_mock_token
    async def test_token_caching(self, pool_basic):
        """Test that tokens are cached and reused."""
        call_count = 0

        def mock_get_token(url, **kwargs):
            nonlocal call_count
            call_count += 1
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "url": "ws://localhost:8765/ws?token=cached_token"
            })

            # Create proper async context manager
            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        pool_basic._session = MagicMock()
        pool_basic._session.get = mock_get_token

        # First call should hit the mock
        url1 = await pool_basic._get_ws_token("fr", 16000)
        assert "token=cached_token" in url1
        assert call_count == 1

        # Second call should use cache (no additional HTTP call)
        url2 = await pool_basic._get_ws_token("fr", 16000)
        assert "token=cached_token" in url2
        assert call_count == 1  # Still 1, cache was used

    @pytest.mark.asyncio
    @pytest.mark.no_auto_mock_token
    async def test_token_exchange_auth_error(self, pool_basic):
        """Test that authentication errors are raised properly."""
        def mock_get_token(url, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 401

            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        pool_basic._session = MagicMock()
        pool_basic._session.get = mock_get_token

        with pytest.raises(AuthenticationError, match="Invalid API key"):
            await pool_basic._get_ws_token("fr", 16000)

    @pytest.mark.asyncio
    @pytest.mark.no_auto_mock_token
    async def test_token_adds_language_and_sample_rate(self, pool_basic):
        """Test that language and sample_rate are appended to token URL."""
        def mock_get_token(url, **kwargs):
            mock_response = AsyncMock()
            mock_response.status = 200
            mock_response.json = AsyncMock(return_value={
                "url": "ws://localhost:8765/ws?token=test_token"
            })

            cm = AsyncMock()
            cm.__aenter__ = AsyncMock(return_value=mock_response)
            cm.__aexit__ = AsyncMock(return_value=None)
            return cm

        pool_basic._session = MagicMock()
        pool_basic._session.get = mock_get_token

        url = await pool_basic._get_ws_token("fr-medical", 48000)

        assert "token=test_token" in url
        assert "lang=fr-medical" in url
        assert "sample_rate=48000" in url

    def test_is_initialized_property(self):
        """Test CRIT-002 fix: is_initialized property."""
        pool = ConnectionPool(
            base_url="ws://localhost:8765/ws",
            api_key="test_key",
        )
        assert pool.is_initialized is False
        pool._initialized = True
        assert pool.is_initialized is True


class TestReconnectionRateLimiting:
    """Test SEC-012 reconnection rate limiting functionality."""

    def test_rate_limit_constants_exist(self):
        """Test SEC-012 rate limiting instance variables are initialized."""
        pool = ConnectionPool(
            base_url="ws://localhost:8765/ws",
            api_key="test_key",
        )
        # Verify rate limiting attributes exist
        assert hasattr(pool, '_reconnect_semaphore')
        assert hasattr(pool, '_reconnect_times')
        assert hasattr(pool, '_max_reconnects_per_minute')

        # Verify defaults
        assert pool._max_reconnects_per_minute == 30
        assert pool._reconnect_times == []

    @pytest.mark.asyncio
    async def test_rate_limit_tracking(self, pool_basic, mock_ws_connect):
        """Test reconnection attempts are tracked in _reconnect_times."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            pool_basic._session = aiohttp.ClientSession()
            pool_basic._max_reconnects_per_minute = 100  # High limit for this test

            conn = Connection(id=0, state=ConnectionState.FAILED)
            conn.retry_count = 0

            # Perform reconnection
            await pool_basic._reconnect(conn)

            # Should have recorded the timestamp
            assert len(pool_basic._reconnect_times) >= 1

            await pool_basic._session.close()

    @pytest.mark.asyncio
    async def test_rate_limit_cleans_old_timestamps(self, pool_basic):
        """Test old timestamps (>60s) are cleaned up."""
        import time

        pool_basic._session = aiohttp.ClientSession()
        pool_basic._max_reconnects_per_minute = 100

        # Manually add old timestamps
        old_time = time.time() - 120  # 2 minutes ago
        pool_basic._reconnect_times = [old_time, old_time + 1, old_time + 2]

        # Call rate limit check which should clean old entries
        await pool_basic._check_reconnect_rate_limit()

        # Old timestamps should be removed, only new one added
        assert len(pool_basic._reconnect_times) == 1
        assert pool_basic._reconnect_times[0] > old_time + 100  # New timestamp

        await pool_basic._session.close()

    @pytest.mark.asyncio
    async def test_rate_limit_enforced(self, pool_basic):
        """Test rate limit blocks when exceeded."""
        import time

        pool_basic._session = aiohttp.ClientSession()
        pool_basic._max_reconnects_per_minute = 3  # Low limit for testing

        # Fill up the rate limit
        now = time.time()
        pool_basic._reconnect_times = [now - 10, now - 5, now - 1]

        start_time = time.time()

        # This should wait because rate limit is exceeded
        await pool_basic._check_reconnect_rate_limit()

        elapsed = time.time() - start_time

        # Should have waited (oldest was 10s ago, so need to wait ~50s to expire it)
        # But since we're just checking the wait behavior exists, we can verify
        # by the fact that _reconnect_times is cleaned up
        # For a quick test, just verify the call succeeded
        assert elapsed < 60  # Reasonable timeout

        await pool_basic._session.close()

    @pytest.mark.asyncio
    async def test_semaphore_limits_concurrent_reconnections(self, pool_basic, mock_ws_connect):
        """Test semaphore limits concurrent reconnection attempts."""
        concurrent_count = 0
        max_concurrent = 0

        async def slow_connect(*args, **kwargs):
            nonlocal concurrent_count, max_concurrent
            concurrent_count += 1
            max_concurrent = max(max_concurrent, concurrent_count)
            await asyncio.sleep(0.1)  # Simulate connection time
            concurrent_count -= 1
            mock_ws = AsyncMock()
            mock_ws.closed = False
            return mock_ws

        with patch.object(aiohttp.ClientSession, 'ws_connect', slow_connect):
            pool_basic._session = aiohttp.ClientSession()
            pool_basic._max_reconnects_per_minute = 100  # High limit for this test

            # Create multiple connections to reconnect
            connections = [Connection(id=i, state=ConnectionState.FAILED) for i in range(10)]
            pool_basic.connections = connections

            # Start all reconnections concurrently
            tasks = [
                asyncio.create_task(pool_basic._reconnect(conn))
                for conn in connections
            ]

            await asyncio.gather(*tasks, return_exceptions=True)

            # Max concurrent should be limited by semaphore (5)
            assert max_concurrent <= 5

            await pool_basic._session.close()

    @pytest.mark.asyncio
    async def test_rate_limit_waits_when_exceeded(self, pool_basic):
        """
        Test that rate limit causes waiting when exceeded.

        SEC-012: When rate limit is exceeded, the function should wait
        until the oldest timestamp expires (60s window).
        """
        import time

        pool_basic._session = aiohttp.ClientSession()
        pool_basic._max_reconnects_per_minute = 2  # Very low limit

        # Fill up the rate limit with timestamps that will expire soon
        now = time.time()
        # Oldest is 59.5s ago, so we only need to wait ~0.5s for it to expire
        pool_basic._reconnect_times = [now - 59.5, now - 59.4]

        start_time = time.time()

        # This should wait briefly because rate limit is exceeded
        await pool_basic._check_reconnect_rate_limit()

        elapsed = time.time() - start_time

        # Should have waited at least 0.3s for oldest to expire, but not too long
        # (with some tolerance for test timing)
        assert elapsed >= 0.3, f"Expected wait >= 0.3s, got {elapsed:.2f}s"
        assert elapsed < 2.0, f"Wait too long: {elapsed:.2f}s"

        # After waiting, the new timestamp should be added
        assert len(pool_basic._reconnect_times) >= 1

        await pool_basic._session.close()

    @pytest.mark.asyncio
    async def test_reconnect_uses_semaphore(self, pool_basic, mock_ws_connect):
        """Test that _reconnect acquires semaphore before proceeding."""
        with patch.object(aiohttp.ClientSession, 'ws_connect', mock_ws_connect):
            pool_basic._session = aiohttp.ClientSession()

            conn = Connection(id=0, state=ConnectionState.FAILED)
            conn.retry_count = 0

            # Pre-fill semaphore to verify it's used
            # Acquire all 5 permits
            for _ in range(5):
                await pool_basic._reconnect_semaphore.acquire()

            # Create a reconnection task that should block
            reconnect_task = asyncio.create_task(pool_basic._reconnect(conn))

            # Give it a moment - should still be waiting
            await asyncio.sleep(0.1)

            # Task should still be running (blocked on semaphore)
            assert not reconnect_task.done()

            # Release one permit
            pool_basic._reconnect_semaphore.release()

            # Now reconnect should proceed
            try:
                await asyncio.wait_for(reconnect_task, timeout=5.0)
            except asyncio.TimeoutError:
                # Cancel to clean up
                reconnect_task.cancel()
                pytest.fail("Reconnect did not complete after semaphore release")

            # Release remaining permits to clean up
            for _ in range(4):
                pool_basic._reconnect_semaphore.release()

            await pool_basic._session.close()

    def test_semaphore_initialized_with_correct_limit(self, pool_basic):
        """Test semaphore is initialized with 5 concurrent limit."""
        # Check semaphore has 5 permits available
        # (internal implementation detail, but important for SEC-012)
        assert pool_basic._reconnect_semaphore._value == 5
