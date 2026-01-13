"""Unit tests for logging configuration and SEC-007 sanitization."""

import logging

import pytest

from livekit.plugins.voxist.log import SanitizingFilter, logger, set_log_level


class TestSanitizingFilter:
    """Test SEC-007 log sanitization functionality."""

    @pytest.fixture
    def log_filter(self):
        """Create SanitizingFilter for testing."""
        return SanitizingFilter()

    def test_sanitize_api_key_in_url(self, log_filter):
        """Test api_key=xxx pattern is sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Connecting to wss://api.voxist.com/ws?api_key=secret123abc&lang=fr",
            args=(),
            exc_info=None,
        )
        log_filter.filter(record)
        assert "secret123abc" not in record.msg
        assert "api_key=***REDACTED***" in record.msg
        assert "lang=fr" in record.msg

    def test_sanitize_voxist_api_key_format(self, log_filter):
        """Test voxist_xxx API key format is sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Using API key: voxist_abc123_xyz789_secret",
            args=(),
            exc_info=None,
        )
        log_filter.filter(record)
        assert "voxist_abc123_xyz789_secret" not in record.msg
        assert "voxist_***" in record.msg

    def test_sanitize_voxist_uppercase(self, log_filter):
        """Test VOXIST_xxx API key format is sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="API key is Voxist_TestKey123",
            args=(),
            exc_info=None,
        )
        log_filter.filter(record)
        assert "Voxist_TestKey123" not in record.msg
        assert "voxist_***" in record.msg

    def test_sanitize_bearer_token(self, log_filter):
        """Test Bearer token is sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg=(
                "Authorization: Bearer "
                "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0"
            ),
            args=(),
            exc_info=None,
        )
        log_filter.filter(record)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in record.msg
        assert "Bearer ***" in record.msg

    def test_sanitize_jwt_token_in_url(self, log_filter):
        """Test JWT token in URL parameter is sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Connecting to ws://localhost:8765/ws?token=eyJhbGciOiJIUzI1NiJ9.payload.signature&lang=fr",
            args=(),
            exc_info=None,
        )
        log_filter.filter(record)
        assert "eyJhbGciOiJIUzI1NiJ9" not in record.msg
        assert "token=***" in record.msg
        assert "lang=fr" in record.msg

    def test_sanitize_generic_token(self, log_filter):
        """Test generic token parameter is sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Received token=abc123xyz&session=test",
            args=(),
            exc_info=None,
        )
        log_filter.filter(record)
        assert "abc123xyz" not in record.msg
        assert "token=***" in record.msg
        assert "session=test" in record.msg

    def test_sanitize_x_api_key_header(self, log_filter):
        """Test X-API-Key header value is sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg='Headers: {"X-API-Key": "secret_api_key_value"}',
            args=(),
            exc_info=None,
        )
        log_filter.filter(record)
        assert "secret_api_key_value" not in record.msg
        assert "X-API-Key: ***" in record.msg

    def test_sanitize_multiple_patterns(self, log_filter):
        """Test multiple patterns in same message are all sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Using voxist_key123 with Bearer abc123 and api_key=secret",
            args=(),
            exc_info=None,
        )
        log_filter.filter(record)
        assert "voxist_key123" not in record.msg
        assert "abc123" not in record.msg
        assert "secret" not in record.msg
        assert "voxist_***" in record.msg
        assert "Bearer ***" in record.msg
        assert "api_key=***REDACTED***" in record.msg

    def test_no_sanitization_needed(self, log_filter):
        """Test messages without sensitive data pass through unchanged."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="AudioProcessor initialized: chunk=1600 samples (100ms)",
            args=(),
            exc_info=None,
        )
        original_msg = record.msg
        log_filter.filter(record)
        assert record.msg == original_msg

    def test_partial_match_not_sanitized(self, log_filter):
        """Test partial matches are not incorrectly sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Processing api_keys config option",  # Note: api_keys not api_key=
            args=(),
            exc_info=None,
        )
        original_msg = record.msg
        log_filter.filter(record)
        assert record.msg == original_msg  # Should NOT be sanitized

    def test_sanitize_patterns_constant_exists(self):
        """Test SANITIZE_PATTERNS constant is defined."""
        assert hasattr(SanitizingFilter, "SANITIZE_PATTERNS")
        assert len(SanitizingFilter.SANITIZE_PATTERNS) >= 5


class TestLoggerConfiguration:
    """Test logger setup and configuration."""

    def test_logger_name(self):
        """Test logger has correct name."""
        assert logger.name == "livekit.plugins.voxist"

    def test_logger_default_level(self):
        """Test logger default level is INFO."""
        assert logger.level == logging.INFO

    def test_logger_has_sanitizing_filter(self):
        """Test logger has SanitizingFilter installed."""
        # Find the sanitizing filter
        for log_filter in logger.filters:
            if isinstance(log_filter, SanitizingFilter):
                return
        pytest.fail("No SanitizingFilter found on logger")

    def test_propagate_enabled(self):
        """Test logger propagates to root logger."""
        assert logger.propagate is True

    def test_no_console_handler(self):
        """Test logger does not have its own console handler."""
        # Logger should not have direct handlers - logs propagate to root
        assert len(logger.handlers) == 0

    def test_set_log_level_valid(self):
        """Test set_log_level accepts valid levels."""
        original_level = logger.level

        try:
            set_log_level("DEBUG")
            assert logger.level == logging.DEBUG

            set_log_level("WARNING")
            assert logger.level == logging.WARNING

            set_log_level("error")  # lowercase should work
            assert logger.level == logging.ERROR
        finally:
            # Restore original level
            logger.setLevel(original_level)

    def test_set_log_level_invalid(self):
        """Test set_log_level raises for invalid levels."""
        with pytest.raises(ValueError, match="Invalid log level"):
            set_log_level("INVALID")


class TestLogSanitizationIntegration:
    """Integration tests for log sanitization in real logging scenarios."""

    def test_actual_log_message_sanitized(self, caplog):
        """Test that actual log output is sanitized."""
        import io

        # Create a test handler to capture logs
        string_buffer = io.StringIO()
        test_handler = logging.StreamHandler(string_buffer)
        test_handler.setLevel(logging.INFO)

        # Add handler to our logger
        logger.addHandler(test_handler)
        original_level = logger.level
        logger.setLevel(logging.INFO)

        try:
            # Log a message with sensitive data
            logger.info("Connection URL: ws://test.com?api_key=secret123&lang=fr")

            # Get the output from our buffer
            output = string_buffer.getvalue()

            # The actual secret should not appear in output
            assert "secret123" not in output
            assert "api_key=***REDACTED***" in output
        finally:
            logger.removeHandler(test_handler)
            logger.setLevel(original_level)

    def test_debug_log_with_voxist_key_sanitized(self, caplog):
        """Test DEBUG level logs sanitize voxist keys."""
        import io

        # Create a test handler to capture logs
        string_buffer = io.StringIO()
        test_handler = logging.StreamHandler(string_buffer)
        test_handler.setLevel(logging.DEBUG)

        # Add handler to our logger
        logger.addHandler(test_handler)
        original_level = logger.level
        logger.setLevel(logging.DEBUG)

        try:
            logger.debug("Using key voxist_test_key_12345 for authentication")

            output = string_buffer.getvalue()
            assert "voxist_test_key_12345" not in output
            assert "voxist_***" in output
        finally:
            logger.removeHandler(test_handler)
            logger.setLevel(original_level)
