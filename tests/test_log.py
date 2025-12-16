"""Unit tests for logging configuration and SEC-007 sanitization."""

import logging
import pytest

from livekit.plugins.voxist.log import SanitizingFormatter, logger, set_log_level


class TestSanitizingFormatter:
    """Test SEC-007 log sanitization functionality."""

    @pytest.fixture
    def formatter(self):
        """Create SanitizingFormatter for testing."""
        return SanitizingFormatter(fmt="%(message)s")

    def test_sanitize_api_key_in_url(self, formatter):
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
        result = formatter.format(record)
        assert "secret123abc" not in result
        assert "api_key=***REDACTED***" in result
        assert "lang=fr" in result

    def test_sanitize_voxist_api_key_format(self, formatter):
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
        result = formatter.format(record)
        assert "voxist_abc123_xyz789_secret" not in result
        assert "voxist_***" in result

    def test_sanitize_voxist_uppercase(self, formatter):
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
        result = formatter.format(record)
        assert "Voxist_TestKey123" not in result
        assert "voxist_***" in result

    def test_sanitize_bearer_token(self, formatter):
        """Test Bearer token is sanitized."""
        record = logging.LogRecord(
            name="test",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0",
            args=(),
            exc_info=None,
        )
        result = formatter.format(record)
        assert "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9" not in result
        assert "Bearer ***" in result

    def test_sanitize_jwt_token_in_url(self, formatter):
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
        result = formatter.format(record)
        assert "eyJhbGciOiJIUzI1NiJ9" not in result
        assert "token=***" in result
        assert "lang=fr" in result

    def test_sanitize_generic_token(self, formatter):
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
        result = formatter.format(record)
        assert "abc123xyz" not in result
        assert "token=***" in result
        assert "session=test" in result

    def test_sanitize_x_api_key_header(self, formatter):
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
        result = formatter.format(record)
        assert "secret_api_key_value" not in result
        assert "X-API-Key: ***" in result

    def test_sanitize_multiple_patterns(self, formatter):
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
        result = formatter.format(record)
        assert "voxist_key123" not in result
        assert "abc123" not in result
        assert "secret" not in result
        assert "voxist_***" in result
        assert "Bearer ***" in result
        assert "api_key=***REDACTED***" in result

    def test_no_sanitization_needed(self, formatter):
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
        result = formatter.format(record)
        assert result == "AudioProcessor initialized: chunk=1600 samples (100ms)"

    def test_partial_match_not_sanitized(self, formatter):
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
        result = formatter.format(record)
        assert "api_keys" in result  # Should NOT be sanitized

    def test_format_preserves_structure(self, formatter):
        """Test formatter preserves log message structure."""
        full_formatter = SanitizingFormatter(
            fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        record = logging.LogRecord(
            name="livekit.plugins.voxist",
            level=logging.INFO,
            pathname="test.py",
            lineno=1,
            msg="Test message with api_key=secret",
            args=(),
            exc_info=None,
        )
        result = full_formatter.format(record)
        assert "livekit.plugins.voxist" in result
        assert "INFO" in result
        assert "api_key=***REDACTED***" in result
        assert "secret" not in result

    def test_sanitize_patterns_constant_exists(self):
        """Test SANITIZE_PATTERNS constant is defined."""
        assert hasattr(SanitizingFormatter, 'SANITIZE_PATTERNS')
        assert len(SanitizingFormatter.SANITIZE_PATTERNS) >= 5


class TestLoggerConfiguration:
    """Test logger setup and configuration."""

    def test_logger_name(self):
        """Test logger has correct name."""
        assert logger.name == "livekit.plugins.voxist"

    def test_logger_default_level(self):
        """Test logger default level is INFO."""
        assert logger.level == logging.INFO

    def test_logger_has_handler(self):
        """Test logger has at least one handler configured."""
        assert len(logger.handlers) >= 1

    def test_handler_uses_sanitizing_formatter(self):
        """Test handler uses SanitizingFormatter."""
        # Find the console handler
        for handler in logger.handlers:
            if hasattr(handler, 'formatter') and handler.formatter:
                assert isinstance(handler.formatter, SanitizingFormatter)
                return
        pytest.fail("No handler with SanitizingFormatter found")

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

    def test_propagate_disabled(self):
        """Test logger doesn't propagate to root logger."""
        assert logger.propagate is False


class TestLogSanitizationIntegration:
    """Integration tests for log sanitization in real logging scenarios."""

    def test_actual_log_message_sanitized(self, capsys):
        """
        Test that actual log output is sanitized.

        Note: We verify by checking that the logger's handler uses
        SanitizingFormatter, which is already tested in unit tests above.
        This test verifies the formatter is actually applied to logged messages.
        """
        import io

        # Create a fresh handler with a string buffer we can inspect
        string_buffer = io.StringIO()
        test_handler = logging.StreamHandler(string_buffer)
        test_handler.setFormatter(SanitizingFormatter(fmt="%(message)s"))

        # Add our test handler
        logger.addHandler(test_handler)

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

    def test_debug_log_with_voxist_key_sanitized(self):
        """Test DEBUG level logs sanitize voxist keys."""
        import io

        # Create a fresh handler with a string buffer we can inspect
        string_buffer = io.StringIO()
        test_handler = logging.StreamHandler(string_buffer)
        test_handler.setFormatter(SanitizingFormatter(fmt="%(message)s"))
        test_handler.setLevel(logging.DEBUG)

        # Add our test handler
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
