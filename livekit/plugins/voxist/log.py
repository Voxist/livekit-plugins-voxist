"""Logging configuration for Voxist STT plugin."""

import logging
import re


class SanitizingFilter(logging.Filter):
    """
    Log filter that sanitizes sensitive information from log messages.

    SEC-007 FIX: Prevents credential leakage in log output by redacting:
    - API keys (api_key=..., voxist_... patterns)
    - Bearer tokens
    - JWT tokens in URLs

    CWE-532: Insertion of Sensitive Information into Log File

    Example:
        >>> logger = logging.getLogger("test")
        >>> logger.addFilter(SanitizingFilter())
        >>> # "api_key=secret123" becomes "api_key=***REDACTED***"
        >>> # "voxist_abc123xyz" becomes "voxist_***"
        >>> # "token=eyJhbG..." becomes "token=***"
    """

    # Patterns to sanitize: (regex_pattern, replacement)
    # Order matters: more specific patterns before generic ones
    SANITIZE_PATTERNS: list[tuple[str, str]] = [
        # API key in URL parameter
        (r"api_key=([^&\s'\"]+)", r"api_key=***REDACTED***"),
        # Voxist API key format (voxist_xxx or VOXIST_xxx)
        (r"[Vv]oxist_[a-zA-Z0-9_-]+", "voxist_***"),
        # JWT tokens in URL (token=eyJ...) - must precede generic token pattern
        (r"token=eyJ[a-zA-Z0-9_.-]+", "token=***"),
        # Generic token parameter
        (r"token=([^&\s'\"]+)", r"token=***"),
        # Bearer tokens
        (r"Bearer\s+[a-zA-Z0-9_.-]+", "Bearer ***"),
        # X-API-Key header value (may appear in debug logs)
        (r'X-API-Key["\']?\s*:\s*["\']?[^"\'}\s,]+', "X-API-Key: ***"),
    ]

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Sanitize log message by removing sensitive information.

        Args:
            record: The log record to sanitize

        Returns:
            True (always allows the log record after sanitization)
        """
        msg = str(record.msg)
        for pattern, replacement in self.SANITIZE_PATTERNS:
            msg = re.sub(pattern, replacement, msg)
        record.msg = msg
        return True


# Create logger for Voxist plugin
logger = logging.getLogger("livekit.plugins.voxist")

# Default to INFO level, can be overridden by application
logger.setLevel(logging.INFO)

# Enable propagation to root logger for monitoring integrations
logger.propagate = True

# Add sanitizing filter to prevent credential leakage in logs
logger.addFilter(SanitizingFilter())


def set_log_level(level: str) -> None:
    """
    Set logging level for Voxist plugin.

    Args:
        level: One of "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"

    Example:
        from livekit.plugins.voxist.log import set_log_level
        set_log_level("DEBUG")  # Enable verbose logging
    """
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f"Invalid log level: {level}")

    logger.setLevel(numeric_level)
    logger.info(f"Voxist plugin log level set to {level.upper()}")
