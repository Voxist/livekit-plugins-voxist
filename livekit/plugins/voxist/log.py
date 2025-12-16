"""Logging configuration for Voxist STT plugin."""

import logging
import re


class SanitizingFormatter(logging.Formatter):
    """
    Log formatter that sanitizes sensitive information from log messages.

    SEC-007 FIX: Prevents credential leakage in log output by redacting:
    - API keys (api_key=..., voxist_... patterns)
    - Bearer tokens
    - JWT tokens in URLs

    CWE-532: Insertion of Sensitive Information into Log File

    Example:
        >>> formatter = SanitizingFormatter(fmt="%(message)s")
        >>> # "api_key=secret123" becomes "api_key=***REDACTED***"
        >>> # "voxist_abc123xyz" becomes "voxist_***"
        >>> # "token=eyJhbG..." becomes "token=***"
    """

    # Patterns to sanitize: (regex_pattern, replacement)
    # Order matters - more specific patterns first
    SANITIZE_PATTERNS: list[tuple[str, str]] = [
        # API key in URL parameter
        (r'api_key=([^&\s"\']+)', r'api_key=***REDACTED***'),
        # Voxist API key format (voxist_xxx or VOXIST_xxx)
        (r'[Vv]oxist_[a-zA-Z0-9_-]+', 'voxist_***'),
        # Bearer tokens
        (r'Bearer\s+[a-zA-Z0-9_\-\.]+', 'Bearer ***'),
        # JWT tokens in URL (token=eyJ...)
        (r'token=eyJ[a-zA-Z0-9_\-\.]+', 'token=***'),
        # Generic token parameter
        (r'token=([^&\s"\']+)', r'token=***'),
        # X-API-Key header value (may appear in debug logs)
        (r'X-API-Key["\']?\s*:\s*["\']?([^"\'}\s,]+)', 'X-API-Key: ***'),
    ]

    def format(self, record: logging.LogRecord) -> str:
        """
        Format log record with sensitive data sanitization.

        Args:
            record: The log record to format

        Returns:
            Formatted and sanitized log message
        """
        # Format the message first using parent
        original = super().format(record)

        # Apply sanitization patterns
        sanitized = original
        for pattern, replacement in self.SANITIZE_PATTERNS:
            sanitized = re.sub(pattern, replacement, sanitized)

        return sanitized


# Create logger for Voxist plugin
logger = logging.getLogger("livekit.plugins.voxist")

# Default to INFO level, can be overridden by application
logger.setLevel(logging.INFO)

# Create console handler with formatting
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.DEBUG)

# Format: timestamp - name - level - message
# SEC-007 FIX: Use SanitizingFormatter to prevent credential leakage
formatter = SanitizingFormatter(
    fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
console_handler.setFormatter(formatter)

# Add handler if not already added
if not logger.handlers:
    logger.addHandler(console_handler)

# Prevent propagation to root logger
logger.propagate = False


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
