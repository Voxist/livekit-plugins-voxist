"""Logging configuration for Voxist STT plugin."""

from __future__ import annotations

import logging
import re
import traceback


class SanitizingFilter(logging.Filter):
    """
    Log filter that sanitizes sensitive information from log records.

    SEC-007 FIX: Prevents credential leakage in log output by redacting:
    - API keys (api_key=..., voxist_... patterns)
    - Bearer tokens
    - JWT tokens in URLs

    CWE-532: Insertion of Sensitive Information into Log File

    Sanitization covers the message template (record.msg), %-style string
    arguments (record.args), and formatted exception tracebacks (via
    record.exc_text, which standard Formatters use as the traceback cache).
    Non-string values are left untouched so structured-logging handlers that
    expect e.g. a dict in record.msg keep receiving the original object.

    Known limitation: logger-level filters only run for records emitted on
    the logger they are attached to. Records emitted on child loggers (e.g.
    logging.getLogger("livekit.plugins.voxist.custom")) propagate directly
    to ancestor handlers without passing this filter; attach a
    SanitizingFilter to any such child logger or to the receiving handler.

    Example:
        >>> logger = logging.getLogger("test")
        >>> logger.addFilter(SanitizingFilter())
        >>> # "api_key=secret123" becomes "api_key=***REDACTED***"
        >>> # "voxist_abc123xyz" becomes "voxist_***"
        >>> # "token=eyJhbG..." becomes "token=***"
    """

    # Patterns to sanitize: (compiled_regex, replacement)
    # Order matters: more specific patterns before generic ones
    # Precompiled once - filter() runs on every record on the logging hot path
    SANITIZE_PATTERNS: list[tuple[re.Pattern[str], str]] = [
        (re.compile(pattern), replacement)
        for pattern, replacement in [
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
    ]

    @classmethod
    def _sanitize_text(cls, text: str) -> str:
        for pattern, replacement in cls.SANITIZE_PATTERNS:
            text = pattern.sub(replacement, text)
        return text

    @classmethod
    def _sanitize_value(cls, value: object) -> object:
        # Only strings can be scrubbed without changing the semantics of
        # %-style placeholders (%d, %f, %r on arbitrary objects).
        if isinstance(value, str):
            return cls._sanitize_text(value)
        return value

    def filter(self, record: logging.LogRecord) -> bool:
        """
        Sanitize a log record in place, covering msg, args and traceback.

        Args:
            record: The log record to sanitize

        Returns:
            True (always allows the log record after sanitization)
        """
        if isinstance(record.msg, str):
            record.msg = self._sanitize_text(record.msg)

        if record.args:
            if isinstance(record.args, dict):
                record.args = {
                    key: self._sanitize_value(value)
                    for key, value in record.args.items()
                }
            else:
                record.args = tuple(
                    self._sanitize_value(value) for value in record.args
                )

        # Pre-render the traceback into exc_text (the Formatter's cache slot)
        # so standard handlers emit the sanitized text instead of formatting
        # exc_info themselves.
        if record.exc_info and record.exc_info[0] is not None and not record.exc_text:
            formatted = "".join(traceback.format_exception(*record.exc_info))
            record.exc_text = self._sanitize_text(formatted.rstrip("\n"))

        return True


# Create logger for Voxist plugin
logger = logging.getLogger("livekit.plugins.voxist")

# Defer level control to the application: with NOTSET the effective level
# comes from ancestor loggers, so e.g. logging.basicConfig(level=DEBUG)
# surfaces the plugin's debug logs without plugin-specific configuration.
logger.setLevel(logging.NOTSET)

# Enable propagation to root logger for monitoring integrations
logger.propagate = True

# Library best practice: a NullHandler avoids "No handlers could be found"
# style fallbacks while leaving output to the application's handlers.
logger.addHandler(logging.NullHandler())

# Add sanitizing filter to prevent credential leakage in logs
logger.addFilter(SanitizingFilter())

# Fallback console handler attached by set_log_level() when the application
# has not configured any logging handlers of its own.
_fallback_handler: logging.Handler | None = None


def _has_configured_handler() -> bool:
    """Return True if a real (non-Null) handler would receive our records."""
    node: logging.Logger | None = logger
    while node is not None:
        for handler in node.handlers:
            if not isinstance(handler, logging.NullHandler):
                return True
        if not node.propagate:
            break
        node = node.parent
    return False


def _ensure_visible_output() -> None:
    """
    Attach a console handler if no application handler exists.

    Without this, set_log_level("DEBUG") in an app that never configured
    logging would be silently swallowed by logging.lastResort (WARNING+).
    The handler carries its own SanitizingFilter so records propagated from
    child loggers are also scrubbed before reaching the console.
    """
    global _fallback_handler
    if _fallback_handler is not None or _has_configured_handler():
        return

    handler = logging.StreamHandler()
    handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    handler.addFilter(SanitizingFilter())
    logger.addHandler(handler)
    _fallback_handler = handler


def set_log_level(level: str) -> None:
    """
    Set logging level for Voxist plugin.

    If the application has not configured any logging handlers, a console
    handler is attached so the requested output is actually visible.

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
    _ensure_visible_output()
    logger.info(f"Voxist plugin log level set to {level.upper()}")
