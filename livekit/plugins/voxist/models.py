"""Data models and enums for Voxist STT plugin."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import Enum

import aiohttp

# SEC-002 FIX: Regex pattern for language code validation
# Matches formats like: fr, en, de, fr-FR, en-US, fr-medical
# Pattern: 2 lowercase letters, optionally followed by hyphen and 2+ lowercase letters
# Using \Z instead of $ to ensure no trailing newlines are accepted
LANGUAGE_FORMAT_PATTERN = re.compile(r'^[a-z]{2}(-[a-zA-Z]{2,8})?\Z')


def validate_language_format(language: str) -> bool:
    """
    Validate language code format using regex.

    SEC-002 FIX: Defense-in-depth validation beyond allowlist check.
    Prevents potential injection if SUPPORTED_LANGUAGES is modified incorrectly.

    Args:
        language: Language code to validate

    Returns:
        True if format is valid, False otherwise

    Examples:
        >>> validate_language_format("fr")
        True
        >>> validate_language_format("en-US")
        True
        >>> validate_language_format("fr-medical")
        True
        >>> validate_language_format("fr; DROP TABLE users")
        False
    """
    if not language or not isinstance(language, str):
        return False
    return bool(LANGUAGE_FORMAT_PATTERN.match(language))


def sanitize_url_param(value: str) -> str:
    """
    Sanitize a value for safe URL parameter inclusion.

    SEC-002 FIX: Ensures special characters are properly encoded.

    Args:
        value: Parameter value to sanitize

    Returns:
        URL-safe encoded string
    """
    from urllib.parse import quote
    return quote(str(value), safe='')


class ConnectionState(Enum):
    """WebSocket connection states."""
    CONNECTING = "connecting"
    READY = "ready"
    IN_USE = "in_use"
    CLOSING = "closing"
    CLOSED = "closed"
    FAILED = "failed"
    RECONNECTING = "reconnecting"


@dataclass
class Connection:
    """
    Represents a single WebSocket connection in the pool.

    Attributes:
        id: Unique connection identifier (0-indexed)
        ws: aiohttp WebSocket connection (None if not connected)
        state: Current connection state
        retry_count: Number of reconnection attempts
        last_heartbeat: Timestamp of last successful heartbeat
        buffered_amount: Estimated buffered bytes (for backpressure)
    """
    id: int
    ws: aiohttp.ClientWebSocketResponse | None = None
    state: ConnectionState = ConnectionState.CLOSED
    retry_count: int = 0
    last_heartbeat: float = 0.0
    buffered_amount: int = 0


# Supported languages with descriptions
SUPPORTED_LANGUAGES = {
    'fr': 'French (Standard)',
    'fr-FR': 'French (Standard)',
    'fr-medical': 'French (Medical with text2num and medical units processing)',
    'en': 'English',
    'en-US': 'English',
    'de': 'German',
    'de-DE': 'German',
    'it': 'Italian',
    'es': 'Spanish',
    'nl': 'Dutch',
    'nl-NL': 'Dutch',
    'pt': 'Portuguese',
    'sv': 'Swedish',
}


# Default configuration values
DEFAULT_CONFIG = {
    'sample_rate': 16000,
    'base_url': 'wss://api-asr.voxist.com/ws',
    'language': 'fr',
    'interim_results': True,
    'connection_pool_size': 2,
    'connection_timeout': 10.0,
    'heartbeat_interval': 30.0,
    'chunk_duration_ms': 100,
    'stride_overlap_ms': 20,
    'max_reconnect_attempts': 10,
    'reconnect_backoff': 1.0,
    'max_backoff': 30.0,
}
