"""Custom exceptions for Voxist STT plugin."""


class VoxistError(Exception):
    """Base exception for all Voxist plugin errors."""
    pass


class AuthenticationError(VoxistError):
    """
    Raised when API key authentication fails.

    This typically indicates:
    - Invalid API key format
    - Expired or revoked API key
    - API key lacks transcription permissions

    Resolution:
    - Verify VOXIST_API_KEY environment variable
    - Check API key in Voxist dashboard
    - Ensure API key has correct permissions
    """
    pass


class InsufficientBalanceError(VoxistError):
    """
    Raised when wallet balance is too low for transcription.

    WebSocket close code: 1008

    Resolution:
    - Add credits to Voxist wallet
    - Check current balance in dashboard
    """
    pass


class ConnectionError(VoxistError):
    """
    Raised when WebSocket connection fails.

    This can indicate:
    - Network connectivity issues
    - Invalid WebSocket URL
    - Server unavailable
    - Firewall blocking WebSocket connections

    Resolution:
    - Check network connectivity
    - Verify base_url is correct
    - Check server status
    """
    pass


class ConnectionPoolExhaustedError(ConnectionError):
    """
    Raised when all connections in the pool have failed.

    This indicates a systemic issue:
    - Voxist API unavailable
    - Network outage
    - Invalid configuration affecting all connections

    Resolution:
    - Check Voxist API status
    - Verify network connectivity
    - Review error logs for specific failure causes
    - Consider increasing connection_timeout
    """
    pass


class LanguageNotSupportedError(VoxistError):
    """
    Raised when requested language is not supported.

    Supported languages:
    - fr, fr-FR: French (standard)
    - fr-medical: French with medical text processing
    - en, en-US: English
    - de, de-DE: German
    - it: Italian
    - es: Spanish
    - nl, nl-NL: Dutch
    - pt: Portuguese
    - sv: Swedish

    Resolution:
    - Use a supported language code
    - Check for typos in language parameter
    """
    pass


class ConfigurationError(VoxistError):
    """
    Raised when plugin configuration is invalid.

    Common causes:
    - Missing required parameters (e.g., api_key)
    - Invalid parameter values (e.g., negative sample_rate)
    - Conflicting configuration options

    Resolution:
    - Review configuration parameters
    - Check documentation for valid values
    - Verify environment variables are set
    """
    pass


class BackpressureError(VoxistError):
    """
    Raised when WebSocket buffer is consistently full.

    This indicates:
    - Sending audio faster than network can handle
    - Server processing slower than audio rate
    - Network congestion

    Resolution:
    - Reduce chunk_duration_ms
    - Increase connection_pool_size
    - Check network bandwidth
    """
    pass


class OwnershipViolationError(VoxistError):
    """
    Raised when a stream attempts to modify connection state without ownership.

    Security: VUL-003 mitigation
    This error indicates a race condition where multiple streams are
    attempting to access the same connection concurrently.

    This is a programming error and should not occur in normal operation.
    If this error is raised, it indicates:
    - A bug in stream lifecycle management
    - Improper connection pool usage
    - Concurrent access to a connection

    Resolution:
    - Report this error - it indicates a bug in the plugin
    - Ensure each stream exclusively owns its connection
    - Check for improper connection sharing
    """
    pass


class TranscriptLostError(VoxistError):
    """
    Raised when streamed audio was consumed but produced no transcript, and
    no retry can recover it.

    Streamed audio cannot be replayed: once frames have left the input
    channel, a fresh connection would receive nothing but a bare "Done" and
    fabricate an empty success. When that state is reached with zero
    FINAL_TRANSCRIPT events delivered, the session's content is gone.

    Deliberately NOT an APIError (same design as AuthenticationError):
    livekit's RecognizeStream._main_task retries every APIError - the
    installed version does not consult `retryable` - so raising
    APIError(retryable=False) here produced max_retry misleading
    "recoverable" error events and ~4s of retry sleeps before the stream
    finally died anyway. A non-APIError takes _main_task's terminal branch:
    it emits exactly ONE error event (recoverable=False, verified in the
    installed source) and kills the stream immediately with the true cause.

    Resolution:
    - Treat the session as lost; the audio must be re-captured, not retried
    - Investigate the connection failure that consumed the audio (see the
      preceding APIConnectionError in the logs)
    """
    pass


class InitializationError(VoxistError):
    """
    Raised when plugin initialization fails and cannot recover.

    This indicates the plugin was unable to initialize properly and
    attempting to use it would result in undefined behavior.

    Common causes:
    - Authentication failure during pool initialization
    - Network issues preventing initial connection
    - Invalid configuration detected at runtime

    Resolution:
    - Check initialization_error property for root cause
    - Verify API key and network connectivity
    - Use is_ready property to check initialization status
    - Call wait_for_initialization() before first use
    """
    pass
