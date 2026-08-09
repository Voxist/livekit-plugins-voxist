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
    DEPRECATED - never raised. Superseded by ConnectionError.

    This dates from the connection pool, which no longer exists: the gateway
    ends every session by closing the socket after "Done", so sockets cannot
    be reused and each stream dials its own. With no pool there is nothing to
    exhaust, and every dial or token-exchange failure is now a plain
    ConnectionError.

    Kept only so `except ConnectionPoolExhaustedError` in existing user code
    still imports and still compiles. It subclasses ConnectionError, so code
    that catches it already catches nothing narrower than what is raised -
    catch ConnectionError instead.
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
    DEPRECATED - never raised. Backpressure is no longer an error condition.

    Sustained inability to keep up with the input is handled inside the
    stream: the input backlog is bounded and trimmed (oldest audio dropped,
    flush sentinels preserved, drops rate-limit-logged) rather than raised.
    A send that fails outright surfaces as ConnectionError, and a session
    whose audio was consumed with nothing transcribable to show for it
    surfaces as TranscriptLostError.

    Kept only so `except BackpressureError` in existing user code still
    imports and still compiles.
    """
    pass


class OwnershipViolationError(VoxistError):
    """
    DEPRECATED - never raised. Superseded by single-owner sockets.

    This dates from the connection pool, where several streams could contend
    for one pooled connection and ownership had to be policed at runtime
    (VUL-003). There is no pool: each stream dials, owns and closes exactly
    one socket for its own lifetime, so there is no shared connection state
    left to violate.

    Kept only so `except OwnershipViolationError` in existing user code still
    imports and still compiles.
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
    - Authentication failure during the startup token pre-fetch
    - Network issues preventing the token exchange
    - The WebSocket reachability probe failing on a deployment whose HTTPS
      token endpoint works but whose WebSocket path is blocked

    Resolution:
    - Check initialization_error property for root cause
    - Verify API key and network connectivity
    - Use is_ready property to check initialization status
    - Call wait_for_initialization() before first use
    """
    pass
