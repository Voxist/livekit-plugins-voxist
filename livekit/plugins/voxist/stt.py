"""Main VoxistSTT plugin class."""

from __future__ import annotations

import asyncio
import contextlib
import os
import ssl
import time
import weakref
from enum import Enum

import aiohttp
from livekit.agents.stt import STT, STTCapabilities
from livekit.agents.types import NOT_GIVEN, APIConnectOptions, NotGivenOr

from .connection import VoxistDialer
from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    InitializationError,
    LanguageNotSupportedError,
)
from .exceptions import (
    ConnectionError as VoxistConnectionError,
)
from .log import logger
from .models import SUPPORTED_LANGUAGES, validate_language_format
from .stream import VoxistSTTStream


class InitializationState(Enum):
    """
    The readiness verdict, as reported by VoxistSTT.initialization_state.

    COMPLETED means READY, by the single definition documented in VoxistSTT
    (above VoxistSTT._reachability_proven) - not merely "the background
    warm-up task finished". A finished token warm-up whose reachability proof
    has not run yet reads as PENDING, because the remaining phase is
    created-but-not-started.

    State transitions:
        PENDING -> RUNNING -> COMPLETED (success)
        PENDING -> RUNNING -> FAILED (error)
        PENDING -> NOT_STARTED (no event loop)
        FAILED -> NOT_STARTED (transient cause, retry cooldown elapsed)

    FAILED is NOT terminal for transient causes: see
    VoxistSTT.wait_for_initialization and VoxistSTT._failure_is_permanent.
    Two cases are terminal - a rejected credential, and a plugin closed with
    aclose().

    Use VoxistSTT.initialization_state property to check current state.
    """
    NOT_STARTED = "not_started"   # No event loop available, init on demand
    PENDING = "pending"           # Task created but not yet started
    RUNNING = "running"           # Initialization in progress
    COMPLETED = "completed"       # Successfully initialized
    FAILED = "failed"             # Initialization failed with error


class VoxistSTT(STT):
    """
    Voxist ASR Speech-to-Text plugin for LiveKit.

    Features:
    - One WebSocket per stream, dialed on demand: the gateway ends every
      session by closing the socket after "Done", so sockets are never reused
      and there is no connection pool
    - Support for 8+ languages including French medical
    - Automatic text2num and medical units processing (fr-medical)
    - Interim and final transcription results
    - Retries owned by livekit's own machinery (conn_options.max_retry on
      stream()); transport liveness by aiohttp's WebSocket heartbeat

    Task Lifecycle (QUAL-002):
        Construction starts a background warm-up that pre-fetches the
        WebSocket token (the one slow step of the first dial). The token
        exchange is plain HTTPS, so it is NOT a readiness proof: readiness
        additionally requires one short-lived WebSocket dial whose application
        layer is watched (unless validate_websocket=False), and
        wait_for_initialization() is the only call that performs it.

        Every member below is a projection of ONE definition of "ready",
        documented in full above _reachability_proven. They cannot disagree:

        - initialization_state: The verdict as an InitializationState.
          COMPLETED means ready and nothing weaker.
        - initialization_error: The cause behind a non-ready verdict.
        - is_ready: initialization_state == COMPLETED.
        - wait_for_initialization(): Does the work that can establish
          readiness (token warm-up + reachability proof), then returns the
          verdict.
        - check_initialization(): Raises InitializationError unless ready -
          including when readiness has merely never been verified.

        State transitions:
            PENDING -> RUNNING -> COMPLETED (success path)
            PENDING -> RUNNING -> FAILED (error path)
            NOT_STARTED (no event loop, warm-up runs on demand)
            FAILED -> NOT_STARTED (transient failure, cooldown elapsed:
                the next wait_for_initialization() re-attempts, so one
                transport blip does not brick the instance. A rejected
                credential stays FAILED, and so does aclose().)

    Example:
        # Minimal usage
        stt = VoxistSTT()  # Uses VOXIST_API_KEY env var

        # With configuration
        stt = VoxistSTT(
            api_key="voxist_...",
            language="fr-medical",
        )

        # Use in LiveKit agent
        agent = agents.VoicePipelineAgent(stt=stt, llm=..., tts=...)
        await agent.start(ctx.room)

        # Explicit initialization check (optional)
        await stt.wait_for_initialization()
        if not stt.is_ready:
            raise stt.initialization_error
    """

    def __init__(
        self,
        *,
        api_key: str | None = None,
        language: str = "fr",
        sample_rate: int = 16000,
        base_url: str = "wss://api-asr.voxist.com/ws",
        interim_results: bool = True,
        connection_pool_size: int = 2,
        connection_timeout: float = 10.0,
        heartbeat_interval: float = 30.0,
        chunk_duration_ms: int = 100,
        stride_overlap_ms: int = 20,
        max_reconnect_attempts: int = 10,
        enable_metrics: bool = True,
        http_session: aiohttp.ClientSession | None = None,
        api_key_header: str = "X-LVL-KEY",
        punctuation_mode: str | None = None,
        ssl_context: ssl.SSLContext | None = None,
        validate_websocket: bool = True,
    ):
        """
        Initialize Voxist STT plugin.

        Args:
            api_key: Voxist API key (or use VOXIST_API_KEY env var)
            language: Language code (fr, en, de, it, es, nl, pt, sv, fr-medical)
            sample_rate: Audio sample rate in Hz (default: 16000)
            base_url: WebSocket endpoint URL
            interim_results: Enable partial transcription results
            connection_pool_size: Accepted for backwards compatibility and
                IGNORED. There is no pool: the gateway closes every socket at
                end of session, so each stream dials its own. Passing a
                non-default value logs a deprecation warning.
            connection_timeout: Bound, in seconds, on both the HTTPS token
                exchange and the WebSocket dial (default: 10.0).
            heartbeat_interval: WebSocket ping/pong interval in seconds.
                Still honoured: it is passed to aiohttp's heartbeat, which is
                how a dead transport is detected during long silences.
            chunk_duration_ms: Audio chunk size in milliseconds
            stride_overlap_ms: Chunk overlap for boundary accuracy
            max_reconnect_attempts: Accepted for backwards compatibility and
                IGNORED. Retries are owned by livekit's own machinery
                (conn_options.max_retry on stream()). Passing a non-default
                value logs a deprecation warning.
            enable_metrics: Emit LiveKit metrics events
            http_session: Optional aiohttp session (for advanced use)
            api_key_header: HTTP header name for API key (default: X-LVL-KEY)
            punctuation_mode: Engine punctuation behaviour, sent on the connect
                URL. None (default) leaves the engine's automatic punctuation
                on. "Dictated" turns it OFF so the speaker's own spoken
                punctuation is used instead - verified live: the same audio
                returns without automatic commas or sentence periods. Requires
                the gateway's V2 feature flag, and is effectively French-only.
            ssl_context: Optional SSL context for TLS. Needed to reach a
                deployment whose certificate is signed by a private CA:
                certificate verification is always enabled, so without a
                context trusting that CA the connection is refused.
            validate_websocket: When True (the default), the explicit
                readiness paths - wait_for_initialization() and therefore
                __aenter__ - prove WebSocket reachability by dialing one
                short-lived WS after the token pre-fetch (see
                _validate_websocket_path for the deployment-health story
                and the cost). Set False to make readiness token-only, as
                it was before this validation existed.

        Raises:
            ConfigurationError: If API key missing or invalid config
            LanguageNotSupportedError: If language not supported
        """
        super().__init__(
            capabilities=STTCapabilities(
                streaming=True,
                interim_results=interim_results
            )
        )

        # API Configuration
        self._api_key = api_key or os.environ.get("VOXIST_API_KEY")
        if not self._api_key:
            raise ConfigurationError(
                "Voxist API key required. Set VOXIST_API_KEY environment "
                "variable or pass api_key parameter.\n\n"
                "Get your API key at: https://asr-demo.voxist.com"
            )

        # Validate language (SEC-002 FIX: both allowlist and format validation)
        if language not in SUPPORTED_LANGUAGES:
            raise LanguageNotSupportedError(
                f"Language '{language}' not supported.\n"
                f"Supported languages: {', '.join(SUPPORTED_LANGUAGES.keys())}\n"
                f"See documentation for language codes."
            )

        # SEC-002 FIX: Defense-in-depth format validation
        if not validate_language_format(language):
            raise LanguageNotSupportedError(
                f"Language '{language}' has invalid format.\n"
                f"Expected format: 'xx' or 'xx-YY' (e.g., 'fr', 'en-US', 'fr-medical')"
            )

        # Validate configuration
        if sample_rate not in [8000, 16000, 44100, 48000]:
            logger.warning(
                f"Unusual sample rate: {sample_rate}. "
                f"Recommended: 16000 Hz for optimal quality."
            )

        # connection_pool_size is deliberately NOT range-checked: it is
        # accepted-and-ignored (see the deprecation warning below), so a hard
        # ConfigurationError for an out-of-range value would reject a setting
        # that does nothing either way.

        if chunk_duration_ms < 50 or chunk_duration_ms > 500:
            raise ConfigurationError(
                f"chunk_duration_ms must be 50-500ms, got {chunk_duration_ms}"
            )

        # No silent lies about dead parameters: both are accepted only for
        # backwards compatibility, and a caller passing a non-default value
        # is told so once, here, instead of wondering why it has no effect.
        if connection_pool_size != 2:
            logger.warning(
                f"connection_pool_size={connection_pool_size} is deprecated "
                "and ignored: there is no connection pool (one socket per "
                "stream, dialed on demand)"
            )
        if max_reconnect_attempts != 10:
            logger.warning(
                f"max_reconnect_attempts={max_reconnect_attempts} is "
                "deprecated and ignored: retries are owned by livekit "
                "(conn_options.max_retry on stream())"
            )

        # Store configuration
        self._config = {
            "language": language,
            "sample_rate": sample_rate,
            "interim_results": interim_results,
            "chunk_duration_ms": chunk_duration_ms,
            "stride_overlap_ms": stride_overlap_ms,
        }

        self._base_url = base_url
        self._session = http_session
        self._enable_metrics = enable_metrics

        # One dialer per plugin; one socket per stream. There is no pool:
        # the gateway ends every session by closing the socket after "Done",
        # so connections cannot be reused. connection_pool_size,
        # max_reconnect_attempts and heartbeat_interval are accepted for
        # backwards compatibility; reconnection is owned by livekit's own
        # retry (conn_options.max_retry) and liveness by aiohttp's heartbeat.
        self._ssl_context = ssl_context
        self._api_key_header = api_key_header
        # Validated here, at the public boundary, because the gateway
        # matches the EXACT string 'Dictated' (and only under its V2 feature
        # flag): any other value - "dictated", "DICTATED", a stray space - is
        # silently ignored server-side, and the caller gets full automatic
        # punctuation with nothing in any log to say their setting did
        # nothing. A dictation product misconfigured that way ships wrong
        # transcripts quietly; a ConfigurationError at construction is the
        # only place the mistake is cheap.
        if punctuation_mode is not None and punctuation_mode != "Dictated":
            raise ConfigurationError(
                f"punctuation_mode={punctuation_mode!r} is not recognised. "
                "The gateway accepts exactly 'Dictated' (case-sensitive), "
                "which disables automatic punctuation so the speaker's own "
                "spoken punctuation is used; omit the parameter (None) to "
                "keep automatic punctuation."
            )
        self._punctuation_mode = punctuation_mode
        self._heartbeat_interval = heartbeat_interval
        self._connection_timeout = connection_timeout
        self._owns_session = http_session is None
        self._dialer: VoxistDialer | None = None
        self._dialer_lock = asyncio.Lock()
        self._closed = False
        # The loop the session is bound to, as recorded by us (see
        # _session_loop): authoritative for plugin-owned sessions, pinned at
        # first use for a caller-supplied one whose binding is unreadable.
        self._session_bound_loop: asyncio.AbstractEventLoop | None = None
        self._validate_websocket = validate_websocket
        self._ws_validated = False
        # Dedicated lock, NOT _dialer_lock: the probe calls _ensure_dialer,
        # which takes _dialer_lock, and asyncio.Lock is not reentrant.
        self._probe_lock = asyncio.Lock()
        self._probe_failed_at: float | None = None
        self._probe_error: Exception | None = None

        # Live streams, tracked weakly so a stream that ends normally
        # disappears on its own. aclose() closes these BEFORE the shared
        # HTTP session: closing the session first left live streams to
        # redial on a closed session and crash with an unmapped
        # RuntimeError('Session is closed').
        self._live_streams: weakref.WeakSet[VoxistSTTStream] = weakref.WeakSet()

        # Task lifecycle tracking (QUAL-002: asr-all-dxe)
        self._init_task: asyncio.Task | None = None
        self._init_error: Exception | None = None
        self._init_state = InitializationState.PENDING
        self._init_failed_at: float | None = None

        logger.info(
            f"VoxistSTT initialized: language={language}, "
            f"sample_rate={sample_rate}, connection_timeout={connection_timeout}"
        )

        # Pre-warm connections asynchronously (non-blocking)
        # Only if event loop is running (avoid issues in tests)
        try:
            loop = asyncio.get_running_loop()
            self._init_task = loop.create_task(self._initialize_pool())
            # The task re-raises AuthenticationError so an awaiter sees the
            # true cause, but nobody is REQUIRED to await it - without this
            # callback, an unawaited failed init surfaces as a GC-time
            # "Task exception was never retrieved" even though the failure
            # is already recorded in _init_error/_init_state.
            self._init_task.add_done_callback(self._retrieve_init_exception)
            logger.debug("Background initialization task created")
        except RuntimeError:
            # No running event loop (e.g., in tests): nothing is pre-fetched
            # now; the token is exchanged on the first dial instead.
            self._init_state = InitializationState.NOT_STARTED
            logger.debug(
                "No running event loop; the WebSocket token will be fetched "
                "on first use instead of pre-fetched"
            )

    # How long a FAILED readiness result is trusted before a readiness call
    # is allowed to re-attempt (see _may_retry_failed_readiness). Per
    # instance: tests override it, and so may a deployment that wants a
    # tighter or looser probe cadence.
    READINESS_RETRY_COOLDOWN_SECONDS = 30.0

    # A readiness probe has already proved reachability when this close runs;
    # do not let a peer that stalls its close handshake hold readiness open.
    READINESS_CLOSE_TIMEOUT_SECONDS = 1.0

    # How long the probe watches a freshly upgraded socket before accepting it
    # as proof (see _confirm_probe_reached_the_application). It is NOT a wait
    # for something to arrive - the gateway sends no greeting frame, so on a
    # healthy deployment nothing ever will. It is a window in which a rejection
    # the peer has ALREADY decided to send (a 1008 close right after the
    # upgrade) can still reach us, so it only has to cover a round trip.
    READINESS_APPLICATION_GRACE_SECONDS = 0.5

    @staticmethod
    def _now() -> float:
        """Monotonic clock seam for the readiness cooldown (tests override it
        per instance instead of tampering with the global clock, which asyncio
        itself depends on)."""
        return time.monotonic()

    def _record_init_failure(self, exc: Exception) -> None:
        """Move to FAILED, recording the cause and WHEN it happened.

        The timestamp is what makes FAILED non-terminal for transient causes:
        see _may_retry_failed_readiness.

        A PERMANENT classification is never downgraded by a later, weaker
        report ([F3b]). Two readiness callers racing (a health endpoint and
        `async with`) both record an outcome, and the LAST write used to win
        unconditionally: caller A recorded the AuthenticationError the gateway
        answered with, then caller B - which never dialed at all, it only
        replayed A's failure through the cooldown branch - overwrote it with a
        transient-looking ConnectionError. _failure_is_permanent then read the
        revoked credential as a blip and re-probed it every cooldown window,
        forever, which is precisely how a rejected key gets banned.
        """
        if (
            self._init_state == InitializationState.FAILED
            and self._failure_is_permanent(self._init_error)
            and not self._failure_is_permanent(exc)
        ):
            logger.debug(
                f"Keeping the settled readiness failure {self._init_error!r} "
                f"rather than downgrading it to {exc!r}"
            )
            return
        self._init_error = exc
        self._init_state = InitializationState.FAILED
        self._init_failed_at = self._now()

    @staticmethod
    def _failure_is_permanent(exc: Exception | None) -> bool:
        """
        Whether a readiness failure can be re-attempted, or is settled.

        The distinction that matters ([2]):

        - CREDENTIAL REJECTED (AuthenticationError): the gateway looked at
          the key and said no. Re-probing cannot change that answer, and
          hammering a rejected key is exactly how a key gets banned. Sticky
          for the instance's lifetime - fix the key and build a new plugin.
        - TRANSIENT TRANSPORT FAILURE (everything else: our ConnectionError
          from a refused/reset dial, a TimeoutError, a proxy hiccup, an
          unexpected error): the deployment may well be reachable a moment
          later. Blocking readiness forever on one blip bricked the plugin
          even though stream() would have dialed and transcribed fine.

        The whole __cause__ CHAIN is inspected, not just the outermost type
        ([F3b]). This used to be a bare isinstance() on the exception handed
        in, so any code path that re-wrapped a rejected credential -
        `raise ConnectionError(...) from AuthenticationError(...)` - silently
        reclassified it as transient and re-probed a revoked key on a 30s
        cadence for the rest of the process's life. Classification must
        survive wrapping, because a wrapper says nothing about whether the
        underlying answer can change.
        """
        seen: set[int] = set()
        current: BaseException | None = exc
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, AuthenticationError):
                return True
            current = current.__cause__
        return False

    def _may_retry_failed_readiness(self) -> bool:
        """True when a FAILED state is stale enough to re-attempt."""
        if self._closed:
            # aclose() is settled, not stale: stream() can never succeed
            # again, so there is nothing a re-probe could discover.
            return False
        if self._failure_is_permanent(self._init_error):
            return False
        if self._init_failed_at is None:
            # FAILED was set without going through _record_init_failure
            # (a test poking _init_state, say): no timestamp, no cooldown to
            # measure - treat it as settled rather than re-probing blindly.
            return False
        elapsed = self._now() - self._init_failed_at
        return elapsed >= self.READINESS_RETRY_COOLDOWN_SECONDS

    # ------------------------------------------------------------------
    # THE readiness definition. Read this before touching anything below.
    # ------------------------------------------------------------------
    #
    # The whole readiness surface - is_ready, initialization_state,
    # initialization_error, check_initialization() and
    # wait_for_initialization() - derives its answer from _readiness_state()
    # and adds NOTHING of its own. That is deliberate and it is the fix for a
    # whole family of defects: each accessor used to compute its own verdict
    # from a slightly different subset of the internal flags, so they
    # disagreed. is_ready consulted _ws_validated but not _closed, so a
    # plugin torn down by aclose() still reported healthy while stream()
    # raised RuntimeError. initialization_state reported COMPLETED as soon as
    # the HTTPS token exchange succeeded, so a deployment whose WebSocket path
    # is blocked read as healthy and check_initialization() - documented as
    # "use this before operations that require successful initialization" -
    # stayed silent. Adding one more guard per accessor is what produced that
    # mess; there is one predicate now, and every accessor is a projection of
    # it.
    #
    # READY means, and only means:
    #
    #     THIS PLUGIN HAS OBSERVED - AND HAS NOT SINCE INVALIDATED - EVERY
    #     FACT ITS CONFIGURED READINESS CONTRACT REQUIRES BEFORE A NEW STREAM
    #     CAN REACH THE ASR APPLICATION.
    #
    # All three of these must hold. Nothing else counts, in either direction:
    #
    #   1. THE PLUGIN IS STILL USABLE. aclose() has not run. A closed plugin
    #      cannot dial (_ensure_dialer refuses) and stream() raises outright,
    #      so "ready" would be a lie no matter what was verified earlier. This
    #      is also permanent: _may_retry_failed_readiness refuses to re-probe
    #      a closed plugin.
    #
    #   2. NO READINESS FAILURE IS ON RECORD. _init_state is not FAILED. A
    #      transient failure clears itself through wait_for_initialization()
    #      once the cooldown elapses; a rejected credential never does (see
    #      _failure_is_permanent).
    #
    #   3. THE CONFIGURED REACHABILITY PROOF EXISTS (_reachability_proven):
    #      - validate_websocket=True (the default): a probe has dialed a real
    #        WebSocket AND watched the application layer survive a grace
    #        window (_validate_websocket_path). A 101 handshake alone is NOT
    #        this proof - see _confirm_probe_reached_the_application.
    #      - validate_websocket=False: the token exchange succeeded. The
    #        caller has explicitly asked for the weaker, HTTPS-only contract.
    #
    # DELIBERATELY NOT part of the definition, and why:
    #
    #   - THAT THE NEXT DIAL WILL SUCCEED. Readiness is past tense: it reports
    #      what was observed, never a prediction. The gateway can die one
    #      millisecond after the probe. Callers get failover from stream()
    #      failing, not from readiness promising.
    #   - A SUCCESSFUL STREAM DIAL. _dial deliberately does not credit
    #     readiness: it only observes the 101 upgrade, which is exactly the
    #     evidence point 3 rejects. A consequence, stated plainly: a
    #     deployment that only ever calls stream() never runs the probe, so
    #     is_ready stays False for it. Awaiting wait_for_initialization() once
    #     is the supported way to get a readiness signal, and it is cheap
    #     after the first call.
    #   - THE DIAL RATE LIMITER'S REMAINING BUDGET (connection.py). A full
    #     window makes the probe fail as a retryable ConnectionError, which is
    #     recorded as a transient failure and re-attempted after the cooldown.
    #     Readiness never inspects the limiter directly, so it stays correct
    #     whatever the limiter's policy is.
    #   - WHETHER THE ENGINE WILL PRODUCE TRANSCRIPTS. That is a per-session
    #     property (see TranscriptLostError), not a deployment property, and
    #     no startup probe can establish it.

    def _reachability_proven(self) -> bool:
        """Point 3 of the readiness definition: does the configured proof
        exist? Nothing else - no state, no _closed, no failure."""
        if self._validate_websocket:
            return self._ws_validated
        # Token-only contract: either the warm-up completed, or a dial already
        # fetched and cached a token on demand.
        return (
            self._init_state == InitializationState.COMPLETED
            or (self._dialer is not None and self._dialer._token_url is not None)
        )

    def _readiness_state(self) -> InitializationState:
        """
        The one readiness verdict, expressed as an InitializationState.

        COMPLETED is returned if and ONLY if the plugin is ready by the
        definition above. In particular, a finished token warm-up whose
        reachability proof has not run yet reads as PENDING, not COMPLETED:
        the remaining phase is created-but-not-started, which is exactly what
        PENDING means, and reporting COMPLETED there is what let a blocked
        WebSocket path pass for a healthy deployment.
        """
        if self._closed:
            return InitializationState.FAILED
        if self._init_state == InitializationState.FAILED:
            return InitializationState.FAILED
        if self._reachability_proven():
            return InitializationState.COMPLETED
        if self._init_state == InitializationState.COMPLETED:
            return InitializationState.PENDING
        return self._init_state

    @staticmethod
    def _retrieve_init_exception(task: asyncio.Task) -> None:
        """Retrieve (and debug-log) the init task's exception so GC never
        reports it as unretrieved. The failure itself is already surfaced
        through initialization_state / initialization_error."""
        if task.cancelled():
            return
        exc = task.exception()
        if exc is not None:
            logger.debug(
                f"Background initialization failed: {exc!r} (already "
                "recorded in initialization_state/initialization_error)"
            )

    # Warn once per process, not once per dial: an aiohttp release that
    # renames ClientSession._loop would otherwise flood every log line.
    _loop_introspection_warned = False

    @classmethod
    def _introspect_session_loop(
        cls,
        session: aiohttp.ClientSession,
    ) -> asyncio.AbstractEventLoop | None:
        """
        The loop an aiohttp session is bound to by reading its private
        attribute, or None if that cannot be determined.

        Failing LOUDLY on purpose. This used to return None silently, which
        disabled BOTH loop guards at once: no defunct-loop rebuild and no
        alive-foreign-loop diagnosis. On an aiohttp release that renames the
        attribute, a plugin built under one asyncio.run() and used under
        another then died inside ws_connect with
        RuntimeError('Event loop is closed'), mapped to a generic retryable
        transport error - the user chasing a network problem that was really
        a loop-affinity problem.

        Callers must not depend on this for plugin-owned sessions: those
        record their loop at creation (_session_bound_loop), which no aiohttp
        rename can break. This introspection is only the fallback for a
        caller-supplied session whose binding predates us.
        """
        loop = getattr(session, "_loop", None)
        if isinstance(loop, asyncio.AbstractEventLoop):
            return loop
        if not cls._loop_introspection_warned:
            cls._loop_introspection_warned = True
            logger.warning(
                "Cannot determine which event loop this aiohttp.ClientSession "
                f"is bound to (aiohttp {aiohttp.__version__} does not expose "
                "ClientSession._loop as expected). Loop-affinity checks fall "
                "back to first-use pinning for caller-supplied sessions: a "
                "session already bound to another loop before this plugin "
                "saw it can no longer be diagnosed, and its dials will fail "
                "as transport errors instead. Plugin-owned sessions are "
                "unaffected. Please report this aiohttp incompatibility."
            )
        return None

    def _session_loop(
        self,
        session: aiohttp.ClientSession,
    ) -> asyncio.AbstractEventLoop | None:
        """
        The loop `session` is bound to, preferring our own record over
        aiohttp introspection.

        Returns None only when the binding is genuinely unknown - which,
        after the first use of a caller-supplied session, cannot happen:
        _ensure_dialer pins it (conservatively, to the loop that first used
        it) precisely so the foreign-loop guard is never silently disabled.
        """
        if self._session_bound_loop is not None and session is self._session:
            return self._session_bound_loop
        return self._introspect_session_loop(session)

    async def _ensure_dialer(self) -> VoxistDialer:
        """
        Create the HTTP session and dialer on first use, in a running loop.

        aiohttp binds a ClientSession to whichever loop first uses it; a
        stream running in a DIFFERENT loop would otherwise die with an
        unmapped RuntimeError('Event loop is closed'). Two situations are
        distinguished:

        - The bound loop is CLOSED (sequential asyncio.run calls, test
          loops): the session is unusable and nothing can race us on it.
          A plugin-owned session is rebuilt transparently for the current
          loop; a caller-supplied session is theirs, so a clear mapped
          error is raised instead of replacing it.
        - The bound loop is ALIVE but not the running one (two live loops
          sharing one VoxistSTT): multi-loop sharing is unsupported, not
          silently accommodated. Destroying the session here would leak
          connectors, thrash the token cache and sabotage the healthy
          loop's streams, so the session is left untouched and RuntimeError
          is raised: create one VoxistSTT instance per event loop.

        Why RuntimeError and not our ConnectionError for that last case
        ([8]): the stream maps ConnectionError to a retryable
        APIConnectionError, so livekit burned its whole retry schedule -
        three misleading recoverable=True events - on a programming error
        that cannot change between attempts, then reported a wrapper instead
        of the real cause. A non-APIError takes _main_task's terminal
        branch: exactly one recoverable=False event and the precise message.
        Same precedent as stream()-after-aclose().

        Which loop a session is bound to is determined WITHOUT aiohttp
        introspection whenever possible: a plugin-owned session records its
        loop at creation, and a caller-supplied session is pinned to the loop
        that first used it. See _introspect_session_loop for what happens
        when neither is available.
        """
        async with self._dialer_lock:
            if self._closed:
                # A stream retry racing aclose() must fail as a mapped,
                # retryable error - NOT resurrect a fresh session after
                # shutdown, and not crash with a raw RuntimeError.
                raise VoxistConnectionError(
                    "VoxistSTT is closed; cannot dial"
                )

            running = asyncio.get_running_loop()
            if self._session is not None:
                bound_loop = self._session_loop(self._session)
                if bound_loop is None:
                    # Binding unknown - only reachable for a CALLER-SUPPLIED
                    # session whose aiohttp binding is unreadable, because a
                    # plugin-owned session records its loop at creation
                    # (which is the conservative half of [11]: an owned
                    # session is rebuilt for the running loop rather than
                    # reused possibly-dead, with no introspection involved).
                    # We cannot rebuild someone else's session, so pin it to
                    # the loop that first used it: the foreign-loop guard
                    # stays armed for every later loop instead of being
                    # silently disabled. _introspect_session_loop has
                    # already logged the incompatibility.
                    assert not self._owns_session, (
                        "a plugin-owned session always has a recorded loop"
                    )
                    self._session_bound_loop = running
                    bound_loop = running
                defunct = self._session.closed or bound_loop.is_closed()
                if defunct:
                    if not self._owns_session:
                        raise VoxistConnectionError(
                            "the caller-supplied http_session is closed or "
                            "bound to a closed event loop; it is not "
                            "plugin-owned, so it will not be replaced"
                        )
                    old, old_loop = self._session, bound_loop
                    self._session = None
                    self._dialer = None
                    self._session_bound_loop = None
                    if not old.closed:
                        if old_loop is running:
                            await old.close()
                        else:
                            # Bound to a closed loop: it cannot be
                            # awaited-closed from here. Dropping it leaks at
                            # most its idle connector, once, and is logged
                            # rather than hidden.
                            logger.warning(
                                "Dropping the plugin-owned HTTP session "
                                "bound to a defunct event loop; it cannot "
                                "be closed from the current loop"
                            )
                elif bound_loop is not running:
                    # Alive-but-different loop: do NOT destroy the healthy
                    # loop's session. RuntimeError (not our ConnectionError)
                    # so livekit fails fast with this exact message instead
                    # of retrying a programming error - see the docstring.
                    raise RuntimeError(
                        "VoxistSTT is bound to a different running event "
                        "loop; sharing one instance across live loops is "
                        "unsupported - create one VoxistSTT per event loop"
                    )

            if self._dialer is None:
                assert self._api_key is not None  # validated in __init__
                if self._session is None:
                    self._session = aiohttp.ClientSession()
                    self._owns_session = True
                    # Recorded, not introspected: no aiohttp rename can break
                    # the loop guards for a session we created ourselves.
                    self._session_bound_loop = running
                self._dialer = VoxistDialer(
                    session=self._session,
                    base_url=self._base_url,
                    api_key=self._api_key,
                    api_key_header=self._api_key_header,
                    ssl_context=self._ssl_context,
                    heartbeat_interval=self._heartbeat_interval,
                    connection_timeout=self._connection_timeout,
                )
            return self._dialer

    async def _dial(self, language: str):
        """
        Open a WebSocket for one stream session.

        Raises:
            AuthenticationError: The key was rejected (fatal, not retried).
            ConnectionError: Transport failure (the stream converts this to
                APIConnectionError so livekit retries it).
        """
        dialer = await self._ensure_dialer()
        # Always 16kHz on the wire; the stream resamples its input.
        #
        # This dial does NOT credit readiness, even though it used to
        # ("keeps is_ready useful for applications that use the stream
        # directly"). All it observes is the 101 upgrade, and a gateway that
        # rejects at the APPLICATION layer - a 1008 close right after the
        # upgrade, the documented shape for a refused app credential or an
        # exhausted balance - answers 101 first. Crediting readiness here
        # therefore made is_ready True on a deployment where every stream
        # dies immediately after connecting. Readiness has one source of
        # proof, the probe, which actually watches for that close; see the
        # readiness definition above _reachability_proven.
        return await dialer.dial(
            language, 16000, punctuation_mode=self._punctuation_mode
        )

    async def _initialize_pool(self) -> None:
        """
        Warm up: pre-fetch the WebSocket token (called asynchronously).

        The "_pool" in the name is a leftover: there is no pool, and this
        method never opens a connection. Renaming it is a separate change -
        the existing test suite patches and calls it by this name.

        The token exchange is the one slow step of the first dial (an HTTPS
        round-trip). Pre-fetching it means the first stream dials without it,
        and a rejected key surfaces at startup instead of on first use.

        Deliberately token-only: this task runs fire-and-forget on EVERY
        construction, and each WebSocket dial opens a real ASR engine
        session server-side (the gateway dials the engine on connect). The
        end-to-end WebSocket reachability proof - which the HTTPS exchange
        alone cannot give - lives in _validate_websocket_path and runs once,
        on the explicit readiness paths (wait_for_initialization /
        __aenter__), where a caller has actually asked for the guarantee.

        The COMPLETED this method sets is therefore the WARM-UP PHASE's own
        state, NOT a readiness verdict, and _init_state is not what any public
        accessor reports: the readiness verdict is computed by
        _readiness_state() from the full definition, and a completed
        token-only warm-up under the default contract reads as PENDING there.
        The distinction is load-bearing. This method opens no socket, so on a
        deployment whose WebSocket path is blocked it succeeds - and while
        initialization_state returned _init_state raw, that deployment
        reported COMPLETED with initialization_error None and
        check_initialization() silent, i.e. "healthy" for a plugin whose every
        stream() dies on the dial.

        Warm-up phase transitions (QUAL-002):
            PENDING -> RUNNING (start)
            RUNNING -> COMPLETED (token cached)
            RUNNING -> FAILED (error)
        """
        self._init_state = InitializationState.RUNNING
        logger.debug("Initialization state: RUNNING")

        try:
            dialer = await self._ensure_dialer()
            await dialer._get_token_url()
            # A prior timeout or transient transport error may have been
            # recorded while this task continued in the background. A later
            # success must clear that stale diagnostic state along with the
            # FAILED transition.
            self._init_error = None
            self._init_failed_at = None
            self._init_state = InitializationState.COMPLETED
            logger.debug("Token pre-fetch complete (state: COMPLETED)")
        except AuthenticationError as e:
            # Store and re-raise critical errors - never swallow auth failures
            # _record_init_failure stamps the failure time so the cooldown
            # logic can tell a transient blip from a rejected credential.
            self._record_init_failure(e)
            logger.error(
                f"Authentication failed during the token pre-fetch: {e} "
                "(state: FAILED)"
            )
            raise
        except Exception as e:
            # Store error for later access, mark as failed
            self._record_init_failure(e)
            logger.error(f"Failed to pre-fetch WebSocket token: {e} (state: FAILED)")
            # Don't re-raise - allow stream() to attempt on-demand initialization

    async def _validate_websocket_path(self, timeout: float) -> None:
        """
        Prove end-to-end WebSocket reachability with one short-lived dial.

        Deployment-health story: the token pre-fetch is plain HTTPS, so it
        succeeds on deployments where the WebSocket path is broken (a proxy
        stripping the Upgrade header, a firewall blocking WS) - "healthy"
        startup, then every real call fails. Dialing one WS and closing it
        immediately restores the old initialize() guarantee: COMPLETED
        readiness means a stream can actually connect.

        Cost, weighed deliberately: the gateway opens a real ASR engine
        session server-side on connect (lang rides the URL). One extra
        short-lived engine session per PLUGIN STARTUP is acceptable; per
        stream it would not be - hence this living on the explicit readiness
        paths, not in the fire-and-forget background warm-up, and hence the
        budget enforced here.

        Probe budget ([9] + [2]): AT MOST ONE PROBE IN FLIGHT, AND AT MOST
        ONE PER COOLDOWN WINDOW. The old guard was a bare check-then-await-
        then-set, so `asyncio.gather(wait_for_initialization(),
        wait_for_initialization())` - or a health endpoint racing
        `async with` - made both callers dial, opening two real sockets and
        two server-side engine sessions. _probe_lock plus a re-check inside
        it makes the success path exactly-once; _probe_failed_at makes the
        FAILURE path at-most-once-per-cooldown, so the bounded re-probe that
        keeps a transient blip from bricking the plugin cannot itself become
        a dial loop.

        What counts as proof: a 101 upgrade is NOT enough, and treating it as
        enough is why this had to be reworked. The probe must also watch the
        application layer accept the session - see
        _confirm_probe_reached_the_application.

        Raises whatever the dial or the application-layer check raises (mapped
        ConnectionError / AuthenticationError, or asyncio.TimeoutError from the
        bound), or REPLAYS the last failure unchanged while the cooldown holds;
        the caller records it as an initialization failure.
        """
        if self._ws_validated or not self._validate_websocket:
            return

        async with self._probe_lock:
            # Re-check under the lock: a racing caller may have completed the
            # probe (or failed it) while we waited here.
            if self._ws_validated:
                return
            probe_error = self._probe_error
            if probe_error is not None and self._probe_failed_at is not None:
                elapsed = self._now() - self._probe_failed_at
                if elapsed < self.READINESS_RETRY_COOLDOWN_SECONDS:
                    # The recorded failure is replayed AS ITSELF ([F3a]). It
                    # used to be re-raised as a fresh VoxistConnectionError
                    # wrapping the real cause, and that lost the only thing
                    # the caller needs from it: its classification. A second
                    # readiness caller arriving here after the first recorded
                    # an AuthenticationError handed
                    # wait_for_initialization a transient-looking
                    # ConnectionError, whose _record_init_failure landed last
                    # and downgraded the sticky rejected credential into
                    # something re-probed every cooldown window forever.
                    # Raising the original object cannot lose that, whatever
                    # any classifier does with wrappers.
                    logger.debug(
                        "Replaying the readiness probe failure recorded "
                        f"{elapsed:.1f}s ago; not re-probing for another "
                        f"{self.READINESS_RETRY_COOLDOWN_SECONDS - elapsed:.1f}s"
                    )
                    raise probe_error

            dialer = await self._ensure_dialer()
            language = self._config["language"]
            assert isinstance(language, str)  # validated in __init__
            try:
                ws = await asyncio.wait_for(
                    dialer.dial(language, 16000), timeout=timeout
                )
            except asyncio.CancelledError:
                raise
            except Exception as e:
                # Stamped so concurrent and subsequent callers share one
                # cooldown window instead of each dialing again.
                self._probe_failed_at = self._now()
                self._probe_error = e
                raise

            # Still inside the probe budget: an application-layer rejection is
            # a probe FAILURE and must share the same cooldown window as a
            # failed dial, or a refused credential would be re-probed by every
            # readiness call.
            try:
                await self._confirm_probe_reached_the_application(ws)
            except asyncio.CancelledError:
                # The socket is ours and must not leak, but awaiting a clean
                # close on a cancelled task raises again immediately - abort
                # the transport synchronously instead.
                self._abort_probe_transport(ws)
                raise
            except Exception as e:
                self._probe_failed_at = self._now()
                self._probe_error = e
                await self._close_probe_socket(ws)
                raise

            self._ws_validated = True
            self._probe_failed_at = None
            self._probe_error = None

        await self._close_probe_socket(ws)

    async def _confirm_probe_reached_the_application(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> None:
        """
        Confirm the peer accepted the session, not merely the handshake.

        Why this exists: aiohttp's ws_connect returns as soon as the 101
        arrives, so a gateway that upgrades and THEN refuses at the
        application layer - close code 1008, the documented answer for an
        invalid or expired app-level credential and for an exhausted wallet
        balance (see InsufficientBalanceError) - satisfied the old probe
        completely. Readiness reported True, __aenter__ succeeded, and every
        real stream then died with "server closed the connection before end of
        input" after burning the whole retry budget. The handshake and the
        session are two different questions and only the second one matters.

        Why NEGATIVE evidence, on a timer: the gateway sends no greeting frame
        (verified in simple-websocket-proxy.gateway.ts, and the mock server's
        handler documents the same contract), so on a healthy deployment there
        is nothing to wait FOR - and no close to wait for either, since the
        Kroko engine that already serves production Swedish does not close the
        socket after Done. A probe that waited for any positive signal would
        therefore hang on a perfectly good deployment. What CAN be observed is
        a rejection the peer has already decided to send: the close travels
        right behind the 101, so a short grace window catches it. Surviving
        the window is the healthy answer.

        Any frame that does arrive is also proof - the gateway's
        {"type": "redirect"} is the realistic case - because it means the
        application layer, not just the HTTP upgrade, is talking to us.

        Raises:
            AuthenticationError: closed with 1008 - the credential or the
                account was refused. Permanent by classification, so readiness
                stays failed instead of re-probing a refused key.
            ConnectionError: closed with any other code, or the socket errored.
        """
        try:
            msg = await asyncio.wait_for(
                ws.receive(),
                timeout=self.READINESS_APPLICATION_GRACE_SECONDS,
            )
        except asyncio.TimeoutError:
            # Survived the window in silence: exactly what a healthy gateway
            # does, since it has nothing to say until audio arrives.
            return

        if msg.type == aiohttp.WSMsgType.ERROR:
            raise VoxistConnectionError(
                "The readiness probe socket errored immediately after the "
                f"WebSocket upgrade: {msg.data!r}"
            )

        if msg.type not in (
            aiohttp.WSMsgType.CLOSE,
            aiohttp.WSMsgType.CLOSING,
            aiohttp.WSMsgType.CLOSED,
        ):
            # A real frame: the application layer answered.
            return

        code = msg.data if isinstance(msg.data, int) else ws.close_code
        reason = msg.extra if isinstance(msg.extra, str) else ""
        if code == 1008:
            # 1008 is the gateway's application-layer refusal, and both of its
            # documented meanings - a rejected/expired app credential and an
            # exhausted balance - are answers no re-probe can change. Mapping
            # it to AuthenticationError makes it permanent for this instance
            # (see _failure_is_permanent), which is the same contract a
            # rejected API key gets: fix the account, build a new plugin.
            raise AuthenticationError(
                "The gateway accepted the WebSocket upgrade and then closed "
                f"it with 1008 ({reason!r}): the credential or the account "
                "was refused at the application layer - an invalid or expired "
                "app credential, or an exhausted wallet balance. Every stream "
                "would fail the same way, and re-probing cannot change it."
            )
        raise VoxistConnectionError(
            "The gateway closed the readiness probe immediately after the "
            f"WebSocket upgrade (close code {code}, reason {reason!r}): the "
            "handshake succeeded but the session did not."
        )

    @staticmethod
    def _abort_probe_transport(ws: aiohttp.ClientWebSocketResponse) -> bool:
        """Drop the probe socket's transport without awaiting anything.

        Used where a clean close is impossible or already gave up: the socket
        is short-lived and about to be discarded either way, and leaking it
        would leak a server-side engine session with it.

        Returns whether a transport was actually aborted, because often none
        is reachable. On the close-timeout path in particular aiohttp has
        already released the connection, so `ws._response.connection` is None
        and there is nothing left to abort - and the caller used to log
        "aborted the probe transport" regardless, telling an operator
        diagnosing a leaked socket that cleanup had happened when it had not.
        """
        response = getattr(ws, "_response", None)
        connection = getattr(response, "connection", None)
        transport = getattr(connection, "transport", None)
        if transport is None:
            return False
        try:
            transport.abort()
        except (AttributeError, RuntimeError):
            return False
        return True

    async def _close_probe_socket(
        self, ws: aiohttp.ClientWebSocketResponse
    ) -> None:
        """Close the probe socket without ever letting it affect the verdict.

        The verdict was decided before this runs - by the dial and the
        application-layer check - so nothing here may change it: a peer that
        stalls its close handshake must not hold readiness open, and a hiccup
        while closing must not turn a proven deployment into a failed one.
        """
        try:
            await asyncio.wait_for(
                ws.close(), timeout=self.READINESS_CLOSE_TIMEOUT_SECONDS
            )
        except asyncio.CancelledError:
            raise
        except asyncio.TimeoutError:
            if self._abort_probe_transport(ws):
                logger.warning(
                    "WebSocket readiness probe close timed out; aborted the "
                    "probe transport"
                )
            else:
                # aiohttp has usually already released the connection by the
                # time close() times out, so there is no transport left to
                # abort. Say so rather than claiming a cleanup that did not
                # happen: the socket is then left to aiohttp's own connector
                # teardown, and the server-side session to the gateway's
                # timeout.
                logger.warning(
                    "WebSocket readiness probe close timed out and its "
                    "transport was already released; leaving it to aiohttp's "
                    "connector teardown"
                )
        except Exception as e:
            logger.debug(f"Closing the warm-up WebSocket failed: {e!r}")

    def stream(
        self,
        *,
        language: NotGivenOr[str] = NOT_GIVEN,
        conn_options: APIConnectOptions | None = None,
    ) -> VoxistSTTStream:
        """
        Create a new streaming recognition session.

        Args:
            language: Override default language for this stream
            conn_options: LiveKit connection options

        Returns:
            VoxistSTTStream instance ready for audio streaming

        Raises:
            RuntimeError: The plugin has been closed with aclose(). Creating
                a stream on a closed plugin is a programming error; failing
                here, at the call site, beats burning livekit's whole retry
                budget against a plugin that can never dial again.
            LanguageNotSupportedError: The language override is invalid.

        Example:
            stream = stt.stream(language="fr-medical")
            stream.push_frame(audio_frame)
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    print(event.alternatives[0].text)
        """
        if self._closed:
            raise RuntimeError(
                "stream() called on a closed VoxistSTT; create a new "
                "instance (a closed plugin cannot dial, and retrying "
                "against it can never succeed)"
            )

        stream_language = language if language is not NOT_GIVEN else self._config["language"]

        # Validate language if overridden (SEC-002 FIX: both allowlist and format)
        if language is not NOT_GIVEN:
            assert isinstance(language, str)  # Type narrowing for mypy
            if language not in SUPPORTED_LANGUAGES:
                raise LanguageNotSupportedError(
                    f"Language '{language}' not supported. "
                    f"Supported: {', '.join(SUPPORTED_LANGUAGES.keys())}"
                )
            # SEC-002 FIX: Defense-in-depth format validation
            if not validate_language_format(language):
                raise LanguageNotSupportedError(
                    f"Language '{language}' has invalid format. "
                    f"Expected: 'xx' or 'xx-YY' (e.g., 'fr', 'en-US')"
                )

        # Ensure stream_language is str (from language or config)
        assert isinstance(stream_language, str)

        stream = VoxistSTTStream(
            stt=self,
            config=self._config,
            language=stream_language,
            conn_options=conn_options if conn_options is not None else APIConnectOptions(),
            enable_metrics=self._enable_metrics,
        )
        # Tracked weakly so aclose() can shut live streams down BEFORE the
        # shared HTTP session goes away (see aclose); streams that finish on
        # their own drop out of the set automatically.
        self._live_streams.add(stream)
        return stream

    async def _recognize_impl(  # type: ignore[override]
        self,
        buffer,
        *,
        language: NotGivenOr[str],
        conn_options: APIConnectOptions,
    ) -> None:
        """
        Batch recognition not implemented.

        Voxist plugin only supports streaming recognition for real-time use cases.
        Use stream() method instead.

        Raises:
            NotImplementedError: Always raised
        """
        raise NotImplementedError(
            "Batch recognition not supported by Voxist plugin. "
            "Use stream() method for real-time transcription."
        )

    async def aclose(self) -> None:
        """
        Cleanup plugin resources.

        Order matters: live streams are closed BEFORE the shared HTTP
        session. Closing the session first left running streams to retry
        their dial on a closed session and crash with an unmapped
        RuntimeError('Session is closed') instead of ending cleanly.

        This teardown is best-effort-COMPLETE: every step runs even if an
        earlier one fails, and failures are logged rather than re-raised.
        aclose() is called from finally blocks and __aexit__; an exception
        escaping mid-teardown would both mask the caller's original error
        and abort the remaining cleanup (leaking streams or the HTTP
        session). Only cancellation of aclose() itself propagates.
        """
        logger.info("Closing VoxistSTT plugin")
        # From here on, _ensure_dialer refuses to build a new session, so a
        # stream retry racing this shutdown dies as a mapped ConnectionError.
        # _closed is also point 1 of the readiness definition, so this single
        # assignment is what makes is_ready / initialization_state /
        # check_initialization() all report a closed plugin as not ready. They
        # used to consult only _init_state and _ws_validated, neither of which
        # aclose() touches, so a torn-down plugin kept reporting itself healthy
        # to readiness endpoints and supervisors while every stream() call
        # raised RuntimeError at the caller's site.
        self._closed = True

        # Readiness reports a cause, so give the deliberate shutdown one.
        # A cause already on record explains the plugin's state better than
        # "closed" does, so it is preserved. This is NOT routed through
        # _record_init_failure: that stamps a retry timestamp, and a closed
        # plugin must never look re-probable (see _may_retry_failed_readiness).
        if self._init_error is None:
            self._init_error = InitializationError(
                "VoxistSTT was closed with aclose(); a closed plugin cannot "
                "dial again - create a new instance"
            )

        # Cancel pending initialization task if running (QUAL-HIGH:
        # asr-all-cga). The task may live on a dead or foreign loop (the
        # plugin was built under one asyncio.run and closed under another);
        # cancelling/awaiting it from here would then raise RuntimeError and,
        # unguarded, abort the rest of this teardown before any stream or
        # the session was closed.
        init_task = self._init_task
        if init_task is not None and not init_task.done():
            if init_task.get_loop().is_closed():
                # The task can never run again, and nothing on this loop can
                # cancel or await it. Drop the reference and move on.
                logger.debug(
                    "Dropping the init task stranded on a closed event loop"
                )
                self._init_task = None
            else:
                try:
                    init_task.cancel()
                    await init_task
                except asyncio.CancelledError:
                    logger.debug("Initialization task cancelled")
                except RuntimeError as e:
                    # Task bound to a live loop that is not this one:
                    # unawaitable from here, but the teardown must go on.
                    logger.warning(
                        f"Could not await the init task during close: {e!r}"
                    )
                except Exception as e:
                    # Lost race: the task completed with an error between
                    # the done() check and the cancel. Already recorded in
                    # initialization_error; not this shutdown's problem.
                    logger.debug(f"Init task finished with {e!r} during close")

        # Close live streams first (RecognizeStream.aclose closes the input
        # channel and cancels _main_task, which tears down the socket).
        # A stream may have been closed concurrently by its owner; that is
        # not this shutdown's failure to report.
        for stream in list(self._live_streams):
            with contextlib.suppress(Exception):
                await stream.aclose()

        try:
            await super().aclose()
        except Exception as e:
            logger.warning(f"Base STT close failed during shutdown: {e!r}")

        # Sockets are per-stream and close with their stream; the only shared
        # resource is the HTTP session, and only if this plugin created it.
        try:
            if (
                self._owns_session
                and self._session is not None
                and not self._session.closed
            ):
                await self._session.close()
        except Exception as e:
            logger.warning(f"HTTP session close failed during shutdown: {e!r}")

    @property
    def initialization_error(self) -> Exception | None:
        """
        The cause behind the current readiness verdict, or None when ready.

        Projection of the one readiness definition (see above
        _reachability_proven): it reports the recorded cause, which aclose()
        also sets so a closed plugin explains itself rather than reporting a
        bare None.
        """
        return self._init_error

    @property
    def initialization_state(self) -> InitializationState:
        """
        Return current initialization state (QUAL-002).

        Projection of the one readiness definition (see above
        _reachability_proven) - NOT the raw state of the background warm-up
        task. COMPLETED therefore means "ready", and only that:

            - NOT_STARTED: No event loop was available, init on demand
            - PENDING: Initialization is not finished. Either the warm-up task
              has not started, or it has finished but the reachability proof
              the configured contract requires has not run yet (the usual case
              until something awaits wait_for_initialization()).
            - RUNNING: The warm-up is in progress
            - COMPLETED: Ready, by the full definition
            - FAILED: A readiness failure is on record, or aclose() has run

        Reporting the warm-up's raw state here is what let a deployment whose
        WebSocket path is blocked read as COMPLETED: the token exchange is
        plain HTTPS and succeeds there, so a health check on this property
        called the plugin healthy while every stream() died on the dial.
        """
        return self._readiness_state()

    @property
    def is_ready(self) -> bool:
        """
        Whether the plugin is ready for use (QUAL-002).

        Projection of the one readiness definition (see above
        _reachability_proven): True if and only if that definition holds. It
        is not an independent opinion, and it must never become one - it used
        to be, and consequently answered a different question from
        initialization_state, most visibly after aclose(), which it ignored
        entirely while stream() raised RuntimeError.

        Synchronous, so it can never re-probe: it reports the last VERIFIED
        outcome, and both "failed" and "not verified yet" read as False. A
        transient failure becomes ready again on the next
        `await wait_for_initialization()`, which does re-probe.

        Returns:
            True if ready, False otherwise
        """
        return self._readiness_state() == InitializationState.COMPLETED

    async def wait_for_initialization(self, timeout: float = 30.0) -> bool:
        """
        Wait for background initialization to complete (QUAL-002).

        The ONLY path that can establish readiness, and therefore the one a
        health check or supervisor must await: beyond the token warm-up, the
        first successful call proves the deployment end to end by dialing one
        short-lived WebSocket and watching the application layer accept it
        (see _validate_websocket_path), unless validate_websocket=False. Its
        return value is the one readiness verdict - it is exactly
        `is_ready` re-read after doing the work that could change it.

        FAILED is not terminal for transient causes ([2]). A single blip in
        the one-shot WS probe used to brick the plugin forever: this method
        returned False on its first line for the rest of the instance's life,
        is_ready stayed False and check_initialization() kept raising, even
        though stream() would have dialed and transcribed fine the moment the
        blip cleared. So a FAILED state whose cause was a transient transport
        failure is re-attempted once READINESS_RETRY_COOLDOWN_SECONDS have
        passed since it was recorded, and success clears it. A rejected
        credential (AuthenticationError) stays sticky - see
        _failure_is_permanent for that distinction. The cooldown, and the
        probe's own in-flight lock, bound the retry: at most one probe in
        flight, and at most one per cooldown window.

        Args:
            timeout: Maximum time to wait in seconds (default: 30.0).
                Bounds the token warm-up and the WS probe separately.

        Returns:
            True if initialization completed successfully, False otherwise

        Example:
            stt = VoxistSTT(api_key="...")
            if await stt.wait_for_initialization(timeout=10.0):
                stream = stt.stream()
            else:
                logger.error(f"Init failed: {stt.initialization_error}")
        """
        if self._closed:
            # Point 1 of the readiness definition, checked before any work:
            # nothing this method could do would make a closed plugin ready,
            # and _ensure_dialer refuses anyway - so fail here rather than
            # recording a fresh "cannot dial" failure per call.
            return False

        if self._init_state == InitializationState.FAILED:
            if not self._may_retry_failed_readiness():
                return False
            # Cooldown elapsed on a transient failure: re-run readiness from
            # scratch. NOT_STARTED routes into the on-demand warm-up below;
            # the token exchange is cached, so this is cheap when the earlier
            # failure was the probe rather than the token.
            logger.info(
                "Re-attempting readiness after a transient failure "
                f"({self._init_error!r}); the "
                f"{self.READINESS_RETRY_COOLDOWN_SECONDS:.0f}s cooldown has "
                "elapsed"
            )
            self._init_state = InitializationState.NOT_STARTED
            self._init_failed_at = None
            # Cleared with the state: a recovered plugin reporting COMPLETED
            # while initialization_error still holds the old blip would be
            # the same kind of lie is_ready used to tell. A repeat failure
            # records a fresh error through _record_init_failure.
            self._init_error = None

        if self._init_state == InitializationState.NOT_STARTED:
            # No background task, initialize on demand
            try:
                await asyncio.wait_for(self._initialize_pool(), timeout=timeout)
                # _initialize_pool records its own outcome and swallows
                # non-auth errors (streams may still succeed on demand), so
                # the state - not the absence of an exception - is the result.
            except asyncio.TimeoutError:
                self._record_init_failure(
                    asyncio.TimeoutError(
                        f"Initialization timed out after {timeout}s"
                    )
                )
                return False
            except Exception as e:
                self._record_init_failure(e)
                return False
        elif (
            self._init_state != InitializationState.COMPLETED
            and self._init_task is not None
        ):
            # Wait for background task
            try:
                await asyncio.wait_for(
                    asyncio.shield(self._init_task),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                self._record_init_failure(
                    asyncio.TimeoutError(
                        f"Initialization timed out after {timeout}s"
                    )
                )
                return False
            except Exception:
                # Error already stored (with its timestamp) by
                # _initialize_pool's _record_init_failure
                pass

        if self._init_state != InitializationState.COMPLETED:
            return False

        # Token warm-up succeeded; now prove the WebSocket path (once).
        try:
            await self._validate_websocket_path(timeout)
        except asyncio.CancelledError:
            raise
        except Exception as e:
            self._record_init_failure(e)
            logger.error(
                f"WebSocket reachability validation failed: {e!r} "
                "(state: FAILED - the token endpoint works but the "
                "WebSocket path does not)"
            )
            return False
        # is_ready, not a bare True: the verdict this method returns is read
        # off the one predicate, so the two can never disagree.
        return self.is_ready

    def check_initialization(self) -> None:
        """
        Raise InitializationError unless the plugin is READY (QUAL-002).

        Use this before operations that require successful initialization.

        Projection of the one readiness definition (see above
        _reachability_proven): it raises whenever that definition does not
        hold, which - deliberately - includes the case where readiness has
        simply never been verified. It used to raise only on a RECORDED
        failure, and that made it useless for the job its own docstring
        advertises: on a deployment whose WebSocket path is blocked, the HTTPS
        token warm-up succeeds and records nothing, so this method stayed
        silent and callers gating on it read the plugin as healthy while every
        stream() died on the dial. Silence now means verified, and nothing
        else.

        The consequence, stated plainly: on a brand-new plugin this raises
        until something has awaited wait_for_initialization() - the only call
        that can verify readiness, and the one this method's own message points
        at. Failing closed on unverified is the whole point; a caller who does
        not want the guarantee should not be calling a checker.

        Synchronous, so it never re-probes: it reports the last VERIFIED
        outcome. A transient failure is recoverable - the message says so -
        and clears on the next `await wait_for_initialization()` once the
        cooldown has elapsed. A rejected credential never clears.

        Raises:
            InitializationError: The plugin is not ready - it failed, it was
                closed, or its readiness has not been verified yet.

        Example:
            await stt.wait_for_initialization()
            stt.check_initialization()  # Raises unless ready
            stream = stt.stream()
        """
        if self.is_ready:
            return

        if self._closed:
            raise InitializationError(
                f"Plugin is not ready: {self._init_error}. A closed plugin "
                "cannot become ready again - create a new instance."
            ) from self._init_error

        if self._init_state == InitializationState.FAILED:
            if self._failure_is_permanent(self._init_error):
                hint = (
                    "The credential was rejected; this will not clear - fix "
                    "the API key and build a new plugin instance."
                )
            else:
                hint = (
                    "This failure looks transient; "
                    "await wait_for_initialization() re-probes once "
                    f"{self.READINESS_RETRY_COOLDOWN_SECONDS:.0f}s have "
                    "passed since it was recorded."
                )
            raise InitializationError(
                f"Plugin initialization failed: {self._init_error}. {hint}"
            ) from self._init_error

        raise InitializationError(
            "Plugin initialization is not verified "
            f"(state: {self._readiness_state().value}). Nothing has failed, "
            "but nothing has proved the deployment reachable either - the "
            "token exchange is plain HTTPS and cannot. Await "
            "wait_for_initialization() (it is cheap and cached after the "
            "first call), or pass validate_websocket=False to accept the "
            "token-only contract."
        )

    async def __aenter__(self) -> VoxistSTT:
        """
        Enter async context manager.

        Ensures the plugin is initialized before use - including, unless
        validate_websocket=False, proof that a WebSocket can actually be
        dialed (not just that the HTTPS token endpoint answers). Raises
        InitializationError if either fails.

        Example:
            async with VoxistSTT(api_key="...") as stt:
                stream = stt.stream()
                # Use stream...

        Returns:
            Self for use in context

        Raises:
            InitializationError: If initialization fails
        """
        # Wait for initialization using lifecycle-aware method (QUAL-002)
        if not await self.wait_for_initialization():
            # Raise with context from the original error
            raise InitializationError(
                f"Plugin initialization failed: {self._init_error}"
            ) from self._init_error

        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb) -> None:
        """
        Exit async context manager.

        Cleans up resources regardless of exception.

        Args:
            exc_type: Exception type if raised in context
            exc_val: Exception value if raised
            exc_tb: Exception traceback if raised
        """
        await self.aclose()
