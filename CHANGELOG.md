# Changelog

All notable changes to `livekit-plugins-voxist` are documented here.
The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/);
versions follow [Semantic Versioning](https://semver.org/) (pre-1.0: a minor
bump may change behaviour).

## [0.10.0] - 2026-09-07

The transport was rewritten. Every name exported from the package root in
0.9.x still imports (see *Removed* for the two submodules that changed), but
runtime behaviour changes in ways you should read before upgrading. See
[Upgrading from 0.9.x](https://github.com/voxist/livekit-plugins-voxist#upgrading-from-09x) in the README for the
short version.

### Changed

- **One WebSocket per stream; the connection pool is gone.** The Voxist
  gateway closes every socket after `Done`, so sockets were never reusable.
  Each `stream()` now dials its own socket on demand, with the stream's
  language in the URL.
- **Retries belong to LiveKit.** The plugin no longer runs its own
  reconnect loop. Interruptions surface as `APIConnectionError`, and
  `RecognizeStream` re-attempts them according to `conn_options.max_retry`
  and `retry_interval` on `stream()`. Authentication failures are not
  retried.
- **`flush()` is a segment boundary, `end_input()` ends the session.**
  0.9.x sent `Done` on every `flush()`, which made the gateway close the
  socket after the first VAD turn. `Done` is now sent exactly once, at
  `end_input()`, and a multi-turn conversation runs on a single socket. A
  mid-session `flush()` injects 400 ms of silence so the engine endpoints
  the utterance.
- **Sessions never report success over a lost transcript.** Every attempt
  ends through a single completion gate. When audio was consumed and no
  transcript can be recovered by a retry, the stream raises
  `TranscriptLostError`: one `recoverable=False` error event, then the
  stream stops. Ambiguous endings (a short unanswered tail that a loaded
  engine could still explain) complete with a warning; unbounded silence
  from the engine is fatal.
- **Mid-session stall detection.** Roughly 30 s of caller audio with no
  transcript back raises `APIConnectionError` so LiveKit redials. Silent
  users, quiet speakers and batch senders do not trip it.
- **Bounded input backlog.** More than 120 s of queued audio is dropped,
  oldest first, with a rate-limited warning; `dropped_frames` reports it.
  Dropped audio bars the session from completing as a clean empty one.
- **Process-wide dial throttle.** At most 30 dial attempts per 60 s per
  credential, shared across every `VoxistSTT` in the process. A caller that
  is early waits for its slot rather than failing.
- **Readiness has one definition.** `wait_for_initialization()` and
  `async with` prove WebSocket reachability with one short probe
  (`validate_websocket=True` by default). `InitializationState.COMPLETED`
  now means ready and nothing weaker. A transient failure is re-probed
  after a 30 s cooldown; a rejected key stays failed. `is_ready` is `False`
  after `aclose()`.
- **Fast turn ending.** The engine's `Done!` acknowledgement is recognised
  (0.9.x logged it as invalid JSON on every session), and a turn ends about
  0.5 s after the engine's last post-`Done` final instead of waiting out a
  5 s drain.
- **Emitted language codes are BCP-47.** LiveKit normalises
  `SpeechData.language`, so `fr-medical` is emitted as `fr-MEDICAL`. The raw
  code is still what Voxist receives.
- **`connection_timeout` bounds the token exchange too.** 0.9.x applied it
  to the WebSocket dial only; the HTTPS token exchange had a fixed 10 s
  timeout.
- **Token lifetime comes from the JWT.** Expiry is read from the token's
  `exp` claim instead of an assumed hour.
- **Programming errors are not retried.** `stream()` on a closed plugin,
  and use from a different live event loop, raise `RuntimeError` at the
  call site instead of burning LiveKit's retry budget.
- `livekit-agents>=1.6.0` and `aiohttp>=3.10` are now required (were
  `>=0.8.0` and `>=3.9.0`).

### Added

- `punctuation_mode="Dictated"` on `VoxistSTT` disables the engine's
  automatic punctuation so spoken punctuation is used. French only, and
  requires the gateway's V2 feature flag. Any other value raises
  `ConfigurationError` because the gateway silently ignores it.
- `ssl_context` on `VoxistSTT`, for deployments whose certificate is signed
  by a private CA. Certificate verification is never disabled.
- `validate_websocket` on `VoxistSTT` to make readiness token-only again.
- `TranscriptLostError`, exported from the package root.
- `VoxistSTTStream.dropped_frames`.
- Detection of a lost trailing utterance from the engine's segment
  numbering, reported as a warning on completion.
- A stale cached token rejected at the handshake is refreshed once and the
  dial retried, instead of failing the stream with `AuthenticationError`.

### Fixed

- Audio stopped being sent after roughly 90 to 110 s of a call. A phantom
  write-buffer counter never decayed, crossed its high-water mark and paused
  sending forever. The plugin no longer gates sends on its own byte counts;
  aiohttp's transport flow control owns backpressure, every send is bounded
  by a 5 s timeout, and teardown of a wedged socket is bounded to 1 s.
- The final transcript of every utterance was discarded: the receive task
  was cancelled as soon as `Done` was written.
- A per-stream `language` override was ignored: a pooled socket opened for
  `fr` transcribed with the French engine and labelled the result `en`.
- A result left unread on a pooled socket could be delivered to the next
  stream under its request id.
- Frames longer than the 2 s ring buffer, or over 1 MB, were truncated to
  their tail. They are now sliced and transcribed whole.
- An invalid API key surfaced as a generic `ConnectionError` and was
  retried.
- A `null` or non-numeric `confidence` in an engine frame is reported as
  1.0 on both the interim and final paths.
- The first dropped-frame warning was suppressed on hosts with short uptime.
- Log sanitisation now covers `%`-style arguments and exception tracebacks,
  and the plugin logger no longer forces its own level.
- Every unmapped failure in the token exchange and dial (timeouts, malformed
  JSON, closed session) now maps to `ConnectionError` so LiveKit can retry.

### Deprecated

- `connection_pool_size` and `max_reconnect_attempts` on `VoxistSTT` are
  accepted and ignored. A non-default value logs a warning each time a
  `VoxistSTT` is constructed. The old 1 to 5 range check on
  `connection_pool_size` is gone.
- `ConnectionPoolExhaustedError`, `BackpressureError` and
  `OwnershipViolationError` are never raised. They stay importable from
  where they were (`ConnectionPoolExhaustedError` from the package root, all
  three from `livekit.plugins.voxist.exceptions`) so existing `except`
  clauses keep working; catch `ConnectionError` and `TranscriptLostError`
  instead.

### Removed

- The `livekit.plugins.voxist.connection_pool` module, with `ConnectionPool`.
- `Connection` and `ConnectionState` from `livekit.plugins.voxist.models`.
  Neither was exported from the package root; code that imported them from
  the submodule will fail at import.

## [0.9.1] - 2026-01-13

Last release of the pooled-connection architecture.

### Changed

- Python 3.9 is no longer supported; `requires-python` is `>=3.10`.
- Plugin logging aligned with LiveKit's conventions, with less noise at
  INFO.

## [0.9.0] - 2025-12-16

First PyPI release.

[0.10.0]: https://github.com/voxist/livekit-plugins-voxist/compare/0.9.1...v0.10.0
[0.9.1]: https://github.com/voxist/livekit-plugins-voxist/compare/v0.9.0...0.9.1
[0.9.0]: https://github.com/voxist/livekit-plugins-voxist/releases/tag/v0.9.0
