"""
LiveKit STT plugin for Voxist ASR.

Features:
- One WebSocket per stream, aligned with the gateway's session protocol
- Support for 8+ languages including French medical
- Automatic text2num and medical units processing
- Reliability owned by livekit's own retry (conn_options.max_retry)

Example:
    from livekit import agents
    from livekit.plugins import voxist

    async def entrypoint(ctx: agents.JobContext):
        stt = voxist.VoxistSTT(language="fr-medical")
        agent = agents.VoicePipelineAgent(stt=stt, llm=..., tts=...)
        await agent.start(ctx.room)

Deprecated exception names:
    ConnectionPoolExhaustedError is a pool-era name that nothing raises any
    more; it is still exported so existing `except` clauses keep importing.
    Catch ConnectionError instead. Two further pool-era names,
    BackpressureError and OwnershipViolationError, remain importable from
    livekit.plugins.voxist.exceptions but are not raised either. Each class's
    docstring names what replaced it.
"""

from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ConnectionPoolExhaustedError,
    InitializationError,
    InsufficientBalanceError,
    LanguageNotSupportedError,
    TranscriptLostError,
    VoxistError,
)
from .stt import InitializationState, VoxistSTT
from .version import __version__

__all__ = [
    "VoxistSTT",
    "InitializationState",
    "__version__",
    "VoxistError",
    "AuthenticationError",
    "InsufficientBalanceError",
    "ConnectionError",
    "TranscriptLostError",
    # Deprecated, never raised; exported so existing user code still imports.
    "ConnectionPoolExhaustedError",
    "LanguageNotSupportedError",
    "ConfigurationError",
    "InitializationError",
]
