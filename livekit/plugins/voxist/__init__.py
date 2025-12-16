"""
LiveKit STT plugin for Voxist ASR.

Features:
- Connection pooling for ultra-low latency (< 300ms)
- Support for 8+ languages including French medical
- Automatic text2num and medical units processing
- Production-ready reliability with auto-reconnection

Example:
    from livekit import agents
    from livekit.plugins import voxist

    async def entrypoint(ctx: agents.JobContext):
        stt = voxist.VoxistSTT(language="fr-medical")
        agent = agents.VoicePipelineAgent(stt=stt, llm=..., tts=...)
        await agent.start(ctx.room)
"""

from .exceptions import (
    AuthenticationError,
    ConfigurationError,
    ConnectionError,
    ConnectionPoolExhaustedError,
    InitializationError,
    InsufficientBalanceError,
    LanguageNotSupportedError,
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
    "ConnectionPoolExhaustedError",
    "LanguageNotSupportedError",
    "ConfigurationError",
    "InitializationError",
]
