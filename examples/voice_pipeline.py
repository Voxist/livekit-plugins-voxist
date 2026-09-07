"""
Voice Pipeline Agent

Full voice agent with STT -> LLM -> TTS pipeline.
Uses Voxist for speech-to-text, OpenAI for LLM and TTS, and Silero for VAD.

Usage:
    export VOXIST_API_KEY="voxist_..."
    export OPENAI_API_KEY="sk-..."
    export LIVEKIT_URL="wss://your-project.livekit.cloud"
    export LIVEKIT_API_KEY="..."
    export LIVEKIT_API_SECRET="..."

    python voice_pipeline.py dev
"""

import logging

from livekit.agents import Agent, AgentSession, cli

from livekit import agents

# Import plugins
from livekit.plugins import openai, silero, voxist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-pipeline")


def prewarm(proc: agents.JobProcess):
    """Load the VAD model once per worker process, not once per room."""
    proc.userdata["vad"] = silero.VAD.load()


async def entrypoint(ctx: agents.JobContext):
    """Main agent entrypoint for voice pipeline."""
    logger.info(f"Starting voice pipeline agent for room: {ctx.room.name}")

    # Connect to room
    await ctx.connect()

    session = AgentSession(
        vad=ctx.proc.userdata["vad"],
        stt=voxist.VoxistSTT(language="fr", interim_results=True),
        llm=openai.LLM(model="gpt-4o-mini", temperature=0.7),
        tts=openai.TTS(voice="alloy"),
    )

    @session.on("user_input_transcribed")
    def on_user_input(event):
        if event.is_final:
            logger.info(f"User said: {event.transcript}")

    agent = Agent(
        instructions=(
            "You are a helpful French-speaking assistant. "
            "Keep your responses concise and natural for voice conversation. "
            "Respond in French unless the user speaks in another language. "
            "Be friendly and conversational."
        ),
    )

    await session.start(agent=agent, room=ctx.room)
    logger.info("Voice pipeline agent started, ready for conversation...")

    await session.say("Bonjour ! Comment puis-je vous aider ?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            prewarm_fnc=prewarm,
        )
    )
