"""
Voice Pipeline Agent

Full voice agent with STT → LLM → TTS pipeline.
Uses Voxist for speech-to-text, OpenAI for LLM, and ElevenLabs for TTS.

Usage:
    export VOXIST_API_KEY="voxist_..."
    export OPENAI_API_KEY="sk-..."
    export ELEVEN_API_KEY="..."  # Optional, for TTS
    export LIVEKIT_URL="wss://your-project.livekit.cloud"
    export LIVEKIT_API_KEY="..."
    export LIVEKIT_API_SECRET="..."

    python voice_pipeline.py dev
"""

import logging
from livekit import agents
from livekit.agents import cli
from livekit.agents.voice_assistant import VoiceAssistant

# Import plugins
from livekit.plugins import voxist, openai, silero

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("voice-pipeline")


async def entrypoint(ctx: agents.JobContext):
    """Main agent entrypoint for voice pipeline."""
    logger.info(f"Starting voice pipeline agent for room: {ctx.room.name}")

    # Connect to room
    await ctx.connect()

    # Initialize STT with Voxist
    stt = voxist.VoxistSTT(
        language="fr",
        interim_results=True,
        connection_pool_size=2,
    )

    # Initialize VAD (Voice Activity Detection)
    vad = silero.VAD.load()

    # Initialize LLM
    llm = openai.LLM(
        model="gpt-4o-mini",
        temperature=0.7,
    )

    # Initialize TTS (optional - can use openai.TTS as alternative)
    try:
        from livekit.plugins import elevenlabs
        tts = elevenlabs.TTS(
            voice="Rachel",
            model_id="eleven_multilingual_v2",
        )
        logger.info("Using ElevenLabs TTS")
    except (ImportError, Exception):
        # Fall back to OpenAI TTS
        tts = openai.TTS(
            voice="alloy",
            model="tts-1",
        )
        logger.info("Using OpenAI TTS (ElevenLabs not available)")

    # System prompt for the assistant
    initial_ctx = llm.ChatContext()
    initial_ctx.messages.append(
        openai.ChatMessage(
            role="system",
            content="""You are a helpful French-speaking assistant.

            Keep your responses concise and natural for voice conversation.
            Respond in French unless the user speaks in another language.
            Be friendly and conversational.""",
        )
    )

    # Create voice assistant
    assistant = VoiceAssistant(
        vad=vad,
        stt=stt,
        llm=llm,
        tts=tts,
        chat_ctx=initial_ctx,
        # Allow interruptions
        allow_interruptions=True,
        # Minimum speech duration before processing
        min_endpointing_delay=0.5,
    )

    # Event handlers
    @assistant.on("user_speech_committed")
    def on_user_speech(user_msg: str):
        logger.info(f"User said: {user_msg}")

    @assistant.on("agent_speech_committed")
    def on_agent_speech(agent_msg: str):
        logger.info(f"Agent said: {agent_msg}")

    @assistant.on("agent_speech_interrupted")
    def on_interrupted():
        logger.info("Agent was interrupted")

    # Wait for participant and start
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Start the assistant
    assistant.start(ctx.room, participant)

    logger.info("Voice pipeline agent started, ready for conversation...")

    # Keep running until room closes
    await assistant.say("Bonjour! Comment puis-je vous aider?", allow_interruptions=True)


if __name__ == "__main__":
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            num_idle_processes=1,
        )
    )
