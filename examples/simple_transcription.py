"""
Simple Transcription Agent

Basic example showing real-time speech transcription with Voxist ASR.
Prints interim and final transcriptions to console.

Usage:
    export VOXIST_API_KEY="voxist_..."
    export LIVEKIT_URL="wss://your-project.livekit.cloud"
    export LIVEKIT_API_KEY="..."
    export LIVEKIT_API_SECRET="..."

    python simple_transcription.py dev
"""

import logging
from livekit import agents, rtc
from livekit.agents import cli, stt

# Import Voxist plugin
from livekit.plugins import voxist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("simple-transcription")


async def entrypoint(ctx: agents.JobContext):
    """Main agent entrypoint."""
    logger.info(f"Connecting to room: {ctx.room.name}")

    # Connect to the LiveKit room
    await ctx.connect()

    # Initialize Voxist STT with French language
    voxist_stt = voxist.VoxistSTT(
        language="fr",
        interim_results=True,  # Get partial transcriptions
        connection_pool_size=2,
    )

    logger.info("Voxist STT initialized, waiting for participants...")

    # Wait for a participant to join
    participant = await ctx.wait_for_participant()
    logger.info(f"Participant joined: {participant.identity}")

    # Process audio from participant
    async def process_track(track: rtc.Track):
        if track.kind != rtc.TrackKind.KIND_AUDIO:
            return

        logger.info(f"Processing audio track: {track.sid}")

        # Create audio stream from track
        audio_stream = rtc.AudioStream(track)

        # Create STT stream
        stt_stream = voxist_stt.stream()

        # Forward audio to STT
        async for audio_event in audio_stream:
            stt_stream.push_frame(audio_event.frame)

        # Signal end of audio
        stt_stream.end_input()

        # Process transcription events
        async for event in stt_stream:
            if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                text = event.alternatives[0].text if event.alternatives else ""
                if text:
                    logger.info(f"[INTERIM] {text}")

            elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                text = event.alternatives[0].text if event.alternatives else ""
                if text:
                    logger.info(f"[FINAL] {text}")
                    # You can do something with the final transcript here
                    # e.g., send to LLM, save to database, etc.

    # Listen for track subscriptions
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track subscribed from {participant.identity}")
            ctx.create_task(process_track(track))

    # Keep agent running
    logger.info("Agent ready, listening for audio...")


if __name__ == "__main__":
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            # Agent will handle one room at a time
            num_idle_processes=1,
        )
    )
