"""
Medical Transcription Agent

Example showing French medical transcription with Voxist ASR.
Uses the specialized fr-medical model for accurate medical terminology.

Usage:
    export VOXIST_API_KEY="voxist_..."
    export LIVEKIT_URL="wss://your-project.livekit.cloud"
    export LIVEKIT_API_KEY="..."
    export LIVEKIT_API_SECRET="..."

    python medical_transcription.py dev
"""

import asyncio
import logging
from datetime import datetime
from livekit import agents, rtc
from livekit.agents import cli, stt

# Import Voxist plugin
from livekit.plugins import voxist

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("medical-transcription")


class TranscriptionSession:
    """Manages a transcription session with buffering."""

    def __init__(self, participant_id: str):
        self.participant_id = participant_id
        self.segments: list[dict] = []
        self.start_time = datetime.now()

    def add_segment(self, text: str, is_final: bool):
        """Add a transcription segment."""
        self.segments.append({
            "text": text,
            "timestamp": datetime.now().isoformat(),
            "is_final": is_final,
        })

    def get_full_transcript(self) -> str:
        """Get the full transcript from final segments only."""
        return " ".join(
            seg["text"] for seg in self.segments if seg["is_final"]
        )


async def entrypoint(ctx: agents.JobContext):
    """Main agent entrypoint for medical transcription."""
    logger.info(f"Starting medical transcription agent for room: {ctx.room.name}")

    await ctx.connect()

    # Initialize Voxist STT with French medical model
    voxist_stt = voxist.VoxistSTT(
        language="fr-medical",  # Specialized medical vocabulary
        interim_results=True,
        connection_pool_size=3,  # More connections for reliability
    )

    logger.info("Voxist Medical STT initialized")

    # Track sessions per participant
    sessions: dict[str, TranscriptionSession] = {}

    participant = await ctx.wait_for_participant()
    session = TranscriptionSession(participant.identity)
    sessions[participant.identity] = session
    logger.info(f"Created session for participant: {participant.identity}")

    async def process_audio_track(track: rtc.Track, participant_id: str):
        """Process audio track and generate transcriptions."""
        session = sessions.get(participant_id)
        if not session:
            return

        audio_stream = rtc.AudioStream(track)
        stt_stream = voxist_stt.stream()

        # Task to forward audio frames
        async def forward_audio():
            async for audio_event in audio_stream:
                stt_stream.push_frame(audio_event.frame)
            stt_stream.end_input()

        # Task to process transcription events
        async def process_transcriptions():
            async for event in stt_stream:
                if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    text = event.alternatives[0].text if event.alternatives else ""
                    if text:
                        session.add_segment(text, is_final=False)
                        # Print interim with indicator
                        print(f"\r[...] {text}", end="", flush=True)

                elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = event.alternatives[0].text if event.alternatives else ""
                    if text:
                        session.add_segment(text, is_final=True)
                        # Print final with newline
                        print(f"\r[MED] {text}")

                        # Log confidence if available
                        if event.alternatives and event.alternatives[0].confidence:
                            logger.debug(
                                f"Confidence: {event.alternatives[0].confidence:.2%}"
                            )

        # Run both tasks concurrently
        await asyncio.gather(forward_audio(), process_transcriptions())

        # Log session summary when done
        full_transcript = session.get_full_transcript()
        logger.info(
            f"Session complete for {participant_id}. "
            f"Total segments: {len(session.segments)}, "
            f"Characters: {len(full_transcript)}"
        )

    @ctx.room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Processing audio from {participant.identity}")
            ctx.create_task(process_audio_track(track, participant.identity))

    @ctx.room.on("participant_disconnected")
    def on_participant_disconnected(participant: rtc.RemoteParticipant):
        if participant.identity in sessions:
            session = sessions[participant.identity]
            logger.info(
                f"Participant {participant.identity} disconnected. "
                f"Final transcript: {session.get_full_transcript()[:100]}..."
            )
            del sessions[participant.identity]

    logger.info("Medical transcription agent ready, waiting for audio...")


if __name__ == "__main__":
    cli.run_app(
        agents.WorkerOptions(
            entrypoint_fnc=entrypoint,
            num_idle_processes=1,
        )
    )
