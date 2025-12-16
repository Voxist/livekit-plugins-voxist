"""
Direct Room Join - Voxist Transcription

Explicitly joins a specific room and transcribes audio.
No dispatch needed - connects directly.

Usage:
    export VOXIST_API_KEY="..."
    # Optional: override default API endpoint
    # export VOXIST_BASE_URL="wss://api-asr.voxist.com/ws"

    python examples/join_room.py
"""

import asyncio
import logging
import os

from livekit import api, rtc
from livekit.agents import stt

# Import Voxist plugin
from livekit.plugins import voxist

# Configure logging - DEBUG to see audio flow
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger("voxist-transcription")
# Reduce noise from other loggers
logging.getLogger("aiohttp").setLevel(logging.WARNING)
logging.getLogger("livekit").setLevel(logging.INFO)

# Configuration
LIVEKIT_URL = os.environ.get("LIVEKIT_URL", "ws://localhost:7880")
LIVEKIT_API_KEY = os.environ.get("LIVEKIT_API_KEY", "devkey")
LIVEKIT_API_SECRET = os.environ.get("LIVEKIT_API_SECRET", "secret")
ROOM_NAME = os.environ.get("ROOM_NAME", "test-room")


async def main():
    logger.info(f"Connecting to room: {ROOM_NAME}")

    # Create access token for agent
    token = api.AccessToken(LIVEKIT_API_KEY, LIVEKIT_API_SECRET)
    token.with_identity("voxist-agent")
    token.with_name("Voxist Transcriber")
    token.with_grants(api.VideoGrants(
        room_join=True,
        room=ROOM_NAME,
    ))
    jwt_token = token.to_jwt()

    # Initialize Voxist STT
    # LiveKit sends audio at 48kHz, so we configure for that
    voxist_stt = voxist.VoxistSTT(
        api_key=os.environ.get("VOXIST_API_KEY"),
        base_url=os.environ.get("VOXIST_BASE_URL", "wss://asr-staging-dev.voxist.com/ws"),
        language="fr",
        sample_rate=48000,  # Match LiveKit's 48kHz output
        interim_results=True,
        connection_pool_size=2,
        chunk_duration_ms=500,  # Larger chunks (500ms instead of 100ms)
    )

    # Create room and connect
    room = rtc.Room()

    @room.on("participant_connected")
    def on_participant_connected(participant: rtc.RemoteParticipant):
        logger.info(f"Participant connected: {participant.identity}")

    @room.on("track_subscribed")
    def on_track_subscribed(
        track: rtc.Track,
        publication: rtc.TrackPublication,
        participant: rtc.RemoteParticipant,
    ):
        if track.kind == rtc.TrackKind.KIND_AUDIO:
            logger.info(f"Audio track from {participant.identity} - starting transcription")
            asyncio.create_task(transcribe_track(track, participant.identity, voxist_stt))

    # Connect to room
    await room.connect(LIVEKIT_URL, jwt_token)
    logger.info(f"Connected to room: {room.name}")
    logger.info(f"Participants: {[p.identity for p in room.remote_participants.values()]}")

    # Process existing participants
    for participant in room.remote_participants.values():
        for publication in participant.track_publications.values():
            if publication.track and publication.track.kind == rtc.TrackKind.KIND_AUDIO:
                logger.info(f"Found existing audio track from {participant.identity}")
                asyncio.create_task(transcribe_track(publication.track, participant.identity, voxist_stt))

    logger.info("Agent ready - speak in the browser!")
    logger.info("Press Ctrl+C to exit")

    # Keep running
    try:
        while True:
            await asyncio.sleep(1)
    except KeyboardInterrupt:
        logger.info("Shutting down...")
    finally:
        await room.disconnect()
        await voxist_stt.aclose()


async def transcribe_track(track: rtc.Track, participant_id: str, voxist_stt):
    """Transcribe audio from a track."""
    logger.info(f"Starting transcription for {participant_id}")

    try:
        audio_stream = rtc.AudioStream(track)
        stt_stream = voxist_stt.stream()

        frame_count = 0

        # Task to read transcriptions
        async def read_transcriptions():
            logger.info("read_transcriptions task started")
            async for event in stt_stream:
                logger.info(f"Got STT event: {event.type}")
                if event.type == stt.SpeechEventType.INTERIM_TRANSCRIPT:
                    text = event.alternatives[0].text if event.alternatives else ""
                    if text.strip():
                        print(f"[{participant_id}] (interim): {text}")

                elif event.type == stt.SpeechEventType.FINAL_TRANSCRIPT:
                    text = event.alternatives[0].text if event.alternatives else ""
                    if text.strip():
                        print(f"[{participant_id}] FINAL: {text}")
            logger.info("read_transcriptions task ended")

        # Start reading transcriptions
        read_task = asyncio.create_task(read_transcriptions())

        # Forward audio to STT
        logger.info("Starting to forward audio frames...")
        async for audio_event in audio_stream:
            frame = audio_event.frame
            frame_count += 1
            if frame_count == 1:
                logger.info(f"First frame: sample_rate={frame.sample_rate}, channels={frame.num_channels}, samples={frame.samples_per_channel}")
                data_bytes = bytes(frame.data)
                logger.info(f"Frame data: {len(data_bytes)} bytes for {frame.samples_per_channel} samples = {len(data_bytes)/frame.samples_per_channel} bytes/sample")
                # 2 bytes/sample = Int16, 4 bytes/sample = Float32
                import numpy as np
                if len(data_bytes) == frame.samples_per_channel * 2:
                    logger.info("Audio format: Int16")
                elif len(data_bytes) == frame.samples_per_channel * 4:
                    logger.info("Audio format: likely Float32 or Int32")
                    # Try to interpret as Float32
                    arr = np.frombuffer(data_bytes, dtype=np.float32)
                    logger.info(f"Float32 range: min={arr.min():.4f}, max={arr.max():.4f}")
            if frame_count % 100 == 0:
                logger.info(f"Forwarded {frame_count} audio frames")
            stt_stream.push_frame(frame)

        # End of audio
        logger.info(f"Audio stream ended, total frames: {frame_count}")
        stt_stream.end_input()
        await read_task

    except Exception as e:
        logger.error(f"Transcription error for {participant_id}: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())
