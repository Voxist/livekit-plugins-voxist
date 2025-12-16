# LiveKit Voxist Plugin Examples

Example agents demonstrating Voxist ASR integration with LiveKit.

## Prerequisites

```bash
# Install the plugin
cd /path/to/livekit-plugins-voxist
pip install -e ".[dev]"

# Set environment variables
export VOXIST_API_KEY="voxist_..."
export LIVEKIT_URL="wss://your-project.livekit.cloud"  # or ws://localhost:7880
export LIVEKIT_API_KEY="..."
export LIVEKIT_API_SECRET="..."
```

## Examples

### 1. Simple Transcription (`simple_transcription.py`)

Basic real-time transcription agent. Prints all transcriptions to console.

```bash
python examples/simple_transcription.py dev
```

### 2. Medical Transcription (`medical_transcription.py`)

French medical transcription with automatic number/unit conversion.

```bash
python examples/medical_transcription.py dev
```

### 3. Voice Pipeline (`voice_pipeline.py`)

Full voice agent with STT → LLM → TTS pipeline.

```bash
# Requires OpenAI API key
export OPENAI_API_KEY="sk-..."
python examples/voice_pipeline.py dev
```

## Running with Local LiveKit Server

```bash
# Start local LiveKit server
docker run -d --name livekit \
  -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
  livekit/livekit-server --dev

# Set local URL
export LIVEKIT_URL="ws://localhost:7880"

# Run example
python examples/simple_transcription.py dev
```

## Testing with LiveKit Playground

1. Go to https://meet.livekit.io
2. Enter your LiveKit server URL and credentials
3. Join a room
4. Speak and watch transcriptions appear in the agent console
