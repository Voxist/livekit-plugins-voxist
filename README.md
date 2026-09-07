# LiveKit Plugin for Voxist ASR

Production-ready LiveKit Speech-to-Text plugin for [Voxist](https://voxist.com) automatic speech recognition API.

## Features

- **Real-time streaming:** one WebSocket per stream, dialed on demand, with the
  token exchange pre-fetched at startup so the first dial is not slowed by it
- **Multi-language support:** French, English, German, Italian, Spanish, Dutch, Portuguese, Swedish
- **Production-ready:** retries owned by LiveKit's own `conn_options.max_retry`,
  mapped error taxonomy, comprehensive testing
- **Simple API:** 3-line integration with LiveKit agents

## Installation

```bash
pip install livekit-plugins-voxist
```

## Upgrading from 0.9.x

0.10.0 replaces the connection pool with one WebSocket per stream and hands
retries to LiveKit. Every name the package root exported in 0.9.x still
imports, but three things deserve a check in your code:

1. **Drop `connection_pool_size` and `max_reconnect_attempts`.** Both are
   accepted and ignored, and a non-default value logs a deprecation warning.
   Retries are configured on the stream instead:

   ```python
   from livekit.agents import APIConnectOptions

   stream = stt.stream(conn_options=APIConnectOptions(max_retry=3, retry_interval=2.0))
   ```

2. **Catch `ConnectionError` and `TranscriptLostError`.**
   `ConnectionPoolExhaustedError` (package root) and `BackpressureError` /
   `OwnershipViolationError` (`livekit.plugins.voxist.exceptions`) still
   import but are never raised. `TranscriptLostError` is new: audio was
   consumed and no transcript can be recovered, so the stream ends with one
   non-recoverable error event instead of a fabricated empty result.

3. **Compare `SpeechData.language` case-insensitively.** LiveKit normalises
   it to BCP-47, so `fr-medical` is emitted as `fr-MEDICAL`. The raw code is
   still what Voxist receives.

Everything else is behavioural: one socket per stream, `flush()` marks a
segment and `end_input()` sends `Done` once, and readiness checks probe the
WebSocket path. The full list is in [CHANGELOG.md](https://github.com/voxist/livekit-plugins-voxist/blob/main/CHANGELOG.md).

## Quick Start

```python
from livekit import agents
from livekit.agents import Agent, AgentSession, cli
from livekit.plugins import voxist, openai, elevenlabs

async def entrypoint(ctx: agents.JobContext):
    await ctx.connect()

    session = AgentSession(
        stt=voxist.VoxistSTT(language="fr"),
        llm=openai.LLM(model="gpt-4o-mini"),
        tts=elevenlabs.TTS(),
    )
    await session.start(
        agent=Agent(instructions="Tu es un assistant vocal. Réponds en français."),
        room=ctx.room,
    )

if __name__ == "__main__":
    cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
```

## Configuration

### Environment Variables

```bash
# Required
export VOXIST_API_KEY="voxist_..."

# Optional
export VOXIST_BASE_URL="wss://api-asr.voxist.com/ws"
export VOXIST_LANGUAGE="fr"
```

### Advanced Configuration

```python
stt = voxist.VoxistSTT(
    api_key="voxist_...",
    language="fr",
    sample_rate=16000,
    interim_results=True,
    chunk_duration_ms=100,       # Audio chunk size
    stride_overlap_ms=20,        # Chunk overlap for accuracy
    connection_timeout=10.0,     # Bounds the token exchange and the dial
    punctuation_mode=None,       # "Dictated" keeps spoken punctuation (fr only, gateway V2 flag)
    ssl_context=None,            # For a private-CA deployment; verification is never disabled
    validate_websocket=True,     # wait_for_initialization() / async with probe the WebSocket once
)
```

## Supported Languages

| Code | Language |
|------|----------|
| `fr`, `fr-FR` | French |
| `en`, `en-US` | English |
| `de`, `de-DE` | German |
| `it` | Italian |
| `es` | Spanish |
| `nl`, `nl-NL` | Dutch |
| `pt` | Portuguese |
| `sv` | Swedish |

## Performance

- **Latency:** < 300ms end-to-end (95th percentile)
- **Startup:** the WebSocket token is pre-fetched at construction, so the first
  stream's dial does not pay for the HTTPS token exchange
- **Recovery:** retries are LiveKit's, governed by `conn_options.max_retry` and
  `retry_interval` on `stream()`; a session whose audio was consumed with no
  transcript raises `TranscriptLostError` rather than faking success
- **Memory:** < 50MB per agent instance
- **CPU:** < 10% for audio processing

## Documentation

- **Changelog:** [CHANGELOG.md](https://github.com/voxist/livekit-plugins-voxist/blob/main/CHANGELOG.md)
- **API Reference:** the `VoxistSTT` constructor docstring documents every parameter
- **Examples:** See `examples/` directory

## Local Development with Docker

### 1. Setup Python Environment

```bash
# Create virtual environment
python -m venv .venv

# Activate (macOS/Linux)
source .venv/bin/activate

# Activate (Windows)
.venv\Scripts\activate

# Install package with dev dependencies
pip install -e ".[dev]"
```

### 2. Start LiveKit Server

```bash
docker run --rm -p 7880:7880 -p 7881:7881 -p 7882:7882/udp \
    livekit/livekit-server \
    --dev \
    --bind 0.0.0.0
```

The server runs with development keys:
- **API Key:** `devkey`
- **API Secret:** `secret`

### 3. Install LiveKit CLI

```bash
# macOS
brew install livekit-cli

# Linux
curl -sSL https://get.livekit.io/cli | bash

# Or download from: https://github.com/livekit/livekit-cli/releases
```

### 4. Generate Access Token

```bash
# Generate token for a test room
livekit-cli create-token \
    --api-key devkey \
    --api-secret secret \
    --join --room test-room \
    --identity user1 \
    --valid-for 24h
```

### 5. Expose with ngrok (for external access)

To test from external devices or meet.livekit.io (which requires HTTPS):

```bash
# Install ngrok: https://ngrok.com/download
# Then expose LiveKit server
ngrok http 7880
```

Note the ngrok URL (e.g., `https://xxxx.ngrok-free.app`)

### 6. Test with meet.livekit.io

1. Go to [meet.livekit.io](https://meet.livekit.io)
2. Click **"Custom Server"**
3. Enter:
   - **LiveKit URL:** `wss://xxxx.ngrok-free.app` (use your ngrok URL with `wss://`)
   - **Token:** *(paste the token from step 3)*
4. Click **"Connect"**

> **Note:** For local testing without ngrok, use `ws://localhost:7880`

### 7. Run the Agent

```bash
# Set environment variables
export LIVEKIT_URL=ws://localhost:7880
export LIVEKIT_API_KEY=devkey
export LIVEKIT_API_SECRET=secret
export VOXIST_API_KEY=your_voxist_api_key

# Run the agent
python examples/simple_transcription.py dev
```

The agent will connect to the room and transcribe audio from participants.

### Staging Backend

For testing against Voxist staging:

```python
stt = voxist.VoxistSTT(
    api_key="your_staging_key",
    base_url="wss://asr-staging-dev.voxist.com/ws",
    language="fr",
)
```

## Development Status

🚧 **Under Active Development**

Track progress with beads issue tracker:
```bash
/beads:stats              # View project statistics
/beads:ready              # See ready-to-work issues
/beads:show livekit-1     # View epic details
```

## License

MIT - See [LICENSE](LICENSE) file

## Support

- **Issues:** https://github.com/voxist/livekit-plugins-voxist/issues
- **Voxist API:** https://api-asr.voxist.com/docs
- **LiveKit:** https://docs.livekit.io
