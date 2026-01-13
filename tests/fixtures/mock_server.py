"""Mock WebSocket server for integration testing."""

import asyncio
import json
from collections.abc import Callable
from typing import Optional

import aiohttp
from aiohttp import web


class MockVoxistServer:
    """
    Mock WebSocket server simulating Voxist API protocol.

    Simulates the complete Voxist WebSocket protocol:
    1. WebSocket connection with API key authentication
    2. Connection confirmation message
    3. Configuration handshake
    4. Binary Float32 audio reception
    5. Partial and final transcription results
    6. Done signal handling

    Example:
        server = MockVoxistServer(port=8765)
        await server.start()

        # Server now accepting connections at ws://localhost:8765/ws

        await server.stop()
    """

    def __init__(
        self,
        port: int = 8765,
        host: str = "localhost",
        *,
        valid_api_key: str = "test_key",
        processing_delay_ms: int = 50,
        transcription_text: str = "bonjour monde",
        transcription_confidence: float = 0.95,
        send_interim: bool = True,
        interim_delay_ms: int = 25,
        error_mode: Optional[str] = None,
        on_audio_received: Optional[Callable] = None,
    ):
        """
        Initialize mock Voxist server.

        Args:
            port: Server port
            host: Server host
            valid_api_key: Expected API key for authentication
            processing_delay_ms: Delay before sending final result (simulates processing)
            transcription_text: Text to return in transcription
            transcription_confidence: Confidence score (0.0-1.0)
            send_interim: Whether to send interim results
            interim_delay_ms: Delay before sending interim result
            error_mode: Error simulation mode (None, "auth_failure", "disconnect")
            on_audio_received: Callback when audio is received (for testing)
        """
        self.port = port
        self.host = host
        self.valid_api_key = valid_api_key
        self.processing_delay_ms = processing_delay_ms
        self.transcription_text = transcription_text
        self.transcription_confidence = transcription_confidence
        self.send_interim = send_interim
        self.interim_delay_ms = interim_delay_ms
        self.error_mode = error_mode
        self.on_audio_received = on_audio_received

        self.app = web.Application()
        self.app.router.add_get("/ws", self.websocket_handler)
        self.runner: Optional[web.AppRunner] = None
        self.site: Optional[web.TCPSite] = None

        self.connections_count = 0
        self.audio_frames_received = 0
        self.total_audio_bytes = 0

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """
        Handle WebSocket connection.

        Implements Voxist protocol:
        1. Authenticate via query parameter
        2. Send connection confirmation
        3. Receive config message
        4. Process audio frames
        5. Send transcription results
        """
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.connections_count += 1

        try:
            # Authenticate via query parameter (support both api_key and token)
            api_key = request.query.get("api_key") or request.query.get("token")

            if self.error_mode == "auth_failure":
                await ws.close(code=1008, message=b"Invalid API key")
                return ws

            # For test tokens, accept mock_jwt_token as valid
            is_valid = api_key == self.valid_api_key or api_key == "mock_jwt_token"
            if not is_valid:
                await ws.close(code=1008, message=b"Invalid API key")
                return ws

            # Send connection confirmation (matches your backend)
            await ws.send_json({"status": "connected"})

            # Track configuration
            config_received = False
            audio_buffer = []

            # Process messages
            async for msg in ws:
                # JSON messages (config or "Done" signal)
                if msg.type == aiohttp.WSMsgType.TEXT:
                    try:
                        data = json.loads(msg.data)

                        # Config message
                        if "config" in data:
                            config_received = True
                            data["config"].get("lang", "fr")
                            data["config"].get("sample_rate", 16000)

                            # Log config (useful for debugging tests)
                            # Could send acknowledgment if needed

                    except json.JSONDecodeError:
                        # Handle "Done" string
                        if "Done" in msg.data:
                            # Client signaling end of audio
                            break

                # Binary messages (Int16 or Float32 audio)
                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self.audio_frames_received += 1
                    self.total_audio_bytes += len(msg.data)

                    # Parse audio (Int16 = 2 bytes/sample, Float32 = 4 bytes/sample)
                    # Plugin sends Int16 PCM audio
                    num_samples = len(msg.data) // 2  # Int16
                    audio_buffer.append(msg.data)

                    # Call callback if provided
                    if self.on_audio_received:
                        self.on_audio_received(msg.data, num_samples)

                    # Simulate processing and send results
                    # Send interim result after short delay
                    if self.send_interim and len(audio_buffer) == 1:
                        await asyncio.sleep(self.interim_delay_ms / 1000.0)

                        await ws.send_json({
                            "type": "partial",
                            "text": self.transcription_text.split()[0],  # First word
                            "confidence": self.transcription_confidence - 0.1,
                        })

                    # Send final result after accumulating some audio
                    if len(audio_buffer) >= 3:
                        await asyncio.sleep(self.processing_delay_ms / 1000.0)

                        await ws.send_json({
                            "type": "final",
                            "text": self.transcription_text,
                            "confidence": self.transcription_confidence,
                        })

                        # Reset for next utterance
                        audio_buffer = []

                elif msg.type == aiohttp.WSMsgType.ERROR:
                    break

            # If we have remaining audio, send final result
            if audio_buffer and config_received:
                await asyncio.sleep(self.processing_delay_ms / 1000.0)

                await ws.send_json({
                    "type": "final",
                    "text": self.transcription_text,
                    "confidence": self.transcription_confidence,
                })

        except Exception as e:
            # Log error but don't crash server
            print(f"MockVoxistServer error: {e}")

        finally:
            if not ws.closed:
                await ws.close()

        return ws

    async def start(self):
        """Start the mock WebSocket server."""
        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        print(f"Mock Voxist server started at ws://{self.host}:{self.port}/ws")

    async def stop(self):
        """Stop the mock WebSocket server."""
        if self.site:
            await self.site.stop()

        if self.runner:
            await self.runner.cleanup()

        print("Mock Voxist server stopped")

    def get_stats(self) -> dict:
        """
        Get server statistics.

        Returns:
            Dictionary with connection and audio stats
        """
        return {
            "connections_count": self.connections_count,
            "audio_frames_received": self.audio_frames_received,
            "total_audio_bytes": self.total_audio_bytes,
        }

    def reset_stats(self):
        """Reset server statistics."""
        self.connections_count = 0
        self.audio_frames_received = 0
        self.total_audio_bytes = 0


class ConfigurableMockServer(MockVoxistServer):
    """
    Extended mock server with configurable behaviors for advanced testing.

    Supports:
    - Multi-utterance handling
    - Custom response sequences
    - Connection drops
    - Latency variations
    """

    def __init__(
        self,
        port: int = 8765,
        *,
        responses: Optional[list[dict]] = None,
        disconnect_after: Optional[int] = None,
        variable_latency: bool = False,
        **kwargs
    ):
        """
        Initialize configurable mock server.

        Args:
            port: Server port
            responses: List of response dicts to send in sequence
            disconnect_after: Disconnect after N audio frames (for reconnection testing)
            variable_latency: Vary processing delay randomly (20-100ms)
            **kwargs: Additional arguments for MockVoxistServer
        """
        super().__init__(port=port, **kwargs)

        self.responses = responses or []
        self.disconnect_after = disconnect_after
        self.variable_latency = variable_latency
        self._response_index = 0

    async def websocket_handler(self, request: web.Request) -> web.WebSocketResponse:
        """Extended handler with configurable behaviors."""
        ws = web.WebSocketResponse()
        await ws.prepare(request)

        self.connections_count += 1

        try:
            # Authenticate
            api_key = request.query.get("api_key")
            if api_key != self.valid_api_key:
                await ws.close(code=1008, message=b"Invalid API key")
                return ws

            # Send connection confirmation
            await ws.send_json({"status": "connected"})

            frame_count = 0

            # Process messages
            async for msg in ws:
                if msg.type == aiohttp.WSMsgType.TEXT:
                    # Handle JSON or "Done"
                    try:
                        json.loads(msg.data)
                        # Config received
                    except json.JSONDecodeError:
                        if "Done" in msg.data:
                            break

                elif msg.type == aiohttp.WSMsgType.BINARY:
                    self.audio_frames_received += 1
                    self.total_audio_bytes += len(msg.data)
                    frame_count += 1

                    # Check disconnect condition
                    if self.disconnect_after and frame_count >= self.disconnect_after:
                        await ws.close(code=1001, message=b"Test disconnect")
                        return ws

                    # Send configured responses
                    if self.responses and self._response_index < len(self.responses):
                        response = self.responses[self._response_index]

                        # Apply variable latency if enabled
                        if self.variable_latency:
                            import random
                            delay = random.uniform(0.02, 0.1)  # 20-100ms
                        else:
                            delay = response.get("delay", 0.05)

                        await asyncio.sleep(delay)
                        await ws.send_json(response["message"])
                        self._response_index += 1

        except Exception as e:
            print(f"ConfigurableMockServer error: {e}")

        finally:
            if not ws.closed:
                await ws.close()

        return ws
