"""AudioProcessor - Optimized audio format conversion and chunking."""


import numpy as np

from .log import logger

# Security limits to prevent resource exhaustion attacks (SEC-HIGH: asr-all-gt1)
MAX_FRAME_SIZE_BYTES = 1024 * 1024  # 1MB max single frame
MAX_BUFFER_SAMPLES = 480000  # 10 seconds at 48kHz (highest supported rate)

# Minimum frame size validation (SEC-004: asr-all-6og)
# 10ms at 8kHz (lowest supported rate) * 2 bytes per sample = 160 bytes
MIN_FRAME_SIZE_BYTES = 160


class AudioProcessor:
    """
    Optimized audio processing pipeline with NumPy vectorization.

    Chunks Int16 PCM audio for Voxist WebSocket API,
    with 100ms chunking and 20ms stride overlap for improved boundary accuracy.

    NOTE: Voxist expects raw Int16 PCM bytes, NOT Float32.

    Performance:
        - < 5ms conversion time for 100ms audio chunk (target)
        - NumPy SIMD operations ~10x faster than Python loops
        - Efficient buffer management for partial frames

    Example:
        processor = AudioProcessor(sample_rate=16000)

        # Process audio frames
        for frame_data in audio_frames:
            chunks = processor.process_audio_frame(frame_data)
            for chunk in chunks:
                await send_to_websocket(chunk)

        # Flush remaining audio
        final_chunks = processor.flush()
    """

    def __init__(
        self,
        sample_rate: int = 16000,
        chunk_duration_ms: int = 100,
        stride_overlap_ms: int = 20,
        target_sample_rate: int = 16000,
    ):
        """
        Initialize audio processor.

        Args:
            sample_rate: Input audio sample rate in Hz (default: 16000)
            chunk_duration_ms: Chunk size in milliseconds (default: 100)
            stride_overlap_ms: Overlap between chunks in milliseconds (default: 20)
            target_sample_rate: Output sample rate for Voxist (default: 16000)

        Raises:
            ValueError: If parameters are invalid
        """
        if sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {sample_rate}")

        if chunk_duration_ms < 50 or chunk_duration_ms > 500:
            raise ValueError(
                f"chunk_duration_ms must be 50-500ms, got {chunk_duration_ms}"
            )

        if stride_overlap_ms < 0 or stride_overlap_ms >= chunk_duration_ms:
            raise ValueError(
                f"stride_overlap_ms must be 0 to {chunk_duration_ms - 1}ms, "
                f"got {stride_overlap_ms}"
            )

        self.sample_rate = sample_rate
        self.target_sample_rate = target_sample_rate
        self.chunk_duration_ms = chunk_duration_ms
        self.stride_overlap_ms = stride_overlap_ms

        # Calculate resampling ratio (e.g., 48000 -> 16000 = ratio of 3)
        self.resample_ratio = sample_rate / target_sample_rate
        self.needs_resampling = abs(self.resample_ratio - 1.0) > 0.01

        if self.needs_resampling:
            logger.debug(
                f"Resampling enabled: {sample_rate}Hz -> {target_sample_rate}Hz "
                f"(ratio={self.resample_ratio:.2f})"
            )

        # Calculate samples at INPUT rate (for buffering)
        self.chunk_samples = int(sample_rate * chunk_duration_ms / 1000)
        self.stride_samples = int(sample_rate * stride_overlap_ms / 1000)

        # Effective advance per chunk (chunk_size - overlap)
        self.advance_samples = self.chunk_samples - self.stride_samples

        # Pre-allocated ring buffer for optimal performance (PERF-HIGH: asr-all-tog)
        # Size: 2 seconds of audio at given sample rate (provides headroom for chunking)
        self._ring_buffer_size = sample_rate * 2
        self._ring_buffer = np.zeros(self._ring_buffer_size, dtype=np.int16)
        self._write_pos = 0  # Next position to write to
        self._read_pos = 0   # Position to read from for chunks

        # Legacy buffer property for backward compatibility with tests
        # This is a view into the ring buffer data
        self._update_legacy_buffer()

        logger.debug(
            f"AudioProcessor initialized: "
            f"chunk={self.chunk_samples} samples ({chunk_duration_ms}ms), "
            f"stride={self.stride_samples} samples ({stride_overlap_ms}ms), "
            f"advance={self.advance_samples} samples, "
            f"ring_buffer={self._ring_buffer_size} samples"
        )

    def _update_legacy_buffer(self) -> None:
        """Update legacy buffer view for backward compatibility."""
        available = self._available_samples()
        if available > 0:
            self.buffer = self._get_ring_data(available)
        else:
            self.buffer = np.array([], dtype=np.int16)

    def _available_samples(self) -> int:
        """Return number of samples available for reading."""
        return self._write_pos - self._read_pos

    def _get_ring_data(self, num_samples: int) -> np.ndarray:
        """Get data from ring buffer without consuming it."""
        if num_samples <= 0:
            return np.array([], dtype=np.int16)

        start = self._read_pos % self._ring_buffer_size
        end = (self._read_pos + num_samples) % self._ring_buffer_size

        if start < end:
            return self._ring_buffer[start:end].copy()
        else:
            # Wrap-around case
            return np.concatenate([
                self._ring_buffer[start:],
                self._ring_buffer[:end]
            ])

    def resample(self, audio: np.ndarray) -> np.ndarray:
        """
        Resample audio from input sample rate to target sample rate.

        Uses simple decimation for integer ratios (e.g., 48kHz -> 16kHz).
        For non-integer ratios, uses linear interpolation.

        Args:
            audio: Int16 audio samples at input sample rate

        Returns:
            Int16 audio samples at target sample rate
        """
        if not self.needs_resampling:
            return audio

        input_len = len(audio)
        output_len = int(input_len / self.resample_ratio)

        if output_len == 0:
            return np.array([], dtype=np.int16)

        # Simple decimation for clean integer ratios (like 3:1)
        ratio_int = int(round(self.resample_ratio))
        if abs(self.resample_ratio - ratio_int) < 0.01:
            # Use simple decimation - take every Nth sample
            resampled = audio[::ratio_int]
        else:
            # Use linear interpolation for non-integer ratios
            indices = np.linspace(0, input_len - 1, output_len)
            resampled = np.interp(indices, np.arange(input_len), audio.astype(np.float32))
            resampled = resampled.astype(np.int16)

        return resampled

    def convert_to_float32(self, int16_data: np.ndarray) -> np.ndarray:
        """
        Convert Int16 PCM audio to Float32 in range [-1.0, 1.0].

        Uses NumPy vectorization for optimal performance (~10x faster than loops).
        The conversion divides by 32768.0 to map Int16 range [-32768, 32767]
        to Float32 range [-1.0, ~1.0].

        Args:
            int16_data: NumPy array of Int16 samples

        Returns:
            NumPy array of Float32 samples in range [-1.0, 1.0]

        Performance:
            ~1-2ms for 1600 samples (100ms at 16kHz)
        """
        # NumPy handles this with SIMD operations - extremely fast
        return int16_data.astype(np.float32) / 32768.0

    def _validate_frame(self, frame_data: bytes) -> bool:
        """
        Validate incoming frame data for size and alignment.

        SEC-004: Comprehensive frame validation (asr-all-6og)

        Args:
            frame_data: Raw audio bytes to validate

        Returns:
            True if frame is valid

        Raises:
            ValueError: If frame_data exceeds MAX_FRAME_SIZE_BYTES
        """
        # Validate maximum frame size to prevent resource exhaustion
        if len(frame_data) > MAX_FRAME_SIZE_BYTES:
            raise ValueError(
                f"Frame size ({len(frame_data)} bytes) exceeds maximum "
                f"allowed ({MAX_FRAME_SIZE_BYTES} bytes)"
            )

        # Validate minimum frame size to prevent processing overhead from tiny frames
        if len(frame_data) < MIN_FRAME_SIZE_BYTES:
            logger.warning(
                f"Frame too small ({len(frame_data)} bytes), minimum is "
                f"{MIN_FRAME_SIZE_BYTES} bytes. Skipping frame."
            )
            return False

        # Validate Int16 alignment (2 bytes per sample)
        if len(frame_data) % 2 != 0:
            logger.error(
                f"Invalid frame size: {len(frame_data)} bytes is not Int16 aligned "
                "(must be even number of bytes). Skipping frame."
            )
            return False

        return True

    def _add_to_buffer(self, int16_audio: np.ndarray) -> None:
        """
        Add audio samples to ring buffer with overflow handling.

        Handles buffer capacity management including:
        - Frames larger than buffer capacity
        - Buffer full conditions
        - Security limits enforcement

        Args:
            int16_audio: Int16 audio samples to add
        """
        num_samples = len(int16_audio)

        # Handle case where incoming data exceeds physical buffer capacity
        if num_samples > self._ring_buffer_size:
            samples_to_skip = num_samples - self._ring_buffer_size
            int16_audio = int16_audio[samples_to_skip:]
            num_samples = len(int16_audio)
            self._read_pos = 0
            self._write_pos = 0
            logger.warning(
                f"Frame ({num_samples + samples_to_skip} samples) exceeds buffer capacity "
                f"({self._ring_buffer_size}), keeping most recent {num_samples} samples"
            )

        # Check if we need to trim oldest samples to make room
        available_space = self._ring_buffer_size - (self._write_pos - self._read_pos)
        if num_samples > available_space:
            samples_to_trim = num_samples - available_space
            self._read_pos += samples_to_trim
            logger.warning(
                f"Ring buffer full, trimming {samples_to_trim} oldest samples"
            )

        # Security: Check total buffer size limit
        total_after_write = self._available_samples() + num_samples
        if total_after_write > MAX_BUFFER_SAMPLES:
            trim_amount = total_after_write - MAX_BUFFER_SAMPLES
            self._read_pos += trim_amount
            logger.warning(
                f"Buffer exceeded max ({total_after_write} > {MAX_BUFFER_SAMPLES}), "
                f"trimming {trim_amount} oldest samples"
            )

        # Write samples to ring buffer (handles wrap-around)
        self._write_to_ring_buffer(int16_audio)

    def _write_to_ring_buffer(self, int16_audio: np.ndarray) -> None:
        """
        Write audio samples to ring buffer handling wrap-around.

        Args:
            int16_audio: Int16 audio samples to write
        """
        num_samples = len(int16_audio)
        write_start = self._write_pos % self._ring_buffer_size
        write_end = (self._write_pos + num_samples) % self._ring_buffer_size

        if write_start < write_end or write_end == 0:
            # No wrap-around (or exactly filling to end)
            actual_end = write_end if write_end > 0 else self._ring_buffer_size
            self._ring_buffer[write_start:actual_end] = int16_audio
        else:
            # Wrap-around case - split the write
            first_part_size = self._ring_buffer_size - write_start
            self._ring_buffer[write_start:] = int16_audio[:first_part_size]
            self._ring_buffer[:write_end] = int16_audio[first_part_size:]

        self._write_pos += num_samples

    def _extract_available_chunks(self) -> list[np.ndarray]:
        """
        Extract all available complete chunks from buffer with stride overlap.

        Returns:
            List of Int16 NumPy arrays ready for transmission
        """
        chunks = []

        while self._available_samples() >= self.chunk_samples:
            # Extract chunk from ring buffer (Int16 format)
            chunk = self._get_ring_data(self.chunk_samples)

            # Log audio levels for debugging
            if len(chunk) > 0:
                logger.debug(
                    f"Audio chunk stats: min={chunk.min()}, max={chunk.max()}, "
                    f"rms={np.sqrt(np.mean(chunk.astype(np.float32)**2)):.1f}"
                )

            # Resample to target rate if needed (e.g., 48kHz -> 16kHz)
            output_chunk = self.resample(chunk)
            chunks.append(output_chunk)

            # Advance read position by stride (creates overlap for next chunk)
            self._read_pos += self.advance_samples

        return chunks

    def process_audio_frame(self, frame_data: bytes) -> list[np.ndarray]:
        """
        Process incoming audio frame into fixed-size Int16 chunks with overlap.

        Maintains internal buffer for partial frames and applies stride overlap
        for improved accuracy at chunk boundaries.

        Args:
            frame_data: Raw audio bytes (Int16 PCM format)

        Returns:
            List of Int16 NumPy arrays ready for WebSocket transmission.
            Voxist backend expects raw Int16 PCM bytes.
            May return empty list if insufficient data for a complete chunk.

        Raises:
            ValueError: If frame_data exceeds MAX_FRAME_SIZE_BYTES

        Example:
            # Process 200ms of audio (3200 samples)
            chunks = processor.process_audio_frame(audio_bytes)
            # Returns 2 chunks with 20ms overlap:
            #   chunk[0]: samples 0-1599
            #   chunk[1]: samples 1280-2879 (320 sample overlap)
        """
        # Validate frame data (SEC-004)
        if not self._validate_frame(frame_data):
            return []

        # Convert bytes to Int16 array
        try:
            int16_audio = np.frombuffer(frame_data, dtype=np.int16)
        except ValueError as e:
            logger.error(f"Failed to parse audio frame: {e}")
            return []

        if len(int16_audio) == 0:
            return []

        # Track original sample count for logging
        num_samples = len(int16_audio)

        # Add samples to ring buffer with overflow handling
        self._add_to_buffer(int16_audio)

        # Extract all available chunks
        chunks = self._extract_available_chunks()

        # Update legacy buffer for backward compatibility
        self._update_legacy_buffer()

        if chunks:
            logger.debug(
                f"Processed {num_samples} samples → "
                f"{len(chunks)} chunks ({self._available_samples()} samples buffered)"
            )

        return chunks

    def flush(self) -> list[np.ndarray]:
        """
        Flush remaining buffered audio as final chunk.

        Call this at the end of audio stream to process any remaining
        samples that didn't complete a full chunk.

        Returns:
            List with single Int16 chunk if buffer has data, empty list otherwise.
            Voxist backend expects raw Int16 PCM bytes.

        Example:
            # End of stream
            final_chunks = processor.flush()
            for chunk in final_chunks:
                await send_to_websocket(chunk)
        """
        available = self._available_samples()
        if available > 0:
            logger.debug(f"Flushing {available} remaining samples")

            # Get remaining data from ring buffer (Int16 format)
            remaining = self._get_ring_data(available)

            # Resample to target rate if needed
            output = self.resample(remaining)

            # Reset positions
            self._read_pos = self._write_pos

            # Update legacy buffer
            self._update_legacy_buffer()

            # Return Int16 directly - Voxist expects raw Int16 PCM bytes
            return [output]

        return []

    def reset(self) -> None:
        """
        Reset internal buffer.

        Useful for reusing the processor for a new audio stream.
        """
        self._write_pos = 0
        self._read_pos = 0
        self._update_legacy_buffer()
        logger.debug("AudioProcessor buffer reset")

    @property
    def buffered_samples(self) -> int:
        """Get number of samples currently in buffer."""
        return self._available_samples()

    @property
    def buffered_duration_ms(self) -> float:
        """Get duration of buffered audio in milliseconds."""
        return (self._available_samples() / self.sample_rate) * 1000
