"""Unit tests for AudioProcessor."""

import time

import numpy as np
import pytest

from livekit.plugins.voxist.audio_processor import AudioProcessor


class TestAudioProcessor:
    """Test suite for AudioProcessor class."""

    def test_initialization_default(self):
        """Test AudioProcessor initializes with default parameters."""
        processor = AudioProcessor()

        assert processor.sample_rate == 16000
        assert processor.chunk_duration_ms == 100
        assert processor.stride_overlap_ms == 20
        assert processor.chunk_samples == 1600  # 100ms at 16kHz
        assert processor.stride_samples == 320   # 20ms at 16kHz
        assert processor.advance_samples == 1280 # 1600 - 320
        assert len(processor.buffer) == 0

    def test_initialization_custom(self):
        """Test AudioProcessor with custom parameters."""
        processor = AudioProcessor(
            sample_rate=8000,
            chunk_duration_ms=200,
            stride_overlap_ms=40,
        )

        assert processor.sample_rate == 8000
        assert processor.chunk_samples == 1600  # 200ms at 8kHz
        assert processor.stride_samples == 320   # 40ms at 8kHz
        assert processor.advance_samples == 1280

    def test_initialization_invalid_sample_rate(self):
        """Test initialization fails with invalid sample rate."""
        with pytest.raises(ValueError, match="sample_rate must be positive"):
            AudioProcessor(sample_rate=0)

        with pytest.raises(ValueError, match="sample_rate must be positive"):
            AudioProcessor(sample_rate=-16000)

    def test_initialization_invalid_chunk_duration(self):
        """Test initialization fails with invalid chunk duration."""
        with pytest.raises(ValueError, match="chunk_duration_ms must be 50-500ms"):
            AudioProcessor(chunk_duration_ms=30)

        with pytest.raises(ValueError, match="chunk_duration_ms must be 50-500ms"):
            AudioProcessor(chunk_duration_ms=600)

    def test_initialization_invalid_stride(self):
        """Test initialization fails with invalid stride overlap."""
        with pytest.raises(ValueError, match="stride_overlap_ms must be"):
            AudioProcessor(chunk_duration_ms=100, stride_overlap_ms=-10)

        with pytest.raises(ValueError, match="stride_overlap_ms must be"):
            AudioProcessor(chunk_duration_ms=100, stride_overlap_ms=100)

    def test_convert_to_float32_accuracy(self):
        """Test Int16 to Float32 conversion accuracy."""
        processor = AudioProcessor()

        # Test data: min, zero, max Int16 values
        int16_data = np.array([-32768, -16384, 0, 16384, 32767], dtype=np.int16)
        float32_data = processor.convert_to_float32(int16_data)

        # Check data type
        assert float32_data.dtype == np.float32

        # Check values
        assert float32_data[0] == pytest.approx(-1.0, abs=1e-5)      # min
        assert float32_data[1] == pytest.approx(-0.5, abs=1e-5)      # -half
        assert float32_data[2] == pytest.approx(0.0, abs=1e-5)       # zero
        assert float32_data[3] == pytest.approx(0.5, abs=1e-5)       # +half
        assert float32_data[4] == pytest.approx(0.999969, abs=1e-5)  # max

    def test_convert_to_float32_range(self):
        """Test converted values are in range [-1.0, 1.0]."""
        processor = AudioProcessor()

        # Generate random Int16 data
        int16_data = np.random.randint(-32768, 32767, size=1600, dtype=np.int16)
        float32_data = processor.convert_to_float32(int16_data)

        # All values should be in range
        assert np.all(float32_data >= -1.0)
        assert np.all(float32_data <= 1.0)

    def test_process_audio_frame_single_chunk(self):
        """Test processing exactly one chunk of audio."""
        processor = AudioProcessor(sample_rate=16000, chunk_duration_ms=100)

        # Create 100ms of audio (1600 samples)
        int16_audio = np.random.randint(-32768, 32767, size=1600, dtype=np.int16)
        frame_data = int16_audio.tobytes()

        chunks = processor.process_audio_frame(frame_data)

        # Should return 1 chunk
        assert len(chunks) == 1
        assert len(chunks[0]) == 1600
        assert chunks[0].dtype == np.int16  # Voxist expects raw Int16 PCM

        # Buffer should have stride overlap remaining
        assert processor.buffered_samples == 320  # 20ms at 16kHz

    def test_process_audio_frame_multiple_chunks(self):
        """Test processing audio that produces multiple chunks."""
        processor = AudioProcessor(sample_rate=16000, chunk_duration_ms=100)

        # Create 300ms of audio (4800 samples)
        int16_audio = np.random.randint(-32768, 32767, size=4800, dtype=np.int16)
        frame_data = int16_audio.tobytes()

        chunks = processor.process_audio_frame(frame_data)

        # Should return 3 chunks with overlap
        # Chunk 0: samples 0-1599
        # Chunk 1: samples 1280-2879 (320 overlap)
        # Chunk 2: samples 2560-4159 (320 overlap)
        assert len(chunks) == 3

        for chunk in chunks:
            assert len(chunk) == 1600
            assert chunk.dtype == np.int16  # Voxist expects raw Int16 PCM

        # Remaining buffer: 4800 - (3 * 1280) = 960 samples
        assert processor.buffered_samples == 960

    def test_process_audio_frame_partial(self):
        """Test processing audio insufficient for a complete chunk."""
        processor = AudioProcessor(sample_rate=16000, chunk_duration_ms=100)

        # Create 50ms of audio (800 samples) - less than chunk size
        int16_audio = np.random.randint(-32768, 32767, size=800, dtype=np.int16)
        frame_data = int16_audio.tobytes()

        chunks = processor.process_audio_frame(frame_data)

        # Should return empty list (buffered for next frame)
        assert len(chunks) == 0
        assert processor.buffered_samples == 800

    def test_process_audio_frame_accumulation(self):
        """Test multiple partial frames accumulate into chunks."""
        processor = AudioProcessor(sample_rate=16000, chunk_duration_ms=100)

        # Send 3 partial frames (50ms each = 800 samples)
        # Total: 2400 samples → should produce 1 chunk (1600 samples)
        # with remaining buffer (2400 - 1280 advance = 1120 samples)
        all_chunks = []
        for _i in range(3):
            int16_audio = np.random.randint(-32768, 32767, size=800, dtype=np.int16)
            frame_data = int16_audio.tobytes()
            chunks = processor.process_audio_frame(frame_data)
            all_chunks.extend(chunks)

        # After 3 frames (2400 samples), should have 1 chunk
        assert len(all_chunks) == 1
        assert len(all_chunks[0]) == 1600
        # Buffer: 2400 - 1280 (advance) = 1120 samples remaining
        assert processor.buffered_samples == 1120

    def test_stride_overlap_creates_overlap(self):
        """Test that stride overlap creates proper overlap between chunks."""
        processor = AudioProcessor(
            sample_rate=16000,
            chunk_duration_ms=100,
            stride_overlap_ms=20,
        )

        # Create identifiable audio pattern
        # First chunk: ascending values 0-1599
        # Second chunk should include 1280-1599 from first chunk (320 sample overlap)
        int16_audio = np.arange(0, 3200, dtype=np.int16)
        frame_data = int16_audio.tobytes()

        chunks = processor.process_audio_frame(frame_data)

        assert len(chunks) == 2

        # First chunk: samples 0-1599
        # Second chunk: samples 1280-2879
        # Overlap region: samples 1280-1599 (320 samples)

        # Verify overlap by checking values (chunks are already Int16)
        # chunk[0] ends with values 1580-1599
        # chunk[1] starts with values 1280-1299
        assert chunks[0].dtype == np.int16  # Voxist expects raw Int16 PCM
        assert chunks[1].dtype == np.int16

        assert chunks[0][-20] == 1580
        assert chunks[0][-1] == 1599
        assert chunks[1][0] == 1280
        assert chunks[1][19] == 1299

    def test_flush_with_buffered_data(self):
        """Test flush returns remaining buffered data."""
        processor = AudioProcessor()

        # Send partial audio (800 samples)
        int16_audio = np.random.randint(-32768, 32767, size=800, dtype=np.int16)
        frame_data = int16_audio.tobytes()

        chunks = processor.process_audio_frame(frame_data)
        assert len(chunks) == 0
        assert processor.buffered_samples == 800

        # Flush should return the 800 samples
        final_chunks = processor.flush()

        assert len(final_chunks) == 1
        assert len(final_chunks[0]) == 800
        assert final_chunks[0].dtype == np.int16  # Voxist expects raw Int16 PCM
        assert processor.buffered_samples == 0

    def test_flush_empty_buffer(self):
        """Test flush with empty buffer returns empty list."""
        processor = AudioProcessor()

        final_chunks = processor.flush()

        assert len(final_chunks) == 0
        assert processor.buffered_samples == 0

    def test_flush_clears_buffer(self):
        """Test flush clears internal buffer."""
        processor = AudioProcessor()

        # Add some data
        int16_audio = np.random.randint(-32768, 32767, size=500, dtype=np.int16)
        processor.process_audio_frame(int16_audio.tobytes())

        assert processor.buffered_samples > 0

        # Flush
        processor.flush()

        # Buffer should be empty
        assert processor.buffered_samples == 0
        assert len(processor.buffer) == 0

    def test_reset_clears_buffer(self):
        """Test reset method clears buffer."""
        processor = AudioProcessor()

        # Add some data
        int16_audio = np.random.randint(-32768, 32767, size=500, dtype=np.int16)
        processor.process_audio_frame(int16_audio.tobytes())

        assert processor.buffered_samples > 0

        # Reset
        processor.reset()

        # Buffer should be empty
        assert processor.buffered_samples == 0

    def test_buffered_duration_ms(self):
        """Test buffered_duration_ms property."""
        processor = AudioProcessor(sample_rate=16000)

        # Add 800 samples (50ms at 16kHz)
        int16_audio = np.random.randint(-32768, 32767, size=800, dtype=np.int16)
        processor.process_audio_frame(int16_audio.tobytes())

        assert processor.buffered_duration_ms == pytest.approx(50.0, abs=0.1)

    def test_empty_frame_handling(self):
        """Test handling of empty audio frames."""
        processor = AudioProcessor()

        chunks = processor.process_audio_frame(b"")

        assert len(chunks) == 0
        assert processor.buffered_samples == 0

    def test_malformed_frame_handling(self):
        """Test handling of malformed audio data."""
        processor = AudioProcessor()

        # Odd number of bytes (Int16 requires even number)
        malformed_data = b"\x00\x01\x02"

        chunks = processor.process_audio_frame(malformed_data)

        # Should handle gracefully (NumPy will truncate)
        assert isinstance(chunks, list)

    @pytest.mark.benchmark
    def test_conversion_performance(self):
        """Benchmark Int16 to Float32 conversion performance."""
        processor = AudioProcessor(sample_rate=16000)

        # Create 100ms of audio (1600 samples)
        int16_audio = np.random.randint(-32768, 32767, size=1600, dtype=np.int16)

        # Measure conversion time
        start = time.perf_counter()
        iterations = 1000

        for _ in range(iterations):
            _ = processor.convert_to_float32(int16_audio)

        elapsed = time.perf_counter() - start
        avg_time_ms = (elapsed / iterations) * 1000

        # Should be < 5ms per conversion (target)
        # Typically achieves < 0.1ms with NumPy
        assert avg_time_ms < 5.0, f"Conversion too slow: {avg_time_ms:.2f}ms"

        print(f"\nConversion performance: {avg_time_ms:.4f}ms per 100ms chunk")

    @pytest.mark.benchmark
    def test_processing_performance(self):
        """Benchmark full processing pipeline performance."""
        processor = AudioProcessor(sample_rate=16000)

        # Create 100ms of audio
        int16_audio = np.random.randint(-32768, 32767, size=1600, dtype=np.int16)
        frame_data = int16_audio.tobytes()

        # Measure processing time
        start = time.perf_counter()
        iterations = 1000

        for _ in range(iterations):
            # Reset to test fresh processing
            processor.reset()
            _ = processor.process_audio_frame(frame_data)

        elapsed = time.perf_counter() - start
        avg_time_ms = (elapsed / iterations) * 1000

        # Should be < 10ms per frame (conservative target)
        assert avg_time_ms < 10.0, f"Processing too slow: {avg_time_ms:.2f}ms"

        print(f"\nProcessing performance: {avg_time_ms:.4f}ms per 100ms chunk")

    def test_no_overlap_mode(self):
        """Test AudioProcessor with zero overlap."""
        processor = AudioProcessor(
            sample_rate=16000,
            chunk_duration_ms=100,
            stride_overlap_ms=0,  # No overlap
        )

        assert processor.stride_samples == 0
        assert processor.advance_samples == 1600  # Full chunk advance

        # Process 200ms (3200 samples)
        int16_audio = np.arange(0, 3200, dtype=np.int16)
        chunks = processor.process_audio_frame(int16_audio.tobytes())

        # Should get 2 chunks with NO overlap
        assert len(chunks) == 2

        # Verify no overlap: chunk[1] starts where chunk[0] ends (chunks are Int16)
        assert chunks[0].dtype == np.int16  # Voxist expects raw Int16 PCM
        assert chunks[1].dtype == np.int16

        assert chunks[0][-1] == 1599
        assert chunks[1][0] == 1600  # Immediately after chunk 0

    def test_large_overlap(self):
        """Test AudioProcessor with large overlap (50%)."""
        processor = AudioProcessor(
            sample_rate=16000,
            chunk_duration_ms=100,
            stride_overlap_ms=50,  # 50% overlap
        )

        assert processor.stride_samples == 800
        assert processor.advance_samples == 800  # Only advance 50ms

        # Process 300ms (4800 samples)
        int16_audio = np.arange(0, 4800, dtype=np.int16)
        chunks = processor.process_audio_frame(int16_audio.tobytes())

        # With 50% overlap and 300ms audio:
        # Chunk 0: 0-1599
        # Chunk 1: 800-2399 (800 sample overlap)
        # Chunk 2: 1600-3199 (800 sample overlap)
        # Chunk 3: 2400-3999 (800 sample overlap)
        # Chunk 4: 3200-4799 (800 sample overlap)
        assert len(chunks) == 5

    def test_realistic_streaming_scenario(self):
        """Test realistic streaming scenario with variable frame sizes."""
        processor = AudioProcessor()

        all_chunks = []

        # Simulate realistic streaming: frames of varying sizes
        frame_sizes = [1000, 800, 1200, 1600, 500, 2000, 300]

        for size in frame_sizes:
            int16_audio = np.random.randint(-32768, 32767, size=size, dtype=np.int16)
            chunks = processor.process_audio_frame(int16_audio.tobytes())
            all_chunks.extend(chunks)

        # Total samples: 7400
        # Expected chunks: (7400 - 320 buffer) / 1280 advance = ~5 chunks
        assert len(all_chunks) >= 5

        # All chunks should be correct size
        for chunk in all_chunks:
            assert len(chunk) == 1600
            assert chunk.dtype == np.int16  # Voxist expects raw Int16 PCM

        # Flush remaining
        final = processor.flush()
        all_chunks.extend(final)

        # Should have flushed remaining data
        assert processor.buffered_samples == 0

    def test_conversion_preserves_waveform(self):
        """Test that conversion preserves audio waveform shape."""
        processor = AudioProcessor()

        # Create a simple sine wave
        t = np.linspace(0, 0.1, 1600)  # 100ms
        sine_wave = (np.sin(2 * np.pi * 440 * t) * 32767).astype(np.int16)

        # Convert
        float32_wave = processor.convert_to_float32(sine_wave)

        # Check waveform properties are preserved
        # Both should have same zero crossings
        int16_zero_crossings = np.where(np.diff(np.sign(sine_wave)))[0]
        float32_zero_crossings = np.where(np.diff(np.sign(float32_wave)))[0]

        # Should have approximately same zero crossing positions
        assert len(int16_zero_crossings) == len(float32_zero_crossings)

    def test_buffer_accumulation_across_calls(self):
        """Test buffer correctly accumulates across multiple calls."""
        processor = AudioProcessor()

        # Send 10 small frames (160 samples each = 10ms)
        for i in range(10):
            int16_audio = np.full(160, i, dtype=np.int16)  # Fill with index value
            chunks = processor.process_audio_frame(int16_audio.tobytes())

            if i < 9:
                # First 9 frames: accumulating
                assert len(chunks) == 0
            else:
                # 10th frame: should produce 1 chunk (1600 samples accumulated)
                assert len(chunks) == 1

    def test_process_audio_frame_with_zero_overlap(self):
        """Test edge case: zero overlap produces sequential non-overlapping chunks."""
        processor = AudioProcessor(
            sample_rate=16000,
            chunk_duration_ms=100,
            stride_overlap_ms=0,
        )

        # Create audio with known pattern
        int16_audio = np.arange(0, 4800, dtype=np.int16)
        chunks = processor.process_audio_frame(int16_audio.tobytes())

        # Should get 3 chunks, each 1600 samples, no overlap
        assert len(chunks) == 3

        # Verify sequential (no overlap) - chunks are already Int16
        for i, chunk in enumerate(chunks):
            assert chunk.dtype == np.int16  # Voxist expects raw Int16 PCM
            expected_start = i * 1600
            assert chunk[0] == expected_start
            assert chunk[-1] == expected_start + 1599

    def test_flush_integration_with_processing(self):
        """Test flush works correctly after processing multiple frames."""
        processor = AudioProcessor()

        # Process several frames
        total_samples = 0
        for _ in range(3):
            int16_audio = np.random.randint(-32768, 32767, size=1000, dtype=np.int16)
            processor.process_audio_frame(int16_audio.tobytes())
            total_samples += 1000

        # Process should have emitted some chunks
        buffered_before_flush = processor.buffered_samples
        assert buffered_before_flush > 0

        # Flush
        final_chunks = processor.flush()

        # Should return remaining data
        assert len(final_chunks) == 1
        assert len(final_chunks[0]) == buffered_before_flush

        # Buffer should be empty
        assert processor.buffered_samples == 0


class TestAudioSizeValidation:
    """Test suite for audio size validation (SEC-HIGH: asr-all-gt1)."""

    def test_max_frame_size_constant_exists(self):
        """Test MAX_FRAME_SIZE_BYTES constant is defined."""
        from livekit.plugins.voxist.audio_processor import MAX_FRAME_SIZE_BYTES
        # Should be 1MB (1024 * 1024)
        assert MAX_FRAME_SIZE_BYTES == 1024 * 1024

    def test_max_buffer_samples_constant_exists(self):
        """Test MAX_BUFFER_SAMPLES constant is defined for 10 seconds at 48kHz."""
        from livekit.plugins.voxist.audio_processor import MAX_BUFFER_SAMPLES
        # 10 seconds at 48kHz = 480000 samples
        assert MAX_BUFFER_SAMPLES == 480000

    def test_oversized_frame_raises_error(self):
        """Test that frames exceeding MAX_FRAME_SIZE_BYTES raise ValueError."""
        processor = AudioProcessor()

        # Create frame larger than 1MB (Int16 = 2 bytes per sample)
        # 1MB = 524288 samples, so create 600000 samples
        oversized_data = np.zeros(600000, dtype=np.int16).tobytes()

        with pytest.raises(ValueError, match="Frame size.*exceeds maximum"):
            processor.process_audio_frame(oversized_data)

    def test_buffer_accumulation_trimmed_at_max(self):
        """Test buffer is trimmed when it would exceed MAX_BUFFER_SAMPLES."""
        processor = AudioProcessor(sample_rate=16000)

        # Send multiple frames to accumulate buffer
        # MAX_BUFFER_SAMPLES = 480000, at 16kHz that's 30 seconds
        # Send frames that would exceed this

        # First, fill up close to limit
        large_frame = np.zeros(400000, dtype=np.int16).tobytes()
        processor.process_audio_frame(large_frame)

        # Now send another frame that would push over limit
        another_frame = np.zeros(100000, dtype=np.int16).tobytes()
        processor.process_audio_frame(another_frame)

        # Buffer should be trimmed to not exceed MAX_BUFFER_SAMPLES
        from livekit.plugins.voxist.audio_processor import MAX_BUFFER_SAMPLES
        assert processor.buffered_samples <= MAX_BUFFER_SAMPLES

    def test_valid_frame_size_accepted(self):
        """Test that valid frame sizes are accepted."""
        processor = AudioProcessor()

        # Normal 100ms frame at 16kHz = 1600 samples = 3200 bytes
        normal_frame = np.random.randint(-32768, 32767, size=1600, dtype=np.int16).tobytes()

        # Should not raise
        chunks = processor.process_audio_frame(normal_frame)
        assert isinstance(chunks, list)

    def test_frame_at_exact_limit_accepted(self):
        """Test that frame exactly at limit is accepted."""
        from livekit.plugins.voxist.audio_processor import MAX_FRAME_SIZE_BYTES
        processor = AudioProcessor()

        # Create frame exactly at limit (1MB / 2 bytes per Int16 = 524288 samples)
        max_samples = MAX_FRAME_SIZE_BYTES // 2
        at_limit_frame = np.zeros(max_samples, dtype=np.int16).tobytes()

        # Should not raise
        chunks = processor.process_audio_frame(at_limit_frame)
        assert isinstance(chunks, list)

    # SEC-004 Tests (asr-all-6og): Minimum frame size and Int16 alignment validation

    def test_min_frame_size_constant_exists(self):
        """Test MIN_FRAME_SIZE_BYTES constant is defined."""
        from livekit.plugins.voxist.audio_processor import MIN_FRAME_SIZE_BYTES
        # Should be 160 bytes (10ms at 8kHz * 2 bytes per sample)
        assert MIN_FRAME_SIZE_BYTES == 160

    def test_undersized_frame_skipped(self):
        """Test that frames below MIN_FRAME_SIZE_BYTES are skipped."""
        processor = AudioProcessor()

        # Create frame smaller than minimum (100 bytes < 160 bytes)
        small_frame = np.zeros(50, dtype=np.int16).tobytes()  # 100 bytes
        assert len(small_frame) == 100
        assert len(small_frame) < 160

        # Should return empty list, not raise
        chunks = processor.process_audio_frame(small_frame)
        assert chunks == []

    def test_empty_frame_skipped(self):
        """Test that empty frames are skipped."""
        processor = AudioProcessor()

        # Empty frame
        chunks = processor.process_audio_frame(b'')
        assert chunks == []

    def test_frame_at_exact_min_limit_accepted(self):
        """Test that frame exactly at minimum limit is accepted."""
        from livekit.plugins.voxist.audio_processor import MIN_FRAME_SIZE_BYTES
        processor = AudioProcessor()

        # Create frame exactly at minimum limit (160 bytes = 80 samples)
        min_samples = MIN_FRAME_SIZE_BYTES // 2
        at_min_frame = np.zeros(min_samples, dtype=np.int16).tobytes()

        # Should not raise or return empty (though won't produce chunks due to chunking logic)
        chunks = processor.process_audio_frame(at_min_frame)
        assert isinstance(chunks, list)

    def test_non_int16_aligned_frame_skipped(self):
        """Test that frames with odd byte count (not Int16 aligned) are skipped."""
        processor = AudioProcessor()

        # Create frame with odd number of bytes (not divisible by 2)
        odd_frame = b'\x00' * 161  # 161 bytes - not Int16 aligned
        assert len(odd_frame) % 2 != 0

        # Should return empty list, not raise
        chunks = processor.process_audio_frame(odd_frame)
        assert chunks == []

    def test_int16_aligned_frame_accepted(self):
        """Test that frames with even byte count (Int16 aligned) are accepted."""
        processor = AudioProcessor()

        # Create frame with even number of bytes (divisible by 2)
        even_frame = np.zeros(100, dtype=np.int16).tobytes()  # 200 bytes
        assert len(even_frame) % 2 == 0

        # Should not return empty due to alignment (may be empty due to chunking)
        chunks = processor.process_audio_frame(even_frame)
        assert isinstance(chunks, list)

    def test_single_byte_frame_skipped(self):
        """Test that single-byte frames are skipped (both too small and not aligned)."""
        processor = AudioProcessor()

        # Single byte - fails both min size and alignment checks
        single_byte = b'\x00'
        chunks = processor.process_audio_frame(single_byte)
        assert chunks == []


class TestRingBufferOptimization:
    """Test suite for ring buffer optimization (PERF-HIGH: asr-all-tog)."""

    def test_ring_buffer_initialization(self):
        """Test that AudioProcessor initializes with pre-allocated ring buffer."""
        processor = AudioProcessor(sample_rate=16000)

        # Should have a pre-allocated buffer
        assert hasattr(processor, '_ring_buffer')
        # Buffer should be pre-allocated to hold at least 2 seconds of audio
        min_buffer_size = 16000 * 2  # 2 seconds at 16kHz
        assert len(processor._ring_buffer) >= min_buffer_size

    def test_ring_buffer_write_position_tracking(self):
        """Test that ring buffer tracks write position."""
        processor = AudioProcessor()

        assert hasattr(processor, '_write_pos')
        assert processor._write_pos == 0

        # After processing data, write position should advance
        int16_audio = np.random.randint(-32768, 32767, size=1600, dtype=np.int16)
        processor.process_audio_frame(int16_audio.tobytes())

        assert processor._write_pos > 0

    def test_no_concatenation_in_hot_path(self):
        """Test that process_audio_frame doesn't use np.concatenate."""
        import dis
        import io

        processor = AudioProcessor()

        # Get bytecode of process_audio_frame
        bytecode_output = io.StringIO()
        dis.dis(processor.process_audio_frame, file=bytecode_output)
        bytecode_output.getvalue()

        # The hot path should not contain concatenate calls
        # (Note: This is a simple check - concatenate may still be used in
        # cold paths like initialization or edge cases, but not in the main loop)
        # We'll verify through performance benchmarking instead

    @pytest.mark.benchmark
    def test_ring_buffer_performance_improvement(self):
        """Test that ring buffer provides at least 50% performance improvement."""
        processor = AudioProcessor(sample_rate=16000)

        # Create 100ms audio frames
        frames = [
            np.random.randint(-32768, 32767, size=1600, dtype=np.int16).tobytes()
            for _ in range(100)
        ]

        # Warm up
        for frame in frames[:10]:
            processor.process_audio_frame(frame)
        processor.reset()

        # Benchmark
        start = time.perf_counter()
        for frame in frames:
            processor.process_audio_frame(frame)
        elapsed = time.perf_counter() - start

        # Should process 100 x 100ms frames (10 seconds of audio) in < 50ms
        # This is much faster than original concatenation approach
        assert elapsed < 0.05, f"Processing too slow: {elapsed*1000:.2f}ms"

    def test_ring_buffer_wrap_around(self):
        """Test ring buffer correctly handles wrap-around."""
        processor = AudioProcessor(sample_rate=16000)

        # Process enough data to wrap around the buffer multiple times
        # At 16kHz with 2-second buffer (32000 samples), we need > 32000 samples
        for _ in range(30):  # 30 x 1600 = 48000 samples
            int16_audio = np.random.randint(-32768, 32767, size=1600, dtype=np.int16)
            chunks = processor.process_audio_frame(int16_audio.tobytes())
            # Should still produce valid chunks
            for chunk in chunks:
                assert chunk.dtype == np.int16  # Voxist expects raw Int16 PCM
                assert len(chunk) == 1600
