"""
Real-time audio engine using sounddevice.

Provides low-latency audio I/O with model inference in the callback.
Thread-safe knob value passing via a lock.
"""

import logging
import threading
from typing import List, Optional, Callable

import numpy as np
import sounddevice as sd
import torch

logger = logging.getLogger(__name__)


class AudioEngine:
    """
    Real-time audio stream that processes input through a NAM model.

    Usage:
        engine = AudioEngine(model, sample_rate=48000, block_size=512)
        engine.set_knob(0, 0.5)
        engine.start()
        # ... later ...
        engine.stop()
    """

    def __init__(
        self,
        model: object,  # MultiKnobModel
        sample_rate: int = 48000,
        block_size: int = 512,
        num_knobs: int = 2,
    ):
        self._model = model
        self._sample_rate = sample_rate
        self._block_size = block_size
        self._num_knobs = num_knobs

        # Thread-safe knob values
        self._knob_values: List[float] = [0.5] * num_knobs
        self._lock = threading.Lock()

        self._stream = None
        self._running = False

        # Optional callback for audio errors
        self._on_error: Optional[Callable] = None

    # ------------------------------------------------------------------
    # Knob values (thread-safe)
    # ------------------------------------------------------------------

    def set_knob(self, idx: int, value: float) -> None:
        """Set a knob value (0.0 to 1.0 normalized range)."""
        if 0 <= idx < self._num_knobs:
            with self._lock:
                self._knob_values[idx] = float(value)

    def get_knob_values(self, count: int) -> List[float]:
        """Get current knob values. Thread-safe."""
        with self._lock:
            return list(self._knob_values[:count])

    # ------------------------------------------------------------------
    # Stream control
    # ------------------------------------------------------------------

    def start(
        self,
        input_device: Optional[int] = None,
        output_device: Optional[int] = None,
    ) -> None:
        """Start the audio stream."""
        if self._running:
            logger.warning("Audio engine already running")
            return

        try:
            self._stream = sd.Stream(
                samplerate=self._sample_rate,
                blocksize=self._block_size,
                dtype="float32",
                channels=1,
                callback=self._audio_callback,
                device=(input_device, output_device),
            )
            self._stream.start()
            self._running = True
            logger.info(
                f"Audio engine started: sr={self._sample_rate}, block={self._block_size}"
            )
        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            raise

    def stop(self) -> None:
        """Stop and close the audio stream."""
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None
        self._running = False
        logger.info("Audio engine stopped")

    @property
    def is_running(self) -> bool:
        return self._running

    # ------------------------------------------------------------------
    # Internal
    # ------------------------------------------------------------------

    def _audio_callback(
        self,
        indata: np.ndarray,
        outdata: np.ndarray,
        frames: int,
        time_info: object,
        status: sd.CallbackFlags,
    ) -> None:
        """sounddevice callback — called from audio thread."""
        if status and self._on_error:
            self._on_error(str(status))

        try:
            knob_values = self.get_knob_values(self._num_knobs)

            # Prepare input tensor
            audio = torch.from_numpy(indata[:, 0].copy()).float()

            # Run model inference
            with torch.no_grad():
                knob_tensors = [torch.tensor(v) for v in knob_values]
                output = self._model(audio.unsqueeze(0), *knob_tensors)

            # Write output
            result = output.squeeze(0).numpy()
            outdata[:, 0] = result[:frames]
            if outdata.shape[1] > 1:
                outdata[:, 1] = result[:frames]
        except Exception as e:
            logger.error(f"Audio callback error: {e}")
            outdata.fill(0.0)

