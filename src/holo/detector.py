from __future__ import annotations

from collections import deque
import queue
from typing import Any
import numpy as np
import sounddevice as sd

try:
    from .config import (
        BG_EMA_ALPHA,
        BG_SNR_RATIO,
        CALIBRATION_BLOCKS,
        CFG,
        COOLDOWN_BLOCKS,
        CREST_THRESHOLD,
        LOWPASS_CUTOFF_HZ,
        PEAK_MULTIPLIER,
        POST_BLOCKS,
        PRE_BLOCKS,
        RMS_THRESHOLD,
        SAMPLE_RATE,
    )
    from .ultrasonic import apply_lowpass_filter
except (ImportError, ValueError):
    from config import (  # type: ignore
        BG_EMA_ALPHA,
        BG_SNR_RATIO,
        CALIBRATION_BLOCKS,
        CFG,
        COOLDOWN_BLOCKS,
        CREST_THRESHOLD,
        LOWPASS_CUTOFF_HZ,
        PEAK_MULTIPLIER,
        POST_BLOCKS,
        PRE_BLOCKS,
        RMS_THRESHOLD,
        SAMPLE_RATE,
    )
    from ultrasonic import apply_lowpass_filter  # type: ignore


def get_input_channels() -> int:
    """Determines available input microphone channels (1 for mono, 2 for stereo)."""
    ch_cfg = CFG.get("audio", {}).get("channels", 1)
    if ch_cfg != "auto":
        return int(ch_cfg)
    try:
        dev = sd.query_devices(kind="input")
        if isinstance(dev, dict):
            return min(2, max(1, int(dev.get("max_input_channels", 1))))
    except Exception:
        pass
    return 1


class ImpulseDetector:
    """
    Real-time acoustic transient impulse detector with pre-trigger ring buffer,
    adaptive background SNR floor tracking, and ultrasonic carrier isolation.
    """

    def __init__(
        self,
        threshold: float | None = None,
        dynamic_multiplier: float | None = None,
        sr: int = SAMPLE_RATE,
    ):
        self.sr = int(sr)
        det_cfg = CFG.get("detector", {})
        self.threshold = float(threshold if threshold is not None else det_cfg.get("rms_threshold", RMS_THRESHOLD))
        self.peak_mult = float(det_cfg.get("peak_multiplier", PEAK_MULTIPLIER))
        self.bg_snr_ratio = float(
            dynamic_multiplier if dynamic_multiplier is not None else det_cfg.get("bg_snr_ratio", BG_SNR_RATIO)
        )
        self.ema_alpha = float(det_cfg.get("bg_ema_alpha", BG_EMA_ALPHA))
        self.crest_thresh = float(det_cfg.get("crest_threshold", CREST_THRESHOLD))

        aud_cfg = CFG.get("audio", {})
        self.pre_blocks = int(aud_cfg.get("pre_blocks", PRE_BLOCKS))
        self.post_blocks = int(aud_cfg.get("post_blocks", POST_BLOCKS))
        self.cooldown_blocks = int(aud_cfg.get("cooldown_blocks", COOLDOWN_BLOCKS))

        self.pre_buffer: deque[np.ndarray] = deque(maxlen=self.pre_blocks)
        self.calibration_buffer: deque[np.ndarray] = deque(maxlen=CALIBRATION_BLOCKS)
        self.capturing = False
        self.captured_blocks: list[np.ndarray] = []
        self.post_counter = 0
        self.cooldown = 0
        self.bg_rms = 0.002
        self.event_queue: queue.Queue[np.ndarray] = queue.Queue()

    def process_block(self, raw_block: np.ndarray) -> None:
        block = np.asarray(raw_block, dtype=np.float32)
        mono_block = block[:, 0] if block.ndim >= 2 else block

        # Low-pass filter block to prevent continuous 19/20.2 kHz ultrasonic carriers
        # from falsely elevating the acoustic background floor or tripping triggers
        lf_block = apply_lowpass_filter(mono_block, cutoff=LOWPASS_CUTOFF_HZ, sr=self.sr)
        rms = float(np.sqrt(np.mean(lf_block ** 2) + 1e-12))
        peak = float(np.max(np.abs(lf_block)) + 1e-12)
        crest = peak / rms

        if self.cooldown > 0:
            self.cooldown -= 1
            self.pre_buffer.append(block)
            return

        if self.capturing:
            self.captured_blocks.append(block)
            self.post_counter -= 1
            if self.post_counter <= 0:
                full_waveform = np.concatenate(self.captured_blocks, axis=0)
                if full_waveform.ndim == 2 and full_waveform.shape[1] == 1:
                    full_waveform = full_waveform[:, 0]
                self.event_queue.put(full_waveform.astype(np.float32))
                self.capturing = False
                self.captured_blocks = []
                self.cooldown = self.cooldown_blocks
            return

        # Adaptive background threshold
        dynamic_thresh = max(self.threshold, self.bg_rms * self.bg_snr_ratio)

        if rms > dynamic_thresh and peak > (self.threshold * self.peak_mult) and crest >= self.crest_thresh:
            self.capturing = True
            pad_needed = self.pre_blocks - len(self.pre_buffer)
            pad_blocks = [np.zeros_like(block) for _ in range(max(0, pad_needed))]
            self.captured_blocks = pad_blocks + list(self.pre_buffer) + [block]
            self.post_counter = self.post_blocks
        else:
            self.pre_buffer.append(block)
            self.calibration_buffer.append(block)
            self.bg_rms = float(self.ema_alpha * self.bg_rms + (1.0 - self.ema_alpha) * rms)

    def audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        self.process_block(indata.copy())
