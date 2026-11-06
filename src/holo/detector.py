import queue
from collections import deque
import numpy as np
from .config import (
    BLOCK_SIZE, PRE_BLOCKS, POST_BLOCKS, COOLDOWN_BLOCKS,
    RMS_THRESHOLD, PEAK_MULTIPLIER, BG_SNR_RATIO, BG_EMA_ALPHA
)

class ImpulseDetector:
    """Captures fixed-length audio snippets around sharp acoustic transients with dynamic noise floor tracking."""
    def __init__(self, threshold: float = RMS_THRESHOLD):
        self.threshold = threshold
        self.pre_buffer = deque(maxlen=PRE_BLOCKS)
        self.capturing = False
        self.captured_blocks = []
        self.cooldown = 0
        self.bg_rms = 0.002
        self.event_queue = queue.Queue()

    def audio_callback(self, indata, frames, time_info, status):
        mono_block = indata[:, 0].copy() if indata.ndim > 1 else indata.copy()
        rms = float(np.sqrt(np.mean(mono_block ** 2)))
        peak = float(np.max(np.abs(mono_block)))

        # During cooldown, buffer audio but do NOT update background noise floor with tap reverberation
        if self.cooldown > 0:
            self.cooldown -= 1
            self.pre_buffer.append(block)
            # Do NOT update bg_rms during cooldown to avoid absorbing chassis reverberation
            return

        if self.capturing:
            self.captured_blocks.append(block)
            if len(self.captured_blocks) >= (self.pre_blocks + 1 + self.post_blocks):
                full_waveform = np.concatenate(self.captured_blocks, axis=0)
                if full_waveform.ndim == 2 and full_waveform.shape[1] == 1:
                    full_waveform = full_waveform[:, 0]
                self.event_queue.put(full_waveform)
                self.capturing = False
                self.captured_blocks = []
                self.cooldown = self.cooldown_blocks
        else:
            dynamic_thresh = max(self.threshold, self.bg_rms * BG_SNR_RATIO)
            # Require both threshold exceeding dynamic floor and sharp crest factor
            if rms > dynamic_thresh and peak > (rms * PEAK_MULTIPLIER):
                self.capturing = True
                # Fix pre-buffer underflow: pad with zeros if stream just started
                pad_count = PRE_BLOCKS - len(self.pre_buffer)
                pad_blocks = [np.zeros_like(mono_block) for _ in range(pad_count)]
                self.captured_blocks = pad_blocks + list(self.pre_buffer) + [mono_block]
            else:
                self.pre_buffer.append(mono_block)
                # Exponential moving average for background noise floor
                self.bg_rms = BG_EMA_ALPHA * self.bg_rms + (1.0 - BG_EMA_ALPHA) * rms
