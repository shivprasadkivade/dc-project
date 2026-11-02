from __future__ import annotations

import threading
import time
from typing import Any
import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt
import sounddevice as sd

try:
    from .config import (
        ALPHA_DOPPLER,
        CARRIER_BW_HZ,
        DOPPLER_MAX_HZ,
        DOPPLER_MIN_HZ,
        GATE_STEEPNESS,
        GATE_THRESHOLD,
        LEFT_ULTRA_HZ,
        LOWPASS_CUTOFF_HZ,
        RAMP_DURATION_MS,
        RIGHT_ULTRA_HZ,
        SAMPLE_RATE,
        SPEAKER_AMPLITUDE,
    )
except (ImportError, ValueError):
    from config import (  # type: ignore
        ALPHA_DOPPLER,
        CARRIER_BW_HZ,
        DOPPLER_MAX_HZ,
        DOPPLER_MIN_HZ,
        GATE_STEEPNESS,
        GATE_THRESHOLD,
        LEFT_ULTRA_HZ,
        LOWPASS_CUTOFF_HZ,
        RAMP_DURATION_MS,
        RIGHT_ULTRA_HZ,
        SAMPLE_RATE,
        SPEAKER_AMPLITUDE,
    )


def apply_lowpass_filter(
    waveform: np.ndarray,
    cutoff: float = LOWPASS_CUTOFF_HZ,
    sr: int = SAMPLE_RATE,
    order: int = 5,
) -> np.ndarray:
    """
    Applies a steep Butterworth low-pass filter (default cutoff: 16 kHz) to isolate
    structure-borne acoustic tap transients from 19 kHz / 20.2 kHz active ultrasonic carriers.
    Guarantees no NaNs or Infs.
    """
    if waveform is None:
        return np.zeros(0, dtype=np.float32)

    data = np.asarray(waveform, dtype=np.float32)
    orig_shape = data.shape
    flat = data.flatten()
    n = len(flat)

    if n == 0:
        return np.zeros(orig_shape, dtype=np.float32)

    nyquist = 0.5 * sr
    if cutoff >= nyquist:
        return np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    norm_cutoff = min(max(cutoff / nyquist, 0.001), 0.999)
    try:
        sos = butter(order, norm_cutoff, btype="low", analog=False, output="sos")
        # Use zero-phase sosfiltfilt if buffer is long enough, else forward sosfilt
        padlen = 3 * (order + 1)
        if n > padlen:
            filtered = sosfiltfilt(sos, flat)
        else:
            filtered = sosfilt(sos, flat)
    except Exception:
        # Fallback if filter design fails
        filtered = flat

    filtered = np.nan_to_num(filtered, nan=0.0, posinf=0.0, neginf=0.0)
    return filtered.reshape(orig_shape).astype(np.float32)


# Backwards compatibility alias
lowpass_for_v2 = apply_lowpass_filter


class UltrasonicSonarGate:
    """
    Continuous Active Stereo Ultrasonic Sonar Gate:
    - Emits continuous stereo ultrasonic carrier tones (19.0 kHz Left, 20.2 kHz Right)
      with smooth fade-in and fade-out cosine envelopes to eliminate speaker clicks.
    - Inspects high-frequency reflection band (18.5 - 21.0 kHz) around carriers for
      Doppler broadening and amplitude perturbations caused by finger/hand motion.
    - Provides two-factor touch verification to reject table bumps, keyboard typing,
      and non-proximity impacts.
    """

    def __init__(
        self,
        sr: int = SAMPLE_RATE,
        left_freq: float = LEFT_ULTRA_HZ,
        right_freq: float = RIGHT_ULTRA_HZ,
        amplitude: float = SPEAKER_AMPLITUDE,
        carrier_bw: float = CARRIER_BW_HZ,
        doppler_min: float = DOPPLER_MIN_HZ,
        doppler_max: float = DOPPLER_MAX_HZ,
        alpha_doppler: float = ALPHA_DOPPLER,
        gate_threshold: float = GATE_THRESHOLD,
        gate_steepness: float = GATE_STEEPNESS,
        ramp_duration_ms: float = RAMP_DURATION_MS,
    ):
        self.sr = int(sr)
        self.left_freq = float(left_freq)
        self.right_freq = float(right_freq)
        self.amplitude = float(amplitude)
        self.carrier_bw = float(carrier_bw)
        self.doppler_min = float(doppler_min)
        self.doppler_max = float(doppler_max)
        self.alpha = float(alpha_doppler)
        self.threshold = float(gate_threshold)
        self.steepness = float(gate_steepness)
        self.ramp_samples = max(64, int((ramp_duration_ms / 1000.0) * self.sr))

        self._phase_l = 0.0
        self._phase_r = 0.0
        self._out_stream: sd.OutputStream | None = None
        self._lock = threading.Lock()

        # Ramp envelope state
        self._envelope_level = 0.0
        self._fading_out = False
        self._active = False

        # Baseline calibration statistics
        self.base_mean = {"left": 1e-3, "right": 1e-3}
        self.base_std = {"left": 2e-4, "right": 2e-4}
        self.base_doppler_mean = {"left": 0.08, "right": 0.08}
        self.base_doppler_std = {"left": 0.02, "right": 0.02}
        self.is_calibrated = False

    def fill_output_buffer(self, outdata: np.ndarray, frames: int) -> None:
        """
        Fills a stereo output buffer with continuous carrier sine waves, smoothly
        modulated by the current fade-in / fade-out envelope.
        Can be called directly from an OutputStream or full-duplex Stream callback.
        """
        t = np.arange(frames, dtype=np.float64) / self.sr
        carrier_l = np.sin(2.0 * np.pi * self.left_freq * t + self._phase_l)
        carrier_r = np.sin(2.0 * np.pi * self.right_freq * t + self._phase_r)

        # Advance continuous phases
        self._phase_l = float((self._phase_l + 2.0 * np.pi * self.left_freq * frames / self.sr) % (2.0 * np.pi))
        self._phase_r = float((self._phase_r + 2.0 * np.pi * self.right_freq * frames / self.sr) % (2.0 * np.pi))

        # Compute smooth envelope progression
        envelope = np.zeros(frames, dtype=np.float64)
        ramp_step = 1.0 / float(self.ramp_samples)

        for i in range(frames):
            if self._fading_out:
                self._envelope_level = max(0.0, self._envelope_level - ramp_step)
            elif self._active:
                self._envelope_level = min(1.0, self._envelope_level + ramp_step)
            # Raised cosine smooth envelope
            # map [0, 1] linearly to smooth cosine [0, 1]
            smooth_gain = 0.5 * (1.0 - np.cos(np.pi * self._envelope_level))
            envelope[i] = smooth_gain

        gain = (self.amplitude * envelope).astype(np.float32)
        outdata[:, 0] = (carrier_l * gain).astype(np.float32)
        if outdata.shape[1] >= 2:
            outdata[:, 1] = (carrier_r * gain).astype(np.float32)
        else:
            outdata[:, 0] = (0.5 * (carrier_l + carrier_r) * gain).astype(np.float32)

    def _speaker_callback(self, outdata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        with self._lock:
            self.fill_output_buffer(outdata, frames)

    def start_emitter(self) -> bool:
        """Starts continuous stereo ultrasonic emission with smooth fade-in envelope."""
        with self._lock:
            self._active = True
            self._fading_out = False
            if self._out_stream is not None:
                return True

            try:
                self._out_stream = sd.OutputStream(
                    samplerate=self.sr,
                    channels=2,
                    dtype="float32",
                    blocksize=1024,
                    callback=self._speaker_callback,
                )
                self._out_stream.start()
                return True
            except Exception as e:
                # Fallback to mono if 2-channel output is unsupported
                try:
                    self._out_stream = sd.OutputStream(
                        samplerate=self.sr,
                        channels=1,
                        dtype="float32",
                        blocksize=1024,
                        callback=self._speaker_callback,
                    )
                    self._out_stream.start()
                    return True
                except Exception as e2:
                    print(f"[Ultrasonic] Warning: Could not initialize speaker output stream ({e2})")
                    self._out_stream = None
                    return False

    def stop_emitter(self) -> None:
        """Smoothly fades out ultrasonic carriers and terminates the output stream."""
        with self._lock:
            if self._out_stream is None:
                self._active = False
                self._fading_out = False
                self._envelope_level = 0.0
                return

            # Signal fade out
            self._fading_out = True
            self._active = False

        # Allow fade-out ramp to play through
        fade_time = (self.ramp_samples / self.sr) + 0.05
        time.sleep(fade_time)

        with self._lock:
            if self._out_stream is not None:
                try:
                    self._out_stream.stop()
                    self._out_stream.close()
                except Exception:
                    pass
                self._out_stream = None
            self._envelope_level = 0.0
            self._fading_out = False

    def _to_mono(self, waveform: np.ndarray) -> np.ndarray:
        if waveform is None:
            return np.zeros(0, dtype=np.float32)
        arr = np.asarray(waveform, dtype=np.float32)
        if arr.ndim >= 2:
            return np.mean(arr, axis=1).flatten()
        return arr.flatten()

    def extract_band_metrics(self, waveform: np.ndarray, carrier_freq: float) -> tuple[float, float]:
        """
        Extracts narrowband carrier peak amplitude and surrounding Doppler sideband energy.
        Returns: (carrier_energy, doppler_ratio)
        """
        y = self._to_mono(waveform)
        n = len(y)
        if n < 64 or self.sr <= (carrier_freq * 2.0):
            return 0.0, 0.0

        # Window to minimize spectral leakage
        windowed = y * np.hanning(n)
        spec = np.abs(np.fft.rfft(windowed))
        freqs = np.fft.rfftfreq(n, d=1.0 / self.sr)

        freq_diff = np.abs(freqs - carrier_freq)
        carrier_mask = freq_diff <= self.carrier_bw
        sideband_mask = (freq_diff >= self.doppler_min) & (freq_diff <= self.doppler_max)

        if not np.any(carrier_mask) or not np.any(sideband_mask):
            return 0.0, 0.0

        carrier_energy = float(np.mean(spec[carrier_mask])) + 1e-8
        sideband_energy = float(np.mean(spec[sideband_mask])) + 1e-8
        doppler_ratio = float(sideband_energy / carrier_energy)

        carrier_energy = float(np.nan_to_num(carrier_energy, nan=0.0, posinf=0.0, neginf=0.0))
        doppler_ratio = float(np.nan_to_num(doppler_ratio, nan=0.0, posinf=0.0, neginf=0.0))
        return carrier_energy, doppler_ratio

    def has_active_carrier(self, waveform: np.ndarray) -> bool:
        """
        Checks whether active ultrasonic carrier tones are present in the recorded microphone signal.
        If no carrier is detected (e.g. offline dataset or speaker off), gating is safely neutral.
        """
        y = self._to_mono(waveform)
        if len(y) < 128 or self.sr < 44100:
            return False

        c_l, _ = self.extract_band_metrics(y, self.left_freq)
        c_r, _ = self.extract_band_metrics(y, self.right_freq)
        return bool((c_l + c_r) > 1e-4)

    def calibrate_baseline(self, empty_windows: list[np.ndarray]) -> None:
        """Calibrates ambient background carrier and Doppler levels from idle microphone frames."""
        if not empty_windows:
            return

        for side, freq in [("left", self.left_freq), ("right", self.right_freq)]:
            c_vals: list[float] = []
            d_vals: list[float] = []
            for w in empty_windows:
                c_en, d_rat = self.extract_band_metrics(w, freq)
                c_vals.append(c_en)
                d_vals.append(d_rat)

            mean_c = float(np.mean(c_vals)) if c_vals else 1e-3
            std_c = float(np.std(c_vals)) if c_vals else 2e-4
            mean_d = float(np.mean(d_vals)) if d_vals else 0.08
            std_d = float(np.std(d_vals)) if d_vals else 0.02

            self.base_mean[side] = max(mean_c, 1e-5)
            self.base_std[side] = max(std_c, self.base_mean[side] * 0.08, 1e-6)
            self.base_doppler_mean[side] = max(mean_d, 1e-4)
            self.base_doppler_std[side] = max(std_d, 0.015)

        self.is_calibrated = True

    def _side_gate(self, waveform: np.ndarray, side: str, freq: float) -> tuple[float, float]:
        """Calculates sigmoid gate [0.0, 1.0] and raw perturbation score for a given carrier."""
        c_en, d_rat = self.extract_band_metrics(waveform, freq)
        z_doppler = max(0.0, (d_rat - self.base_doppler_mean[side]) / self.base_doppler_std[side])
        z_reflect = abs(c_en - self.base_mean[side]) / self.base_std[side]

        score = float(self.alpha * z_doppler + (1.0 - self.alpha) * z_reflect)
        # Sigmoid gate centered at threshold
        gate = float(1.0 / (1.0 + np.exp(-self.steepness * (score - self.threshold))))
        gate = float(np.clip(np.nan_to_num(gate, nan=0.5), 0.0, 1.0))
        score = float(np.nan_to_num(score, nan=0.0))
        return gate, score

    def evaluate_window(self, raw_waveform: np.ndarray) -> dict[str, Any]:
        """
        Evaluates the raw pre-filtered audio window for ultrasonic motion/reflection.
        Returns gate confidence and raw scores for both Left and Right channels.
        """
        if not self.has_active_carrier(raw_waveform):
            return {
                "gate_left": 1.0,
                "gate_right": 1.0,
                "score_left": 0.0,
                "score_right": 0.0,
                "active": False,
            }

        gate_l, score_l = self._side_gate(raw_waveform, "left", self.left_freq)
        gate_r, score_r = self._side_gate(raw_waveform, "right", self.right_freq)

        return {
            "gate_left": gate_l,
            "gate_right": gate_r,
            "score_left": score_l,
            "score_right": score_r,
            "active": True,
        }

    def extract_ultrasonic_feature_vector(self, raw_waveform: np.ndarray) -> np.ndarray:
        """
        Extracts 6-dimensional ultrasonic feature vector:
        [log1p(c_left), log1p(c_right), doppler_left, doppler_right, carrier_asym, doppler_asym]
        """
        if not self.has_active_carrier(raw_waveform):
            return np.zeros(6, dtype=np.float32)

        c_l, d_l = self.extract_band_metrics(raw_waveform, self.left_freq)
        c_r, d_r = self.extract_band_metrics(raw_waveform, self.right_freq)

        c_asym = (c_l - c_r) / (c_l + c_r + 1e-8)
        d_asym = (d_l - d_r) / (d_l + d_r + 1e-8)

        vec = np.array(
            [np.log1p(c_l), np.log1p(c_r), d_l, d_r, c_asym, d_asym],
            dtype=np.float32,
        )
        return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
