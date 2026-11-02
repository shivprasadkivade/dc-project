from __future__ import annotations

from typing import Any
import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import StandardScaler

try:
    from .config import CFG, LOWPASS_CUTOFF_HZ, SAMPLE_RATE
    from .ultrasonic import apply_lowpass_filter
except (ImportError, ValueError):
    from config import CFG, LOWPASS_CUTOFF_HZ, SAMPLE_RATE  # type: ignore
    from ultrasonic import apply_lowpass_filter  # type: ignore


def make_onset_window(length: int, peak_offset: int = 32) -> np.ndarray:
    """
    Creates an asymmetric onset-preserving window.
    - [0 : peak_offset]: Smooth rising cosine taper from 0 to 1, reaching 1.0 right at peak_offset.
    - [peak_offset : length]: Smooth decaying cosine taper from 1 down to 0 at the window end.
    This guarantees that the transient direct-wave impact spike at peak_offset receives
    full 1.0 gain (100% transmission), while preventing boundary truncation artifacts.
    """
    w = np.ones(length, dtype=np.float64)
    if peak_offset > 0:
        rise = 0.5 * (1.0 - np.cos(np.pi * np.arange(peak_offset) / max(1, peak_offset)))
        w[:peak_offset] = rise
    tail_len = length - peak_offset
    if tail_len > 0:
        fall = 0.5 * (1.0 + np.cos(np.pi * np.arange(tail_len) / max(1, tail_len)))
        w[peak_offset:] = fall
    return w


def align_and_slice_waveform(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> tuple[np.ndarray, int, np.ndarray, np.ndarray, np.ndarray]:
    """
    Low-passes the raw waveform at 16 kHz to remove ultrasonic carriers,
    locks onto the microsecond impact peak, and slices direct-wave onset and cavity wake.
    Returns: (filtered_mono, peak_idx, pre_noise, onset, wake)
    """
    win_cfg = CFG.get("signal_window", {})
    s_start_cfg = int(win_cfg.get("search_start_samples", 256))
    s_end_margin = int(win_cfg.get("search_end_margin", 512))
    pre_guard = int(win_cfg.get("pre_noise_guard_samples", 96))
    pre_peak = int(win_cfg.get("onset_pre_peak_samples", 32))
    onset_len = int(win_cfg.get("onset_length_samples", 256))
    wake_len = int(win_cfg.get("wake_length_samples", 1024))

    # Convert to 1D mono
    if waveform.ndim == 2:
        raw_mono = np.mean(waveform, axis=1)
    else:
        raw_mono = waveform
    raw_mono = np.asarray(raw_mono, dtype=np.float32).flatten()

    # Low-pass filter to isolate impact impulse from 19 kHz & 20.2 kHz tones
    mono = apply_lowpass_filter(raw_mono, cutoff=LOWPASS_CUTOFF_HZ, sr=sr)
    mono = np.asarray(mono, dtype=np.float64)

    total_len = len(mono)
    if total_len == 0:
        zeros = np.zeros(onset_len, dtype=np.float64)
        return zeros, 0, np.array([1e-6]), zeros, np.zeros(wake_len, dtype=np.float64)

    search_start = min(s_start_cfg, total_len // 6)
    search_end = max(search_start + 128, total_len - s_end_margin)
    if search_start >= search_end:
        search_start, search_end = 0, total_len

    sub_slice = np.abs(mono[search_start:search_end])
    peak_offset = int(np.argmax(sub_slice)) if len(sub_slice) > 0 else 0
    peak_idx = search_start + peak_offset

    # Pre-noise window strictly before peak
    if peak_idx > (pre_guard + 32):
        noise_end = min(256, peak_idx - pre_guard)
    else:
        noise_end = max(16, peak_idx // 2) if peak_idx > 16 else max(1, peak_idx)

    pre_noise = mono[:noise_end]
    if len(pre_noise) == 0:
        pre_noise = np.array([1e-6], dtype=np.float64)

    # Ballistic direct-wave onset
    onset_start = max(0, peak_idx - pre_peak)
    onset = mono[onset_start : onset_start + onset_len]
    if len(onset) < onset_len:
        onset = np.pad(onset, (0, onset_len - len(onset)))

    # Cavity resonance wake
    wake_start = max(0, peak_idx - pre_peak)
    wake = mono[wake_start : wake_start + wake_len]
    if len(wake) < wake_len:
        wake = np.pad(wake, (0, wake_len - len(wake)))

    return mono, peak_idx, pre_noise, onset, wake


def extract_fourier_triangulation_vector(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
) -> np.ndarray:
    """
    Extracts spatial acoustic resonance, Lamb-wave dispersion delay, and phase
    triangulation features strictly from the 16 kHz low-passed impact window.
    Guarantees no NaNs or Infs.
    """
    ft_cfg = CFG.get("fourier_triangulation", {})
    win_cfg = CFG.get("signal_window", {})
    pre_peak = int(win_cfg.get("onset_pre_peak_samples", 32))
    num_bands = int(ft_cfg.get("num_bands", 12))
    f_min = float(ft_cfg.get("freq_min_hz", 150.0))
    f_max = float(ft_cfg.get("freq_max_hz", 15000.0))
    hf_band = ft_cfg.get("hf_band_hz", [3000.0, 14000.0])
    lf_band = ft_cfg.get("lf_band_hz", [150.0, 900.0])
    hf_low, hf_high = float(hf_band[0]), float(hf_band[1])
    lf_low, lf_high = float(lf_band[0]), float(lf_band[1])
    max_lag = int(ft_cfg.get("gcc_max_lag_samples", 20))

    mono, peak_idx, pre_noise, onset, wake = align_and_slice_waveform(waveform, sr=sr)

    peak_amp = float(np.max(np.abs(wake))) + 1e-8
    peak_offset = min(pre_peak, peak_idx)
    onset_window = make_onset_window(len(onset), peak_offset=peak_offset)
    wake_window = make_onset_window(len(wake), peak_offset=peak_offset)

    norm_onset = (onset / peak_amp) * onset_window
    norm_wake = (wake / peak_amp) * wake_window

    # Onset spatial spectrum
    fft_onset = np.fft.rfft(norm_onset)
    mag_onset = np.abs(fft_onset) + 1e-8
    freqs_onset = np.fft.rfftfreq(len(norm_onset), d=1.0 / sr)

    band_edges = np.geomspace(f_min, f_max, num=num_bands + 1)
    onset_band_energies = []
    for i in range(num_bands):
        mask = (freqs_onset >= band_edges[i]) & (freqs_onset < band_edges[i + 1])
        if np.any(mask):
            val = float(np.log10(np.mean(mag_onset[mask] ** 2) + 1e-10))
        else:
            val = -10.0
        onset_band_energies.append(val)

    # Wake spatial resonance spectrum
    fft_wake = np.fft.rfft(norm_wake)
    mag_wake = np.abs(fft_wake) + 1e-8
    freqs_wake = np.fft.rfftfreq(len(norm_wake), d=1.0 / sr)

    wake_band_energies = []
    for i in range(num_bands):
        mask = (freqs_wake >= band_edges[i]) & (freqs_wake < band_edges[i + 1])
        if np.any(mask):
            val = float(np.log10(np.mean(mag_wake[mask] ** 2) + 1e-10))
        else:
            val = -10.0
        wake_band_energies.append(val)

    # Spatial resonance sub-band ratios
    band_diffs = list(np.diff(wake_band_energies))

    # Flexural Lamb-wave dispersion delay
    hf_fft = np.where((freqs_wake >= hf_low) & (freqs_wake <= hf_high), fft_wake, 0.0)
    lf_fft = np.where((freqs_wake >= lf_low) & (freqs_wake <= lf_high), fft_wake, 0.0)
    hf_time_env = np.abs(np.fft.irfft(hf_fft))
    lf_time_env = np.abs(np.fft.irfft(lf_fft))

    search_lim = min(256, len(hf_time_env))
    hf_arrival = int(np.argmax(hf_time_env[:search_lim])) if search_lim > 0 else 0
    lf_arrival = int(np.argmax(lf_time_env[:search_lim])) if search_lim > 0 else 0
    dispersion_lag_ms = float(((lf_arrival - hf_arrival) / sr) * 1000.0)
    hf_to_lf_ratio = float(np.log10((np.sum(hf_time_env ** 2) + 1e-8) / (np.sum(lf_time_env ** 2) + 1e-8)))

    # Phase unwrapping and group delay
    unwrapped_phase = np.unwrap(np.angle(fft_onset))
    group_delay = -np.diff(unwrapped_phase)
    gd_low = float(np.mean(group_delay[1:min(8, len(group_delay))])) if len(group_delay) > 1 else 0.0
    gd_mid = float(np.mean(group_delay[8:min(32, len(group_delay))])) if len(group_delay) > 8 else 0.0
    gd_high = float(np.mean(group_delay[32:min(90, len(group_delay))])) if len(group_delay) > 32 else 0.0

    snr_db = float(10.0 * np.log10((np.mean(onset ** 2) + 1e-10) / (np.mean(pre_noise ** 2) + 1e-10)))

    # Stereo GCC-PHAT cross correlation (if stereo recording available)
    if waveform.ndim == 2 and waveform.shape[1] >= 2:
        start_idx = max(0, peak_idx - 32)
        ch_l = apply_lowpass_filter(waveform[start_idx : start_idx + 512, 0], cutoff=LOWPASS_CUTOFF_HZ, sr=sr)
        ch_r = apply_lowpass_filter(waveform[start_idx : start_idx + 512, 1], cutoff=LOWPASS_CUTOFF_HZ, sr=sr)
        if len(ch_l) < 512:
            ch_l = np.pad(ch_l, (0, 512 - len(ch_l)))
            ch_r = np.pad(ch_r, (0, 512 - len(ch_r)))
        F_l = np.fft.rfft(ch_l, n=1024)
        F_r = np.fft.rfft(ch_r, n=1024)
        cross_spec = F_l * np.conj(F_r)
        gcc_phat = np.fft.irfft(cross_spec / (np.abs(cross_spec) + 1e-8), n=1024)
        gcc_shifted = np.fft.fftshift(gcc_phat)
        center = len(gcc_shifted) // 2
        lag_sub = gcc_shifted[center - max_lag : center + max_lag + 1]
        lag_samples = float(int(np.argmax(lag_sub)) - max_lag) if len(lag_sub) > 0 else 0.0
        ild_db = float(10.0 * np.log10((np.sum(ch_l ** 2) + 1e-8) / (np.sum(ch_r ** 2) + 1e-8)))
    else:
        lag_samples = 0.0
        ild_db = 0.0

    feature_list = (
        onset_band_energies
        + wake_band_energies
        + band_diffs
        + [dispersion_lag_ms, hf_to_lf_ratio, gd_low, gd_mid, gd_high, snr_db, lag_samples, ild_db]
    )
    vec = np.array(feature_list, dtype=np.float32)
    return np.nan_to_num(vec, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


class FourierTriangulator:
    """
    Fourier-domain Acoustic Triangulation Classifier:
    Combines Shrinkage Linear Discriminant Analysis (LDA) with cosine
    spectral centroid template matching in the Fourier feature domain.
    """

    def __init__(
        self,
        lda_weight: float | None = None,
        cos_temp: float | None = None,
        solver: str = "lsqr",
        shrinkage: str | float = "auto",
    ):
        ft_cfg = CFG.get("fourier_triangulation", {})
        self.lda_weight = float(lda_weight if lda_weight is not None else ft_cfg.get("lda_weight", 0.65))
        self.cos_temp = float(cos_temp if cos_temp is not None else ft_cfg.get("cosine_temperature", 2.5))
        self.scaler = StandardScaler()
        self.lda = LinearDiscriminantAnalysis(solver=solver, shrinkage=shrinkage)
        self.centroids: np.ndarray | None = None
        self.classes_: np.ndarray | None = None
        self.has_lda: bool = False

    def fit(self, X_fourier: np.ndarray, y: np.ndarray) -> FourierTriangulator:
        X_mat = np.asarray(X_fourier, dtype=np.float32)
        y_arr = np.asarray(y)

        X_scaled = self.scaler.fit_transform(X_mat)
        self.classes_ = np.unique(y_arr)

        if len(y_arr) > len(self.classes_):
            try:
                self.lda.fit(X_scaled, y_arr)
                self.has_lda = True
            except Exception:
                self.has_lda = False
        else:
            self.has_lda = False

        centroids = []
        for c in self.classes_:
            cls_samples = X_scaled[y_arr == c]
            if len(cls_samples) > 0:
                centroids.append(np.mean(cls_samples, axis=0))
            else:
                centroids.append(np.zeros(X_scaled.shape[1], dtype=np.float32))
        self.centroids = np.array(centroids, dtype=np.float32)
        return self

    def predict_proba(self, X_fourier: np.ndarray) -> np.ndarray:
        X_mat = np.asarray(X_fourier, dtype=np.float32)
        if X_mat.ndim == 1:
            X_mat = X_mat.reshape(1, -1)

        X_scaled = self.scaler.transform(X_mat)
        if self.has_lda:
            lda_probs = self.lda.predict_proba(X_scaled)
        else:
            n_cls = len(self.classes_) if self.classes_ is not None and len(self.classes_) > 0 else 3
            lda_probs = np.full((len(X_scaled), n_cls), 1.0 / n_cls, dtype=np.float64)

        if self.centroids is None or self.classes_ is None:
            return lda_probs

        # Normalized cosine similarity to class centroids
        norm_x = X_scaled / (np.linalg.norm(X_scaled, axis=1, keepdims=True) + 1e-8)
        norm_c = self.centroids / (np.linalg.norm(self.centroids, axis=1, keepdims=True) + 1e-8)
        cos_sim = np.dot(norm_x, norm_c.T) * self.cos_temp
        exp_sim = np.exp(cos_sim - np.max(cos_sim, axis=1, keepdims=True))
        cos_probs = exp_sim / (np.sum(exp_sim, axis=1, keepdims=True) + 1e-8)

        if self.has_lda:
            fused = self.lda_weight * lda_probs + (1.0 - self.lda_weight) * cos_probs
        else:
            fused = cos_probs
        return np.nan_to_num(fused, nan=1.0 / len(self.classes_))

    def to_dict(self) -> dict[str, Any]:
        """Serializes triangulator components into standard scikit-learn / numpy primitives."""
        return {
            "scaler": self.scaler,
            "lda": self.lda,
            "centroids": self.centroids,
            "classes": self.classes_,
            "lda_weight": self.lda_weight,
            "cos_temp": self.cos_temp,
            "has_lda": self.has_lda,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> FourierTriangulator:
        """Restores a FourierTriangulator from a dictionary bundle."""
        instance = cls(
            lda_weight=data.get("lda_weight", 0.65),
            cos_temp=data.get("cos_temp", 2.5),
        )
        instance.scaler = data.get("scaler", StandardScaler())
        instance.lda = data.get("lda", LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"))
        instance.centroids = data.get("centroids")
        instance.classes_ = data.get("classes")
        instance.has_lda = bool(data.get("has_lda", True))
        return instance
