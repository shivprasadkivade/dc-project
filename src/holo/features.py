from __future__ import annotations

from typing import Any
import librosa
import numpy as np

try:
    from .config import CFG, LOWPASS_CUTOFF_HZ, SAMPLE_RATE
    from .triangulation import align_and_slice_waveform
    from .ultrasonic import apply_lowpass_filter
except (ImportError, ValueError):
    from config import CFG, LOWPASS_CUTOFF_HZ, SAMPLE_RATE  # type: ignore
    from triangulation import align_and_slice_waveform  # type: ignore
    from ultrasonic import apply_lowpass_filter  # type: ignore

_feat_cfg = CFG.get("features", {})
N_FFT: int = int(_feat_cfg.get("n_fft", 256))
HOP_LENGTH: int = int(_feat_cfg.get("hop_length", 64))
N_MFCC: int = int(_feat_cfg.get("n_mfcc", 20))
N_MELS: int = int(_feat_cfg.get("n_mels", 40))


def extract_features(
    waveform: np.ndarray,
    sr: int = SAMPLE_RATE,
    feature_set: str | None = "enhanced",
) -> np.ndarray:
    """
    Deterministic feature extraction function used identically across collect,
    train, evaluate, and listen.

    Pipeline:
    1. Isolates acoustics by applying a steep Butterworth low-pass filter (16 kHz)
       to completely remove any 19 kHz / 20.2 kHz active ultrasonic carrier bleed.
    2. Aligns and slices strictly to the direct-wave impact peak (eliminating background silence).
    3. Extracts peak-normalized MFCCs (attack onset, wake decay, temporal std).
    4. Extracts spectral centroid, bandwidth, rolloff, flatness, and zero-crossing rate.
    5. Computes physical impact metrics: Crest factor, RMS, attack-decay ratio, and SNR.
    6. Guards all outputs against NaN/Inf values with 1e-8 epsilon and np.nan_to_num.
    """
    if waveform is None:
        return np.zeros(73, dtype=np.float32)

    # Convert to 1D mono float32
    if waveform.ndim == 2:
        raw_mono = np.mean(waveform, axis=1)
    else:
        raw_mono = waveform
    raw_mono = np.asarray(raw_mono, dtype=np.float32).flatten()

    # Pre-filtering with Butterworth LPF (16 kHz cutoff)
    filtered_mono = apply_lowpass_filter(raw_mono, cutoff=LOWPASS_CUTOFF_HZ, sr=sr)

    # Peak alignment and transient slicing
    mono, peak_idx, pre_noise, onset, wake = align_and_slice_waveform(filtered_mono, sr=sr)

    # Peak normalization prevents tap amplitude (light vs hard) from distorting spectral shape
    max_val = float(np.max(np.abs(wake))) + 1e-8
    norm_wake = (wake / max_val).astype(np.float32)
    norm_onset = (onset / max_val).astype(np.float32)

    # MFCC Extraction with upper frequency bound capped at 16 kHz low-pass cutoff
    n_fft_safe = min(N_FFT, max(64, 2 ** int(np.floor(np.log2(len(norm_wake))))))
    hop_safe = min(HOP_LENGTH, n_fft_safe // 4)

    mfccs = librosa.feature.mfcc(
        y=norm_wake,
        sr=sr,
        n_mfcc=N_MFCC,
        n_fft=n_fft_safe,
        hop_length=hop_safe,
        n_mels=N_MELS,
        fmax=min(float(LOWPASS_CUTOFF_HZ), sr / 2.0 - 50.0),
    )

    n_frames = mfccs.shape[1]
    split_frame = min(3, max(1, n_frames // 3))
    onset_mfcc = np.mean(mfccs[:, :split_frame], axis=1)
    decay_mfcc = np.mean(mfccs[:, split_frame:], axis=1) if n_frames > split_frame else onset_mfcc
    std_mfcc = np.std(mfccs, axis=1)

    features: list[np.ndarray] = [onset_mfcc, decay_mfcc, std_mfcc]

    if feature_set == "baseline":
        rms_norm_wake = float(np.sqrt(np.mean(norm_wake ** 2)) + 1e-8)
        features.append(np.array([rms_norm_wake], dtype=np.float32))
        res = np.concatenate(features)
        return np.nan_to_num(res, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Enhanced Features: Spectral & Physical Waveform Mechanics
    centroid = librosa.feature.spectral_centroid(
        y=norm_wake, sr=sr, n_fft=n_fft_safe, hop_length=hop_safe
    ) / float(sr)
    bandwidth = librosa.feature.spectral_bandwidth(
        y=norm_wake, sr=sr, n_fft=n_fft_safe, hop_length=hop_safe
    ) / float(sr)
    rolloff = librosa.feature.spectral_rolloff(
        y=norm_wake, sr=sr, n_fft=n_fft_safe, hop_length=hop_safe, roll_percent=0.85
    ) / float(sr)
    flatness = librosa.feature.spectral_flatness(
        y=norm_wake, n_fft=n_fft_safe, hop_length=hop_safe
    )
    zcr = librosa.feature.zero_crossing_rate(
        y=norm_wake, frame_length=n_fft_safe, hop_length=hop_safe
    )

    rms_norm_wake = float(np.sqrt(np.mean(norm_wake ** 2)) + 1e-8)
    crest_factor = float(1.0 / (rms_norm_wake + 1e-8))

    decay_tail = norm_wake[256:] if len(norm_wake) > 256 else norm_wake
    attack_decay_ratio = float(
        np.log10((np.mean(norm_onset ** 2) + 1e-8) / (np.mean(decay_tail ** 2) + 1e-8))
    )

    rms_wake = float(np.sqrt(np.mean(wake ** 2)) + 1e-8)
    rms_pre = float(np.sqrt(np.mean(pre_noise ** 2)) + 1e-8)
    snr_db = float(np.clip(20.0 * np.log10(rms_wake / rms_pre), 0.0, 35.0))

    features.extend([
        np.mean(centroid, axis=1),
        np.std(centroid, axis=1),
        np.mean(bandwidth, axis=1),
        np.std(bandwidth, axis=1),
        np.mean(rolloff, axis=1),
        np.std(rolloff, axis=1),
        np.mean(flatness, axis=1),
        np.mean(zcr, axis=1),
        np.std(zcr, axis=1),
        np.array([snr_db, crest_factor, attack_decay_ratio, rms_norm_wake], dtype=np.float32),
    ])

    concatenated = np.concatenate(features)
    return np.nan_to_num(concatenated, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
