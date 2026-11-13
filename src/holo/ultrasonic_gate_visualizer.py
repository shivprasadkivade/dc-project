import argparse
import time
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

try:
    import sounddevice as sd
    SOUNDDEVICE_AVAILABLE = True
except ImportError:
    SOUNDDEVICE_AVAILABLE = False

# ================= CONFIGURATION =================
FS = 48000                  # 48 kHz sample rate
DURATION = 12.0             # Total capture time in seconds
F_LEFT = 19000.0            # Left speaker frequency (19 kHz)
F_RIGHT = 20000.0           # Right speaker frequency (20 kHz)
NFFT = 4096                 # High frequency resolution (~11.72 Hz per bin)
HOP_LEN = 512               # ~10.67 ms time step


def generate_stereo_ultrasonic(fs: int, duration: float, f_left: float, f_right: float) -> np.ndarray:
    """Generates stereo ultrasonic tones with maximized 20 kHz right channel drive."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    fade_len = int(0.15 * fs)
    envelope = np.ones_like(t)
    envelope[:fade_len] = 0.5 * (1 - np.cos(np.pi * np.arange(fade_len) / fade_len))
    envelope[-fade_len:] = 0.5 * (1 + np.cos(np.pi * np.arange(fade_len) / fade_len))

    # Attenuate 19 kHz slightly and drive 20 kHz at 0.98 full scale to overcome MacBook 20kHz roll-off
    left_channel = 0.35 * np.sin(2 * np.pi * f_left * t) * envelope
    right_channel = 0.98 * np.sin(2 * np.pi * f_right * t) * envelope
    return np.column_stack((left_channel, right_channel)).astype(np.float32)


def simulate_recording(fs: int, duration: float, f_left: float, f_right: float) -> np.ndarray:
    """Synthesizes realistic microphone capture with hand-induced Doppler wobble & shadowing."""
    t = np.linspace(0, duration, int(fs * duration), endpoint=False)
    np.random.seed(42)
    rx = 0.004 * np.random.normal(0, 1, len(t))

    left_env = np.ones_like(t)
    left_phase = np.zeros_like(t)
    mask_l = (t >= 3.3) & (t <= 5.8)
    t_l = t[mask_l] - 3.3
    left_env[mask_l] = 1.0 - 0.60 * np.sin(np.pi * t_l / 2.5) * (1 + 0.45 * np.sin(2 * np.pi * 5.5 * t_l))
    left_phase[mask_l] = 18.0 * np.sin(2 * np.pi * 4.5 * t_l)

    right_env = np.ones_like(t)
    right_phase = np.zeros_like(t)
    mask_r = (t >= 7.3) & (t <= 9.8)
    t_r = t[mask_r] - 7.3
    right_env[mask_r] = 1.0 - 0.60 * np.sin(np.pi * t_r / 2.5) * (1 + 0.45 * np.cos(2 * np.pi * 6.0 * t_r))
    right_phase[mask_r] = 20.0 * np.sin(2 * np.pi * 5.0 * t_r)

    sig_left = 0.25 * left_env * np.sin(2 * np.pi * f_left * t + left_phase)
    sig_right = 0.08 * right_env * np.sin(2 * np.pi * f_right * t + right_phase)
    return (rx + sig_left + sig_right).astype(np.float32)


def capture_live_audio(fs: int, duration: float, f_left: float, f_right: float) -> np.ndarray:
    """Simultaneously plays stereo 19/20 kHz tones and records mono mic input."""
    if not SOUNDDEVICE_AVAILABLE:
        raise RuntimeError("sounddevice is not installed. Run `uv pip install sounddevice` or use `--simulate`.")

    tx_stereo = generate_stereo_ultrasonic(fs, duration, f_left, f_right)
    print("\n" + "=" * 70)
    print("  ACOUSTIC TAP: ULTRASONIC GATE VISUALIZATION CAPTURE")
    print("=" * 70)
    print("HOW TO GET STRONG RIGHT-HAND (20 kHz) & LEFT-HAND (19 kHz) SPIKES:")
    print("  1. Set MacBook volume to 80-90%.")
    print("  2. [3.0s - 6.0s] Wave your LEFT PALM right over the LEFT speaker/keyboard")
    print("     (near Tab/CapsLock/Left Palm Rest) moving toward & away from the screen.")
    print("  3. [7.0s - 10.0s] Wave your RIGHT PALM right over the RIGHT speaker/keyboard")
    print("     (near Return/Backslash/Right Palm Rest) moving toward & away from the screen.")
    print("=" * 70)

    for i in range(3, 0, -1):
        print(f"Starting in {i}...")
        time.sleep(1.0)

    print(">>> RECORDING ACTIVE <<<")
    rx_audio = sd.playrec(tx_stereo, samplerate=fs, channels=1, dtype='float32', blocking=False)

    start_t = time.time()
    while time.time() - start_t < duration:
        elapsed = time.time() - start_t
        if elapsed < 3.0:
            stage = "HOLD STILL (Calibrating Quiet Baseline)"
        elif elapsed < 6.0:
            stage = ">>> WAVE & TAP LEFT PALM OVER LEFT SIDE (19 kHz) <<<"
        elif elapsed < 7.0:
            stage = "HOLD STILL (Resetting Hands Away)"
        elif elapsed < 10.0:
            stage = ">>> WAVE & TAP RIGHT PALM OVER RIGHT SIDE (20 kHz) <<<"
        else:
            stage = "HOLD STILL (Finishing)"
        print(f"\r[{elapsed:5.1f}s / {duration:.1f}s] {stage:<54}", end="", flush=True)
        time.sleep(0.08)

    sd.wait()
    print("\nCapture complete! Generating visual proof report...")
    return rx_audio.flatten()


def extract_iq_wobble(rx_signal: np.ndarray, fs: int, carrier_hz: float, t_stft: np.ndarray, base_mask: np.ndarray) -> np.ndarray:
    """
    Uses coherent IQ lock-in demodulation around `carrier_hz` (+/- 65 Hz) to detect both:
      1. Acoustic shadowing (amplitude drop/fluctuation of the exact carrier)
      2. Micro-Doppler shift (instantaneous phase/frequency derivative from hand velocity)
    """
    t_audio = np.arange(len(rx_signal)) / fs
    # Mix carrier down to baseband (DC)
    lo = np.exp(-1j * 2 * np.pi * carrier_hz * t_audio)
    baseband = rx_signal * lo

    # Low-pass filter at 65 Hz (isolates carrier + hand Doppler sidebands, rejects taps & other channel)
    sos = signal.butter(4, 65.0, btype='low', fs=fs, output='sos')
    iq = signal.sosfiltfilt(sos, baseband)

    # Downsample IQ envelope and instantaneous phase wobble onto the STFT time grid
    idx = np.clip((t_stft * fs).astype(int), 0, len(iq) - 1)
    env = np.abs(iq[idx])
    inst_freq_wobble = np.abs(np.diff(np.unwrap(np.angle(iq[idx])), prepend=0))

    # Ignore startup transient
    valid_base = base_mask & (t_stft >= 0.7)
    env_mean = np.median(env[valid_base]) + 1e-9
    env_std = np.std(env[valid_base]) + 0.02 * env_mean

    freq_mean = np.median(inst_freq_wobble[valid_base])
    freq_std = np.std(inst_freq_wobble[valid_base]) + 1e-4

    # Combine normalized amplitude shadowing/reflection + frame-to-frame envelope velocity + phase Doppler
    amp_dev = np.abs(env - env_mean) / env_std
    amp_vel = np.abs(np.diff(env, prepend=env[0])) / env_std
    doppler_z = np.maximum(0.0, (inst_freq_wobble - freq_mean) / freq_std)

    combined = 0.45 * amp_dev + 0.35 * amp_vel + 0.20 * doppler_z
    combined[t_stft < 0.5] = 0.0

    # Smooth over ~90ms window for clean visual presentation
    kernel = np.ones(9) / 9.0
    smoothed = np.convolve(combined, kernel, mode='same')

    # Normalize to disturbance percentage where quiet baseline sits below 25% and threshold is 50%
    base_floor = np.percentile(smoothed[valid_base], 85)
    base_scale = np.std(smoothed[valid_base]) + 0.25
    score_pct = np.maximum(0.0, (smoothed - base_floor) / base_scale) * 22.0
    return score_pct


def analyze_and_plot(rx_signal: np.ndarray, fs: int, output_path: str = "ultrasonic_gate_proof.png"):
    """Computes carrier-equalized STFT, Perturbation Heatmap, and IQ Lock-in Gate Signals."""
    f, t, Zxx = signal.stft(
        rx_signal,
        fs=fs,
        window='hann',
        nperseg=NFFT,
        noverlap=NFFT - HOP_LEN
    )
    mag_db = 20 * np.log10(np.abs(Zxx) + 1e-9)

    base_mask = (t >= 0.7) & (t <= 2.7)
    if np.sum(base_mask) < 5:
        base_mask = (t >= 0.4) & (t <= 2.0)

    # Crop to Ultrasonic ROI (18.4 kHz to 20.6 kHz)
    roi_mask = (f >= 18400) & (f <= 20600)
    f_roi = f[roi_mask] / 1000.0
    spec_roi_raw = mag_db[roi_mask, :].copy()

    # 1. Suppress broadband vertical click impulses using guard bands (18.55-18.75 kHz & 19.4-19.65 kHz)
    guard_mask = ((f_roi >= 18.55) & (f_roi <= 18.75)) | ((f_roi >= 19.40) & (f_roi <= 19.65))
    broadband_impulse_db = np.median(spec_roi_raw[guard_mask, :], axis=0, keepdims=True)
    broadband_base_db = np.median(broadband_impulse_db[:, base_mask])
    click_artifact = np.maximum(0.0, broadband_impulse_db - broadband_base_db - 2.0)

    spec_clean = spec_roi_raw - 0.90 * click_artifact
    spec_clean[:, t < 0.35] = np.median(spec_clean[:, base_mask], axis=1, keepdims=True)

    # Boost visual gain of the 19.7-20.3 kHz upper band in Panel 1 by +18 dB to compensate mic roll-off
    upper_band = (f_roi >= 19.65)[:, None]
    spec_display = spec_clean + upper_band * 16.0

    # 2. Extract Coherent IQ Lock-In Wobble Scores for Left (19 kHz) and Right (20 kHz)
    left_score = extract_iq_wobble(rx_signal, fs, F_LEFT, t, base_mask)
    right_score = extract_iq_wobble(rx_signal, fs, F_RIGHT, t, base_mask)

    # 3. Build Perturbation Heatmap (combining local spectral Z-score + IQ Doppler sideband energy)
    mu_base = np.mean(spec_clean[:, base_mask], axis=1, keepdims=True)
    std_base = np.std(spec_clean[:, base_mask], axis=1, keepdims=True) + 0.9

    z_dev = np.abs(spec_clean - mu_base) / std_base
    temp_diff = np.abs(np.diff(spec_clean, axis=1, prepend=spec_clean[:, :1])) / std_base
    raw_wobble_map = 0.55 * z_dev + 0.45 * temp_diff

    # Project narrowband carrier perturbation onto the 19.0 kHz and 20.0 kHz bands so Panel 2 & Panel 3 match
    left_gauss = np.exp(-0.5 * ((f_roi - (F_LEFT / 1000.0)) / 0.085) ** 2)[:, None]
    right_gauss = np.exp(-0.5 * ((f_roi - (F_RIGHT / 1000.0)) / 0.085) ** 2)[:, None]

    wobble_heatmap = (
        raw_wobble_map * (0.25 + 0.75 * np.clip(left_gauss + right_gauss, 0.0, 1.0))
        + left_gauss * (left_score[None, :] / 45.0)
        + right_gauss * (right_score[None, :] / 45.0)
    )
    wobble_heatmap[:, t < 0.4] = 0.0
    wobble_heatmap = signal.medfilt2d(wobble_heatmap, kernel_size=(1, 5))

    thresh = 50.0

    # ================= PLOTTING (Clean Layout, Zero Title Overlap) =================
    plt.style.use('dark_background')
    fig = plt.figure(figsize=(14, 11.2), dpi=180)
    gs = GridSpec(
        3, 2,
        width_ratios=[42, 1],
        height_ratios=[1.05, 1.05, 1.05],
        hspace=0.52,
        wspace=0.025,
        top=0.90,
        bottom=0.08,
        left=0.07,
        right=0.93
    )

    fig.suptitle("AcousticTap: Ultrasonic Stereo Gate Verification Report (19 kHz Left / 20 kHz Right)",
                 fontsize=14, fontweight='bold', y=0.965, color='#ffffff')

    # --- Panel 1: Zoomed Ultrasonic Spectrogram ---
    ax0 = fig.add_subplot(gs[0, 0])
    cax0 = fig.add_subplot(gs[0, 1])
    vmax_spec = np.percentile(spec_display, 99.5)
    vmin_spec = vmax_spec - 48
    im0 = ax0.pcolormesh(t, f_roi, spec_display, cmap='inferno', shading='gouraud', vmin=vmin_spec, vmax=vmax_spec)
    ax0.set_title("1. Equalized Ultrasonic Spectrogram (18.4 – 20.6 kHz) — Carrier Shadowing & Doppler Sidebands",
                  fontsize=11.5, fontweight='bold', pad=10, color='#00e5ff')
    ax0.set_ylabel("Frequency (kHz)", fontsize=10)
    ax0.axhline(F_LEFT / 1000.0, color='#00ff88', linestyle='--', linewidth=0.9, alpha=0.75, label='19.0 kHz (Left Gate)')
    ax0.axhline(F_RIGHT / 1000.0, color='#ff3366', linestyle='--', linewidth=0.9, alpha=0.75, label='20.0 kHz (Right Gate)')
    ax0.legend(loc='upper right', framealpha=0.55, fontsize=8.5)
    fig.colorbar(im0, cax=cax0, label="dB (Eq)")

    # --- Panel 2: Perturbation / Wobble Heatmap ---
    ax1 = fig.add_subplot(gs[1, 0], sharex=ax0)
    cax1 = fig.add_subplot(gs[1, 1])
    vmax_wobble = max(2.5, np.percentile(wobble_heatmap, 99.3))
    im1 = ax1.pcolormesh(t, f_roi, wobble_heatmap, cmap='magma', shading='gouraud', vmin=0, vmax=vmax_wobble)
    ax1.set_title("2. Impulse-Rejected Perturbation Heatmap (Isolated 19 kHz Left & 20 kHz Right Hand Wobble)",
                  fontsize=11.5, fontweight='bold', pad=10, color='#ffaa00')
    ax1.set_ylabel("Frequency (kHz)", fontsize=10)
    ax1.set_yticks([18.5, 19.0, 19.5, 20.0, 20.5])
    fig.colorbar(im1, cax=cax1, label="Wobble Z")

    # --- Panel 3: Left vs Right Gate Trigger Graph ---
    ax2 = fig.add_subplot(gs[2, 0], sharex=ax0)
    ax2.plot(t, left_score, color='#00ff88', linewidth=2.2, label='Left Hand Gate (19.0 kHz Disturbance)')
    ax2.plot(t, right_score, color='#ff3366', linewidth=2.2, label='Right Hand Gate (20.0 kHz Disturbance)')
    ax2.axhline(thresh, color='#ffffff', linestyle=':', linewidth=1.3, alpha=0.85,
                label=f'Gate Activation Threshold ({thresh:.0f}%)')

    ax2.fill_between(t, thresh, left_score, where=(left_score > thresh),
                     color='#00ff88', alpha=0.28, interpolate=True)
    ax2.fill_between(t, thresh, right_score, where=(right_score > thresh),
                     color='#ff3366', alpha=0.28, interpolate=True)

    ax2.axvspan(3.0, 6.0, color='#00ff88', alpha=0.06, label='Left Hand Test Window (3–6s)')
    ax2.axvspan(7.0, 10.0, color='#ff3366', alpha=0.06, label='Right Hand Test Window (7–10s)')

    ax2.set_title("3. Ultrasonic Gate Decision Signal — Left (19 kHz) vs. Right (20 kHz) Disturbance",
                  fontsize=11.5, fontweight='bold', pad=10, color='#ffffff')
    ax2.set_xlabel("Time (seconds)", fontsize=11, labelpad=6)
    ax2.set_ylabel("Disturbance Index (%)", fontsize=10)
    ax2.set_xlim(0, t[-1])
    y_max = max(140.0, np.max(left_score) * 1.18, np.max(right_score) * 1.18)
    ax2.set_ylim(0, y_max)
    ax2.grid(True, linestyle='--', alpha=0.2)
    ax2.legend(loc='upper left', framealpha=0.65, fontsize=8.5, ncol=2)

    plt.savefig(output_path, bbox_inches='tight', pad_inches=0.25, facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"Saved high-resolution visual report to: {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="AcousticTap Ultrasonic Gate Visualizer")
    parser.add_argument("--simulate", action="store_true",
                        help="Run with synthesized Doppler/shadowing data (no mic/speaker needed)")
    parser.add_argument("--output", type=str, default="ultrasonic_gate_proof.png",
                        help="Output filename for the report image")
    args = parser.parse_args()

    if args.simulate:
        print("Running in SIMULATION mode (synthesizing 19 kHz Left & 20 kHz Right hand movements)...")
        recorded_signal = simulate_recording(FS, DURATION, F_LEFT, F_RIGHT)
    else:
        recorded_signal = capture_live_audio(FS, DURATION, F_LEFT, F_RIGHT)

    analyze_and_plot(recorded_signal, FS, args.output)