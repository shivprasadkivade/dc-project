from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

SAMPLE_RATE = 44100
BLOCK_SIZE = 512          # ~11.6 ms per audio block
PRE_BLOCKS = 2            # ~23.2 ms captured BEFORE the impulse trigger
POST_BLOCKS = 8           # ~92.8 ms captured AFTER the impulse trigger
COOLDOWN_BLOCKS = 25      # ~290 ms cooldown between taps
RMS_THRESHOLD = 0.015     # Base trigger threshold

# Dynamic noise floor and crest factor settings
PEAK_MULTIPLIER = 1.8     # Peak must exceed rms * peak_multiplier (crest check)
BG_SNR_RATIO = 2.0        # RMS must exceed rolling background floor * bg_snr_ratio
BG_EMA_ALPHA = 0.95       # Rolling background floor exponential smoothing

# Transient alignment window constants
WINDOW_PRE_PEAK = 32      # Samples before peak included in active window
WINDOW_WAKE = 1024        # Active impact transient and resonance window (1024 samples ≈ 23.2 ms)

CFG = load_config()

# Directory Paths
DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
VERSIONS_DIR = MODEL_DIR / "versions"
RESULTS_DIR = MODEL_DIR / "test_results"
MODEL_PATH = MODEL_DIR / "acoustictap_model.joblib"

# Backwards-compatible top-level constants read from config.yaml
SAMPLE_RATE = int(CFG["audio"]["sample_rate"])
BLOCK_SIZE = int(CFG["audio"]["block_size"])
PRE_BLOCKS = int(CFG["audio"]["pre_blocks"])
POST_BLOCKS = int(CFG["audio"]["post_blocks"])
COOLDOWN_BLOCKS = int(CFG["audio"]["cooldown_blocks"])
RMS_THRESHOLD = float(CFG["detector"]["rms_threshold"])
FOURIER_WEIGHT = float(CFG["hybrid_fusion"]["fourier_weight"])
CLASSES = list(CFG["training"]["classes"])
