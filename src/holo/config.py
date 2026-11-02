from __future__ import annotations

import os
from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]

def find_config_path() -> Path:
    candidates = [
        PROJECT_ROOT / "config" / "config.yaml",
        PROJECT_ROOT / "config.yaml",
        Path.cwd() / "config" / "config.yaml",
        Path.cwd() / "config.yaml",
    ]
    for p in candidates:
        if p.is_file():
            return p
    return PROJECT_ROOT / "config" / "config.yaml"

CONFIG_PATH = find_config_path()

def load_config(path: Path | str | None = None) -> dict[str, Any]:
    cfg_path = Path(path) if path is not None else CONFIG_PATH
    if cfg_path.is_file():
        try:
            with open(cfg_path, "r", encoding="utf-8") as f:
                data = yaml.safe_load(f)
                if isinstance(data, dict):
                    return data
        except Exception as e:
            print(f"[Config] Warning: Failed to read {cfg_path}: {e}")
    return {}

CFG = load_config()

# Data & Model Directories
DATA_DIR = PROJECT_ROOT / "dataset"
if not DATA_DIR.exists() and (PROJECT_ROOT / "data").exists():
    DATA_DIR = PROJECT_ROOT / "data"

MODEL_DIR = PROJECT_ROOT / "models"
VERSIONS_DIR = MODEL_DIR / "versions"
RESULTS_DIR = MODEL_DIR / "test_results"
MODEL_PATH = MODEL_DIR / "acoustictap_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Audio Constants
AUDIO_CFG = CFG.get("audio", {})
SAMPLE_RATE: int = int(AUDIO_CFG.get("sample_rate", 48000))
BLOCK_SIZE: int = int(AUDIO_CFG.get("block_size", 512))
PRE_BLOCKS: int = int(AUDIO_CFG.get("pre_blocks", 4))
POST_BLOCKS: int = int(AUDIO_CFG.get("post_blocks", 8))
COOLDOWN_BLOCKS: int = int(AUDIO_CFG.get("cooldown_blocks", 25))
CHANNELS: int = int(AUDIO_CFG.get("channels", 1))
LOWPASS_CUTOFF_HZ: float = float(AUDIO_CFG.get("lowpass_cutoff_hz", 16000.0))

# Detector Constants
DET_CFG = CFG.get("detector", {})
RMS_THRESHOLD: float = float(DET_CFG.get("rms_threshold", 0.008))
PEAK_MULTIPLIER: float = float(DET_CFG.get("peak_multiplier", 1.8))
BG_SNR_RATIO: float = float(DET_CFG.get("bg_snr_ratio", 2.0))
BG_EMA_ALPHA: float = float(DET_CFG.get("bg_ema_alpha", 0.95))
CREST_THRESHOLD: float = float(DET_CFG.get("crest_threshold", 1.85))
DYNAMIC_MULTIPLIER: float = BG_SNR_RATIO

# Ultrasonic Constants
ULTRASONIC_CFG = CFG.get("ultrasonic", {})
ULTRASONIC_ENABLED: bool = bool(ULTRASONIC_CFG.get("enabled", True))
LEFT_ULTRA_HZ: float = float(ULTRASONIC_CFG.get("left_freq_hz", 19000.0))
RIGHT_ULTRA_HZ: float = float(ULTRASONIC_CFG.get("right_freq_hz", 20200.0))
SPEAKER_AMPLITUDE: float = float(ULTRASONIC_CFG.get("speaker_amplitude", 0.03))
CARRIER_BW_HZ: float = float(ULTRASONIC_CFG.get("carrier_bw_hz", 30.0))
DOPPLER_MIN_HZ: float = float(ULTRASONIC_CFG.get("doppler_min_hz", 20.0))
DOPPLER_MAX_HZ: float = float(ULTRASONIC_CFG.get("doppler_max_hz", 180.0))
ALPHA_DOPPLER: float = float(ULTRASONIC_CFG.get("alpha_doppler", 0.65))
GATE_THRESHOLD: float = float(ULTRASONIC_CFG.get("gate_threshold", 1.5))
GATE_STEEPNESS: float = float(ULTRASONIC_CFG.get("gate_steepness", 2.5))
CALIBRATION_BLOCKS: int = int(ULTRASONIC_CFG.get("calibration_blocks", 35))
RAMP_DURATION_MS: float = float(ULTRASONIC_CFG.get("ramp_duration_ms", 60.0))

# Fusion Constants
FUSION_CFG = CFG.get("hybrid_fusion", {})
FOURIER_WEIGHT: float = float(FUSION_CFG.get("fourier_weight", 0.50))
AUTO_TUNE_WEIGHT: bool = bool(FUSION_CFG.get("auto_tune_weight", True))
ULTRASONIC_2FA_ENABLED: bool = bool(FUSION_CFG.get("ultrasonic_2fa_enabled", True))
WEIGHT_SEARCH_GRID: list[float] = [float(w) for w in FUSION_CFG.get("weight_search_grid", [0.20, 0.30, 0.40, 0.50, 0.60, 0.70])]

# Training & Classes
TRAIN_CFG = CFG.get("training", {})
CLASSES: list[str] = list(TRAIN_CFG.get("classes", ["left_tap", "right_tap", "noise"]))
CV_FOLDS: int = int(TRAIN_CFG.get("cv_folds", 5))
TEST_SIZE: float = float(TRAIN_CFG.get("test_size", 0.20))
RANDOM_SEED: int = int(TRAIN_CFG.get("random_seed", 42))

# Listener Constants
LIS_CFG = CFG.get("listener", {})
CONFIDENCE_THRESHOLD: float = float(LIS_CFG.get("default_confidence", 0.60))
ENABLE_SHORTCUTS: bool = bool(LIS_CFG.get("enable_shortcuts", False))
