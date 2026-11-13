from pathlib import Path
from typing import Any
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"

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

# Directory Paths
DATA_DIR = PROJECT_ROOT / "dataset"
MODEL_DIR = PROJECT_ROOT / "models"
VERSIONS_DIR = MODEL_DIR / "versions"
RESULTS_DIR = MODEL_DIR / "test_results"
MODEL_PATH = MODEL_DIR / "acoustictap_model.joblib"

MODEL_DIR.mkdir(parents=True, exist_ok=True)
VERSIONS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Audio Constants
AUDIO_CFG = CFG.get("audio", {})
SAMPLE_RATE = int(AUDIO_CFG.get("sample_rate", 44100))
BLOCK_SIZE = int(AUDIO_CFG.get("block_size", 512))
PRE_BLOCKS = int(AUDIO_CFG.get("pre_blocks", 3))
POST_BLOCKS = int(AUDIO_CFG.get("post_blocks", 8))
COOLDOWN_BLOCKS = int(AUDIO_CFG.get("cooldown_blocks", 25))
CHANNELS = int(AUDIO_CFG.get("channels", 1))

# Detector Constants
DET_CFG = CFG.get("detector", {})
RMS_THRESHOLD = float(DET_CFG.get("rms_threshold", 0.02))
PEAK_MULTIPLIER = float(DET_CFG.get("peak_multiplier", 1.8))
BG_SNR_RATIO = float(DET_CFG.get("bg_snr_ratio", 2.3))
BG_EMA_ALPHA = float(DET_CFG.get("bg_ema_alpha", 0.95))

# Fusion Constants
FUSION_CFG = CFG.get("hybrid_fusion", {})
FOURIER_WEIGHT = float(FUSION_CFG.get("fourier_weight", 0.50))

# Training Constants
TRAIN_CFG = CFG.get("training", {})
CLASSES = list(TRAIN_CFG.get("classes", ["left_tap", "right_tap", "noise"]))

# Geofence Constants
GEOFENCE_CFG = CFG.get("geofence", {
    "enabled": True,
    "min_hf_lf_ratio_db": -14.0,
    "min_attack_decay_ratio": 1.85,
    "min_crest_factor": 4.2,
    "dispersion_window_ms": [-5.5, 2.5],
    "mahalanobis_threshold": 5.90,
    "surface_tolerance_scale": 1.15,
    "class_conditional": True,
})
