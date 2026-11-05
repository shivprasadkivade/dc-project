import time
import platform
import argparse
from pathlib import Path
import numpy as np
import sounddevice as sd
import joblib
from pynput import keyboard
from .config import CFG, SAMPLE_RATE, BLOCK_SIZE, RMS_THRESHOLD, MODEL_PATH
from .detector import ImpulseDetector, get_input_channels
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector

def run_listener(model_file: Path, threshold: float | None, confidence_min: float | None, dry_run: bool):
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}. Run train first.")

def run_listener(model_file: Path, threshold: float, confidence_min: float, dry_run: bool):
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}. Run train first.")

    saved = joblib.load(model_file)
    model = saved["model"]
    triangulator = saved.get("triangulator", None)
    fourier_weight = saved.get("fourier_weight", CFG["hybrid_fusion"]["fourier_weight"])
    model_name = saved.get("model_name", "HybridModel")
    feature_set = saved["feature_set"]
    classes = saved["classes"]
    channels = get_input_channels()

    lis_cfg = CFG.get("listener", {})
    default_conf = float(confidence_min if confidence_min is not None else lis_cfg.get("default_confidence", 0.65))
    class_thresholds = lis_cfg.get("class_thresholds", {})
    typing_suppression_window = float(lis_cfg.get("typing_suppression_seconds", 0.15))

    cfg_mod = str(lis_cfg.get("os_modifier_key", "auto")).lower()
    is_macos = platform.system() == "Darwin"
    if cfg_mod == "cmd" or (cfg_mod == "auto" and is_macos):
        mod_key = keyboard.Key.cmd
        mod_name = "Cmd"
    else:
        mod_key = keyboard.Key.ctrl
        mod_name = "Ctrl"

    shortcuts_map = lis_cfg.get("shortcuts", {"left_tap": "c", "right_tap": "v"})

    is_macos = platform.system() == "Darwin"
    mod_key = keyboard.Key.cmd if is_macos else keyboard.Key.ctrl
    mod_name = "Cmd" if is_macos else "Ctrl"

    kb_controller = keyboard.Controller()
    last_keypress_time = 0.0
    typing_suppression_window = 0.15
    is_emitting_synthetic = False

    def on_press(key):
        nonlocal last_keypress_time
        if not is_emitting_synthetic:
            last_keypress_time = time.time()

    kb_listener = keyboard.Listener(on_press=on_press)
    kb_listener.start()

    def trigger_macro(action_label: str):
        nonlocal is_emitting_synthetic
        if dry_run:
            print(f"  [DRY-RUN] Would have triggered shortcut for: {action_label} ({mod_name}+C/V)")
            return

        is_emitting_synthetic = True
        try:
            if action_label == "left_tap":
                print(f"  >>> ACTION: Left Palm Rest -> Copy ({mod_name}+C)")
                with kb_controller.pressed(mod_key):
                    kb_controller.tap("c")
            elif action_label == "right_tap":
                print(f"  >>> ACTION: Right Palm Rest -> Paste ({mod_name}+V)")
                with kb_controller.pressed(mod_key):
                    kb_controller.tap("v")
        finally:
            time.sleep(0.02)
            is_emitting_synthetic = False

    detector = ImpulseDetector(threshold=threshold)
    mode_label = "DRY-RUN TEST MODE" if dry_run else f"LIVE MACRO MODE ({mod_name.upper()})"
    print(f"\n[LISTEN - {mode_label}] Loaded {model_name} ({feature_set} features) from {model_file.name}")
    print(f"OS: {platform.system()} (Modifier: {mod_name}) | Confidence Threshold: {confidence_min:.2f}")
    print("Tap left or right palm rest. Press Ctrl+C to stop.\n")

    with sd.InputStream(
        samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=channels, callback=detector.audio_callback
    ):
        while True:
            waveform = detector.event_queue.get()

            if (time.time() - last_keypress_time) < typing_suppression_window:
                print("  [Ignored: Active keyboard typing]")
                continue

            t0 = time.perf_counter()
            ml_feats = extract_features(waveform, feature_set=feature_set).reshape(1, -1)
            p_ml = model.predict_proba(ml_feats)[0]

            if triangulator is not None:
                fourier_feats = extract_fourier_triangulation_vector(waveform).reshape(1, -1)
                p_fourier = triangulator.predict_proba(fourier_feats)[0]
                probs = (1.0 - fourier_weight) * p_ml + fourier_weight * p_fourier
                fourier_top = classes[int(np.argmax(p_fourier))]
                ml_top = classes[int(np.argmax(p_ml))]
                breakdown = f"ML:{ml_top}({np.max(p_ml):.2f}) + FFT:{fourier_top}({np.max(p_fourier):.2f})"
            else:
                probs = p_ml
                breakdown = "ML-Only"

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_label = classes[pred_idx]
            latency_ms = (time.perf_counter() - t0) * 1000

            req_conf = float(class_thresholds.get(pred_label, default_conf))
            print(f"Pred: {pred_label:10s} | Conf: {confidence:.2f} (Req: {req_conf:.2f}) [{breakdown}] | {latency_ms:.1f} ms")

            if pred_label != "noise" and confidence >= req_conf:
                trigger_macro(pred_label)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test or run a trained AcousticTap model")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to a .joblib model file")
    parser.add_argument("--threshold", type=float, default=RMS_THRESHOLD, help="Mic RMS trigger threshold")
    parser.add_argument("--confidence", type=float, default=0.75, help="Minimum confidence to trigger")
    parser.add_argument("--dry-run", action="store_true", help="Test predictions without firing shortcuts")
    args = parser.parse_args()
    run_listener(args.model, args.threshold, args.confidence, args.dry_run)
