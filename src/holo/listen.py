from __future__ import annotations

import argparse
from pathlib import Path
import platform
import time
from typing import Any
import joblib
import numpy as np
from pynput import keyboard
import sounddevice as sd

try:
    from .config import (
        BLOCK_SIZE,
        CALIBRATION_BLOCKS,
        CFG,
        CLASSES,
        CONFIDENCE_THRESHOLD,
        ENABLE_SHORTCUTS,
        FOURIER_WEIGHT,
        MODEL_PATH,
        RMS_THRESHOLD,
        SAMPLE_RATE,
        ULTRASONIC_2FA_ENABLED,
        ULTRASONIC_ENABLED,
    )
    from .detector import ImpulseDetector, get_input_channels
    from .features import extract_features
    from .triangulation import FourierTriangulator, extract_fourier_triangulation_vector
    from .ultrasonic import UltrasonicSonarGate
except (ImportError, ValueError):
    from config import (  # type: ignore
        BLOCK_SIZE,
        CALIBRATION_BLOCKS,
        CFG,
        CLASSES,
        CONFIDENCE_THRESHOLD,
        ENABLE_SHORTCUTS,
        FOURIER_WEIGHT,
        MODEL_PATH,
        RMS_THRESHOLD,
        SAMPLE_RATE,
        ULTRASONIC_2FA_ENABLED,
        ULTRASONIC_ENABLED,
    )
    from detector import ImpulseDetector, get_input_channels  # type: ignore
    from features import extract_features  # type: ignore
    from triangulation import FourierTriangulator, extract_fourier_triangulation_vector  # type: ignore
    from ultrasonic import UltrasonicSonarGate  # type: ignore


def run_listener(
    model_file: Path = MODEL_PATH,
    threshold: float | None = None,
    confidence_min: float | None = None,
    dry_run: bool = False,
    enable_ultrasonic: bool = ULTRASONIC_ENABLED,
) -> None:
    """
    Real-time AcousticTap V3 listener daemon:
    - Runs continuous stereo ultrasonic carrier generation & active reflection sonar gate.
    - Monitors acoustic impact transients with pre-trigger ring buffer and 16 kHz LPF.
    - Fuses ML scikit-learn classifier with Fourier-domain acoustic triangulation.
    - Requires two-factor touch verification (Acoustic Impact + Ultrasonic Hand Proximity)
      before triggering OS keyboard shortcuts.
    """
    if not model_file.is_file():
        raise FileNotFoundError(f"Model file not found at {model_file}. Run train first.")

    bundle: dict[str, Any] = joblib.load(model_file)
    model = bundle["model"]
    classes = list(bundle.get("classes", CLASSES))
    w_four = float(bundle.get("fourier_weight", FOURIER_WEIGHT))
    feature_set = bundle.get("feature_set", "enhanced")
    expected_dim = bundle.get("feature_dim", getattr(model, "n_features_in_", None))

    tri_data = bundle.get("triangulator")
    if isinstance(tri_data, dict):
        triangulator = FourierTriangulator.from_dict(tri_data)
    else:
        triangulator = tri_data

    # Feature dimension verification
    dummy_input = np.zeros(2048, dtype=np.float32)
    dummy_feats = extract_features(dummy_input, sr=SAMPLE_RATE, feature_set=feature_set)
    if expected_dim is not None and dummy_feats.shape[0] != expected_dim:
        raise ValueError(
            f"Feature dimension mismatch: extract_features produced {dummy_feats.shape[0]} features, "
            f"but model expects {expected_dim}"
        )

    lis_cfg = CFG.get("listener", {})
    default_conf = float(confidence_min if confidence_min is not None else lis_cfg.get("default_confidence", CONFIDENCE_THRESHOLD))
    class_thresholds = lis_cfg.get("class_thresholds", {"left_tap": 0.58, "right_tap": 0.65})
    typing_suppression_window = float(lis_cfg.get("typing_suppression_seconds", 0.15))
    should_fire_shortcuts = ENABLE_SHORTCUTS and not dry_run

    # OS modifier keys
    cfg_mod = str(lis_cfg.get("os_modifier_key", "auto")).lower()
    is_macos = platform.system() == "Darwin"
    if cfg_mod == "cmd" or (cfg_mod == "auto" and is_macos):
        mod_key = keyboard.Key.cmd
        mod_name = "Cmd"
    else:
        mod_key = keyboard.Key.ctrl
        mod_name = "Ctrl"

    shortcuts_map = lis_cfg.get("shortcuts", {"left_tap": "c", "right_tap": "v"})
    kb_controller = keyboard.Controller()
    last_keypress_time = 0.0
    is_emitting_synthetic = False

    def on_press(key: Any) -> None:
        nonlocal last_keypress_time
        if not is_emitting_synthetic:
            last_keypress_time = time.time()

    kb_listener = keyboard.Listener(on_press=on_press)
    kb_listener.daemon = True
    kb_listener.start()

    def trigger_macro(action_label: str) -> None:
        nonlocal is_emitting_synthetic
        target_key = shortcuts_map.get(action_label)
        if not target_key:
            return
        if not should_fire_shortcuts:
            print(f"  [DRY-RUN] Would trigger {mod_name}+{target_key.upper()} for {action_label}")
            return

        print(f"  >>> ACTION: {action_label} -> {mod_name}+{target_key.upper()}")
        is_emitting_synthetic = True
        try:
            with kb_controller.pressed(mod_key):
                kb_controller.tap(target_key)
        finally:
            time.sleep(0.02)
            is_emitting_synthetic = False

    channels = get_input_channels()
    detector = ImpulseDetector(threshold=threshold, sr=SAMPLE_RATE)
    sonar = UltrasonicSonarGate(sr=SAMPLE_RATE)

    print("\n" + "=" * 70)
    print(f" AcousticTap V3 Listener: Live Touch Sensing Daemon (OS: {platform.system()})")
    print(f" Model: {bundle.get('model_name', 'HybridModel')} | Dimension: {expected_dim} features")
    print(f" Decision Fusion: {int((1-w_four)*100)}% ML + {int(w_four*100)}% Fourier Triangulation")
    print(f" 2FA Sonar Gate: {'ENABLED' if (enable_ultrasonic and ULTRASONIC_2FA_ENABLED) else 'DISABLED'}")
    print(f" Shortcuts: {'ENABLED' if should_fire_shortcuts else 'DISABLED (Dry Run)'} | Mod: {mod_name}")
    print("=" * 70)

    # Start ultrasonic carrier emitter
    if enable_ultrasonic:
        sonar.start_emitter()
        print("\n[Sonar] Ultrasonic emitter active (19.0 kHz L / 20.2 kHz R).")
        print("[Sonar] Calibrating idle baseline. Keep hands off palm rests...")

    last_trigger_time = 0.0
    debounce_cooldown = 0.28  # seconds

    # Use unified duplex stream or independent input stream
    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=channels,
            callback=detector.audio_callback,
        ):
            if enable_ultrasonic:
                # Collect baseline quiet blocks
                while len(detector.calibration_buffer) < CALIBRATION_BLOCKS:
                    time.sleep(0.04)
                sonar.calibrate_baseline(list(detector.calibration_buffer))
                print("[Sonar] Baseline calibration locked! System ready for taps.\n")
            else:
                print("[System] Ready for taps.\n")

            while True:
                try:
                    raw_waveform = detector.event_queue.get(timeout=0.1)
                except Exception:
                    continue

                now = time.time()
                # Cooldown debounce check
                if (now - last_trigger_time) < debounce_cooldown:
                    continue

                # Keyboard typing suppression
                if (now - last_keypress_time) < typing_suppression_window:
                    print("  [Telemetry] Ignored: Active keyboard typing detected")
                    continue

                # 1. Acoustic ML Feature Prediction
                feats = extract_features(raw_waveform, sr=SAMPLE_RATE, feature_set=feature_set)
                p_ml = model.predict_proba(feats.reshape(1, -1))[0]

                # 2. Fourier Triangulation Score
                if triangulator is not None:
                    four_feats = extract_fourier_triangulation_vector(raw_waveform, sr=SAMPLE_RATE)
                    p_four = triangulator.predict_proba(four_feats.reshape(1, -1))[0]
                else:
                    p_four = p_ml

                # 3. Hybrid Fusion
                p_hyb = (1.0 - w_four) * p_ml + w_four * p_four
                probs = {cls_name: float(p_hyb[i]) for i, cls_name in enumerate(classes)}

                ml_top_cls = classes[int(np.argmax(p_ml))]
                ml_top_prob = float(np.max(p_ml))
                four_top_cls = classes[int(np.argmax(p_four))]
                four_top_prob = float(np.max(p_four))

                # 4. Ultrasonic Sonar Gate Check
                us = sonar.evaluate_window(raw_waveform)
                gate_l, gate_r = us["gate_left"], us["gate_right"]
                is_us_active = us["active"]

                # Two-Factor Touch Verification
                cand_cls = max(probs, key=lambda k: probs.get(k, 0.0))
                cand_conf = probs[cand_cls]
                req_conf = float(class_thresholds.get(cand_cls, default_conf))

                # Gate status string
                if is_us_active:
                    us_status_str = f"L={gate_l:.2f} R={gate_r:.2f} (Active)"
                else:
                    us_status_str = "Neutral (No Carrier)"

                # Verification decision
                gate_confirmed = True
                reject_reason = ""

                if cand_cls == "left_tap":
                    if is_us_active and ULTRASONIC_2FA_ENABLED and gate_l < 0.35:
                        gate_confirmed = False
                        reject_reason = "Left Ultrasonic Gate Blocked"
                elif cand_cls == "right_tap":
                    if is_us_active and ULTRASONIC_2FA_ENABLED and gate_r < 0.35:
                        gate_confirmed = False
                        reject_reason = "Right Ultrasonic Gate Blocked"

                # Telemetry formatting: [ML Prob | Fourier Score | Ultrasonic Gate | Final Decision]
                ml_str = f"ML: {ml_top_cls} {ml_top_prob:.2f}"
                four_str = f"Fourier: {four_top_cls} {four_top_prob:.2f}"
                us_str = f"US Gate: {us_status_str}"

                if cand_cls != "noise" and cand_conf >= req_conf:
                    if gate_confirmed:
                        decision_str = f"TRIGGER: {cand_cls.upper()} (Conf: {cand_conf:.2f} >= {req_conf:.2f})"
                        print(f"[{ml_str} | {four_str} | {us_str} | {decision_str}]")
                        trigger_macro(cand_cls)
                        last_trigger_time = now
                    else:
                        decision_str = f"REJECTED: {reject_reason}"
                        print(f"[{ml_str} | {four_str} | {us_str} | {decision_str}]")
                else:
                    decision_str = f"REJECTED: {cand_cls} (Conf: {cand_conf:.2f} < {req_conf:.2f})"
                    print(f"[{ml_str} | {four_str} | {us_str} | {decision_str}]")

    except KeyboardInterrupt:
        print("\n[Listen] Stopping listener...")
    finally:
        if enable_ultrasonic:
            sonar.stop_emitter()
            print("[Sonar] Ultrasonic emitter smoothly terminated.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run AcousticTap V3 real-time listener daemon")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to .joblib model")
    parser.add_argument("--threshold", type=float, default=None, help="Override RMS trigger threshold")
    parser.add_argument("--confidence", type=float, default=None, help="Override confidence threshold")
    parser.add_argument("--dry-run", action="store_true", help="Print decisions without triggering OS keystrokes")
    parser.add_argument("--no-ultrasonic", action="store_true", help="Disable active ultrasonic emitter")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_listener(
        model_file=args.model,
        threshold=args.threshold,
        confidence_min=args.confidence,
        dry_run=args.dry_run,
        enable_ultrasonic=not args.no_ultrasonic,
    )


if __name__ == "__main__":
    main()
