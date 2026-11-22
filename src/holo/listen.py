from __future__ import annotations

import time
import queue
import platform
import argparse
from pathlib import Path
from typing import Optional
import numpy as np
import sounddevice as sd
import joblib
from pynput import keyboard
from .config import (
    CFG,
    SAMPLE_RATE,
    BLOCK_SIZE,
    RMS_THRESHOLD,
    MODEL_PATH,
    SPLIT_INPUT,
    TAP_DEVICE,
    SECONDARY_DEVICE,
    CAPTURE_SECONDARY,
    AUTO_RECONNECT,
)
from .detector import ImpulseDetector, get_input_channels
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector
from .audio_manager import SplitAudioEngine, print_audio_devices

def run_listener(
    model_file: Path = MODEL_PATH,
    threshold: Optional[float] = None,
    confidence_min: Optional[float] = None,
    dry_run: bool = False,
    no_geofence: bool = False,
    tap_device: Optional[str | int] = None,
    secondary_device: Optional[str | int] = None,
    no_split: bool = False,
    no_secondary_capture: bool = False,
):
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}. Run 'uv run python -m src.holo.train' first.")

    saved = joblib.load(model_file)
    model = saved["model"]
    triangulator = saved.get("triangulator", None)
    geofence = saved.get("geofence", None)
    fourier_weight = float(saved.get("fourier_weight", CFG.get("hybrid_fusion", {}).get("fourier_weight", 0.50)))
    model_name = saved.get("model_name", "HybridModel")
    feature_set = saved.get("feature_set", "enhanced")
    classes = saved.get("classes", ["left_tap", "right_tap", "noise"])
    channels = get_input_channels()

    lis_cfg = CFG.get("listener", {})
    default_conf = float(confidence_min if confidence_min is not None else lis_cfg.get("default_confidence", 0.65))
    class_thresholds = lis_cfg.get("class_thresholds", {})
    typing_suppression_window = float(lis_cfg.get("typing_suppression_seconds", 0.15))

    cfg_mod = str(lis_cfg.get("os_modifier_key", "auto")).lower()
    is_macos = (platform.system() == "Darwin")
    if cfg_mod == "cmd" or (cfg_mod == "auto" and is_macos):
        mod_key = keyboard.Key.cmd
        mod_name = "Cmd"
    else:
        mod_key = keyboard.Key.ctrl
        mod_name = "Ctrl"

    kb_controller = keyboard.Controller()
    last_keypress_time = 0.0
    is_emitting_synthetic = False

    def on_press(key):
        nonlocal last_keypress_time
        if not is_emitting_synthetic:
            last_keypress_time = time.time()

    try:
        kb_listener = keyboard.Listener(on_press=on_press)
        kb_listener.start()
    except Exception as e:
        print(f"[Warning] Could not initialize keyboard listener for typing suppression: {e}")

    def trigger_macro(action_label: str):
        nonlocal is_emitting_synthetic
        if dry_run:
            print(f"      [DRY-RUN] Would trigger shortcut for: {action_label} ({mod_name}+C/V)")
            return

        is_emitting_synthetic = True
        try:
            if "left" in action_label:
                print(f"      >>> ACTION: Left Palm Rest -> Copy ({mod_name}+C)")
                with kb_controller.pressed(mod_key):
                    kb_controller.tap("c")
            elif "right" in action_label:
                print(f"      >>> ACTION: Right Palm Rest -> Paste ({mod_name}+V)")
                with kb_controller.pressed(mod_key):
                    kb_controller.tap("v")
        except Exception as ex:
            print(f"      [Error] Failed to emit keystroke: {ex}")
        finally:
            time.sleep(0.02)
            is_emitting_synthetic = False

    effective_thresh = float(threshold if threshold is not None else RMS_THRESHOLD)
    detector = ImpulseDetector(threshold=effective_thresh)
    mode_label = "DRY-RUN TEST MODE" if dry_run else f"LIVE MACRO MODE ({mod_name.upper()})"

    if no_geofence:
        gf_status = "DISABLED via --no-geofence"
    elif geofence is not None:
        mode_str = "Supervised Boundary Discriminator (Inside vs Outside)" if geofence.has_outside_data else "Spatial Manifold"
        gf_status = f"ACTIVE ({mode_str}, Thresh: {geofence.boundary_threshold:.2f})"
    else:
        gf_status = "NOT PRESENT (Legacy v2 model)"

    print("\n" + "=" * 70)
    print(f"[LISTEN - {mode_label}] Loaded {model_name} from {model_file.name}")
    print(f"OS: {platform.system()} (Modifier: {mod_name}) | Confidence Threshold: {default_conf:.2f}")
    print(f"Passive Acoustic Geofence: {gf_status}")
    print("Tap left or right palm rest. Knock desk to test boundary rejection.")
    print("Press Ctrl+C to stop.\n" + "=" * 70 + "\n")

    effective_tap_dev = tap_device if tap_device is not None else TAP_DEVICE
    effective_sec_dev = secondary_device if secondary_device is not None else SECONDARY_DEVICE
    capture_sec = (not no_secondary_capture) and CAPTURE_SECONDARY

    if not no_split:
        audio_ctx = SplitAudioEngine(
            tap_callback=detector.audio_callback,
            tap_device_pref=effective_tap_dev,
            secondary_device_pref=effective_sec_dev,
            capture_secondary=capture_sec,
            auto_reconnect=AUTO_RECONNECT,
            sample_rate=SAMPLE_RATE,
            block_size=BLOCK_SIZE,
            channels=channels,
        )
    else:
        # Legacy fallback mode
        audio_ctx = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=channels, callback=detector.audio_callback
        )

    with audio_ctx:
        if hasattr(audio_ctx, "print_summary"):
            audio_ctx.print_summary()

        while True:
            try:
                waveform = detector.event_queue.get(timeout=1.0)
            except queue.Empty:
                if hasattr(audio_ctx, "ensure_active"):
                    audio_ctx.ensure_active()
                continue

            # Active keyboard typing suppression
            if (time.time() - last_keypress_time) < typing_suppression_window:
                print("  [Ignored: Active keyboard typing]")
                continue

            t0 = time.perf_counter()
            ml_1d = extract_features(waveform, feature_set=feature_set)
            fourier_1d = (
                extract_fourier_triangulation_vector(waveform)
                if triangulator is not None
                else np.zeros(43, dtype=np.float64)
            )

            # -----------------------------------------------------------------
            # STEP 1: Passive Acoustic Geofence Boundary Check (FIRST!)
            # Evaluates whether the transient came from INSIDE or OUTSIDE the boundary
            # -----------------------------------------------------------------
            if geofence is not None and not no_geofence:
                gf_decision = geofence.verify(ml_1d, fourier_1d)
                if not gf_decision.passed:
                    latency_ms = (time.perf_counter() - t0) * 1000.0
                    print(
                        f"  [GEOFENCE: OUTSIDE BOUNDARY] Blocked external impact "
                        f"({gf_decision.reason}) | {latency_ms:.1f} ms"
                    )
                    continue
                gf_tag = f"[GEOFENCE: INSIDE (prob={gf_decision.prob_inside:.2f}, d_in={gf_decision.dist_inside:.2f}/{gf_decision.threshold:.2f})]"
            else:
                gf_tag = "[NO-GEOFENCE]"

            # -----------------------------------------------------------------
            # STEP 2: Gesture Classification (ONLY for verified in-bounds taps!)
            # -----------------------------------------------------------------
            ml_feats = ml_1d.reshape(1, -1)
            p_ml = model.predict_proba(ml_feats)[0]

            if triangulator is not None:
                fourier_feats = fourier_1d.reshape(1, -1)
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
            latency_ms = (time.perf_counter() - t0) * 1000.0

            req_conf = float(class_thresholds.get(pred_label, default_conf))
            if geofence is not None and not no_geofence:
                if gf_decision.prob_inside < (geofence.boundary_threshold + 0.12):
                    req_conf = min(0.92, req_conf + 0.10)

            print(f"  {gf_tag} Pred: {pred_label:10s} | Conf: {confidence:.2f} (Req: {req_conf:.2f}) [{breakdown}] | {latency_ms:.1f} ms")

            if pred_label != "noise" and confidence >= req_conf:
                trigger_macro(pred_label)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Test or run trained AcousticTap with Split-Input Audio & Geofencing")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to a .joblib model file")
    parser.add_argument("--threshold", type=float, default=RMS_THRESHOLD, help="Mic RMS trigger threshold")
    parser.add_argument("--confidence", type=float, default=0.65, help="Minimum confidence to trigger")
    parser.add_argument("--dry-run", action="store_true", help="Test predictions without firing shortcuts")
    parser.add_argument("--no-geofence", action="store_true", help="Disable passive acoustic geofence spatial verification")
    parser.add_argument("--tap-device", type=str, default=None, help="Device name or index for Point 1 (MacBook mic)")
    parser.add_argument("--secondary-device", type=str, default=None, help="Device name or index for Point 2 (Earphones/Headset)")
    parser.add_argument("--no-secondary-capture", action="store_true", help="Do not capture secondary mic (leave earphones purely for system/music)")
    parser.add_argument("--no-split", action="store_true", help="Disable audio splitting and use default system input")
    parser.add_argument("--list-devices", action="store_true", help="List all audio devices with split input roles and exit")
    return parser

if __name__ == "__main__":
    parser = build_parser()
    args = parser.parse_args()
    if args.list_devices:
        print_audio_devices()
    else:
        run_listener(
            model_file=args.model,
            threshold=args.threshold,
            confidence_min=args.confidence,
            dry_run=args.dry_run,
            no_geofence=args.no_geofence,
            tap_device=args.tap_device,
            secondary_device=args.secondary_device,
            no_split=args.no_split,
            no_secondary_capture=args.no_secondary_capture,
        )
