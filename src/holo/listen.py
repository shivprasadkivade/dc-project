from __future__ import annotations

import os
import sys
import time
import queue
import platform
import argparse
import subprocess
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

def show_toast(title: str, subtitle: str, message: str, sound: str = "Tink"):
    """Displays a native macOS Notification Center toast notification in a non-blocking subprocess."""
    if platform.system() != "Darwin":
        return
    try:
        clean_title = title.replace('"', '\\"').replace("\n", " ")
        clean_sub = subtitle.replace('"', '\\"').replace("\n", " ")
        clean_msg = message.replace('"', '\\"').replace("\n", " ")
        sound_clause = f' sound name "{sound}"' if sound else ""
        script = f'display notification "{clean_msg}" with title "{clean_title}" subtitle "{clean_sub}"{sound_clause}'
        subprocess.Popen(
            ["osascript", "-e", script],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
    except Exception:
        pass

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
    verbose: bool = False,
    no_notify: bool = False,
):
    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}. Run 'uv run python -m src.holo.train' first.")

    saved = joblib.load(model_file)
    model = saved["model"]
    triangulator = saved.get("triangulator", None)
    geofence = saved.get("geofence", None)
    fourier_weight = float(saved.get("fourier_weight", CFG.get("hybrid_fusion", {}).get("fourier_weight", 0.50)))
    classes = saved.get("classes", ["left_tap", "right_tap", "noise"])
    channels = get_input_channels()

    lis_cfg = CFG.get("listener", {})
    default_conf = float(confidence_min if confidence_min is not None else lis_cfg.get("default_confidence", 0.65))
    class_thresholds = lis_cfg.get("class_thresholds", {})
    typing_suppression_window = float(lis_cfg.get("typing_suppression_seconds", 0.15))
    enable_notifications = (not no_notify) and bool(lis_cfg.get("notifications", True))

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

    # Initialize keyboard listener quietly if accessibility is granted
    can_monitor = True
    if is_macos:
        try:
            from ApplicationServices import AXIsProcessTrusted
            can_monitor = bool(AXIsProcessTrusted())
        except Exception:
            can_monitor = False

    if can_monitor:
        try:
            kb_listener = keyboard.Listener(on_press=on_press)
            kb_listener.start()
        except Exception:
            pass

    def emit_shortcut(action_label: str, conf_pct: float, latency_ms: float):
        """Dispatches Cmd+C / Cmd+V using full 4-event sequence (Quartz + pynput)."""
        is_left = "left" in action_label.lower()
        key_char = "c" if is_left else "v"
        action_name = "Copy" if is_left else "Paste"
        shortcut_str = f"{mod_name}+{key_char.upper()}"

        if dry_run:
            print(f"  [DRY-RUN] {action_label.upper():9s} -> Would {action_name} ({shortcut_str})  [Conf: {conf_pct:.0f}% | {latency_ms:.1f} ms]")
            if enable_notifications:
                show_toast("AcousticTap [DRY-RUN]", f"{action_label.replace('_', ' ').title()}", f"Preview: Would {action_name} ({shortcut_str})", sound="Tink")
            return

        nonlocal is_emitting_synthetic
        is_emitting_synthetic = True
        dispatched = False

        # Read initial clipboard count on macOS to verify if Copy succeeded
        initial_count = 0
        if is_macos:
            try:
                from AppKit import NSPasteboard
                initial_count = NSPasteboard.generalPasteboard().changeCount()
            except Exception:
                pass

        try:
            # 1. Native macOS CoreGraphics Quartz Event (Full 4-event hardware sequence)
            if is_macos:
                try:
                    import Quartz
                    src = Quartz.CGEventSourceCreate(Quartz.kCGEventSourceStateCombinedSessionState)
                    vk = 8 if key_char == "c" else 9  # virtual keycode: 8='c', 9='v'

                    # Step 1: Command Key Down (virtual keycode 55 = Left Command)
                    cmd_down = Quartz.CGEventCreateKeyboardEvent(src, 55, True)
                    Quartz.CGEventSetFlags(cmd_down, Quartz.kCGEventFlagMaskCommand)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_down)
                    time.sleep(0.02)

                    # Step 2: Character Key Down with Command Flag
                    key_down = Quartz.CGEventCreateKeyboardEvent(src, vk, True)
                    Quartz.CGEventSetFlags(key_down, Quartz.kCGEventFlagMaskCommand)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_down)
                    time.sleep(0.025)

                    # Step 3: Character Key Up with Command Flag
                    key_up = Quartz.CGEventCreateKeyboardEvent(src, vk, False)
                    Quartz.CGEventSetFlags(key_up, Quartz.kCGEventFlagMaskCommand)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, key_up)
                    time.sleep(0.02)

                    # Step 4: Command Key Up (virtual keycode 55)
                    cmd_up = Quartz.CGEventCreateKeyboardEvent(src, 55, False)
                    Quartz.CGEventPost(Quartz.kCGHIDEventTap, cmd_up)
                    time.sleep(0.02)
                    dispatched = True
                except Exception:
                    dispatched = False

            # 2. Pynput fallback
            if not dispatched:
                with kb_controller.pressed(mod_key):
                    time.sleep(0.025)
                    kb_controller.press(key_char)
                    time.sleep(0.025)
                    kb_controller.release(key_char)
                    time.sleep(0.02)
                dispatched = True

            # 3. Report detailed clipboard feedback & dispatch macOS toast notifications
            clip_info = ""
            if is_macos:
                time.sleep(0.05)
                try:
                    from AppKit import NSPasteboard, NSPasteboardTypeString
                    pb = NSPasteboard.generalPasteboard()
                    new_count = pb.changeCount()
                    items = pb.pasteboardItems()
                    current_clip = items[0].stringForType_(NSPasteboardTypeString) if items else ""
                    if current_clip:
                        short = current_clip.strip().replace("\n", " ")
                        short_preview = (short[:24] + "...") if len(short) > 24 else short
                        if is_left:
                            if new_count > initial_count:
                                clip_info = f' -> Copied: "{short_preview}"'
                                if enable_notifications:
                                    show_toast("AcousticTap - Copy (Cmd+C)", "Text Copied to Clipboard ✓", f'"{short_preview}"', sound="Tink")
                            else:
                                clip_info = ' (Highlight text in active app to copy)'
                                if enable_notifications:
                                    show_toast("AcousticTap - Copy (Cmd+C)", "Left Tap Recognized", "No new text was selected to copy.", sound="Purr")
                        else:
                            clip_info = f' -> Pasted: "{short_preview}"'
                            if enable_notifications:
                                show_toast("AcousticTap - Paste (Cmd+V)", "Text Pasted ✓", f'"{short_preview}"', sound="Pop")
                    else:
                        if is_left:
                            clip_info = ' (Highlight text in active app to copy)'
                            if enable_notifications:
                                show_toast("AcousticTap - Copy (Cmd+C)", "Left Tap Recognized", "No text selected in active window.", sound="Purr")
                        else:
                            clip_info = ' (Clipboard empty)'
                            if enable_notifications:
                                show_toast("AcousticTap - Paste (Cmd+V)", "Clipboard Empty", "Nothing to paste. Tap left palm rest to copy first.", sound="Basso")
                except Exception:
                    if enable_notifications:
                        show_toast("AcousticTap", f"{action_name} ({shortcut_str})", f"{action_label.replace('_', ' ').title()} Triggered", sound="Tink")
            elif enable_notifications:
                show_toast("AcousticTap", f"{action_name} ({shortcut_str})", f"{action_label.replace('_', ' ').title()} Triggered", sound="Tink")

            print(f"  >>> [{action_label.upper()}] -> {action_name} ({shortcut_str}) executed!{clip_info}  [Conf: {conf_pct:.0f}% | {latency_ms:.1f} ms]")
        except Exception as ex:
            print(f"  [Error] Failed to execute {shortcut_str}: {ex}")
        finally:
            time.sleep(0.02)
            is_emitting_synthetic = False

    effective_thresh = float(threshold if threshold is not None else RMS_THRESHOLD)
    detector = ImpulseDetector(threshold=effective_thresh)

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
        audio_ctx = sd.InputStream(
            samplerate=SAMPLE_RATE, blocksize=BLOCK_SIZE, channels=channels, callback=detector.audio_callback
        )

    with audio_ctx:
        # Resolve device name for the clean banner
        tap_name = "MacBook Microphone"
        if hasattr(audio_ctx, "tap_device") and audio_ctx.tap_device:
            tap_name = f"{audio_ctx.tap_device.name} (#{audio_ctx.tap_device.index})"

        print("\n" + "=" * 65)
        print("  AcousticTap - Palm Rest Acoustic Gestures")
        if dry_run:
            print("  Mode:      DRY-RUN (Preview Only - Shortcuts are NOT pressed)")
        else:
            print(f"  Mode:      LIVE MACRO ({mod_name}+C / {mod_name}+V Active)")
        print(f"  Hardware:  {tap_name} (Chassis Isolated)")
        print(f"  Geofence:  {'DISABLED' if no_geofence else 'ACTIVE (Desk Knock Filter)'}")
        print("=" * 65)
        print("  Shortcuts:")
        print(f"    • Left Palm Rest  -> Copy  ({mod_name}+C)")
        print(f"    • Right Palm Rest -> Paste ({mod_name}+V)")
        print("-----------------------------------------------------------------")
        if dry_run:
            print("  [Notice] In --dry-run mode, keys are simulated and not sent.")
            print("  To ACTUALLY copy and paste, run without --dry-run:")
            print("  >>> uv run python -m src.holo.listen <<<")
            print("-----------------------------------------------------------------")
        print("  Ready! Tap your palm rest... (Press Ctrl+C to stop)\n")

        while True:
            try:
                waveform = detector.event_queue.get(timeout=1.0)
            except queue.Empty:
                if hasattr(audio_ctx, "ensure_active"):
                    audio_ctx.ensure_active()
                continue

            # Active keyboard typing suppression
            if (time.time() - last_keypress_time) < typing_suppression_window:
                if verbose:
                    print("  -- [Typing] Ignored during active keyboard use")
                continue

            t0 = time.perf_counter()
            ml_1d = extract_features(waveform, feature_set="enhanced")
            fourier_1d = (
                extract_fourier_triangulation_vector(waveform)
                if triangulator is not None
                else np.zeros(43, dtype=np.float64)
            )

            # -----------------------------------------------------------------
            # STEP 1: Passive Acoustic Geofence Boundary Check (FIRST!)
            # -----------------------------------------------------------------
            if geofence is not None and not no_geofence:
                gf_decision = geofence.verify(ml_1d, fourier_1d)
                if not gf_decision.passed:
                    if verbose:
                        latency_ms = (time.perf_counter() - t0) * 1000.0
                        print(f"  -- [Desk Knock] Filtered ({gf_decision.reason}) | {latency_ms:.1f} ms")
                    else:
                        print("  -- [Desk Knock] Filtered (Outside Boundary)")
                    continue

            # -----------------------------------------------------------------
            # STEP 2: Gesture Classification (ONLY for verified in-bounds taps!)
            # -----------------------------------------------------------------
            ml_feats = ml_1d.reshape(1, -1)
            p_ml = model.predict_proba(ml_feats)[0]

            if triangulator is not None:
                fourier_feats = fourier_1d.reshape(1, -1)
                p_fourier = triangulator.predict_proba(fourier_feats)[0]
                probs = (1.0 - fourier_weight) * p_ml + fourier_weight * p_fourier
            else:
                probs = p_ml

            pred_idx = int(np.argmax(probs))
            confidence = float(probs[pred_idx])
            pred_label = classes[pred_idx]
            latency_ms = (time.perf_counter() - t0) * 1000.0

            req_conf = float(class_thresholds.get(pred_label, default_conf))
            if geofence is not None and not no_geofence:
                if gf_decision.prob_inside < (geofence.boundary_threshold + 0.12):
                    req_conf = min(0.92, req_conf + 0.10)

            if pred_label == "noise":
                if verbose:
                    print(f"  -- [Noise] Clatter filtered (Conf: {confidence*100:.0f}%) | {latency_ms:.1f} ms")
                continue

            if confidence < req_conf:
                if verbose:
                    print(f"  -- [Ignored] Low confidence: {pred_label} ({confidence:.2f} < {req_conf:.2f})")
                continue

            if verbose:
                fourier_top = classes[int(np.argmax(p_fourier))] if triangulator else "N/A"
                ml_top = classes[int(np.argmax(p_ml))]
                print(f"     [Breakdown: ML={ml_top} ({np.max(p_ml):.2f}) + FFT={fourier_top} ({np.max(p_fourier):.2f})]")

            emit_shortcut(pred_label, confidence * 100.0, latency_ms)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AcousticTap - Palm Rest Gestures with Hardware Audio Isolation")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to a .joblib model file")
    parser.add_argument("--threshold", type=float, default=RMS_THRESHOLD, help="Mic RMS trigger threshold")
    parser.add_argument("--confidence", type=float, default=0.65, help="Minimum confidence to trigger")
    parser.add_argument("--dry-run", action="store_true", help="Preview predictions without firing OS shortcuts")
    parser.add_argument("--no-geofence", action="store_true", help="Disable passive acoustic geofence spatial verification")
    parser.add_argument("--tap-device", type=str, default=None, help="Device name or index for Point 1 (MacBook mic)")
    parser.add_argument("--secondary-device", type=str, default=None, help="Device name or index for Point 2 (Earphones/Headset)")
    parser.add_argument("--no-secondary-capture", action="store_true", help="Do not capture secondary mic (leave earphones purely for system/music)")
    parser.add_argument("--no-split", action="store_true", help="Disable audio splitting and use default system input")
    parser.add_argument("--no-notify", action="store_true", help="Disable macOS desktop toast notifications")
    parser.add_argument("--list-devices", action="store_true", help="List all audio devices with split input roles and exit")
    parser.add_argument("-v", "--verbose", action="store_true", help="Print detailed mathematical probabilities and latency stats")
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
            verbose=args.verbose,
            no_notify=args.no_notify,
        )
