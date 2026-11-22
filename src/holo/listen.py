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


def emit_shortcut(
    action_label: str,
    conf_pct: float = 100.0,
    latency_ms: float = 0.0,
    dry_run: bool = False,
    enable_notifications: bool = True,
    mod_name: Optional[str] = None,
    kb_controller: Optional[Any] = None,
    on_synthetic_change: Optional[Callable[[bool], None]] = None,
) -> tuple[bool, str]:
    """Dispatches Cmd+C / Cmd+V using full 4-event sequence (Quartz + pynput fallback)."""
    is_left = "left" in action_label.lower()
    key_char = "c" if is_left else "v"
    action_name = "Copy" if is_left else "Paste"
    is_macos = (platform.system() == "Darwin")
    resolved_mod_name = mod_name if mod_name is not None else ("Cmd" if is_macos else "Ctrl")
    shortcut_str = f"{resolved_mod_name}+{key_char.upper()}"

    if dry_run:
        msg = f"[DRY-RUN] {action_label.upper():9s} -> Would {action_name} ({shortcut_str})  [Conf: {conf_pct:.0f}% | {latency_ms:.1f} ms]"
        if enable_notifications:
            show_toast("AcousticTap [DRY-RUN]", f"{action_label.replace('_', ' ').title()}", f"Preview: Would {action_name} ({shortcut_str})", sound="Tink")
        return True, msg

    if on_synthetic_change:
        try:
            on_synthetic_change(True)
        except Exception:
            pass

    dispatched = False
    clip_info = ""

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
            try:
                if kb_controller is None:
                    kb_controller = keyboard.Controller()
                mod_key = keyboard.Key.cmd if resolved_mod_name == "Cmd" else keyboard.Key.ctrl
                with kb_controller.pressed(mod_key):
                    time.sleep(0.025)
                    kb_controller.press(key_char)
                    time.sleep(0.025)
                    kb_controller.release(key_char)
                    time.sleep(0.02)
                dispatched = True
            except Exception:
                dispatched = False

        # 3. Report detailed clipboard feedback & dispatch macOS toast notifications
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

        msg = f"[{action_label.upper()}] -> {action_name} ({shortcut_str}) executed!{clip_info}  [Conf: {conf_pct:.0f}% | {latency_ms:.1f} ms]"
        return dispatched, msg

    except Exception as ex:
        return False, f"[Error] Failed to execute {shortcut_str}: {ex}"
    finally:
        time.sleep(0.02)
        if on_synthetic_change:
            try:
                on_synthetic_change(False)
            except Exception:
                pass


class AcousticTapCore:
    """Core gesture recognition and geofence verification engine.
    Shared across both CLI (listen.py) and Desktop UI (engine_bridge.py)."""

    def __init__(
        self,
        model_file: Path = MODEL_PATH,
        confidence_min: Optional[float] = None,
        no_geofence: bool = False,
        no_notify: bool = False,
        autoload: bool = True,
    ):
        self.model_file = Path(model_file)
        self.confidence_min = confidence_min
        self.no_geofence = no_geofence
        self.no_notify = no_notify
        self.is_macos = (platform.system() == "Darwin")

        self.model = None
        self.triangulator = None
        self.geofence = None
        self.fourier_weight = 0.50
        self.feature_set = "enhanced"
        self.classes = ["left_tap", "right_tap", "noise"]
        self.class_thresholds = {"left_tap": 0.50, "right_tap": 0.55}
        self.default_conf = 0.55
        self.typing_suppression_window = 0.15
        self.enable_notifications = not no_notify

        self.mod_name = "Cmd" if self.is_macos else "Ctrl"
        self.mod_key = keyboard.Key.cmd if self.is_macos else keyboard.Key.ctrl
        self.kb_controller = keyboard.Controller()
        self.last_keypress_time = 0.0
        self.is_emitting_synthetic = False

        if autoload:
            self.load_model()
        self._init_typing_listener()

    def _set_synthetic(self, val: bool) -> None:
        self.is_emitting_synthetic = val

    def load_model(self, path: Optional[Path] = None) -> bool:
        target = Path(path) if path is not None else self.model_file
        if not target.exists():
            return False
        try:
            saved = joblib.load(target)
            self.model = saved.get("model")
            self.triangulator = saved.get("triangulator")
            self.geofence = saved.get("geofence")
            self.fourier_weight = float(saved.get("fourier_weight", CFG.get("hybrid_fusion", {}).get("fourier_weight", 0.50)))
            self.feature_set = saved.get("feature_set", "enhanced")
            self.classes = saved.get("classes", ["left_tap", "right_tap", "noise"])

            lis_cfg = CFG.get("listener", {})
            self.default_conf = float(self.confidence_min if self.confidence_min is not None else lis_cfg.get("default_confidence", 0.55))
            self.class_thresholds = dict(lis_cfg.get("class_thresholds", {"left_tap": 0.50, "right_tap": 0.55}))
            self.typing_suppression_window = float(lis_cfg.get("typing_suppression_seconds", 0.15))
            self.enable_notifications = (not self.no_notify) and bool(lis_cfg.get("notifications", True))

            cfg_mod = str(lis_cfg.get("os_modifier_key", "auto")).lower()
            if cfg_mod == "cmd" or (cfg_mod == "auto" and self.is_macos):
                self.mod_key = keyboard.Key.cmd
                self.mod_name = "Cmd"
            else:
                self.mod_key = keyboard.Key.ctrl
                self.mod_name = "Ctrl"
            return True
        except Exception:
            return False

    def _init_typing_listener(self) -> None:
        can_monitor = True
        if self.is_macos:
            try:
                from ApplicationServices import AXIsProcessTrusted
                can_monitor = bool(AXIsProcessTrusted())
            except Exception:
                can_monitor = False

        if can_monitor:
            def on_press(key):
                if not self.is_emitting_synthetic:
                    self.last_keypress_time = time.time()
            try:
                self.kb_listener = keyboard.Listener(on_press=on_press)
                self.kb_listener.daemon = True
                self.kb_listener.start()
            except Exception:
                pass

    def evaluate_waveform(
        self,
        waveform: np.ndarray,
        enable_shortcuts: bool = False,
        dry_run: bool = False,
        override_confidence: Optional[float] = None,
    ) -> Dict[str, Any]:
        t0 = time.perf_counter()
        now_ts = time.time()

        # 0. Active typing suppression
        if (time.time() - self.last_keypress_time) < self.typing_suppression_window:
            return {
                "timestamp": now_ts,
                "ignored_typing": True,
                "label": "typing",
                "confidence": 0.0,
                "required_confidence": 1.0,
                "meets_min_confidence": False,
                "geofence_passed": False,
                "geofence_reason": "Typing suppression window active",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "breakdown": "Ignored (Typing)",
                "waveform": waveform,
                "action_taken": False,
            }

        # If model not loaded, emit raw uncalibrated event
        if self.model is None:
            peak = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
            return {
                "timestamp": now_ts,
                "ignored_typing": False,
                "label": "uncalibrated_tap",
                "confidence": 1.0,
                "required_confidence": 0.55,
                "meets_min_confidence": True,
                "geofence_passed": True,
                "geofence_reason": "No model loaded (raw transient)",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "breakdown": f"Raw Peak: {peak:.2f}",
                "waveform": waveform,
                "action_taken": False,
            }

        ml_1d = extract_features(waveform, feature_set=self.feature_set)
        fourier_1d = (
            extract_fourier_triangulation_vector(waveform)
            if self.triangulator is not None
            else np.zeros(43, dtype=np.float64)
        )

        # 1. Geofence verification
        gf_passed = True
        gf_reason = "No geofence present"
        gf_prob_in = 1.0
        gf_details: Dict[str, Any] = {}
        if self.geofence is not None and not self.no_geofence:
            gf_decision = self.geofence.verify(ml_1d, fourier_1d)
            gf_passed = bool(getattr(gf_decision, "passed", True))
            gf_reason = str(getattr(gf_decision, "reason", ""))
            try:
                gf_prob_in = float(getattr(gf_decision, "prob_inside", 1.0))
            except (TypeError, ValueError):
                gf_prob_in = 1.0
            gf_details = {
                "passed": gf_passed,
                "stage_failed": getattr(gf_decision, "stage_failed", 0),
                "prob_inside": gf_prob_in,
                "dist_inside": getattr(gf_decision, "dist_inside", 0.0),
                "dist_outside": getattr(gf_decision, "dist_outside", 0.0),
                "threshold": getattr(gf_decision, "threshold", 0.62),
                "reason": gf_reason,
                "timestamp": now_ts,
            }

        # 2. Hybrid classification
        ml_feats = ml_1d.reshape(1, -1)
        p_ml = self.model.predict_proba(ml_feats)[0]

        if self.triangulator is not None:
            fourier_feats = fourier_1d.reshape(1, -1)
            p_fourier = self.triangulator.predict_proba(fourier_feats)[0]
            probs = (1.0 - self.fourier_weight) * p_ml + self.fourier_weight * p_fourier
            fourier_top = self.classes[int(np.argmax(p_fourier))]
            ml_top = self.classes[int(np.argmax(p_ml))]
            breakdown = f"ML:{ml_top}({np.max(p_ml):.2f}) + FFT:{fourier_top}({np.max(p_fourier):.2f})"
        else:
            probs = p_ml
            breakdown = "ML-Only"

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = self.classes[pred_idx]
        latency_ms = (time.perf_counter() - t0) * 1000.0

        # Required confidence calculation with geofence borderline boost
        def_conf = self.default_conf
        if override_confidence is not None:
            try:
                def_conf = float(override_confidence)
            except (TypeError, ValueError):
                pass

        req_conf = float(self.class_thresholds.get(pred_label, def_conf))
        if override_confidence is not None:
            try:
                req_conf = max(req_conf, float(override_confidence))
            except (TypeError, ValueError):
                pass

        if self.geofence is not None and not self.no_geofence:
            try:
                raw_bth = getattr(self.geofence, "boundary_threshold", 0.62)
                boundary_th = float(raw_bth)
                prob_val = float(gf_prob_in)
                if prob_val < (boundary_th + 0.12):
                    req_conf = min(0.92, req_conf + 0.10)
            except (TypeError, ValueError):
                pass

        meets_confidence = (confidence >= req_conf) and (pred_label != "noise")
        action_taken = False
        action_msg = ""

        # 3. Macro Shortcut execution
        if gf_passed and meets_confidence and enable_shortcuts:
            dispatched, action_msg = emit_shortcut(
                action_label=pred_label,
                conf_pct=confidence * 100.0,
                latency_ms=latency_ms,
                dry_run=dry_run,
                enable_notifications=self.enable_notifications,
                mod_name=self.mod_name,
                kb_controller=self.kb_controller,
                on_synthetic_change=self._set_synthetic,
            )
            action_taken = dispatched

        return {
            "timestamp": now_ts,
            "ignored_typing": False,
            "label": pred_label,
            "confidence": confidence,
            "required_confidence": req_conf,
            "meets_min_confidence": meets_confidence,
            "geofence_passed": gf_passed,
            "geofence_reason": gf_reason,
            "latency_ms": latency_ms,
            "breakdown": breakdown,
            "waveform": waveform,
            "action_taken": action_taken,
            "action_message": action_msg,
            "geofence_details": gf_details,
        }

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

    core = AcousticTapCore(
        model_file=model_file,
        confidence_min=confidence_min,
        no_geofence=no_geofence,
        no_notify=no_notify,
    )

    channels = get_input_channels()
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
        tap_name = "MacBook Microphone"
        if hasattr(audio_ctx, "tap_device") and audio_ctx.tap_device:
            tap_name = f"{audio_ctx.tap_device.name} (#{audio_ctx.tap_device.index})"

        print("\n" + "=" * 65)
        print("  AcousticTap - Palm Rest Acoustic Gestures")
        if dry_run:
            print("  Mode:      DRY-RUN (Preview Only - Shortcuts are NOT pressed)")
        else:
            print(f"  Mode:      LIVE MACRO ({core.mod_name}+C / {core.mod_name}+V Active)")
        print(f"  Hardware:  {tap_name} (Chassis Isolated)")
        print(f"  Geofence:  {'DISABLED' if no_geofence else 'ACTIVE (Desk Knock Filter)'}")
        print("=" * 65)
        print("  Shortcuts:")
        print(f"    • Left Palm Rest  -> Copy  ({core.mod_name}+C)")
        print(f"    • Right Palm Rest -> Paste ({core.mod_name}+V)")
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

            res = core.evaluate_waveform(
                waveform,
                enable_shortcuts=(not dry_run),
                dry_run=dry_run,
                override_confidence=confidence_min,
            )

            if res.get("ignored_typing"):
                if verbose:
                    print("  -- [Typing] Ignored during active keyboard use")
                continue

            if not res.get("geofence_passed"):
                if verbose:
                    print(f"  -- [Desk Knock] Filtered ({res.get('geofence_reason')}) | {res.get('latency_ms'):.1f} ms")
                else:
                    print("  -- [Desk Knock] Filtered (Outside Boundary)")
                continue

            pred_label = res.get("label", "unknown")
            conf = float(res.get("confidence", 0.0))
            req_c = float(res.get("required_confidence", 0.55))
            lat = float(res.get("latency_ms", 0.0))

            if pred_label == "noise":
                if verbose:
                    print(f"  -- [Noise] Clatter filtered (Conf: {conf*100:.0f}%) | {lat:.1f} ms")
                continue

            if not res.get("meets_min_confidence"):
                if verbose:
                    print(f"  -- [Ignored] Low confidence: {pred_label} ({conf:.2f} < {req_c:.2f})")
                continue

            if verbose:
                print(f"     [Breakdown: {res.get('breakdown')}]")

            if dry_run:
                action_name = "Copy" if "left" in pred_label else "Paste"
                key_c = "C" if "left" in pred_label else "V"
                print(f"  [DRY-RUN] {pred_label.upper():9s} -> Would {action_name} ({core.mod_name}+{key_c})  [Conf: {conf*100:.0f}% | {lat:.1f} ms]")
            else:
                act_msg = res.get("action_message", "")
                if act_msg:
                    print(f"  >>> {act_msg}")

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
