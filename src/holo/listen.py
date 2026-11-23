from __future__ import annotations

import os
import sys
import time
import queue
import threading
import platform
import argparse
import subprocess
from pathlib import Path
from typing import Optional, Dict, Any, Callable
import numpy as np
import sounddevice as sd
import joblib
from pynput import keyboard
from .config import (
    CFG,
    SAMPLE_RATE,
    BLOCK_SIZE,
    RMS_THRESHOLD,
    MODEL_DIR,
    MODEL_PATH,
    SPLIT_INPUT,
    TAP_DEVICE,
    SECONDARY_DEVICE,
    CAPTURE_SECONDARY,
    AUTO_RECONNECT,
    DOUBLE_TAP_ENABLED,
    DOUBLE_TAP_MIN_MS,
    DOUBLE_TAP_MAX_MS,
    DOUBLE_TAP_REQUIRE_SAME_ZONE,
    DOUBLE_TAP_ACTIONS,
)
from .detector import ImpulseDetector, get_input_channels
from .features import extract_features
from .triangulation import extract_fourier_triangulation_vector
from .audio_manager import SplitAudioEngine, print_audio_devices
from .double_tap import DoubleTapStateMachine

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
    inter_tap_ms: float = 0.0,
) -> tuple[bool, str]:
    """Dispatches Cmd+C / Cmd+V / Cmd+X / Cmd+Z using full 4-event sequence (Quartz + pynput fallback)."""
    lbl_lower = action_label.lower()
    is_double = "double" in lbl_lower
    is_left = "left" in lbl_lower

    if is_double:
        if is_left:
            key_char = "x"
            action_name = "Cut"
        else:
            key_char = "z"
            action_name = "Undo"
    else:
        if is_left:
            key_char = "c"
            action_name = "Copy"
        else:
            key_char = "v"
            action_name = "Paste"

    is_macos = (platform.system() == "Darwin")
    resolved_mod_name = mod_name if mod_name is not None else ("Cmd" if is_macos else "Ctrl")
    shortcut_str = f"{resolved_mod_name}+{key_char.upper()}"
    gap_info = f" | Interval: {inter_tap_ms:.0f} ms" if inter_tap_ms > 0 else ""

    if dry_run:
        msg = f"[DRY-RUN] {action_label.upper():12s} -> Would {action_name} ({shortcut_str}){gap_info}  [Conf: {conf_pct:.0f}% | {latency_ms:.1f} ms]"
        if enable_notifications:
            show_toast(
                "AcousticTap [DRY-RUN]",
                f"{action_label.replace('_', ' ').title()}",
                f"Preview: Would {action_name} ({shortcut_str}){gap_info}",
                sound="Tink",
            )
        return True, msg

    if on_synthetic_change:
        try:
            on_synthetic_change(True)
        except Exception:
            pass

    dispatched = False
    clip_info = ""

    # Read initial clipboard count on macOS to verify if Copy/Cut succeeded
    initial_count = 0
    if is_macos and not is_double:
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
                vk_map = {"c": 8, "v": 9, "x": 7, "z": 6}
                vk = vk_map.get(key_char, 8)

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
        if is_double:
            if enable_notifications:
                side_str = "Left" if is_left else "Right"
                show_toast(
                    f"AcousticTap - {action_name} ({shortcut_str})",
                    f"Double {side_str} Tap Recognized",
                    f"{action_name} ({shortcut_str}) executed!{gap_info}",
                    sound="Tink" if is_left else "Pop",
                )
        elif is_macos:
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

        msg = f"[{action_label.upper()}] -> {action_name} ({shortcut_str}) executed!{clip_info}{gap_info}  [Conf: {conf_pct:.0f}% | {latency_ms:.1f} ms]"
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


class ShortcutDispatcher:
    """
    Non-blocking background daemon worker thread.
    Consumes queued shortcut events and dispatches Quartz HID keystroke sequences
    without stalling the audio processing or state machine event loop.
    """
    def __init__(self, core: AcousticTapCore, dry_run: bool = False):
        self.core = core
        self.dry_run = dry_run
        self.queue: queue.Queue = queue.Queue()
        self.running = True
        self.worker = threading.Thread(target=self._run, daemon=True, name="AcousticTap-HID-Worker")
        self.worker.start()

    def dispatch(self, ev: Dict[str, Any]):
        self.queue.put(ev)

    def _run(self):
        while self.running:
            try:
                ev = self.queue.get(timeout=0.2)
            except queue.Empty:
                continue
            try:
                ev_label = ev.get("label", "")
                ev_conf = float(ev.get("confidence", 0.0))
                ev_lat = float(ev.get("latency_ms", 0.0))
                ev_dt = float(ev.get("inter_tap_ms", 0.0))
                dispatched, act_msg = emit_shortcut(
                    action_label=ev_label,
                    conf_pct=ev_conf * 100.0,
                    latency_ms=ev_lat,
                    dry_run=self.dry_run,
                    enable_notifications=self.core.enable_notifications,
                    mod_name=self.core.mod_name,
                    kb_controller=self.core.kb_controller,
                    on_synthetic_change=self.core._set_synthetic,
                    inter_tap_ms=ev_dt,
                )
                if self.dry_run:
                    print(f"  {act_msg}")
                elif act_msg:
                    print(f"  >>> {act_msg}")
            except Exception:
                pass
            finally:
                self.queue.task_done()

    def stop(self):
        self.running = False


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
        self.last_synthetic_emit_ts = 0.0

        if autoload:
            self.load_model()
        self._init_typing_listener()

    def _set_synthetic(self, val: bool) -> None:
        self.is_emitting_synthetic = val
        self.last_synthetic_emit_ts = time.time()

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

            # Warm up JIT and feature extraction so first live tap runs in ~5 ms
            try:
                dummy_wav = np.zeros(4096, dtype=np.float32)
                dummy_wav[1024] = 0.05
                self.evaluate_waveform(dummy_wav, enable_shortcuts=False)
            except Exception:
                pass

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
                now = time.time()
                # Immunize against self-injected keystrokes from Quartz/pynput
                if (now - self.last_synthetic_emit_ts) < 0.35 or self.is_emitting_synthetic:
                    return
                self.last_keypress_time = now
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
        ringing_tolerant: bool = False,
        tap2_boundary_threshold: float = 0.38,
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
                "feat_ms": 0.0,
                "gf_ms": 0.0,
                "inf_ms": 0.0,
                "total_compute_ms": (time.perf_counter() - t0) * 1000.0,
                "breakdown": "Ignored (Typing)",
                "waveform": waveform,
                "action_taken": False,
            }

        # If model not loaded, emit raw uncalibrated event
        if self.model is None:
            peak = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
            total_compute = (time.perf_counter() - t0) * 1000.0
            return {
                "timestamp": now_ts,
                "ignored_typing": False,
                "label": "uncalibrated_tap",
                "confidence": 1.0,
                "required_confidence": 0.55,
                "meets_min_confidence": True,
                "geofence_passed": True,
                "geofence_reason": "No model loaded (raw transient)",
                "latency_ms": total_compute,
                "feat_ms": 0.0,
                "gf_ms": 0.0,
                "inf_ms": 0.0,
                "total_compute_ms": total_compute,
                "breakdown": f"Raw Peak: {peak:.2f}",
                "waveform": waveform,
                "action_taken": False,
            }

        t_feat0 = time.perf_counter()
        ml_1d = extract_features(waveform, feature_set=self.feature_set)
        fourier_1d = (
            extract_fourier_triangulation_vector(waveform)
            if self.triangulator is not None
            else np.zeros(43, dtype=np.float64)
        )
        feat_ms = (time.perf_counter() - t_feat0) * 1000.0

        # 1. Geofence verification
        t_gf0 = time.perf_counter()
        gf_passed = True
        gf_reason = "No geofence present"
        gf_prob_in = 1.0
        gf_details: Dict[str, Any] = {}
        if self.geofence is not None and not self.no_geofence:
            gf_decision = self.geofence.verify(
                ml_1d,
                fourier_1d,
                ringing_tolerant=ringing_tolerant,
                tap2_boundary_threshold=tap2_boundary_threshold,
            )
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
        gf_ms = (time.perf_counter() - t_gf0) * 1000.0

        # 2. Hybrid classification
        t_inf0 = time.perf_counter()
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

        inf_ms = (time.perf_counter() - t_inf0) * 1000.0
        total_compute_ms = (time.perf_counter() - t0) * 1000.0

        pred_idx = int(np.argmax(probs))
        confidence = float(probs[pred_idx])
        pred_label = self.classes[pred_idx]
        latency_ms = total_compute_ms

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
            "feat_ms": feat_ms,
            "gf_ms": gf_ms,
            "inf_ms": inf_ms,
            "total_compute_ms": total_compute_ms,
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
    double_tap: Optional[bool] = None,
    no_double_tap: bool = False,
):
    if double_tap is True:
        double_tap_active = True
    elif no_double_tap is True or double_tap is False:
        double_tap_active = False
    else:
        double_tap_active = bool(DOUBLE_TAP_ENABLED)

    # Intelligently load the double-tap augmented model if double-tap mode is active
    if model_file == MODEL_PATH and double_tap_active:
        dt_model = MODEL_DIR / "acoustictap_doubletap_model.joblib"
        if dt_model.exists():
            model_file = dt_model
            print(f"  [Model Selection] Auto-selected double-tap model: {dt_model.name}")
        else:
            print(f"  [Model Selection] Dedicated double-tap model ({dt_model.name}) not found. Gracefully falling back to base model ({MODEL_PATH.name}).")

    if not model_file.exists():
        raise FileNotFoundError(f"Model file not found at {model_file}. Run 'uv run python -m src.holo.train' first.")

    core = AcousticTapCore(
        model_file=model_file,
        confidence_min=confidence_min,
        no_geofence=no_geofence,
        no_notify=no_notify,
    )
    state_machine = (
        DoubleTapStateMachine(
            min_inter_tap_ms=DOUBLE_TAP_MIN_MS,
            max_inter_tap_ms=DOUBLE_TAP_MAX_MS,
            require_same_zone=DOUBLE_TAP_REQUIRE_SAME_ZONE,
            actions=DOUBLE_TAP_ACTIONS,
        )
        if double_tap_active
        else None
    )

    channels = get_input_channels()
    effective_thresh = float(threshold if threshold is not None else RMS_THRESHOLD)
    detector = ImpulseDetector(
        threshold=effective_thresh,
        double_tap_mode=double_tap_active,
        min_inter_tap_ms=DOUBLE_TAP_MIN_MS,
    )

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
            print("  Mode:        DRY-RUN (Preview Only - Shortcuts are NOT pressed)")
        else:
            print(f"  Mode:        LIVE MACRO Active ({core.mod_name}+C/V/X/Z)")
        print(f"  Model:       {model_file.name} ({'Double-Tap Augmented' if 'doubletap' in model_file.name else 'Standard'})")
        print(f"  Hardware:    {tap_name} (Chassis Isolated)")
        print(f"  Geofence:    {'DISABLED' if no_geofence else 'ACTIVE (Desk Knock Filter)'}")
        print(f"  Double-Tap:  {'ACTIVE (Cut: Cmd+X, Undo: Cmd+Z)' if double_tap_active else 'DISABLED (Instant Single-Tap)'}")
        print("=" * 65)
        print("  Shortcuts:")
        print(f"    • Left Palm Rest        -> Copy  ({core.mod_name}+C)")
        print(f"    • Right Palm Rest       -> Paste ({core.mod_name}+V)")
        if double_tap_active:
            print(f"    • Double-Tap Left       -> Cut   ({core.mod_name}+X)")
            print(f"    • Double-Tap Right      -> Undo  ({core.mod_name}+Z)")
        print("-----------------------------------------------------------------")
        if dry_run:
            print("  [Notice] In --dry-run mode, keys are simulated and not sent.")
            print("  To ACTUALLY execute shortcuts, run without --dry-run:")
            print("  >>> uv run python -m src.holo.listen <<<")
            print("-----------------------------------------------------------------")
        print("  Ready! Tap your palm rest... (Press Ctrl+C to stop)\n")

        dispatcher = ShortcutDispatcher(core=core, dry_run=dry_run)

        while True:
            ready_events = []
            try:
                waveform = detector.event_queue.get(timeout=0.025)

                # Check candidate Tap 2 timing window strictly clamped to [min_inter_tap_ms, max_inter_tap_ms]
                now_ts = time.time()
                delta_ms = (
                    (now_ts - state_machine.pending_time) * 1000.0
                    if (double_tap_active and state_machine is not None and state_machine.pending_tap is not None)
                    else 0.0
                )
                is_candidate_tap2_window = bool(
                    double_tap_active
                    and state_machine is not None
                    and state_machine.pending_tap is not None
                    and (state_machine.min_inter_tap_ms <= delta_ms <= state_machine.max_inter_tap_ms)
                )

                res = core.evaluate_waveform(
                    waveform,
                    enable_shortcuts=False,
                    dry_run=dry_run,
                    override_confidence=confidence_min,
                    ringing_tolerant=is_candidate_tap2_window,
                    tap2_boundary_threshold=0.38,
                )

                if res.get("ignored_typing"):
                    if verbose:
                        print("  -- [Typing] Ignored during active keyboard use")
                    continue

                is_candidate_tap2 = False
                if is_candidate_tap2_window:
                    cand_label = res.get("label", "unknown")
                    cand_conf = float(res.get("confidence", 0.0))
                    pending_label = state_machine.pending_tap.get("label", "")
                    same_zone = (cand_label == pending_label) or ("left" in cand_label and "left" in pending_label) or ("right" in cand_label and "right" in pending_label)
                    # Candidate Tap 2 must match zone (if require_same_zone) and achieve confidence >= 0.60
                    if (same_zone or not state_machine.require_same_zone) and cand_label in ("left_tap", "right_tap") and cand_conf >= 0.60:
                        is_candidate_tap2 = True

                # Enforce Geofence: NEVER bypass on Tap 2! Desk knocks must still be rejected.
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

                if not res.get("meets_min_confidence") and not (is_candidate_tap2 and conf >= 0.60):
                    if verbose:
                        print(f"  -- [Ignored] Low confidence: {pred_label} ({conf:.2f} < {req_c:.2f})")
                    continue

                if is_candidate_tap2 and verbose:
                    print(f"     [Double-Tap Tap 2: Ringing tolerance applied | Gap: {delta_ms:.1f} ms]")

                if verbose:
                    feat_ms = res.get("feat_ms", 0.0)
                    gf_ms = res.get("gf_ms", 0.0)
                    inf_ms = res.get("inf_ms", 0.0)
                    tot_ms = res.get("total_compute_ms", lat)
                    print(f"     [Breakdown: {res.get('breakdown')} | Timing: feat={feat_ms:.1f}ms gf={gf_ms:.1f}ms inf={inf_ms:.1f}ms total={tot_ms:.1f}ms]")

                if double_tap_active and state_machine is not None:
                    ready_events.extend(state_machine.process_tap(res))
                else:
                    ready_events.append(res)

            except queue.Empty:
                if hasattr(audio_ctx, "ensure_active"):
                    audio_ctx.ensure_active()

            # Poll state machine for expired single tap
            if double_tap_active and state_machine is not None:
                ready_events.extend(state_machine.check_timeout(time.time()))

            for ev in ready_events:
                dispatcher.dispatch(ev)

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="AcousticTap - Palm Rest Gestures with Hardware Audio Isolation")
    parser.add_argument("--model", type=Path, default=MODEL_PATH, help="Path to a .joblib model file")
    parser.add_argument("--threshold", type=float, default=RMS_THRESHOLD, help="Mic RMS trigger threshold")
    parser.add_argument("--confidence", type=float, default=None, help="Minimum confidence to trigger (default: from config/class thresholds)")
    parser.add_argument("--dry-run", action="store_true", help="Preview predictions without firing OS shortcuts")
    parser.add_argument("--no-geofence", action="store_true", help="Disable passive acoustic geofence spatial verification")
    parser.add_argument("--double-tap", dest="double_tap", action="store_true", default=None, help="Force enable double-tap gestures (Cut: Cmd+X, Undo: Cmd+Z)")
    parser.add_argument("--no-double-tap", dest="no_double_tap", action="store_true", default=False, help="Force disable double-tap gestures (instant zero-latency single taps)")
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
            double_tap=args.double_tap,
            no_double_tap=args.no_double_tap,
        )
