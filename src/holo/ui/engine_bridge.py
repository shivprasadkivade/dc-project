"""Engine bridge connecting Holo core audio processing and geofencing to Qt UI.

Runs the audio engine and queue drain loop inside a dedicated background QThread,
emitting typed Qt Signals to cleanly decouple UI views from backend internals.
"""

from __future__ import annotations

import sys
import time
import queue
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import joblib

from PySide6.QtCore import QObject, QThread, Signal, Slot, QTimer

# Import audio hardware helpers cleanly
# If src/holo/audio_manager.py is not present yet, audio_adapter provides safe fallback
try:
    from src.holo.audio_manager import (
        list_audio_devices,
        find_builtin_microphone,
        find_secondary_microphone,
        SplitAudioEngine,
    )
except ImportError:
    from src.holo.ui.audio_adapter import (  # type: ignore # noqa: F401
        list_audio_devices,
        find_builtin_microphone,
        find_secondary_microphone,
        SplitAudioEngine,
    )

from src.holo.config import CFG, MODEL_PATH
from src.holo.features import extract_features
from src.holo.triangulation import extract_fourier_triangulation_vector
from src.holo.ui.config_store import config_store

logger = logging.getLogger(__name__)


class EngineWorker(QObject):
    """Background worker executing inside a dedicated QThread."""

    # Typed Qt Signals
    tap_detected = Signal(dict)
    geofence_event = Signal(dict)
    audio_chunk_ready = Signal(np.ndarray)
    engine_state_changed = Signal(bool, str)
    error_occurred = Signal(str)

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_confidence: float = 0.65,
    ) -> None:
        super().__init__()
        self.model_path = model_path or MODEL_PATH
        self.min_confidence = min_confidence

        self.engine: Optional[SplitAudioEngine] = None
        self._drain_timer = QTimer()
        self._drain_timer.setInterval(25)
        self._drain_timer.timeout.connect(self._drain_queues)

        # Model state
        self.model = None
        self.triangulator = None
        self.geofence = None
        self.fourier_weight: float = 0.50
        self.feature_set: str = "enhanced"
        self.classes: List[str] = ["left_tap", "right_tap", "noise"]
        self.class_thresholds: Dict[str, float] = {}

        self.enable_shortcuts: bool = False
        self._is_active: bool = False

    def load_model(self, path: Optional[Path] = None) -> bool:
        """Loads and prepares trained model, triangulator, and geofence."""
        target_path = Path(path) if path is not None else self.model_path
        if not target_path.exists():
            logger.info("Model file not found at %s. Running in uncalibrated mode.", target_path)
            self.model = None
            self.triangulator = None
            self.geofence = None
            return False

        try:
            saved = joblib.load(target_path)
            self.model = saved.get("model")
            self.triangulator = saved.get("triangulator")
            self.geofence = saved.get("geofence")
            self.fourier_weight = float(saved.get("fourier_weight", 0.50))
            self.feature_set = saved.get("feature_set", "enhanced")
            self.classes = saved.get("classes", ["left_tap", "right_tap", "noise"])

            lis_cfg = CFG.get("listener", {})
            self.class_thresholds = lis_cfg.get("class_thresholds", {})
            logger.info("Loaded model successfully from %s", target_path.name)
            return True
        except Exception as e:
            logger.error("Failed to load model from %s: %s", target_path, e)
            self.error_occurred.emit(f"Model load error: {e}")
            return False

    @Slot(int, int, bool)
    def start_engine(
        self,
        primary_device: Optional[int] = None,
        secondary_device: Optional[int] = None,
        enable_shortcuts: bool = False,
    ) -> None:
        """Starts audio engine and queue drain timer."""
        if self._is_active:
            return

        self.enable_shortcuts = enable_shortcuts
        self.load_model()

        try:
            self.engine = SplitAudioEngine()
            self.engine.start(primary_device=primary_device, secondary_device=secondary_device)
            self._is_active = True

            # Start high-cadence non-blocking queue drain timer (25 ms interval)
            self._drain_timer.start()

            self.engine_state_changed.emit(True, "Engine listening")
        except Exception as e:
            self._is_active = False
            self.error_occurred.emit(f"Failed to start audio engine: {e}")
            self.engine_state_changed.emit(False, f"Engine error: {e}")

    @Slot()
    def stop_engine(self) -> None:
        """Stops audio engine and drains cleanly."""
        if not self._is_active:
            return

        if self._drain_timer is not None and self._drain_timer.isActive():
            self._drain_timer.stop()

        if self.engine is not None:
            try:
                self.engine.stop()
            except Exception as e:
                logger.warning("Error stopping engine: %s", e)
            self.engine = None

        self._is_active = False
        self.engine_state_changed.emit(False, "Engine stopped")

    @Slot(float)
    def set_min_confidence(self, val: float) -> None:
        """Updates UI-level min confidence threshold."""
        self.min_confidence = max(0.0, min(1.0, float(val)))

    def _drain_queues(self) -> None:
        """Non-blocking queue consumer driven by QTimer."""
        if not self._is_active or self.engine is None:
            return

        # 1. Drain continuous audio chunks from secondary_queue (bounded batch)
        chunks_drained = 0
        while chunks_drained < 4:
            try:
                chunk = self.engine.secondary_queue.get_nowait()
                self.audio_chunk_ready.emit(chunk)
                chunks_drained += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.debug("Error draining audio chunk: %s", e)
                break

        # 2. Drain transient events from detector.event_queue (bounded batch)
        events_drained = 0
        while events_drained < 5:
            try:
                waveform = self.engine.detector.event_queue.get_nowait()
                self._process_tap_waveform(waveform)
                events_drained += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.error("Error processing tap event: %s", e)
                break

    def _process_tap_waveform(self, waveform: np.ndarray) -> None:
        """Extracts features, performs geofence verification, and predicts gesture."""
        t0 = time.perf_counter()
        now_ts = time.time()

        # If model is not loaded, emit basic event for diagnostics
        if self.model is None:
            peak = float(np.max(np.abs(waveform))) if len(waveform) > 0 else 0.0
            event = {
                "timestamp": now_ts,
                "label": "uncalibrated_tap",
                "confidence": 1.0,
                "required_confidence": self.min_confidence,
                "meets_min_confidence": True,
                "geofence_passed": True,
                "geofence_reason": "No model loaded (raw transient)",
                "latency_ms": (time.perf_counter() - t0) * 1000.0,
                "breakdown": f"Raw Peak: {peak:.2f}",
                "waveform": waveform,
                "action_taken": False,
            }
            self.tap_detected.emit(event)
            return

        try:
            ml_1d = extract_features(waveform, feature_set=self.feature_set)
            fourier_1d = (
                extract_fourier_triangulation_vector(waveform)
                if self.triangulator is not None
                else np.zeros(43, dtype=np.float64)
            )

            # Step 1: Geofence Verification
            geofence_passed = True
            gf_reason = "No geofence present"
            gf_details: Dict[str, Any] = {}

            if self.geofence is not None:
                gf_decision = self.geofence.verify(ml_1d, fourier_1d)
                geofence_passed = bool(gf_decision.passed)
                gf_reason = str(gf_decision.reason)
                gf_details = {
                    "passed": gf_decision.passed,
                    "stage_failed": gf_decision.stage_failed,
                    "prob_inside": gf_decision.prob_inside,
                    "dist_inside": gf_decision.dist_inside,
                    "dist_outside": gf_decision.dist_outside,
                    "threshold": gf_decision.threshold,
                    "reason": gf_decision.reason,
                    "timestamp": now_ts,
                }
                self.geofence_event.emit(gf_details)

            # Step 2: Classification (even if outside boundary, compute for diagnostics display)
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

            # Determine required confidence threshold
            req_conf = float(self.class_thresholds.get(pred_label, self.min_confidence))
            # Also ensure it respects user's ui.min_confidence
            effective_req = max(req_conf, self.min_confidence)

            meets_confidence = (confidence >= effective_req)
            action_taken = False
            if geofence_passed and meets_confidence and pred_label != "noise" and self.enable_shortcuts:
                action_taken = True
                self._trigger_macro(pred_label)

            event = {
                "timestamp": now_ts,
                "label": pred_label,
                "confidence": confidence,
                "required_confidence": effective_req,
                "meets_min_confidence": meets_confidence,
                "geofence_passed": geofence_passed,
                "geofence_reason": gf_reason,
                "latency_ms": latency_ms,
                "breakdown": breakdown,
                "waveform": waveform,
                "action_taken": action_taken,
                "geofence_details": gf_details,
            }
            self.tap_detected.emit(event)

        except Exception as e:
            logger.error("Error evaluating tap transient: %s", e)
            self.error_occurred.emit(f"Evaluation error: {e}")

    def _trigger_macro(self, pred_label: str) -> None:
        """Triggers keyboard shortcut if enabled."""
        try:
            from pynput.keyboard import Controller, Key
            kb = Controller()
            import platform
            mod = Key.cmd if platform.system() == "Darwin" else Key.ctrl
            if "left" in pred_label:
                with kb.pressed(mod):
                    kb.tap("c")
            elif "right" in pred_label:
                with kb.pressed(mod):
                    kb.tap("v")
        except Exception as ex:
            logger.warning("Could not execute keyboard macro: %s", ex)


class EngineBridge(QObject):
    """Main-thread coordinator managing the background worker and QThread."""

    tap_detected = Signal(dict)
    geofence_event = Signal(dict)
    audio_chunk_ready = Signal(np.ndarray)
    engine_state_changed = Signal(bool, str)
    error_occurred = Signal(str)

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        ui_cfg = config_store.get_ui_config()
        self.min_confidence = float(ui_cfg.get("min_confidence", 0.65))

        self.worker_thread = QThread()
        self.worker = EngineWorker(min_confidence=self.min_confidence)
        self.worker.moveToThread(self.worker_thread)

        # Forward worker signals to bridge signals
        self.worker.tap_detected.connect(self.tap_detected)
        self.worker.geofence_event.connect(self.geofence_event)
        self.worker.audio_chunk_ready.connect(self.audio_chunk_ready)
        self.worker.engine_state_changed.connect(self.engine_state_changed)
        self.worker.error_occurred.connect(self.error_occurred)

        self.worker_thread.start()
        self._is_running = False

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start_engine(
        self,
        primary_device: Optional[int] = None,
        secondary_device: Optional[int] = None,
        enable_shortcuts: bool = False,
    ) -> None:
        """Directs worker to start audio engine."""
        self._is_running = True
        self.worker.start_engine(primary_device, secondary_device, enable_shortcuts)

    def stop_engine(self) -> None:
        """Directs worker to stop audio engine."""
        self._is_running = False
        self.worker.stop_engine()

    def set_min_confidence(self, val: float) -> None:
        """Updates min confidence threshold."""
        self.min_confidence = val
        self.worker.set_min_confidence(val)

    def shutdown(self) -> None:
        """Gracefully terminates worker and background thread."""
        self.stop_engine()
        self.worker_thread.quit()
        if not self.worker_thread.wait(2000):
            logger.warning("Worker thread did not terminate within timeout; terminating.")
            self.worker_thread.terminate()
            self.worker_thread.wait(500)
