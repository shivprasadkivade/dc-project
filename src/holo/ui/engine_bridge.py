"""Engine bridge connecting Holo core audio processing and geofencing to Qt UI.

Runs the audio engine and queue drain loop inside a dedicated background QThread,
emitting typed Qt Signals to cleanly decouple UI views from backend internals.
Uses the unified AcousticTapCore and SplitAudioEngine backends shared with holo.listen.
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

from src.holo.config import CFG, MODEL_PATH, SAMPLE_RATE, BLOCK_SIZE, RMS_THRESHOLD
from src.holo.detector import ImpulseDetector, get_input_channels
from src.holo.audio_manager import SplitAudioEngine
from src.holo.listen import AcousticTapCore, emit_shortcut, show_toast

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
        min_confidence: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.model_path = model_path or MODEL_PATH
        self.min_confidence = min_confidence

        self.core = AcousticTapCore(
            model_file=self.model_path,
            confidence_min=self.min_confidence,
            autoload=False,
        )

        self.detector = ImpulseDetector(threshold=RMS_THRESHOLD)
        self.secondary_queue: queue.Queue = queue.Queue(maxsize=200)
        self.engine: Optional[SplitAudioEngine] = None

        self._drain_timer = QTimer()
        self._drain_timer.setInterval(25)
        self._drain_timer.timeout.connect(self._drain_queues)

        self.enable_shortcuts: bool = False
        self._is_active: bool = False

    # Backwards-compatible properties mapping to self.core
    @property
    def model(self):
        return self.core.model

    @model.setter
    def model(self, val):
        self.core.model = val

    @property
    def triangulator(self):
        return self.core.triangulator

    @triangulator.setter
    def triangulator(self, val):
        self.core.triangulator = val

    @property
    def geofence(self):
        return self.core.geofence

    @geofence.setter
    def geofence(self, val):
        self.core.geofence = val

    @property
    def classes(self):
        return self.core.classes

    @classes.setter
    def classes(self, val):
        self.core.classes = val

    @property
    def class_thresholds(self):
        return self.core.class_thresholds

    @class_thresholds.setter
    def class_thresholds(self, val):
        self.core.class_thresholds = val

    @property
    def fourier_weight(self):
        return self.core.fourier_weight

    @fourier_weight.setter
    def fourier_weight(self, val):
        self.core.fourier_weight = val

    @property
    def feature_set(self):
        return self.core.feature_set

    @feature_set.setter
    def feature_set(self, val):
        self.core.feature_set = val

    def load_model(self, path: Optional[Path] = None) -> bool:
        """Loads and prepares trained model, triangulator, and geofence."""
        success = self.core.load_model(path or self.model_path)
        if not success:
            logger.info("Model file not found at %s. Running in uncalibrated mode.", path or self.model_path)
        return success

    def _audio_callback(self, indata: np.ndarray, frames: int, time_info: Any, status: Any) -> None:
        """Servicing both transient impulse detector and live visualizer audio chunks."""
        # 1. Feed detector
        self.detector.audio_callback(indata, frames, time_info, status)

        # 2. Extract mono audio chunk for visualizer
        chunk = indata[:, 0].copy() if (indata.ndim == 2 and indata.shape[1] > 0) else indata.flatten().copy()
        try:
            self.secondary_queue.put_nowait(chunk)
        except queue.Full:
            try:
                self.secondary_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self.secondary_queue.put_nowait(chunk)
            except queue.Full:
                pass

    @Slot(object, object, bool)
    def start_engine(
        self,
        primary_device: Optional[int | str] = None,
        secondary_device: Optional[int | str] = None,
        enable_shortcuts: bool = False,
    ) -> None:
        """Starts audio engine and queue drain timer."""
        if self._is_active:
            return

        self.enable_shortcuts = enable_shortcuts
        self.load_model()

        try:
            channels = get_input_channels()
            self.engine = SplitAudioEngine(
                tap_callback=self._audio_callback,
                tap_device_pref=primary_device,
                secondary_device_pref=secondary_device,
                capture_secondary=False,  # strictly keep Bluetooth headphones untouched
                auto_reconnect=True,
                sample_rate=SAMPLE_RATE,
                block_size=BLOCK_SIZE,
                channels=channels,
            )
            # Attach references for mocks in tests
            self.engine.detector = self.detector
            self.engine.secondary_queue = self.secondary_queue

            self.engine.start()
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
        self.core.confidence_min = self.min_confidence

    @Slot(bool)
    def set_enable_shortcuts(self, enabled: bool) -> None:
        """Updates shortcut macro trigger state dynamically."""
        self.enable_shortcuts = enabled

    def _drain_queues(self) -> None:
        """Non-blocking queue consumer driven by QTimer."""
        if not self._is_active or self.engine is None:
            return

        # 1. Drain continuous audio chunks from secondary_queue (bounded batch)
        chunks_drained = 0
        sec_q = getattr(self.engine, "secondary_queue", self.secondary_queue)
        while chunks_drained < 4:
            try:
                chunk = sec_q.get_nowait()
                self.audio_chunk_ready.emit(chunk)
                chunks_drained += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.debug("Error draining audio chunk: %s", e)
                break

        # 2. Drain transient events from detector.event_queue (bounded batch)
        events_drained = 0
        det = getattr(self.engine, "detector", self.detector)
        while events_drained < 5:
            try:
                waveform = det.event_queue.get_nowait()
                self._process_tap_waveform(waveform)
                events_drained += 1
            except queue.Empty:
                break
            except Exception as e:
                logger.error("Error processing tap event: %s", e)
                break

    def _process_tap_waveform(self, waveform: np.ndarray) -> None:
        """Extracts features, performs geofence verification, and predicts gesture."""
        try:
            override_conf = self.min_confidence
            res = self.core.evaluate_waveform(
                waveform=waveform,
                enable_shortcuts=self.enable_shortcuts,
                dry_run=False,
                override_confidence=override_conf,
            )

            # If active typing suppression was triggered, drop silently without UI flicker
            if res.get("ignored_typing"):
                return

            self.tap_detected.emit(res)
            gf_details = res.get("geofence_details")
            if gf_details:
                self.geofence_event.emit(gf_details)

        except Exception as e:
            logger.error("Error evaluating tap transient: %s", e)
            self.error_occurred.emit(f"Evaluation error: {e}")

    def _trigger_macro(self, pred_label: str) -> None:
        """Triggers keyboard shortcut with full Quartz sequence and toast notification."""
        try:
            emit_shortcut(
                action_label=pred_label,
                conf_pct=100.0,
                latency_ms=0.0,
                dry_run=False,
                enable_notifications=True,
            )
        except Exception as ex:
            logger.warning("Could not execute keyboard macro: %s", ex)


class EngineBridge(QObject):
    """Facade for Qt views to control EngineWorker across thread boundaries."""

    # Re-emitted typed signals from worker
    tap_detected = Signal(dict)
    geofence_event = Signal(dict)
    audio_chunk_ready = Signal(np.ndarray)
    engine_state_changed = Signal(bool, str)
    error_occurred = Signal(str)

    def __init__(
        self,
        model_path: Optional[Path] = None,
        min_confidence: Optional[float] = None,
        parent: Optional[QObject] = None,
    ) -> None:
        super().__init__(parent)
        self.min_confidence = min_confidence

        self.worker_thread = QThread()
        self.worker = EngineWorker(model_path=model_path, min_confidence=min_confidence)
        self.worker.moveToThread(self.worker_thread)

        # Forward worker signals to bridge signals
        self.worker.tap_detected.connect(self.tap_detected)
        self.worker.geofence_event.connect(self.geofence_event)
        self.worker.audio_chunk_ready.connect(self.audio_chunk_ready)
        self.worker.engine_state_changed.connect(self.engine_state_changed)
        self.worker.error_occurred.connect(self.error_occurred)

        self.worker_thread.start()
        self._is_running = False

    def __del__(self) -> None:
        try:
            self.shutdown()
        except Exception:
            pass

    @property
    def is_running(self) -> bool:
        return self._is_running

    def start_engine(
        self,
        primary_device: Optional[int | str] = None,
        secondary_device: Optional[int | str] = None,
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

    def set_enable_shortcuts(self, enabled: bool) -> None:
        """Updates shortcut macro trigger state."""
        self.worker.set_enable_shortcuts(enabled)

    def shutdown(self) -> None:
        """Gracefully terminates worker and background thread."""
        self.stop_engine()
        if hasattr(self, "worker_thread") and self.worker_thread.isRunning():
            self.worker_thread.quit()
            if not self.worker_thread.wait(2000):
                logger.warning("Worker thread did not terminate within timeout; terminating.")
                self.worker_thread.terminate()
                self.worker_thread.wait(500)
