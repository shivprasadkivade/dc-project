"""Tests for EngineBridge and background worker thread-safe event consumption."""

from __future__ import annotations

import os
import queue
import time
import unittest
from unittest.mock import MagicMock
import numpy as np

# Ensure offscreen platform for Qt
os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from src.holo.ui.engine_bridge import EngineBridge, EngineWorker
from src.holo.ui.audio_adapter import SplitAudioEngine


class TestEngineBridge(unittest.TestCase):
    """Test suite for EngineBridge, EngineWorker, and queue consumers."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_worker_initialization_and_min_confidence(self) -> None:
        """Worker initializes with default confidence and updates properly."""
        worker = EngineWorker(min_confidence=0.75)
        self.assertAlmostEqual(worker.min_confidence, 0.75)

        worker.set_min_confidence(0.85)
        self.assertAlmostEqual(worker.min_confidence, 0.85)

        # Clamping out-of-range values
        worker.set_min_confidence(1.5)
        self.assertAlmostEqual(worker.min_confidence, 1.0)
        worker.set_min_confidence(-0.2)
        self.assertAlmostEqual(worker.min_confidence, 0.0)

    def test_synthetic_audio_chunk_drain(self) -> None:
        """Continuous audio chunks pushed into secondary_queue are drained and emitted."""
        worker = EngineWorker()
        worker._is_active = True
        worker.engine = MagicMock()
        worker.engine.secondary_queue = queue.Queue()
        worker.engine.detector.event_queue = queue.Queue()

        received_chunks = []
        worker.audio_chunk_ready.connect(lambda ch: received_chunks.append(ch))

        # Push 3 synthetic audio chunks
        chunk1 = np.ones(512, dtype=np.float32) * 0.1
        chunk2 = np.ones(512, dtype=np.float32) * 0.2
        chunk3 = np.ones(512, dtype=np.float32) * 0.3

        worker.engine.secondary_queue.put(chunk1)
        worker.engine.secondary_queue.put(chunk2)
        worker.engine.secondary_queue.put(chunk3)

        # Drain one batch
        worker._drain_queues()

        self.assertEqual(len(received_chunks), 3)
        np.testing.assert_array_almost_equal(received_chunks[0], chunk1)
        np.testing.assert_array_almost_equal(received_chunks[2], chunk3)

    def test_synthetic_tap_event_uncalibrated_mode(self) -> None:
        """When no model is loaded, tap waveform emits uncalibrated_tap event."""
        worker = EngineWorker(min_confidence=0.60)
        worker._is_active = True
        worker.model = None  # Uncalibrated mode
        worker.engine = MagicMock()
        worker.engine.secondary_queue = queue.Queue()
        worker.engine.detector.event_queue = queue.Queue()

        received_taps = []
        worker.tap_detected.connect(lambda ev: received_taps.append(ev))

        # Synthetic transient waveform (128 ms at 44.1 kHz = 5632 samples)
        synth_waveform = np.zeros(5632, dtype=np.float32)
        synth_waveform[1000] = 0.85
        synth_waveform[1001] = -0.60

        worker.engine.detector.event_queue.put(synth_waveform)
        worker._drain_queues()

        self.assertEqual(len(received_taps), 1)
        event = received_taps[0]
        self.assertEqual(event["label"], "uncalibrated_tap")
        self.assertTrue(event["meets_min_confidence"])
        self.assertTrue(event["geofence_passed"])
        self.assertAlmostEqual(event["confidence"], 1.0)
        self.assertIn("Raw Peak: 0.85", event["breakdown"])

    def test_min_confidence_filtering(self) -> None:
        """Tap classification below min_confidence threshold must flag meets_min_confidence=False."""
        worker = EngineWorker(min_confidence=0.80)
        worker._is_active = True

        # Mock model and geofence
        mock_model = MagicMock()
        # Return left_tap with 0.70 confidence (< 0.80 required)
        mock_model.predict_proba.return_value = np.array([[0.70, 0.20, 0.10]])
        worker.model = mock_model
        worker.classes = ["left_tap", "right_tap", "noise"]
        worker.class_thresholds = {"left_tap": 0.50}  # lower than ui.min_confidence 0.80

        mock_geofence = MagicMock()
        gf_decision = MagicMock()
        gf_decision.passed = True
        gf_decision.reason = "inside palm rest boundary"
        gf_decision.prob_inside = 0.95
        gf_decision.dist_inside = 1.2
        gf_decision.dist_outside = 4.0
        gf_decision.threshold = 5.0
        gf_decision.stage_failed = None
        mock_geofence.verify.return_value = gf_decision
        worker.geofence = mock_geofence

        received_taps = []
        worker.tap_detected.connect(lambda ev: received_taps.append(ev))

        synth_waveform = np.zeros(5632, dtype=np.float32)
        worker._process_tap_waveform(synth_waveform)

        self.assertEqual(len(received_taps), 1)
        ev = received_taps[0]
        self.assertEqual(ev["label"], "left_tap")
        self.assertAlmostEqual(ev["confidence"], 0.70)
        self.assertFalse(ev["meets_min_confidence"])  # 0.70 < 0.80
        self.assertFalse(ev["action_taken"])

    def test_geofence_blocked_event(self) -> None:
        """Geofence failure marks geofence_passed=False and action_taken=False."""
        worker = EngineWorker(min_confidence=0.60)
        worker._is_active = True

        mock_model = MagicMock()
        mock_model.predict_proba.return_value = np.array([[0.95, 0.03, 0.02]])
        worker.model = mock_model
        worker.classes = ["left_tap", "right_tap", "noise"]

        mock_geofence = MagicMock()
        gf_decision = MagicMock()
        gf_decision.passed = False
        gf_decision.reason = "HF/LF ratio too low - desk bump"
        gf_decision.prob_inside = 0.05
        gf_decision.dist_inside = 9.0
        gf_decision.dist_outside = 1.0
        gf_decision.threshold = 5.0
        gf_decision.stage_failed = "stage1_hf"
        mock_geofence.verify.return_value = gf_decision
        worker.geofence = mock_geofence

        received_taps = []
        received_gf = []
        worker.tap_detected.connect(lambda ev: received_taps.append(ev))
        worker.geofence_event.connect(lambda gfe: received_gf.append(gfe))

        synth_waveform = np.zeros(5632, dtype=np.float32)
        worker._process_tap_waveform(synth_waveform)

        self.assertEqual(len(received_taps), 1)
        self.assertEqual(len(received_gf), 1)
        ev = received_taps[0]
        self.assertFalse(ev["geofence_passed"])
        self.assertIn("HF/LF ratio too low", ev["geofence_reason"])
        self.assertFalse(ev["action_taken"])

    def test_bridge_start_stop_and_shutdown(self) -> None:
        """EngineBridge coordinates start, stop, and clean background thread shutdown."""
        bridge = EngineBridge()
        self.assertFalse(bridge.is_running)

        # Start and stop safely
        bridge.start_engine()
        self.assertTrue(bridge.is_running)

        bridge.stop_engine()
        self.assertFalse(bridge.is_running)

        # Shutdown terminates thread cleanly
        bridge.shutdown()
        self.assertFalse(bridge.worker_thread.isRunning())


if __name__ == "__main__":
    unittest.main()
