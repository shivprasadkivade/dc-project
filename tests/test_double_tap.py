"""Unit tests for AcousticTap double-tap state machine, peak detection, and detector refractory tuning."""

import unittest
import numpy as np

from src.holo.double_tap import (
    DoubleTapStateMachine,
    DoubleTapEvent,
    find_double_tap_peaks,
    extract_tap_subwindow,
)
from src.holo.detector import ImpulseDetector
from src.holo.config import SAMPLE_RATE


class TestDoubleTapStateMachine(unittest.TestCase):
    """Test suite for DoubleTapStateMachine time-windowed logic."""

    def setUp(self):
        self.sm = DoubleTapStateMachine(
            min_inter_tap_ms=80.0,
            max_inter_tap_ms=350.0,
            require_same_zone=True,
            actions={"double_left": "x", "double_right": "z"},
        )

    def test_single_tap_timeout_emission(self):
        """A single tap must buffer initially and emit as a single tap once max_inter_tap_ms expires."""
        tap1 = {"timestamp": 1000.0, "label": "left_tap", "confidence": 0.88}
        emitted = self.sm.process_tap(tap1)
        self.assertEqual(len(emitted), 0, "Tap 1 should be buffered while awaiting potential Tap 2")

        # Before timeout (<350 ms)
        early = self.sm.check_timeout(now=1000.20)
        self.assertEqual(len(early), 0, "No event should be emitted before timeout")

        # After timeout (>=350 ms)
        timeout_events = self.sm.check_timeout(now=1000.36)
        self.assertEqual(len(timeout_events), 1)
        ev = timeout_events[0]
        self.assertEqual(ev["label"], "left_tap")
        self.assertFalse(ev["is_double"])
        self.assertAlmostEqual(ev["confidence"], 0.88)
        self.assertEqual(ev["action_key"], "c")
        self.assertEqual(ev["action_name"], "Copy")

    def test_valid_double_left_detection(self):
        """Two left taps separated by 150 ms (within 80-350 ms) must trigger double_left."""
        tap1 = {"timestamp": 1000.0, "label": "left_tap", "confidence": 0.80}
        tap2 = {"timestamp": 1000.15, "label": "left_tap", "confidence": 0.90}

        self.sm.process_tap(tap1)
        emitted = self.sm.process_tap(tap2)

        self.assertEqual(len(emitted), 1)
        ev = emitted[0]
        self.assertEqual(ev["label"], "double_left")
        self.assertTrue(ev["is_double"])
        self.assertAlmostEqual(ev["confidence"], 0.85)
        self.assertAlmostEqual(ev["inter_tap_ms"], 150.0, places=1)
        self.assertEqual(ev["action_key"], "x")
        self.assertEqual(ev["action_name"], "Cut")

    def test_valid_double_right_detection(self):
        """Two right taps separated by 200 ms must trigger double_right."""
        tap1 = {"timestamp": 1000.0, "label": "right_tap", "confidence": 0.76}
        tap2 = {"timestamp": 1000.20, "label": "right_tap", "confidence": 0.84}

        self.sm.process_tap(tap1)
        emitted = self.sm.process_tap(tap2)

        self.assertEqual(len(emitted), 1)
        ev = emitted[0]
        self.assertEqual(ev["label"], "double_right")
        self.assertTrue(ev["is_double"])
        self.assertAlmostEqual(ev["confidence"], 0.80)
        self.assertAlmostEqual(ev["inter_tap_ms"], 200.0, places=1)
        self.assertEqual(ev["action_key"], "z")
        self.assertEqual(ev["action_name"], "Undo")

    def test_echo_rejection_below_min_inter_tap(self):
        """Transients arriving <80 ms after Tap 1 must be ignored as acoustic chassis ringing."""
        tap1 = {"timestamp": 1000.0, "label": "left_tap", "confidence": 0.85}
        rebound = {"timestamp": 1000.045, "label": "left_tap", "confidence": 0.92}  # 45 ms (< 80 ms)
        tap2 = {"timestamp": 1000.170, "label": "left_tap", "confidence": 0.87}     # 170 ms (valid)

        self.sm.process_tap(tap1)
        rebound_emitted = self.sm.process_tap(rebound)
        self.assertEqual(len(rebound_emitted), 0, "Chassis ringing at 45 ms must be dropped")

        # Legitimate tap 2 arrives
        valid_emitted = self.sm.process_tap(tap2)
        self.assertEqual(len(valid_emitted), 1)
        self.assertEqual(valid_emitted[0]["label"], "double_left")
        self.assertAlmostEqual(valid_emitted[0]["inter_tap_ms"], 170.0, places=1)

    def test_mismatched_zone_handling(self):
        """A left tap followed by a right tap within 150 ms must emit Tap 1 and buffer Tap 2 without hanging."""
        tap1 = {"timestamp": 1000.0, "label": "left_tap", "confidence": 0.85}
        tap2 = {"timestamp": 1000.15, "label": "right_tap", "confidence": 0.90}

        self.sm.process_tap(tap1)
        emitted = self.sm.process_tap(tap2)

        # Should immediately flush Tap 1 as single tap
        self.assertEqual(len(emitted), 1)
        self.assertEqual(emitted[0]["label"], "left_tap")
        self.assertFalse(emitted[0]["is_double"])

        # Tap 2 should now be buffered and time out to single right tap
        timeout_events = self.sm.check_timeout(now=1000.52)  # 370 ms after tap2
        self.assertEqual(len(timeout_events), 1)
        self.assertEqual(timeout_events[0]["label"], "right_tap")
        self.assertFalse(timeout_events[0]["is_double"])

    def test_flush_and_reset(self):
        """flush() emits buffered tap immediately; reset() clears without emitting."""
        tap1 = {"timestamp": 1000.0, "label": "left_tap", "confidence": 0.85}
        self.sm.process_tap(tap1)
        flushed = self.sm.flush()
        self.assertEqual(len(flushed), 1)
        self.assertEqual(flushed[0]["label"], "left_tap")

        self.sm.process_tap(tap1)
        self.sm.reset()
        self.assertEqual(len(self.sm.check_timeout(now=2000.0)), 0)


class TestImpulseDetectorDoubleTapMode(unittest.TestCase):
    """Test suite verifying ImpulseDetector refractory tuning for double tap."""

    def test_detector_cooldown_single_vs_double_mode(self):
        """ImpulseDetector in double_tap_mode must have reduced refractory period (~70-80 ms) vs single mode (~290 ms)."""
        single_det = ImpulseDetector(double_tap_mode=False)
        double_det = ImpulseDetector(double_tap_mode=True, min_inter_tap_ms=80.0)

        # Single tap mode cooldown blocks default to 25 blocks (~290 ms)
        self.assertEqual(single_det.cooldown_blocks, 25)

        # Double tap mode refractory period = (post_blocks + cooldown_blocks) * 11.61 ms
        block_ms = (512 / 44100.0) * 1000.0
        refractory_ms = (double_det.post_blocks + double_det.cooldown_blocks) * block_ms

        self.assertGreaterEqual(refractory_ms, 70.0, "Refractory lockout must block <60 ms rebound")
        self.assertLessEqual(refractory_ms, 90.0, "Refractory lockout must re-arm in time for >=80 ms Tap 2")


class TestPeakDetectionAndSubwindow(unittest.TestCase):
    """Test suite for double-tap peak locator and subwindow extractor."""

    def test_find_double_tap_peaks_synthetic(self):
        """find_double_tap_peaks accurately locates two transient spikes and computes inter_tap_ms."""
        sr = 44100
        length = int(0.65 * sr)
        waveform = np.zeros(length, dtype=np.float32)

        # Create spike 1 at sample 2000
        p1_expected = 2000
        waveform[p1_expected] = 0.50
        waveform[p1_expected + 1] = 0.30

        # Create spike 2 150 ms later (6615 samples)
        gap_samples = int(0.150 * sr)
        p2_expected = p1_expected + gap_samples
        waveform[p2_expected] = 0.45
        waveform[p2_expected + 1] = 0.25

        p1, p2, dt_ms = find_double_tap_peaks(waveform, sr=sr, min_gap_ms=80.0, max_gap_ms=350.0, threshold=0.01)

        self.assertIsNotNone(p1)
        self.assertIsNotNone(p2)
        self.assertAlmostEqual(p1, p1_expected, delta=10)
        self.assertAlmostEqual(p2, p2_expected, delta=10)
        self.assertAlmostEqual(dt_ms, 150.0, delta=2.0)

    def test_extract_tap_subwindow_bounds(self):
        """extract_tap_subwindow returns properly sized array even near boundaries."""
        waveform = np.ones(10000, dtype=np.float32)
        sub = extract_tap_subwindow(waveform, peak_idx=2000, window_samples=4096, pre_peak_samples=1024)
        self.assertEqual(len(sub), 4096)
        self.assertEqual(sub.dtype, np.float32)


if __name__ == "__main__":
    unittest.main()
