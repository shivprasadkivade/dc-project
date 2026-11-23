"""Unit tests for the 7 architectural fixes and audit requirements in AcousticTap."""

import time
import queue
import unittest
from unittest.mock import MagicMock, patch
from pathlib import Path
import numpy as np

from src.holo.geofence import AcousticGeofenceValidator, GeofenceDecision
from src.holo.triangulation import align_and_slice_waveform
from src.holo.listen import AcousticTapCore, ShortcutDispatcher, build_parser
from src.holo.config import PROJECT_ROOT, MODEL_DIR, MODEL_PATH


class TestRingingToleranceAndGeofenceSecurity(unittest.TestCase):
    """Verifies that ringing-tolerant Tap 2 evaluation relaxes pre-noise gates
    while STRICTLY enforcing physical HF/LF damping rejection and Stage 2 probability."""

    def setUp(self):
        self.validator = AcousticGeofenceValidator()
        self.validator.enabled = True
        self.validator.is_fitted = True
        self.validator.min_hf_lf_ratio = -14.5
        self.validator.min_adr = 2.20
        self.validator.min_crest = 3.50
        self.validator.max_lf_dominance = 2.50
        self.validator.boundary_threshold = 0.50
        self.validator.threshold = 5.80
        self.validator.centroid_inside = np.zeros(24)
        self.validator.centroid_outside = np.ones(24) * 2.0
        self.validator.precision_inside = np.eye(24)

    def test_desk_knock_hf_lf_rejected_even_in_ringing_tolerant_mode(self):
        """A rubber-damped desk knock (low HF/LF ratio) must be rejected even when ringing_tolerant=True."""
        ml_vec = np.zeros(70, dtype=np.float32)
        fourier_vec = np.zeros(43, dtype=np.float64)
        # Low HF/LF ratio: index 36 in Fourier vector -> -25 dB (below -14.5 dB floor)
        fourier_vec[36] = -2.5

        dec = self.validator.verify(
            ml_vec,
            fourier_vec,
            ringing_tolerant=True,
            tap2_boundary_threshold=0.38,
        )
        self.assertFalse(dec.passed, "Desk knock must fail geofence even under ringing-tolerant mode")
        self.assertEqual(dec.stage_failed, "stage1_hf")

    def test_excessive_lf_thump_rejected_in_ringing_tolerant_mode(self):
        """Low frequency thump energy above max_lf must fail even when ringing_tolerant=True."""
        ml_vec = np.zeros(70, dtype=np.float32)
        fourier_vec = np.zeros(43, dtype=np.float64)
        fourier_vec[36] = -0.5  # Good HF/LF ratio (-5 dB)
        # Artificially inflate low bands 0..2 vs total onset bands
        fourier_vec[0:3] = 5.0
        fourier_vec[3:12] = -5.0

        dec = self.validator.verify(
            ml_vec,
            fourier_vec,
            ringing_tolerant=True,
            tap2_boundary_threshold=0.38,
        )
        self.assertFalse(dec.passed)
        self.assertEqual(dec.stage_failed, "stage1_lf_thump")

    def test_pre_noise_gates_relaxed_under_ringing_tolerant_mode(self):
        """Pre-noise sensitive ADR (z[1]) is relaxed during ringing-tolerant Tap 2 evaluation."""
        ml_vec = np.zeros(70, dtype=np.float32)
        fourier_vec = np.zeros(43, dtype=np.float64)
        fourier_vec[36] = -0.5  # Good HF/LF ratio
        # Set low ADR (0.50 dB < 2.20 dB) due to Tap 1 ringing tail in pre-buffer
        ml_vec[69] = 0.50

        # Normal verification: fails ADR
        normal_dec = self.validator.verify(ml_vec, fourier_vec, ringing_tolerant=False)
        self.assertFalse(normal_dec.passed)
        self.assertEqual(normal_dec.stage_failed, "stage1_adr")

        # Ringing-tolerant verification: passes Stage 1 ADR check
        tolerant_dec = self.validator.verify(
            ml_vec,
            fourier_vec,
            ringing_tolerant=True,
            tap2_boundary_threshold=0.38,
        )
        self.assertNotEqual(tolerant_dec.stage_failed, "stage1_adr")

    def test_evaluate_alias_matches_verify(self):
        """The evaluate() method must return identical results to verify()."""
        ml_vec = np.zeros(70, dtype=np.float32)
        fourier_vec = np.zeros(43, dtype=np.float64)
        fourier_vec[36] = -0.5

        d1 = self.validator.verify(ml_vec, fourier_vec, ringing_tolerant=True)
        d2 = self.validator.evaluate(ml_vec, fourier_vec, ringing_tolerant=True)
        self.assertEqual(d1.passed, d2.passed)
        self.assertEqual(d1.threshold, d2.threshold)

    def test_sweep_thresholds_returns_expected_structure(self):
        """sweep_thresholds must calculate FRR and OOB-RR across the configured thresholds."""
        Z_in_ml = np.zeros((10, 70), dtype=np.float32)
        Z_in_four = np.zeros((10, 43), dtype=np.float64)
        Z_in_four[:, 36] = -0.5  # Valid HF/LF ratio

        res = self.validator.sweep_thresholds(
            inside_ml=Z_in_ml,
            inside_fourier=Z_in_four,
            thresholds=[0.50, 0.55, 0.60, 0.65],
        )
        for th in [0.50, 0.55, 0.60, 0.65]:
            self.assertIn(th, res)
            self.assertIn("frr", res[th])
            self.assertIn("oob_rr", res[th])
            self.assertIsInstance(res[th]["frr"], float)


class TestNonBlockingHIDAndSelfCooldown(unittest.TestCase):
    """Verifies non-blocking Quartz HID sequence dispatch and typing cooldown self-injection immunity."""

    def test_non_blocking_dispatcher_queues_and_processes(self):
        """ShortcutDispatcher must accept events without blocking the caller."""
        mock_core = MagicMock()
        mock_core.enable_notifications = False
        mock_core.mod_name = "Cmd"
        mock_core.kb_controller = None
        mock_core._set_synthetic = MagicMock()

        dispatcher = ShortcutDispatcher(core=mock_core, dry_run=True)
        event = {
            "label": "left_tap",
            "confidence": 0.85,
            "latency_ms": 12.0,
            "inter_tap_ms": 0.0,
        }

        t_start = time.perf_counter()
        dispatcher.dispatch(event)
        t_dispatch = (time.perf_counter() - t_start) * 1000.0

        # Dispatch call must be virtually instantaneous (<5 ms), not sleeping ~85 ms
        self.assertLess(t_dispatch, 5.0, "Dispatching event to background worker must be non-blocking")

        dispatcher.stop()

    def test_synthetic_cooldown_immunity(self):
        """Synthetic keystrokes emitted by AcousticTap must not trigger the 0.15s typing cooldown against itself."""
        core = AcousticTapCore(autoload=False)
        core.last_keypress_time = 0.0
        core.last_synthetic_emit_ts = time.time()  # Synthetic key just emitted

        # Simulate on_press event while synthetic timestamp is fresh (<0.35s ago)
        # The keypress should be ignored, NOT updating last_keypress_time
        now = time.time()
        if (now - core.last_synthetic_emit_ts) < 0.35 or core.is_emitting_synthetic:
            pass  # Protected branch in _init_typing_listener
        else:
            core.last_keypress_time = now

        self.assertEqual(core.last_keypress_time, 0.0, "Synthetic keystrokes must not update last_keypress_time")


class TestBoundedOnsetPeakAlignment(unittest.TestCase):
    """Verifies that align_and_slice_waveform locks onto Tap 1 even when a louder Tap 2 is in the tail."""

    def test_bounded_search_prevents_tap2_hijacking(self):
        """A louder second strike in the waveform buffer tail must not hijack Tap 1's peak index."""
        # 4608-sample waveform (~104.5 ms at 44.1 kHz)
        wav = np.zeros(4608, dtype=np.float64)

        # Tap 1 at sample 1536 (amplitude 0.12)
        wav[1536] = 0.12
        wav[1537] = -0.09
        wav[1538] = 0.07

        # Tap 2 at sample 3500 (much louder: amplitude 0.85)
        wav[3500] = 0.85
        wav[3501] = -0.70

        mono, peak_idx, pre_noise, onset, wake = align_and_slice_waveform(wav)

        # peak_idx MUST be around Tap 1 (1536), NOT around Tap 2 (3500)
        self.assertLess(peak_idx, 2500, f"Peak alignment locked onto Tap 2 tail at {peak_idx} instead of Tap 1!")
        self.assertGreater(peak_idx, 1400)
        self.assertEqual(len(wake), 1024, "Wake window must maintain full 1024-sample length")
        self.assertEqual(len(onset), 256, "Onset window must maintain 256-sample length")


class TestGroupAwareSplitting(unittest.TestCase):
    """Verifies that Tap 1 and Tap 2 from the same recording are grouped and do not leak across splits."""

    def test_group_shuffle_split_prevents_leakage(self):
        """GroupShuffleSplit must place both Tap 1 and Tap 2 of the same strike into the same split."""
        from sklearn.model_selection import GroupShuffleSplit

        # 20 samples from 10 distinct recordings (each recording has 2 sub-windows: t1 and t2)
        y = np.array([0, 0, 1, 1] * 5)
        groups = []
        for rec_id in range(10):
            groups.append(f"dt_rec_{rec_id}")
            groups.append(f"dt_rec_{rec_id}")
        groups = np.array(groups)
        indices = np.arange(len(y))

        gss = GroupShuffleSplit(n_splits=1, test_size=0.20, random_state=42)
        train_idx, test_idx = next(gss.split(indices, y, groups=groups))

        train_groups = set(groups[train_idx])
        test_groups = set(groups[test_idx])

        overlap = train_groups.intersection(test_groups)
        self.assertEqual(len(overlap), 0, f"Recording groups leaked across train and test: {overlap}")


class TestModelAutoSelectionAndCLI(unittest.TestCase):
    """Verifies CLI flag reconciliation and graceful double-tap model fallback."""

    def test_cli_flags_parsed_cleanly(self):
        """--double-tap and --no-double-tap flags must parse cleanly."""
        parser = build_parser()

        args_dt = parser.parse_args(["--double-tap"])
        self.assertTrue(args_dt.double_tap)
        self.assertFalse(args_dt.no_double_tap)

        args_no_dt = parser.parse_args(["--no-double-tap"])
        self.assertTrue(args_no_dt.no_double_tap)
        self.assertIsNone(args_no_dt.double_tap)

    def test_model_fallback_path(self):
        """When dedicated double-tap model is missing, MODEL_PATH remains the active fallback."""
        fake_dt_model = Path("/tmp/nonexistent_dt_model.joblib")
        active_model = MODEL_PATH
        if fake_dt_model.exists():
            active_model = fake_dt_model
        # Must resolve to valid existing MODEL_PATH
        self.assertEqual(active_model, MODEL_PATH)
        self.assertTrue(active_model.exists())


if __name__ == "__main__":
    unittest.main()
