"""Tests for Holo UI configuration store and surgical line-preserving persistence."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from src.holo.ui.config_store import UIConfigStore, DEFAULT_UI_CONFIG

SAMPLE_CONFIG_WITH_COMMENTS = """# =====================================================================
# AcousticTap Master Configuration (config/config.yaml)
# Edit any value here to tune detection, features, models, or shortcuts
# without touching the Python source code.
# =====================================================================

audio:
  sample_rate: 44100
  block_size: 512               # 512 samples ≈ 11.6 ms per block
  pre_blocks: 3                 # Blocks captured BEFORE impact spike
  post_blocks: 8                # Blocks captured AFTER impact spike
  cooldown_blocks: 25           # Cooldown blocks (~290 ms) to prevent double triggers
  channels: 1                   # Default 1 (mono) for Mac and single-channel hardware; set 2 for stereo

detector:
  rms_threshold: 0.02          # Base microphone RMS trigger threshold
  peak_multiplier: 1.8          # Peak amplitude must exceed rms_threshold * peak_multiplier
  bg_snr_ratio: 2.3             # RMS must exceed rolling background floor * bg_snr_ratio
  bg_ema_alpha: 0.95            # Exponential moving average decay for background noise floor

geofence:
  enabled: true
  boundary_threshold: 0.62
  min_hf_lf_ratio_db: -14.5
  min_attack_decay_ratio: 3.50   # Updated for proper 10*log10 dB scale
  min_crest_factor: 3.80
  dispersion_window_ms: [-5.0, 3.0]
  max_inside_distance: 5.80
  surface_tolerance_scale: 1.15
  class_conditional: true
"""


class TestUIConfigStore(unittest.TestCase):
    """Test suite verifying safe config read/write and byte-for-byte line preservation."""

    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.temp_config_path = Path(self.temp_dir.name) / "config.yaml"
        self.temp_config_path.write_text(SAMPLE_CONFIG_WITH_COMMENTS, encoding="utf-8")
        self.store = UIConfigStore(config_path=self.temp_config_path)

    def tearDown(self) -> None:
        self.temp_dir.cleanup()

    def test_get_ui_config_defaults_when_missing(self) -> None:
        """When ui: block is absent, defaults should be cleanly returned."""
        cfg = self.store.get_ui_config()
        self.assertEqual(cfg["theme"], "system")
        self.assertTrue(cfg["show_spectrogram"])
        self.assertAlmostEqual(cfg["min_confidence"], 0.65)

    def test_get_core_config_excludes_ui(self) -> None:
        """Core config dictionary should exclude ui: section and contain core sections."""
        core = self.store.get_core_config()
        self.assertIn("audio", core)
        self.assertIn("detector", core)
        self.assertIn("geofence", core)
        self.assertNotIn("ui", core)
        self.assertEqual(core["audio"]["sample_rate"], 44100)

    def test_save_ui_config_appends_and_preserves_bytes(self) -> None:
        """Saving ui: to a config lacking it must leave all previous bytes 100% identical."""
        original_bytes = self.temp_config_path.read_bytes()

        # Save UI config
        new_ui = {
            "theme": "dark",
            "show_spectrogram": False,
            "min_confidence": 0.80,
        }
        self.store.save_ui_config(new_ui)

        updated_bytes = self.temp_config_path.read_bytes()
        # The beginning of updated_bytes must be byte-for-byte identical to original_bytes
        self.assertTrue(updated_bytes.startswith(original_bytes))

        # Check that UI section now reads back accurately
        read_back = self.store.get_ui_config()
        self.assertEqual(read_back["theme"], "dark")
        self.assertFalse(read_back["show_spectrogram"])
        self.assertAlmostEqual(read_back["min_confidence"], 0.80)

        # Core config must remain intact
        core = self.store.get_core_config()
        self.assertEqual(core["audio"]["sample_rate"], 44100)
        self.assertEqual(core["geofence"]["boundary_threshold"], 0.62)

    def test_update_existing_ui_config_preserves_preceding_sections(self) -> None:
        """Subsequent saves must only update the ui: section and preserve everything above."""
        # First save
        self.store.save_ui_config({"theme": "dark", "min_confidence": 0.70})

        # Capture content before the ui: block
        full_text = self.temp_config_path.read_text(encoding="utf-8")
        ui_idx = full_text.index("ui:")
        prefix_before_ui = full_text[:ui_idx]

        # Second save with different parameters
        self.store.save_ui_config({"theme": "light", "min_confidence": 0.90, "show_spectrogram": True})

        full_text_after = self.temp_config_path.read_text(encoding="utf-8")
        ui_idx_after = full_text_after.index("ui:")
        prefix_after_ui = full_text_after[:ui_idx_after]

        # Everything before ui: must be exactly identical
        self.assertEqual(prefix_before_ui, prefix_after_ui)

        # Values in ui: must match updated values
        updated_ui = self.store.get_ui_config()
        self.assertEqual(updated_ui["theme"], "light")
        self.assertAlmostEqual(updated_ui["min_confidence"], 0.90)
        self.assertTrue(updated_ui["show_spectrogram"])

    def test_invalid_values_are_normalized(self) -> None:
        """Invalid theme or out-of-range confidence values must be normalized safely."""
        self.store.save_ui_config({"theme": "invalid_neon", "min_confidence": 2.5})
        cfg = self.store.get_ui_config()
        self.assertEqual(cfg["theme"], "system")
        self.assertEqual(cfg["min_confidence"], 1.0)

        self.store.save_ui_config({"min_confidence": -0.5})
        cfg2 = self.store.get_ui_config()
        self.assertEqual(cfg2["min_confidence"], 0.0)


if __name__ == "__main__":
    unittest.main()
