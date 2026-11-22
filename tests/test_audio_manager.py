import unittest
from unittest.mock import patch, MagicMock
import numpy as np

from src.holo.audio_manager import (
    AudioDeviceInfo,
    is_builtin_device_name,
    is_external_or_bluetooth_name,
    find_builtin_microphone,
    find_builtin_speakers,
    find_secondary_microphone,
    SplitAudioEngine,
)


class TestAudioManager(unittest.TestCase):

    def test_builtin_device_naming(self):
        self.assertTrue(is_builtin_device_name("MacBook Air Microphone"))
        self.assertTrue(is_builtin_device_name("MacBook Pro Microphone"))
        self.assertTrue(is_builtin_device_name("Built-in Microphone"))
        self.assertTrue(is_builtin_device_name("Internal Microphone"))

        # External / Bluetooth devices must NOT be classified as built-in
        self.assertFalse(is_builtin_device_name("AirPods Pro"))
        self.assertFalse(is_builtin_device_name("Sony WH-1000XM4"))
        self.assertFalse(is_builtin_device_name("Bluetooth Audio Device"))
        self.assertFalse(is_builtin_device_name("USB Headset"))
        self.assertFalse(is_builtin_device_name("BlackHole 2ch"))

    def test_external_device_naming(self):
        self.assertTrue(is_external_or_bluetooth_name("AirPods Pro"))
        self.assertTrue(is_external_or_bluetooth_name("AirPods Max"))
        self.assertTrue(is_external_or_bluetooth_name("WH-1000XM5"))
        self.assertTrue(is_external_or_bluetooth_name("Bose QuietComfort"))
        self.assertTrue(is_external_or_bluetooth_name("Bluetooth Handsfree Headset"))
        self.assertTrue(is_external_or_bluetooth_name("USB Earphones with Mic"))

        # MacBook mic must NOT be classified as external
        self.assertFalse(is_external_or_bluetooth_name("MacBook Air Microphone"))
        self.assertFalse(is_external_or_bluetooth_name("Built-in Microphone"))

    @patch("src.holo.audio_manager.sd.query_devices")
    def test_find_builtin_microphone_isolation(self, mock_query):
        mock_devices = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0, "hostapi": 0},
            {"name": "AirPods Pro", "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 24000.0, "hostapi": 0},
            {"name": "MacBook Pro Speakers", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0, "hostapi": 0},
        ]
        mock_query.return_value = mock_devices

        builtin = find_builtin_microphone()
        self.assertIsNotNone(builtin)
        self.assertEqual(builtin.index, 0)
        self.assertEqual(builtin.name, "MacBook Pro Microphone")
        self.assertTrue(builtin.is_builtin)

    @patch("src.holo.audio_manager.sd.query_devices")
    def test_find_builtin_speakers_isolation(self, mock_query):
        mock_devices = [
            {"name": "MacBook Pro Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0, "hostapi": 0},
            {"name": "AirPods Pro", "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 24000.0, "hostapi": 0},
            {"name": "MacBook Pro Speakers", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 44100.0, "hostapi": 0},
        ]
        mock_query.return_value = mock_devices

        speaker = find_builtin_speakers()
        self.assertIsNotNone(speaker)
        self.assertEqual(speaker.index, 2)
        self.assertEqual(speaker.name, "MacBook Pro Speakers")
        self.assertTrue(speaker.is_builtin)

    @patch("src.holo.audio_manager.sd.InputStream")
    @patch("src.holo.audio_manager.sd.query_devices")
    def test_default_engine_strictly_ignores_bluetooth_microphone(self, mock_query, mock_input_stream):
        """Verifies that by default, Bluetooth/headset mic is NEVER opened."""
        mock_devices = [
            {"name": "MacBook Air Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0, "hostapi": 0},
            {"name": "AirPods Pro", "max_input_channels": 1, "max_output_channels": 2, "default_samplerate": 24000.0, "hostapi": 0},
        ]
        mock_query.return_value = mock_devices

        mock_stream = MagicMock()
        mock_stream.active = True
        mock_input_stream.return_value = mock_stream

        engine = SplitAudioEngine(
            tap_callback=lambda indata, f, t, s: None,
            capture_secondary=False,  # default
        )

        with engine:
            # Stream must only be opened ONCE for the MacBook built-in mic
            self.assertEqual(mock_input_stream.call_count, 1)
            opened_device = mock_input_stream.call_args.kwargs["device"]
            self.assertEqual(opened_device, 0)
            self.assertFalse(engine.is_secondary_active())

    @patch("src.holo.audio_manager.sd.InputStream")
    @patch("src.holo.audio_manager.sd.query_devices")
    def test_no_background_polling_thread(self, mock_query, mock_input_stream):
        """Verifies that no continuous background polling loop runs."""
        mock_devices = [
            {"name": "MacBook Air Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0, "hostapi": 0},
        ]
        mock_query.return_value = mock_devices

        mock_stream = MagicMock()
        mock_stream.active = True
        mock_input_stream.return_value = mock_stream

        engine = SplitAudioEngine(
            tap_callback=lambda *args: None,
            capture_secondary=False,
        )

        with engine:
            # Must not have any active monitor thread
            self.assertFalse(hasattr(engine, "_monitor_thread") and engine._monitor_thread is not None)

    @patch("src.holo.audio_manager.sd.InputStream")
    @patch("src.holo.audio_manager.sd.query_devices")
    def test_passive_error_recovery(self, mock_query, mock_input_stream):
        """Verifies passive error recovery restores stream only when interrupted."""
        mock_devices = [
            {"name": "MacBook Air Microphone", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 44100.0, "hostapi": 0},
        ]
        mock_query.return_value = mock_devices

        mock_stream1 = MagicMock()
        mock_stream1.active = True
        mock_stream2 = MagicMock()
        mock_stream2.active = True
        mock_input_stream.side_effect = [mock_stream1, mock_stream2]

        engine = SplitAudioEngine(
            tap_callback=lambda *args: None,
            capture_secondary=False,
        )

        with engine:
            self.assertEqual(mock_input_stream.call_count, 1)

            # Simulate stream stopping due to CoreAudio event (e.g. Bluetooth connection)
            mock_stream1.active = False

            # Passive recovery is invoked
            recovered = engine.ensure_active()
            self.assertTrue(recovered)
            self.assertEqual(mock_input_stream.call_count, 2)


if __name__ == "__main__":
    unittest.main()
