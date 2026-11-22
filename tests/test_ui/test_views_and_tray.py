"""Tests for MainWindow, views, theme manager, and system tray."""

from __future__ import annotations

import os
import unittest
from unittest.mock import patch
import numpy as np

os.environ["QT_QPA_PLATFORM"] = "offscreen"

from PySide6.QtWidgets import QApplication
from PySide6.QtGui import QIcon

from src.holo.ui.theme import theme_manager, generate_qss
from src.holo.ui.tray import HoloSystemTray, create_tray_icon
from src.holo.ui.engine_bridge import EngineBridge
from src.holo.ui.main_window import MainWindow
from src.holo.ui.views.monitor_view import MonitorView
from src.holo.ui.views.diagnostics_view import DiagnosticsView
from src.holo.ui.views.calibration_view import CalibrationView
from src.holo.ui.views.settings_view import SettingsView

MOCK_DEVICES = [
    {
        "index": 0,
        "name": "MacBook Air Microphone",
        "channels": 1,
        "samplerate": 44100.0,
        "is_default": True,
    },
    {
        "index": 1,
        "name": "USB Audio Interface",
        "channels": 2,
        "samplerate": 48000.0,
        "is_default": False,
    },
]


class TestViewsAndTray(unittest.TestCase):
    """Test suite verifying all UI views, navigation, tray, and theming."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.app = QApplication.instance()
        if cls.app is None:
            cls.app = QApplication([])

    def test_theme_manager_modes_and_qss(self) -> None:
        """ThemeManager switches between dark, light, and system cleanly."""
        theme_manager.set_mode("dark")
        self.assertEqual(theme_manager.mode, "dark")
        self.assertTrue(theme_manager.is_dark)

        theme_manager.set_mode("light")
        self.assertEqual(theme_manager.mode, "light")
        self.assertFalse(theme_manager.is_dark)

        theme_manager.set_mode("system")
        self.assertEqual(theme_manager.mode, "system")

        pens = theme_manager.get_plot_pens()
        self.assertEqual(len(pens), 3)

    def test_create_tray_icon(self) -> None:
        """Programmatic vector tray icons generate valid QIcon objects."""
        icon_active = create_tray_icon(is_active=True, is_dark=True)
        self.assertIsInstance(icon_active, QIcon)
        self.assertFalse(icon_active.isNull())

        icon_idle = create_tray_icon(is_active=False, is_dark=False)
        self.assertIsInstance(icon_idle, QIcon)
        self.assertFalse(icon_idle.isNull())

    def test_system_tray_menu_actions_and_state(self) -> None:
        """HoloSystemTray manages menu items and updates dynamically."""
        tray = HoloSystemTray()
        self.assertIsNotNone(tray.tray_icon)
        self.assertIsNotNone(tray.menu)

        tray.update_state(is_active=True, is_dark=True)
        self.assertEqual(tray.status_action.text(), "Holo: Listening")
        self.assertEqual(tray.toggle_action.text(), "Stop Listening")

        tray.update_state(is_active=False, is_dark=False)
        self.assertEqual(tray.status_action.text(), "Holo: Idle")
        self.assertEqual(tray.toggle_action.text(), "Start Listening")

        tray.update_last_tap("left_tap", 0.94, geofence_passed=True)
        self.assertIn("left_tap (94%) [IN]", tray.last_tap_action.text())

    @patch("src.holo.ui.views.monitor_view.list_audio_devices", return_value=MOCK_DEVICES)
    @patch("src.holo.ui.views.monitor_view.find_builtin_microphone", return_value=0)
    @patch("src.holo.ui.views.monitor_view.find_secondary_microphone", return_value=1)
    def test_monitor_view_lifecycle(self, mock_sec, mock_built, mock_list) -> None:
        """MonitorView correctly populates devices and handles tap events."""
        view = MonitorView()
        self.assertEqual(view.primary_combo.count(), 2)
        self.assertEqual(view.secondary_combo.count(), 3)  # None + 2 devices

        # Update engine state to listening
        view.update_engine_state(True, "Listening")
        self.assertIn("LISTENING", view.status_indicator.text())
        self.assertFalse(view.primary_combo.isEnabled())

        # Simulate incoming in-bounds tap
        tap_event = {
            "label": "left_tap",
            "confidence": 0.88,
            "meets_min_confidence": True,
            "geofence_passed": True,
            "geofence_reason": "inside palm rest boundary",
            "latency_ms": 3.8,
            "breakdown": "ML:left_tap(0.90) + FFT:left_tap(0.86)",
            "action_taken": True,
        }
        view.handle_tap_event(tap_event)
        self.assertIn("Left Palm Rest", view.gesture_label.text())
        self.assertIn("IN-BOUNDS", view.geofence_badge.text())
        self.assertEqual(view.events_list.count(), 1)

        # Simulate audio chunk preview
        chunk = np.sin(np.linspace(0, np.pi * 2, 512)).astype(np.float32)
        view.update_audio_preview(chunk)

    def test_diagnostics_view_oscilloscope_and_table(self) -> None:
        """DiagnosticsView consumes audio chunks into ring buffer and logs events to table."""
        view = DiagnosticsView()

        # Push chunk
        chunk = np.ones(256, dtype=np.float32) * 0.2
        view.push_audio_chunk(chunk)
        self.assertEqual(len(view.waveform_buffer), view.buffer_size)

        # Force render
        view._render_plots()

        # Push tap event
        tap_event = {
            "timestamp": 1700000000.0,
            "label": "right_tap",
            "confidence": 0.91,
            "geofence_passed": True,
            "geofence_reason": "inside palm rest boundary",
            "latency_ms": 4.1,
            "geofence_details": {"prob_inside": 0.95},
        }
        view.handle_tap_event(tap_event)
        self.assertEqual(view.table.rowCount(), 1)
        self.assertEqual(view.table.item(0, 1).text(), "RIGHT_TAP")
        self.assertEqual(view.table.item(0, 3).text(), "PASS")

    def test_calibration_view_state(self) -> None:
        """CalibrationView renders buttons, count displays, and refresh actions."""
        view = CalibrationView()
        self.assertIsNotNone(view.lbl_center_count)
        self.assertIsNotNone(view.btn_run_step1)
        self.assertIsNotNone(view.btn_run_step2)
        self.assertIsNotNone(view.btn_run_step3)
        self.assertIsNotNone(view.btn_run_step4)

        # Ensure refresh counts runs without exception
        view.refresh_dataset_counts()

    def test_calibration_view_cancel_and_clear_console(self) -> None:
        """Verifies clearing console during active task and cancelling never crashes."""
        view = CalibrationView()
        from src.holo.ui.views.calibration_view import CalibrationWorker
        from PySide6.QtCore import QThread

        worker = CalibrationWorker("collect_center", {"class": "all", "samples": 5})
        view.worker = worker
        view.worker_thread = QThread()
        view.btn_stop_task.setEnabled(True)

        # Append logs
        view._append_log("Collecting samples [1/5]...\n")
        # User clears console while running
        view._clear_console()
        self.assertEqual(view.console.toPlainText(), "")

        # Cancel task
        view._cancel_task()
        self.assertFalse(view.btn_stop_task.isEnabled())
        self.assertIsNone(view.worker)
        self.assertIsNone(view.worker_thread)

    def test_settings_view_load_and_save(self) -> None:
        """SettingsView interacts with theme and confidence controls."""
        view = SettingsView()
        self.assertIsNotNone(view.theme_combo)
        self.assertIsNotNone(view.conf_slider)
        self.assertIsNotNone(view.conf_spin)
        self.assertIsNotNone(view.inspector_text)

        # Change slider and spinbox
        view.conf_slider.setValue(80)
        self.assertAlmostEqual(view.conf_spin.value(), 0.80)

        # Toggle inspector
        view._toggle_inspector()
        self.assertTrue(view.inspector_text.isHidden())
        view._toggle_inspector()
        self.assertFalse(view.inspector_text.isHidden())

    @patch("src.holo.ui.views.monitor_view.list_audio_devices", return_value=MOCK_DEVICES)
    def test_main_window_navigation_and_close_event(self, mock_list) -> None:
        """MainWindow sidebar navigation switches views and closeEvent hides window."""
        bridge = EngineBridge()
        window = MainWindow(engine_bridge=bridge)

        # Switch to each tab
        window.btn_diagnostics.click()
        self.assertEqual(window.stack.currentIndex(), 1)

        window.btn_calibration.click()
        self.assertEqual(window.stack.currentIndex(), 2)

        window.btn_settings.click()
        self.assertEqual(window.stack.currentIndex(), 3)

        window.btn_monitor.click()
        self.assertEqual(window.stack.currentIndex(), 0)

        # Test close event (should hide window, not close)
        window.show()
        self.assertTrue(window.isVisible())
        window.close()
        self.assertFalse(window.isVisible())  # Hidden, keeping app alive in menu bar

        bridge.shutdown()


if __name__ == "__main__":
    unittest.main()
