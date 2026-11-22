"""Application bootstrap and lifecycle controller for Holo UI."""

from __future__ import annotations

import sys
import os
import signal
import argparse
import logging
from typing import Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication

from src.holo.ui.config_store import config_store
from src.holo.ui.theme import theme_manager
from src.holo.ui.engine_bridge import EngineBridge
from src.holo.ui.tray import HoloSystemTray
from src.holo.ui.main_window import MainWindow

logger = logging.getLogger(__name__)


class HoloApp:
    """Manages the QApplication, theme lifecycle, system tray, and main window."""

    def __init__(self, headless: bool = False, smoke_test: bool = False) -> None:
        self.headless = headless
        self.smoke_test = smoke_test

        if self.headless or os.environ.get("QT_QPA_PLATFORM") == "offscreen":
            os.environ["QT_QPA_PLATFORM"] = "offscreen"

        # Initialize QApplication
        self.app = QApplication.instance()
        if self.app is None:
            self.app = QApplication(sys.argv)

        self.app.setApplicationName("Holo")
        self.app.setOrganizationName("AcousticTap")
        self.app.setQuitOnLastWindowClosed(False)

        # Apply initial theme
        ui_cfg = config_store.get_ui_config()
        initial_theme = ui_cfg.get("theme", "system")
        theme_manager.set_mode(initial_theme)

        # Core components
        self.engine_bridge = EngineBridge()
        self.main_window = MainWindow(engine_bridge=self.engine_bridge)
        self.system_tray = HoloSystemTray()

        self._wire_signals()
        self._setup_interrupt_handling()

    def _wire_signals(self) -> None:
        """Connects tray actions and engine signals."""
        # Tray -> Engine / Window
        self.system_tray.toggle_engine_requested.connect(self._toggle_engine)
        self.system_tray.show_dashboard_requested.connect(self.show_main_window)
        self.system_tray.quit_requested.connect(self.quit)

        # Engine -> Tray
        self.engine_bridge.engine_state_changed.connect(self._on_engine_state_changed)
        self.engine_bridge.tap_detected.connect(self._on_tap_detected)

        # Theme changes -> Tray icon update
        theme_manager.theme_changed.connect(
            lambda is_dark: self.system_tray.update_state(self.engine_bridge.is_running, is_dark=is_dark)
        )

    def _toggle_engine(self) -> None:
        if self.engine_bridge.is_running:
            self.engine_bridge.stop_engine()
        else:
            self.engine_bridge.start_engine()

    def _on_engine_state_changed(self, is_running: bool, status_msg: str) -> None:
        self.system_tray.update_state(is_running, is_dark=theme_manager.is_dark)

    def _on_tap_detected(self, event: dict) -> None:
        label = event.get("label", "tap")
        conf = float(event.get("confidence", 0.0))
        passed = bool(event.get("geofence_passed", False))
        self.system_tray.update_last_tap(label, conf, passed)

    def _setup_interrupt_handling(self) -> None:
        """Allows Python to process SIGINT (Ctrl+C) smoothly in Qt event loop."""
        signal.signal(signal.SIGINT, lambda *args: self.quit())
        self.sigint_timer = QTimer()
        self.sigint_timer.setInterval(250)
        self.sigint_timer.timeout.connect(lambda: None)  # Wakes Python GIL
        self.sigint_timer.start()

    def show_main_window(self) -> None:
        """Displays and brings main window to front."""
        self.main_window.show()
        if os.environ.get("QT_QPA_PLATFORM") != "offscreen":
            try:
                self.main_window.raise_()
                self.main_window.activateWindow()
            except Exception:
                pass

    def quit(self) -> None:
        """Performs graceful shutdown releasing audio resources."""
        logger.info("Quitting Holo...")
        self.main_window.set_force_quit(True)
        self.engine_bridge.shutdown()
        self.system_tray.hide()
        self.app.quit()

    def run(self) -> int:
        """Runs the event loop or executes smoke test."""
        if self.smoke_test:
            logger.info("Running smoke test verification...")
            assert self.main_window is not None
            assert self.system_tray is not None
            assert self.engine_bridge is not None
            # Teardown cleanly
            self.quit()
            return 0

        self.system_tray.show()
        self.show_main_window()
        return self.app.exec()


def main() -> int:
    """CLI entry point for Holo Desktop & Menu Bar UI."""
    parser = argparse.ArgumentParser(description="Holo - macOS Desktop & Menu Bar Acoustic Tap Controller")
    parser.add_argument("--smoke-test", action="store_true", help="Instantiate components, verify integrity, and exit cleanly")
    parser.add_argument("--headless", action="store_true", help="Run with offscreen QPA platform")
    args = parser.parse_args()

    app_instance = HoloApp(headless=args.headless, smoke_test=args.smoke_test)
    return app_instance.run()


if __name__ == "__main__":
    sys.exit(main())
