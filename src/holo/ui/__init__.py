"""Holo UI package.

macOS Desktop & Menu Bar UI for Acoustic Tap Recognition & Geofencing.
"""

from src.holo.ui.app import HoloApp, main
from src.holo.ui.main_window import MainWindow
from src.holo.ui.tray import HoloSystemTray
from src.holo.ui.engine_bridge import EngineBridge

__all__ = [
    "HoloApp",
    "main",
    "MainWindow",
    "HoloSystemTray",
    "EngineBridge",
]
