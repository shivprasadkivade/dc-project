"""Main application window for Holo.

Features a macOS-style fixed-width (~210 px) sidebar navigation shell
with smooth switching across the 4 primary views and seamless menu bar integration.
"""

from __future__ import annotations

import logging
from typing import Optional
from PySide6.QtCore import Qt, QSize
from PySide6.QtGui import QCloseEvent, QIcon
from PySide6.QtWidgets import (
    QMainWindow,
    QWidget,
    QHBoxLayout,
    QVBoxLayout,
    QStackedWidget,
    QPushButton,
    QFrame,
    QLabel,
    QButtonGroup,
)

from src.holo.ui.engine_bridge import EngineBridge
from src.holo.ui.theme import theme_manager
from src.holo.ui.views.monitor_view import MonitorView
from src.holo.ui.views.diagnostics_view import DiagnosticsView
from src.holo.ui.views.calibration_view import CalibrationView
from src.holo.ui.views.settings_view import SettingsView

logger = logging.getLogger(__name__)


class MainWindow(QMainWindow):
    """macOS-inspired desktop navigation shell."""

    def __init__(
        self,
        engine_bridge: EngineBridge,
        parent: Optional[QWidget] = None,
    ) -> None:
        super().__init__(parent)
        self.engine_bridge = engine_bridge
        self._force_quit: bool = False

        self._init_window()
        self._init_ui()
        self._connect_signals()

    def _init_window(self) -> None:
        self.setWindowTitle("Holo — Acoustic Tap & Geofencing")
        self.setMinimumSize(980, 660)
        self.resize(1080, 720)

    def _init_ui(self) -> None:
        central = QWidget(self)
        central.setObjectName("centralWidget")
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Left Sidebar Panel (Fixed-width ~210 px)
        sidebar = QFrame()
        sidebar.setObjectName("sidebarPanel")
        sidebar.setFixedWidth(210)

        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(14, 20, 14, 16)
        sidebar_layout.setSpacing(6)

        # App Brand Header
        brand_box = QVBoxLayout()
        brand_box.setSpacing(2)
        brand_title = QLabel("HOLO")
        brand_title.setStyleSheet("font-size: 16px; font-weight: 800; letter-spacing: 1px;")
        brand_subtitle = QLabel("Acoustic Geofence v4")
        brand_subtitle.setStyleSheet("font-size: 11px; color: #8E8E93;")
        brand_box.addWidget(brand_title)
        brand_box.addWidget(brand_subtitle)
        brand_box.setContentsMargins(8, 0, 8, 14)
        sidebar_layout.addLayout(brand_box)

        # Section Label
        lbl_section = QLabel("NAVIGATION")
        lbl_section.setStyleSheet("font-size: 10px; font-weight: 700; color: #8E8E93; margin-left: 8px; margin-bottom: 4px;")
        sidebar_layout.addWidget(lbl_section)

        # Navigation Buttons Group
        self.nav_group = QButtonGroup(self)
        self.nav_group.setExclusive(True)

        self.btn_monitor = QPushButton("  Live Monitor")
        self.btn_monitor.setObjectName("navButton")
        self.btn_monitor.setCheckable(True)
        self.btn_monitor.setChecked(True)
        self.btn_monitor.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_diagnostics = QPushButton("  Signal & Geofence")
        self.btn_diagnostics.setObjectName("navButton")
        self.btn_diagnostics.setCheckable(True)
        self.btn_diagnostics.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_calibration = QPushButton("  Calibration")
        self.btn_calibration.setObjectName("navButton")
        self.btn_calibration.setCheckable(True)
        self.btn_calibration.setCursor(Qt.CursorShape.PointingHandCursor)

        self.btn_settings = QPushButton("  Settings")
        self.btn_settings.setObjectName("navButton")
        self.btn_settings.setCheckable(True)
        self.btn_settings.setCursor(Qt.CursorShape.PointingHandCursor)

        self.nav_group.addButton(self.btn_monitor, 0)
        self.nav_group.addButton(self.btn_diagnostics, 1)
        self.nav_group.addButton(self.btn_calibration, 2)
        self.nav_group.addButton(self.btn_settings, 3)

        sidebar_layout.addWidget(self.btn_monitor)
        sidebar_layout.addWidget(self.btn_diagnostics)
        sidebar_layout.addWidget(self.btn_calibration)
        sidebar_layout.addWidget(self.btn_settings)

        sidebar_layout.addStretch()

        # Bottom Status Indicator
        self.sidebar_status = QLabel("● Engine Idle")
        self.sidebar_status.setStyleSheet("font-size: 11px; color: #8E8E93; padding: 8px;")
        sidebar_layout.addWidget(self.sidebar_status)

        main_layout.addWidget(sidebar)

        # 2. Right Content Stack
        self.stack = QStackedWidget()
        self.monitor_view = MonitorView()
        self.diagnostics_view = DiagnosticsView()
        self.calibration_view = CalibrationView()
        self.settings_view = SettingsView()

        self.stack.addWidget(self.monitor_view)       # Index 0
        self.stack.addWidget(self.diagnostics_view)   # Index 1
        self.stack.addWidget(self.calibration_view)   # Index 2
        self.stack.addWidget(self.settings_view)      # Index 3

        main_layout.addWidget(self.stack, stretch=1)

    def _connect_signals(self) -> None:
        """Wires navigation buttons and bridge signals."""
        self.nav_group.idClicked.connect(self.stack.setCurrentIndex)

        # Connect Monitor View controls to Engine Bridge
        self.monitor_view.start_engine_requested.connect(self.engine_bridge.start_engine)
        self.monitor_view.stop_engine_requested.connect(self.engine_bridge.stop_engine)
        self.monitor_view.macro_toggled.connect(self.engine_bridge.set_enable_shortcuts)

        # Forward Engine Bridge signals to Views
        self.engine_bridge.engine_state_changed.connect(self._on_engine_state_changed)
        self.engine_bridge.tap_detected.connect(self.monitor_view.handle_tap_event)
        self.engine_bridge.tap_detected.connect(self.diagnostics_view.handle_tap_event)
        self.engine_bridge.audio_chunk_ready.connect(self.monitor_view.update_audio_preview)
        self.engine_bridge.audio_chunk_ready.connect(self.diagnostics_view.push_audio_chunk)

        # Settings View updates
        self.settings_view.min_confidence_changed.connect(self.engine_bridge.set_min_confidence)
        self.settings_view.min_confidence_changed.connect(lambda val: self.monitor_view.apply_config_preferences())

    def _on_engine_state_changed(self, is_running: bool, status_msg: str) -> None:
        self.monitor_view.update_engine_state(is_running, status_msg)
        if is_running:
            self.sidebar_status.setText("● Listening")
            self.sidebar_status.setStyleSheet("font-size: 11px; color: #30D158; padding: 8px; font-weight: 600;")
        else:
            self.sidebar_status.setText("● Engine Idle")
            self.sidebar_status.setStyleSheet("font-size: 11px; color: #8E8E93; padding: 8px;")

    def set_force_quit(self, force: bool = True) -> None:
        """Allows clean exit without intercepting closeEvent."""
        self._force_quit = force

    def closeEvent(self, event: QCloseEvent) -> None:
        """Hides window on close so listener continues in menu bar."""
        if self._force_quit:
            event.accept()
        else:
            event.ignore()
            self.hide()
