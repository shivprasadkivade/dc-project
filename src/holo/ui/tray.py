"""macOS Menu Bar System Tray Controller for Holo.

Features a crisp programmatic status vector icon generated with QPainter
and provides quick access to listener status, controls, and recent gestures.
"""

from __future__ import annotations

import logging
from typing import Optional
from PySide6.QtCore import QObject, Signal, Qt, QRectF
from PySide6.QtGui import QIcon, QPixmap, QPainter, QColor, QPen, QBrush
from PySide6.QtWidgets import QSystemTrayIcon, QMenu

logger = logging.getLogger(__name__)


def create_tray_icon(is_active: bool, is_dark: bool = True) -> QIcon:
    """Renders a crisp programmatic vector icon for macOS menu bar.

    Generates high-DPI @2x pixmaps without committing binary image assets.
    """
    size = 44  # Retina 22pt @2x
    pixmap = QPixmap(size, size)
    pixmap.fill(Qt.GlobalColor.transparent)

    painter = QPainter(pixmap)
    painter.setRenderHint(QPainter.RenderHint.Antialiasing)

    center_x = size / 2.0
    center_y = size / 2.0

    if is_active:
        # Listening / Active: vibrant pulse ring with solid inner dot
        pulse_color = QColor(10, 132, 255, 90) if is_dark else QColor(0, 122, 255, 90)
        core_color = QColor(10, 132, 255) if is_dark else QColor(0, 122, 255)

        # Outer glowing pulse ring
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(pulse_color))
        painter.drawEllipse(QRectF(center_x - 17, center_y - 17, 34, 34))

        # Acoustic waveform ripples
        pen = QPen(core_color, 2.5)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(center_x - 11, center_y - 11, 22, 22))

        # Solid inner dot
        painter.setBrush(QBrush(core_color))
        painter.drawEllipse(QRectF(center_x - 5, center_y - 5, 10, 10))
    else:
        # Idle / Paused: clean monochrome concentric ring
        ring_color = QColor(255, 255, 255, 180) if is_dark else QColor(30, 30, 30, 180)
        dot_color = QColor(255, 255, 255, 120) if is_dark else QColor(30, 30, 30, 120)

        pen = QPen(ring_color, 2.2)
        painter.setPen(pen)
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawEllipse(QRectF(center_x - 12, center_y - 12, 24, 24))

        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(QBrush(dot_color))
        painter.drawEllipse(QRectF(center_x - 4, center_y - 4, 8, 8))

    painter.end()
    return QIcon(pixmap)


class HoloSystemTray(QObject):
    """macOS Menu Bar System Tray Controller."""

    toggle_engine_requested = Signal()
    show_dashboard_requested = Signal()
    quit_requested = Signal()

    def __init__(self, parent: Optional[QObject] = None) -> None:
        super().__init__(parent)
        self._is_active: bool = False
        self._is_dark: bool = True

        self.tray_icon = QSystemTrayIcon(self)
        self.menu = QMenu()

        self._init_menu()
        self.update_state(is_active=False)

        self.tray_icon.activated.connect(self._on_tray_activated)

    def _init_menu(self) -> None:
        """Constructs macOS menu bar dropdown menu."""
        self.status_action = self.menu.addAction("Holo: Idle")
        self.status_action.setEnabled(False)

        self.menu.addSeparator()

        self.toggle_action = self.menu.addAction("Start Listening")
        self.toggle_action.triggered.connect(lambda: self.toggle_engine_requested.emit())

        self.last_tap_action = self.menu.addAction("Last Tap: None")
        self.last_tap_action.setEnabled(False)

        self.menu.addSeparator()

        self.show_action = self.menu.addAction("Show Dashboard")
        self.show_action.triggered.connect(lambda: self.show_dashboard_requested.emit())

        self.menu.addSeparator()

        self.quit_action = self.menu.addAction("Quit Holo")
        self.quit_action.triggered.connect(lambda: self.quit_requested.emit())

        self.tray_icon.setContextMenu(self.menu)

    def update_state(self, is_active: bool, is_dark: bool = True) -> None:
        """Updates menu item labels and redraws programmatic status icon."""
        self._is_active = is_active
        self._is_dark = is_dark

        icon = create_tray_icon(is_active, is_dark)
        self.tray_icon.setIcon(icon)

        if is_active:
            self.status_action.setText("Holo: Listening")
            self.toggle_action.setText("Stop Listening")
            self.tray_icon.setToolTip("Holo: Active & Listening")
        else:
            self.status_action.setText("Holo: Idle")
            self.toggle_action.setText("Start Listening")
            self.tray_icon.setToolTip("Holo: Idle")

    def update_last_tap(self, label: str, confidence: float, geofence_passed: bool) -> None:
        """Updates last detected tap action item."""
        status_sym = "[IN]" if geofence_passed else "[BLOCKED]"
        self.last_tap_action.setText(f"Last Tap: {label} ({confidence * 100:.0f}%) {status_sym}")

    def show(self) -> None:
        """Displays system tray icon."""
        self.tray_icon.show()

    def hide(self) -> None:
        """Hides system tray icon."""
        self.tray_icon.hide()

    def _on_tray_activated(self, reason: QSystemTrayIcon.ActivationReason) -> None:
        """Brings dashboard to front on icon click / double click."""
        if reason in (
            QSystemTrayIcon.ActivationReason.Trigger,
            QSystemTrayIcon.ActivationReason.DoubleClick,
        ):
            self.show_dashboard_requested.emit()
