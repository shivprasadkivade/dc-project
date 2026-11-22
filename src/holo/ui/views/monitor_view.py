"""Live Monitor View for Holo UI.

Provides listener controls, audio device selection, live gesture card,
confidence threshold meter, geofence status badge, and recent tap feed.
"""

from __future__ import annotations

import time
from typing import Optional, Dict, Any, List
import numpy as np

from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QComboBox,
    QProgressBar,
    QFrame,
    QListWidget,
    QListWidgetItem,
    QCheckBox,
    QScrollArea,
)
import pyqtgraph as pg

try:
    from src.holo.audio_manager import (
        list_audio_devices,
        find_builtin_microphone,
        find_secondary_microphone,
    )
except ImportError:
    from src.holo.ui.audio_adapter import (  # type: ignore # noqa: F401
        list_audio_devices,
        find_builtin_microphone,
        find_secondary_microphone,
    )

from src.holo.ui.theme import theme_manager, ThemeColors
from src.holo.ui.config_store import config_store


class MonitorView(QWidget):
    """Main dashboard tab for live gesture monitoring and hardware selection."""

    start_engine_requested = Signal(object, object, bool)  # (primary_dev, secondary_dev, enable_shortcuts)
    stop_engine_requested = Signal()

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._is_listening: bool = False
        self._show_spectrogram: bool = True
        self._min_confidence: float = 0.65

        self._init_ui()
        self.refresh_devices()
        self.apply_config_preferences()

    def _init_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(28, 24, 28, 24)
        main_layout.setSpacing(20)

        # 1. Header Row
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(4)

        self.header_title = QLabel("Live Monitor")
        self.header_title.setObjectName("headerTitle")
        self.sub_title = QLabel("Real-time acoustic palm rest gesture detection and geofencing")
        self.sub_title.setObjectName("subTitle")

        title_box.addWidget(self.header_title)
        title_box.addWidget(self.sub_title)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        # Listener Toggle Pill
        self.status_indicator = QLabel("● IDLE")
        self.status_indicator.setStyleSheet("color: #8E8E93; font-weight: 700; font-size: 13px; margin-right: 8px;")

        self.toggle_button = QPushButton("Start Listening")
        self.toggle_button.setObjectName("primaryButton")
        self.toggle_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.toggle_button.clicked.connect(self._on_toggle_clicked)

        header_layout.addWidget(self.status_indicator)
        header_layout.addWidget(self.toggle_button)
        main_layout.addLayout(header_layout)

        # 2. Audio Hardware Configuration Bar (Card)
        device_card = QFrame()
        device_card.setObjectName("cardFrame")
        dev_layout = QHBoxLayout(device_card)
        dev_layout.setContentsMargins(16, 12, 16, 12)
        dev_layout.setSpacing(16)

        # Primary Microphone Selector
        primary_box = QVBoxLayout()
        primary_box.setSpacing(4)
        lbl_primary = QLabel("Primary Built-in Microphone:")
        lbl_primary.setObjectName("subTitle")
        self.primary_combo = QComboBox()
        self.primary_combo.setMinimumWidth(240)
        primary_box.addWidget(lbl_primary)
        primary_box.addWidget(self.primary_combo)
        dev_layout.addLayout(primary_box)

        # Secondary Microphone Selector
        secondary_box = QVBoxLayout()
        secondary_box.setSpacing(4)
        lbl_secondary = QLabel("Secondary External Microphone (Optional):")
        lbl_secondary.setObjectName("subTitle")
        self.secondary_combo = QComboBox()
        self.secondary_combo.setMinimumWidth(240)
        secondary_box.addWidget(lbl_secondary)
        secondary_box.addWidget(self.secondary_combo)
        dev_layout.addLayout(secondary_box)

        # Refresh Devices Button
        self.refresh_btn = QPushButton("Refresh")
        self.refresh_btn.setObjectName("secondaryButton")
        self.refresh_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.refresh_btn.clicked.connect(self.refresh_devices)
        dev_layout.addWidget(self.refresh_btn, alignment=Qt.AlignmentFlag.AlignBottom)

        # Macro Trigger Checkbox
        self.macro_checkbox = QCheckBox("Trigger OS Macros (Cmd+C/V)")
        self.macro_checkbox.setToolTip("When checked, in-bounds gestures will execute system keyboard shortcuts.")
        self.macro_checkbox.setChecked(False)
        dev_layout.addWidget(self.macro_checkbox, alignment=Qt.AlignmentFlag.AlignBottom)

        dev_layout.addStretch()
        main_layout.addWidget(device_card)

        # 3. Middle Section: Live Tap Status Card + Mini Visualizer
        middle_layout = QHBoxLayout()
        middle_layout.setSpacing(16)

        # Live Tap Card
        self.tap_card = QFrame()
        self.tap_card.setObjectName("cardFrame")
        tap_layout = QVBoxLayout(self.tap_card)
        tap_layout.setContentsMargins(20, 18, 20, 18)
        tap_layout.setSpacing(12)

        card_header = QHBoxLayout()
        lbl_latest = QLabel("LATEST DETECTED IMPACT")
        lbl_latest.setObjectName("subTitle")
        self.geofence_badge = QLabel("WAITING")
        self.geofence_badge.setStyleSheet(
            "background-color: #383838; color: #FFFFFF; border-radius: 6px; padding: 3px 8px; font-weight: 700; font-size: 11px;"
        )
        card_header.addWidget(lbl_latest)
        card_header.addStretch()
        card_header.addWidget(self.geofence_badge)
        tap_layout.addLayout(card_header)

        # Detected Gesture Label
        self.gesture_label = QLabel("Awaiting Tap...")
        self.gesture_label.setObjectName("metricValue")
        tap_layout.addWidget(self.gesture_label)

        # Sub-info: breakdown, shortcut info, latency
        self.gesture_detail = QLabel("Tap left or right palm rest on chassis.")
        self.gesture_detail.setObjectName("subTitle")
        tap_layout.addWidget(self.gesture_detail)

        # Confidence Bar
        conf_header = QHBoxLayout()
        lbl_conf = QLabel("Confidence:")
        lbl_conf.setObjectName("subTitle")
        self.conf_val_label = QLabel("0.0%")
        self.conf_val_label.setStyleSheet("font-weight: 700;")
        conf_header.addWidget(lbl_conf)
        conf_header.addStretch()
        conf_header.addWidget(self.conf_val_label)
        tap_layout.addLayout(conf_header)

        self.conf_bar = QProgressBar()
        self.conf_bar.setRange(0, 100)
        self.conf_bar.setValue(0)
        self.conf_bar.setFixedHeight(10)
        self.conf_bar.setTextVisible(False)
        tap_layout.addWidget(self.conf_bar)

        middle_layout.addWidget(self.tap_card, stretch=5)

        # Mini Live Waveform / Spectrogram Preview
        self.preview_card = QFrame()
        self.preview_card.setObjectName("cardFrame")
        prev_layout = QVBoxLayout(self.preview_card)
        prev_layout.setContentsMargins(16, 14, 16, 14)
        prev_layout.setSpacing(8)

        lbl_prev_title = QLabel("LIVE INPUT PREVIEW")
        lbl_prev_title.setObjectName("subTitle")
        prev_layout.addWidget(lbl_prev_title)

        self.mini_plot = pg.PlotWidget()
        self.mini_plot.setFixedHeight(120)
        self.mini_plot.showGrid(x=False, y=True, alpha=0.3)
        self.mini_plot.setYRange(-0.25, 0.25)
        self.mini_plot.getPlotItem().hideAxis("bottom")
        self.mini_plot.getPlotItem().hideAxis("left")
        self.mini_curve = self.mini_plot.plot(pen=pg.mkPen(color=ThemeColors.DARK_ACCENT, width=1.5))
        prev_layout.addWidget(self.mini_plot)

        middle_layout.addWidget(self.preview_card, stretch=4)
        main_layout.addLayout(middle_layout)

        # 4. Recent Events Feed Card
        events_card = QFrame()
        events_card.setObjectName("cardFrame")
        ev_layout = QVBoxLayout(events_card)
        ev_layout.setContentsMargins(18, 16, 18, 16)
        ev_layout.setSpacing(10)

        feed_header = QHBoxLayout()
        lbl_feed = QLabel("RECENT EVENTS FEED")
        lbl_feed.setObjectName("subTitle")
        self.clear_feed_btn = QPushButton("Clear Feed")
        self.clear_feed_btn.setObjectName("secondaryButton")
        self.clear_feed_btn.setFixedHeight(26)
        self.clear_feed_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_feed_btn.clicked.connect(self._clear_feed)

        feed_header.addWidget(lbl_feed)
        feed_header.addStretch()
        feed_header.addWidget(self.clear_feed_btn)
        ev_layout.addLayout(feed_header)

        self.events_list = QListWidget()
        self.events_list.setSelectionMode(QListWidget.SelectionMode.NoSelection)
        self.events_list.setStyleSheet(
            "QListWidget { border: none; background: transparent; } "
            "QListWidget::item { padding: 8px 12px; border-bottom: 1px solid #383838; border-radius: 4px; margin-bottom: 4px; }"
        )
        ev_layout.addWidget(self.events_list)
        main_layout.addWidget(events_card, stretch=1)

    def apply_config_preferences(self) -> None:
        """Applies UI configuration settings."""
        ui_cfg = config_store.get_ui_config()
        self._show_spectrogram = bool(ui_cfg.get("show_spectrogram", True))
        self._min_confidence = float(ui_cfg.get("min_confidence", 0.65))
        self.preview_card.setVisible(self._show_spectrogram)

    def refresh_devices(self) -> None:
        """Discovers input audio devices and populates dropdowns."""
        devices = list_audio_devices()
        builtin_idx = find_builtin_microphone(devices)
        secondary_idx = find_secondary_microphone(devices)

        self.primary_combo.clear()
        self.secondary_combo.clear()

        self.secondary_combo.addItem("None (Single-Mic Mono)", None)

        selected_primary_row = 0
        selected_sec_row = 0

        for row, dev in enumerate(devices):
            label = f"[{dev['index']}] {dev['name']} ({dev['channels']} ch)"
            self.primary_combo.addItem(label, dev["index"])
            if dev["index"] == builtin_idx:
                selected_primary_row = row

            sec_row = row + 1  # Offset by 1 due to "None"
            self.secondary_combo.addItem(label, dev["index"])
            if secondary_idx is not None and dev["index"] == secondary_idx:
                selected_sec_row = sec_row

        if self.primary_combo.count() > 0:
            self.primary_combo.setCurrentIndex(selected_primary_row)
        if self.secondary_combo.count() > 0:
            self.secondary_combo.setCurrentIndex(selected_sec_row)

    def _on_toggle_clicked(self) -> None:
        """Handles Start/Stop listener click."""
        if self._is_listening:
            self.stop_engine_requested.emit()
        else:
            prim_dev = self.primary_combo.currentData()
            sec_dev = self.secondary_combo.currentData()
            enable_macros = self.macro_checkbox.isChecked()
            self.start_engine_requested.emit(prim_dev, sec_dev, enable_macros)

    @Slot(bool, str)
    def update_engine_state(self, is_running: bool, status_msg: str) -> None:
        """Updates UI elements based on engine running state."""
        self._is_listening = is_running
        if is_running:
            self.status_indicator.setText("● LISTENING")
            self.status_indicator.setStyleSheet("color: #30D158; font-weight: 700; font-size: 13px; margin-right: 8px;")
            self.toggle_button.setText("Stop Listening")
            self.toggle_button.setStyleSheet("background-color: #FF453A; color: #FFFFFF;")
            self.primary_combo.setEnabled(False)
            self.secondary_combo.setEnabled(False)
            self.refresh_btn.setEnabled(False)
        else:
            self.status_indicator.setText("● IDLE")
            self.status_indicator.setStyleSheet("color: #8E8E93; font-weight: 700; font-size: 13px; margin-right: 8px;")
            self.toggle_button.setText("Start Listening")
            self.toggle_button.setStyleSheet("")
            self.primary_combo.setEnabled(True)
            self.secondary_combo.setEnabled(True)
            self.refresh_btn.setEnabled(True)

    @Slot(dict)
    def handle_tap_event(self, event: Dict[str, Any]) -> None:
        """Processes tap event dictionary emitted by EngineBridge."""
        label = event.get("label", "unknown")
        conf = float(event.get("confidence", 0.0))
        meets_conf = bool(event.get("meets_min_confidence", False))
        gf_passed = bool(event.get("geofence_passed", False))
        gf_reason = event.get("geofence_reason", "")
        latency = float(event.get("latency_ms", 0.0))
        breakdown = event.get("breakdown", "")
        action = bool(event.get("action_taken", False))

        # Update Live Tap Card
        conf_pct = conf * 100.0
        self.conf_bar.setValue(int(conf_pct))
        self.conf_val_label.setText(f"{conf_pct:.1f}%")

        if gf_passed and meets_conf:
            action_tag = " [Macro Executed]" if action else ""
            if "left" in label:
                self.gesture_label.setText("Left Palm Rest" + action_tag)
                self.gesture_label.setStyleSheet("color: #0A84FF; font-size: 26px; font-weight: 700;")
                self.gesture_detail.setText(f"Shortcut: Cmd+C (Copy) | {breakdown} | {latency:.1f} ms")
            elif "right" in label:
                self.gesture_label.setText("Right Palm Rest" + action_tag)
                self.gesture_label.setStyleSheet("color: #30D158; font-size: 26px; font-weight: 700;")
                self.gesture_detail.setText(f"Shortcut: Cmd+V (Paste) | {breakdown} | {latency:.1f} ms")
            else:
                self.gesture_label.setText(f"{label.upper()}")
                self.gesture_label.setStyleSheet("color: #A0A0A0; font-size: 26px; font-weight: 700;")
                self.gesture_detail.setText(f"Ambient transient | {breakdown} | {latency:.1f} ms")

            self.geofence_badge.setText("IN-BOUNDS (ACCEPTED)")
            self.geofence_badge.setStyleSheet(
                "background-color: #1B432C; color: #30D158; border: 1px solid #30D158; border-radius: 6px; padding: 3px 8px; font-weight: 700; font-size: 11px;"
            )
        elif not gf_passed:
            self.gesture_label.setText("Desk Knock / External Vibration")
            self.gesture_label.setStyleSheet("color: #FF9F0A; font-size: 22px; font-weight: 700;")
            self.gesture_detail.setText(f"Blocked by Geofence: {gf_reason} | {latency:.1f} ms")
            self.geofence_badge.setText("OUT-OF-BOUNDS (BLOCKED)")
            self.geofence_badge.setStyleSheet(
                "background-color: #4A2711; color: #FF9F0A; border: 1px solid #FF9F0A; border-radius: 6px; padding: 3px 8px; font-weight: 700; font-size: 11px;"
            )
        else:
            # Below confidence threshold
            self.gesture_label.setText(f"Low Confidence ({label})")
            self.gesture_label.setStyleSheet("color: #8E8E93; font-size: 22px; font-weight: 700;")
            self.gesture_detail.setText(f"Confidence {conf_pct:.1f}% below required threshold | {latency:.1f} ms")
            self.geofence_badge.setText("LOW CONFIDENCE")
            self.geofence_badge.setStyleSheet(
                "background-color: #383838; color: #A0A0A0; border-radius: 6px; padding: 3px 8px; font-weight: 700; font-size: 11px;"
            )

        # Add item to recent events feed
        time_str = time.strftime("%H:%M:%S")
        status_icon = "✓" if (gf_passed and meets_conf and label != "noise") else "✗"
        feed_item_text = (
            f"[{time_str}] {status_icon}  {label.upper():12s}  |  Conf: {conf_pct:4.1f}%  |  "
            f"Geofence: {'PASS' if gf_passed else 'BLOCKED'}  |  {latency:.1f} ms"
        )
        item = QListWidgetItem(feed_item_text)
        if gf_passed and meets_conf and label != "noise":
            item.setForeground(Qt.GlobalColor.green)
        elif not gf_passed:
            item.setForeground(Qt.GlobalColor.yellow)
        else:
            item.setForeground(Qt.GlobalColor.gray)

        self.events_list.insertItem(0, item)
        # Limit to 50 recent items
        while self.events_list.count() > 50:
            self.events_list.takeItem(self.events_list.count() - 1)

    @Slot(np.ndarray)
    def update_audio_preview(self, chunk: np.ndarray) -> None:
        """Updates mini live waveform plot."""
        if not self._show_spectrogram or len(chunk) == 0:
            return
        # Plot up to 512 points smoothly
        data = chunk[:512] if len(chunk) > 512 else chunk
        self.mini_curve.setData(data)

    def _clear_feed(self) -> None:
        self.events_list.clear()
