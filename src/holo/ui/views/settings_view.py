"""Settings View for Holo UI.

Allows editing UI-specific configuration (theme, spectrogram toggle, min_confidence)
with surgical line-preserving persistence, alongside a read-only core config inspector.
"""

from __future__ import annotations

import json
from typing import Optional, Dict, Any
from PySide6.QtCore import Qt, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QComboBox,
    QCheckBox,
    QSlider,
    QDoubleSpinBox,
    QPlainTextEdit,
    QScrollArea,
)

from src.holo.ui.config_store import config_store, DEFAULT_UI_CONFIG
from src.holo.ui.theme import theme_manager


class SettingsView(QWidget):
    """User preferences and read-only core engine configuration inspector."""

    min_confidence_changed = Signal(float)
    theme_changed_signal = Signal(str)

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._init_ui()
        self.load_settings()

    def _init_ui(self) -> None:
        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(20)

        # Header
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        header_title = QLabel("Settings & Preferences")
        header_title.setObjectName("headerTitle")
        sub_title = QLabel("Manage interface appearance, visualizer preferences, and inspect core engine configuration")
        sub_title.setObjectName("subTitle")
        title_box.addWidget(header_title)
        title_box.addWidget(sub_title)
        layout.addLayout(title_box)

        # --- Section 1: UI Preferences Card ---
        ui_card = QFrame()
        ui_card.setObjectName("cardFrame")
        card_layout = QVBoxLayout(ui_card)
        card_layout.setContentsMargins(20, 18, 20, 18)
        card_layout.setSpacing(16)

        lbl_ui_title = QLabel("UI & DISPLAY PREFERENCES")
        lbl_ui_title.setObjectName("cardTitle")
        card_layout.addWidget(lbl_ui_title)

        # Theme Selector Row
        theme_row = QHBoxLayout()
        lbl_theme = QLabel("Appearance Theme:")
        lbl_theme.setObjectName("subTitle")
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("System (Follows macOS)", "system")
        self.theme_combo.addItem("Dark Mode", "dark")
        self.theme_combo.addItem("Light Mode", "light")
        self.theme_combo.currentIndexChanged.connect(self._on_theme_changed)

        theme_row.addWidget(lbl_theme)
        theme_row.addStretch()
        theme_row.addWidget(self.theme_combo)
        card_layout.addLayout(theme_row)

        # Show Spectrogram Checkbox Row
        spec_row = QHBoxLayout()
        lbl_spec = QLabel("Live Waveform & Spectrogram Preview:")
        lbl_spec.setObjectName("subTitle")
        self.spec_checkbox = QCheckBox("Show on Live Monitor")
        spec_row.addWidget(lbl_spec)
        spec_row.addStretch()
        spec_row.addWidget(self.spec_checkbox)
        card_layout.addLayout(spec_row)

        # Minimum Confidence Filter Row (Slider + DoubleSpinBox)
        conf_box = QVBoxLayout()
        conf_box.setSpacing(6)
        conf_label_row = QHBoxLayout()
        lbl_conf_title = QLabel("UI Minimum Confidence Threshold:")
        lbl_conf_title.setObjectName("subTitle")
        self.lbl_conf_pct = QLabel("65%")
        self.lbl_conf_pct.setStyleSheet("font-weight: 700;")
        conf_label_row.addWidget(lbl_conf_title)
        conf_label_row.addStretch()
        conf_label_row.addWidget(self.lbl_conf_pct)
        conf_box.addLayout(conf_label_row)

        slider_row = QHBoxLayout()
        self.conf_slider = QSlider(Qt.Orientation.Horizontal)
        self.conf_slider.setRange(0, 100)
        self.conf_slider.setValue(65)
        self.conf_slider.valueChanged.connect(self._on_slider_changed)

        self.conf_spin = QDoubleSpinBox()
        self.conf_spin.setRange(0.0, 1.0)
        self.conf_spin.setSingleStep(0.01)
        self.conf_spin.setValue(0.65)
        self.conf_spin.valueChanged.connect(self._on_spin_changed)

        slider_row.addWidget(self.conf_slider, stretch=1)
        slider_row.addWidget(self.conf_spin)
        conf_box.addLayout(slider_row)
        card_layout.addLayout(conf_box)

        # Buttons Row: Save & Reset
        btn_row = QHBoxLayout()
        self.save_btn = QPushButton("Save Preferences")
        self.save_btn.setObjectName("primaryButton")
        self.save_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.save_btn.clicked.connect(self.save_settings)

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setObjectName("secondaryButton")
        self.reset_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.reset_btn.clicked.connect(self.reset_defaults)

        self.status_msg = QLabel("")
        self.status_msg.setStyleSheet("color: #30D158; font-weight: 600;")

        btn_row.addWidget(self.save_btn)
        btn_row.addWidget(self.reset_btn)
        btn_row.addWidget(self.status_msg)
        btn_row.addStretch()
        card_layout.addLayout(btn_row)

        layout.addWidget(ui_card)

        # --- Section 2: Read-Only Core Config Inspector Card ---
        inspector_card = QFrame()
        inspector_card.setObjectName("cardFrame")
        insp_layout = QVBoxLayout(inspector_card)
        insp_layout.setContentsMargins(20, 18, 20, 18)
        insp_layout.setSpacing(12)

        insp_header = QHBoxLayout()
        lbl_insp_title = QLabel("READ-ONLY CORE CONFIG INSPECTOR")
        lbl_insp_title.setObjectName("cardTitle")

        self.btn_toggle_insp = QPushButton("Collapse")
        self.btn_toggle_insp.setObjectName("secondaryButton")
        self.btn_toggle_insp.setFixedHeight(26)
        self.btn_toggle_insp.clicked.connect(self._toggle_inspector)

        btn_refresh_insp = QPushButton("Reload")
        btn_refresh_insp.setObjectName("secondaryButton")
        btn_refresh_insp.setFixedHeight(26)
        btn_refresh_insp.clicked.connect(self.refresh_inspector)

        insp_header.addWidget(lbl_insp_title)
        insp_header.addStretch()
        insp_header.addWidget(btn_refresh_insp)
        insp_header.addWidget(self.btn_toggle_insp)
        insp_layout.addLayout(insp_header)

        lbl_insp_desc = QLabel(
            "Displays active audio, detector, geofence, and ML settings from config/config.yaml. "
            "Core parameters are locked from accidental editing."
        )
        lbl_insp_desc.setObjectName("subTitle")
        insp_layout.addWidget(lbl_insp_desc)

        self.inspector_text = QPlainTextEdit()
        self.inspector_text.setReadOnly(True)
        self.inspector_text.setMinimumHeight(240)
        insp_layout.addWidget(self.inspector_text)

        layout.addWidget(inspector_card)
        layout.addStretch()

        scroll.setWidget(container)
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)
        outer_layout.addWidget(scroll)

    def load_settings(self) -> None:
        """Loads UI settings from config_store and sets widget values."""
        ui_cfg = config_store.get_ui_config()
        theme = str(ui_cfg.get("theme", "system")).lower()
        idx = self.theme_combo.findData(theme)
        if idx >= 0:
            self.theme_combo.setCurrentIndex(idx)

        show_spec = bool(ui_cfg.get("show_spectrogram", True))
        self.spec_checkbox.setChecked(show_spec)

        min_conf = float(ui_cfg.get("min_confidence", 0.55))
        self.conf_slider.setValue(int(min_conf * 100))
        self.conf_spin.setValue(min_conf)
        self.lbl_conf_pct.setText(f"{int(min_conf * 100)}%")

        self.refresh_inspector()

    def refresh_inspector(self) -> None:
        """Reloads core configuration YAML and displays as formatted JSON."""
        core_cfg = config_store.get_core_config()
        formatted = json.dumps(core_cfg, indent=2)
        self.inspector_text.setPlainText(formatted)

    def _on_slider_changed(self, val: int) -> None:
        float_val = val / 100.0
        if abs(self.conf_spin.value() - float_val) > 0.005:
            self.conf_spin.setValue(float_val)
        self.lbl_conf_pct.setText(f"{val}%")
        self.min_confidence_changed.emit(float_val)

    def _on_spin_changed(self, val: float) -> None:
        slider_val = int(val * 100)
        if self.conf_slider.value() != slider_val:
            self.conf_slider.setValue(slider_val)
        self.lbl_conf_pct.setText(f"{slider_val}%")
        self.min_confidence_changed.emit(val)

    def _on_theme_changed(self, idx: int) -> None:
        theme = self.theme_combo.currentData()
        theme_manager.set_mode(theme)
        self.theme_changed_signal.emit(theme)

    def save_settings(self) -> None:
        """Saves UI settings surgically to config/config.yaml."""
        theme = self.theme_combo.currentData()
        show_spec = self.macro_or_spec = self.spec_checkbox.isChecked()
        min_conf = self.conf_spin.value()

        ui_dict = {
            "theme": theme,
            "show_spectrogram": show_spec,
            "min_confidence": min_conf,
        }
        config_store.save_ui_config(ui_dict)
        theme_manager.set_mode(theme)
        self.status_msg.setText("Settings saved.")
        self.refresh_inspector()

    def reset_defaults(self) -> None:
        """Resets UI settings to default values."""
        config_store.save_ui_config(DEFAULT_UI_CONFIG)
        self.load_settings()
        self.status_msg.setText("Reset to defaults.")

    def _toggle_inspector(self) -> None:
        is_hidden = self.inspector_text.isHidden()
        self.inspector_text.setHidden(not is_hidden)
        self.btn_toggle_insp.setText("Collapse" if is_hidden else "Expand")
