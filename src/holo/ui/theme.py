"""macOS-inspired Dark/Light/System theme engine and pyqtgraph color synchronisation."""

from __future__ import annotations

import logging
from typing import Literal, Tuple
from PySide6.QtCore import QObject, Signal, Qt
from PySide6.QtGui import QGuiApplication, QColor, QFont
import pyqtgraph as pg

logger = logging.getLogger(__name__)

ThemeMode = Literal["system", "dark", "light"]


class ThemeColors:
    """Design tokens matching macOS Sonoma/Sequoia guidelines."""

    # Dark Mode
    DARK_BG = "#1E1E1E"
    DARK_CARD = "#282828"
    DARK_SIDEBAR = "#181818"
    DARK_HOVER = "#323232"
    DARK_BORDER = "#383838"
    DARK_TEXT_PRIMARY = "#FFFFFF"
    DARK_TEXT_SECONDARY = "#A0A0A0"
    DARK_TEXT_MUTED = "#6C6C6C"
    DARK_ACCENT = "#0A84FF"  # macOS System Blue Dark
    DARK_ACCENT_HOVER = "#409CFF"
    DARK_SUCCESS = "#30D158"
    DARK_ERROR = "#FF453A"
    DARK_WARNING = "#FF9F0A"

    # Light Mode
    LIGHT_BG = "#F5F5F7"
    LIGHT_CARD = "#FFFFFF"
    LIGHT_SIDEBAR = "#EBEBF0"
    LIGHT_HOVER = "#E1E1E6"
    LIGHT_BORDER = "#D6D6DC"
    LIGHT_TEXT_PRIMARY = "#1D1D1F"
    LIGHT_TEXT_SECONDARY = "#6E6E73"
    LIGHT_TEXT_MUTED = "#86868B"
    LIGHT_ACCENT = "#007AFF"  # macOS System Blue Light
    LIGHT_ACCENT_HOVER = "#0062CC"
    LIGHT_SUCCESS = "#34C759"
    LIGHT_ERROR = "#FF3B30"
    LIGHT_WARNING = "#FF9500"


def generate_qss(is_dark: bool) -> str:
    """Generates complete macOS-style QSS stylesheet."""
    c = ThemeColors
    bg = c.DARK_BG if is_dark else c.LIGHT_BG
    card = c.DARK_CARD if is_dark else c.LIGHT_CARD
    sidebar = c.DARK_SIDEBAR if is_dark else c.LIGHT_SIDEBAR
    hover = c.DARK_HOVER if is_dark else c.LIGHT_HOVER
    border = c.DARK_BORDER if is_dark else c.LIGHT_BORDER
    txt = c.DARK_TEXT_PRIMARY if is_dark else c.LIGHT_TEXT_PRIMARY
    sec_txt = c.DARK_TEXT_SECONDARY if is_dark else c.LIGHT_TEXT_SECONDARY
    accent = c.DARK_ACCENT if is_dark else c.LIGHT_ACCENT
    accent_hover = c.DARK_ACCENT_HOVER if is_dark else c.LIGHT_ACCENT_HOVER

    return f"""
    * {{
        font-family: -apple-system, 'SF Pro Text', 'SF Pro Display', 'Helvetica Neue', Arial, sans-serif;
        font-size: 13px;
        color: {txt};
    }}

    QMainWindow, QDialog, QWidget#centralWidget {{
        background-color: {bg};
    }}

    /* Sidebar Navigation */
    QFrame#sidebarPanel {{
        background-color: {sidebar};
        border-right: 1px solid {border};
    }}

    QPushButton#navButton {{
        background-color: transparent;
        border: none;
        border-radius: 6px;
        padding: 8px 14px;
        text-align: left;
        font-weight: 500;
        color: {sec_txt};
    }}

    QPushButton#navButton:hover {{
        background-color: {hover};
        color: {txt};
    }}

    QPushButton#navButton:checked {{
        background-color: {accent};
        color: #FFFFFF;
        font-weight: 600;
    }}

    /* Card Panels */
    QFrame#cardFrame {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 10px;
        padding: 16px;
    }}

    /* Headings and Labels */
    QLabel#headerTitle {{
        font-size: 20px;
        font-weight: 700;
        color: {txt};
    }}

    QLabel#subTitle {{
        font-size: 13px;
        font-weight: 400;
        color: {sec_txt};
    }}

    QLabel#cardTitle {{
        font-size: 15px;
        font-weight: 600;
        color: {txt};
    }}

    QLabel#metricValue {{
        font-size: 26px;
        font-weight: 700;
        color: {accent};
    }}

    /* Buttons */
    QPushButton#primaryButton {{
        background-color: {accent};
        color: #FFFFFF;
        border: none;
        border-radius: 8px;
        padding: 8px 18px;
        font-weight: 600;
    }}

    QPushButton#primaryButton:hover {{
        background-color: {accent_hover};
    }}

    QPushButton#secondaryButton {{
        background-color: {card};
        color: {txt};
        border: 1px solid {border};
        border-radius: 8px;
        padding: 7px 16px;
        font-weight: 500;
    }}

    QPushButton#secondaryButton:hover {{
        background-color: {hover};
    }}

    QPushButton#pillToggleActive {{
        background-color: {accent};
        color: #FFFFFF;
        border: none;
        border-radius: 14px;
        padding: 6px 16px;
        font-weight: 600;
    }}

    QPushButton#pillToggleInactive {{
        background-color: {card};
        color: {sec_txt};
        border: 1px solid {border};
        border-radius: 14px;
        padding: 6px 16px;
        font-weight: 500;
    }}

    /* Inputs, ComboBox, SpinBox */
    QSpinBox, QDoubleSpinBox, QLineEdit {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 6px;
        padding: 6px 10px;
        color: {txt};
        selection-background-color: {accent};
    }}

    QComboBox {{
        background-color: {card};
        border: 1px solid {border};
        border-radius: 6px;
        padding-left: 12px;
        padding-right: 28px;
        padding-top: 0px;
        padding-bottom: 0px;
        min-height: 32px;
        max-height: 34px;
        color: {txt};
        selection-background-color: {accent};
    }}

    QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover, QLineEdit:hover {{
        border: 1px solid {accent};
    }}

    QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus, QLineEdit:focus {{
        border: 1px solid {accent};
    }}

    QComboBox::drop-down {{
        subcontrol-origin: padding;
        subcontrol-position: top right;
        width: 24px;
        border-left: none;
    }}

    QComboBox::down-arrow {{
        image: none;
        border-left: 4px solid transparent;
        border-right: 4px solid transparent;
        border-top: 5px solid {sec_txt};
        width: 0px;
        height: 0px;
        margin-right: 10px;
    }}

    QComboBox QAbstractItemView {{
        background-color: {card};
        border: 1px solid {border};
        selection-background-color: {accent};
        selection-color: #FFFFFF;
        border-radius: 6px;
        padding: 4px;
    }}

    QComboBox QAbstractItemView::item {{
        min-height: 28px;
        padding-left: 8px;
    }}

    /* Checkbox */
    QCheckBox {{
        color: {txt};
        font-weight: 500;
        spacing: 8px;
        min-height: 34px;
        max-height: 34px;
    }}

    /* Tables */
    QTableWidget {{
        background-color: {card};
        alternate-background-color: {hover};
        border: 1px solid {border};
        border-radius: 8px;
        gridline-color: {border};
        color: {txt};
    }}

    QHeaderView::section {{
        background-color: {card};
        color: {sec_txt};
        font-weight: 600;
        font-size: 11px;
        padding: 6px;
        border: none;
        border-bottom: 1px solid {border};
    }}

    /* Sliders */
    QSlider::groove:horizontal {{
        border: none;
        height: 4px;
        background: {border};
        border-radius: 2px;
    }}

    QSlider::sub-page:horizontal {{
        background: {accent};
        border-radius: 2px;
    }}

    QSlider::handle:horizontal {{
        background: #FFFFFF;
        border: 1px solid {border};
        width: 16px;
        height: 16px;
        margin: -6px 0;
        border-radius: 8px;
    }}

    /* Progress Bar */
    QProgressBar {{
        border: 1px solid {border};
        border-radius: 6px;
        text-align: center;
        background-color: {card};
        color: {txt};
        font-weight: 600;
        font-size: 11px;
    }}

    QProgressBar::chunk {{
        background-color: {accent};
        border-radius: 5px;
    }}

    /* Scrollbars */
    QScrollBar:vertical {{
        background: transparent;
        width: 8px;
        margin: 0;
    }}

    QScrollBar::handle:vertical {{
        background: {border};
        border-radius: 4px;
        min-height: 20px;
    }}

    QScrollBar::handle:vertical:hover {{
        background: {sec_txt};
    }}

    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
        height: 0px;
    }}

    /* Plain Text & Console */
    QPlainTextEdit {{
        background-color: {'#141414' if is_dark else '#FAFAFC'};
        border: 1px solid {border};
        border-radius: 6px;
        font-family: 'SF Mono', Menlo, Monaco, 'Courier New', monospace;
        font-size: 11px;
        padding: 8px;
        color: {txt};
    }}
    """


class ThemeManager(QObject):
    """Manages active theme state and synchronisation across Qt and pyqtgraph."""

    theme_changed = Signal(bool)  # Emits True if dark, False if light

    def __init__(self, mode: ThemeMode = "system") -> None:
        super().__init__()
        self._mode: ThemeMode = mode
        self._is_dark: bool = self._compute_is_dark(mode)

    @property
    def mode(self) -> ThemeMode:
        return self._mode

    @property
    def is_dark(self) -> bool:
        return self._is_dark

    def _compute_is_dark(self, mode: ThemeMode) -> bool:
        if mode == "dark":
            return True
        if mode == "light":
            return False

        # Mode is "system": detect from OS styleHints
        app = QGuiApplication.instance()
        if app is not None:
            hints = app.styleHints()
            if hints is not None:
                color_scheme = hints.colorScheme()
                return color_scheme == Qt.ColorScheme.Dark

        return True  # Modern default fallback is dark

    def set_mode(self, mode: ThemeMode) -> None:
        """Sets the mode and reapplies theme."""
        if mode not in ("system", "dark", "light"):
            mode = "system"
        self._mode = mode
        self.apply()

    def apply(self) -> None:
        """Applies stylesheet to QApplication and syncs pyqtgraph."""
        new_is_dark = self._compute_is_dark(self._mode)
        self._is_dark = new_is_dark

        app = QGuiApplication.instance()
        if app is not None:
            qss = generate_qss(new_is_dark)
            app.setStyleSheet(qss)

        self._sync_pyqtgraph(new_is_dark)
        self.theme_changed.emit(new_is_dark)

    def _sync_pyqtgraph(self, is_dark: bool) -> None:
        """Synchronises pyqtgraph styling."""
        c = ThemeColors
        bg = c.DARK_CARD if is_dark else c.LIGHT_CARD
        fg = c.DARK_TEXT_PRIMARY if is_dark else c.LIGHT_TEXT_PRIMARY

        pg.setConfigOption("background", bg)
        pg.setConfigOption("foreground", fg)
        pg.setConfigOption("antialias", True)

    def get_plot_pens(self) -> Tuple[Any, Any, Any]:
        """Returns pyqtgraph pens for waveform, spectrum, and grid."""
        c = ThemeColors
        if self._is_dark:
            wave_pen = pg.mkPen(color=c.DARK_ACCENT, width=1.5)
            spec_pen = pg.mkPen(color=c.DARK_SUCCESS, width=1.5)
            grid_pen = pg.mkPen(color=c.DARK_BORDER, width=0.8, style=Qt.PenStyle.DotLine)
        else:
            wave_pen = pg.mkPen(color=c.LIGHT_ACCENT, width=1.5)
            spec_pen = pg.mkPen(color=c.LIGHT_SUCCESS, width=1.5)
            grid_pen = pg.mkPen(color=c.LIGHT_BORDER, width=0.8, style=Qt.PenStyle.DotLine)
        return wave_pen, spec_pen, grid_pen


# Global ThemeManager instance
theme_manager = ThemeManager()
