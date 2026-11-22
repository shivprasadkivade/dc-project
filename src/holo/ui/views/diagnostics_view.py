"""Signal & Geofence Diagnostics View for Holo UI.

Features high-cadence 60 FPS pyqtgraph real-time oscilloscope and FFT spectrum plots
backed by fixed-size ring buffers, alongside a comprehensive geofence decision log table.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Optional, Dict, Any
import numpy as np

from PySide6.QtCore import Qt, Slot, QTimer
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QTableWidget,
    QTableWidgetItem,
    QHeaderView,
    QSplitter,
)
import pyqtgraph as pg

from src.holo.config import SAMPLE_RATE
from src.holo.ui.theme import theme_manager, ThemeColors


class DiagnosticsView(QWidget):
    """Real-time signal analysis and geofence verification diagnostics."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self._is_paused: bool = False
        self.buffer_size: int = 2048
        self.sample_rate: int = SAMPLE_RATE

        # Pre-allocated fixed-size ring buffer for 60 FPS oscilloscope
        self.waveform_buffer = np.zeros(self.buffer_size, dtype=np.float32)

        # Ring buffer for FFT spectrum calculation
        self.fft_buffer = deque(maxlen=self.buffer_size)

        self._init_ui()

        # Render throttler timer (runs at ~40 FPS = 25 ms interval)
        self._render_timer = QTimer(self)
        self._render_timer.setInterval(25)
        self._render_timer.timeout.connect(self._render_plots)
        self._render_timer.start()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        # Header
        header_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title_box.setSpacing(4)

        header_title = QLabel("Signal & Geofence Diagnostics")
        header_title.setObjectName("headerTitle")
        sub_title = QLabel("Real-time oscilloscope, Lamb-wave frequency analysis, and geofence audit log")
        sub_title.setObjectName("subTitle")

        title_box.addWidget(header_title)
        title_box.addWidget(sub_title)
        header_layout.addLayout(title_box)

        header_layout.addStretch()

        # Pause / Resume Button
        self.pause_btn = QPushButton("Pause Visualizer")
        self.pause_btn.setObjectName("secondaryButton")
        self.pause_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.pause_btn.clicked.connect(self._toggle_pause)

        # Clear Log Button
        self.clear_btn = QPushButton("Clear Log")
        self.clear_btn.setObjectName("secondaryButton")
        self.clear_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.clear_btn.clicked.connect(self._clear_table)

        header_layout.addWidget(self.pause_btn)
        header_layout.addWidget(self.clear_btn)
        layout.addLayout(header_layout)

        # Splitter dividing top plots from bottom event log
        splitter = QSplitter(Qt.Orientation.Vertical)
        splitter.setChildrenCollapsible(False)

        # --- Top Section: Real-Time Plots ---
        plots_container = QFrame()
        plots_container.setObjectName("cardFrame")
        plots_layout = QHBoxLayout(plots_container)
        plots_layout.setContentsMargins(14, 14, 14, 14)
        plots_layout.setSpacing(14)

        # Waveform Oscilloscope Plot
        wave_box = QVBoxLayout()
        wave_header = QLabel("REAL-TIME OSCILLOSCOPE (TIME-DOMAIN)")
        wave_header.setObjectName("subTitle")
        self.wave_plot = pg.PlotWidget()
        self.wave_plot.setYRange(-0.5, 0.5)
        self.wave_plot.setXRange(0, self.buffer_size)
        self.wave_plot.showGrid(x=True, y=True, alpha=0.3)
        self.wave_plot.setLabel("left", "Amplitude")
        self.wave_plot.setLabel("bottom", "Samples")
        wave_pen, spec_pen, _ = theme_manager.get_plot_pens()
        self.wave_curve = self.wave_plot.plot(pen=wave_pen)
        wave_box.addWidget(wave_header)
        wave_box.addWidget(self.wave_plot)
        plots_layout.addLayout(wave_box)

        # FFT Spectrum Plot
        spec_box = QVBoxLayout()
        spec_header = QLabel("FFT SPATIAL FREQUENCY SPECTRUM (0 - 20 kHz)")
        spec_header.setObjectName("subTitle")
        self.spec_plot = pg.PlotWidget()
        self.spec_plot.setYRange(0, 1.0)
        self.spec_plot.setXRange(0, 20000)
        self.spec_plot.showGrid(x=True, y=True, alpha=0.3)
        self.spec_plot.setLabel("left", "Magnitude")
        self.spec_plot.setLabel("bottom", "Frequency", units="Hz")

        # Lamb-wave dispersion reference band highlight regions
        # High frequency Lamb-wave band: [3000 Hz, 14000 Hz]
        # Low frequency Lamb-wave band: [150 Hz, 900 Hz]
        hf_region = pg.LinearRegionItem([3000, 14000], movable=False, brush=pg.mkBrush(10, 132, 255, 30))
        lf_region = pg.LinearRegionItem([150, 900], movable=False, brush=pg.mkBrush(255, 159, 10, 30))
        self.spec_plot.addItem(hf_region)
        self.spec_plot.addItem(lf_region)

        self.spec_curve = self.spec_plot.plot(pen=spec_pen)
        spec_box.addWidget(spec_header)
        spec_box.addWidget(self.spec_plot)
        plots_layout.addLayout(spec_box)

        splitter.addWidget(plots_container)

        # --- Bottom Section: Event Audit Table ---
        table_container = QFrame()
        table_container.setObjectName("cardFrame")
        table_layout = QVBoxLayout(table_container)
        table_layout.setContentsMargins(14, 14, 14, 14)
        table_layout.setSpacing(10)

        table_header = QLabel("GEOFENCE AUDIT & TAP INSPECTION LOG")
        table_header.setObjectName("subTitle")
        table_layout.addWidget(table_header)

        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Timestamp",
            "Zone / Gesture",
            "Confidence",
            "Geofence",
            "Inside Prob",
            "Latency",
            "Decision Reason / Physical Stage",
        ])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(6, QHeaderView.ResizeMode.Stretch)

        self.table.setAlternatingRowColors(True)
        self.table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        table_layout.addWidget(self.table)

        splitter.addWidget(table_container)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 4)

        layout.addWidget(splitter)

    @Slot(np.ndarray)
    def push_audio_chunk(self, chunk: np.ndarray) -> None:
        """Pushes raw audio chunk into fixed-size ring buffer."""
        if self._is_paused or len(chunk) == 0:
            return

        chunk_len = len(chunk)
        if chunk_len >= self.buffer_size:
            self.waveform_buffer[:] = chunk[-self.buffer_size:]
        else:
            # Shift buffer left and insert new chunk at right
            self.waveform_buffer[:-chunk_len] = self.waveform_buffer[chunk_len:]
            self.waveform_buffer[-chunk_len:] = chunk

        self.fft_buffer.extend(chunk)

    def _render_plots(self) -> None:
        """Throttled render callback updating pyqtgraph curves."""
        if self._is_paused:
            return

        # 1. Update oscilloscope
        self.wave_curve.setData(self.waveform_buffer)

        # 2. Update FFT spectrum
        if len(self.fft_buffer) >= 512:
            sig = np.array(list(self.fft_buffer)[-1024:], dtype=np.float32)
            n = len(sig)
            # Apply Hanning window
            win = np.hanning(n)
            fft_vals = np.abs(np.fft.rfft(sig * win)) / (n / 2.0)
            freqs = np.fft.rfftfreq(n, d=1.0 / self.sample_rate)
            self.spec_curve.setData(freqs, fft_vals)

    @Slot(dict)
    def handle_tap_event(self, event: Dict[str, Any]) -> None:
        """Logs tap classification and geofence decision to table."""
        ts = event.get("timestamp", time.time())
        time_str = time.strftime("%H:%M:%S", time.localtime(ts))
        label = str(event.get("label", "unknown")).upper()
        conf = float(event.get("confidence", 0.0))
        passed = bool(event.get("geofence_passed", False))
        latency = float(event.get("latency_ms", 0.0))
        reason = str(event.get("geofence_reason", ""))

        gf_details = event.get("geofence_details", {})
        prob_in = float(gf_details.get("prob_inside", 1.0 if passed else 0.0))

        # Insert new row at top
        row = 0
        self.table.insertRow(row)

        self.table.setItem(row, 0, QTableWidgetItem(time_str))
        self.table.setItem(row, 1, QTableWidgetItem(label))
        self.table.setItem(row, 2, QTableWidgetItem(f"{conf * 100:.1f}%"))

        gf_item = QTableWidgetItem("PASS" if passed else "BLOCKED")
        if passed:
            gf_item.setForeground(Qt.GlobalColor.green)
        else:
            gf_item.setForeground(Qt.GlobalColor.red)
        self.table.setItem(row, 3, gf_item)

        self.table.setItem(row, 4, QTableWidgetItem(f"{prob_in:.2f}"))
        self.table.setItem(row, 5, QTableWidgetItem(f"{latency:.1f} ms"))
        self.table.setItem(row, 6, QTableWidgetItem(reason))

        # Limit to 100 rows
        while self.table.rowCount() > 100:
            self.table.removeRow(self.table.rowCount() - 1)

    def _toggle_pause(self) -> None:
        self._is_paused = not self._is_paused
        if self._is_paused:
            self.pause_btn.setText("Resume Visualizer")
        else:
            self.pause_btn.setText("Pause Visualizer")

    def _clear_table(self) -> None:
        self.table.setRowCount(0)
