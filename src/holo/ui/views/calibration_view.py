"""Calibration View for Holo UI.

Provides a guided step-by-step runner for data collection, geofence boundary calibration,
model training, and holdout evaluation using existing backend functions in a background thread.
"""

from __future__ import annotations

import sys
import io
import logging
from pathlib import Path
from typing import Optional, Callable
from PySide6.QtCore import Qt, QObject, QThread, Signal, Slot
from PySide6.QtWidgets import (
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QFrame,
    QPlainTextEdit,
    QProgressBar,
    QSpinBox,
    QComboBox,
    QGroupBox,
)

from src.holo.config import DATA_DIR, BOUNDARY_INSIDE_DIR, BOUNDARY_OUTSIDE_DIR, MODEL_PATH

logger = logging.getLogger(__name__)


class StreamRedirector(io.StringIO):
    """Intercepts stdout/stderr writes and emits them via Qt Signal."""

    def __init__(self, callback: Callable[[str], None]) -> None:
        super().__init__()
        self.callback = callback

    def write(self, s: str) -> int:
        if s:
            self.callback(s)
        return len(s)

    def flush(self) -> None:
        pass


class CalibrationWorker(QObject):
    """Executes collection, training, or evaluation backend tasks in background."""

    log_message = Signal(str)
    task_finished = Signal(bool, str)

    def __init__(self, task_type: str, params: dict) -> None:
        super().__init__()
        self.task_type = task_type
        self.params = params
        self._is_cancelled = False

    def cancel(self) -> None:
        self._is_cancelled = True

    def run(self) -> None:
        """Invokes backend functions while redirecting stdout."""
        old_stdout = sys.stdout
        old_stderr = sys.stderr

        redirector = StreamRedirector(self.log_message.emit)
        sys.stdout = redirector
        sys.stderr = redirector

        success = True
        err_msg = ""

        try:
            if self.task_type == "collect_center":
                from src.holo.collect import collect_single_class, collect_group_interleaved
                cls = self.params.get("class", "all")
                samples = int(self.params.get("samples", 15))
                if cls == "all":
                    collect_group_interleaved(samples_per_class=samples, threshold=0.02)
                else:
                    collect_single_class(label=cls, num_samples=samples, threshold=0.02)

            elif self.task_type == "calibrate_geofence":
                from src.holo.collect import calibrate_geofence_session
                calibrate_geofence_session(threshold=0.02)

            elif self.task_type == "train":
                from src.holo.train import train_and_benchmark
                train_and_benchmark()

            elif self.task_type == "evaluate":
                from src.holo.evaluate import evaluate_saved_model
                holdout_only = bool(self.params.get("holdout_only", True))
                evaluate_saved_model(model_file=MODEL_PATH, holdout_only=holdout_only)

            else:
                raise ValueError(f"Unknown task type: {self.task_type}")

        except Exception as e:
            success = False
            err_msg = str(e)
            self.log_message.emit(f"\n[Error] Task failed: {e}\n")
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.task_finished.emit(success, err_msg)


class CalibrationView(QWidget):
    """Step-by-step workflow runner for calibration, dataset collection, and training."""

    def __init__(self, parent: Optional[QWidget] = None) -> None:
        super().__init__(parent)
        self.worker_thread: Optional[QThread] = None
        self.worker: Optional[CalibrationWorker] = None

        self._init_ui()
        self.refresh_dataset_counts()

    def _init_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(28, 24, 28, 24)
        layout.setSpacing(18)

        # Header
        title_box = QVBoxLayout()
        title_box.setSpacing(4)
        header_title = QLabel("Guided Calibration & Training")
        header_title.setObjectName("headerTitle")
        sub_title = QLabel("Step-by-step wizard to collect palm rest taps, calibrate boundary geofence, and train model")
        sub_title.setObjectName("subTitle")
        title_box.addWidget(header_title)
        title_box.addWidget(sub_title)
        layout.addLayout(title_box)

        # Status Summary Bar (Card)
        summary_card = QFrame()
        summary_card.setObjectName("cardFrame")
        sum_layout = QHBoxLayout(summary_card)
        sum_layout.setContentsMargins(16, 12, 16, 12)
        sum_layout.setSpacing(24)

        self.lbl_center_count = QLabel("Center Samples: 0")
        self.lbl_inside_count = QLabel("Inside Boundary: 0")
        self.lbl_outside_count = QLabel("Outside Boundary: 0")
        self.lbl_model_status = QLabel("Model: None")

        sum_layout.addWidget(self.lbl_center_count)
        sum_layout.addWidget(self.lbl_inside_count)
        sum_layout.addWidget(self.lbl_outside_count)
        sum_layout.addWidget(self.lbl_model_status)
        sum_layout.addStretch()

        btn_refresh_counts = QPushButton("Refresh Counts")
        btn_refresh_counts.setObjectName("secondaryButton")
        btn_refresh_counts.setCursor(Qt.CursorShape.PointingHandCursor)
        btn_refresh_counts.clicked.connect(self.refresh_dataset_counts)
        sum_layout.addWidget(btn_refresh_counts)
        layout.addWidget(summary_card)

        # Steps Row (4 Action Cards)
        steps_layout = QHBoxLayout()
        steps_layout.setSpacing(14)

        # Step 1: Collect Center Gestures
        card_step1 = QFrame()
        card_step1.setObjectName("cardFrame")
        c1_layout = QVBoxLayout(card_step1)
        c1_layout.setContentsMargins(14, 14, 14, 14)
        c1_layout.setSpacing(8)

        lbl_s1_title = QLabel("Step 1: Center Gestures")
        lbl_s1_title.setObjectName("cardTitle")
        lbl_s1_desc = QLabel("Record center taps for left & right palm rests and ambient noise.")
        lbl_s1_desc.setObjectName("subTitle")
        lbl_s1_desc.setWordWrap(True)

        self.s1_samples_spin = QSpinBox()
        self.s1_samples_spin.setRange(5, 100)
        self.s1_samples_spin.setValue(15)
        self.s1_samples_spin.setPrefix("Samples: ")

        self.btn_run_step1 = QPushButton("Run Collection")
        self.btn_run_step1.setObjectName("secondaryButton")
        self.btn_run_step1.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_step1.clicked.connect(self._run_collect_center)

        c1_layout.addWidget(lbl_s1_title)
        c1_layout.addWidget(lbl_s1_desc)
        c1_layout.addStretch()
        c1_layout.addWidget(self.s1_samples_spin)
        c1_layout.addWidget(self.btn_run_step1)
        steps_layout.addWidget(card_step1)

        # Step 2: Calibrate Geofence Boundaries
        card_step2 = QFrame()
        card_step2.setObjectName("cardFrame")
        c2_layout = QVBoxLayout(card_step2)
        c2_layout.setContentsMargins(14, 14, 14, 14)
        c2_layout.setSpacing(8)

        lbl_s2_title = QLabel("Step 2: Geofence Boundaries")
        lbl_s2_title.setObjectName("cardTitle")
        lbl_s2_desc = QLabel("Calibrate perimeter taps and external desk knocks to reject desk bumps.")
        lbl_s2_desc.setObjectName("subTitle")
        lbl_s2_desc.setWordWrap(True)

        self.btn_run_step2 = QPushButton("Calibrate Geofence")
        self.btn_run_step2.setObjectName("secondaryButton")
        self.btn_run_step2.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_step2.clicked.connect(self._run_calibrate_geofence)

        c2_layout.addWidget(lbl_s2_title)
        c2_layout.addWidget(lbl_s2_desc)
        c2_layout.addStretch()
        c2_layout.addWidget(self.btn_run_step2)
        steps_layout.addWidget(card_step2)

        # Step 3: Train & Benchmark Model
        card_step3 = QFrame()
        card_step3.setObjectName("cardFrame")
        c3_layout = QVBoxLayout(card_step3)
        c3_layout.setContentsMargins(14, 14, 14, 14)
        c3_layout.setSpacing(8)

        lbl_s3_title = QLabel("Step 3: Train Model")
        lbl_s3_title.setObjectName("cardTitle")
        lbl_s3_desc = QLabel("Fit hybrid ML + Fourier triangulation model and geofence classifier.")
        lbl_s3_desc.setObjectName("subTitle")
        lbl_s3_desc.setWordWrap(True)

        self.btn_run_step3 = QPushButton("Train & Benchmark")
        self.btn_run_step3.setObjectName("primaryButton")
        self.btn_run_step3.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_step3.clicked.connect(self._run_train)

        c3_layout.addWidget(lbl_s3_title)
        c3_layout.addWidget(lbl_s3_desc)
        c3_layout.addStretch()
        c3_layout.addWidget(self.btn_run_step3)
        steps_layout.addWidget(card_step3)

        # Step 4: Evaluate Model
        card_step4 = QFrame()
        card_step4.setObjectName("cardFrame")
        c4_layout = QVBoxLayout(card_step4)
        c4_layout.setContentsMargins(14, 14, 14, 14)
        c4_layout.setSpacing(8)

        lbl_s4_title = QLabel("Step 4: Evaluate")
        lbl_s4_title.setObjectName("cardTitle")
        lbl_s4_desc = QLabel("Validate holdout accuracy, confusion matrix, and geofence rejection.")
        lbl_s4_desc.setObjectName("subTitle")
        lbl_s4_desc.setWordWrap(True)

        self.btn_run_step4 = QPushButton("Evaluate Model")
        self.btn_run_step4.setObjectName("secondaryButton")
        self.btn_run_step4.setCursor(Qt.CursorShape.PointingHandCursor)
        self.btn_run_step4.clicked.connect(self._run_evaluate)

        c4_layout.addWidget(lbl_s4_title)
        c4_layout.addWidget(lbl_s4_desc)
        c4_layout.addStretch()
        c4_layout.addWidget(self.btn_run_step4)
        steps_layout.addWidget(card_step4)

        layout.addLayout(steps_layout)

        # Log Output Console (Card)
        console_card = QFrame()
        console_card.setObjectName("cardFrame")
        con_layout = QVBoxLayout(console_card)
        con_layout.setContentsMargins(16, 14, 16, 14)
        con_layout.setSpacing(10)

        con_header = QHBoxLayout()
        lbl_console = QLabel("WORKFLOW CONSOLE OUTPUT")
        lbl_console.setObjectName("subTitle")

        self.btn_stop_task = QPushButton("Cancel Task")
        self.btn_stop_task.setObjectName("secondaryButton")
        self.btn_stop_task.setEnabled(False)
        self.btn_stop_task.clicked.connect(self._cancel_task)

        btn_clear_console = QPushButton("Clear Console")
        btn_clear_console.setObjectName("secondaryButton")
        btn_clear_console.clicked.connect(lambda: self.console.clear())

        con_header.addWidget(lbl_console)
        con_header.addStretch()
        con_header.addWidget(self.btn_stop_task)
        con_header.addWidget(btn_clear_console)
        con_layout.addLayout(con_header)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        con_layout.addWidget(self.console)

        layout.addWidget(console_card, stretch=1)

    def refresh_dataset_counts(self) -> None:
        """Counts recorded .npy samples on disk."""
        left_count = len(list((DATA_DIR / "left_tap").glob("*.npy"))) if (DATA_DIR / "left_tap").exists() else 0
        right_count = len(list((DATA_DIR / "right_tap").glob("*.npy"))) if (DATA_DIR / "right_tap").exists() else 0
        noise_count = len(list((DATA_DIR / "noise").glob("*.npy"))) if (DATA_DIR / "noise").exists() else 0
        in_count = len(list(BOUNDARY_INSIDE_DIR.glob("*.npy"))) if BOUNDARY_INSIDE_DIR.exists() else 0
        out_count = len(list(BOUNDARY_OUTSIDE_DIR.glob("*.npy"))) if BOUNDARY_OUTSIDE_DIR.exists() else 0

        self.lbl_center_count.setText(f"Center Taps: Left={left_count}, Right={right_count}, Noise={noise_count}")
        self.lbl_inside_count.setText(f"Inside Boundary: {in_count}")
        self.lbl_outside_count.setText(f"Outside Boundary: {out_count}")
        self.lbl_model_status.setText(f"Model: {'Found' if MODEL_PATH.exists() else 'Missing'}")

    def _start_worker(self, task_type: str, params: dict) -> None:
        """Starts a background calibration worker."""
        if self.worker_thread is not None and self.worker_thread.isRunning():
            self.console.appendPlainText("[Warning] Another task is already running.\n")
            return

        self._set_buttons_enabled(False)
        self.btn_stop_task.setEnabled(True)
        self.console.appendPlainText(f"\n>>> Starting {task_type}...\n")

        self.worker_thread = QThread()
        self.worker = CalibrationWorker(task_type, params)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.log_message.connect(self._append_log)
        self.worker.task_finished.connect(self._on_task_finished)

        self.worker_thread.start()

    def _append_log(self, text: str) -> None:
        self.console.insertPlainText(text)
        # Auto-scroll to end
        sb = self.console.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _on_task_finished(self, success: bool, err_msg: str) -> None:
        if success:
            self.console.appendPlainText("\n>>> Task completed successfully.\n")
        else:
            self.console.appendPlainText(f"\n>>> Task finished with error: {err_msg}\n")

        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(1000)
            self.worker_thread = None
            self.worker = None

        self._set_buttons_enabled(True)
        self.btn_stop_task.setEnabled(False)
        self.refresh_dataset_counts()

    def _cancel_task(self) -> None:
        if self.worker is not None:
            self.worker.cancel()
            self.console.appendPlainText("\n[Cancelling task...]\n")
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(1500)
            self.worker_thread = None
            self.worker = None
        self._set_buttons_enabled(True)
        self.btn_stop_task.setEnabled(False)

    def _set_buttons_enabled(self, enabled: bool) -> None:
        self.btn_run_step1.setEnabled(enabled)
        self.btn_run_step2.setEnabled(enabled)
        self.btn_run_step3.setEnabled(enabled)
        self.btn_run_step4.setEnabled(enabled)

    def _run_collect_center(self) -> None:
        samples = self.s1_samples_spin.value()
        self._start_worker("collect_center", {"class": "all", "samples": samples})

    def _run_calibrate_geofence(self) -> None:
        self._start_worker("calibrate_geofence", {})

    def _run_train(self) -> None:
        self._start_worker("train", {})

    def _run_evaluate(self) -> None:
        self._start_worker("evaluate", {"holdout_only": True})
