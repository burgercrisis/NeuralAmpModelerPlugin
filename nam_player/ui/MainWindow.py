"""
PyQt6 UI for the NAM Multi-Knob Player.

MainWindow with dynamic knob dials, model info display, file browser, and transport controls.
"""

import logging
import sys
from pathlib import Path
from typing import List, Optional

from PyQt6.QtCore import Qt, pyqtSignal, QSize
from PyQt6.QtGui import QFont
from PyQt6.QtWidgets import (
    QApplication,
    QDial,
    QFileDialog,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QPushButton,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# KnobWidget — dial + label + value display
# ---------------------------------------------------------------------------

class KnobWidget(QGroupBox):
    """A single knob control for a model parameter."""

    valueChanged = pyqtSignal(int, float)  # (knob_index, normalized_value)

    def __init__(
        self,
        name: str,
        min_val: float = 0.0,
        max_val: float = 1.0,
        default: float = 0.5,
        parent=None,
    ):
        super().__init__(name, parent)
        self._knob_index = 0
        self._min = float(min_val)
        self._max = float(max_val)
        self._default = float(default)

        self._setup_ui()

    def _setup_ui(self):
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setFixedSize(90, 110)

        layout = QVBoxLayout()
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)

        # Dial
        self._dial = QDial()
        self._dial.setMinimumWidth(60)
        self._dial.setMaximumWidth(60)
        self._dial.setNotchesVisible(True)
        self._dial.setWrapping(False)
        self._dial.setRange(0, 1000)
        self._setValueFromNormalized(self._default)
        self._dial.valueChanged.connect(self._on_dial_changed)

        # Value label
        self._label = QLabel()
        self._label.setFont(QFont("Monospace", 9))
        self._label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._update_label(self._default)

        layout.addWidget(self._dial, alignment=Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self._label)
        self.setLayout(layout)

    def _setValueFromNormalized(self, value: float):
        """Set dial position from a value in [min, max] range."""
        if self._max == self._min:
            normalized = 0.5
        else:
            normalized = (value - self._min) / (self._max - self._min)
        self._dial.setValue(int(normalized * 1000))

    def _on_dial_changed(self, raw: int):
        """Dial value changed (0-1000)."""
        value = self._min + (raw / 1000.0) * (self._max - self._min)
        self._update_label(value)
        self.valueChanged.emit(self._knob_index, value)

    def _update_label(self, value: float):
        self._label.setText(f"{value:.2f}")

    def set_value(self, value: float):
        """Set knob value externally (updates dial position)."""
        self._setValueFromNormalized(value)
        self._update_label(value)

    @property
    def knob_index(self) -> int:
        return self._knob_index

    @knob_index.setter
    def knob_index(self, idx: int):
        self._knob_index = idx


# ---------------------------------------------------------------------------
# MainWindow
# ---------------------------------------------------------------------------

class MainWindow(QMainWindow):
    """Main application window for the NAM Multi-Knob Player."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NAM Multi-Knob Player")
        self.setMinimumSize(500, 350)
        self.resize(600, 400)

        # State
        self._model = None
        self._metadata = None
        self._engine = None
        self._knob_widgets: List[KnobWidget] = []

        self._setup_ui()

    # ------------------------------------------------------------------
    # UI Setup
    # ------------------------------------------------------------------

    def _setup_ui(self):
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        layout.setSpacing(8)

        # --- Menu bar ---
        menubar = self.menuBar()
        file_menu = menubar.addMenu("&File")
        open_action = file_menu.addAction("&Open .nam...")
        open_action.triggered.connect(self._open_nam)
        close_action = file_menu.addAction("&Close Model")
        close_action.triggered.connect(self._close_model)
        file_menu.addSeparator()
        exit_action = file_menu.addAction("E&xit")
        exit_action.triggered.connect(self.close)

        audio_menu = menubar.addMenu("&Audio")
        devices_action = audio_menu.addAction("&Devices...")
        devices_action.triggered.connect(self._show_audio_devices)

        # --- Model info ---
        self._info_group = QGroupBox("Model")
        info_layout = QHBoxLayout()
        self._info_label = QLabel("No model loaded")
        self._info_label.setFont(QFont("Sans", 10))
        info_layout.addWidget(self._info_label)
        self._info_group.setLayout(info_layout)
        layout.addWidget(self._info_group)

        # --- Knob area ---
        self._knob_group = QGroupBox("Model Knobs")
        self._knob_layout = QHBoxLayout()
        self._knob_layout.setAlignment(Qt.AlignmentFlag.AlignLeft)
        self._empty_label = QLabel("Load a multi-knob model to see parameters")
        self._empty_label.setFont(QFont("Sans", 10))
        self._empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._knob_layout.addWidget(self._empty_label)
        self._knob_group.setLayout(self._knob_layout)
        layout.addWidget(self._knob_group)

        # --- Transport bar ---
        transport = QHBoxLayout()
        self._play_btn = QPushButton("▶ Play")
        self._stop_btn = QPushButton("■ Stop")
        self._stop_btn.setEnabled(False)
        self._play_btn.setMinimumHeight(40)
        self._stop_btn.setMinimumHeight(40)
        self._play_btn.clicked.connect(self._start_audio)
        self._stop_btn.clicked.connect(self._stop_audio)
        transport.addWidget(self._play_btn)
        transport.addWidget(self._stop_btn)
        layout.addLayout(transport)

        # --- Status bar ---
        self._status = QStatusBar()
        self.setStatusBar(self._status)
        self._status.showMessage("Ready")

    # ------------------------------------------------------------------
    # File actions
    # ------------------------------------------------------------------

    def _open_nam(self):
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Open .nam Model",
            "",
            "NAM Files (*.nam);;All Files (*)",
        )
        if path:
            self._load_model(path)

    def _load_model(self, path: str):
        """Load a .nam model."""
        from ..model.loader import load_nam

        try:
            self._status.showMessage(f"Loading {Path(path).name}...")
            QApplication.processEvents()

            model, metadata = load_nam(path)
            self._model = model
            self._metadata = metadata

            # Update info display
            knob_list = ", ".join(metadata["knob_names"]) if metadata["knob_names"] else "(none)"
            sr = metadata["sample_rate"]
            arch = metadata["architecture"]
            self._info_label.setText(
                f"{Path(path).name}  |  Architecture: {arch}  |  Knobs: {knob_list}  |  SR: {sr:.0f} Hz"
            )

            # Rebuild knob UI
            self._rebuild_knobs()

            self._status.showMessage(f"Loaded: {Path(path).name}")

        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            self._status.showMessage(f"Error: {e}")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.critical(self, "Load Error", f"Failed to load model:\n\n{e}")

    def _close_model(self):
        """Unload the current model."""
        self._stop_audio()
        self._model = None
        self._metadata = None
        self._info_label.setText("No model loaded")
        self._rebuild_knobs()
        self._status.showMessage("Model unloaded")

    # ------------------------------------------------------------------
    # Knob management
    # ------------------------------------------------------------------

    def _rebuild_knobs(self):
        """Clear and rebuild knob controls based on loaded model."""

        # Remove old knob widgets
        for w in self._knob_widgets:
            self._knob_layout.removeWidget(w)
            w.deleteLater()
        self._knob_widgets.clear()

        # Show/hide empty label
        self._empty_label.setVisible(self._model is None or len(self._metadata.get("knob_names", [])) == 0)

        if self._model is None:
            return

        knob_names = self._metadata.get("knob_names", [])
        knob_meta = self._metadata.get("knob_metadata", {})

        if not knob_names:
            return

        for i, name in enumerate(knob_names):
            meta = knob_meta.get(name, {})
            knob = KnobWidget(
                name,
                min_val=meta.get("min_value", 0.0),
                max_val=meta.get("max_value", 1.0),
                default=meta.get("default_value", 0.5),
            )
            knob.knob_index = i
            knob.valueChanged.connect(self._on_knob_changed)
            self._knob_layout.addWidget(knob)
            self._knob_widgets.append(knob)

    def _on_knob_changed(self, knob_idx: int, value: float):
        """A knob value changed."""
        if self._engine is not None:
            self._engine.set_knob(knob_idx, value)

    # ------------------------------------------------------------------
    # Audio controls
    # ------------------------------------------------------------------

    def _start_audio(self):
        if self._model is None:
            return
        if self._engine is not None and self._engine.is_running:
            return

        from .audio.engine import AudioEngine

        try:
            sr = self._metadata.get("sample_rate", 48000)
            knob_count = len(self._metadata.get("knob_names", []))
            self._engine = AudioEngine(
                self._model,
                sample_rate=int(sr),
                num_knobs=knob_count,
            )

            # Initialize knob values on engine
            for i, w in enumerate(self._knob_widgets):
                self._engine.set_knob(i, w._min + (w._dial.value() / 1000.0) * (w._max - w._min))

            self._engine.start()
            self._play_btn.setEnabled(False)
            self._stop_btn.setEnabled(True)
            self._status.showMessage("Playing...")
        except Exception as e:
            logger.error(f"Failed to start audio: {e}")
            self._status.showMessage(f"Audio error: {e}")

    def _stop_audio(self):
        if self._engine is not None:
            self._engine.stop()
            self._engine = None
        self._play_btn.setEnabled(True)
        self._stop_btn.setEnabled(False)
        self._status.showMessage("Stopped")

    def _show_audio_devices(self):
        """Show available audio devices."""
        import sounddevice as sd
        try:
            devices = sd.query_devices()
            info_lines = ["Available Audio Devices:", ""]
            for i, dev in enumerate(devices):
                kind = "in" if dev["max_input_channels"] > 0 else ""
                kind += "out" if dev["max_output_channels"] > 0 else ""
                info_lines.append(f"  [{i}] {dev['name']} ({kind})")
            from PyQt6.QtWidgets import QMessageBox
            QMessageBox.information(self, "Audio Devices", "\n".join(info_lines))
        except Exception as e:
            self._status.showMessage(f"Error querying devices: {e}")

    # ------------------------------------------------------------------
    # Cleanup
    # ------------------------------------------------------------------

    def closeEvent(self, event):
        """Clean up on window close."""
        self._stop_audio()
        event.accept()
