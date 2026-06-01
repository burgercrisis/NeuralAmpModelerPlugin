"""Entry point for the NAM Multi-Knob Player."""

import sys
import traceback

from PyQt6.QtWidgets import QApplication

# Import after QApplication exists (some Qt backends need it)
from .ui.MainWindow import MainWindow


def main():
    """Run the NAM Multi-Knob Player application."""
    # Enable high-DPI support
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(sys.argv)
    app.setApplicationName("NAM Multi-Knob Player")
    app.setApplicationVersion("0.1.0")

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


# Import Qt after function definition
from PyQt6.QtCore import Qt  # noqa: E402


if __name__ == "__main__":
    main()
