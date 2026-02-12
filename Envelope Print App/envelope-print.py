#!/usr/bin/env python3
"""
PyQt5 UI to edit envelope addresses and print envelope PDFs.
"""

from __future__ import annotations

import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from PyQt5.QtWidgets import (
    QApplication,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QCheckBox,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)

from make_envelope_pdf import make_envelope_pdf


DEFAULT_RETURN_LINES = [
    "Geoff Dutton",
    "NedDog's Stamps & Cards",
    "2644 Ridge Rd",
    "Nederland, CO 80466",
]

DEFAULT_TO_LINES = [
    "Andrew Attard",
    "13 Lady Smith Drive",
    "ABN#64652016681 Code:PAID",
    "Edmondson Park NSW 2174",
    "Australia",
]

PRINTER_NAME = "_192_168_86_174"
MEDIA_NAME_6X9 = "Custom.6x9in"
MEDIA_NAME_5X7 = "Custom.5x7in"
ENVELOPE_SLOW_PRESET_NAME = "Envelopes SLOW"
ENVELOPE_SLOW_OPTIONS = (
    "MediaType=MidWeight96110",
    "HPPrintQuality=ProRes1200",
)


@dataclass(frozen=True)
class EnvelopeSpec:
    label: str
    page_w_in: float
    page_h_in: float
    content_rotation: str
    media: str


@dataclass(frozen=True)
class PrintSettings:
    printer_name: str
    output_filename: str
    options: tuple[str, ...] = ()


class EnvelopePrinter:
    def __init__(self, settings: PrintSettings, specs: dict[str, EnvelopeSpec]) -> None:
        self._settings = settings
        self._specs = specs

    def available_labels(self) -> list[str]:
        return list(self._specs.keys())

    def settings(self) -> PrintSettings:
        return self._settings

    def get_spec(self, label: str) -> EnvelopeSpec:
        return self._specs.get(label, next(iter(self._specs.values())))

    def output_path(self) -> Path:
        return Path(__file__).resolve().parent / self._settings.output_filename

    def render_pdf(self, spec: EnvelopeSpec, return_lines: list[str], to_lines: list[str]) -> None:
        make_envelope_pdf(
            self.output_path(),
            return_lines,
            to_lines,
            page_w_in=spec.page_w_in,
            page_h_in=spec.page_h_in,
            content_rotation=spec.content_rotation,
        )

    def print_pdf(self, spec: EnvelopeSpec) -> subprocess.CompletedProcess[str]:
        cmd = [
            "lp",
            "-d",
            self._settings.printer_name,
            "-o",
            f"media={spec.media}",
        ]
        for option in self._settings.options:
            cmd.extend(["-o", option])
        cmd.append(str(self.output_path()))
        return subprocess.run(cmd, capture_output=True, text=True)


ENVELOPE_SPECS = {
    "5x7": EnvelopeSpec(
        label="5x7",
        page_w_in=5.0,
        page_h_in=7.0,
        content_rotation="ccw",
        media=MEDIA_NAME_5X7,
    ),
    "6x9": EnvelopeSpec(
        label="6x9",
        page_w_in=6.0,
        page_h_in=9.0,
        content_rotation="ccw",
        media=MEDIA_NAME_6X9,
    ),
}

PRINT_SETTINGS = PrintSettings(
    printer_name=PRINTER_NAME,
    output_filename="envelope_print.pdf",
    options=ENVELOPE_SLOW_OPTIONS,
)


def _clean_lines(text: str) -> list[str]:
    return [line.strip() for line in text.splitlines() if line.strip()]


class EnvelopeWindow(QWidget):
    def __init__(self, printer: EnvelopePrinter) -> None:
        super().__init__()
        self.setWindowTitle("Envelope Print")
        self.printer = printer

        self.size_combo = QComboBox()
        self.size_combo.addItems(self.printer.available_labels())
        self.size_combo.currentTextChanged.connect(self.update_info_label)
        self.remove_pdf_toggle = QCheckBox("Remove PDF after print")
        self.remove_pdf_toggle.setChecked(True)

        self.return_edit = QPlainTextEdit()
        self.return_edit.setPlainText("\n".join(DEFAULT_RETURN_LINES))

        self.to_edit = QPlainTextEdit()
        self.to_edit.setPlainText("\n".join(DEFAULT_TO_LINES))

        return_box = QGroupBox("Return address")
        return_layout = QVBoxLayout()
        return_layout.addWidget(self.return_edit)
        return_box.setLayout(return_layout)

        to_box = QGroupBox("Send to")
        to_layout = QVBoxLayout()
        to_layout.addWidget(self.to_edit)
        to_box.setLayout(to_layout)

        self.print_button = QPushButton("Print")
        self.print_button.clicked.connect(self.handle_print)

        size_box = QGroupBox("Envelope size")
        size_layout = QHBoxLayout()
        size_layout.addWidget(self.size_combo)
        size_layout.addWidget(self.remove_pdf_toggle)
        size_box.setLayout(size_layout)

        self.info_label = QLabel()
        self.update_info_label()

        layout = QVBoxLayout()
        layout.addWidget(size_box)
        layout.addWidget(return_box)
        layout.addWidget(to_box)
        layout.addWidget(self.print_button)
        layout.addWidget(self.info_label)
        self.setLayout(layout)
        self.resize(540, 520)

    def _size_spec(self) -> EnvelopeSpec:
        size = self.size_combo.currentText()
        return self.printer.get_spec(size)

    def update_info_label(self) -> None:
        spec = self._size_spec()
        size = spec.label
        settings = self.printer.settings()
        self.info_label.setText(
            f"Size: {size} | Printer: {settings.printer_name} | Media: {spec.media} | Output: {settings.output_filename}"
        )

    def handle_print(self) -> None:
        return_lines = _clean_lines(self.return_edit.toPlainText())
        to_lines = _clean_lines(self.to_edit.toPlainText())

        if not return_lines:
            QMessageBox.warning(self, "Missing return address", "Enter a return address.")
            return
        if not to_lines:
            QMessageBox.warning(self, "Missing destination address", "Enter a destination address.")
            return

        spec = self._size_spec()

        try:
            self.printer.render_pdf(spec, return_lines, to_lines)
        except Exception as exc:
            QMessageBox.critical(self, "PDF error", f"Failed to write PDF:\n{exc}")
            return

        result = self.printer.print_pdf(spec)
        if result.returncode != 0:
            details = result.stderr.strip() or result.stdout.strip() or "Unknown error"
            QMessageBox.critical(self, "Print error", f"lp failed:\n{details}")
            return

        if self.remove_pdf_toggle.isChecked():
            try:
                self.printer.output_path().unlink()
            except FileNotFoundError:
                pass
            except OSError as exc:
                QMessageBox.warning(
                    self,
                    "Cleanup warning",
                    f"Printed, but failed to remove PDF:\n{exc}",
                )
                return

        QMessageBox.information(self, "Printed", "Envelope sent to printer.")


def main() -> None:
    app = QApplication(sys.argv)
    printer = EnvelopePrinter(PRINT_SETTINGS, ENVELOPE_SPECS)
    window = EnvelopeWindow(printer)
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
