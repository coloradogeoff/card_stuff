#!/usr/bin/env python3
import os
import sys
import webbrowser
from pathlib import Path

import requests
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QApplication,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

APP_STYLE = """
QWidget {
    background-color: #f0f2f5;
    font-family: -apple-system, "Helvetica Neue", Arial, sans-serif;
    font-size: 13px;
    color: #1a1a2e;
}
QGroupBox {
    background-color: #ffffff;
    border: 1px solid #dde1e7;
    border-radius: 10px;
    margin-top: 14px;
    padding: 10px 12px 12px 12px;
    font-weight: 600;
    font-size: 12px;
    color: #6b7280;
    text-transform: uppercase;
    letter-spacing: 0.5px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 4px;
}
QLineEdit {
    background-color: #fafbfc;
    border: 1px solid #dde1e7;
    border-radius: 7px;
    padding: 6px 10px;
    min-height: 28px;
    font-size: 13px;
    color: #1a1a2e;
}
QLineEdit:focus {
    border-color: #4f8ef7;
    background-color: #ffffff;
}
QLineEdit[readOnly="true"] {
    background-color: #f5f5f5;
    color: #6b7280;
}
QTextEdit {
    background-color: #fafbfc;
    border: 1px solid #dde1e7;
    border-radius: 7px;
    padding: 8px 10px;
    font-size: 13px;
    color: #1a1a2e;
}
QTextEdit:focus {
    border-color: #4f8ef7;
    background-color: #ffffff;
}
QPushButton {
    background-color: #ffffff;
    color: #374151;
    border: 1px solid #dde1e7;
    border-radius: 7px;
    padding: 6px 14px;
    font-size: 13px;
    min-height: 30px;
}
QPushButton:hover {
    background-color: #f3f4f6;
    border-color: #c0c8d4;
}
QPushButton:pressed {
    background-color: #e9ebee;
}
QPushButton#primaryBtn {
    background-color: #4f8ef7;
    color: #ffffff;
    border: none;
    border-radius: 9px;
    font-size: 14px;
    font-weight: 700;
    min-height: 44px;
    letter-spacing: 0.3px;
}
QPushButton#primaryBtn:hover {
    background-color: #3a7de8;
}
QPushButton#primaryBtn:pressed {
    background-color: #2d6dd4;
}
QFrame#divider {
    color: #e5e7eb;
}
QLabel#infoLabel {
    color: #9ca3af;
    font-size: 11px;
}
"""


class TrackingApp(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Letter Track")

        # ── Quick-link buttons ──────────────────────────────────────────
        links_box = QGroupBox("Quick Links")
        links_layout = QHBoxLayout()
        links_layout.setContentsMargins(8, 8, 8, 8)
        links_layout.setSpacing(8)

        awaiting_btn = QPushButton("Awaiting Shipping")
        awaiting_btn.setCursor(Qt.PointingHandCursor)
        awaiting_btn.clicked.connect(self.open_awaiting_shipping)

        lettertrackpro_btn = QPushButton("LetterTrackPro")
        lettertrackpro_btn.setCursor(Qt.PointingHandCursor)
        lettertrackpro_btn.clicked.connect(self.open_lettertrackpro)

        links_layout.addWidget(awaiting_btn)
        links_layout.addWidget(lettertrackpro_btn)
        links_box.setLayout(links_layout)

        # ── Tracking number input ───────────────────────────────────────
        tracking_box = QGroupBox("Tracking Number")
        tracking_layout = QHBoxLayout()
        tracking_layout.setContentsMargins(8, 8, 8, 8)
        tracking_layout.setSpacing(8)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Enter USPS tracking number…")

        clear_btn = QPushButton("Clear")
        clear_btn.setCursor(Qt.PointingHandCursor)
        clear_btn.clicked.connect(self.clear_fields)
        clear_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        tracking_layout.addWidget(self.input_field)
        tracking_layout.addWidget(clear_btn)
        tracking_box.setLayout(tracking_layout)

        # ── TinyURL ─────────────────────────────────────────────────────
        tinyurl_box = QGroupBox("Short Link")
        tinyurl_layout = QHBoxLayout()
        tinyurl_layout.setContentsMargins(8, 8, 8, 8)
        tinyurl_layout.setSpacing(8)

        tinyurl_btn = QPushButton("Generate TinyURL")
        tinyurl_btn.setCursor(Qt.PointingHandCursor)
        tinyurl_btn.clicked.connect(self.generate_tinyurl)
        tinyurl_btn.setSizePolicy(QSizePolicy.Fixed, QSizePolicy.Fixed)

        self.tinyurl_field = QLineEdit()
        self.tinyurl_field.setReadOnly(True)
        self.tinyurl_field.setPlaceholderText("Shortened URL will appear here")

        tinyurl_layout.addWidget(tinyurl_btn)
        tinyurl_layout.addWidget(self.tinyurl_field)
        tinyurl_box.setLayout(tinyurl_layout)

        # ── Message ─────────────────────────────────────────────────────
        message_box = QGroupBox("Message")
        message_layout = QVBoxLayout()
        message_layout.setContentsMargins(8, 8, 8, 8)
        message_layout.setSpacing(8)

        self.message_edit = QTextEdit()
        self.message_edit.setMinimumHeight(120)

        gen_btn = QPushButton("Generate Message")
        gen_btn.setObjectName("primaryBtn")
        gen_btn.setCursor(Qt.PointingHandCursor)
        gen_btn.clicked.connect(self.generate_message)

        copy_btn = QPushButton("Copy to Clipboard")
        copy_btn.setCursor(Qt.PointingHandCursor)
        copy_btn.clicked.connect(self.copy_to_clipboard)

        message_layout.addWidget(self.message_edit)
        message_layout.addWidget(gen_btn)
        message_layout.addWidget(copy_btn)
        message_box.setLayout(message_layout)

        # ── Root layout ──────────────────────────────────────────────────
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(10)
        layout.addWidget(links_box)
        layout.addWidget(tracking_box)
        layout.addWidget(tinyurl_box)
        layout.addWidget(message_box)
        self.setLayout(layout)
        self.setFixedWidth(560)
        self.adjustSize()

    # ── Helpers ──────────────────────────────────────────────────────────

    def _tracking_url(self, tracking_number: str) -> str:
        return f"https://www.lettertrackpro.com/uspstracking/?TrackingNumber={tracking_number}"

    def get_tinyurl_token(self) -> str:
        token = os.getenv("TINYURL_TOKEN", "").strip()
        if token:
            return token
        token_path = Path(__file__).with_name(".tinyurl-token.txt")
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()
            if token:
                return token
        raise RuntimeError(
            "Missing TinyURL token. Set TINYURL_TOKEN or create .tinyurl-token.txt "
            "next to lettertrack.py."
        )

    def shorten_tinyurl(self, long_url: str) -> str:
        resp = requests.post(
            "https://api.tinyurl.com/create",
            json={"url": long_url},
            headers={
                "Authorization": f"Bearer {self.get_tinyurl_token()}",
                "Content-Type": "application/json",
            },
        )
        resp.raise_for_status()
        return resp.json()["data"]["tiny_url"]

    # ── Slots ─────────────────────────────────────────────────────────────

    def open_awaiting_shipping(self):
        webbrowser.open("https://www.ebay.com/sh/ord/?filter=status:AWAITING_SHIPMENT")

    def open_lettertrackpro(self):
        webbrowser.open("https://www.lettertrackpro.com/Process_Mail.asp")

    def generate_tinyurl(self):
        tracking_number = self.input_field.text().strip()
        if not tracking_number:
            self.tinyurl_field.clear()
            return
        try:
            tiny_url = self.shorten_tinyurl(self._tracking_url(tracking_number))
            self.tinyurl_field.setText(tiny_url)
        except Exception as e:
            self.tinyurl_field.setText(f"Error: {e}")

    def generate_message(self):
        tracking_number = self.input_field.text().strip()
        if not tracking_number:
            self.message_edit.setPlainText("Please enter a tracking number.")
            return
        try:
            tiny_url = self.shorten_tinyurl(self._tracking_url(tracking_number))
            self.tinyurl_field.setText(tiny_url)
        except Exception as e:
            self.message_edit.setPlainText(f"Error generating TinyURL: {e}")
            return
        message = (
            f"Hello, and thank you for your business. I'm using a mail-tracking service "
            f"called LetterTrackPro. You can track your order with this link: {tiny_url}\n\n"
            f"If you have any questions, please don't hesitate to reach out.\n"
            f"Geoff\n"
        )
        self.message_edit.setPlainText(message)
        QApplication.clipboard().setText(message)

    def clear_fields(self):
        self.input_field.clear()
        self.tinyurl_field.clear()
        self.message_edit.clear()

    def copy_to_clipboard(self):
        QApplication.clipboard().setText(self.message_edit.toPlainText())


if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)
    window = TrackingApp()
    window.show()
    sys.exit(app.exec_())
