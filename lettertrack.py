import sys
import webbrowser
import os
from pathlib import Path
from PyQt5.QtWidgets import (
    QApplication,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTextEdit
)
import requests


class TrackingApp(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Tracking Message Generator")
        self.setGeometry(100, 100, 600, 400)

        # Main Layout
        main_layout = QVBoxLayout()

        # Top Buttons Layout (Awaiting Shipping & LetterTrackPro)
        button_layout = QHBoxLayout()

        # Awaiting Shipping Button
        self.awaiting_shipping_button = QPushButton("Awaiting Shipping")
        self.awaiting_shipping_button.clicked.connect(self.open_awaiting_shipping)
        button_layout.addWidget(self.awaiting_shipping_button)

        # LetterTrackPro Button
        self.lettertrackpro_button = QPushButton("LetterTrackPro")
        self.lettertrackpro_button.clicked.connect(self.open_lettertrackpro)
        button_layout.addWidget(self.lettertrackpro_button)

        # Add top buttons to main layout
        main_layout.addLayout(button_layout)

        # Tracking Input Layout (Label Above, Input Below, Clear Button Next to Input)
        main_layout.addWidget(QLabel("Enter Tracking Number:"))

        tracking_layout = QHBoxLayout()

        # Input Field
        self.input_field = QLineEdit()
        tracking_layout.addWidget(self.input_field)

        # Clear Button (to clear tracking number + message box)
        self.clear_button = QPushButton("Clear")
        self.clear_button.clicked.connect(self.clear_fields)
        tracking_layout.addWidget(self.clear_button)

        # Add tracking – number layout to main layout
        main_layout.addLayout(tracking_layout)

        # ───────────────────────────────────────────────────────────
        # NEW: TinyUrl Button + Text Field Layout
        #
        # Between the tracking input and Generate Message button, we add:
        #  • a QPushButton("TinyUrl") that calls self.generate_tinyurl()
        #  • a QLineEdit (read-only) to display the shortened URL
        #
        tinyurl_layout = QHBoxLayout()

        # TinyUrl Button
        self.tinyurl_button = QPushButton("Generate TinyUrl")
        self.tinyurl_button.clicked.connect(self.generate_tinyurl)
        tinyurl_layout.addWidget(self.tinyurl_button)

        # Read-only field to hold the shortened URL
        self.tinyurl_field = QLineEdit()
        self.tinyurl_field.setReadOnly(True)
        tinyurl_layout.addWidget(self.tinyurl_field)

        # Add the TinyUrl layout to main layout
        main_layout.addLayout(tinyurl_layout)
        # ───────────────────────────────────────────────────────────

        # Generate Button (original)
        self.generate_button = QPushButton("Generate Message")
        self.generate_button.clicked.connect(self.generate_message)
        main_layout.addWidget(self.generate_button)

        # Message Display (Editable)
        self.message_box = QTextEdit()
        main_layout.addWidget(self.message_box)

        # Copy to Clipboard Button (original)
        self.copy_button = QPushButton("Copy to Clipboard")
        self.copy_button.clicked.connect(self.copy_to_clipboard)
        main_layout.addWidget(self.copy_button)

        # Set the main layout on this QWidget
        self.setLayout(main_layout)

    def shorten_tinyurl(self, long_url: str) -> str:
        """
        Shorten a long URL via the TinyURL API using a Bearer token.

        :param long_url: The full URL to shorten (e.g.
                         "https://www.example.com/very/long/path")
        :return: The shortened TinyURL (e.g. "https://tinyurl.com/abc123")
        """
        endpoint = "https://api.tinyurl.com/create"
        tinyurl_token = self.get_tinyurl_token()

        headers = {
            "Authorization": f"Bearer {tinyurl_token}",
            "Content-Type": "application/json"
        }

        payload = {
            "url": long_url
        }

        resp = requests.post(endpoint, json=payload, headers=headers)
        resp.raise_for_status()

        data = resp.json()
        # The API returns something like:
        # {"data": {"tiny_url": "https://tinyurl.com/abc123", …}, "code": 200, …}
        short_url = data["data"]["tiny_url"]
        return short_url

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

    def generate_tinyurl(self):
        """
        Called when user clicks the “TinyUrl” button. Reads the tracking number,
        builds lt_url = "https://www.lettertrackpro.com/uspstracking/?TrackingNumber=…",
        then calls shorten_tinyurl(...) and puts the result into tinyurl_field.
        """
        tracking_number = self.input_field.text().strip()
        if tracking_number:
            lt_url = f"https://www.lettertrackpro.com/uspstracking/?TrackingNumber={tracking_number}"
            try:
                tiny_url = self.shorten_tinyurl(lt_url)
                self.tinyurl_field.setText(tiny_url)
            except Exception as e:
                # If there’s an error contacting TinyURL, show a brief message
                self.tinyurl_field.setText(f"Error: {e}")
        else:
            self.tinyurl_field.clear()

    def open_awaiting_shipping(self):
        """Opens the eBay 'Awaiting Shipping' page in the default browser."""
        webbrowser.open("https://www.ebay.com/sh/ord/?filter=status:AWAITING_SHIPMENT")

    def open_lettertrackpro(self):
        """Opens the LetterTrackPro 'Process Mail' page in the default browser."""
        webbrowser.open("https://www.lettertrackpro.com/Process_Mail.asp")

    def generate_message(self):
        """Generates the message based on the input tracking number."""
        tracking_number = self.input_field.text().strip()
        lt_url = f"https://www.lettertrackpro.com/uspstracking/?TrackingNumber={tracking_number}"
        tiny_url = self.shorten_tinyurl(lt_url)

        if tracking_number:
            message = f"""Hello, and thank you for your business. I’m using a mail-tracking service called LetterTrackPro. You can track your order with this link: {tiny_url}

If you have any questions, please don’t hesitate to reach out.
Geoff
"""
            self.message_box.setPlainText(message)
        else:
            self.message_box.setPlainText("Please enter a tracking number.")

    def clear_fields(self):
        """Clears both the tracking number input and message box."""
        self.input_field.clear()
        self.message_box.clear()
        self.tinyurl_field.clear()

    def copy_to_clipboard(self):
        """Copies the generated message to the clipboard."""
        clipboard = QApplication.clipboard()
        clipboard.setText(self.message_box.toPlainText())


# Run the Application
if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TrackingApp()
    window.show()
    sys.exit(app.exec_())  # PyQt5 uses `exec_()` instead of `exec()`
