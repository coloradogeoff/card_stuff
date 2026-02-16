#! /usr/bin/env python3

import sys
import csv
import webbrowser
from datetime import date
from urllib.parse import quote_plus
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QPushButton, QHBoxLayout, QAbstractItemView, QFileDialog, QLineEdit,
    QHeaderView, QLabel
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor

HIGHLIGHT_COLOR = "#d9f7d9"

class DescriptionTable(QWidget):
    def __init__(self, csv_path):
        super().__init__()
        self.last_copied_row = None
        self.highlighted_row = None
        self.setWindowTitle("Ebay Title Descriptions")
        self.resize(900, 600)
        layout = QVBoxLayout(self)

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Potential SKU"))
        self.sku_input = QLineEdit(date.today().strftime("%Y%m%d"))
        self.sku_input.setToolTip("Potential SKU value")
        controls.addWidget(self.sku_input)
        self.copy_sku_btn = QPushButton("Copy")
        self.copy_sku_btn.setToolTip("Copy Potential SKU")
        self.copy_sku_btn.clicked.connect(self.copy_potential_sku)
        controls.addWidget(self.copy_sku_btn)
        controls.addStretch()
        self.copy_next_btn = QPushButton("Copy Next")
        self.copy_next_btn.setToolTip("Copy the next title")
        self.copy_next_btn.clicked.connect(self.copy_next_title)
        controls.addWidget(self.copy_next_btn)
        layout.addLayout(controls)

        self.table = QTableWidget(self)
        layout.addWidget(self.table)
        self.load_csv(csv_path)

    def load_csv(self, csv_path):
        # Read CSV
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Image", "Title", "", ""])
        self.table.setRowCount(len(rows))
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        for row_idx, row in enumerate(rows):
            # Image cell
            img_item = QTableWidgetItem(row["front"])
            self.table.setItem(row_idx, 0, img_item)
            # Title cell
            title_edit = QLineEdit(row["title"])
            title_edit.setReadOnly(True)
            title_edit.setToolTip("Select text and copy with Ctrl+C / Cmd+C")
            self.table.setCellWidget(row_idx, 1, title_edit)
            # Title Copy button
            btn_title = QPushButton("Copy")
            btn_title.setToolTip("Copy Title")
            btn_title.clicked.connect(lambda _, r=row_idx: self.copy_row_title(r))
            self.table.setCellWidget(row_idx, 2, btn_title)
            # eBay Search button
            btn_ebay = QPushButton("eBay")
            btn_ebay.setToolTip("Search title on eBay")
            btn_ebay.clicked.connect(lambda _, r=row_idx: self.search_row_ebay(r))
            self.table.setCellWidget(row_idx, 3, btn_ebay)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        self.table.resizeRowsToContents()
        self.update_copy_next_state()

    def copy_row_title(self, row):
        if row < 0 or row >= self.table.rowCount():
            return
        title_widget = self.table.cellWidget(row, 1)
        if not isinstance(title_widget, QLineEdit):
            return
        self.copy_to_clipboard(title_widget.text())
        self.last_copied_row = row
        self.highlight_row(row)
        self.update_copy_next_state()

    def copy_next_title(self):
        total_rows = self.table.rowCount()
        if total_rows == 0:
            return
        if self.last_copied_row is None:
            next_row = 0
        else:
            next_row = self.last_copied_row + 1
        if next_row >= total_rows:
            return
        self.copy_row_title(next_row)

    def search_row_ebay(self, row):
        if row < 0 or row >= self.table.rowCount():
            return
        title_widget = self.table.cellWidget(row, 1)
        if not isinstance(title_widget, QLineEdit):
            return
        title = title_widget.text().strip()
        if not title:
            return
        url = f"https://www.ebay.com/sch/i.html?_nkw={quote_plus(title)}"
        webbrowser.open(url, new=2)

    def highlight_row(self, row):
        if self.highlighted_row is not None and self.highlighted_row != row:
            self.set_row_highlight(self.highlighted_row, highlighted=False)
        self.set_row_highlight(row, highlighted=True)
        self.highlighted_row = row

    def set_row_highlight(self, row, highlighted):
        color = HIGHLIGHT_COLOR if highlighted else ""
        img_item = self.table.item(row, 0)
        if img_item is not None:
            img_item.setData(Qt.BackgroundRole, QColor(HIGHLIGHT_COLOR) if highlighted else None)
        title_widget = self.table.cellWidget(row, 1)
        if title_widget is not None:
            title_widget.setStyleSheet(f"background-color: {color};")
        copy_btn = self.table.cellWidget(row, 2)
        if copy_btn is not None:
            copy_btn.setStyleSheet(f"background-color: {color};")
        ebay_btn = self.table.cellWidget(row, 3)
        if ebay_btn is not None:
            ebay_btn.setStyleSheet(f"background-color: {color};")

    def update_copy_next_state(self):
        total_rows = self.table.rowCount()
        enabled = total_rows > 0 and (
            self.last_copied_row is None or self.last_copied_row < total_rows - 1
        )
        self.copy_next_btn.setEnabled(enabled)

    def copy_potential_sku(self):
        self.copy_to_clipboard(self.sku_input.text())

    def copy_to_clipboard(self, text):
        clipboard = QApplication.clipboard()
        clipboard.setText(text)

def main(csv_path=None):
    import os
    app = QApplication(sys.argv)
    if not csv_path:
        # Default to description.csv in cwd if present
        default_csv = os.path.join(os.getcwd(), "description.csv")
        if not os.path.exists(default_csv):
            # Ask user to select a CSV file
            fname, _ = QFileDialog.getOpenFileName(None, "Open description.csv", "", "CSV Files (*.csv)")
            if not fname:
                sys.exit(0)
            csv_path = fname
        else:
            csv_path = default_csv
    win = DescriptionTable(csv_path)
    win.show()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
