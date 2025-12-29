#! /usr/bin/env python3

import sys
import csv
from PyQt5.QtWidgets import (
    QApplication, QWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QPushButton, QHBoxLayout, QAbstractItemView, QFileDialog, QLineEdit,
    QHeaderView
)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QClipboard

class DescriptionTable(QWidget):
    def __init__(self, csv_path):
        super().__init__()
        self.setWindowTitle("Postcard Descriptions")
        self.resize(900, 600)
        layout = QVBoxLayout(self)
        self.table = QTableWidget(self)
        layout.addWidget(self.table)
        self.load_csv(csv_path)

    def load_csv(self, csv_path):
        # Read CSV
        with open(csv_path, newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Image", "Title", ""])
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
            btn_title.clicked.connect(lambda _, text=row["title"]: self.copy_to_clipboard(text))
            self.table.setCellWidget(row_idx, 2, btn_title)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        self.table.resizeRowsToContents()

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
