#!/usr/bin/env python3

import base64
import csv
import datetime as dt
import os
import re
import subprocess
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from pathlib import Path
from threading import Lock

import pytesseract
import yaml
from PyQt5.QtCore import QObject, QThread, Qt, pyqtSignal
from PyQt5.QtGui import QCursor, QGuiApplication, QPixmap
from PyQt5.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDialog,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMessageBox,
    QPushButton,
    QProgressBar,
    QPlainTextEdit,
    QVBoxLayout,
    QWidget,
)
from PIL import Image


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
PREVIEW_HEIGHT = 800
_rate_lock = Lock()
_last_call_time = 0.0
_MIN_INTERVAL = 0.5
_client = None
_openai_api_key = None


def _ensure_typing_extensions_override():
    try:
        import typing_extensions
    except Exception:
        return
    if not hasattr(typing_extensions, "override"):
        def override(func):
            return func
        typing_extensions.override = override


_ensure_typing_extensions_override()
from openai import OpenAI


def load_instructions(category: str) -> str:
    rules_path = Path(__file__).parent / "instructions.yaml"
    with open(rules_path, "r", encoding="utf-8") as file_handle:
        data = yaml.safe_load(file_handle) or {}
    return data.get(category, "")


def instructions_file_path() -> Path:
    return Path(__file__).parent / "instructions.yaml"


def load_openai_api_key() -> str:
    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    candidates = [
        Path(__file__).with_name(".openai-api-key.txt"),
        Path(__file__).resolve().parent.parent / ".openai-api-key.txt",
    ]
    for key_path in candidates:
        if key_path.exists():
            key = key_path.read_text(encoding="utf-8").strip()
            if key:
                return key

    raise RuntimeError(
        "Missing OPENAI_API_KEY. Set the environment variable or create "
        ".openai-api-key.txt next to ebay_title_gui.py (or in its parent folder)."
    )


def resize_image(input_path: Path, max_size: int = 1024) -> Image.Image:
    image = Image.open(input_path)
    if image.width > max_size:
        image.thumbnail((max_size, max_size), Image.LANCZOS)
    return image


def compress_image(image: Image.Image, quality: int = 85) -> bytes:
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="JPEG", quality=quality, optimize=True)
    return buffer.getvalue()


def validate_image(path: Path) -> bool:
    try:
        with Image.open(path) as image:
            image.verify()
        return True
    except Exception:
        return False


def extract_ocr_text(image_path: Path) -> str:
    try:
        image = Image.open(image_path)
        text = pytesseract.image_to_string(image)
        return text.strip()
    except Exception:
        return ""


def image_to_base64(image_path: Path, compress: bool = True) -> str:
    if compress:
        resized = resize_image(image_path)
        jpeg_bytes = compress_image(resized)
        return base64.b64encode(jpeg_bytes).decode("utf-8")
    with open(image_path, "rb") as file_handle:
        return base64.b64encode(file_handle.read()).decode("utf-8")


def build_messages(
    ocr_text: str,
    images: list,
    category: str = "postcards",
    set_override: str = None,
    variety_override: str = None,
) -> list:
    system = load_instructions(category)
    if ocr_text:
        system += f"Text found on the front of the item: {ocr_text}\n"
    if category == "sports_cards" and set_override:
        system += (
            "Card set override: Use exactly the following card set string in the title "
            f"(season years + manufacturer + set name): {set_override}\n"
        )
    if category == "sports_cards" and variety_override:
        system += (
            "Variety override: Use exactly the following value for the "
            "[Variety/Parallel/Color] section in the title: "
            f"{variety_override}\n"
        )

    messages = [{"role": "system", "content": system}]
    for image in images:
        messages.append(
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": image["desc"]},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{image['data']}"},
                    },
                ],
            }
        )
    return messages


def chat_with_openai(messages):
    global _last_call_time
    global _client
    global _openai_api_key

    if _client is None:
        if _openai_api_key is None:
            _openai_api_key = load_openai_api_key()
        _client = OpenAI(api_key=_openai_api_key)

    with _rate_lock:
        now = time.time()
        wait = _MIN_INTERVAL - (now - _last_call_time)
        if wait > 0:
            time.sleep(wait)
        _last_call_time = time.time()

    return _client.chat.completions.create(
        model="gpt-5.2",
        messages=messages,
        max_completion_tokens=500,
    )


def process_pair(
    front: Path,
    back: Path,
    compress: bool,
    category: str = "postcards",
    set_override: str = None,
    variety_override: str = None,
):
    if not (validate_image(front) and validate_image(back)):
        return None

    ocr_text = extract_ocr_text(front)
    images = [
        {"desc": "This is the front of the item.", "data": image_to_base64(front, compress)},
        {"desc": "This is the back of the item.", "data": image_to_base64(back, compress)},
    ]
    messages = build_messages(
        ocr_text,
        images,
        category=category,
        set_override=set_override,
        variety_override=variety_override,
    )
    response = chat_with_openai(messages)

    content = (response.choices[0].message.content or "").strip()
    lines = [line.strip() for line in content.splitlines() if line.strip()]
    title = ""
    title_pattern = re.compile(r"^\s*(?:\d+\.\s*)?\**\s*title\s*:\s*(.+)$", re.IGNORECASE)
    for line in lines:
        title_match = title_pattern.match(line)
        if title_match:
            title = title_match.group(1).strip().replace("*", "")
            break
    if not title and lines:
        title = lines[0].replace("*", "")

    return {"front": str(front.name), "title": title}


def default_sales_directory() -> Path:
    root = Path(os.environ.get("SALES_ROOT") or str(Path.home() / "Sales")).expanduser()
    if root.is_symlink() and not root.exists():
        link_target = os.readlink(root)
        raise RuntimeError(f"Sales root symlink is broken: {root} -> {link_target}")
    now = dt.date.today()
    target = root / f"{now.year:04d}" / f"{now.month:02d}"
    target.mkdir(parents=True, exist_ok=True)
    return target


def list_image_files(directory: Path) -> list:
    return sorted(
        [
            path
            for path in directory.iterdir()
            if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
        ]
    )


def build_image_pairs(image_files: list) -> list:
    pairs = [
        (image_files[index], image_files[index + 1])
        for index in range(0, len(image_files), 2)
    ]
    return sorted(
        pairs,
        key=lambda pair: max(pair[0].stat().st_mtime, pair[1].stat().st_mtime),
        reverse=True,
    )


def is_recent_pair(pair: tuple, max_age_seconds: int = 3600) -> bool:
    latest_mtime = max(pair[0].stat().st_mtime, pair[1].stat().st_mtime)
    return latest_mtime >= (dt.datetime.now().timestamp() - max_age_seconds)


def natural_sort_key(value: str):
    return [int(part) if part.isdigit() else part.lower() for part in re.split(r"(\d+)", value)]


def pair_filename_sort_key(pair: tuple):
    front, back = pair
    return (natural_sort_key(front.name), natural_sort_key(back.name))


class ClickableLabel(QLabel):
    clicked = pyqtSignal()

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.clicked.emit()
        super().mousePressEvent(event)


class InstructionsEditorDialog(QDialog):
    def __init__(self, path: Path, parent=None):
        super().__init__(parent)
        self.path = path
        self.setWindowTitle("Edit instructions.yaml")
        self.resize(900, 700)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(f"Editing: {self.path}"))

        self.editor = QPlainTextEdit()
        layout.addWidget(self.editor, stretch=1)

        button_row = QHBoxLayout()
        self.reload_button = QPushButton("Reload")
        self.reload_button.clicked.connect(self.load_file)
        button_row.addWidget(self.reload_button)
        button_row.addStretch()

        self.save_button = QPushButton("Save")
        self.save_button.clicked.connect(self.save_file)
        button_row.addWidget(self.save_button)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        button_row.addWidget(self.close_button)
        layout.addLayout(button_row)

        self.load_file()

    def load_file(self):
        try:
            content = self.path.read_text(encoding="utf-8")
        except Exception as exc:
            QMessageBox.critical(self, "Load Failed", f"Could not read instructions file:\n{exc}")
            return
        self.editor.setPlainText(content)

    def save_file(self):
        text = self.editor.toPlainText()
        try:
            parsed = yaml.safe_load(text) or {}
            if not isinstance(parsed, dict):
                QMessageBox.warning(
                    self,
                    "Invalid YAML",
                    "instructions.yaml must contain a mapping of categories.",
                )
                return
        except Exception as exc:
            QMessageBox.warning(self, "Invalid YAML", f"YAML parse error:\n{exc}")
            return

        try:
            if text and not text.endswith("\n"):
                text += "\n"
            self.path.write_text(text, encoding="utf-8")
            QMessageBox.information(self, "Saved", "instructions.yaml updated.")
        except Exception as exc:
            QMessageBox.critical(self, "Save Failed", f"Could not write instructions file:\n{exc}")


class ProcessorWorker(QObject):
    progress = pyqtSignal(str)
    percent = pyqtSignal(int)
    finished = pyqtSignal(str, int)
    failed = pyqtSignal(str)

    def __init__(
        self,
        directory: Path,
        pairs: list,
        category: str,
        compress: bool,
        set_override: str,
        variety_override: str,
    ):
        super().__init__()
        self.directory = directory
        self.pairs = pairs
        self.category = category
        self.compress = compress
        self.set_override = set_override.strip() or None
        self.variety_override = variety_override.strip() or None

    def run(self):
        try:
            if not self.pairs:
                raise ValueError("No image pairs selected.")

            total_pairs = len(self.pairs)
            completed = 0
            results_by_index = {}
            self.progress.emit(f"Submitting {total_pairs} selected item(s) for processing...")

            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {
                    executor.submit(
                        process_pair,
                        front,
                        back,
                        self.compress,
                        self.category,
                        self.set_override,
                        self.variety_override,
                    ): (index, front, back)
                    for index, (front, back) in enumerate(self.pairs)
                }

                for future in as_completed(futures):
                    index, front, back = futures[future]
                    result = future.result()
                    completed += 1
                    self.progress.emit(
                        f"[{completed}/{total_pairs}] Completed {front.name} + {back.name}"
                    )
                    if result is not None:
                        results_by_index[index] = result
                        self.progress.emit(f"Title: {result['title']}")
                    self.percent.emit(int((completed / total_pairs) * 100))

            results = [results_by_index[idx] for idx in sorted(results_by_index)]

            csv_path = self.directory / "description.csv"
            with open(csv_path, "w", newline="", encoding="utf-8") as csvfile:
                writer = csv.DictWriter(
                    csvfile, fieldnames=["front", "title"], quoting=csv.QUOTE_ALL
                )
                writer.writeheader()
                for row in results:
                    writer.writerow(row)

            self.progress.emit(f"Generated description.csv with {len(results)} entries.")
            self.finished.emit(str(csv_path), len(results))
        except Exception as exc:
            details = traceback.format_exc()
            self.failed.emit(f"{exc}\n{details}")


class EbayTitleGui(QWidget):
    def __init__(self):
        super().__init__()
        self.worker = None
        self.thread = None
        self.current_front_path = None
        self.current_back_path = None
        self.current_showing_back = False
        self.image_pairs = []
        self.startup_message = ""

        try:
            self.default_directory = default_sales_directory()
        except Exception as exc:
            self.default_directory = Path.cwd()
            self.startup_message = (
                f"Could not prepare Sales directory ({exc}). Using {self.default_directory}."
            )
        else:
            self.startup_message = f"Using Sales directory: {self.default_directory}"

        self.setWindowTitle("eBay Title Generator")
        self.resize(1100, 700)
        screen = None
        if hasattr(QGuiApplication, "screenAt"):
            screen = QGuiApplication.screenAt(QCursor.pos())
        if screen is not None:
            self.move(screen.availableGeometry().topLeft())
        else:
            desktop = QApplication.desktop()
            screen_number = desktop.screenNumber(QCursor.pos())
            if screen_number < 0:
                screen_number = desktop.primaryScreen()
            self.move(desktop.availableGeometry(screen_number).topLeft())

        root = QVBoxLayout(self)

        top = QHBoxLayout()
        category_block = QVBoxLayout()
        category_block.addWidget(QLabel("Category"))
        self.category_combo = QComboBox()
        self.category_combo.addItems(["Sports Cards", "Postcards", "Postal History"])
        self.category_combo.currentIndexChanged.connect(self.on_category_changed)
        category_block.addWidget(self.category_combo)
        self.edit_instructions_button = QPushButton("Edit Instructions")
        self.edit_instructions_button.clicked.connect(self.open_instructions_editor)
        category_block.addWidget(self.edit_instructions_button)
        top.addLayout(category_block)

        top.addWidget(QLabel("Image Directory"))
        self.directory_edit = QLineEdit(str(self.default_directory))
        top.addWidget(self.directory_edit, stretch=1)

        self.browse_button = QPushButton("Browse")
        self.browse_button.clicked.connect(self.choose_directory)
        top.addWidget(self.browse_button)

        self.refresh_button = QPushButton("Refresh")
        self.refresh_button.clicked.connect(self.refresh_images)
        top.addWidget(self.refresh_button)
        root.addLayout(top)

        middle = QHBoxLayout()
        left = QVBoxLayout()
        self.preview_side_label = QLabel("Front (click image to toggle)")
        self.preview_side_label.setAlignment(Qt.AlignCenter)
        left.addWidget(self.preview_side_label)
        self.preview_label = ClickableLabel("Select an image pair")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumWidth(650)
        self.preview_label.setFixedHeight(PREVIEW_HEIGHT)
        self.preview_label.setStyleSheet(
            "border: 1px solid #cccccc; background: #f5f5f5;"
        )
        self.preview_label.clicked.connect(self.toggle_preview_side)
        left.addWidget(self.preview_label)
        middle.addLayout(left, stretch=2)

        self.image_list = QListWidget()
        self.image_list.currentItemChanged.connect(self.on_image_selected)
        self.image_list.itemChanged.connect(self.on_pair_check_changed)
        right = QVBoxLayout()
        right.addWidget(self.image_list, stretch=1)
        pair_select = QHBoxLayout()
        self.select_all_button = QPushButton("Select All")
        self.select_all_button.clicked.connect(self.select_all_pairs)
        pair_select.addWidget(self.select_all_button)
        self.select_none_button = QPushButton("Select None")
        self.select_none_button.clicked.connect(self.select_no_pairs)
        pair_select.addWidget(self.select_none_button)
        right.addLayout(pair_select)
        middle.addLayout(right, stretch=1)
        root.addLayout(middle, stretch=1)

        options = QHBoxLayout()
        self.compress_checkbox = QCheckBox("Compress images")
        self.compress_checkbox.setChecked(True)
        options.addWidget(self.compress_checkbox)

        self.viewer_checkbox = QCheckBox("Open viewer.py after completion")
        self.viewer_checkbox.setChecked(True)
        options.addWidget(self.viewer_checkbox)
        self.view_descriptions_button = QPushButton("View Descriptions")
        self.view_descriptions_button.clicked.connect(self.view_descriptions_now)
        self.view_descriptions_button.setVisible(False)
        options.addWidget(self.view_descriptions_button)
        options.addStretch()
        root.addLayout(options)

        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        root.addWidget(self.progress_bar)

        self.overrides_widget = QWidget()
        overrides = QHBoxLayout(self.overrides_widget)
        overrides.setContentsMargins(0, 0, 0, 0)
        overrides.addWidget(QLabel("Override set"))
        self.set_override_edit = QLineEdit("")
        self.set_override_edit.setPlaceholderText("e.g. 2024-25 Panini Select")
        overrides.addWidget(self.set_override_edit, stretch=1)
        overrides.addWidget(QLabel("Override variety"))
        self.variety_override_edit = QLineEdit("")
        self.variety_override_edit.setPlaceholderText("e.g. Silver Prizm")
        overrides.addWidget(self.variety_override_edit, stretch=1)
        root.addWidget(self.overrides_widget)

        self.submit_button = QPushButton("Submit")
        self.submit_button.clicked.connect(self.start_processing)
        root.addWidget(self.submit_button)

        self.feedback = QPlainTextEdit()
        self.feedback.setReadOnly(True)
        root.addWidget(self.feedback, stretch=1)

        self.update_override_visibility()
        self.append_log(self.startup_message)
        self.refresh_images()

    def category_key(self) -> str:
        label = self.category_combo.currentText()
        mapping = {
            "Sports Cards": "sports_cards",
            "Postcards": "postcards",
            "Postal History": "postal_history",
        }
        return mapping[label]

    def on_category_changed(self, _index):
        self.update_override_visibility()

    def update_override_visibility(self):
        show_overrides = self.category_key() == "sports_cards"
        self.overrides_widget.setVisible(show_overrides)

    def choose_directory(self):
        directory = QFileDialog.getExistingDirectory(
            self, "Select Image Directory", self.directory_edit.text()
        )
        if directory:
            self.directory_edit.setText(directory)
            self.refresh_images()

    def refresh_images(self):
        self.image_list.clear()
        self.image_pairs = []
        self.current_front_path = None
        self.current_back_path = None
        self.current_showing_back = False
        self.clear_previews()

        try:
            directory = Path(self.directory_edit.text()).expanduser().resolve()
            if not directory.exists() or not directory.is_dir():
                self.append_log("Selected directory is invalid.")
                self.submit_button.setEnabled(False)
                self.select_all_button.setEnabled(False)
                self.select_none_button.setEnabled(False)
                self.update_view_descriptions_button(None)
                return
            files = list_image_files(directory)
        except Exception as exc:
            self.append_log(f"Failed to read directory: {exc}")
            self.submit_button.setEnabled(False)
            self.select_all_button.setEnabled(False)
            self.select_none_button.setEnabled(False)
            self.update_view_descriptions_button(None)
            return

        if len(files) % 2 == 0 and files:
            self.image_pairs = build_image_pairs(files)
        for index, pair in enumerate(self.image_pairs):
            front, back = pair
            item = QListWidgetItem(f"{index + 1:03d}. {front.name} | {back.name}")
            item.setData(Qt.UserRole, str(front))
            item.setData(Qt.UserRole + 1, str(back))
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if is_recent_pair(pair) else Qt.Unchecked)
            self.image_list.addItem(item)

        if self.image_pairs:
            self.image_list.setCurrentRow(0)

        if not files:
            self.append_log("No images found in selected directory.")
        elif len(files) % 2 != 0:
            self.append_log(
                f"Found {len(files)} images. Need an even number for front/back pairs."
            )
        else:
            self.append_log(
                f"Loaded {len(files)} images ({len(self.image_pairs)} pairs) from {directory}."
            )

        self.select_all_button.setEnabled(bool(self.image_pairs))
        self.select_none_button.setEnabled(bool(self.image_pairs))
        self.update_submit_enabled()
        self.update_view_descriptions_button(directory)

    def on_image_selected(self, current, _previous):
        if current is None:
            self.current_front_path = None
            self.current_back_path = None
            self.current_showing_back = False
            self.clear_previews()
            return
        self.current_front_path = Path(current.data(Qt.UserRole))
        self.current_back_path = Path(current.data(Qt.UserRole + 1))
        self.current_showing_back = False
        self.update_preview_pixmap()

    def on_pair_check_changed(self, _item):
        self.update_submit_enabled()

    def clear_previews(self):
        self.preview_side_label.setText("Front (click image to toggle)")
        self.preview_label.setText("Select an image pair")
        self.preview_label.setPixmap(QPixmap())

    def set_preview_pixmap(self, image_path: Path, fallback: str):
        pixmap = QPixmap(str(image_path))
        if pixmap.isNull():
            self.preview_label.setText(fallback)
            self.preview_label.setPixmap(QPixmap())
            return
        scaled = pixmap.scaled(
            self.preview_label.width(),
            PREVIEW_HEIGHT,
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation,
        )
        self.preview_label.setText("")
        self.preview_label.setPixmap(scaled)

    def update_preview_pixmap(self):
        if self.current_front_path is None or self.current_back_path is None:
            return
        if self.current_showing_back:
            self.preview_side_label.setText("Back (click image to toggle)")
            self.set_preview_pixmap(
                self.current_back_path,
                f"Cannot preview: {self.current_back_path.name}",
            )
        else:
            self.preview_side_label.setText("Front (click image to toggle)")
            self.set_preview_pixmap(
                self.current_front_path,
                f"Cannot preview: {self.current_front_path.name}",
            )

    def toggle_preview_side(self):
        if self.current_front_path is None or self.current_back_path is None:
            return
        self.current_showing_back = not self.current_showing_back
        self.update_preview_pixmap()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.update_preview_pixmap()

    def append_log(self, message: str):
        self.feedback.appendPlainText(message)

    def selected_pairs(self) -> list:
        selected = []
        for row in range(self.image_list.count()):
            item = self.image_list.item(row)
            if item.checkState() != Qt.Checked:
                continue
            front = Path(item.data(Qt.UserRole))
            back = Path(item.data(Qt.UserRole + 1))
            selected.append((front, back))
        return selected

    def update_submit_enabled(self):
        has_selection = len(self.selected_pairs()) > 0
        self.submit_button.setEnabled(bool(self.image_pairs) and has_selection)

    def select_all_pairs(self):
        for row in range(self.image_list.count()):
            self.image_list.item(row).setCheckState(Qt.Checked)
        self.update_submit_enabled()

    def select_no_pairs(self):
        for row in range(self.image_list.count()):
            self.image_list.item(row).setCheckState(Qt.Unchecked)
        self.update_submit_enabled()

    def set_controls_enabled(self, enabled: bool):
        self.category_combo.setEnabled(enabled)
        self.directory_edit.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.refresh_button.setEnabled(enabled)
        self.image_list.setEnabled(enabled)
        self.select_all_button.setEnabled(enabled and bool(self.image_pairs))
        self.select_none_button.setEnabled(enabled and bool(self.image_pairs))
        self.compress_checkbox.setEnabled(enabled)
        self.viewer_checkbox.setEnabled(enabled)
        self.view_descriptions_button.setEnabled(
            enabled and self.view_descriptions_button.isVisible()
        )
        self.edit_instructions_button.setEnabled(enabled)
        self.overrides_widget.setEnabled(enabled and self.overrides_widget.isVisible())
        if enabled:
            self.update_submit_enabled()
        else:
            self.submit_button.setEnabled(False)

    def update_view_descriptions_button(self, directory: Path):
        if directory is None:
            self.view_descriptions_button.setVisible(False)
            return
        has_csv = (directory / "description.csv").exists()
        self.view_descriptions_button.setVisible(has_csv)
        self.view_descriptions_button.setEnabled(has_csv and self.thread is None)

    def view_descriptions_now(self):
        directory = Path(self.directory_edit.text()).expanduser().resolve()
        csv_path = directory / "description.csv"
        if not csv_path.exists():
            QMessageBox.warning(
                self, "Missing File", "description.csv was not found in this directory."
            )
            self.update_view_descriptions_button(directory)
            return
        self.launch_viewer(directory)

    def open_instructions_editor(self):
        dialog = InstructionsEditorDialog(instructions_file_path(), self)
        dialog.exec_()

    def start_processing(self):
        if self.thread is not None:
            return

        directory = Path(self.directory_edit.text()).expanduser().resolve()
        if not directory.exists() or not directory.is_dir():
            QMessageBox.warning(self, "Invalid Directory", "Select a valid image directory.")
            return

        files = list_image_files(directory)
        if not files:
            QMessageBox.warning(self, "No Images", "No image files found in the directory.")
            return
        if len(files) % 2 != 0:
            QMessageBox.warning(
                self,
                "Pairing Error",
                "Image count must be even so each item has front/back images.",
            )
            return
        selected_pairs = self.selected_pairs()
        if not selected_pairs:
            QMessageBox.warning(
                self,
                "No Pairs Selected",
                "Select at least one image pair to process.",
            )
            return
        selected_pairs = sorted(selected_pairs, key=pair_filename_sort_key)

        self.progress_bar.setValue(0)
        self.append_log(
            f"Starting {self.category_combo.currentText()} processing on {len(selected_pairs)} selected items..."
        )

        self.set_controls_enabled(False)

        self.thread = QThread(self)
        self.worker = ProcessorWorker(
            directory,
            selected_pairs,
            self.category_key(),
            self.compress_checkbox.isChecked(),
            self.set_override_edit.text(),
            self.variety_override_edit.text(),
        )
        self.worker.moveToThread(self.thread)

        self.thread.started.connect(self.worker.run)
        self.worker.progress.connect(self.append_log)
        self.worker.percent.connect(self.progress_bar.setValue)
        self.worker.finished.connect(self.on_processing_finished)
        self.worker.failed.connect(self.on_processing_failed)
        self.worker.finished.connect(self.thread.quit)
        self.worker.failed.connect(self.thread.quit)
        self.thread.finished.connect(self.cleanup_thread)
        self.thread.start()

    def on_processing_finished(self, csv_path: str, count: int):
        self.progress_bar.setValue(100)
        self.append_log(f"Done. {count} rows written to {csv_path}.")
        if self.viewer_checkbox.isChecked():
            self.launch_viewer(Path(csv_path).parent)

    def on_processing_failed(self, message: str):
        self.append_log(f"Failed:\n{message}")
        QMessageBox.critical(self, "Processing Failed", message.splitlines()[0])

    def cleanup_thread(self):
        if self.worker is not None:
            self.worker.deleteLater()
            self.worker = None
        if self.thread is not None:
            self.thread.deleteLater()
            self.thread = None
        directory = Path(self.directory_edit.text()).expanduser().resolve()
        self.update_view_descriptions_button(directory)
        self.set_controls_enabled(True)

    def launch_viewer(self, directory: Path):
        viewer_script = Path(__file__).with_name("viewer.py")
        if not viewer_script.exists():
            self.append_log("viewer.py not found. Skipping viewer launch.")
            return
        try:
            subprocess.Popen([sys.executable, str(viewer_script)], cwd=str(directory))
            self.append_log("Opened viewer.py.")
        except Exception as exc:
            self.append_log(f"Failed to open viewer.py: {exc}")


def main():
    app = QApplication(sys.argv)
    win = EbayTitleGui()
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
