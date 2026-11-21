from __future__ import annotations

import csv
import json
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from fluent_blue import Alert, Button, ConfirmDialog, Text, ThemeManager
from fluent_blue.compatibility.qt import (
    QApplication, QCheckBox, QFileDialog, QFrame, QGridLayout, QHBoxLayout,
    QImageReader, QLabel, QLineEdit, QMessageBox, QPixmap, QScrollArea,
    QSize, Qt, QThread, QVBoxLayout, QWidget, Signal,
)

CLASS_COLUMNS = (
    "antelope_duiker", "bird", "blank", "civet_genet", "hog", "leopard",
    "monkey_prosimian", "rodent",
)

ANIMAL_CLASSES = tuple(name for name in CLASS_COLUMNS if name != "blank")

CLASS_NAMES_ZH = {
    "antelope_duiker": "羚羊／麂羚",
    "bird": "鸟类",
    "blank": "空拍",
    "civet_genet": "灵猫／獛",
    "hog": "猪类",
    "leopard": "豹",
    "monkey_prosimian": "猴类／原猴类",
    "rodent": "啮齿类",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV has no header: {path.name}")
        return list(reader.fieldnames), list(reader)


def load_dataset(folder: Path) -> dict:
    feature_header, features = read_csv(folder / "train_features.csv")
    label_header, labels = read_csv(folder / "train_labels.csv")
    if not {"id", "filepath", "site"}.issubset(feature_header):
        raise ValueError("train_features.csv must contain id, filepath, and site columns.")
    if "id" not in label_header:
        raise ValueError("train_labels.csv must contain an id column.")
    classes = [name for name in label_header if name != "id"]
    if tuple(classes) != CLASS_COLUMNS:
        raise ValueError("Unexpected label columns. Expected the eight Conservision classes.")

    labels_by_id: dict[str, dict[str, str]] = {}
    for row in labels:
        image_id = row["id"]
        if image_id in labels_by_id:
            raise ValueError(f"Duplicate label id: {image_id}")
        labels_by_id[image_id] = row

    site_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    class_counts: dict[str, Counter] = defaultdict(Counter)
    for row in features:
        image_id, site = row["id"], row["site"].strip()
        if not site:
            raise ValueError(f"Missing site for image {image_id}.")
        if image_id not in labels_by_id:
            raise ValueError(f"No label row for image {image_id}.")
        label = labels_by_id[image_id]
        positive = [name for name in classes if float(label[name]) == 1.0]
        if len(positive) != 1:
            raise ValueError(f"Expected one class per image; found {len(positive)} for {image_id}.")
        site_rows[site].append(row)
        class_counts[site][positive[0]] += 1

    feature_ids = {row["id"] for row in features}
    if feature_ids != set(labels_by_id):
        raise ValueError("Feature and label CSVs have different image IDs.")

    site_summaries = []
    for site in sorted(site_rows):
        site_summaries.append({
            "site": site, "images": len(site_rows[site]),
            **{name: class_counts[site][name] for name in classes},
        })
    return {
        "feature_header": feature_header, "label_header": label_header,
        "features": features, "labels_by_id": labels_by_id,
        "site_rows": site_rows, "site_summaries": site_summaries, "classes": classes,
    }


class BuildSplitWorker(QThread):
    succeeded = Signal(str, dict)
    failed = Signal(str)
    progress = Signal(str)

    def __init__(self, folder: Path, dataset: dict, validation_sites: set[str]):
        super().__init__()
        self.folder, self.dataset = folder, dataset
        self.validation_sites = validation_sites

    def run(self) -> None:
        run_name = f"run_{datetime.now():%Y%m%d_%H%M%S}_{uuid.uuid4().hex[:6]}"
        output_root = self.folder / "split_runs"
        final_dir = output_root / run_name
        staging_dir = output_root / f".{run_name}.building"
        try:
            output_root.mkdir(parents=True, exist_ok=True)
            train_rows = [row for site, rows in self.dataset["site_rows"].items()
                          if site not in self.validation_sites for row in rows]
            val_rows = [row for site in sorted(self.validation_sites)
                        for row in self.dataset["site_rows"][site]]
            staging_dir.mkdir()
            total, done = len(train_rows) + len(val_rows), 0
            for split_name, rows in (("train", train_rows), ("validation", val_rows)):
                split_dir = staging_dir / split_name
                image_dir = split_dir / "train_features"
                image_dir.mkdir(parents=True)
                labels = []
                for row in rows:
                    source = self.folder / row["filepath"]
                    if not source.resolve().is_relative_to(self.folder.resolve()):
                        raise ValueError(f"Image path escapes the source directory: {row['filepath']}")
                    if not source.is_file():
                        raise FileNotFoundError(f"Image not found: {source}")
                    target = image_dir / source.name
                    try:
                        target.hardlink_to(source)
                    except OSError:
                        shutil.copy2(source, target)
                    labels.append(self.dataset["labels_by_id"][row["id"]])
                    done += 1
                    if done % 250 == 0 or done == total:
                        self.progress.emit(f"正在准备图片：{done:,} / {total:,}")
                self._write_csv(split_dir / "train_features.csv", self.dataset["feature_header"], rows)
                self._write_csv(split_dir / "train_labels.csv", self.dataset["label_header"], labels)

            train_sites = sorted(set(self.dataset["site_rows"]) - self.validation_sites)
            report = {
                "source_directory": str(self.folder),
                "validation_sites": sorted(self.validation_sites), "train_sites": train_sites,
                "total_images": total, "train_images": len(train_rows),
                "validation_images": len(val_rows),
                "train_percent": round(100 * len(train_rows) / total, 4),
                "validation_percent": round(100 * len(val_rows) / total, 4),
            }
            (staging_dir / "split_summary.json").write_text(
                json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            staging_dir.rename(final_dir)
            self.succeeded.emit(str(final_dir), report)
        except Exception as error:
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)
            self.failed.emit(str(error))

    @staticmethod
    def _write_csv(path: Path, header: list[str], rows: list[dict[str, str]]) -> None:
        with path.open("w", newline="", encoding="utf-8-sig") as handle:
            writer = csv.DictWriter(handle, fieldnames=header, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)


class SiteCard(QFrame):
    SIZE = 176
    DETAILS_WIDTH = 152

    def __init__(self, summary: dict, classes: list[str], image_path: Path):
        super().__init__()
        self.site = summary["site"]
        self.present_classes = {name for name in classes if summary[name] > 0}
        self.setObjectName("siteCard")
        self.setFixedWidth(self.SIZE)
        self.setStyleSheet(
            'QFrame#siteCard { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 8px; }'
            'QFrame#siteCard[selected="true"] { background: #eff6ff; border: 2px solid #2563eb; }'
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(6)
        self.checkbox = QCheckBox(self.site)
        self.checkbox.setAccessibleName(f"站点 {self.site}，划入验证集")
        self.checkbox.toggled.connect(self.highlight)
        layout.addWidget(self.checkbox)

        self.thumbnail = QLabel("暂无预览")
        self.thumbnail.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.thumbnail.setFixedHeight(126)
        self.thumbnail.setStyleSheet("background: #e9eef4; color: #475569; border-radius: 4px;")
        reader = QImageReader(str(image_path))
        reader.setAutoTransform(True)
        if reader.size().isValid():
            reader.setScaledSize(reader.size().scaled(QSize(152, 124), Qt.AspectRatioMode.KeepAspectRatio))
            image = reader.read()
            if not image.isNull():
                self.thumbnail.setPixmap(QPixmap.fromImage(image).scaled(
                    152, 124, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation
                ))
        layout.addWidget(self.thumbnail)
        self.detail_text = "\n".join([
            f"共 {summary['images']:,} 张图片",
            *(f"{name} ({CLASS_NAMES_ZH[name]}): {summary[name]:,} 张" for name in classes),
        ])
        self.details = QLabel(self.detail_text)
        self.details.setWordWrap(True)
        self.details.setFixedWidth(self.DETAILS_WIDTH)
        self.details.setTextFormat(Qt.TextFormat.PlainText)
        layout.addWidget(self.details)
        self.show_details(False)

    def highlight(self, checked: bool) -> None:
        self.setProperty("selected", checked)
        self.style().unpolish(self)
        self.style().polish(self)
        self.update()

    def show_details(self, expanded: bool) -> None:
        if expanded:
            # Reserve space for wrapped text before the grid computes its row heights.
            self.details.setMinimumHeight(self.details.heightForWidth(self.DETAILS_WIDTH))
        self.details.setVisible(expanded)
        self.setMinimumHeight(0 if expanded else self.SIZE)
        hint = "" if expanded else self.detail_text
        for widget in (self, self.checkbox, self.thumbnail):
            widget.setToolTip(hint)


class SiteGrid(QScrollArea):
    selection_changed = Signal()
    visible_count_changed = Signal(int)
    GAP = 10

    def __init__(self):
        super().__init__()
        self.cards: dict[str, SiteCard] = {}
        self.columns = 0
        self.site_filter = ""
        self.required_classes: set[str] = set()
        self.setWidgetResizable(True)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)
        self.content = QWidget()
        self.grid = QGridLayout(self.content)
        self.grid.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
        self.grid.setContentsMargins(self.GAP, self.GAP, self.GAP, self.GAP)
        self.grid.setSpacing(self.GAP)
        self.setWidget(self.content)

    def set_sites(self, folder: Path, data: dict | None) -> None:
        for card in self.cards.values():
            self.grid.removeWidget(card)
            card.hide()
            card.deleteLater()
        self.cards.clear()
        if data:
            for summary in data["site_summaries"]:
                site = summary["site"]
                image_path = folder / data["site_rows"][site][0]["filepath"]
                card = SiteCard(summary, data["classes"], image_path)
                card.checkbox.toggled.connect(self.selection_changed.emit)
                self.cards[site] = card
        self.reflow(force=True)
        self.verticalScrollBar().setValue(0)

    def filter_sites(self, text: str) -> None:
        self.site_filter = text.strip().casefold()
        self.reflow(force=True)
        self.verticalScrollBar().setValue(0)

    def filter_classes(self, required_classes: set[str]) -> None:
        self.required_classes = required_classes
        self.reflow(force=True)
        self.verticalScrollBar().setValue(0)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self.reflow()

    def reflow(self, *, force: bool = False) -> None:
        columns = max(1, (self.viewport().width() - self.GAP) // (SiteCard.SIZE + self.GAP))
        if not force and columns == self.columns:
            return
        self.columns = columns
        for card in self.cards.values():
            self.grid.removeWidget(card)
        visible_cards = []
        for site, card in self.cards.items():
            matches = (
                self.site_filter in site.casefold()
                and self.required_classes.issubset(card.present_classes)
            )
            card.setVisible(matches)
            if matches:
                visible_cards.append(card)
        for index, card in enumerate(visible_cards):
            row, column = divmod(index, columns)
            self.grid.addWidget(card, row, column, Qt.AlignmentFlag.AlignTop)
        self.visible_count_changed.emit(len(visible_cards))

    def show_details(self, expanded: bool) -> None:
        for card in self.cards.values():
            card.show_details(expanded)

    def clear_selection(self) -> None:
        for card in self.cards.values():
            card.checkbox.setChecked(False)


class SiteSplitter(QWidget):
    def __init__(self, initial_folder: Path):
        super().__init__()
        self.dataset: dict | None = None
        self.worker: BuildSplitWorker | None = None
        self.setWindowTitle("野生动物数据集站点划分")
        self.resize(1420, 850)
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(12)
        root.addWidget(Text("按站点划分训练 / 验证集", style="heading-2"))
        root.addWidget(Text(
            "选择验证站点；同一站点的图片会整体进入同一个集合。原始文件不会被修改。",
            color="text-secondary", wrap=True,
        ))

        path_row = QHBoxLayout()
        path_row.addWidget(Text("数据目录", color="text-secondary"))
        self.path_edit = QLineEdit(str(initial_folder))
        self.path_edit.setClearButtonEnabled(True)
        path_row.addWidget(self.path_edit, 1)
        self.browse_button = Button("浏览…", variant="secondary")
        self.browse_button.clicked.connect(self.choose_folder)
        path_row.addWidget(self.browse_button)
        self.load_button = Button("读取数据", variant="outline", icon="refresh-cw")
        self.load_button.clicked.connect(self.load_data)
        path_row.addWidget(self.load_button)
        root.addLayout(path_row)

        self.alert = Alert(
            "数据要求", "目录需包含 train_features.csv、train_labels.csv 和 train_features 图片文件夹。",
            variant="info", dense=True,
        )
        root.addWidget(self.alert)
        self.summary = Text("尚未读取数据", color="text-secondary")
        site_toolbar = QHBoxLayout()
        site_toolbar.addWidget(self.summary, 1)
        self.site_search = QLineEdit()
        self.site_search.setPlaceholderText("搜索站点编号")
        self.site_search.setAccessibleName("搜索站点编号")
        self.site_search.setClearButtonEnabled(True)
        site_toolbar.addWidget(self.site_search)
        root.addLayout(site_toolbar)
        grid_toolbar = QHBoxLayout()
        self.selection_summary = Text("尚未读取数据", style="body-medium", wrap=True)
        grid_toolbar.addWidget(self.selection_summary, 1)
        self.details_toggle = Button("全部展示", variant="outline")
        self.details_toggle.setCheckable(True)
        self.details_toggle.toggled.connect(self.toggle_details)
        grid_toolbar.addWidget(self.details_toggle)
        root.addLayout(grid_toolbar)

        self.site_grid = SiteGrid()
        self.site_search.textChanged.connect(self.site_grid.filter_sites)
        self.site_grid.selection_changed.connect(self.update_selection_summary)
        class_filter_layout = QGridLayout()
        self.all_animals_filter = QCheckBox("所有动物")
        self.all_animals_filter.clicked.connect(self.toggle_all_animals)
        class_filter_layout.addWidget(self.all_animals_filter, 0, 0)
        self.class_filters = {}
        for index, name in enumerate(CLASS_COLUMNS, start=1):
            checkbox = QCheckBox(CLASS_NAMES_ZH[name])
            checkbox.setToolTip(name)
            checkbox.toggled.connect(self.update_class_filter)
            self.class_filters[name] = checkbox
            row, col = divmod(index, 3)
            class_filter_layout.addWidget(checkbox, row, col)
        root.addWidget(QLabel("按类别筛选（须同时包含勾选的类别）"))
        root.addLayout(class_filter_layout)
        self.filter_result = QLabel()
        self.site_grid.visible_count_changed.connect(self.update_filter_result)
        root.addWidget(self.filter_result)
        root.addWidget(self.site_grid, 1)

        footer = QHBoxLayout()
        footer.addStretch()
        self.clear_button = Button("清除选择", variant="secondary", icon="x")
        self.clear_button.clicked.connect(self.site_grid.clear_selection)
        footer.addWidget(self.clear_button)
        self.create_button = Button("确认并创建划分", variant="primary", icon="folder-plus")
        self.create_button.setEnabled(False)
        self.create_button.clicked.connect(self.confirm_create)
        footer.addWidget(self.create_button)
        root.addLayout(footer)
        self.status_label = QLabel("")
        root.addWidget(self.status_label)
        self.load_data()

    def choose_folder(self) -> None:
        selected = QFileDialog.getExistingDirectory(self, "选择比赛数据目录", self.path_edit.text())
        if selected:
            self.path_edit.setText(selected)
            self.load_data()

    def load_data(self) -> None:
        folder = Path(self.path_edit.text()).expanduser()
        try:
            data = load_dataset(folder)
        except Exception as error:
            self.dataset = None
            self.site_grid.set_sites(folder, None)
            self.details_toggle.setEnabled(False)
            self.create_button.setEnabled(False)
            self.summary.setText("数据读取失败")
            self.selection_summary.setText(str(error))
            return

        self.dataset = data
        self.dataset_folder = folder.resolve()
        self.site_grid.set_sites(self.dataset_folder, data)
        self.site_grid.show_details(self.details_toggle.isChecked())
        self.details_toggle.setEnabled(bool(self.site_grid.cards))
        self.summary.setText(f"共 {len(data['features']):,} 张已标注图片，{len(data['site_summaries'])} 个站点")
        self.update_selection_summary()

    def toggle_all_animals(self, checked: bool) -> None:
        for name in ANIMAL_CLASSES:
            checkbox = self.class_filters[name]
            previous = checkbox.blockSignals(True)
            checkbox.setChecked(checked)
            checkbox.blockSignals(previous)
        self.update_class_filter()

    def update_class_filter(self) -> None:
        required = {name for name, box in self.class_filters.items() if box.isChecked()}
        self.all_animals_filter.setChecked(all(name in required for name in ANIMAL_CLASSES))
        self.site_grid.filter_classes(required)

    def update_filter_result(self, count: int) -> None:
        result = f"显示 {count} 个站点"
        if count == 0 and self.dataset and self.site_grid.cards:
            result += "，无符合条件的站点"
        self.filter_result.setText(result)

    def toggle_details(self, expanded: bool) -> None:
        self.details_toggle.setText("收起详情" if expanded else "全部展示")
        self.site_grid.show_details(expanded)

    def selected_sites(self) -> set[str]:
        return {site for site, card in self.site_grid.cards.items() if card.checkbox.isChecked()}

    def update_selection_summary(self) -> None:
        if not self.dataset:
            return
        chosen_sites = self.selected_sites()
        val_count = sum(len(self.dataset["site_rows"][site]) for site in chosen_sites)
        total = len(self.dataset["features"])
        train_count = total - val_count
        self.selection_summary.setText(
            f"训练集 {train_count:,} 张（{train_count / total if total else 0:.1%}）  |  "
            f"验证集 {val_count:,} 张（{val_count / total if total else 0:.1%}）  |  "
            f"验证站点 {len(chosen_sites)} 个"
        )
        self.create_button.setEnabled(train_count > 0 and val_count > 0 and self.worker is None)

    def confirm_create(self) -> None:
        if not self.dataset:
            return
        validation_sites = self.selected_sites()
        if not validation_sites:
            QMessageBox.warning(self, "尚未选择站点", "请至少选择一个站点作为验证集。")
            return
        val_images = sum(len(self.dataset["site_rows"][site]) for site in validation_sites)
        total = len(self.dataset["features"])
        train_images = total - val_images
        if train_images == 0:
            QMessageBox.warning(self, "划分无效", "不能把所有站点都选为验证集。")
            return
        message = (
            f"验证集：{val_images:,} 张，{val_images / total:.1%}，{len(validation_sites)} 个站点\n"
            f"训练集：{train_images:,} 张，{train_images / total:.1%}\n\n"
            f"验证站点：{', '.join(sorted(validation_sites)[:20])}"
            f"{' …' if len(validation_sites) > 20 else ''}\n\n"
            "将在数据目录下新建 split_runs 实验文件夹；图片优先使用硬链接，原数据不变。"
        )
        if not ConfirmDialog.ask(self, "创建这次数据划分？", message, confirm_text="创建划分"):
            return
        self.create_button.setEnabled(False)
        self.load_button.setEnabled(False)
        self.browse_button.setEnabled(False)
        self.status_label.setText("正在创建划分…")
        self.worker = BuildSplitWorker(self.dataset_folder, self.dataset, validation_sites)
        self.worker.progress.connect(self.status_label.setText)
        self.worker.succeeded.connect(self.on_success)
        self.worker.failed.connect(self.on_failure)
        self.worker.finished.connect(self.on_worker_finished)
        self.worker.start()

    def on_success(self, path: str, report: dict) -> None:
        self.status_label.setText(f"创建完成：{path}")
        QMessageBox.information(
            self, "划分已创建",
            f"训练集：{report['train_images']:,} 张（{report['train_percent']:.2f}%）\n"
            f"验证集：{report['validation_images']:,} 张（{report['validation_percent']:.2f}%）\n\n"
            f"目录：\n{path}\n\n每个子目录中的 filepath 均可相对该子目录读取。",
        )

    def on_failure(self, message: str) -> None:
        self.status_label.setText("创建失败")
        QMessageBox.critical(self, "创建失败", message)

    def on_worker_finished(self) -> None:
        self.worker = None
        self.load_button.setEnabled(True)
        self.browse_button.setEnabled(True)
        self.update_selection_summary()


def main() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    ThemeManager.set_theme("light")
    window = SiteSplitter(Path(__file__).resolve().parent)
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
