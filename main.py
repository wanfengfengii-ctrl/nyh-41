import sys
from datetime import datetime
from typing import List, Optional

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt, QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Signal
from PySide6.QtGui import QColor, QFont, QIcon
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QTabWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QFormLayout, QLabel, QLineEdit, QPushButton, QTableView, QHeaderView,
    QMessageBox, QComboBox, QDateEdit, QDoubleSpinBox, QFileDialog, QSplitter,
    QListWidget, QListWidgetItem, QGroupBox, QStatusBar, QTextEdit, QAbstractItemView,
    QCheckBox, QScrollArea, QSizePolicy
)

import db
import models
import csv_importer
import anomaly_detection


pg.setConfigOptions(antialias=True, background='w', foreground='k')


class SampleTableModel(QAbstractTableModel):
    HEADERS = ["编号", "试样编号", "纸张类型", "点墨日期", "批次", "基线", "风险", "判断", "创建时间"]

    def __init__(self, data: Optional[List] = None):
        super().__init__()
        self._data = data or []

    def update_data(self, data: List):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._data):
            return None

        row = self._data[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return row["id"]
            elif col == 1:
                return row["sample_no"]
            elif col == 2:
                return row["paper_type"]
            elif col == 3:
                return row["ink_date"]
            elif col == 4:
                return row["batch_code"] if row["batch_code"] else "-"
            elif col == 5:
                return "★ 基线" if row.get("is_baseline", 0) else "-"
            elif col == 6:
                return row["risk_flag"]
            elif col == 7:
                return row["judgment"] if row["judgment"] else "-"
            elif col == 8:
                return row["created_at"]

        if role == Qt.ForegroundRole:
            if col == 5 and row.get("is_baseline", 0):
                return QColor("#2980b9")
            if col == 6:
                risk = row["risk_flag"]
                if risk == "高风险":
                    return QColor("#c0392b")
                elif risk == "中风险":
                    return QColor("#e67e22")
                elif risk == "低风险":
                    return QColor("#f1c40f")
                return QColor("#27ae60")

        if role == Qt.FontRole and col == 5 and row.get("is_baseline", 0):
            f = QFont()
            f.setBold(True)
            return f

        return None


class MeasurementTableModel(QAbstractTableModel):
    HEADERS = ["时间(s)", "半径(mm)", "毛糙度", "是否异常", "异常类型"]

    def __init__(self, data: Optional[List] = None):
        super().__init__()
        self._data = data or []

    def update_data(self, data: List):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._data):
            return None

        row = self._data[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return f"{row['adsorb_time']:.1f}"
            elif col == 1:
                return f"{row['radius']:.2f}"
            elif col == 2:
                return f"{row['roughness']:.2f}"
            elif col == 3:
                return "是" if row["is_anomaly"] else "否"
            elif col == 4:
                return row["anomaly_type"] if row["anomaly_type"] else "-"

        if role == Qt.ForegroundRole and col == 3 and row["is_anomaly"]:
            return QColor("#c0392b")

        return None


class BatchTableModel(QAbstractTableModel):
    HEADERS = ["编号", "批次号", "风险等级", "连续异常", "备注", "更新时间"]

    def __init__(self, data: Optional[List] = None):
        super().__init__()
        self._data = data or []

    def update_data(self, data: List):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._data):
            return None

        row = self._data[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return row["id"]
            elif col == 1:
                return row["batch_code"]
            elif col == 2:
                return row["risk_level"]
            elif col == 3:
                return str(row["consecutive_anomalies"])
            elif col == 4:
                return row["remark"] if row["remark"] else "-"
            elif col == 5:
                return row["updated_at"]

        if role == Qt.ForegroundRole and col == 2:
            risk = row["risk_level"]
            if risk == "高风险":
                return QColor("#c0392b")
            elif risk == "中风险":
                return QColor("#e67e22")
            elif risk == "低风险":
                return QColor("#f1c40f")
            return QColor("#27ae60")

        return None


class ImportFailureTableModel(QAbstractTableModel):
    HEADERS = ["源文件", "行号", "失败原因", "原始数据", "导入时间"]

    def __init__(self, data: Optional[List] = None):
        super().__init__()
        self._data = data or []

    def update_data(self, data: List):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._data):
            return None

        row = self._data[index.row()]
        col = index.column()

        if role == Qt.DisplayRole:
            if col == 0:
                return row["source_file"]
            elif col == 1:
                return row["row_number"]
            elif col == 2:
                return row["failure_reason"]
            elif col == 3:
                return row["raw_data"]
            elif col == 4:
                return row["imported_at"]

        return None


class SampleManagementTab(QWidget):
    data_updated = Signal()

    def __init__(self):
        super().__init__()
        self._data_loaded = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        form_group = QGroupBox("录入试样")
        form_layout = QFormLayout(form_group)

        self.sample_no_edit = QLineEdit()
        self.sample_no_edit.setPlaceholderText("请输入试样编号（必须唯一）")

        self.paper_type_combo = QComboBox()
        self.paper_type_combo.setEditable(True)
        self.paper_type_combo.setPlaceholderText("选择或输入纸型，如：宣纸、毛边纸、皮纸")
        self.paper_type_combo.currentTextChanged.connect(self._on_paper_type_changed)

        self.ink_date_edit = QDateEdit()
        self.ink_date_edit.setCalendarPopup(True)
        self.ink_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.ink_date_edit.setDate(datetime.now())

        self.batch_code_edit = QLineEdit()
        self.batch_code_edit.setPlaceholderText("可选，批次号将自动关联")

        self.is_baseline_cb = QCheckBox("设为该纸型基线试样（自动生成基线模板）")

        self.baseline_hint_label = QLabel()
        self.baseline_hint_label.setStyleSheet("color: #2980b9; font-size: 12px; padding: 4px;")
        self.baseline_hint_label.setWordWrap(True)

        form_layout.addRow("试样编号 *", self.sample_no_edit)
        form_layout.addRow("纸张类型 *", self.paper_type_combo)
        form_layout.addRow("点墨日期 *", self.ink_date_edit)
        form_layout.addRow("批次号", self.batch_code_edit)
        form_layout.addRow("", self.is_baseline_cb)
        form_layout.addRow("", self.baseline_hint_label)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加试样")
        self.add_btn.clicked.connect(self._add_sample)
        self.clear_btn = QPushButton("清空")
        self.clear_btn.clicked.connect(self._clear_form)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.clear_btn)
        btn_layout.addStretch()

        form_layout.addRow(btn_layout)

        main_layout.addWidget(form_group)

        list_group = QGroupBox("试样列表")
        list_layout = QVBoxLayout(list_group)

        self.sample_table = QTableView()
        self.sample_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.sample_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.sample_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.sample_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.sample_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.ResizeToContents)

        self.sample_model = SampleTableModel()
        self.filter_proxy = QSortFilterProxyModel()
        self.filter_proxy.setSourceModel(self.sample_model)
        self.filter_proxy.setFilterCaseSensitivity(Qt.CaseInsensitive)
        self.filter_proxy.setFilterKeyColumn(1)
        self.sample_table.setModel(self.filter_proxy)

        filter_layout = QHBoxLayout()
        filter_layout.addWidget(QLabel("搜索试样编号："))
        self.filter_edit = QLineEdit()
        self.filter_edit.textChanged.connect(self.filter_proxy.setFilterFixedString)
        filter_layout.addWidget(self.filter_edit)
        list_layout.addLayout(filter_layout)

        list_layout.addWidget(self.sample_table)

        btn_row = QHBoxLayout()
        self.delete_btn = QPushButton("删除选中试样")
        self.delete_btn.clicked.connect(self._delete_sample)
        self.set_baseline_btn = QPushButton("★ 设为纸型基线")
        self.set_baseline_btn.clicked.connect(self._set_selected_as_baseline)
        self.unset_baseline_btn = QPushButton("取消基线标记")
        self.unset_baseline_btn.clicked.connect(self._unset_selected_baseline)
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._load_samples)
        btn_row.addWidget(self.delete_btn)
        btn_row.addWidget(self.set_baseline_btn)
        btn_row.addWidget(self.unset_baseline_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.refresh_btn)
        list_layout.addLayout(btn_row)

        main_layout.addWidget(list_group, 1)

    def _refresh_paper_types(self):
        current_text = self.paper_type_combo.currentText()
        self.paper_type_combo.blockSignals(True)
        self.paper_type_combo.clear()
        paper_types = db.get_all_paper_types()
        self.paper_type_combo.addItems(paper_types)
        self.paper_type_combo.setCurrentText(current_text)
        self.paper_type_combo.blockSignals(False)
        self._on_paper_type_changed(current_text)

    def _on_paper_type_changed(self, text):
        pt = text.strip()
        if not pt:
            self.baseline_hint_label.setText("")
            return
        template = db.get_baseline_template_by_paper(pt)
        baselines = db.get_baselines_by_paper_type(pt)
        if template and baselines:
            sample_row = db.get_sample_by_id(template["baseline_sample_id"]) if template["baseline_sample_id"] else None
            sample_no = sample_row["sample_no"] if sample_row else "-"
            self.baseline_hint_label.setText(
                f"✓ 纸型「{pt}」已有基线模板（来源试样：{sample_no}，共 {len(baselines)} 个基线试样）。"
                f"新增试样将自动匹配此基线进行对照分析。"
            )
        elif baselines:
            self.baseline_hint_label.setText(
                f"△ 纸型「{pt}」有 {len(baselines)} 个基线试样，建议创建基线模板以获得更稳定的对照。"
            )
        else:
            self.baseline_hint_label.setText(
                f"△ 纸型「{pt}」尚未建立基线模板，建议勾选「设为该纸型基线试样」或后续标记一个典型试样作为基线。"
            )

    def _load_samples(self):
        samples = db.get_all_samples()
        self.sample_model.update_data(samples)
        self._refresh_paper_types()
        self._data_loaded = True
        self.data_updated.emit()

    def showEvent(self, event):
        super().showEvent(event)
        if not self._data_loaded:
            self._load_samples()

    def _add_sample(self):
        sample_no = self.sample_no_edit.text().strip()
        paper_type = self.paper_type_combo.currentText().strip()
        ink_date = self.ink_date_edit.date().toString("yyyy-MM-dd")
        batch_code = self.batch_code_edit.text().strip() or None
        is_baseline = 1 if self.is_baseline_cb.isChecked() else 0

        sd = models.SampleData(
            sample_no=sample_no,
            paper_type=paper_type,
            ink_date=ink_date,
            batch_code=batch_code
        )

        vr = models.validate_sample(sd)
        if not vr:
            QMessageBox.warning(self, "验证失败", models.errors_to_text(vr.errors))
            return

        batch_id = None
        if batch_code:
            batch_row = db.get_batch_by_code(batch_code)
            if batch_row:
                batch_id = batch_row["id"]
            else:
                try:
                    batch_id = db.create_batch(batch_code)
                except Exception as e:
                    QMessageBox.warning(self, "批次创建失败", str(e))
                    return

        try:
            new_id = db.create_sample(sample_no, paper_type, ink_date, batch_id, is_baseline=is_baseline)
            msg = f"试样 {sample_no} 添加成功！"
            if is_baseline:
                try:
                    bl = anomaly_detection.build_baseline_from_sample(new_id)
                    if bl:
                        msg += "\n已根据该试样创建基线模板（待录入测量数据后自动生效）。"
                except Exception:
                    msg += "\n基线模板创建将在录入测量数据后进行。"
            else:
                baseline = anomaly_detection.get_paper_type_baseline(paper_type)
                if baseline:
                    msg += f"\n已自动匹配纸型「{paper_type}」基线模板，录入测量数据后将进行对照分析。"
            QMessageBox.information(self, "成功", msg)
            self._clear_form()
            self._load_samples()
        except Exception as e:
            QMessageBox.warning(self, "添加失败", str(e))

    def _clear_form(self):
        self.sample_no_edit.clear()
        self.paper_type_combo.setCurrentText("")
        self.ink_date_edit.setDate(datetime.now())
        self.batch_code_edit.clear()
        self.is_baseline_cb.setChecked(False)

    def _get_selected_sample(self):
        proxy_idx = self.sample_table.currentIndex()
        if not proxy_idx.isValid():
            return None
        source_idx = self.filter_proxy.mapToSource(proxy_idx)
        return self.sample_model._data[source_idx.row()]

    def _set_selected_as_baseline(self):
        row = self._get_selected_sample()
        if not row:
            QMessageBox.information(self, "提示", "请先选择一个试样")
            return
        sample_id = row["id"]
        sample_no = row["sample_no"]
        paper_type = row["paper_type"]

        measurements = db.get_measurements_by_sample(sample_id)
        if len(measurements) < 3:
            reply = QMessageBox.question(
                self, "测量数据不足",
                f"试样 '{sample_no}' 仅有 {len(measurements)} 个测量点，建议至少 3 个点。\n仍要设为基线吗？",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply != QMessageBox.Yes:
                return

        try:
            bl = anomaly_detection.build_baseline_from_sample(sample_id)
            if bl:
                QMessageBox.information(
                    self, "成功",
                    f"已将 '{sample_no}' 设为「{paper_type}」纸型基线，并生成基线模板！\n"
                    f"后续同纸型试样将自动匹配该基线进行对照分析。"
                )
                self._load_samples()
                self.data_updated.emit()
            else:
                QMessageBox.warning(self, "失败", "基线模板构建失败，请检查测量数据")
        except Exception as e:
            QMessageBox.warning(self, "失败", str(e))

    def _unset_selected_baseline(self):
        row = self._get_selected_sample()
        if not row:
            QMessageBox.information(self, "提示", "请先选择一个试样")
            return
        if not row.get("is_baseline", 0):
            QMessageBox.information(self, "提示", "该试样当前不是基线试样")
            return
        sample_id = row["id"]
        sample_no = row["sample_no"]
        paper_type = row["paper_type"]
        reply = QMessageBox.question(
            self, "确认",
            f"确定要取消试样 '{sample_no}' 的基线标记吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        try:
            db.set_sample_baseline(sample_id, 0)

            remaining_baselines = db.get_baselines_by_paper_type(paper_type)
            template = db.get_baseline_template_by_paper(paper_type)

            if remaining_baselines:
                try:
                    bl = anomaly_detection.build_aggregated_baseline(paper_type)
                    if bl:
                        remaining_nos = ", ".join(
                            db.get_sample_by_id(sid)["sample_no"]
                            for sid in bl.sample_ids
                            if db.get_sample_by_id(sid)
                        )
                        QMessageBox.information(
                            self, "成功",
                            f"已取消试样「{sample_no}」的基线标记。\n"
                            f"纸型「{paper_type}」基线模板已用剩余基线试样重建：{remaining_nos}"
                        )
                    else:
                        if template:
                            db.delete_baseline_template(template["id"])
                        QMessageBox.information(
                            self, "成功",
                            f"已取消试样「{sample_no}」的基线标记。\n"
                            f"剩余基线试样数据不足，已删除「{paper_type}」基线模板。"
                        )
                except Exception:
                    if template:
                        db.delete_baseline_template(template["id"])
                    QMessageBox.information(
                        self, "成功",
                        f"已取消试样「{sample_no}」的基线标记。\n"
                        f"重建模板失败，已删除「{paper_type}」基线模板。"
                    )
            else:
                if template:
                    db.delete_baseline_template(template["id"])
                QMessageBox.information(
                    self, "成功",
                    f"已取消试样「{sample_no}」的基线标记。\n"
                    f"纸型「{paper_type}」已无基线试样，基线模板已一并删除。"
                )

            self._load_samples()
            self.data_updated.emit()
        except Exception as e:
            QMessageBox.warning(self, "失败", str(e))

    def _delete_sample(self):
        row = self._get_selected_sample()
        if not row:
            QMessageBox.information(self, "提示", "请先选择一个试样")
            return
        sample_id = row["id"]
        sample_no = row["sample_no"]

        reply = QMessageBox.question(
            self, "确认删除",
            f"确定要删除试样 '{sample_no}' 及其所有测量数据吗？\n此操作不可撤销！",
            QMessageBox.Yes | QMessageBox.No
        )

        if reply == QMessageBox.Yes:
            try:
                db.delete_sample(sample_id)
                self._load_samples()
                QMessageBox.information(self, "成功", "删除成功")
            except Exception as e:
                QMessageBox.warning(self, "删除失败", str(e))


class MeasurementTab(QWidget):
    data_updated = Signal()

    def __init__(self):
        super().__init__()
        self._selected_sample_id = None
        self._data_loaded = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        form_group = QGroupBox("录入测量数据")
        form_layout = QFormLayout(form_group)

        self.sample_combo = QComboBox()
        self.sample_combo.currentIndexChanged.connect(self._on_sample_changed)

        self.adsorb_time_spin = QDoubleSpinBox()
        self.adsorb_time_spin.setRange(0.1, 10000.0)
        self.adsorb_time_spin.setDecimals(1)
        self.adsorb_time_spin.setSingleStep(1.0)
        self.adsorb_time_spin.setSuffix(" s")

        self.radius_spin = QDoubleSpinBox()
        self.radius_spin.setRange(0.01, 1000.0)
        self.radius_spin.setDecimals(2)
        self.radius_spin.setSingleStep(0.1)
        self.radius_spin.setSuffix(" mm")

        self.roughness_spin = QDoubleSpinBox()
        self.roughness_spin.setRange(0.0, 100.0)
        self.roughness_spin.setDecimals(2)
        self.roughness_spin.setSingleStep(0.1)

        form_layout.addRow("选择试样 *", self.sample_combo)
        form_layout.addRow("吸附时间 *", self.adsorb_time_spin)
        form_layout.addRow("扩散半径 *", self.radius_spin)
        form_layout.addRow("边缘毛糙度 *", self.roughness_spin)

        btn_layout = QHBoxLayout()
        self.add_btn = QPushButton("添加测量")
        self.add_btn.clicked.connect(self._add_measurement)
        self.analyze_btn = QPushButton("重新分析异常")
        self.analyze_btn.clicked.connect(self._reanalyze)
        self.analyze_btn.setEnabled(False)
        btn_layout.addWidget(self.add_btn)
        btn_layout.addWidget(self.analyze_btn)
        btn_layout.addStretch()

        form_layout.addRow(btn_layout)

        main_layout.addWidget(form_group)

        list_group = QGroupBox("测量数据")
        list_layout = QVBoxLayout(list_group)

        self.info_label = QLabel("请选择一个试样查看测量数据")
        list_layout.addWidget(self.info_label)

        self.measurement_table = QTableView()
        self.measurement_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.measurement_model = MeasurementTableModel()
        self.measurement_table.setModel(self.measurement_model)
        list_layout.addWidget(self.measurement_table, 1)

        self.judgment_group = QGroupBox("试样分析结果")
        judgment_layout = QVBoxLayout(self.judgment_group)
        self.judgment_text = QTextEdit()
        self.judgment_text.setReadOnly(True)
        self.judgment_text.setMaximumHeight(120)
        judgment_layout.addWidget(self.judgment_text)
        self.judgment_group.setVisible(False)
        list_layout.addWidget(self.judgment_group)

        refresh_btn = QPushButton("刷新列表")
        refresh_btn.clicked.connect(self._load_measurements)
        list_layout.addWidget(refresh_btn)

        main_layout.addWidget(list_group, 1)

    def _load_samples(self):
        self.sample_combo.blockSignals(True)
        self.sample_combo.clear()
        self.sample_combo.addItem("-- 请选择 --", None)
        samples = db.get_all_samples()
        for s in samples:
            text = f"{s['sample_no']} ({s['paper_type']})"
            self.sample_combo.addItem(text, s["id"])
        self.sample_combo.blockSignals(False)
        self._data_loaded = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._data_loaded:
            self._load_samples()

    def _on_sample_changed(self, idx):
        self._selected_sample_id = self.sample_combo.itemData(idx)
        self.analyze_btn.setEnabled(self._selected_sample_id is not None)
        self._load_measurements()

    def _load_measurements(self):
        if self._selected_sample_id is None:
            self.measurement_model.update_data([])
            self.info_label.setText("请选择一个试样查看测量数据")
            self.judgment_group.setVisible(False)
            return

        sample = db.get_sample_by_id(self._selected_sample_id)
        measurements = db.get_measurements_by_sample(self._selected_sample_id)
        self.measurement_model.update_data(measurements)
        self.info_label.setText(f"试样: {sample['sample_no']} | 共 {len(measurements)} 个测量点")

        if sample and sample["judgment"]:
            self.judgment_group.setVisible(True)
            judgment_text = f"判断结果: {sample['judgment']}\n风险等级: {sample['risk_flag']}"
            all_samples = db.get_all_samples()
            ref_ids = [s["id"] for s in all_samples if s["id"] != self._selected_sample_id]
            j = anomaly_detection.judge_sample(self._selected_sample_id, ref_ids if ref_ids else None)
            reasons = "\n".join(f"- {r}" for r in j.reasons)
            self.judgment_text.setPlainText(f"{judgment_text}\n\n分析依据:\n{reasons}")
        else:
            self.judgment_group.setVisible(False)

    def _add_measurement(self):
        if self._selected_sample_id is None:
            QMessageBox.warning(self, "提示", "请先选择一个试样")
            return

        sample_idx = self.sample_combo.currentIndex()
        sample_data = self.sample_combo.itemText(sample_idx).split(" ")[0]

        md = models.MeasurementData(
            sample_no=sample_data,
            adsorb_time=self.adsorb_time_spin.value(),
            radius=self.radius_spin.value(),
            roughness=self.roughness_spin.value()
        )

        vr, sample_id = models.validate_measurement(md)
        if not vr:
            QMessageBox.warning(self, "验证失败", models.errors_to_text(vr.errors))
            return

        try:
            db.create_measurement(
                sample_id,
                self.adsorb_time_spin.value(),
                self.radius_spin.value(),
                self.roughness_spin.value()
            )

            result = anomaly_detection.process_measurement_and_update_risk(sample_id)

            msg_parts = ["测量添加成功！"]
            if result["sample_judgment"]:
                j = result["sample_judgment"]
                msg_parts.append(f"\n试样判断: {j.judgment}")
                msg_parts.append(f"风险等级: {j.risk_flag}")
            if result["batch_risk"]:
                msg_parts.append(f"\n批次风险: {result['batch_risk']}")
                msg_parts.append(f"连续异常: {result['batch_consecutive']} 点")

            QMessageBox.information(self, "成功", "\n".join(msg_parts))
            self._load_measurements()
            self.data_updated.emit()
        except Exception as e:
            QMessageBox.warning(self, "添加失败", str(e))

    def _reanalyze(self):
        if self._selected_sample_id is None:
            return

        result = anomaly_detection.process_measurement_and_update_risk(self._selected_sample_id)
        self._load_measurements()
        self.data_updated.emit()

        j = result["sample_judgment"]
        msg = f"重新分析完成！\n\n判断结果: {j.judgment}\n风险等级: {j.risk_flag}"
        if result["batch_risk"]:
            msg += f"\n\n批次风险: {result['batch_risk']}\n连续异常: {result['batch_consecutive']} 点"
        QMessageBox.information(self, "分析完成", msg)

    def refresh_samples(self):
        current_idx = self.sample_combo.currentIndex()
        current_id = self._selected_sample_id
        self._load_samples()
        if current_id:
            for i in range(self.sample_combo.count()):
                if self.sample_combo.itemData(i) == current_id:
                    self.sample_combo.setCurrentIndex(i)
                    return
        self.sample_combo.setCurrentIndex(0)


class SampleDetailCard(QWidget):
    def __init__(self, sample, color, measurements, judgment, parent=None):
        super().__init__(parent)
        self._sample = sample
        self._color = color
        self._measurements = measurements
        self._judgment = judgment
        self._init_ui()

    def _risk_color(self, risk: str) -> str:
        if risk == "高风险":
            return "#c0392b"
        elif risk == "中风险":
            return "#e67e22"
        elif risk == "低风险":
            return "#f1c40f"
        return "#27ae60"

    def _init_ui(self):
        import statistics

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        self.setStyleSheet(f"""
            SampleDetailCard {{
                border: 2px solid {self._color};
                border-radius: 6px;
                background-color: #fafafa;
            }}
        """)
        self.setMinimumWidth(300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        header = QHBoxLayout()
        color_bar = QWidget()
        color_bar.setFixedWidth(6)
        color_bar.setStyleSheet(f"background-color: {self._color}; border-radius: 3px;")
        header.addWidget(color_bar)

        title_layout = QVBoxLayout()
        title = QLabel(f"{self._sample['sample_no']}  ·  {self._sample['paper_type']}")
        f = title.font()
        f.setBold(True)
        f.setPointSize(f.pointSize() + 1)
        title.setFont(f)
        title.setStyleSheet(f"color: {self._color};")

        meta = QLabel(f"点墨日期: {self._sample['ink_date']}")
        meta.setStyleSheet("color: #666; font-size: 11px;")
        title_layout.addWidget(title)
        title_layout.addWidget(meta)
        header.addLayout(title_layout, 1)

        risk_label = QLabel(self._judgment.risk_flag)
        risk_label.setAlignment(Qt.AlignCenter)
        risk_label.setFixedWidth(64)
        risk_label.setStyleSheet(f"""
            background-color: {self._risk_color(self._judgment.risk_flag)};
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            font-size: 12px;
        """)
        header.addWidget(risk_label)
        layout.addLayout(header)

        plot = pg.PlotWidget()
        plot.setMaximumHeight(180)
        plot.setLabel('left', '半径(mm)')
        plot.setLabel('bottom', '时间(s)')
        plot.showGrid(x=True, y=True, alpha=0.3)
        plot.setMouseEnabled(x=True, y=False)
        plot.setBackground('#ffffff')

        if self._measurements:
            times_arr = np.array([float(m["adsorb_time"]) for m in self._measurements])
            radii_arr = np.array([float(m["radius"]) for m in self._measurements])
            sort_idx = np.argsort(times_arr)
            times_arr = times_arr[sort_idx]
            radii_arr = radii_arr[sort_idx]
            ms_sorted = [self._measurements[i] for i in sort_idx]

            plot.plot(times_arr, radii_arr, pen=pg.mkPen(color=self._color, width=2),
                      symbol='o', symbolSize=6, symbolBrush=self._color, symbolPen=self._color)

            anomaly_t = [float(m["adsorb_time"]) for m in ms_sorted if m["is_anomaly"]]
            anomaly_r = [float(m["radius"]) for m in ms_sorted if m["is_anomaly"]]
            if anomaly_t:
                plot.plot(np.array(anomaly_t), np.array(anomaly_r),
                          pen=None, symbol='x', symbolSize=14,
                          symbolBrush=QColor("#c0392b"), symbolPen=QColor("#c0392b"))
        else:
            plot.setTitle("无测量数据")

        layout.addWidget(plot)

        if self._measurements:
            times_list = [float(m["adsorb_time"]) for m in self._measurements]
            radii_list = [float(m["radius"]) for m in self._measurements]
            rough_list = [float(m["roughness"]) for m in self._measurements]
            slope = anomaly_detection._fit_slope(times_list, radii_list)
            avg_r = statistics.mean(radii_list)
            max_r = max(radii_list)
            avg_ro = statistics.mean(rough_list)
            anom_cnt = sum(1 for m in self._measurements if m["is_anomaly"])

            stats_layout = QHBoxLayout()
            stats = [
                ("渗化斜率", f"{slope:.3f}"),
                ("平均半径", f"{avg_r:.2f}mm"),
                ("最大半径", f"{max_r:.2f}mm"),
                ("平均毛糙", f"{avg_ro:.2f}"),
                (f"异常点", f"{anom_cnt}/{len(self._measurements)}"),
            ]
            for label, val in stats:
                stat_box = QVBoxLayout()
                lbl = QLabel(label)
                lbl.setStyleSheet("color: #888; font-size: 10px;")
                lbl.setAlignment(Qt.AlignCenter)
                val_lbl = QLabel(val)
                val_lbl.setAlignment(Qt.AlignCenter)
                vf = val_lbl.font()
                vf.setBold(True)
                val_lbl.setFont(vf)
                stat_box.addWidget(lbl)
                stat_box.addWidget(val_lbl)
                stats_layout.addLayout(stat_box)
            layout.addLayout(stats_layout)

        judge_box = QVBoxLayout()
        judge_title = QLabel(f"判断: {self._judgment.judgment}")
        jf = judge_title.font()
        jf.setBold(True)
        judge_title.setFont(jf)
        judge_title.setStyleSheet(f"color: {self._risk_color(self._judgment.risk_flag)};")
        judge_box.addWidget(judge_title)

        for r in self._judgment.reasons[:4]:
            reason_lbl = QLabel(f"  · {r}")
            reason_lbl.setStyleSheet("color: #555; font-size: 11px;")
            reason_lbl.setWordWrap(True)
            judge_box.addWidget(reason_lbl)
        layout.addLayout(judge_box)

        layout.addStretch()


class ComparisonTab(QWidget):
    def __init__(self):
        super().__init__()
        self._selected_ids: List[int] = []
        self._plot_curves = {}
        self._plot_scatters = {}
        self._detail_cards = []
        self._data_loaded = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(310)

        list_group = QGroupBox("选择试样对比")
        list_layout = QVBoxLayout(list_group)

        self.sample_list = QListWidget()
        self.sample_list.setSelectionMode(QAbstractItemView.MultiSelection)
        self.sample_list.itemSelectionChanged.connect(self._on_selection_changed)
        list_layout.addWidget(self.sample_list)

        btn_row = QHBoxLayout()
        select_all_btn = QPushButton("全选")
        select_all_btn.clicked.connect(lambda: self.sample_list.selectAll())
        clear_btn = QPushButton("清空")
        clear_btn.clicked.connect(lambda: self.sample_list.clearSelection())
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._load_samples)
        btn_row.addWidget(select_all_btn)
        btn_row.addWidget(clear_btn)
        btn_row.addWidget(refresh_btn)
        list_layout.addLayout(btn_row)

        left_layout.addWidget(list_group)

        options_group = QGroupBox("显示选项")
        options_layout = QVBoxLayout(options_group)

        self.show_anomaly_cb = QCheckBox("高亮异常点")
        self.show_anomaly_cb.setChecked(True)
        self.show_anomaly_cb.stateChanged.connect(self._update_plot)

        self.show_legend_cb = QCheckBox("显示图例")
        self.show_legend_cb.setChecked(True)
        self.show_legend_cb.stateChanged.connect(self._update_plot)

        self.show_grid_cb = QCheckBox("显示网格")
        self.show_grid_cb.setChecked(True)
        self.show_grid_cb.stateChanged.connect(self._on_grid_changed)

        self.show_details_cb = QCheckBox("显示并排明细")
        self.show_details_cb.setChecked(True)
        self.show_details_cb.stateChanged.connect(self._on_details_toggled)

        self.auto_baseline_cb = QCheckBox("自动叠加同纸型基线（虚线）")
        self.auto_baseline_cb.setChecked(True)
        self.auto_baseline_cb.stateChanged.connect(self._update_plot)

        self.show_deviation_cb = QCheckBox("显示「与基线偏差」叠加视图")
        self.show_deviation_cb.setChecked(True)
        self.show_deviation_cb.stateChanged.connect(self._on_deviation_toggled)

        self.deviation_mode_combo = QComboBox()
        self.deviation_mode_combo.addItem("绝对偏差 (mm)", "abs")
        self.deviation_mode_combo.addItem("相对偏差 (%)", "pct")
        self.deviation_mode_combo.currentIndexChanged.connect(self._update_deviation_plot)

        options_layout.addWidget(self.show_anomaly_cb)
        options_layout.addWidget(self.show_legend_cb)
        options_layout.addWidget(self.show_grid_cb)
        options_layout.addWidget(self.show_details_cb)
        options_layout.addSpacing(6)
        options_layout.addWidget(self.auto_baseline_cb)
        options_layout.addWidget(self.show_deviation_cb)
        dev_layout = QHBoxLayout()
        dev_layout.addWidget(QLabel("偏差模式:"))
        dev_layout.addWidget(self.deviation_mode_combo)
        options_layout.addLayout(dev_layout)

        left_layout.addWidget(options_group)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.right_splitter = QSplitter(Qt.Vertical)

        self.curves_splitter = QSplitter(Qt.Vertical)

        merge_group = QGroupBox("合并渗化曲线（可叠加同纸型基线）")
        merge_layout = QVBoxLayout(merge_group)

        self.plot_widget = pg.PlotWidget()
        self.plot_widget.setLabel('left', '扩散半径', units='mm')
        self.plot_widget.setLabel('bottom', '吸附时间', units='s')
        self.plot_widget.setTitle("多试样渗化曲线对比")
        self.plot_widget.showGrid(x=True, y=True)
        self.plot_widget.setMouseEnabled(x=True, y=True)

        if self.show_legend_cb.isChecked():
            self.plot_widget.addLegend()

        merge_layout.addWidget(self.plot_widget)
        self.curves_splitter.addWidget(merge_group)

        self.deviation_group = QGroupBox("与基线偏差叠加视图（正值=大于基线，负值=小于基线）")
        deviation_layout = QVBoxLayout(self.deviation_group)

        self.deviation_plot = pg.PlotWidget()
        self.deviation_plot.setLabel('bottom', '吸附时间', units='s')
        self.deviation_plot.setTitle("与纸型基线偏差")
        self.deviation_plot.showGrid(x=True, y=True)
        self.deviation_plot.setMouseEnabled(x=True, y=True)
        self.deviation_plot.addLegend()

        deviation_layout.addWidget(self.deviation_plot)
        self.curves_splitter.addWidget(self.deviation_group)
        self.curves_splitter.setStretchFactor(0, 3)
        self.curves_splitter.setStretchFactor(1, 2)

        self.right_splitter.addWidget(self.curves_splitter)

        self.detail_group = QGroupBox("试样并排明细")
        detail_layout = QVBoxLayout(self.detail_group)

        self.detail_scroll = QScrollArea()
        self.detail_scroll.setWidgetResizable(True)
        self.detail_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.detail_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)

        self.detail_container = QWidget()
        self.detail_scroll_layout = QHBoxLayout(self.detail_container)
        self.detail_scroll_layout.setSpacing(8)
        self.detail_scroll_layout.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self.detail_scroll.setWidget(self.detail_container)
        detail_layout.addWidget(self.detail_scroll)

        self.right_splitter.addWidget(self.detail_group)
        self.right_splitter.setStretchFactor(0, 4)
        self.right_splitter.setStretchFactor(1, 2)

        right_layout.addWidget(self.right_splitter)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 5)
        main_layout.addWidget(splitter)

        self._on_deviation_toggled()

    def _load_samples(self):
        self.sample_list.clear()
        self._plot_curves.clear()
        self._plot_scatters.clear()
        self.plot_widget.clear()
        self.deviation_plot.clear()
        self._clear_detail_cards()

        samples = db.get_all_samples()
        self._all_samples = {s["id"]: s for s in samples}

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                  '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4',
                  '#ff9800', '#8bc34a', '#ff5722', '#607d8b', '#795548']

        for i, s in enumerate(samples):
            mark = " ★" if s.get("is_baseline", 0) else ""
            item = QListWidgetItem(f"{s['sample_no']} ({s['paper_type']}){mark}")
            item.setData(Qt.UserRole, s["id"])
            color = QColor(colors[i % len(colors)])
            item.setData(Qt.ForegroundRole, color)
            if s.get("is_baseline", 0):
                f = item.font()
                f.setBold(True)
                item.setFont(f)
            self.sample_list.addItem(item)
            self._plot_curves[s["id"]] = color

        self._data_loaded = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._data_loaded:
            self._load_samples()

    def _clear_detail_cards(self):
        while self.detail_scroll_layout.count():
            item = self.detail_scroll_layout.takeAt(0)
            w = item.widget()
            if w:
                w.setParent(None)
                w.deleteLater()
        self._detail_cards = []

    def _on_selection_changed(self):
        self._selected_ids = []
        for item in self.sample_list.selectedItems():
            sid = item.data(Qt.UserRole)
            self._selected_ids.append(sid)
        self._update_plot()
        self._update_details()

    def _on_grid_changed(self):
        show = self.show_grid_cb.isChecked()
        self.plot_widget.showGrid(x=show, y=show)
        self.deviation_plot.showGrid(x=show, y=show)

    def _on_details_toggled(self):
        self.detail_group.setVisible(self.show_details_cb.isChecked())

    def _on_deviation_toggled(self):
        show = self.show_deviation_cb.isChecked()
        self.deviation_group.setVisible(show)
        if show:
            self._update_deviation_plot()

    def _update_details(self):
        self._clear_detail_cards()
        if not self._selected_ids:
            hint = QLabel("请从左侧选择要对比的试样")
            hint.setAlignment(Qt.AlignCenter)
            hint.setStyleSheet("color: #888; padding: 30px;")
            self.detail_scroll_layout.addWidget(hint)
            return

        all_measurements = db.get_measurements_by_samples(self._selected_ids)

        for sid in self._selected_ids:
            sample = self._all_samples.get(sid)
            if not sample:
                continue

            measurements = all_measurements.get(sid, [])
            peer_refs = [s for s in self._selected_ids if s != sid]
            judgment = anomaly_detection.judge_sample(sid, peer_refs if peer_refs else None)

            color = self._plot_curves.get(sid, "#3498db")
            card = SampleDetailCard(sample, color, measurements, judgment)
            self.detail_scroll_layout.addWidget(card)
            self._detail_cards.append(card)

        self.detail_scroll_layout.addStretch()

    def _collect_baselines_for_plot(self):
        baseline_map = {}
        if not self.auto_baseline_cb.isChecked():
            return baseline_map
        paper_types_in_selection = set()
        for sid in self._selected_ids:
            s = self._all_samples.get(sid)
            if s:
                paper_types_in_selection.add(s["paper_type"])
        for pt in paper_types_in_selection:
            bl = anomaly_detection.get_paper_type_baseline(pt)
            if bl and bl.times:
                baseline_map[pt] = bl
        return baseline_map

    def _update_plot(self):
        self.plot_widget.clear()
        if self.show_legend_cb.isChecked():
            self.plot_widget.addLegend()

        if not self._selected_ids and not self.auto_baseline_cb.isChecked():
            self._update_deviation_plot()
            return

        all_measurements = db.get_measurements_by_samples(self._selected_ids)

        baseline_map = self._collect_baselines_for_plot()
        for pt, bl in baseline_map.items():
            bl_times = np.array(bl.times, dtype=float)
            bl_radii = np.array(bl.radii, dtype=float)
            baseline_pen = pg.mkPen(color='#2c3e50', width=3, style=Qt.DashLine)
            self.plot_widget.plot(
                bl_times, bl_radii,
                pen=baseline_pen,
                name=f"[基线] {pt}",
                symbol=None
            )

        for sid in self._selected_ids:
            sample = self._all_samples.get(sid)
            if not sample:
                continue

            measurements = all_measurements.get(sid, [])
            if not measurements:
                continue

            color = self._plot_curves[sid]
            times = np.array([float(m["adsorb_time"]) for m in measurements])
            radii = np.array([float(m["radius"]) for m in measurements])

            sort_idx = np.argsort(times)
            times = times[sort_idx]
            radii = radii[sort_idx]
            measurements_sorted = [measurements[i] for i in sort_idx]

            line_width = 3 if sample.get("is_baseline", 0) else 2
            self.plot_widget.plot(
                times, radii,
                pen=pg.mkPen(color=color, width=line_width),
                name=sample["sample_no"],
                symbol=None
            )

            if self.show_anomaly_cb.isChecked():
                normal_t = []
                normal_r = []
                anomaly_t = []
                anomaly_r = []
                for m in measurements_sorted:
                    t = float(m["adsorb_time"])
                    r = float(m["radius"])
                    if m["is_anomaly"]:
                        anomaly_t.append(t)
                        anomaly_r.append(r)
                    else:
                        normal_t.append(t)
                        normal_r.append(r)

                if normal_t:
                    self.plot_widget.plot(
                        np.array(normal_t), np.array(normal_r),
                        pen=None,
                        symbol='o',
                        symbolSize=7,
                        symbolBrush=color,
                        symbolPen=color,
                        name=f"{sample['sample_no']} 正常"
                    )

                if anomaly_t:
                    self.plot_widget.plot(
                        np.array(anomaly_t), np.array(anomaly_r),
                        pen=None,
                        symbol='x',
                        symbolSize=12,
                        symbolBrush=QColor("#c0392b"),
                        symbolPen=QColor("#c0392b"),
                        name=f"{sample['sample_no']} 异常"
                    )
            else:
                self.plot_widget.plot(
                    times, radii,
                    pen=None,
                    symbol='o',
                    symbolSize=7,
                    symbolBrush=color,
                    symbolPen=color,
                    name=f"{sample['sample_no']} 数据点"
                )

        self._update_deviation_plot()

    def _update_deviation_plot(self):
        self.deviation_plot.clear()
        if not self.show_deviation_cb.isChecked():
            return
        if not self._selected_ids:
            return

        mode = self.deviation_mode_combo.currentData()
        self.deviation_plot.setLabel(
            'left',
            '偏差 (mm)' if mode == "abs" else '偏差 (%)'
        )
        self.deviation_plot.setTitle(
            f"与纸型基线偏差 - {'绝对偏差' if mode == 'abs' else '相对偏差'}"
        )

        all_measurements = db.get_measurements_by_samples(self._selected_ids)
        threshold_line = None

        for sid in self._selected_ids:
            sample = self._all_samples.get(sid)
            if not sample:
                continue
            bl = anomaly_detection.get_paper_type_baseline(sample["paper_type"])
            if not bl or not bl.times:
                continue
            measurements = all_measurements.get(sid, [])
            if not measurements:
                continue

            sample_times = [float(m["adsorb_time"]) for m in measurements]
            sample_radii = [float(m["radius"]) for m in measurements]
            sample_rough = [float(m["roughness"]) for m in measurements]

            devs = anomaly_detection.compute_baseline_deviations(
                sample_times, sample_radii, sample_rough, bl
            )
            if not devs:
                continue

            color = self._plot_curves[sid]
            times_arr = np.array([d.time for d in devs], dtype=float)
            if mode == "abs":
                values_arr = np.array([d.diff for d in devs], dtype=float)
            else:
                values_arr = np.array([d.diff_pct for d in devs], dtype=float)

            sort_idx = np.argsort(times_arr)
            times_arr = times_arr[sort_idx]
            values_arr = values_arr[sort_idx]

            self.deviation_plot.plot(
                times_arr, values_arr,
                pen=pg.mkPen(color=color, width=2),
                name=sample["sample_no"],
                symbol='o',
                symbolSize=6,
                symbolBrush=color,
                symbolPen=color
            )

            if threshold_line is None:
                if mode == "abs":
                    pass
                else:
                    threshold_line = 20.0

        zero_pen = pg.mkPen(color='#7f8c8d', width=1, style=Qt.SolidLine)
        self.deviation_plot.plot(
            np.array([0.0, 1.0]), np.array([0.0, 0.0]),
            pen=zero_pen, name="零线"
        )
        view_range = self.deviation_plot.viewRange()
        if view_range and view_range[0]:
            x_max = max(view_range[0][1], np.max(times_arr) if len(times_arr) else 200.0)
            self.deviation_plot.plot(
                np.array([0.0, x_max]), np.array([0.0, 0.0]),
                pen=zero_pen, name=None
            )

        if mode == "pct" and threshold_line:
            th_pen = pg.mkPen(color='#e74c3c', width=1, style=Qt.DashLine)
            self.deviation_plot.plot(
                np.array([0.0, x_max if 'x_max' in locals() else 200.0]),
                np.array([threshold_line, threshold_line]),
                pen=th_pen, name="+阈值(20%)"
            )
            self.deviation_plot.plot(
                np.array([0.0, x_max if 'x_max' in locals() else 200.0]),
                np.array([-threshold_line, -threshold_line]),
                pen=th_pen, name="-阈值(20%)"
            )

    def refresh_samples(self):
        current_selected = self._selected_ids
        self._load_samples()
        for i in range(self.sample_list.count()):
            item = self.sample_list.item(i)
            if item.data(Qt.UserRole) in current_selected:
                item.setSelected(True)


class CsvImportTab(QWidget):
    data_updated = Signal()

    def __init__(self):
        super().__init__()
        self._data_loaded = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        import_group = QGroupBox("CSV 导入")
        import_layout = QVBoxLayout(import_group)

        file_row = QHBoxLayout()
        self.file_edit = QLineEdit()
        self.file_edit.setPlaceholderText("选择要导入的 CSV 文件...")
        browse_btn = QPushButton("浏览...")
        browse_btn.clicked.connect(self._browse_file)
        file_row.addWidget(self.file_edit, 1)
        file_row.addWidget(browse_btn)
        import_layout.addLayout(file_row)

        btn_row = QHBoxLayout()
        self.import_btn = QPushButton("导入")
        self.import_btn.clicked.connect(self._do_import)
        self.import_btn.setEnabled(False)
        self.import_btn.setMinimumWidth(100)
        btn_row.addWidget(self.import_btn)
        btn_row.addStretch()
        import_layout.addLayout(btn_row)

        self.result_text = QTextEdit()
        self.result_text.setReadOnly(True)
        self.result_text.setMaximumHeight(150)
        self.result_text.setPlaceholderText("导入结果将显示在这里...")
        import_layout.addWidget(self.result_text)

        main_layout.addWidget(import_group)

        failures_group = QGroupBox("导入失败记录")
        failures_layout = QVBoxLayout(failures_group)

        self.failures_table = QTableView()
        self.failures_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.failures_table.horizontalHeader().setSectionResizeMode(3, QHeaderView.Interactive)
        self.failures_model = ImportFailureTableModel()
        self.failures_table.setModel(self.failures_model)
        failures_layout.addWidget(self.failures_table, 1)

        btn_row2 = QHBoxLayout()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._load_failures)
        clear_btn = QPushButton("清空失败记录")
        clear_btn.clicked.connect(self._clear_failures)
        btn_row2.addWidget(refresh_btn)
        btn_row2.addStretch()
        btn_row2.addWidget(clear_btn)
        failures_layout.addLayout(btn_row2)

        main_layout.addWidget(failures_group, 1)

    def _browse_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择 CSV 文件", "", "CSV 文件 (*.csv);;所有文件 (*.*)"
        )
        if file_path:
            self.file_edit.setText(file_path)
            self.import_btn.setEnabled(True)

    def _do_import(self):
        file_path = self.file_edit.text().strip()
        if not file_path:
            return

        summary = csv_importer.import_csv(file_path)

        result_lines = [
            f"文件: {summary.source_file}",
            f"总行数: {summary.total_rows}",
            f"试样: 新增 {summary.samples_created}，跳过 {summary.samples_skipped}",
            f"测量: 新增 {summary.measurements_created}，跳过 {summary.measurements_skipped}",
            f"失败: {summary.failures}",
        ]
        result_lines.extend(summary.messages)

        self.result_text.setPlainText("\n".join(result_lines))

        self._load_failures()
        self.data_updated.emit()

        if summary.failures > 0:
            QMessageBox.warning(
                self, "导入完成",
                f"导入完成，但有 {summary.failures} 条记录导入失败\n请查看下方失败记录列表"
            )
        else:
            QMessageBox.information(self, "导入完成", "导入成功！")

    def _load_failures(self):
        failures = db.get_import_failures()
        self.failures_model.update_data(failures)
        self._data_loaded = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._data_loaded:
            self._load_failures()

    def _clear_failures(self):
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有失败记录吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            db.clear_import_failures()
            self._load_failures()


class BaselineTemplateTableModel(QAbstractTableModel):
    HEADERS = ["纸型", "基线试样", "基线点数", "平均斜率", "平均半径", "平均毛糙", "异常率%", "备注", "更新时间"]

    def __init__(self, data: Optional[List] = None):
        super().__init__()
        self._data = data or []

    def update_data(self, data: List):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._data):
            return None
        row = self._data[index.row()]
        col = index.column()
        if role == Qt.DisplayRole:
            if col == 0:
                return row["paper_type"]
            elif col == 1:
                return row.get("sample_no", "-") or "-"
            elif col == 2:
                return str(row.get("point_count", 0))
            elif col == 3:
                return f"{row['avg_slope']:.4f}" if row["avg_slope"] is not None else "-"
            elif col == 4:
                return f"{row['avg_radius']:.2f}" if row["avg_radius"] is not None else "-"
            elif col == 5:
                return f"{row['avg_roughness']:.2f}" if row["avg_roughness"] is not None else "-"
            elif col == 6:
                anomaly_rate = row.get("anomaly_rate_pct")
                return f"{anomaly_rate:.1f}%" if anomaly_rate is not None else "-"
            elif col == 7:
                return row["remark"] if row["remark"] else "-"
            elif col == 8:
                return row["updated_at"]
        if role == Qt.ForegroundRole and col == 6:
            anomaly_rate = row.get("anomaly_rate_pct")
            if anomaly_rate is not None:
                if anomaly_rate >= 30:
                    return QColor("#c0392b")
                elif anomaly_rate >= 15:
                    return QColor("#e67e22")
                elif anomaly_rate >= 5:
                    return QColor("#f1c40f")
        return None


class PaperRiskTableModel(QAbstractTableModel):
    HEADERS = ["纸张类型", "试样总数", "高风险", "中风险", "低风险", "正常", "异常率%", "基线试样数"]

    def __init__(self, data: Optional[List] = None):
        super().__init__()
        self._data = data or []

    def update_data(self, data: List):
        self.beginResetModel()
        self._data = data
        self.endResetModel()

    def rowCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self._data)

    def columnCount(self, parent: QModelIndex = QModelIndex()) -> int:
        return len(self.HEADERS)

    def headerData(self, section: int, orientation: Qt.Orientation, role: int = Qt.DisplayRole):
        if role == Qt.DisplayRole and orientation == Qt.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index: QModelIndex, role: int = Qt.DisplayRole):
        if not index.isValid() or index.row() >= len(self._data):
            return None
        row = self._data[index.row()]
        col = index.column()
        if role == Qt.DisplayRole:
            if col == 0:
                return row["paper_type"]
            elif col == 1:
                return str(row["total_samples"])
            elif col == 2:
                return str(row["high_risk"])
            elif col == 3:
                return str(row["mid_risk"])
            elif col == 4:
                return str(row["low_risk"])
            elif col == 5:
                return str(row["normal"])
            elif col == 6:
                rate = row.get("anomaly_rate_pct")
                return f"{rate:.1f}%" if rate is not None else "-"
            elif col == 7:
                return str(row.get("baseline_count", 0))
        if role == Qt.ForegroundRole and col == 6:
            rate = row.get("anomaly_rate_pct")
            if rate is not None:
                if rate >= 40:
                    return QColor("#c0392b")
                elif rate >= 20:
                    return QColor("#e67e22")
                elif rate >= 5:
                    return QColor("#f1c40f")
                else:
                    return QColor("#27ae60")
        if role == Qt.ForegroundRole and col in (2, 3, 4):
            val = row[self.HEADERS[col].replace("风险", "_risk").lower()] if False else 0
            if col == 2 and row["high_risk"] > 0:
                return QColor("#c0392b")
            if col == 3 and row["mid_risk"] > 0:
                return QColor("#e67e22")
            if col == 4 and row["low_risk"] > 0:
                return QColor("#f1c40f")
        return None


class BaselineTemplateTab(QWidget):
    data_updated = Signal()

    def __init__(self):
        super().__init__()
        self._data_loaded = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        hint_group = QGroupBox("纸型基线模板说明")
        hint_layout = QVBoxLayout(hint_group)
        hint = QLabel(
            "• 每个纸张类型可建立一个基线模板，用于同纸型试样的对照分析\n"
            "• 基线来源：在「试样管理」中选择典型试样，点击「★ 设为纸型基线」\n"
            "• 新增试样时自动匹配同纸型的基线模板，测量数据录入后自动进行偏差分析\n"
            "• 异常判断将增加「偏离纸型基线」作为原因之一"
        )
        hint.setStyleSheet("color: #555; font-size: 12px; padding: 4px;")
        hint_layout.addWidget(hint)
        main_layout.addWidget(hint_group)

        list_group = QGroupBox("基线模板列表（按纸型）")
        list_layout = QVBoxLayout(list_group)

        self.template_table = QTableView()
        self.template_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.template_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.template_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.template_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.template_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeToContents)

        self.template_model = BaselineTemplateTableModel()
        self.template_table.setModel(self.template_model)
        list_layout.addWidget(self.template_table, 1)

        btn_row = QHBoxLayout()
        self.refresh_btn = QPushButton("刷新列表")
        self.refresh_btn.clicked.connect(self._load_templates)
        self.rebuild_btn = QPushButton("重建选中纸型基线")
        self.rebuild_btn.clicked.connect(self._rebuild_selected)
        self.delete_btn = QPushButton("删除选中基线模板")
        self.delete_btn.clicked.connect(self._delete_selected)
        self.rebuild_all_btn = QPushButton("重建所有基线（聚合基线试样）")
        self.rebuild_all_btn.clicked.connect(self._rebuild_all)
        btn_row.addWidget(self.refresh_btn)
        btn_row.addWidget(self.rebuild_btn)
        btn_row.addWidget(self.delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.rebuild_all_btn)
        list_layout.addLayout(btn_row)

        main_layout.addWidget(list_group, 1)

        stats_group = QGroupBox("各纸型异常率统计（用于快速发现风险纸型）")
        stats_layout = QVBoxLayout(stats_group)
        self.paper_risk_table = QTableView()
        self.paper_risk_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.paper_risk_model = PaperRiskTableModel()
        self.paper_risk_table.setModel(self.paper_risk_model)
        stats_layout.addWidget(self.paper_risk_table)

        btn_row2 = QHBoxLayout()
        self.refresh_risk_btn = QPushButton("刷新统计")
        self.refresh_risk_btn.clicked.connect(self._load_risk_stats)
        btn_row2.addWidget(self.refresh_risk_btn)
        btn_row2.addStretch()
        stats_layout.addLayout(btn_row2)

        main_layout.addWidget(stats_group, 1)

    def _load_templates(self):
        templates = db.get_all_baseline_templates()
        rows = []
        for t in templates:
            d = dict(t)
            bl = anomaly_detection.get_paper_type_baseline(t["paper_type"])
            d["point_count"] = len(bl.times) if bl else 0
            samples = db.get_samples_by_paper_type(t["paper_type"])
            ids = [s["id"] for s in samples]
            if ids:
                anom = sum(1 for s in samples if s["risk_flag"] != "正常")
                d["anomaly_rate_pct"] = round(anom / len(samples) * 100, 1) if samples else 0.0
            else:
                d["anomaly_rate_pct"] = None
            rows.append(d)
        self.template_model.update_data(rows)
        self._load_risk_stats()
        self._data_loaded = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._data_loaded:
            self._load_templates()

    def _load_risk_stats(self):
        stats = db.get_paper_type_risk_summary()
        self.paper_risk_model.update_data(stats)

    def _get_selected_template(self):
        idx = self.template_table.currentIndex()
        if not idx.isValid():
            return None
        return self.template_model._data[idx.row()]

    def _rebuild_selected(self):
        t = self._get_selected_template()
        if not t:
            QMessageBox.information(self, "提示", "请先选择一个基线模板")
            return
        paper_type = t["paper_type"]
        baselines = db.get_baselines_by_paper_type(paper_type)
        if not baselines:
            QMessageBox.warning(self, "失败", f"纸型「{paper_type}」下没有标记为基线的试样，请先在试样管理中标记")
            return
        try:
            bl = anomaly_detection.build_aggregated_baseline(paper_type)
            if bl:
                sample_nos = ", ".join(
                    db.get_sample_by_id(sid)["sample_no"]
                    for sid in bl.sample_ids
                    if db.get_sample_by_id(sid)
                )
                QMessageBox.information(
                    self, "成功",
                    f"已重建「{paper_type}」基线模板\n"
                    f"聚合基线试样: {sample_nos}\n"
                    f"平均斜率: {bl.avg_slope:.4f}，平均半径: {bl.avg_radius:.2f}mm"
                )
                self._load_templates()
                self.data_updated.emit()
            else:
                QMessageBox.warning(self, "失败", "重建失败，请检查基线试样的测量数据（需至少3个点）")
        except Exception as e:
            QMessageBox.warning(self, "失败", str(e))

    def _delete_selected(self):
        t = self._get_selected_template()
        if not t:
            QMessageBox.information(self, "提示", "请先选择一个基线模板")
            return
        paper_type = t["paper_type"]
        reply = QMessageBox.question(
            self, "确认",
            f"确定要删除「{paper_type}」的基线模板吗？\n（不会删除试样及其基线标记，仅删除模板数据）",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        try:
            db.delete_baseline_template(t["id"])
            QMessageBox.information(self, "成功", f"已删除「{paper_type}」基线模板")
            self._load_templates()
            self.data_updated.emit()
        except Exception as e:
            QMessageBox.warning(self, "失败", str(e))

    def _rebuild_all(self):
        paper_types = db.get_all_paper_types()
        rebuilt = 0
        failed = 0
        details = []
        for pt in paper_types:
            baselines = db.get_baselines_by_paper_type(pt)
            if not baselines:
                continue
            try:
                bl = anomaly_detection.build_aggregated_baseline(pt)
                if bl:
                    rebuilt += 1
                    details.append(f"「{pt}」: 聚合 {len(bl.sample_ids)} 个基线试样")
                else:
                    failed += 1
                    details.append(f"「{pt}」: 重建失败（数据不足）")
            except Exception as e:
                failed += 1
                details.append(f"「{pt}」: 错误 - {e}")
        self._load_templates()
        self.data_updated.emit()
        msg = f"重建完成：成功 {rebuilt} 个，失败 {failed} 个\n\n" + "\n".join(details)
        QMessageBox.information(self, "完成", msg)


class BatchTab(QWidget):
    def __init__(self):
        super().__init__()
        self._data_loaded = False
        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        form_group = QGroupBox("创建批次")
        form_layout = QFormLayout(form_group)

        self.batch_code_edit = QLineEdit()
        self.batch_code_edit.setPlaceholderText("输入批次号")

        self.remark_edit = QLineEdit()
        self.remark_edit.setPlaceholderText("备注（可选）")

        form_layout.addRow("批次号 *", self.batch_code_edit)
        form_layout.addRow("备注", self.remark_edit)

        btn_row = QHBoxLayout()
        create_btn = QPushButton("创建批次")
        create_btn.clicked.connect(self._create_batch)
        btn_row.addWidget(create_btn)
        btn_row.addStretch()
        form_layout.addRow(btn_row)

        main_layout.addWidget(form_group)

        center_splitter = QSplitter(Qt.Vertical)

        list_group = QGroupBox("批次列表")
        list_layout = QVBoxLayout(list_group)

        self.batch_table = QTableView()
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.batch_model = BatchTableModel()
        self.batch_table.setModel(self.batch_model)
        self.batch_table.selectionModel().selectionChanged.connect(self._on_batch_selected)
        list_layout.addWidget(self.batch_table, 1)

        btn_row2 = QHBoxLayout()
        refresh_btn = QPushButton("刷新")
        refresh_btn.clicked.connect(self._load_batches)
        reassess_btn = QPushButton("重新评估选中批次")
        reassess_btn.clicked.connect(self._reassess_batch)
        btn_row2.addWidget(refresh_btn)
        btn_row2.addStretch()
        btn_row2.addWidget(reassess_btn)
        list_layout.addLayout(btn_row2)

        center_splitter.addWidget(list_group)

        detail_group = QGroupBox("批次详情 & 按纸型聚合")
        detail_container = QWidget()
        detail_splitter = QSplitter(Qt.Horizontal, detail_container)

        left_detail = QWidget()
        left_layout = QVBoxLayout(left_detail)
        left_layout.setContentsMargins(0, 0, 0, 0)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        left_layout.addWidget(QLabel("批次详情："))
        left_layout.addWidget(self.detail_text, 1)

        right_detail = QWidget()
        right_layout = QVBoxLayout(right_detail)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(QLabel("按纸型聚合风险统计（选中批次）："))
        self.paper_agg_table = QTableView()
        self.paper_agg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.paper_agg_model = PaperRiskTableModel()
        self.paper_agg_table.setModel(self.paper_agg_model)
        right_layout.addWidget(self.paper_agg_table, 1)

        refresh_agg_btn = QPushButton("刷新聚合统计")
        refresh_agg_btn.clicked.connect(self._refresh_aggregation)
        right_layout.addWidget(refresh_agg_btn)

        detail_splitter.addWidget(left_detail)
        detail_splitter.addWidget(right_detail)
        detail_splitter.setStretchFactor(0, 1)
        detail_splitter.setStretchFactor(1, 1)

        inner_layout = QVBoxLayout(detail_group)
        inner_layout.addWidget(detail_container)

        center_splitter.addWidget(detail_group)
        center_splitter.setStretchFactor(0, 1)
        center_splitter.setStretchFactor(1, 1)

        main_layout.addWidget(center_splitter, 1)

        global_agg_group = QGroupBox("全局按纸型聚合风险统计（所有批次）")
        global_agg_layout = QVBoxLayout(global_agg_group)
        self.global_paper_agg_table = QTableView()
        self.global_paper_agg_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.global_paper_agg_model = PaperRiskTableModel()
        self.global_paper_agg_table.setModel(self.global_paper_agg_model)
        global_agg_layout.addWidget(self.global_paper_agg_table, 1)

        global_btn_row = QHBoxLayout()
        refresh_global_btn = QPushButton("刷新全局统计")
        refresh_global_btn.clicked.connect(self._refresh_global_aggregation)
        global_btn_row.addWidget(refresh_global_btn)
        global_btn_row.addStretch()
        global_agg_layout.addWidget(global_btn_row)

        main_layout.addWidget(global_agg_group, 1)

    def _load_batches(self):
        batches = db.get_all_batches()
        self.batch_model.update_data(batches)
        self._refresh_global_aggregation()
        self._data_loaded = True

    def showEvent(self, event):
        super().showEvent(event)
        if not self._data_loaded:
            self._load_batches()

    def _create_batch(self):
        batch_code = self.batch_code_edit.text().strip()
        remark = self.remark_edit.text().strip()

        if not batch_code:
            QMessageBox.warning(self, "提示", "请输入批次号")
            return

        existing = db.get_batch_by_code(batch_code)
        if existing:
            QMessageBox.warning(self, "提示", f"批次号 '{batch_code}' 已存在")
            return

        try:
            db.create_batch(batch_code, remark)
            QMessageBox.information(self, "成功", f"批次 '{batch_code}' 创建成功")
            self.batch_code_edit.clear()
            self.remark_edit.clear()
            self._load_batches()
        except Exception as e:
            QMessageBox.warning(self, "创建失败", str(e))

    def _get_selected_batch(self):
        idx = self.batch_table.currentIndex()
        if not idx.isValid():
            return None
        return self.batch_model._data[idx.row()]

    def _on_batch_selected(self):
        row = self._get_selected_batch()
        if not row:
            self.detail_text.setPlainText("")
            self.paper_agg_model.update_data([])
            return

        batch_id = row["id"]

        samples = db.get_samples_by_batch(batch_id)
        all_sample_ids = [s["id"] for s in samples]
        measurements_map = db.get_measurements_by_samples(all_sample_ids)

        total_measurements = sum(len(m) for m in measurements_map.values())
        total_anomalies = sum(
            1 for ms in measurements_map.values() for m in ms if m["is_anomaly"]
        )

        detail_lines = [
            f"批次号: {row['batch_code']}",
            f"风险等级: {row['risk_level']}",
            f"连续异常: {row['consecutive_anomalies']} 个点",
            f"试样数量: {len(samples)} 个",
            f"测量点数: {total_measurements} 个",
            f"异常点数: {total_anomalies} 个",
        ]
        if row["remark"]:
            detail_lines.append(f"备注: {row['remark']}")

        if samples:
            detail_lines.append("")
            detail_lines.append("包含试样:")
            for s in samples:
                judge = s["judgment"] if s["judgment"] else "未分析"
                risk = s["risk_flag"]
                baseline_mark = " ★基线" if s.get("is_baseline", 0) else ""
                detail_lines.append(f"  • {s['sample_no']} - {s['paper_type']}{baseline_mark} | {judge} | {risk}")

        self.detail_text.setPlainText("\n".join(detail_lines))

        agg = db.get_risk_aggregated_by_paper(batch_id)
        for a in agg:
            total = a.get("total_samples", 0)
            anom = a.get("high_risk", 0) + a.get("mid_risk", 0) + a.get("low_risk", 0)
            a["anomaly_rate_pct"] = round(anom / total * 100, 1) if total > 0 else 0.0
        agg.sort(key=lambda x: x.get("anomaly_rate_pct", 0), reverse=True)
        self.paper_agg_model.update_data(agg)

    def _refresh_aggregation(self):
        self._on_batch_selected()

    def _refresh_global_aggregation(self):
        stats = db.get_paper_type_risk_summary()
        self.global_paper_agg_model.update_data(stats)

    def _reassess_batch(self):
        row = self._get_selected_batch()
        if not row:
            QMessageBox.information(self, "提示", "请选择一个批次")
            return

        batch_id = row["id"]

        samples = db.get_samples_by_batch(batch_id)
        for s in samples:
            anomaly_detection.process_measurement_and_update_risk(s["id"])

        risk_level, consecutive = anomaly_detection.update_batch_risk_by_anomalies(batch_id)

        self._load_batches()
        self._on_batch_selected()
        QMessageBox.information(
            self, "评估完成",
            f"批次 '{row['batch_code']}' 重新评估完成\n风险等级: {risk_level}\n连续异常: {consecutive} 点"
        )


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("古籍修复墨迹试样分析系统")
        self.resize(1400, 900)

        self._init_ui()
        self._init_db()
        self._apply_styles()
        self._load_initial_data()

    def _init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.sample_tab = SampleManagementTab()
        self.measurement_tab = MeasurementTab()
        self.comparison_tab = ComparisonTab()
        self.baseline_tab = BaselineTemplateTab()
        self.csv_tab = CsvImportTab()
        self.batch_tab = BatchTab()

        self.tabs.addTab(self.sample_tab, "试样管理")
        self.tabs.addTab(self.measurement_tab, "测量录入")
        self.tabs.addTab(self.comparison_tab, "渗化曲线对比")
        self.tabs.addTab(self.baseline_tab, "纸型基线模板")
        self.tabs.addTab(self.csv_tab, "CSV 导入")
        self.tabs.addTab(self.batch_tab, "批次&纸型风险")

        self._connect_signals()

    def _connect_signals(self):
        self.sample_tab.data_updated.connect(self.measurement_tab.refresh_samples)
        self.sample_tab.data_updated.connect(self.comparison_tab.refresh_samples)
        self.sample_tab.data_updated.connect(self.baseline_tab._load_templates)
        self.sample_tab.data_updated.connect(self.batch_tab._load_batches)

        self.measurement_tab.data_updated.connect(self.sample_tab._load_samples)
        self.measurement_tab.data_updated.connect(self.comparison_tab.refresh_samples)
        self.measurement_tab.data_updated.connect(self.baseline_tab._load_templates)
        self.measurement_tab.data_updated.connect(self.batch_tab._load_batches)

        self.baseline_tab.data_updated.connect(self.sample_tab._load_samples)
        self.baseline_tab.data_updated.connect(self.measurement_tab.refresh_samples)
        self.baseline_tab.data_updated.connect(self.comparison_tab.refresh_samples)
        self.baseline_tab.data_updated.connect(self.batch_tab._load_batches)

        self.csv_tab.data_updated.connect(self.sample_tab._load_samples)
        self.csv_tab.data_updated.connect(self.measurement_tab.refresh_samples)
        self.csv_tab.data_updated.connect(self.comparison_tab.refresh_samples)
        self.csv_tab.data_updated.connect(self.baseline_tab._load_templates)
        self.csv_tab.data_updated.connect(self.batch_tab._load_batches)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("就绪")

    def _init_db(self):
        try:
            db.init_db()
            self.status_bar.showMessage("数据库初始化完成")
        except Exception as e:
            QMessageBox.critical(self, "数据库错误", f"数据库初始化失败: {e}")

    def _load_initial_data(self):
        self.sample_tab._load_samples()

    def _apply_styles(self):
        style = """
        QMainWindow {
            background-color: #f8f9fa;
        }
        QTabWidget::pane {
            border: 1px solid #dee2e6;
            background: white;
        }
        QTabBar::tab {
            background: #e9ecef;
            padding: 8px 20px;
            border: 1px solid #dee2e6;
            border-bottom: none;
            margin-right: 2px;
        }
        QTabBar::tab:selected {
            background: white;
            border-bottom: 2px solid #3498db;
        }
        QGroupBox {
            font-weight: bold;
            border: 1px solid #dee2e6;
            border-radius: 5px;
            margin-top: 10px;
            padding-top: 10px;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            left: 10px;
            padding: 0 5px;
        }
        QPushButton {
            background-color: #3498db;
            color: white;
            border: none;
            padding: 8px 16px;
            border-radius: 4px;
            min-width: 80px;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:disabled {
            background-color: #95a5a6;
        }
        QLineEdit, QComboBox, QDateEdit, QDoubleSpinBox {
            padding: 6px;
            border: 1px solid #ced4da;
            border-radius: 4px;
            min-height: 20px;
        }
        QLineEdit:focus, QComboBox:focus, QDateEdit:focus, QDoubleSpinBox:focus {
            border: 1px solid #3498db;
        }
        QTableView {
            gridline-color: #dee2e6;
            selection-background-color: #3498db;
        }
        QHeaderView::section {
            background-color: #e9ecef;
            padding: 6px;
            border: 1px solid #dee2e6;
            font-weight: bold;
        }
        """
        self.setStyleSheet(style)


def main():
    app = QApplication(sys.argv)
    app.setApplicationName("古籍修复墨迹试样分析系统")

    font = QFont("Microsoft YaHei", 9)
    app.setFont(font)

    window = MainWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
