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
    HEADERS = ["编号", "试样编号", "纸张类型", "点墨日期", "批次", "风险", "判断", "创建时间"]

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
                return row["risk_flag"]
            elif col == 6:
                return row["judgment"] if row["judgment"] else "-"
            elif col == 7:
                return row["created_at"]

        if role == Qt.ForegroundRole and col == 5:
            risk = row["risk_flag"]
            if risk == "高风险":
                return QColor("#c0392b")
            elif risk == "中风险":
                return QColor("#e67e22")
            elif risk == "低风险":
                return QColor("#f1c40f")
            return QColor("#27ae60")

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
        self._init_ui()
        self._load_samples()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)

        form_group = QGroupBox("录入试样")
        form_layout = QFormLayout(form_group)

        self.sample_no_edit = QLineEdit()
        self.sample_no_edit.setPlaceholderText("请输入试样编号（必须唯一）")

        self.paper_type_edit = QLineEdit()
        self.paper_type_edit.setPlaceholderText("如：宣纸、毛边纸、皮纸")

        self.ink_date_edit = QDateEdit()
        self.ink_date_edit.setCalendarPopup(True)
        self.ink_date_edit.setDisplayFormat("yyyy-MM-dd")
        self.ink_date_edit.setDate(datetime.now())

        self.batch_code_edit = QLineEdit()
        self.batch_code_edit.setPlaceholderText("可选，批次号将自动关联")

        form_layout.addRow("试样编号 *", self.sample_no_edit)
        form_layout.addRow("纸张类型 *", self.paper_type_edit)
        form_layout.addRow("点墨日期 *", self.ink_date_edit)
        form_layout.addRow("批次号", self.batch_code_edit)

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
        self.refresh_btn = QPushButton("刷新")
        self.refresh_btn.clicked.connect(self._load_samples)
        btn_row.addWidget(self.delete_btn)
        btn_row.addStretch()
        btn_row.addWidget(self.refresh_btn)
        list_layout.addLayout(btn_row)

        main_layout.addWidget(list_group, 1)

    def _load_samples(self):
        samples = db.get_all_samples()
        self.sample_model.update_data(samples)
        self.data_updated.emit()

    def _add_sample(self):
        sample_no = self.sample_no_edit.text().strip()
        paper_type = self.paper_type_edit.text().strip()
        ink_date = self.ink_date_edit.date().toString("yyyy-MM-dd")
        batch_code = self.batch_code_edit.text().strip() or None

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
            db.create_sample(sample_no, paper_type, ink_date, batch_id)
            QMessageBox.information(self, "成功", f"试样 {sample_no} 添加成功！")
            self._clear_form()
            self._load_samples()
        except Exception as e:
            QMessageBox.warning(self, "添加失败", str(e))

    def _clear_form(self):
        self.sample_no_edit.clear()
        self.paper_type_edit.clear()
        self.ink_date_edit.setDate(datetime.now())
        self.batch_code_edit.clear()

    def _delete_sample(self):
        proxy_idx = self.sample_table.currentIndex()
        if not proxy_idx.isValid():
            QMessageBox.information(self, "提示", "请先选择一个试样")
            return

        source_idx = self.filter_proxy.mapToSource(proxy_idx)
        row = self.sample_model._data[source_idx.row()]
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
        self._init_ui()
        self._load_samples()

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
        self._init_ui()
        self._load_samples()

    def _init_ui(self):
        main_layout = QHBoxLayout(self)

        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(300)

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

        options_layout.addWidget(self.show_anomaly_cb)
        options_layout.addWidget(self.show_legend_cb)
        options_layout.addWidget(self.show_grid_cb)
        options_layout.addWidget(self.show_details_cb)

        left_layout.addWidget(options_group)
        left_layout.addStretch()

        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)

        self.right_splitter = QSplitter(Qt.Vertical)

        merge_group = QGroupBox("合并渗化曲线")
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
        self.right_splitter.addWidget(merge_group)

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
        self.right_splitter.setStretchFactor(0, 3)
        self.right_splitter.setStretchFactor(1, 2)

        right_layout.addWidget(self.right_splitter)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(1, 5)
        main_layout.addWidget(splitter)

    def _load_samples(self):
        self.sample_list.clear()
        self._plot_curves.clear()
        self._plot_scatters.clear()
        self.plot_widget.clear()
        self._clear_detail_cards()

        samples = db.get_all_samples()
        self._all_samples = {s["id"]: s for s in samples}

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6',
                  '#1abc9c', '#e67e22', '#34495e', '#e91e63', '#00bcd4',
                  '#ff9800', '#8bc34a', '#ff5722', '#607d8b', '#795548']

        for i, s in enumerate(samples):
            item = QListWidgetItem(f"{s['sample_no']} ({s['paper_type']})")
            item.setData(Qt.UserRole, s["id"])
            color = QColor(colors[i % len(colors)])
            item.setData(Qt.ForegroundRole, color)
            self.sample_list.addItem(item)
            self._plot_curves[s["id"]] = color

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

    def _on_details_toggled(self):
        self.detail_group.setVisible(self.show_details_cb.isChecked())

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

    def _update_plot(self):
        self.plot_widget.clear()
        if self.show_legend_cb.isChecked():
            self.plot_widget.addLegend()

        if not self._selected_ids:
            return

        all_measurements = db.get_measurements_by_samples(self._selected_ids)

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

            self.plot_widget.plot(
                times, radii,
                pen=pg.mkPen(color=color, width=2),
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
        self._init_ui()
        self._load_failures()

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

    def _clear_failures(self):
        reply = QMessageBox.question(
            self, "确认", "确定要清空所有失败记录吗？",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            db.clear_import_failures()
            self._load_failures()


class BatchTab(QWidget):
    def __init__(self):
        super().__init__()
        self._init_ui()
        self._load_batches()

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

        list_group = QGroupBox("批次列表")
        list_layout = QVBoxLayout(list_group)

        self.batch_table = QTableView()
        self.batch_table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.batch_table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.batch_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.batch_table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeToContents)
        self.batch_table.selectionModel().selectionChanged.connect(self._on_batch_selected)
        self.batch_model = BatchTableModel()
        self.batch_table.setModel(self.batch_model)
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

        main_layout.addWidget(list_group, 1)

        detail_group = QGroupBox("批次详情")
        detail_layout = QVBoxLayout(detail_group)
        self.detail_text = QTextEdit()
        self.detail_text.setReadOnly(True)
        self.detail_text.setMaximumHeight(180)
        detail_layout.addWidget(self.detail_text)
        main_layout.addWidget(detail_group)

    def _load_batches(self):
        batches = db.get_all_batches()
        self.batch_model.update_data(batches)

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

    def _on_batch_selected(self):
        idx = self.batch_table.currentIndex()
        if not idx.isValid():
            self.detail_text.setPlainText("")
            return

        row = self.batch_model._data[idx.row()]
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
                detail_lines.append(f"  • {s['sample_no']} - {s['paper_type']} | {judge} | {risk}")

        self.detail_text.setPlainText("\n".join(detail_lines))

    def _reassess_batch(self):
        idx = self.batch_table.currentIndex()
        if not idx.isValid():
            QMessageBox.information(self, "提示", "请选择一个批次")
            return

        row = self.batch_model._data[idx.row()]
        batch_id = row["id"]

        samples = db.get_samples_by_batch(batch_id)
        for s in samples:
            anomaly_detection.process_measurement_and_update_risk(s["id"])

        risk_level, consecutive = anomaly_detection.update_batch_risk_by_anomalies(batch_id)

        self._load_batches()
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

    def _init_ui(self):
        self.tabs = QTabWidget()
        self.setCentralWidget(self.tabs)

        self.sample_tab = SampleManagementTab()
        self.measurement_tab = MeasurementTab()
        self.comparison_tab = ComparisonTab()
        self.csv_tab = CsvImportTab()
        self.batch_tab = BatchTab()

        self.tabs.addTab(self.sample_tab, "试样管理")
        self.tabs.addTab(self.measurement_tab, "测量录入")
        self.tabs.addTab(self.comparison_tab, "渗化曲线对比")
        self.tabs.addTab(self.csv_tab, "CSV 导入")
        self.tabs.addTab(self.batch_tab, "批次管理")

        self.sample_tab.data_updated.connect(self.measurement_tab.refresh_samples)
        self.sample_tab.data_updated.connect(self.comparison_tab.refresh_samples)
        self.sample_tab.data_updated.connect(self.batch_tab._load_batches)

        self.measurement_tab.data_updated.connect(self.sample_tab._load_samples)
        self.measurement_tab.data_updated.connect(self.comparison_tab.refresh_samples)
        self.measurement_tab.data_updated.connect(self.batch_tab._load_batches)

        self.csv_tab.data_updated.connect(self.sample_tab._load_samples)
        self.csv_tab.data_updated.connect(self.measurement_tab.refresh_samples)
        self.csv_tab.data_updated.connect(self.comparison_tab.refresh_samples)
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
