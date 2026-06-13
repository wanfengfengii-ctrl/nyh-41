import csv
import os
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

import db
import models


SAMPLE_COLUMN_ALIASES = {
    "sample_no": ["试样编号", "编号", "sample_no", "sample", "sampleid", "sample_id"],
    "paper_type": ["纸张类型", "纸型", "paper_type", "paper", "papertype"],
    "ink_date": ["点墨日期", "日期", "ink_date", "date", "inkdate"],
    "batch_code": ["批次号", "批次", "batch_code", "batch", "batchcode"],
}

MEASUREMENT_COLUMN_ALIASES = {
    "sample_no": ["试样编号", "编号", "sample_no", "sample", "sampleid", "sample_id"],
    "adsorb_time": ["吸附时间", "时间", "adsorb_time", "time", "adsorbtime", "t"],
    "radius": ["扩散半径", "半径", "radius", "r"],
    "roughness": ["边缘毛糙度", "毛糙度", "粗糙度", "roughness", "rough"],
}


@dataclass
class ImportSummary:
    source_file: str
    total_rows: int = 0
    samples_created: int = 0
    samples_skipped: int = 0
    measurements_created: int = 0
    measurements_skipped: int = 0
    failures: int = 0
    messages: List[str] = field(default_factory=list)


def _normalize_header(header: str) -> str:
    return header.strip().lower().replace(" ", "").replace("_", "")


def _detect_columns(headers: List[str], alias_map: Dict[str, List[str]]) -> Dict[str, Optional[int]]:
    normalized_headers = {_normalize_header(h): i for i, h in enumerate(headers)}
    result: Dict[str, Optional[int]] = {}
    for col_name, aliases in alias_map.items():
        idx: Optional[int] = None
        for alias in aliases:
            key = _normalize_header(alias)
            if key in normalized_headers:
                idx = normalized_headers[key]
                break
        result[col_name] = idx
    return result


def _get_value(row: List[str], idx: Optional[int]) -> str:
    if idx is None or idx >= len(row):
        return ""
    val = row[idx]
    return val.strip() if val else ""


def _detect_file_mode(headers: List[str]) -> str:
    normalized = [_normalize_header(h) for h in headers]
    measurement_keys = {_normalize_header(a) for aliases in MEASUREMENT_COLUMN_ALIASES.values() for a in aliases}
    sample_keys = {_normalize_header(a) for aliases in SAMPLE_COLUMN_ALIASES.values() for a in aliases}

    has_measurement = any(k in measurement_keys for k in normalized) and (
        _normalize_header("吸附时间") in normalized
        or _normalize_header("时间") in normalized
        or _normalize_header("adsorb_time") in normalized
        or _normalize_header("radius") in normalized
        or _normalize_header("扩散半径") in normalized
    )
    has_sample = _normalize_header("纸张类型") in normalized or _normalize_header("paper_type") in normalized

    if has_measurement and has_sample:
        return "mixed"
    elif has_measurement:
        return "measurement"
    else:
        return "sample"


def import_csv(file_path: str) -> ImportSummary:
    summary = ImportSummary(source_file=os.path.basename(file_path))

    if not os.path.isfile(file_path):
        summary.failures += 1
        summary.messages.append(f"文件不存在: {file_path}")
        return summary

    try:
        with open(file_path, "r", encoding="utf-8-sig", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except UnicodeDecodeError:
        with open(file_path, "r", encoding="gbk", newline="") as f:
            reader = csv.reader(f)
            rows = list(reader)
    except Exception as e:
        summary.failures += 1
        summary.messages.append(f"读取文件失败: {e}")
        return summary

    if not rows:
        summary.messages.append("CSV 文件为空")
        return summary

    headers = rows[0]
    data_rows = rows[1:]
    summary.total_rows = len(data_rows)

    mode = _detect_file_mode(headers)

    if mode in ("sample", "mixed"):
        sample_cols = _detect_columns(headers, SAMPLE_COLUMN_ALIASES)
        sample_cache: Dict[str, int] = {}
        for i, row in enumerate(data_rows, start=2):
            sample_no = _get_value(row, sample_cols.get("sample_no"))
            if not sample_no:
                if mode == "mixed":
                    pass
                else:
                    raw = ",".join(row)
                    reason = "试样编号为空"
                    db.record_import_failure(summary.source_file, i, raw, reason)
                    summary.failures += 1
                continue

            if db.sample_no_exists(sample_no):
                existing = db.get_sample_by_no(sample_no)
                if existing:
                    sample_cache[sample_no] = existing["id"]
                summary.samples_skipped += 1
                continue

            paper_type = _get_value(row, sample_cols.get("paper_type"))
            ink_date_raw = _get_value(row, sample_cols.get("ink_date"))
            batch_code_raw = _get_value(row, sample_cols.get("batch_code"))
            ink_date = models.normalize_ink_date(ink_date_raw) if ink_date_raw else ""

            sd = models.SampleData(
                sample_no=sample_no,
                paper_type=paper_type,
                ink_date=ink_date,
                batch_code=batch_code_raw or None,
            )
            vr = models.validate_sample(sd)
            if not vr:
                raw = ",".join(row)
                reason = models.errors_to_text(vr.errors)
                db.record_import_failure(summary.source_file, i, raw, reason)
                summary.failures += 1
                continue

            batch_id = None
            if sd.batch_code:
                batch_row = db.get_batch_by_code(sd.batch_code)
                if batch_row:
                    batch_id = batch_row["id"]
                else:
                    try:
                        batch_id = db.create_batch(sd.batch_code)
                    except Exception:
                        batch_id = None

            try:
                sid = db.create_sample(sd.sample_no.strip(), sd.paper_type.strip(), sd.ink_date, batch_id)
                sample_cache[sample_no] = sid
                summary.samples_created += 1
            except Exception as e:
                raw = ",".join(row)
                reason = f"数据库写入失败: {e}"
                db.record_import_failure(summary.source_file, i, raw, reason)
                summary.failures += 1

    if mode in ("measurement", "mixed"):
        meas_cols = _detect_columns(headers, MEASUREMENT_COLUMN_ALIASES)
        sample_id_cache: Dict[str, int] = {}
        samples = db.get_all_samples()
        for s in samples:
            sample_id_cache[s["sample_no"]] = s["id"]

        for i, row in enumerate(data_rows, start=2):
            sample_no = _get_value(row, meas_cols.get("sample_no"))
            adsorb_time_raw = _get_value(row, meas_cols.get("adsorb_time"))
            radius_raw = _get_value(row, meas_cols.get("radius"))
            roughness_raw = _get_value(row, meas_cols.get("roughness"))

            if adsorb_time_raw == "" and radius_raw == "" and roughness_raw == "":
                continue

            md = models.MeasurementData(
                sample_no=sample_no,
                adsorb_time=adsorb_time_raw,
                radius=radius_raw,
                roughness=roughness_raw,
            )
            vr, sample_id = models.validate_measurement(md, sample_id_cache)
            if not vr:
                raw = ",".join(row)
                reason = models.errors_to_text(vr.errors)
                db.record_import_failure(summary.source_file, i, raw, reason)
                summary.failures += 1
                continue

            try:
                db.create_measurement(
                    sample_id,
                    float(md.adsorb_time),
                    float(md.radius),
                    float(md.roughness),
                )
                summary.measurements_created += 1
                if sample_id not in sample_id_cache and sample_no:
                    sample_id_cache[sample_no] = sample_id
            except Exception as e:
                raw = ",".join(row)
                reason = f"数据库写入失败: {e}"
                db.record_import_failure(summary.source_file, i, raw, reason)
                summary.failures += 1

    summary.messages.append(f"导入模式: {mode}")
    summary.messages.append(
        f"试样: 新增 {summary.samples_created}，跳过 {summary.samples_skipped}"
    )
    summary.messages.append(
        f"测量: 新增 {summary.measurements_created}，跳过 {summary.measurements_skipped}"
    )
    summary.messages.append(f"失败记录: {summary.failures}")

    return summary
