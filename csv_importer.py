import csv
import os
from typing import List, Dict, Tuple, Any, Optional
from dataclasses import dataclass, field

import db
import models
import anomaly_detection


SAMPLE_COLUMN_ALIASES = {
    "sample_no": ["试样编号", "编号", "sample_no", "sample", "sampleid", "sample_id"],
    "paper_type": ["纸张类型", "纸型", "paper_type", "paper", "papertype"],
    "ink_date": ["点墨日期", "日期", "ink_date", "date", "inkdate"],
    "batch_code": ["批次号", "批次", "batch_code", "batch", "batchcode"],
    "is_baseline": ["是否基线", "基线标记", "基线", "is_baseline", "baseline", "isbaseline"],
}

MEASUREMENT_COLUMN_ALIASES = {
    "sample_no": ["试样编号", "编号", "sample_no", "sample", "sampleid", "sample_id"],
    "adsorb_time": ["吸附时间", "时间", "adsorb_time", "time", "adsorbtime", "t"],
    "radius": ["扩散半径", "半径", "radius", "r"],
    "roughness": ["边缘毛糙度", "毛糙度", "粗糙度", "roughness", "rough"],
}

PRESCRIPTION_COLUMN_ALIASES = {
    "sample_no": ["试样编号", "编号", "原试样编号", "sample_no", "sample", "sampleid", "sample_id"],
    "dilution_ratio": ["稀释比例", "稀释比", "稀释浓度", "dilution_ratio", "dilution", "dilutionratio"],
    "ink_amount": ["点墨量", "施墨量", "墨量", "ink_amount", "inkamount", "ink"],
    "environment": ["处理环境", "环境条件", "环境", "温度湿度", "environment", "env"],
    "retest_time": ["复测时间", "检测时点", "复测时点", "retest_time", "retesttime"],
    "observation_focus": ["观察重点", "观察要点", "注意事项", "observation_focus", "observationfocus"],
    "remark": ["备注", "说明", "注释", "remark", "note", "comment"],
    "source": ["处方来源", "来源", "source"],
    "confidence_score": ["置信度", "置信分数", "confidence_score", "confidence", "score"],
}

EXPERIMENT_COLUMN_ALIASES = {
    "sample_no": ["试样编号", "编号", "复测试样编号", "sample_no", "sample", "sampleid", "sample_id"],
    "prescription_id": ["处方编号", "处方ID", "prescription_id", "prescriptionid", "prescid", "presc_id"],
    "retest_sample_no": ["复测试样", "复测编号", "复测", "retest_sample_no", "retestno", "retest_sampleno"],
    "execute_date": ["执行日期", "实验日期", "操作日期", "execute_date", "executedate", "date"],
    "diffusion_improved": ["是否改善扩散", "扩散改善", "扩散是否改善", "diffusion_improved", "diffusionimproved", "diff_improved"],
    "anomaly_reduced": ["是否降低异常", "异常降低", "异常是否减少", "anomaly_reduced", "anomalyreduced", "anom_reduced"],
    "paper_judgment_affected": ["是否影响纸性", "影响纸性", "纸性是否受影响", "paper_judgment_affected", "paperaffected", "paper_affected"],
    "effect_rating": ["效果评级", "效果", "评级", "effect_rating", "effectrating", "rating"],
    "operator": ["操作员", "操作人", "执行者", "operator", "user", "executor"],
    "remark": ["实验备注", "实验说明", "备注", "remark", "note", "comment"],
    "pre_risk_flag": ["修复前风险", "前风险", "pre_risk", "pre_risk_flag", "prerisk"],
    "post_risk_flag": ["修复后风险", "后风险", "post_risk", "post_risk_flag", "postrisk"],
    "pre_anomaly_ratio": ["修复前异常比例", "前异常率", "pre_anomaly", "pre_anomaly_ratio", "preanomalyratio"],
    "post_anomaly_ratio": ["修复后异常比例", "后异常率", "post_anomaly", "post_anomaly_ratio", "postanomalyratio"],
}


@dataclass
class ImportSummary:
    source_file: str
    total_rows: int = 0
    samples_created: int = 0
    samples_skipped: int = 0
    measurements_created: int = 0
    measurements_skipped: int = 0
    prescriptions_created: int = 0
    prescriptions_skipped: int = 0
    experiment_records_created: int = 0
    experiment_records_skipped: int = 0
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
    presc_keys = {_normalize_header(a) for aliases in PRESCRIPTION_COLUMN_ALIASES.values() for a in aliases}
    exp_keys = {_normalize_header(a) for aliases in EXPERIMENT_COLUMN_ALIASES.values() for a in aliases}

    has_measurement = any(k in measurement_keys for k in normalized) and (
        _normalize_header("吸附时间") in normalized
        or _normalize_header("时间") in normalized
        or _normalize_header("adsorb_time") in normalized
        or _normalize_header("radius") in normalized
        or _normalize_header("扩散半径") in normalized
    )
    has_sample = _normalize_header("纸张类型") in normalized or _normalize_header("paper_type") in normalized
    has_prescription = (
        _normalize_header("稀释比例") in normalized
        or _normalize_header("dilution_ratio") in normalized
        or _normalize_header("点墨量") in normalized
        or _normalize_header("ink_amount") in normalized
        or _normalize_header("处理环境") in normalized
        or _normalize_header("environment") in normalized
        or _normalize_header("观察重点") in normalized
        or _normalize_header("observation_focus") in normalized
    )
    has_experiment = (
        _normalize_header("效果评级") in normalized
        or _normalize_header("effect_rating") in normalized
        or _normalize_header("是否改善扩散") in normalized
        or _normalize_header("diffusion_improved") in normalized
        or _normalize_header("是否降低异常") in normalized
        or _normalize_header("anomaly_reduced") in normalized
        or _normalize_header("是否影响纸性") in normalized
        or _normalize_header("paper_judgment_affected") in normalized
    )

    if has_experiment:
        return "experiment"
    elif has_prescription:
        return "prescription"
    elif has_measurement and has_sample:
        return "mixed"
    elif has_measurement:
        return "measurement"
    else:
        return "sample"


def _parse_is_baseline(raw: str) -> int:
    if not raw:
        return 0
    r = raw.strip().lower()
    if r in ("1", "true", "yes", "y", "是", "基线", "标记"):
        return 1
    return 0


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
        baseline_sample_ids_created: List[int] = []
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
            is_baseline_raw = _get_value(row, sample_cols.get("is_baseline"))
            ink_date = models.normalize_ink_date(ink_date_raw) if ink_date_raw else ""
            is_baseline = _parse_is_baseline(is_baseline_raw)

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
                sid = db.create_sample(
                    sd.sample_no.strip(), sd.paper_type.strip(), sd.ink_date, batch_id,
                    is_baseline=is_baseline
                )
                sample_cache[sample_no] = sid
                summary.samples_created += 1
                if is_baseline:
                    baseline_sample_ids_created.append(sid)
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
            if not sample_no:
                if mode == "mixed":
                    continue
                raw = ",".join(row)
                db.record_import_failure(summary.source_file, i, raw, "试样编号为空")
                summary.failures += 1
                continue

            if sample_id_cache.get(sample_no) is None:
                existing = db.get_sample_by_no(sample_no)
                if existing:
                    sample_id_cache[sample_no] = existing["id"]

            sample_id = sample_id_cache.get(sample_no)
            if sample_id is None:
                raw = ",".join(row)
                db.record_import_failure(summary.source_file, i, raw,
                                         f"试样编号 '{sample_no}' 不存在")
                summary.failures += 1
                continue

            try:
                adsorb_time = float(adsorb_time_raw)
                radius = float(radius_raw)
                roughness = float(roughness_raw) if roughness_raw != "" else 0.0
            except (ValueError, TypeError):
                raw = ",".join(row)
                db.record_import_failure(summary.source_file, i, raw,
                                         "吸附时间/半径/毛糙度必须是数字")
                summary.failures += 1
                continue

            if adsorb_time <= 0:
                raw = ",".join(row)
                db.record_import_failure(summary.source_file, i, raw,
                                         f"吸附时间必须大于 0，当前值: {adsorb_time}")
                summary.failures += 1
                continue
            if radius <= 0:
                raw = ",".join(row)
                db.record_import_failure(summary.source_file, i, raw,
                                         f"扩散半径必须大于 0，当前值: {radius}")
                summary.failures += 1
                continue
            if roughness < 0:
                raw = ",".join(row)
                db.record_import_failure(summary.source_file, i, raw,
                                         f"边缘毛糙度不能为负，当前值: {roughness}")
                summary.failures += 1
                continue

            if db.measurement_exists(sample_id, adsorb_time):
                summary.measurements_skipped += 1
                continue

            try:
                db.create_measurement(sample_id, adsorb_time, radius, roughness)
                summary.measurements_created += 1
            except Exception as e:
                raw = ",".join(row)
                reason = f"数据库写入失败: {e}"
                db.record_import_failure(summary.source_file, i, raw, reason)
                summary.failures += 1

    if mode == "prescription":
        presc_cols = _detect_columns(headers, PRESCRIPTION_COLUMN_ALIASES)
        sample_id_cache: Dict[str, int] = {}
        samples = db.get_all_samples()
        for s in samples:
            sample_id_cache[s["sample_no"]] = s["id"]

        for i, row in enumerate(data_rows, start=2):
            sample_no = _get_value(row, presc_cols.get("sample_no"))
            if not sample_no:
                raw = ",".join(row)
                db.record_import_failure(summary.source_file, i, raw, "试样编号为空")
                summary.failures += 1
                continue

            if sample_no not in sample_id_cache:
                existing = db.get_sample_by_no(sample_no)
                if existing:
                    sample_id_cache[sample_no] = existing["id"]
                else:
                    raw = ",".join(row)
                    db.record_import_failure(
                        summary.source_file, i, raw,
                        f"试样编号 '{sample_no}' 不存在，请先录入试样"
                    )
                    summary.failures += 1
                    continue

            sample_id = sample_id_cache[sample_no]
            sample_row = db.get_sample_by_id(sample_id)
            dilution_ratio = _get_value(row, presc_cols.get("dilution_ratio")) or None
            ink_amount = _get_value(row, presc_cols.get("ink_amount")) or None
            environment = _get_value(row, presc_cols.get("environment")) or None
            retest_time = _get_value(row, presc_cols.get("retest_time")) or None
            observation_focus = _get_value(row, presc_cols.get("observation_focus")) or None
            remark = _get_value(row, presc_cols.get("remark")) or None
            source = _get_value(row, presc_cols.get("source")) or "manual"
            confidence_raw = _get_value(row, presc_cols.get("confidence_score"))
            confidence_score = 0.0
            if confidence_raw:
                try:
                    confidence_score = float(confidence_raw)
                except ValueError:
                    confidence_score = 0.0

            pd = models.PrescriptionData(
                sample_no=sample_no,
                dilution_ratio=dilution_ratio,
                ink_amount=ink_amount,
                environment=environment,
                retest_time=retest_time,
                observation_focus=observation_focus,
                remark=remark,
                source=source,
            )
            vr, _ = models.validate_prescription(pd)
            if not vr:
                raw = ",".join(row)
                reason = models.errors_to_text(vr.errors)
                db.record_import_failure(summary.source_file, i, raw, reason)
                summary.failures += 1
                continue

            anomaly_type_text = None
            measurements = db.get_measurements_by_sample(sample_id)
            types = []
            for m in measurements:
                at = m.get("anomaly_type")
                if at:
                    for part in at.split(";"):
                        p = part.strip()
                        if p and p not in types:
                            types.append(p)
            if types:
                anomaly_type_text = "; ".join(types)

            try:
                db.create_prescription(
                    sample_id=sample_id,
                    paper_type=sample_row["paper_type"] if sample_row else "",
                    anomaly_type=anomaly_type_text,
                    dilution_ratio=dilution_ratio,
                    ink_amount=ink_amount,
                    environment=environment,
                    retest_time=retest_time,
                    observation_focus=observation_focus,
                    source=source,
                    confidence_score=confidence_score,
                    remark=remark,
                )
                summary.prescriptions_created += 1
            except Exception as e:
                raw = ",".join(row)
                reason = f"数据库写入失败: {e}"
                db.record_import_failure(summary.source_file, i, raw, reason)
                summary.failures += 1

    if mode == "experiment":
        exp_cols = _detect_columns(headers, EXPERIMENT_COLUMN_ALIASES)
        sample_id_cache: Dict[str, int] = {}
        presc_id_cache: Dict[str, int] = {}
        samples = db.get_all_samples()
        for s in samples:
            sample_id_cache[s["sample_no"]] = s["id"]

        for i, row in enumerate(data_rows, start=2):
            sample_no = _get_value(row, exp_cols.get("sample_no"))
            presc_id_raw = _get_value(row, exp_cols.get("prescription_id"))
            if not sample_no and not presc_id_raw:
                raw = ",".join(row)
                db.record_import_failure(summary.source_file, i, raw, "试样编号和处方编号不能同时为空")
                summary.failures += 1
                continue

            prescription_id: Optional[int] = None
            if presc_id_raw:
                try:
                    prescription_id = int(presc_id_raw)
                except ValueError:
                    raw = ",".join(row)
                    db.record_import_failure(
                        summary.source_file, i, raw,
                        f"处方编号格式错误: '{presc_id_raw}'"
                    )
                    summary.failures += 1
                    continue

            sample_id: Optional[int] = None
            if sample_no:
                if sample_no in sample_id_cache:
                    sample_id = sample_id_cache[sample_no]
                else:
                    existing = db.get_sample_by_no(sample_no)
                    if existing:
                        sample_id = existing["id"]
                        sample_id_cache[sample_no] = existing["id"]
                    else:
                        raw = ",".join(row)
                        db.record_import_failure(
                            summary.source_file, i, raw,
                            f"试样编号 '{sample_no}' 不存在"
                        )
                        summary.failures += 1
                        continue

            if prescription_id is None and sample_id is not None:
                prescs = db.get_prescriptions_by_sample(sample_id)
                if prescs:
                    prescription_id = prescs[0]["id"]

            if prescription_id is not None:
                presc = db.get_prescription_by_id(prescription_id)
                if presc is None:
                    raw = ",".join(row)
                    db.record_import_failure(
                        summary.source_file, i, raw,
                        f"处方编号 {prescription_id} 不存在"
                    )
                    summary.failures += 1
                    continue
                if sample_id is None:
                    sample_id = presc["sample_id"]

            if sample_id is None:
                raw = ",".join(row)
                db.record_import_failure(summary.source_file, i, raw, "无法确定试样ID")
                summary.failures += 1
                continue

            retest_sample_no = _get_value(row, exp_cols.get("retest_sample_no")) or None
            execute_date_raw = _get_value(row, exp_cols.get("execute_date"))
            execute_date = models.normalize_date(execute_date_raw) if execute_date_raw else None
            diffusion_improved = models.parse_bool_int(_get_value(row, exp_cols.get("diffusion_improved")))
            anomaly_reduced = models.parse_bool_int(_get_value(row, exp_cols.get("anomaly_reduced")))
            paper_judgment_affected = models.parse_bool_int(_get_value(row, exp_cols.get("paper_judgment_affected")))
            effect_rating_raw = _get_value(row, exp_cols.get("effect_rating"))
            effect_rating = effect_rating_raw.strip() if effect_rating_raw and effect_rating_raw.strip() in ("优", "良", "中", "差") else None
            operator = _get_value(row, exp_cols.get("operator")) or None
            remark = _get_value(row, exp_cols.get("remark")) or None
            pre_risk_flag = _get_value(row, exp_cols.get("pre_risk_flag")) or None
            post_risk_flag = _get_value(row, exp_cols.get("post_risk_flag")) or None

            pre_anomaly_ratio = None
            pre_raw = _get_value(row, exp_cols.get("pre_anomaly_ratio"))
            if pre_raw:
                try:
                    v = float(pre_raw)
                    if 0 <= v <= 1:
                        pre_anomaly_ratio = v
                    elif v > 1:
                        pre_anomaly_ratio = v / 100.0
                except ValueError:
                    pass
            post_anomaly_ratio = None
            post_raw = _get_value(row, exp_cols.get("post_anomaly_ratio"))
            if post_raw:
                try:
                    v = float(post_raw)
                    if 0 <= v <= 1:
                        post_anomaly_ratio = v
                    elif v > 1:
                        post_anomaly_ratio = v / 100.0
                except ValueError:
                    pass

            sample_row = db.get_sample_by_id(sample_id)
            paper_type = sample_row["paper_type"] if sample_row else ""
            batch_id = sample_row["batch_id"] if sample_row else None

            try:
                db.create_experiment_record(
                    prescription_id=prescription_id if prescription_id else 0,
                    sample_id=sample_id,
                    paper_type=paper_type,
                    batch_id=batch_id,
                    retest_sample_no=retest_sample_no,
                    execute_date=execute_date,
                    diffusion_improved=diffusion_improved,
                    anomaly_reduced=anomaly_reduced,
                    paper_judgment_affected=paper_judgment_affected,
                    pre_risk_flag=pre_risk_flag,
                    post_risk_flag=post_risk_flag,
                    pre_anomaly_ratio=pre_anomaly_ratio if pre_anomaly_ratio is not None else 0.0,
                    post_anomaly_ratio=post_anomaly_ratio if post_anomaly_ratio is not None else 0.0,
                    effect_rating=effect_rating,
                    operator=operator,
                    remark=remark,
                )
                summary.experiment_records_created += 1
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
    if mode == "prescription":
        summary.messages.append(
            f"处方: 新增 {summary.prescriptions_created}，跳过 {summary.prescriptions_skipped}"
        )
    if mode == "experiment":
        summary.messages.append(
            f"实验记录: 新增 {summary.experiment_records_created}，跳过 {summary.experiment_records_skipped}"
        )
    summary.messages.append(f"失败记录: {summary.failures}")

    if baseline_sample_ids_created:
        built = 0
        for sid in baseline_sample_ids_created:
            try:
                bl = anomaly_detection.build_baseline_from_sample(sid)
                if bl:
                    built += 1
            except Exception:
                pass
        if built > 0:
            summary.messages.append(f"基线模板: 成功构建 {built} 个纸型基线")

    return summary
