from dataclasses import dataclass
from typing import Optional, List, Tuple, Any
from datetime import datetime
import db


@dataclass
class ValidationError:
    field: str
    message: str


@dataclass
class SampleData:
    sample_no: str
    paper_type: str
    ink_date: str
    batch_code: Optional[str] = None


@dataclass
class MeasurementData:
    sample_no: str
    adsorb_time: float
    radius: float
    roughness: float


@dataclass
class ValidationResult:
    valid: bool
    errors: List[ValidationError]

    def __bool__(self) -> bool:
        return self.valid


def validate_sample(data: SampleData) -> ValidationResult:
    errors: List[ValidationError] = []

    if not data.sample_no or not data.sample_no.strip():
        errors.append(ValidationError("sample_no", "试样编号不能为空"))
    elif db.sample_no_exists(data.sample_no.strip()):
        errors.append(ValidationError("sample_no", f"试样编号 '{data.sample_no}' 已存在，必须唯一"))

    if not data.paper_type or not data.paper_type.strip():
        errors.append(ValidationError("paper_type", "纸张类型不能为空"))

    if not data.ink_date or not data.ink_date.strip():
        errors.append(ValidationError("ink_date", "点墨日期不能为空"))
    else:
        try:
            datetime.strptime(data.ink_date.strip(), "%Y-%m-%d")
        except ValueError:
            try:
                datetime.strptime(data.ink_date.strip(), "%Y/%m/%d")
            except ValueError:
                errors.append(ValidationError("ink_date",
                                              f"点墨日期格式错误: '{data.ink_date}'，请使用 YYYY-MM-DD"))

    return ValidationResult(valid=len(errors) == 0, errors=errors)


def validate_measurement(data: MeasurementData,
                         sample_id_cache: Optional[dict] = None) -> Tuple[ValidationResult, Optional[int]]:
    errors: List[ValidationError] = []
    sample_id: Optional[int] = None

    if not data.sample_no or not data.sample_no.strip():
        errors.append(ValidationError("sample_no", "试样编号不能为空"))
    else:
        if sample_id_cache and data.sample_no in sample_id_cache:
            sample_id = sample_id_cache[data.sample_no]
        else:
            row = db.get_sample_by_no(data.sample_no.strip())
            if row is None:
                errors.append(ValidationError("sample_no",
                                              f"试样编号 '{data.sample_no}' 不存在，请先录入试样"))
            else:
                sample_id = row["id"]

    try:
        adsorb_time = float(data.adsorb_time)
        if adsorb_time <= 0:
            errors.append(ValidationError("adsorb_time",
                                          f"吸附时间必须大于 0，当前值: {adsorb_time}"))
    except (ValueError, TypeError):
        errors.append(ValidationError("adsorb_time",
                                      f"吸附时间格式错误: '{data.adsorb_time}'，必须是数字"))
        adsorb_time = 0.0

    try:
        radius = float(data.radius)
        if radius <= 0:
            errors.append(ValidationError("radius",
                                          f"扩散半径必须大于 0，当前值: {radius}"))
    except (ValueError, TypeError):
        errors.append(ValidationError("radius",
                                      f"扩散半径格式错误: '{data.radius}'，必须是数字"))
        radius = 0.0

    try:
        roughness = float(data.roughness)
        if roughness < 0:
            errors.append(ValidationError("roughness",
                                          f"边缘毛糙度不能为负，当前值: {roughness}"))
    except (ValueError, TypeError):
        errors.append(ValidationError("roughness",
                                      f"边缘毛糙度格式错误: '{data.roughness}'，必须是数字"))

    if sample_id is not None and adsorb_time > 0:
        if db.measurement_exists(sample_id, adsorb_time):
            errors.append(ValidationError("adsorb_time",
                                          f"试样 '{data.sample_no}' 在 {adsorb_time}s 处已有测量记录，同一时间点不能重复测量"))

    return ValidationResult(valid=len(errors) == 0, errors=errors), sample_id


def normalize_ink_date(date_str: str) -> str:
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def errors_to_text(errors: List[ValidationError]) -> str:
    return "; ".join(f"[{e.field}] {e.message}" for e in errors)


@dataclass
class PrescriptionData:
    sample_no: str
    dilution_ratio: Optional[str] = None
    ink_amount: Optional[str] = None
    environment: Optional[str] = None
    retest_time: Optional[str] = None
    observation_focus: Optional[str] = None
    remark: Optional[str] = None
    source: str = "manual"


@dataclass
class ExperimentRecordData:
    sample_no: str
    prescription_id: Optional[int] = None
    retest_sample_no: Optional[str] = None
    execute_date: Optional[str] = None
    diffusion_improved: int = 0
    anomaly_reduced: int = 0
    paper_judgment_affected: int = 0
    effect_rating: Optional[str] = None
    operator: Optional[str] = None
    remark: Optional[str] = None
    pre_risk_flag: Optional[str] = None
    post_risk_flag: Optional[str] = None
    pre_anomaly_ratio: Optional[float] = None
    post_anomaly_ratio: Optional[float] = None


def validate_prescription(data: PrescriptionData) -> Tuple[ValidationResult, Optional[int]]:
    errors: List[ValidationError] = []
    sample_id: Optional[int] = None

    if not data.sample_no or not data.sample_no.strip():
        errors.append(ValidationError("sample_no", "试样编号不能为空"))
    else:
        row = db.get_sample_by_no(data.sample_no.strip())
        if row is None:
            errors.append(ValidationError("sample_no",
                                          f"试样编号 '{data.sample_no}' 不存在，请先录入试样"))
        else:
            sample_id = row["id"]

    has_any_field = any([
        data.dilution_ratio and data.dilution_ratio.strip(),
        data.ink_amount and data.ink_amount.strip(),
        data.environment and data.environment.strip(),
        data.retest_time and data.retest_time.strip(),
        data.observation_focus and data.observation_focus.strip(),
    ])
    if not has_any_field and data.source != "auto":
        errors.append(ValidationError("prescription_fields",
                                      "处方内容不能为空，请至少填写一项（稀释比例/点墨量/环境/复测时间/观察重点）"))

    if data.execute_date and data.execute_date.strip():
        try:
            datetime.strptime(data.execute_date.strip(), "%Y-%m-%d")
        except ValueError:
            try:
                datetime.strptime(data.execute_date.strip(), "%Y/%m/%d")
            except ValueError:
                errors.append(ValidationError("execute_date",
                                              f"执行日期格式错误: '{data.execute_date}'，请使用 YYYY-MM-DD"))

    if data.effect_rating and data.effect_rating.strip():
        if data.effect_rating.strip() not in ("优", "良", "中", "差"):
            errors.append(ValidationError("effect_rating",
                                          f"效果评级只能是 优/良/中/差，当前值: '{data.effect_rating}'"))

    return ValidationResult(valid=len(errors) == 0, errors=errors), sample_id


def validate_experiment_record(
    data: ExperimentRecordData,
    sample_id_cache: Optional[dict] = None,
) -> Tuple[ValidationResult, Tuple[Optional[int], Optional[int]]]:
    errors: List[ValidationError] = []
    sample_id: Optional[int] = None
    prescription_id: Optional[int] = data.prescription_id

    if not data.sample_no or not data.sample_no.strip():
        errors.append(ValidationError("sample_no", "试样编号不能为空"))
    else:
        sno = data.sample_no.strip()
        if sample_id_cache and sno in sample_id_cache:
            sample_id = sample_id_cache[sno]
        else:
            row = db.get_sample_by_no(sno)
            if row is None:
                errors.append(ValidationError("sample_no",
                                              f"试样编号 '{sno}' 不存在"))
            else:
                sample_id = row["id"]

    if prescription_id is not None:
        presc = db.get_prescription_by_id(prescription_id)
        if presc is None:
            errors.append(ValidationError("prescription_id",
                                          f"处方编号 {prescription_id} 不存在"))

    if data.execute_date and data.execute_date.strip():
        try:
            datetime.strptime(data.execute_date.strip(), "%Y-%m-%d")
        except ValueError:
            try:
                datetime.strptime(data.execute_date.strip(), "%Y/%m/%d")
            except ValueError:
                errors.append(ValidationError("execute_date",
                                              f"执行日期格式错误: '{data.execute_date}'，请使用 YYYY-MM-DD"))

    if data.effect_rating and data.effect_rating.strip():
        if data.effect_rating.strip() not in ("优", "良", "中", "差"):
            errors.append(ValidationError("effect_rating",
                                          f"效果评级只能是 优/良/中/差，当前值: '{data.effect_rating}'"))

    def _to_int(field: str, val: Any) -> Optional[int]:
        if val is None or val == "":
            return 0
        if isinstance(val, int):
            return 1 if val else 0
        s = str(val).strip().lower()
        if s in ("1", "true", "yes", "y", "是", "有"):
            return 1
        if s in ("0", "false", "no", "n", "否", "无"):
            return 0
        errors.append(ValidationError(field, f"{field} 必须是 是/否 或 1/0"))
        return None

    for fname, fval in [
        ("diffusion_improved", data.diffusion_improved),
        ("anomaly_reduced", data.anomaly_reduced),
        ("paper_judgment_affected", data.paper_judgment_affected),
    ]:
        _to_int(fname, fval)

    return ValidationResult(valid=len(errors) == 0, errors=errors), (sample_id, prescription_id)


def normalize_date(date_str: Optional[str]) -> Optional[str]:
    if not date_str:
        return None
    date_str = date_str.strip()
    for fmt in ("%Y-%m-%d", "%Y/%m/%d", "%Y.%m.%d"):
        try:
            return datetime.strptime(date_str, fmt).strftime("%Y-%m-%d")
        except ValueError:
            continue
    return date_str


def parse_bool_int(raw: Any) -> int:
    if raw is None or raw == "":
        return 0
    if isinstance(raw, int):
        return 1 if raw else 0
    if isinstance(raw, bool):
        return 1 if raw else 0
    s = str(raw).strip().lower()
    if s in ("1", "true", "yes", "y", "是", "有"):
        return 1
    return 0

