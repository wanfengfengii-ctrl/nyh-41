from dataclasses import dataclass
from typing import Optional, List, Tuple
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
