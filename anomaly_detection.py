from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import statistics
import db


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_type: Optional[str]
    details: str


@dataclass
class SampleJudgment:
    sample_id: int
    sample_no: str
    judgment: str
    risk_flag: str
    reasons: List[str]


def detect_anomaly_in_series(
    times: List[float],
    radii: List[float],
    roughness_list: List[float],
    idx: int
) -> AnomalyResult:
    if len(radii) < 4:
        return AnomalyResult(False, None, "数据点过少，暂不判定")

    current_r = radii[idx]
    current_t = times[idx]
    current_rough = roughness_list[idx]

    other_radii = [r for i, r in enumerate(radii) if i != idx]
    other_rough = [ro for i, ro in enumerate(roughness_list) if i != idx]

    mean_r = statistics.mean(other_radii)
    stdev_r = statistics.pstdev(other_radii) if len(other_radii) >= 2 else 0.0
    mean_ro = statistics.mean(other_rough)
    stdev_ro = statistics.pstdev(other_rough) if len(other_rough) >= 2 else 0.0

    z_r = abs(current_r - mean_r) / stdev_r if stdev_r > 0 else 0.0
    z_ro = abs(current_rough - mean_ro) / stdev_ro if stdev_ro > 0 else 0.0

    if idx > 0:
        dt = current_t - times[idx - 1]
        dr = current_r - radii[idx - 1]
        if dt > 0:
            rate = dr / dt
            other_rates = []
            for j in range(1, len(times)):
                if j != idx and j - 1 != idx:
                    dtj = times[j] - times[j - 1]
                    if dtj > 0:
                        other_rates.append((radii[j] - radii[j - 1]) / dtj)
            if other_rates:
                mean_rate = statistics.mean(other_rates)
                stdev_rate = statistics.pstdev(other_rates) if len(other_rates) >= 2 else 0.0
                z_rate = abs(rate - mean_rate) / stdev_rate if stdev_rate > 0 else 0.0
                if z_rate > 2.5:
                    direction = "过快" if rate > mean_rate else "过慢"
                    return AnomalyResult(
                        True,
                        f"扩散速率{direction}",
                        f"速率 z={z_rate:.2f}，偏离同试样其他时段超过 2.5σ"
                    )

    if z_r > 2.5 and z_ro > 2.0:
        return AnomalyResult(
            True,
            "半径与毛糙度同时异常",
            f"半径 z={z_r:.2f}，毛糙度 z={z_ro:.2f}"
        )
    if z_r > 2.8:
        return AnomalyResult(
            True,
            "扩散半径异常",
            f"半径 z={z_r:.2f}，偏离均值超过 2.8σ"
        )
    if z_ro > 2.5:
        return AnomalyResult(
            True,
            "边缘毛糙度异常",
            f"毛糙度 z={z_ro:.2f}，偏离均值超过 2.5σ"
        )

    return AnomalyResult(False, None, "正常")


def analyze_sample_anomalies(sample_id: int) -> List[Tuple[int, AnomalyResult]]:
    rows = db.get_measurements_by_sample(sample_id)
    if len(rows) < 4:
        return []

    times = [float(r["adsorb_time"]) for r in rows]
    radii = [float(r["radius"]) for r in rows]
    roughness = [float(r["roughness"]) for r in rows]

    results = []
    for i, row in enumerate(rows):
        ar = detect_anomaly_in_series(times, radii, roughness, i)
        results.append((row["id"], ar))
    return results


def reclassify_measurement_anomalies(sample_id: int) -> int:
    results = analyze_sample_anomalies(sample_id)
    count = 0
    with db.get_connection() as conn:
        for mid, ar in results:
            conn.execute(
                "UPDATE measurements SET is_anomaly = ?, anomaly_type = ? WHERE id = ?",
                (1 if ar.is_anomaly else 0, ar.anomaly_type, mid)
            )
            if ar.is_anomaly:
                count += 1
    return count


def _fit_slope(times: List[float], values: List[float]) -> float:
    if len(times) < 2:
        return 0.0
    n = len(times)
    mean_t = sum(times) / n
    mean_v = sum(values) / n
    num = sum((t - mean_t) * (v - mean_v) for t, v in zip(times, values))
    den = sum((t - mean_t) ** 2 for t in times)
    if den == 0:
        return 0.0
    return num / den


def judge_sample(sample_id: int, reference_samples: Optional[List[int]] = None) -> SampleJudgment:
    rows = db.get_measurements_by_sample(sample_id)
    sample_row = db.get_sample_by_id(sample_id)
    sample_no = sample_row["sample_no"] if sample_row else str(sample_id)

    if not rows:
        return SampleJudgment(sample_id, sample_no, "数据不足", "正常", ["无测量数据"])

    times = [float(r["adsorb_time"]) for r in rows]
    radii = [float(r["radius"]) for r in rows]
    roughness = [float(r["roughness"]) for r in rows]
    anomaly_count = sum(1 for r in rows if r["is_anomaly"])

    reasons: List[str] = []
    judgment_parts: List[str] = []
    risk_flags: List[str] = []

    slope = _fit_slope(times, radii)
    avg_roughness = statistics.mean(roughness) if roughness else 0.0
    avg_radius = statistics.mean(radii) if radii else 0.0
    max_radius = max(radii) if radii else 0.0

    anomaly_ratio = anomaly_count / len(rows) if rows else 0.0
    if anomaly_ratio >= 0.4:
        reasons.append(f"异常测量点占比 {anomaly_ratio:.0%}")
        risk_flags.append("高风险")

    ref_avg_slope: Optional[float] = None
    ref_avg_radius: Optional[float] = None
    ref_avg_rough: Optional[float] = None

    if reference_samples:
        ref_slopes = []
        ref_radii = []
        ref_roughs = []
        for ref_id in reference_samples:
            if ref_id == sample_id:
                continue
            ref_rows = db.get_measurements_by_sample(ref_id)
            if len(ref_rows) >= 3:
                rt = [float(r["adsorb_time"]) for r in ref_rows]
                rr = [float(r["radius"]) for r in ref_rows]
                rro = [float(r["roughness"]) for r in ref_rows]
                ref_slopes.append(_fit_slope(rt, rr))
                ref_radii.append(statistics.mean(rr))
                ref_roughs.append(statistics.mean(rro))
        if ref_slopes:
            ref_avg_slope = statistics.mean(ref_slopes)
        if ref_radii:
            ref_avg_radius = statistics.mean(ref_radii)
        if ref_roughs:
            ref_avg_rough = statistics.mean(ref_roughs)

    if ref_avg_slope is not None and ref_avg_slope > 0:
        slope_ratio = slope / ref_avg_slope
        if slope_ratio > 1.6:
            judgment_parts.append("疑似过浓")
            reasons.append(f"渗化斜率为参考组的 {slope_ratio:.1f} 倍，扩散过快")
            risk_flags.append("中风险")
        elif slope_ratio < 0.55:
            judgment_parts.append("疑似过稀")
            reasons.append(f"渗化斜率仅为参考组的 {slope_ratio:.1f} 倍，扩散过慢")
            risk_flags.append("中风险")
    else:
        if slope > 1.2:
            judgment_parts.append("疑似过浓")
            reasons.append(f"渗化斜率={slope:.3f} 偏高，无参考组情况下判定扩散过快")
        elif slope < 0.25:
            judgment_parts.append("疑似过稀")
            reasons.append(f"渗化斜率={slope:.3f} 偏低，无参考组情况下判定扩散过慢")

    if ref_avg_rough is not None and ref_avg_rough > 0:
        rough_ratio = avg_roughness / ref_avg_rough
        if rough_ratio > 1.8:
            judgment_parts.append("纸性异常")
            reasons.append(f"毛糙度均值为参考组的 {rough_ratio:.1f} 倍，边缘明显偏糙")
            risk_flags.append("中风险")
        elif rough_ratio < 0.45:
            judgment_parts.append("纸性异常")
            reasons.append(f"毛糙度均值仅为参考组的 {rough_ratio:.1f} 倍，边缘异常光滑")
            risk_flags.append("中风险")
    else:
        if avg_roughness > 3.5:
            judgment_parts.append("纸性异常(可疑)")
            reasons.append(f"毛糙度均值={avg_roughness:.2f} 偏高")

    if ref_avg_radius is not None and ref_avg_radius > 0:
        radius_ratio = max_radius / ref_avg_radius
        if radius_ratio > 1.7:
            if "疑似过浓" not in judgment_parts:
                judgment_parts.append("疑似过浓")
            reasons.append(f"最大半径为参考组均值的 {radius_ratio:.1f} 倍")

    if not judgment_parts:
        judgment_parts.append("正常")
    if not reasons:
        reasons.append("各项指标在正常范围内")

    final_judgment = " / ".join(dict.fromkeys(judgment_parts))
    if "高风险" in risk_flags:
        final_risk = "高风险"
    elif "中风险" in risk_flags:
        final_risk = "中风险"
    else:
        final_risk = "正常"

    return SampleJudgment(sample_id, sample_no, final_judgment, final_risk, reasons)


def update_batch_risk_by_anomalies(batch_id: int) -> Tuple[str, int]:
    samples = db.get_samples_by_batch(batch_id)
    sample_ids = [s["id"] for s in samples]
    if not sample_ids:
        return "正常", 0

    all_measurements: List[Dict] = []
    ms_map = db.get_measurements_by_samples(sample_ids)
    for sid in sample_ids:
        for m in ms_map.get(sid, []):
            all_measurements.append({
                "sample_id": sid,
                "time": float(m["adsorb_time"]),
                "is_anomaly": int(m["is_anomaly"])
            })

    all_measurements.sort(key=lambda x: x["time"])

    consecutive = 0
    max_consecutive = 0
    for m in all_measurements:
        if m["is_anomaly"]:
            consecutive += 1
            max_consecutive = max(max_consecutive, consecutive)
        else:
            consecutive = 0

    if max_consecutive >= 5:
        risk_level = "高风险"
    elif max_consecutive >= 3:
        risk_level = "中风险"
    elif max_consecutive >= 1:
        risk_level = "低风险"
    else:
        risk_level = "正常"

    db.update_batch_risk(batch_id, risk_level, max_consecutive)
    return risk_level, max_consecutive


def process_measurement_and_update_risk(sample_id: int) -> Dict:
    reclassify_measurement_anomalies(sample_id)
    batch_id = db.get_batch_id_of_sample(sample_id)

    result = {
        "anomalies_reclassified": 0,
        "batch_id": batch_id,
        "batch_risk": None,
        "batch_consecutive": 0,
        "sample_judgment": None,
    }

    if batch_id is not None:
        risk_level, consecutive = update_batch_risk_by_anomalies(batch_id)
        result["batch_risk"] = risk_level
        result["batch_consecutive"] = consecutive

    samples = db.get_all_samples()
    ref_ids = [s["id"] for s in samples if s["id"] != sample_id]
    j = judge_sample(sample_id, ref_ids if ref_ids else None)
    db.update_sample_judgment(sample_id, j.judgment, j.risk_flag)
    result["sample_judgment"] = j

    return result
