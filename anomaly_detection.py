from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
import statistics
import json
import db


@dataclass
class AnomalyResult:
    is_anomaly: bool
    anomaly_type: Optional[str]
    details: str


@dataclass
class BaselineDeviation:
    time: float
    sample_radius: float
    baseline_radius: float
    diff: float
    diff_pct: float
    sample_roughness: float
    baseline_roughness: float
    rough_diff: float
    rough_diff_pct: float


@dataclass
class PaperTypeBaseline:
    paper_type: str
    template_id: Optional[int]
    baseline_sample_id: Optional[int]
    sample_ids: List[int]
    avg_slope: float
    avg_radius: float
    avg_roughness: float
    times: List[float]
    radii: List[float]
    roughness: List[float]


@dataclass
class SampleJudgment:
    sample_id: int
    sample_no: str
    judgment: str
    risk_flag: str
    reasons: List[str]
    baseline_deviation_score: float = 0.0
    matched_baseline_paper: Optional[str] = None


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
    baseline_deviation_score = 0.0
    matched_baseline_paper: Optional[str] = None

    slope = _fit_slope(times, radii)
    avg_roughness = statistics.mean(roughness) if roughness else 0.0
    avg_radius = statistics.mean(radii) if radii else 0.0
    max_radius = max(radii) if radii else 0.0

    anomaly_ratio = anomaly_count / len(rows) if rows else 0.0
    if anomaly_ratio >= 0.4:
        reasons.append(f"异常测量点占比 {anomaly_ratio:.0%}")
        risk_flags.append("高风险")

    baseline_anom_count = sum(
        1 for r in rows if r["anomaly_type"] and "偏离纸型基线" in (r["anomaly_type"] or "")
    )
    if baseline_anom_count > 0:
        baseline_ratio = baseline_anom_count / len(rows)
        baseline_deviation_score = baseline_ratio * 100
        matched_baseline_paper = sample_row["paper_type"] if sample_row else None
        if baseline_ratio >= 0.3:
            judgment_parts.append("偏离纸型基线")
            reasons.append(
                f"偏离纸型「{matched_baseline_paper}」基线 {baseline_anom_count} 个点"
                f"（占比 {baseline_ratio:.0%}），扩散行为显著偏离同纸型"
            )
            risk_flags.append("高风险" if baseline_ratio >= 0.5 else "中风险")
        elif baseline_ratio >= 0.1:
            reasons.append(
                f"部分偏离「{matched_baseline_paper}」纸型基线 {baseline_anom_count} 个点"
                f"（占比 {baseline_ratio:.0%}）"
            )
            risk_flags.append("中风险")

    baseline = None
    if sample_row:
        baseline = get_paper_type_baseline(sample_row["paper_type"])
    if baseline and baseline.avg_slope > 0:
        matched_baseline_paper = baseline.paper_type
        slope_bl_ratio = slope / baseline.avg_slope if baseline.avg_slope > 0 else 1.0
        if slope_bl_ratio > 1.5:
            if "疑似过浓" not in judgment_parts:
                judgment_parts.append("偏离纸型基线-过浓")
            reasons.append(
                f"渗化斜率为「{baseline.paper_type}」基线的 {slope_bl_ratio:.1f} 倍，扩散过快（纸型对照）"
            )
            risk_flags.append("中风险")
            baseline_deviation_score = max(baseline_deviation_score, abs(slope_bl_ratio - 1) * 50)
        elif slope_bl_ratio < 0.6:
            if "疑似过稀" not in judgment_parts:
                judgment_parts.append("偏离纸型基线-过稀")
            reasons.append(
                f"渗化斜率仅为「{baseline.paper_type}」基线的 {slope_bl_ratio:.1f} 倍，扩散过慢（纸型对照）"
            )
            risk_flags.append("中风险")
            baseline_deviation_score = max(baseline_deviation_score, abs(slope_bl_ratio - 1) * 50)

        if baseline.avg_roughness > 0:
            rough_bl_ratio = avg_roughness / baseline.avg_roughness
            if rough_bl_ratio > 1.6:
                judgment_parts.append("偏离纸型基线-纸性异常")
                reasons.append(
                    f"毛糙度均值为「{baseline.paper_type}」基线的 {rough_bl_ratio:.1f} 倍（纸型对照）"
                )
                risk_flags.append("中风险")
            elif rough_bl_ratio < 0.5:
                judgment_parts.append("偏离纸型基线-纸性异常")
                reasons.append(
                    f"毛糙度均值仅为「{baseline.paper_type}」基线的 {rough_bl_ratio:.1f} 倍（纸型对照）"
                )
                risk_flags.append("中风险")

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
        if slope_ratio > 1.6 and not any("过浓" in j for j in judgment_parts):
            judgment_parts.append("疑似过浓")
            reasons.append(f"渗化斜率为参考组的 {slope_ratio:.1f} 倍，扩散过快")
            risk_flags.append("中风险")
        elif slope_ratio < 0.55 and not any("过稀" in j for j in judgment_parts):
            judgment_parts.append("疑似过稀")
            reasons.append(f"渗化斜率仅为参考组的 {slope_ratio:.1f} 倍，扩散过慢")
            risk_flags.append("中风险")
    elif not baseline or baseline.avg_slope == 0:
        if slope > 1.2:
            judgment_parts.append("疑似过浓")
            reasons.append(f"渗化斜率={slope:.3f} 偏高，无参考组情况下判定扩散过快")
        elif slope < 0.25:
            judgment_parts.append("疑似过稀")
            reasons.append(f"渗化斜率={slope:.3f} 偏低，无参考组情况下判定扩散过慢")

    if ref_avg_rough is not None and ref_avg_rough > 0:
        rough_ratio = avg_roughness / ref_avg_rough
        if rough_ratio > 1.8 and not any("纸性异常" in j for j in judgment_parts):
            judgment_parts.append("纸性异常")
            reasons.append(f"毛糙度均值为参考组的 {rough_ratio:.1f} 倍，边缘明显偏糙")
            risk_flags.append("中风险")
        elif rough_ratio < 0.45 and not any("纸性异常" in j for j in judgment_parts):
            judgment_parts.append("纸性异常")
            reasons.append(f"毛糙度均值仅为参考组的 {rough_ratio:.1f} 倍，边缘异常光滑")
            risk_flags.append("中风险")
    elif not baseline or baseline.avg_roughness == 0:
        if avg_roughness > 3.5:
            judgment_parts.append("纸性异常(可疑)")
            reasons.append(f"毛糙度均值={avg_roughness:.2f} 偏高")

    if ref_avg_radius is not None and ref_avg_radius > 0:
        radius_ratio = max_radius / ref_avg_radius
        if radius_ratio > 1.7:
            if not any("过浓" in j for j in judgment_parts):
                judgment_parts.append("疑似过浓")
            reasons.append(f"最大半径为参考组均值的 {radius_ratio:.1f} 倍")

    if not judgment_parts:
        judgment_parts.append("正常")
    if not reasons:
        if matched_baseline_paper:
            reasons.append(f"各项指标在「{matched_baseline_paper}」纸型基线正常范围内")
        else:
            reasons.append("各项指标在正常范围内")

    final_judgment = " / ".join(dict.fromkeys(judgment_parts))
    if "高风险" in risk_flags:
        final_risk = "高风险"
    elif "中风险" in risk_flags:
        final_risk = "中风险"
    else:
        final_risk = "正常"

    return SampleJudgment(
        sample_id, sample_no, final_judgment, final_risk, reasons,
        baseline_deviation_score=round(baseline_deviation_score, 1),
        matched_baseline_paper=matched_baseline_paper
    )


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
    anom_count = reclassify_with_baseline(sample_id)
    batch_id = db.get_batch_id_of_sample(sample_id)

    result = {
        "anomalies_reclassified": anom_count,
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


# ==================== Baseline Matching & Deviation ====================

def _parse_float_list(raw: Optional[str]) -> List[float]:
    if not raw:
        return []
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, list):
            return [float(x) for x in parsed]
    except (json.JSONDecodeError, ValueError, TypeError):
        pass
    try:
        return [float(x) for x in raw.split(",") if x.strip()]
    except (ValueError, TypeError):
        return []


def get_paper_type_baseline(paper_type: str) -> Optional[PaperTypeBaseline]:
    template = db.get_baseline_template_by_paper(paper_type)

    baseline_sample_rows = db.get_baselines_by_paper_type(paper_type)
    baseline_sample_ids = [r["id"] for r in baseline_sample_rows]

    if not template and not baseline_sample_rows:
        return None

    if template:
        times = _parse_float_list(template["baseline_times"])
        radii = _parse_float_list(template["baseline_radii"])
        roughness = _parse_float_list(template["baseline_roughness"])
        avg_slope = float(template["avg_slope"]) if template["avg_slope"] else 0.0
        avg_radius = float(template["avg_radius"]) if template["avg_radius"] else 0.0
        avg_rough = float(template["avg_roughness"]) if template["avg_roughness"] else 0.0
        template_id = int(template["id"])
        baseline_sample_id = template["baseline_sample_id"]
    else:
        times, radii, roughness = [], [], []
        avg_slope_list, avg_radius_list, avg_rough_list = [], [], []
        for bs in baseline_sample_rows:
            bs_rows = db.get_measurements_by_sample(bs["id"])
            if len(bs_rows) >= 3:
                bt = [float(m["adsorb_time"]) for m in bs_rows]
                br = [float(m["radius"]) for m in bs_rows]
                bro = [float(m["roughness"]) for m in bs_rows]
                s_idx = sorted(range(len(bt)), key=lambda i: bt[i])
                bt = [bt[i] for i in s_idx]
                br = [br[i] for i in s_idx]
                bro = [bro[i] for i in s_idx]
                if not times:
                    times = bt
                if not radii:
                    radii = br
                if not roughness:
                    roughness = bro
                avg_slope_list.append(_fit_slope(bt, br))
                avg_radius_list.append(statistics.mean(br))
                avg_rough_list.append(statistics.mean(bro))
        avg_slope = statistics.mean(avg_slope_list) if avg_slope_list else 0.0
        avg_radius = statistics.mean(avg_radius_list) if avg_radius_list else 0.0
        avg_rough = statistics.mean(avg_rough_list) if avg_rough_list else 0.0
        template_id = None
        baseline_sample_id = baseline_sample_ids[0] if baseline_sample_ids else None

    return PaperTypeBaseline(
        paper_type=paper_type,
        template_id=template_id,
        baseline_sample_id=baseline_sample_id,
        sample_ids=baseline_sample_ids,
        avg_slope=avg_slope,
        avg_radius=avg_radius,
        avg_roughness=avg_rough,
        times=times,
        radii=radii,
        roughness=roughness,
    )


def _interpolate_baseline_value(times: List[float], values: List[float], t: float) -> Optional[float]:
    if not times or not values or len(times) != len(values):
        return None
    if t <= times[0]:
        return values[0]
    if t >= times[-1]:
        return values[-1]
    for i in range(len(times) - 1):
        t0, t1 = times[i], times[i + 1]
        v0, v1 = values[i], values[i + 1]
        if t0 <= t <= t1:
            if t1 == t0:
                return v0
            ratio = (t - t0) / (t1 - t0)
            return v0 + ratio * (v1 - v0)
    return values[-1]


def compute_baseline_deviations(
    sample_times: List[float],
    sample_radii: List[float],
    sample_roughness: List[float],
    baseline: PaperTypeBaseline
) -> List[BaselineDeviation]:
    deviations: List[BaselineDeviation] = []
    for i, t in enumerate(sample_times):
        bl_r = _interpolate_baseline_value(baseline.times, baseline.radii, t)
        bl_ro = _interpolate_baseline_value(baseline.times, baseline.roughness, t)
        if bl_r is None:
            continue
        r = sample_radii[i]
        ro = sample_roughness[i] if i < len(sample_roughness) else 0.0
        diff = r - bl_r
        diff_pct = (diff / bl_r * 100.0) if bl_r != 0 else 0.0
        rough_diff = ro - (bl_ro if bl_ro is not None else 0.0)
        rough_bl = bl_ro if bl_ro is not None else 0.0
        rough_diff_pct = (rough_diff / rough_bl * 100.0) if rough_bl != 0 else 0.0
        deviations.append(BaselineDeviation(
            time=t,
            sample_radius=r,
            baseline_radius=bl_r,
            diff=diff,
            diff_pct=diff_pct,
            sample_roughness=ro,
            baseline_roughness=rough_bl,
            rough_diff=rough_diff,
            rough_diff_pct=rough_diff_pct,
        ))
    return deviations


def detect_baseline_deviation_anomalies(
    sample_id: int,
    threshold_pct: float = 20.0
) -> List[Tuple[int, AnomalyResult]]:
    rows = db.get_measurements_by_sample(sample_id)
    if len(rows) < 3:
        return []

    sample = db.get_sample_by_id(sample_id)
    if not sample:
        return []

    baseline = get_paper_type_baseline(sample["paper_type"])
    if not baseline or not baseline.times:
        return []

    sample_times = [float(r["adsorb_time"]) for r in rows]
    sample_radii = [float(r["radius"]) for r in rows]
    sample_rough = [float(r["roughness"]) for r in rows]

    deviations = compute_baseline_deviations(sample_times, sample_radii, sample_rough, baseline)

    results: List[Tuple[int, AnomalyResult]] = []
    for i, dev in enumerate(deviations):
        row = rows[i]
        abs_pct = abs(dev.diff_pct)
        abs_rough_pct = abs(dev.rough_diff_pct)

        if abs_pct >= threshold_pct:
            direction = "偏大" if dev.diff > 0 else "偏小"
            anomaly = AnomalyResult(
                True,
                f"偏离纸型基线-半径{direction}",
                (f"时间 {dev.time:.1f}s，与「{baseline.paper_type}」基线偏差 "
                 f"{dev.diff_pct:+.1f}%（{dev.diff:+.2f}mm），超过阈值 {threshold_pct}%")
            )
            results.append((row["id"], anomaly))
        elif abs_rough_pct >= threshold_pct * 1.5:
            direction = "偏糙" if dev.rough_diff > 0 else "偏光滑"
            anomaly = AnomalyResult(
                True,
                f"偏离纸型基线-毛糙度{direction}",
                (f"时间 {dev.time:.1f}s，与「{baseline.paper_type}」基线毛糙度偏差 "
                 f"{dev.rough_diff_pct:+.1f}%，超过阈值 {threshold_pct * 1.5}%")
            )
            results.append((row["id"], anomaly))

    return results


def build_baseline_from_sample(sample_id: int, remark: str = "") -> Optional[PaperTypeBaseline]:
    sample = db.get_sample_by_id(sample_id)
    if not sample:
        return None

    rows = db.get_measurements_by_sample(sample_id)
    if len(rows) < 3:
        return None

    times = [float(r["adsorb_time"]) for r in rows]
    radii = [float(r["radius"]) for r in rows]
    roughness = [float(r["roughness"]) for r in rows]
    s_idx = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in s_idx]
    radii = [radii[i] for i in s_idx]
    roughness = [roughness[i] for i in s_idx]

    avg_slope = _fit_slope(times, radii)
    avg_radius = statistics.mean(radii)
    avg_rough = statistics.mean(roughness)

    times_json = json.dumps(times, ensure_ascii=False)
    radii_json = json.dumps(radii, ensure_ascii=False)
    rough_json = json.dumps(roughness, ensure_ascii=False)

    existing = db.get_baseline_template_by_paper(sample["paper_type"])
    if existing:
        db.update_baseline_template(
            existing["id"], sample_id, avg_slope, avg_radius, avg_rough,
            times_json, radii_json, rough_json, remark
        )
    else:
        db.create_baseline_template(
            sample["paper_type"], sample_id, avg_slope, avg_radius, avg_rough,
            times_json, radii_json, rough_json, remark
        )

    db.set_sample_baseline(sample_id, 1)

    return PaperTypeBaseline(
        paper_type=sample["paper_type"],
        template_id=None,
        baseline_sample_id=sample_id,
        sample_ids=[sample_id],
        avg_slope=avg_slope,
        avg_radius=avg_radius,
        avg_roughness=avg_rough,
        times=times,
        radii=radii,
        roughness=roughness,
    )


def reclassify_with_baseline(sample_id: int) -> int:
    baseline_anoms = detect_baseline_deviation_anomalies(sample_id)
    series_anoms = analyze_sample_anomalies(sample_id)

    anom_map: Dict[int, AnomalyResult] = {}
    for mid, ar in series_anoms:
        anom_map[mid] = ar
    for mid, ar in baseline_anoms:
        if mid in anom_map:
            existing = anom_map[mid]
            combined = AnomalyResult(
                True,
                f"{existing.anomaly_type}; {ar.anomaly_type}",
                f"{existing.details}; {ar.details}"
            )
            anom_map[mid] = combined
        else:
            anom_map[mid] = ar

    count = 0
    with db.get_connection() as conn:
        conn.execute(
            "UPDATE measurements SET is_anomaly = 0, anomaly_type = NULL WHERE sample_id = ?",
            (sample_id,)
        )
        for mid, ar in anom_map.items():
            conn.execute(
                "UPDATE measurements SET is_anomaly = ?, anomaly_type = ? WHERE id = ?",
                (1 if ar.is_anomaly else 0, ar.anomaly_type, mid)
            )
            if ar.is_anomaly:
                count += 1
    return count
