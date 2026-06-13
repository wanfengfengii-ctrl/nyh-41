from typing import List, Dict, Optional, Tuple, Any
from dataclasses import dataclass, field
import statistics
import json

import db
import anomaly_detection


@dataclass
class PrescParams:
    sample_id: int
    paper_type: str
    anomaly_type: Optional[str]
    avg_slope: float
    avg_radius: float
    avg_roughness: float
    max_radius: float
    anomaly_ratio: float
    risk_flag: str
    baseline_slope_ratio: float
    baseline_rough_ratio: float
    slope_deviation_dir: str
    rough_deviation_dir: str


@dataclass
class RecommendedPrescription:
    dilution_ratio: str
    ink_amount: str
    environment: str
    retest_time: str
    observation_focus: str
    confidence_score: float
    matched_history_ids: List[int] = field(default_factory=list)
    match_reasons: List[str] = field(default_factory=list)
    source: str = "auto"


_PRESET_RULES: Dict[str, Dict[str, Any]] = {
    "过浓_快": {
        "dilution_ratio": "原墨:蒸馏水 = 1:1.2 ~ 1:1.5（增加稀释度以减缓扩散）",
        "ink_amount": "单次点墨量减少 20%~30%，建议从中心向外缘渐变施墨",
        "environment": "温度 20~24℃，湿度 50%~55% RH，避免高湿环境加速扩散",
        "retest_time": "复测建议在点墨后 30s、60s、120s、240s、360s 记录，重点观察前期扩散速率",
        "observation_focus": "关注 0~120s 渗化斜率是否回落至基线的 1.1 倍以内；边缘毛糙度不应超过基线 1.4 倍",
    },
    "过稀_慢": {
        "dilution_ratio": "原墨:蒸馏水 = 1:0.7 ~ 1:0.9（减少稀释度，提高浓度以加快扩散）",
        "ink_amount": "单次点墨量增加 15%~20%，注意保持中心点墨均匀",
        "environment": "温度 22~26℃，湿度 55%~65% RH，适度提高湿度以促进渗化",
        "retest_time": "复测建议在点墨后 60s、120s、240s、480s、600s 记录，关注中后期扩散是否充分",
        "observation_focus": "关注 120~480s 渗化斜率是否回升至基线的 0.8 倍以上；最大半径是否达到纸型预期值",
    },
    "毛糙_偏糙": {
        "dilution_ratio": "在原稀释基础上增加蒸馏水 10%~15%，减少墨料颗粒聚集",
        "ink_amount": "保持或略减点墨量（-5%~-10%），避免边缘堆积",
        "environment": "温度 20~23℃，湿度 55%~60% RH，确保纸张纤维舒展均匀",
        "retest_time": "复测建议在 60s、180s、300s、480s 记录，重点记录边缘形态变化",
        "observation_focus": "边缘毛糙度均值应回落至基线 1.3 倍以内；目测边缘是否仍有明显锯齿状",
    },
    "毛糙_偏滑": {
        "dilution_ratio": "在原稀释基础上减少蒸馏水 5%~10%，保持适当墨料颗粒比例",
        "ink_amount": "保持或略增点墨量（+5%~+10%），保证边缘充分着色",
        "environment": "温度 22~25℃，湿度 50%~55% RH，避免过度光滑导致晕染不足",
        "retest_time": "复测建议在 60s、180s、300s、480s 记录，对比边缘着色深度",
        "observation_focus": "边缘毛糙度均值应回升至基线 0.7 倍以上；边缘过渡区是否自然",
    },
    "基线_半径大": {
        "dilution_ratio": "稀释比例提高 15%~20%，从源头控制扩散范围",
        "ink_amount": "点墨量减少 25%~35%，采用多次少量施墨法",
        "environment": "温度控制在 19~22℃，湿度 48%~52% RH，低温低湿抑制快速扩散",
        "retest_time": "复测建议 20s、40s、80s、160s、320s 高频记录，捕捉早期扩散峰值",
        "observation_focus": "各时间点半径偏差应控制在基线 +15% 以内；扩散速率拐点是否提前出现",
    },
    "基线_半径小": {
        "dilution_ratio": "稀释比例降低 10%~15%，增加有效墨浓度",
        "ink_amount": "点墨量增加 20%~30%，确保充足渗化驱动力",
        "environment": "温度 24~27℃，湿度 60%~65% RH，温湿双提促进纤维吸墨",
        "retest_time": "复测建议 120s、240s、480s、720s、900s 长周期记录，观察后期是否持续扩散",
        "observation_focus": "最大半径偏差应回升至基线 -15% 以上；最终渗化形态是否饱满",
    },
    "混合_多重异常": {
        "dilution_ratio": "先针对主异常调整（通常为扩散过快→增稀 10%），再据复测微调",
        "ink_amount": "点墨量调整 ±10%，采用试探量，避免多重调整相互干扰",
        "environment": "标准环境：温度 22±2℃，湿度 55%±5% RH，减少环境变量干扰",
        "retest_time": "全周期记录：30s、60s、120s、240s、480s、720s，必要时加测 10s、20s 早期点",
        "observation_focus": "优先控制主异常指标，关注各指标是否联动变化；判断是否存在纸性本身异常",
    },
    "通用_高风险": {
        "dilution_ratio": "维持现有稀释比例，先通过点墨量和环境进行保守调整",
        "ink_amount": "调整幅度不超过 ±10%，小步试验防止过度矫正",
        "environment": "严格控制温湿度波动 ≤±2℃/±5% RH，记录实时环境数据",
        "retest_time": "建议至少 5 个时间点，覆盖早期（<60s）、中期（60~300s）、后期（>300s）",
        "observation_focus": "综合观察扩散速率、半径范围、边缘毛糙三项指标是否协同变化；复测 2~3 次确认趋势",
    },
}


def _extract_sample_params(sample_id: int) -> Optional[PrescParams]:
    sample = db.get_sample_by_id(sample_id)
    if not sample:
        return None
    measurements = db.get_measurements_by_sample(sample_id)
    if not measurements:
        return None

    times = [float(m["adsorb_time"]) for m in measurements]
    radii = [float(m["radius"]) for m in measurements]
    rough = [float(m["roughness"]) for m in measurements]
    s_idx = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in s_idx]
    radii = [radii[i] for i in s_idx]
    rough = [rough[i] for i in s_idx]

    avg_slope = anomaly_detection._fit_slope(times, radii)
    avg_radius = statistics.mean(radii) if radii else 0.0
    avg_roughness = statistics.mean(rough) if rough else 0.0
    max_radius = max(radii) if radii else 0.0
    anomaly_count = sum(1 for m in measurements if m["is_anomaly"])
    anomaly_ratio = anomaly_count / len(measurements) if measurements else 0.0
    risk_flag = sample["risk_flag"] or "正常"

    baseline = anomaly_detection.get_paper_type_baseline(sample["paper_type"])
    baseline_slope_ratio = 1.0
    baseline_rough_ratio = 1.0
    slope_dir = "normal"
    rough_dir = "normal"

    if baseline and baseline.avg_slope > 0:
        baseline_slope_ratio = avg_slope / baseline.avg_slope
        if baseline_slope_ratio > 1.3:
            slope_dir = "fast"
        elif baseline_slope_ratio < 0.7:
            slope_dir = "slow"

    if baseline and baseline.avg_roughness > 0:
        baseline_rough_ratio = avg_roughness / baseline.avg_roughness
        if baseline_rough_ratio > 1.4:
            rough_dir = "rough"
        elif baseline_rough_ratio < 0.6:
            rough_dir = "smooth"

    anomaly_types = []
    for m in measurements:
        at = m["anomaly_type"]
        if at:
            for part in at.split(";"):
                part = part.strip()
                if part and part not in anomaly_types:
                    anomaly_types.append(part)
    primary_anomaly = anomaly_types[0] if anomaly_types else None

    return PrescParams(
        sample_id=sample_id,
        paper_type=sample["paper_type"],
        anomaly_type=primary_anomaly,
        avg_slope=avg_slope,
        avg_radius=avg_radius,
        avg_roughness=avg_roughness,
        max_radius=max_radius,
        anomaly_ratio=anomaly_ratio,
        risk_flag=risk_flag,
        baseline_slope_ratio=baseline_slope_ratio,
        baseline_rough_ratio=baseline_rough_ratio,
        slope_deviation_dir=slope_dir,
        rough_deviation_dir=rough_dir,
    )


def _match_rule_key(params: PrescParams) -> str:
    has_slope_issue = params.slope_deviation_dir in ("fast", "slow")
    has_rough_issue = params.rough_deviation_dir in ("rough", "smooth")
    has_baseline_radius = False
    baseline = anomaly_detection.get_paper_type_baseline(params.paper_type)
    if baseline and baseline.avg_radius > 0:
        ratio = params.max_radius / baseline.avg_radius
        if ratio > 1.3 or ratio < 0.7:
            has_baseline_radius = True

    issues = 0
    if has_slope_issue:
        issues += 1
    if has_rough_issue:
        issues += 1
    if has_baseline_radius:
        issues += 1

    if issues >= 2:
        return "混合_多重异常"
    if params.risk_flag == "高风险" and issues == 0:
        return "通用_高风险"

    if has_slope_issue:
        return "过浓_快" if params.slope_deviation_dir == "fast" else "过稀_慢"
    if has_rough_issue:
        return "毛糙_偏糙" if params.rough_deviation_dir == "rough" else "毛糙_偏滑"
    if has_baseline_radius and baseline:
        ratio = params.max_radius / baseline.avg_radius
        return "基线_半径大" if ratio > 1.0 else "基线_半径小"
    if params.risk_flag != "正常":
        return "通用_高风险"
    return "通用_高风险"


def _find_similar_historical_samples(
    params: PrescParams, top_k: int = 3
) -> List[Tuple[int, float, List[str]]]:
    paper_samples = db.get_samples_by_paper_type(params.paper_type)
    candidates: List[Tuple[int, float, List[str]]] = []

    for s in paper_samples:
        sid = s["id"]
        if sid == params.sample_id:
            continue
        prescs = db.get_prescriptions_by_sample(sid)
        if not prescs:
            continue
        has_record = any(
            db.get_experiment_records_by_prescription(p["id"]) for p in prescs
        )
        if not has_record:
            continue

        pms = db.get_measurements_by_sample(sid)
        if len(pms) < 3:
            continue
        t2 = [float(m["adsorb_time"]) for m in pms]
        r2 = [float(m["radius"]) for m in pms]
        ro2 = [float(m["roughness"]) for m in pms]
        s2 = sorted(range(len(t2)), key=lambda i: t2[i])
        t2 = [t2[i] for i in s2]
        r2 = [r2[i] for i in s2]
        ro2 = [ro2[i] for i in s2]

        slope2 = anomaly_detection._fit_slope(t2, r2)
        avg_r2 = statistics.mean(r2)
        avg_ro2 = statistics.mean(ro2)
        anom2 = sum(1 for m in pms if m["is_anomaly"]) / len(pms)

        score = 0.0
        reasons: List[str] = []
        total_weight = 0.0

        if params.avg_slope > 0 and slope2 > 0:
            w = 0.35
            diff = abs(params.avg_slope - slope2) / max(params.avg_slope, slope2)
            comp = max(0.0, 1.0 - diff)
            score += comp * w
            total_weight += w
            if comp > 0.7:
                reasons.append(f"渗化斜率相近（差异 {diff*100:.0f}%）")

        if params.avg_roughness > 0 and avg_ro2 > 0:
            w = 0.25
            diff = abs(params.avg_roughness - avg_ro2) / max(params.avg_roughness, avg_ro2)
            comp = max(0.0, 1.0 - diff)
            score += comp * w
            total_weight += w
            if comp > 0.7:
                reasons.append(f"毛糙度水平相近（差异 {diff*100:.0f}%）")

        if params.anomaly_ratio > 0 or anom2 > 0:
            w = 0.2
            diff = abs(params.anomaly_ratio - anom2)
            comp = max(0.0, 1.0 - diff * 2)
            score += comp * w
            total_weight += w
            if comp > 0.6:
                reasons.append(f"异常比例相近（当前 {params.anomaly_ratio:.0%}，历史 {anom2:.0%}）")

        if params.avg_radius > 0 and avg_r2 > 0:
            w = 0.2
            diff = abs(params.avg_radius - avg_r2) / max(params.avg_radius, avg_r2)
            comp = max(0.0, 1.0 - diff)
            score += comp * w
            total_weight += w

        if s["risk_flag"] == params.risk_flag:
            score += 0.15
            total_weight += 0.15
            reasons.append(f"风险等级同为「{params.risk_flag}」")

        normalized = score / total_weight if total_weight > 0 else 0.0
        if normalized > 0.45:
            candidates.append((sid, normalized, reasons))

    candidates.sort(key=lambda x: x[1], reverse=True)
    return candidates[:top_k]


def recommend_prescription(sample_id: int) -> Optional[RecommendedPrescription]:
    params = _extract_sample_params(sample_id)
    if params is None:
        return None

    rule_key = _match_rule_key(params)
    rule = _PRESET_RULES.get(rule_key, _PRESET_RULES["通用_高风险"])

    similar = _find_similar_historical_samples(params, top_k=3)
    matched_ids = [s[0] for s in similar]
    match_reasons: List[str] = []
    if similar:
        for sid, score, reasons in similar:
            s = db.get_sample_by_id(sid)
            sno = s["sample_no"] if s else str(sid)
            reason_parts = "；".join(reasons) if reasons else f"相似度 {score:.0%}"
            match_reasons.append(f"参考试样「{sno}」（相似度 {score*100:.0f}%）：{reason_parts}")

    if similar:
        avg_score = statistics.mean([s[1] for s in similar])
        confidence = round(0.4 + avg_score * 0.6, 3)
    else:
        confidence = 0.4

    if params.slope_deviation_dir == "fast":
        match_reasons.append(
            f"当前渗化斜率为纸型基线的 {params.baseline_slope_ratio:.1f} 倍，判定为扩散过快，参考「过浓/过快」类规则"
        )
    elif params.slope_deviation_dir == "slow":
        match_reasons.append(
            f"当前渗化斜率仅为纸型基线的 {params.baseline_slope_ratio:.1f} 倍，判定为扩散过慢，参考「过稀/过慢」类规则"
        )

    if params.rough_deviation_dir == "rough":
        match_reasons.append(
            f"当前毛糙度为纸型基线的 {params.baseline_rough_ratio:.1f} 倍，判定为边缘偏糙"
        )
    elif params.rough_deviation_dir == "smooth":
        match_reasons.append(
            f"当前毛糙度仅为纸型基线的 {params.baseline_rough_ratio:.1f} 倍，判定为边缘偏光滑"
        )

    if params.anomaly_ratio >= 0.4:
        match_reasons.append(f"异常点占比 {params.anomaly_ratio:.0%}，属高异常比例试样，建议保守小步调整")

    dilution_ratio = rule["dilution_ratio"]
    ink_amount = rule["ink_amount"]
    environment = rule["environment"]
    retest_time = rule["retest_time"]
    observation_focus = rule["observation_focus"]

    if similar:
        top_sid = similar[0][0]
        top_prescs = db.get_prescriptions_by_sample(top_sid)
        for tp in top_prescs:
            records = db.get_experiment_records_by_prescription(tp["id"])
            if records:
                best = max(records, key=lambda r: (
                    1 if r["diffusion_improved"] else 0,
                    1 if r["anomaly_reduced"] else 0,
                    r["effect_rating"] in ("优", "良"),
                ))
                if best and best.get("effect_rating") in ("优", "良"):
                    s_row = db.get_sample_by_id(top_sid)
                    sno = s_row["sample_no"] if s_row else str(top_sid)
                    match_reasons.append(
                        f"※ 历史试样「{sno}」使用类似方案后效果评级「{best['effect_rating']}」，"
                        f"异常比例从 {best['pre_anomaly_ratio']:.0%} 降至 {best['post_anomaly_ratio']:.0%}"
                    )
                    if tp["dilution_ratio"] and tp["dilution_ratio"] != dilution_ratio:
                        dilution_ratio = tp["dilution_ratio"]
                    if tp["ink_amount"] and tp["ink_amount"] != ink_amount:
                        ink_amount = tp["ink_amount"]
                    if tp["environment"] and tp["environment"] != environment:
                        environment = tp["environment"]
                    confidence = min(1.0, confidence + 0.1)
                    break

    return RecommendedPrescription(
        dilution_ratio=dilution_ratio,
        ink_amount=ink_amount,
        environment=environment,
        retest_time=retest_time,
        observation_focus=observation_focus,
        confidence_score=round(confidence, 3),
        matched_history_ids=matched_ids,
        match_reasons=match_reasons,
        source=f"rule:{rule_key}" if not similar else f"rule:{rule_key}+history",
    )


def generate_and_save_prescription(sample_id: int, remark: Optional[str] = None) -> Optional[int]:
    sample = db.get_sample_by_id(sample_id)
    if not sample:
        return None
    rec = recommend_prescription(sample_id)
    if rec is None:
        return None
    matched_ids_json = json.dumps(rec.matched_history_ids, ensure_ascii=False) if rec.matched_history_ids else None
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

    return db.create_prescription(
        sample_id=sample_id,
        paper_type=sample["paper_type"],
        anomaly_type=anomaly_type_text,
        dilution_ratio=rec.dilution_ratio,
        ink_amount=rec.ink_amount,
        environment=rec.environment,
        retest_time=rec.retest_time,
        observation_focus=rec.observation_focus,
        source=rec.source,
        matched_history_ids=matched_ids_json,
        confidence_score=rec.confidence_score,
        remark=remark,
    )


def capture_pre_metrics(sample_id: int) -> Dict[str, Any]:
    sample = db.get_sample_by_id(sample_id)
    measurements = db.get_measurements_by_sample(sample_id)
    result: Dict[str, Any] = {
        "pre_risk_flag": sample["risk_flag"] if sample else None,
        "pre_anomaly_ratio": 0.0,
        "pre_avg_slope": None,
        "pre_avg_radius": None,
        "pre_avg_roughness": None,
    }
    if not measurements:
        return result
    times = [float(m["adsorb_time"]) for m in measurements]
    radii = [float(m["radius"]) for m in measurements]
    rough = [float(m["roughness"]) for m in measurements]
    s_idx = sorted(range(len(times)), key=lambda i: times[i])
    times = [times[i] for i in s_idx]
    radii = [radii[i] for i in s_idx]
    rough = [rough[i] for i in s_idx]
    result["pre_avg_slope"] = anomaly_detection._fit_slope(times, radii)
    result["pre_avg_radius"] = statistics.mean(radii)
    result["pre_avg_roughness"] = statistics.mean(rough)
    anom = sum(1 for m in measurements if m["is_anomaly"])
    result["pre_anomaly_ratio"] = round(anom / len(measurements), 4)
    return result


def capture_post_metrics(sample_id: int) -> Dict[str, Any]:
    return capture_pre_metrics(sample_id)


def save_experiment_record(
    prescription_id: int,
    retest_sample_id: int,
    diffusion_improved: bool = False,
    anomaly_reduced: bool = False,
    paper_judgment_affected: bool = False,
    effect_rating: Optional[str] = None,
    operator: Optional[str] = None,
    remark: Optional[str] = None,
    execute_date: Optional[str] = None,
    retest_sample_no: Optional[str] = None,
    pre_sample_id: Optional[int] = None,
) -> Optional[int]:
    presc = db.get_prescription_by_id(prescription_id)
    if not presc:
        return None

    original_sample_id = pre_sample_id if pre_sample_id is not None else presc["sample_id"]
    pre_metrics = capture_pre_metrics(original_sample_id)
    post_metrics = capture_post_metrics(retest_sample_id)

    sample = db.get_sample_by_id(retest_sample_id)
    if not sample:
        return None
    if retest_sample_no is None:
        retest_sample_no = sample["sample_no"]

    return db.create_experiment_record(
        prescription_id=prescription_id,
        sample_id=retest_sample_id,
        paper_type=presc["paper_type"],
        batch_id=sample["batch_id"],
        retest_sample_no=retest_sample_no,
        execute_date=execute_date,
        diffusion_improved=1 if diffusion_improved else 0,
        anomaly_reduced=1 if anomaly_reduced else 0,
        paper_judgment_affected=1 if paper_judgment_affected else 0,
        pre_risk_flag=pre_metrics.get("pre_risk_flag"),
        post_risk_flag=post_metrics.get("pre_risk_flag"),
        pre_anomaly_ratio=pre_metrics.get("pre_anomaly_ratio", 0.0),
        post_anomaly_ratio=post_metrics.get("pre_anomaly_ratio", 0.0),
        pre_avg_slope=pre_metrics.get("pre_avg_slope"),
        post_avg_slope=post_metrics.get("pre_avg_slope"),
        pre_avg_radius=pre_metrics.get("pre_avg_radius"),
        post_avg_radius=post_metrics.get("pre_avg_radius"),
        pre_avg_roughness=pre_metrics.get("pre_avg_roughness"),
        post_avg_roughness=post_metrics.get("pre_avg_roughness"),
        effect_rating=effect_rating,
        operator=operator,
        remark=remark,
    )


def update_prescription_manually(
    prescription_id: int,
    dilution_ratio: Optional[str] = None,
    ink_amount: Optional[str] = None,
    environment: Optional[str] = None,
    retest_time: Optional[str] = None,
    observation_focus: Optional[str] = None,
    remark: Optional[str] = None,
) -> bool:
    presc = db.get_prescription_by_id(prescription_id)
    if not presc:
        return False
    try:
        db.update_prescription(
            prescription_id=prescription_id,
            dilution_ratio=dilution_ratio,
            ink_amount=ink_amount,
            environment=environment,
            retest_time=retest_time,
            observation_focus=observation_focus,
            remark=remark,
        )
        return True
    except Exception:
        return False


def get_prescription_with_records(prescription_id: int) -> Optional[Dict[str, Any]]:
    presc = db.get_prescription_by_id(prescription_id)
    if not presc:
        return None
    records = db.get_experiment_records_by_prescription(prescription_id)
    return {
        "prescription": dict(presc),
        "records": [dict(r) for r in records],
    }


def get_prescription_history_summary(
    sample_id: Optional[int] = None,
    batch_id: Optional[int] = None,
    paper_type: Optional[str] = None,
) -> List[Dict[str, Any]]:
    prescs = db.get_all_prescriptions(paper_type=paper_type, batch_id=batch_id)
    result = []
    for p in prescs:
        if sample_id is not None and p["sample_id"] != sample_id:
            continue
        d = dict(p)
        records = db.get_experiment_records_by_prescription(p["id"])
        d["record_count"] = len(records)
        d["best_rating"] = None
        d["last_effect_date"] = None
        if records:
            rating_order = {"优": 4, "良": 3, "中": 2, "差": 1}
            sorted_recs = sorted(
                records,
                key=lambda r: (
                    rating_order.get(r["effect_rating"], 0),
                    1 if r["diffusion_improved"] else 0,
                    1 if r["anomaly_reduced"] else 0,
                ),
                reverse=True,
            )
            best = sorted_recs[0]
            d["best_rating"] = best["effect_rating"]
            d["last_effect_date"] = best["execute_date"] or best["created_at"]
        result.append(d)
    return result
