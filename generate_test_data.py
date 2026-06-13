import csv
import random
import math
import os
from datetime import datetime, timedelta


def generate_samples(num_samples: int = 6, seed: int = 42) -> list:
    random.seed(seed)
    samples = []
    paper_types = ["宣纸", "毛边纸", "皮纸", "竹纸", "麻纸", "高丽纸"]
    now = datetime.now()

    for i in range(num_samples):
        sample_no = f"S{2024001 + i}"
        paper_type = random.choice(paper_types)
        ink_date = (now - timedelta(days=random.randint(0, 30))).strftime("%Y-%m-%d")
        batch_code = f"BATCH-2024-{random.randint(1, 3):02d}"
        samples.append({
            "sample_no": sample_no,
            "paper_type": paper_type,
            "ink_date": ink_date,
            "batch_code": batch_code,
        })
    return samples


def generate_measurements(sample_no: str, base_slope: float = 0.6,
                          base_radius: float = 2.0, base_roughness: float = 1.5,
                          noise: float = 0.08, num_points: int = 10,
                          inject_anomaly: bool = False) -> list:
    measurements = []
    times = [5.0, 10.0, 15.0, 20.0, 30.0, 45.0, 60.0, 90.0, 120.0, 180.0]
    times = times[:num_points]

    for idx, t in enumerate(times):
        radius = base_radius + base_slope * math.sqrt(t) * 0.8
        radius += random.gauss(0, noise * radius)

        roughness = base_roughness + random.gauss(0, 0.15 * base_roughness)
        roughness = max(0.1, roughness)

        if inject_anomaly and idx == 4:
            radius *= 1.8
            roughness *= 2.0

        measurements.append({
            "sample_no": sample_no,
            "adsorb_time": round(t, 1),
            "radius": round(radius, 3),
            "roughness": round(roughness, 3),
        })

    return measurements


def generate_sample_csv(file_path: str, num_samples: int = 6, mark_baselines: bool = True):
    samples = generate_samples(num_samples)
    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)

    with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["试样编号", "纸张类型", "点墨日期", "批次号", "是否基线"])
        seen_papers = set()
        for s in samples:
            is_baseline = ""
            if mark_baselines and s["paper_type"] not in seen_papers:
                is_baseline = "是"
                seen_papers.add(s["paper_type"])
            writer.writerow([s["sample_no"], s["paper_type"], s["ink_date"], s["batch_code"], is_baseline])

    print(f"试样 CSV 已生成: {file_path} ({len(samples)} 条)")
    return samples


def generate_measurement_csv(file_path: str, samples: list,
                             inject_anomalies: bool = True):
    all_meas = []
    profiles = [
        {"base_slope": 0.8, "base_radius": 2.5, "base_roughness": 1.2},
        {"base_slope": 0.5, "base_radius": 1.8, "base_roughness": 2.0},
        {"base_slope": 1.1, "base_radius": 3.0, "base_roughness": 0.8},
        {"base_slope": 0.35, "base_radius": 1.2, "base_roughness": 1.5},
        {"base_slope": 0.7, "base_radius": 2.2, "base_roughness": 2.8},
        {"base_slope": 0.9, "base_radius": 2.8, "base_roughness": 1.0},
    ]

    for i, s in enumerate(samples):
        profile = profiles[i % len(profiles)]
        meas = generate_measurements(
            s["sample_no"],
            base_slope=profile["base_slope"],
            base_radius=profile["base_radius"],
            base_roughness=profile["base_roughness"],
            inject_anomaly=inject_anomalies and i in (1, 4),
        )
        all_meas.extend(meas)

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["试样编号", "吸附时间", "扩散半径", "边缘毛糙度"])
        for m in all_meas:
            writer.writerow([m["sample_no"], m["adsorb_time"], m["radius"], m["roughness"]])

    print(f"测量 CSV 已生成: {file_path} ({len(all_meas)} 条)")
    return all_meas


def generate_mixed_csv(file_path: str, num_samples: int = 4, mark_baselines: bool = True):
    samples = generate_samples(num_samples, seed=123)
    all_rows = []

    profiles = [
        {"base_slope": 0.75, "base_radius": 2.3, "base_roughness": 1.4},
        {"base_slope": 0.45, "base_radius": 1.6, "base_roughness": 2.2},
        {"base_slope": 1.0, "base_radius": 2.9, "base_roughness": 0.9},
        {"base_slope": 0.4, "base_radius": 1.4, "base_roughness": 1.8},
    ]

    seen_papers = set()
    for i, s in enumerate(samples):
        profile = profiles[i % len(profiles)]
        meas = generate_measurements(
            s["sample_no"],
            base_slope=profile["base_slope"],
            base_radius=profile["base_radius"],
            base_roughness=profile["base_roughness"],
            num_points=8,
        )
        is_baseline = ""
        if mark_baselines and s["paper_type"] not in seen_papers:
            is_baseline = "1"
            seen_papers.add(s["paper_type"])
        for m in meas:
            all_rows.append({
                "sample_no": s["sample_no"],
                "paper_type": s["paper_type"],
                "ink_date": s["ink_date"],
                "batch_code": s["batch_code"],
                "is_baseline": is_baseline,
                "adsorb_time": m["adsorb_time"],
                "radius": m["radius"],
                "roughness": m["roughness"],
            })
            is_baseline = ""

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["试样编号", "纸张类型", "点墨日期", "批次号", "是否基线",
                         "吸附时间(s)", "扩散半径(mm)", "边缘毛糙度"])
        for r in all_rows:
            writer.writerow([
                r["sample_no"], r["paper_type"], r["ink_date"], r["batch_code"], r["is_baseline"],
                r["adsorb_time"], r["radius"], r["roughness"]
            ])

    print(f"混合 CSV 已生成: {file_path} ({len(all_rows)} 条)")
    return all_rows


def generate_invalid_csv(file_path: str):
    rows = [
        ["试样编号", "纸张类型", "点墨日期", "批次号", "吸附时间", "扩散半径", "边缘毛糙度"],
        ["", "宣纸", "2024-01-15", "BATCH-TEST-1", "10", "2.5", "1.2"],
        ["S-INV-001", "", "2024-01-15", "BATCH-TEST-1", "10", "2.5", "1.2"],
        ["S-INV-002", "毛边纸", "bad-date", "BATCH-TEST-1", "10", "2.5", "1.2"],
        ["S-INV-003", "皮纸", "2024-01-15", "BATCH-TEST-1", "-5", "2.5", "1.2"],
        ["S-INV-003", "皮纸", "2024-01-15", "BATCH-TEST-1", "10", "0", "1.2"],
        ["S-INV-004", "竹纸", "2024-01-15", "BATCH-TEST-1", "abc", "2.5", "1.2"],
    ]

    os.makedirs(os.path.dirname(file_path) if os.path.dirname(file_path) else ".", exist_ok=True)
    with open(file_path, "w", encoding="utf-8-sig", newline="") as f:
        writer = csv.writer(f)
        writer.writerows(rows)

    print(f"无效数据 CSV 已生成: {file_path} ({len(rows) - 1} 条无效记录)")


def generate_all_test_data(output_dir: str = "test_data"):
    os.makedirs(output_dir, exist_ok=True)

    print("=" * 50)
    print("生成测试数据...")
    print("=" * 50)

    samples = generate_sample_csv(os.path.join(output_dir, "samples.csv"))

    generate_measurement_csv(os.path.join(output_dir, "measurements.csv"), samples)

    generate_mixed_csv(os.path.join(output_dir, "mixed.csv"))

    generate_invalid_csv(os.path.join(output_dir, "invalid_data.csv"))

    print("=" * 50)
    print("测试数据生成完成！")
    print(f"输出目录: {os.path.abspath(output_dir)}")
    print("=" * 50)


if __name__ == "__main__":
    generate_all_test_data()
