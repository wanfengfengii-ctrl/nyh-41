import sqlite3
import os
from datetime import datetime
from contextlib import contextmanager
from typing import Optional, List, Dict, Any, Tuple, Iterator

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ink_samples.db")


@contextmanager
def get_connection() -> Iterator[sqlite3.Connection]:
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def init_db() -> None:
    with get_connection() as conn:
        cursor = conn.cursor()

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS batches (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                batch_code TEXT UNIQUE NOT NULL,
                risk_level TEXT NOT NULL DEFAULT '正常',
                consecutive_anomalies INTEGER NOT NULL DEFAULT 0,
                remark TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS samples (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_no TEXT UNIQUE NOT NULL,
                batch_id INTEGER,
                paper_type TEXT NOT NULL,
                ink_date TEXT NOT NULL,
                risk_flag TEXT NOT NULL DEFAULT '正常',
                judgment TEXT,
                is_baseline INTEGER NOT NULL DEFAULT 0,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE SET NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS baseline_templates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                paper_type TEXT UNIQUE NOT NULL,
                baseline_sample_id INTEGER,
                avg_slope REAL,
                avg_radius REAL,
                avg_roughness REAL,
                baseline_times TEXT,
                baseline_radii TEXT,
                baseline_roughness TEXT,
                remark TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (baseline_sample_id) REFERENCES samples(id) ON DELETE SET NULL
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS measurements (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER NOT NULL,
                adsorb_time REAL NOT NULL,
                radius REAL NOT NULL,
                roughness REAL NOT NULL,
                is_anomaly INTEGER NOT NULL DEFAULT 0,
                anomaly_type TEXT,
                measured_at TEXT NOT NULL,
                FOREIGN KEY (sample_id) REFERENCES samples(id) ON DELETE CASCADE,
                UNIQUE(sample_id, adsorb_time)
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS import_failures (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_file TEXT NOT NULL,
                row_number INTEGER NOT NULL,
                raw_data TEXT NOT NULL,
                failure_reason TEXT NOT NULL,
                imported_at TEXT NOT NULL
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_batch ON samples(batch_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_samples_paper ON samples(paper_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_measurements_sample ON measurements(sample_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_measurements_time ON measurements(adsorb_time)")

        try:
            cursor.execute("ALTER TABLE samples ADD COLUMN is_baseline INTEGER NOT NULL DEFAULT 0")
        except sqlite3.OperationalError:
            pass

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS repair_prescriptions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                sample_id INTEGER NOT NULL,
                paper_type TEXT NOT NULL,
                anomaly_type TEXT,
                dilution_ratio TEXT,
                ink_amount TEXT,
                environment TEXT,
                retest_time TEXT,
                observation_focus TEXT,
                source TEXT DEFAULT 'auto',
                matched_history_ids TEXT,
                confidence_score REAL DEFAULT 0.0,
                remark TEXT,
                created_at TEXT NOT NULL,
                updated_at TEXT NOT NULL,
                FOREIGN KEY (sample_id) REFERENCES samples(id) ON DELETE CASCADE
            )
        """)

        cursor.execute("""
            CREATE TABLE IF NOT EXISTS experiment_records (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prescription_id INTEGER NOT NULL,
                sample_id INTEGER NOT NULL,
                paper_type TEXT NOT NULL,
                batch_id INTEGER,
                retest_sample_no TEXT,
                execute_date TEXT,
                diffusion_improved INTEGER DEFAULT 0,
                anomaly_reduced INTEGER DEFAULT 0,
                paper_judgment_affected INTEGER DEFAULT 0,
                pre_risk_flag TEXT,
                post_risk_flag TEXT,
                pre_anomaly_ratio REAL DEFAULT 0.0,
                post_anomaly_ratio REAL DEFAULT 0.0,
                pre_avg_slope REAL,
                post_avg_slope REAL,
                pre_avg_radius REAL,
                post_avg_radius REAL,
                pre_avg_roughness REAL,
                post_avg_roughness REAL,
                effect_rating TEXT,
                operator TEXT,
                remark TEXT,
                created_at TEXT NOT NULL,
                FOREIGN KEY (prescription_id) REFERENCES repair_prescriptions(id) ON DELETE CASCADE,
                FOREIGN KEY (sample_id) REFERENCES samples(id) ON DELETE CASCADE,
                FOREIGN KEY (batch_id) REFERENCES batches(id) ON DELETE SET NULL
            )
        """)

        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_sample ON repair_prescriptions(sample_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_prescriptions_paper ON repair_prescriptions(paper_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_prescription ON experiment_records(prescription_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_sample ON experiment_records(sample_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_paper ON experiment_records(paper_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_records_batch ON experiment_records(batch_id)")


def now_str() -> str:
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ==================== Batches ====================

def create_batch(batch_code: str, remark: str = "") -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        t = now_str()
        cursor.execute(
            "INSERT INTO batches (batch_code, remark, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (batch_code, remark, t, t)
        )
        return cursor.lastrowid


def get_all_batches() -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM batches ORDER BY created_at DESC").fetchall()


def get_batch_by_code(batch_code: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM batches WHERE batch_code = ?", (batch_code,)).fetchone()


def update_batch_risk(batch_id: int, risk_level: str, consecutive_anomalies: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE batches SET risk_level = ?, consecutive_anomalies = ?, updated_at = ? WHERE id = ?",
            (risk_level, consecutive_anomalies, now_str(), batch_id)
        )


def increment_batch_anomaly(batch_id: int) -> Tuple[str, int]:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT consecutive_anomalies FROM batches WHERE id = ?", (batch_id,)
        ).fetchone()
        new_count = (row["consecutive_anomalies"] if row else 0) + 1
        if new_count >= 3:
            risk_level = "高风险"
        elif new_count >= 2:
            risk_level = "中风险"
        else:
            risk_level = "低风险"
        conn.execute(
            "UPDATE batches SET risk_level = ?, consecutive_anomalies = ?, updated_at = ? WHERE id = ?",
            (risk_level, new_count, now_str(), batch_id)
        )
        return risk_level, new_count


def reset_batch_anomaly(batch_id: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE batches SET risk_level = '正常', consecutive_anomalies = 0, updated_at = ? WHERE id = ?",
            (now_str(), batch_id)
        )


# ==================== Samples ====================

def sample_no_exists(sample_no: str) -> bool:
    with get_connection() as conn:
        row = conn.execute("SELECT 1 FROM samples WHERE sample_no = ?", (sample_no,)).fetchone()
        return row is not None


def create_sample(sample_no: str, paper_type: str, ink_date: str,
                  batch_id: Optional[int] = None, is_baseline: int = 0) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        t = now_str()
        cursor.execute(
            """INSERT INTO samples
               (sample_no, batch_id, paper_type, ink_date, is_baseline, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sample_no, batch_id, paper_type, ink_date, is_baseline, t, t)
        )
        return cursor.lastrowid


def update_sample_judgment(sample_id: int, judgment: str, risk_flag: str = "正常") -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE samples SET judgment = ?, risk_flag = ?, updated_at = ? WHERE id = ?",
            (judgment, risk_flag, now_str(), sample_id)
        )


def get_all_samples() -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT s.*, b.batch_code FROM samples s
            LEFT JOIN batches b ON s.batch_id = b.id
            ORDER BY s.created_at DESC
        """).fetchall()


def get_sample_by_no(sample_no: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT s.*, b.batch_code FROM samples s
            LEFT JOIN batches b ON s.batch_id = b.id
            WHERE s.sample_no = ?
        """, (sample_no,)).fetchone()


def get_sample_by_id(sample_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT s.*, b.batch_code FROM samples s
            LEFT JOIN batches b ON s.batch_id = b.id
            WHERE s.id = ?
        """, (sample_id,)).fetchone()


def get_samples_by_batch(batch_id: int) -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("SELECT * FROM samples WHERE batch_id = ? ORDER BY sample_no",
                            (batch_id,)).fetchall()


def delete_sample(sample_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM samples WHERE id = ?", (sample_id,))


# ==================== Measurements ====================

def measurement_exists(sample_id: int, adsorb_time: float) -> bool:
    with get_connection() as conn:
        row = conn.execute(
            "SELECT 1 FROM measurements WHERE sample_id = ? AND adsorb_time = ?",
            (sample_id, adsorb_time)
        ).fetchone()
        return row is not None


def create_measurement(sample_id: int, adsorb_time: float, radius: float,
                       roughness: float, is_anomaly: int = 0,
                       anomaly_type: Optional[str] = None) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO measurements
               (sample_id, adsorb_time, radius, roughness, is_anomaly, anomaly_type, measured_at)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (sample_id, adsorb_time, radius, roughness, is_anomaly, anomaly_type, now_str())
        )
        return cursor.lastrowid


def get_measurements_by_sample(sample_id: int) -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM measurements WHERE sample_id = ? ORDER BY adsorb_time ASC",
            (sample_id,)
        ).fetchall()


def get_measurements_by_samples(sample_ids: List[int]) -> Dict[int, List[sqlite3.Row]]:
    if not sample_ids:
        return {}
    placeholders = ",".join("?" for _ in sample_ids)
    with get_connection() as conn:
        rows = conn.execute(
            f"SELECT * FROM measurements WHERE sample_id IN ({placeholders}) ORDER BY adsorb_time ASC",
            sample_ids
        ).fetchall()
    result: Dict[int, List[sqlite3.Row]] = {sid: [] for sid in sample_ids}
    for row in rows:
        result[row["sample_id"]].append(row)
    return result


def get_batch_id_of_sample(sample_id: int) -> Optional[int]:
    with get_connection() as conn:
        row = conn.execute("SELECT batch_id FROM samples WHERE id = ?", (sample_id,)).fetchone()
        return row["batch_id"] if row else None


# ==================== Import Failures ====================

def record_import_failure(source_file: str, row_number: int,
                          raw_data: str, failure_reason: str) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO import_failures
               (source_file, row_number, raw_data, failure_reason, imported_at)
               VALUES (?, ?, ?, ?, ?)""",
            (source_file, row_number, raw_data, failure_reason, now_str())
        )
        return cursor.lastrowid


def get_import_failures(limit: int = 500) -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM import_failures ORDER BY imported_at DESC LIMIT ?",
            (limit,)
        ).fetchall()


def clear_import_failures() -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM import_failures")


# ==================== Baseline Templates ====================

def set_sample_baseline(sample_id: int, is_baseline: int) -> None:
    with get_connection() as conn:
        conn.execute(
            "UPDATE samples SET is_baseline = ?, updated_at = ? WHERE id = ?",
            (1 if is_baseline else 0, now_str(), sample_id)
        )


def get_baselines_by_paper_type(paper_type: str) -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT s.*, b.batch_code FROM samples s
            LEFT JOIN batches b ON s.batch_id = b.id
            WHERE s.paper_type = ? AND s.is_baseline = 1
            ORDER BY s.created_at DESC
        """, (paper_type,)).fetchall()


def get_all_paper_types() -> List[str]:
    with get_connection() as conn:
        rows = conn.execute(
            "SELECT DISTINCT paper_type FROM samples ORDER BY paper_type"
        ).fetchall()
        return [r["paper_type"] for r in rows]


def get_samples_by_paper_type(paper_type: str) -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT s.*, b.batch_code FROM samples s
            LEFT JOIN batches b ON s.batch_id = b.id
            WHERE s.paper_type = ?
            ORDER BY s.created_at DESC
        """, (paper_type,)).fetchall()


def create_baseline_template(
    paper_type: str,
    baseline_sample_id: Optional[int],
    avg_slope: float,
    avg_radius: float,
    avg_roughness: float,
    baseline_times: str,
    baseline_radii: str,
    baseline_roughness: str,
    remark: str = ""
) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        t = now_str()
        cursor.execute(
            """INSERT INTO baseline_templates
               (paper_type, baseline_sample_id, avg_slope, avg_radius, avg_roughness,
                baseline_times, baseline_radii, baseline_roughness, remark, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (paper_type, baseline_sample_id, avg_slope, avg_radius, avg_roughness,
             baseline_times, baseline_radii, baseline_roughness, remark, t, t)
        )
        return cursor.lastrowid


def update_baseline_template(
    template_id: int,
    baseline_sample_id: Optional[int],
    avg_slope: float,
    avg_radius: float,
    avg_roughness: float,
    baseline_times: str,
    baseline_radii: str,
    baseline_roughness: str,
    remark: str = ""
) -> None:
    with get_connection() as conn:
        conn.execute(
            """UPDATE baseline_templates
               SET baseline_sample_id = ?, avg_slope = ?, avg_radius = ?, avg_roughness = ?,
                   baseline_times = ?, baseline_radii = ?, baseline_roughness = ?,
                   remark = ?, updated_at = ?
               WHERE id = ?""",
            (baseline_sample_id, avg_slope, avg_radius, avg_roughness,
             baseline_times, baseline_radii, baseline_roughness, remark, now_str(), template_id)
        )


def get_baseline_template_by_paper(paper_type: str) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT * FROM baseline_templates WHERE paper_type = ?",
            (paper_type,)
        ).fetchone()


def get_all_baseline_templates() -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute(
            "SELECT bt.*, s.sample_no FROM baseline_templates bt "
            "LEFT JOIN samples s ON bt.baseline_sample_id = s.id "
            "ORDER BY bt.paper_type ASC"
        ).fetchall()


def delete_baseline_template(template_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM baseline_templates WHERE id = ?", (template_id,))


def get_risk_aggregated_by_paper(batch_id: Optional[int] = None) -> List[Dict[str, Any]]:
    with get_connection() as conn:
        if batch_id is not None:
            rows = conn.execute("""
                SELECT
                    s.paper_type AS paper_type,
                    COUNT(*) AS total_samples,
                    SUM(CASE WHEN s.risk_flag = '高风险' THEN 1 ELSE 0 END) AS high_risk,
                    SUM(CASE WHEN s.risk_flag = '中风险' THEN 1 ELSE 0 END) AS mid_risk,
                    SUM(CASE WHEN s.risk_flag = '低风险' THEN 1 ELSE 0 END) AS low_risk,
                    SUM(CASE WHEN s.risk_flag = '正常' THEN 1 ELSE 0 END) AS normal,
                    SUM(CASE WHEN s.is_baseline = 1 THEN 1 ELSE 0 END) AS baseline_count
                FROM samples s
                WHERE s.batch_id = ?
                GROUP BY s.paper_type
                ORDER BY s.paper_type ASC
            """, (batch_id,)).fetchall()
        else:
            rows = conn.execute("""
                SELECT
                    s.paper_type AS paper_type,
                    COUNT(*) AS total_samples,
                    SUM(CASE WHEN s.risk_flag = '高风险' THEN 1 ELSE 0 END) AS high_risk,
                    SUM(CASE WHEN s.risk_flag = '中风险' THEN 1 ELSE 0 END) AS mid_risk,
                    SUM(CASE WHEN s.risk_flag = '低风险' THEN 1 ELSE 0 END) AS low_risk,
                    SUM(CASE WHEN s.risk_flag = '正常' THEN 1 ELSE 0 END) AS normal,
                    SUM(CASE WHEN s.is_baseline = 1 THEN 1 ELSE 0 END) AS baseline_count
                FROM samples s
                GROUP BY s.paper_type
                ORDER BY s.paper_type ASC
            """).fetchall()
        return [dict(r) for r in rows]


def get_paper_type_risk_summary() -> List[Dict[str, Any]]:
    with get_connection() as conn:
        rows = conn.execute("""
            SELECT
                s.paper_type AS paper_type,
                COUNT(*) AS total_samples,
                SUM(CASE WHEN s.risk_flag = '高风险' THEN 1 ELSE 0 END) AS high_risk,
                SUM(CASE WHEN s.risk_flag = '中风险' THEN 1 ELSE 0 END) AS mid_risk,
                SUM(CASE WHEN s.risk_flag = '低风险' THEN 1 ELSE 0 END) AS low_risk,
                SUM(CASE WHEN s.risk_flag = '正常' THEN 1 ELSE 0 END) AS normal,
                SUM(CASE WHEN s.is_baseline = 1 THEN 1 ELSE 0 END) AS baseline_count,
                ROUND(100.0 * SUM(CASE WHEN s.risk_flag != '正常' THEN 1 ELSE 0 END) /
                      NULLIF(COUNT(*), 0), 1) AS anomaly_rate_pct
            FROM samples s
            GROUP BY s.paper_type
            ORDER BY anomaly_rate_pct DESC, paper_type ASC
        """).fetchall()
        return [dict(r) for r in rows]


def get_baseline_matched_sample_ids(sample_id: int) -> List[int]:
    sample = get_sample_by_id(sample_id)
    if not sample:
        return []
    paper_type = sample["paper_type"]
    baselines = get_baselines_by_paper_type(paper_type)
    return [b["id"] for b in baselines if b["id"] != sample_id]


# ==================== Repair Prescriptions ====================

def create_prescription(
    sample_id: int,
    paper_type: str,
    anomaly_type: Optional[str] = None,
    dilution_ratio: Optional[str] = None,
    ink_amount: Optional[str] = None,
    environment: Optional[str] = None,
    retest_time: Optional[str] = None,
    observation_focus: Optional[str] = None,
    source: str = "auto",
    matched_history_ids: Optional[str] = None,
    confidence_score: float = 0.0,
    remark: Optional[str] = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        t = now_str()
        cursor.execute(
            """INSERT INTO repair_prescriptions
               (sample_id, paper_type, anomaly_type, dilution_ratio, ink_amount, environment,
                retest_time, observation_focus, source, matched_history_ids, confidence_score,
                remark, created_at, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (sample_id, paper_type, anomaly_type, dilution_ratio, ink_amount, environment,
             retest_time, observation_focus, source, matched_history_ids, confidence_score,
             remark, t, t)
        )
        return cursor.lastrowid


def update_prescription(
    prescription_id: int,
    dilution_ratio: Optional[str] = None,
    ink_amount: Optional[str] = None,
    environment: Optional[str] = None,
    retest_time: Optional[str] = None,
    observation_focus: Optional[str] = None,
    remark: Optional[str] = None,
) -> None:
    fields = []
    values = []
    for name, val in [
        ("dilution_ratio", dilution_ratio),
        ("ink_amount", ink_amount),
        ("environment", environment),
        ("retest_time", retest_time),
        ("observation_focus", observation_focus),
        ("remark", remark),
    ]:
        if val is not None:
            fields.append(f"{name} = ?")
            values.append(val)
    if not fields:
        return
    fields.append("updated_at = ?")
    values.append(now_str())
    values.append(prescription_id)
    with get_connection() as conn:
        conn.execute(f"UPDATE repair_prescriptions SET {', '.join(fields)} WHERE id = ?", values)


def get_prescriptions_by_sample(sample_id: int) -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT rp.*, s.sample_no
            FROM repair_prescriptions rp
            LEFT JOIN samples s ON rp.sample_id = s.id
            WHERE rp.sample_id = ?
            ORDER BY rp.created_at DESC
        """, (sample_id,)).fetchall()


def get_prescription_by_id(prescription_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT rp.*, s.sample_no, s.batch_id, b.batch_code
            FROM repair_prescriptions rp
            LEFT JOIN samples s ON rp.sample_id = s.id
            LEFT JOIN batches b ON s.batch_id = b.id
            WHERE rp.id = ?
        """, (prescription_id,)).fetchone()


def get_all_prescriptions(
    paper_type: Optional[str] = None,
    batch_id: Optional[int] = None,
) -> List[sqlite3.Row]:
    with get_connection() as conn:
        sql = """
            SELECT rp.*, s.sample_no, s.batch_id, b.batch_code
            FROM repair_prescriptions rp
            LEFT JOIN samples s ON rp.sample_id = s.id
            LEFT JOIN batches b ON s.batch_id = b.id
            WHERE 1=1
        """
        params: List[Any] = []
        if paper_type:
            sql += " AND rp.paper_type = ?"
            params.append(paper_type)
        if batch_id is not None:
            sql += " AND s.batch_id = ?"
            params.append(batch_id)
        sql += " ORDER BY rp.created_at DESC"
        return conn.execute(sql, params).fetchall()


def delete_prescription(prescription_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM repair_prescriptions WHERE id = ?", (prescription_id,))


# ==================== Experiment Records ====================

def create_experiment_record(
    prescription_id: int,
    sample_id: int,
    paper_type: str,
    batch_id: Optional[int] = None,
    retest_sample_no: Optional[str] = None,
    execute_date: Optional[str] = None,
    diffusion_improved: int = 0,
    anomaly_reduced: int = 0,
    paper_judgment_affected: int = 0,
    pre_risk_flag: Optional[str] = None,
    post_risk_flag: Optional[str] = None,
    pre_anomaly_ratio: float = 0.0,
    post_anomaly_ratio: float = 0.0,
    pre_avg_slope: Optional[float] = None,
    post_avg_slope: Optional[float] = None,
    pre_avg_radius: Optional[float] = None,
    post_avg_radius: Optional[float] = None,
    pre_avg_roughness: Optional[float] = None,
    post_avg_roughness: Optional[float] = None,
    effect_rating: Optional[str] = None,
    operator: Optional[str] = None,
    remark: Optional[str] = None,
) -> int:
    with get_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO experiment_records
               (prescription_id, sample_id, paper_type, batch_id, retest_sample_no, execute_date,
                diffusion_improved, anomaly_reduced, paper_judgment_affected,
                pre_risk_flag, post_risk_flag, pre_anomaly_ratio, post_anomaly_ratio,
                pre_avg_slope, post_avg_slope, pre_avg_radius, post_avg_radius,
                pre_avg_roughness, post_avg_roughness, effect_rating, operator, remark, created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (prescription_id, sample_id, paper_type, batch_id, retest_sample_no, execute_date,
             diffusion_improved, anomaly_reduced, paper_judgment_affected,
             pre_risk_flag, post_risk_flag, pre_anomaly_ratio, post_anomaly_ratio,
             pre_avg_slope, post_avg_slope, pre_avg_radius, post_avg_radius,
             pre_avg_roughness, post_avg_roughness, effect_rating, operator, remark, now_str())
        )
        return cursor.lastrowid


def get_experiment_records_by_prescription(prescription_id: int) -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT er.*, s.sample_no, rp.dilution_ratio, rp.ink_amount, rp.environment,
                   rp.retest_time AS presc_retest_time, rp.observation_focus,
                   b.batch_code
            FROM experiment_records er
            LEFT JOIN samples s ON er.sample_id = s.id
            LEFT JOIN repair_prescriptions rp ON er.prescription_id = rp.id
            LEFT JOIN batches b ON er.batch_id = b.id
            WHERE er.prescription_id = ?
            ORDER BY er.created_at DESC
        """, (prescription_id,)).fetchall()


def get_experiment_records_by_sample(sample_id: int) -> List[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT er.*, s.sample_no, rp.dilution_ratio, rp.ink_amount, rp.environment,
                   rp.retest_time AS presc_retest_time, rp.observation_focus,
                   b.batch_code
            FROM experiment_records er
            LEFT JOIN samples s ON er.sample_id = s.id
            LEFT JOIN repair_prescriptions rp ON er.prescription_id = rp.id
            LEFT JOIN batches b ON er.batch_id = b.id
            WHERE er.sample_id = ?
            ORDER BY er.created_at DESC
        """, (sample_id,)).fetchall()


def get_all_experiment_records(
    paper_type: Optional[str] = None,
    batch_id: Optional[int] = None,
    sample_id: Optional[int] = None,
) -> List[sqlite3.Row]:
    with get_connection() as conn:
        sql = """
            SELECT er.*, s.sample_no, rp.dilution_ratio, rp.ink_amount, rp.environment,
                   rp.retest_time AS presc_retest_time, rp.observation_focus,
                   b.batch_code
            FROM experiment_records er
            LEFT JOIN samples s ON er.sample_id = s.id
            LEFT JOIN repair_prescriptions rp ON er.prescription_id = rp.id
            LEFT JOIN batches b ON er.batch_id = b.id
            WHERE 1=1
        """
        params: List[Any] = []
        if paper_type:
            sql += " AND er.paper_type = ?"
            params.append(paper_type)
        if batch_id is not None:
            sql += " AND er.batch_id = ?"
            params.append(batch_id)
        if sample_id is not None:
            sql += " AND er.sample_id = ?"
            params.append(sample_id)
        sql += " ORDER BY er.created_at DESC"
        return conn.execute(sql, params).fetchall()


def get_experiment_record_by_id(record_id: int) -> Optional[sqlite3.Row]:
    with get_connection() as conn:
        return conn.execute("""
            SELECT er.*, s.sample_no, rp.dilution_ratio, rp.ink_amount, rp.environment,
                   rp.retest_time AS presc_retest_time, rp.observation_focus,
                   b.batch_code
            FROM experiment_records er
            LEFT JOIN samples s ON er.sample_id = s.id
            LEFT JOIN repair_prescriptions rp ON er.prescription_id = rp.id
            LEFT JOIN batches b ON er.batch_id = b.id
            WHERE er.id = ?
        """, (record_id,)).fetchone()


def delete_experiment_record(record_id: int) -> None:
    with get_connection() as conn:
        conn.execute("DELETE FROM experiment_records WHERE id = ?", (record_id,))


def get_prescription_hit_stats(paper_type: Optional[str] = None) -> Dict[str, Any]:
    with get_connection() as conn:
        sql = """
            SELECT
                COUNT(DISTINCT rp.id) AS total_prescriptions,
                COUNT(DISTINCT er.id) AS total_records,
                SUM(CASE WHEN er.diffusion_improved = 1 THEN 1 ELSE 0 END) AS diffusion_improved_count,
                SUM(CASE WHEN er.anomaly_reduced = 1 THEN 1 ELSE 0 END) AS anomaly_reduced_count,
                SUM(CASE WHEN er.paper_judgment_affected = 0 THEN 1 ELSE 0 END) AS no_paper_impact_count,
                SUM(CASE WHEN er.effect_rating = '优' THEN 1 ELSE 0 END) AS rating_excellent,
                SUM(CASE WHEN er.effect_rating = '良' THEN 1 ELSE 0 END) AS rating_good,
                SUM(CASE WHEN er.effect_rating = '中' THEN 1 ELSE 0 END) AS rating_mid,
                SUM(CASE WHEN er.effect_rating = '差' THEN 1 ELSE 0 END) AS rating_poor,
                AVG(CASE WHEN er.pre_anomaly_ratio > 0
                    THEN (er.pre_anomaly_ratio - er.post_anomaly_ratio) / er.pre_anomaly_ratio * 100
                    ELSE NULL END) AS avg_anomaly_reduction_pct
            FROM repair_prescriptions rp
            LEFT JOIN experiment_records er ON rp.id = er.prescription_id
            WHERE 1=1
        """
        params: List[Any] = []
        if paper_type:
            sql += " AND rp.paper_type = ?"
            params.append(paper_type)
        row = conn.execute(sql, params).fetchone()
        if not row:
            return {}
        d = dict(row)
        total = d.get("total_records") or 0
        if total > 0:
            d["hit_rate_diffusion_pct"] = round(
                (d.get("diffusion_improved_count") or 0) / total * 100, 1
            )
            d["hit_rate_anomaly_pct"] = round(
                (d.get("anomaly_reduced_count") or 0) / total * 100, 1
            )
            d["rate_no_paper_impact_pct"] = round(
                (d.get("no_paper_impact_count") or 0) / total * 100, 1
            )
            good_plus = (d.get("rating_excellent") or 0) + (d.get("rating_good") or 0)
            d["overall_satisfaction_pct"] = round(good_plus / total * 100, 1)
        else:
            d["hit_rate_diffusion_pct"] = 0.0
            d["hit_rate_anomaly_pct"] = 0.0
            d["rate_no_paper_impact_pct"] = 0.0
            d["overall_satisfaction_pct"] = 0.0
        return d


def get_risk_trend_records(
    paper_type: Optional[str] = None,
    batch_id: Optional[int] = None,
) -> List[Dict[str, Any]]:
    records = get_all_experiment_records(paper_type=paper_type, batch_id=batch_id)
    result = []
    for r in records:
        d = dict(r)
        risk_order = {"正常": 0, "低风险": 1, "中风险": 2, "高风险": 3}
        d["pre_risk_level"] = risk_order.get(d.get("pre_risk_flag") or "正常", 0)
        d["post_risk_level"] = risk_order.get(d.get("post_risk_flag") or "正常", 0)
        d["risk_delta"] = d["post_risk_level"] - d["pre_risk_level"]
        result.append(d)
    result.sort(key=lambda x: x.get("created_at") or "")
    return result

