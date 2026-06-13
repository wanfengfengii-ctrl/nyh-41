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
