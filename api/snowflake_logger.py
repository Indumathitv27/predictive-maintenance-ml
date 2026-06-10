import os
import uuid
import logging
import snowflake.connector
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)


# ── Snowflake connection ───────────────────────────────────
def get_connection():
    """
    Creates and returns a Snowflake connection
    using credentials from .env file.
    """
    try:
        conn = snowflake.connector.connect(
            account=os.getenv("SNOWFLAKE_ACCOUNT"),
            user=os.getenv("SNOWFLAKE_USER"),
            password=os.getenv("SNOWFLAKE_PASSWORD"),
            database=os.getenv("SNOWFLAKE_DATABASE"),
            schema=os.getenv("SNOWFLAKE_SCHEMA"),
            warehouse=os.getenv("SNOWFLAKE_WAREHOUSE")
        )
        return conn
    except Exception as e:
        logger.error(f"Snowflake connection failed: {e}")
        return None


# ── Log prediction to Snowflake ────────────────────────────
def log_prediction(
    engine_id: int,
    cycle: int,
    predicted_rul: float,
    risk_level: str,
    recommendation: str,
    model_version: str
) -> bool:
    """
    Logs a single prediction to Snowflake PREDICTIONS table.
    Returns True if successful, False otherwise.
    """
    conn = get_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        prediction_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO PREDICTIONS (
                prediction_id,
                engine_id,
                cycle,
                predicted_rul,
                risk_level,
                recommendation,
                model_version,
                predicted_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s,
                CURRENT_TIMESTAMP()
            )
        """, (
            prediction_id,
            engine_id,
            cycle,
            predicted_rul,
            risk_level,
            recommendation,
            model_version
        ))

        conn.commit()
        logger.info(
            f"Prediction logged to Snowflake | "
            f"engine_id={engine_id} | RUL={predicted_rul}"
        )
        return True

    except Exception as e:
        logger.error(f"Snowflake logging failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ── Log sensor reading to Snowflake ───────────────────────
def log_sensor_reading(reading: dict) -> bool:
    """
    Logs raw sensor readings to Snowflake
    SENSOR_READINGS table for audit trail.
    """
    conn = get_connection()
    if conn is None:
        return False

    try:
        cursor = conn.cursor()
        reading_id = str(uuid.uuid4())

        cursor.execute("""
            INSERT INTO SENSOR_READINGS (
                reading_id,
                engine_id,
                cycle,
                op_setting_1,
                op_setting_2,
                op_setting_3,
                sensor_2,
                sensor_3,
                sensor_4,
                sensor_7,
                sensor_11,
                sensor_12,
                created_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s,
                %s, %s, %s, %s, %s, %s,
                CURRENT_TIMESTAMP()
            )
        """, (
            reading_id,
            reading.get("engine_id"),
            reading.get("cycle"),
            reading.get("op_setting_1"),
            reading.get("op_setting_2"),
            reading.get("op_setting_3"),
            reading.get("sensor_2"),
            reading.get("sensor_3"),
            reading.get("sensor_4"),
            reading.get("sensor_7"),
            reading.get("sensor_11"),
            reading.get("sensor_12")
        ))

        conn.commit()
        logger.info(
            f"Sensor reading logged | "
            f"engine_id={reading.get('engine_id')}"
        )
        return True

    except Exception as e:
        logger.error(f"Sensor logging failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()


# ── Fetch prediction history ───────────────────────────────
def get_prediction_history(limit: int = 100) -> list:
    """
    Fetches recent predictions from Snowflake
    for dashboard display.
    """
    conn = get_connection()
    if conn is None:
        return []

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                engine_id,
                cycle,
                predicted_rul,
                risk_level,
                recommendation,
                model_version,
                predicted_at
            FROM PREDICTIONS
            ORDER BY predicted_at DESC
            LIMIT %s
        """, (limit,))

        rows = cursor.fetchall()
        columns = [
            "engine_id", "cycle", "predicted_rul",
            "risk_level", "recommendation",
            "model_version", "predicted_at"
        ]
        return [dict(zip(columns, row)) for row in rows]

    except Exception as e:
        logger.error(f"Fetch failed: {e}")
        return []
    finally:
        cursor.close()
        conn.close()


# ── Fetch risk summary ─────────────────────────────────────
def get_risk_summary() -> dict:
    """
    Returns count of predictions by risk level
    for dashboard summary metrics.
    """
    conn = get_connection()
    if conn is None:
        return {}

    try:
        cursor = conn.cursor()
        cursor.execute("""
            SELECT
                risk_level,
                COUNT(*) as count
            FROM PREDICTIONS
            GROUP BY risk_level
            ORDER BY count DESC
        """)

        rows = cursor.fetchall()
        return {row[0]: row[1] for row in rows}

    except Exception as e:
        logger.error(f"Risk summary failed: {e}")
        return {}
    finally:
        cursor.close()
        conn.close()


# ── Test connection ────────────────────────────────────────
def test_connection() -> bool:
    """
    Tests Snowflake connectivity.
    Returns True if connected successfully.
    """
    conn = get_connection()
    if conn is None:
        return False
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT CURRENT_VERSION()")
        version = cursor.fetchone()[0]
        print(f"Snowflake connected — version: {version}")
        return True
    except Exception as e:
        print(f"Connection test failed: {e}")
        return False
    finally:
        cursor.close()
        conn.close()