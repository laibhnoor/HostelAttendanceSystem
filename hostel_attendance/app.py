from __future__ import annotations

import logging
from typing import Dict, List

from flask import Flask, jsonify

from config import LOG_LEVEL
from database import get_all_students, get_attendance_today, init_db

logger = logging.getLogger(__name__)
logger.setLevel(LOG_LEVEL)

app = Flask(__name__)


def _format_attendance(records: List[Dict[str, str]]) -> List[Dict[str, str]]:
    """Format attendance records for display.

    Args:
        records: Attendance records.

    Returns:
        Formatted attendance records.
    """
    formatted = []
    for record in records:
        formatted.append(
            {
                "student_id": record["student_id"],
                "name": record["name"],
                "timestamp": str(record["timestamp"]),
                "confidence": record["confidence"],
            }
        )
    return formatted


@app.route("/")
def index() -> str:
    """Render today's attendance table.

    Args:
        None.

    Returns:
        HTML string.
    """
    try:
        records = _format_attendance(get_attendance_today())
    except Exception as exc:
        logger.error("Failed to load attendance: %s", exc)
        records = []

    rows = ""
    for record in records:
        confidence_pct = f"{float(record['confidence']) * 100:.1f}%"
        rows += (
            "<tr>"
            f"<td>{record['student_id']}</td>"
            f"<td>{record['name']}</td>"
            f"<td>{record['timestamp']}</td>"
            f"<td>{confidence_pct}</td>"
            "</tr>"
        )

    html = f"""
    <!doctype html>
    <html>
    <head>
        <meta http-equiv=\"refresh\" content=\"10\">
        <title>Hostel Attendance</title>
        <style>
            body {{ font-family: Arial, sans-serif; background: #f4f6f8; }}
            table {{ width: 90%; margin: 20px auto; border-collapse: collapse; }}
            th {{ background: #222; color: #fff; padding: 12px; }}
            td {{ padding: 10px; text-align: center; }}
            tr:nth-child(even) {{ background: #e8ecef; }}
            tr:nth-child(odd) {{ background: #ffffff; }}
            h1 {{ text-align: center; color: #222; }}
        </style>
    </head>
    <body>
        <h1>Today's Attendance</h1>
        <table>
            <tr>
                <th>Student ID</th>
                <th>Name</th>
                <th>Time</th>
                <th>Confidence</th>
            </tr>
            {rows}
        </table>
    </body>
    </html>
    """
    return html


@app.route("/api/attendance/today")
def api_attendance_today():
    """Return today's attendance records as JSON.

    Args:
        None.

    Returns:
        JSON response.
    """
    return jsonify(_format_attendance(get_attendance_today()))


@app.route("/api/students")
def api_students():
    """Return all enrolled students as JSON.

    Args:
        None.

    Returns:
        JSON response.
    """
    return jsonify(get_all_students())


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
