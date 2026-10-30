# -*- coding: utf-8 -*-
"""SQLite-backed review history and lightweight human workflow."""

from __future__ import annotations

import csv
import io
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .schemas import ReviewResponse


ACTION_LABELS = {
    "allow": "放行",
    "limit": "限流",
    "review": "人工复核",
    "block": "下架",
    "unavailable": "流程不可用",
}
ACTION_SEVERITY = {
    "allow": 0,
    "limit": 1,
    "review": 2,
    "block": 3,
}
ACTION_LEVELS = {
    "allow": "L0",
    "limit": "L1",
    "review": "L2",
    "block": "L3",
}


class ReviewStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _initialize(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS review_records (
                    id TEXT PRIMARY KEY,
                    workspace_id TEXT NOT NULL DEFAULT 'default',
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    filename TEXT NOT NULL,
                    file_extension TEXT NOT NULL,
                    image_path TEXT NOT NULL,
                    action TEXT NOT NULL,
                    final_action TEXT NOT NULL,
                    effective_risk TEXT NOT NULL,
                    risk_raw TEXT,
                    domain TEXT NOT NULL,
                    queue TEXT NOT NULL,
                    manual_status TEXT NOT NULL,
                    adjustment_direction TEXT NOT NULL,
                    feedback_reason TEXT NOT NULL,
                    response_json TEXT NOT NULL
                )
                """
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_queue "
                "ON review_records(queue, updated_at)"
            )
            connection.execute(
                "CREATE INDEX IF NOT EXISTS idx_review_filters "
                "ON review_records(domain, final_action, effective_risk)"
            )
            columns = {
                row["name"]
                for row in connection.execute(
                    "PRAGMA table_info(review_records)"
                ).fetchall()
            }
            if "adjustment_direction" not in columns:
                connection.execute(
                    "ALTER TABLE review_records ADD COLUMN "
                    "adjustment_direction TEXT NOT NULL DEFAULT ''"
                )
            if "workspace_id" not in columns:
                connection.execute(
                    "ALTER TABLE review_records ADD COLUMN "
                    "workspace_id TEXT NOT NULL DEFAULT 'default'"
                )

    @staticmethod
    def _now() -> str:
        return datetime.now(timezone.utc).isoformat()

    @staticmethod
    def _display_filename(filename: str) -> str:
        return Path(str(filename or "image")).name[:255]

    def save_response(
        self,
        response: ReviewResponse,
        *,
        filename: str,
        image_path: str = "",
        workspace_id: str = "default",
    ) -> None:
        result = response.result
        action = result.action if result else "unavailable"
        domain = result.domain if result else "unknown"
        effective = result.effective_risk if result else "无有效预测"
        risk_raw = result.risk_raw if result else None
        queue = "human" if action == "review" else "auto"
        manual_status = "pending" if action == "review" else "none"
        now = self._now()
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO review_records (
                    id, created_at, updated_at, filename, file_extension,
                    image_path, action, final_action, effective_risk,
                    risk_raw, domain, queue, manual_status, adjustment_direction,
                    feedback_reason, response_json, workspace_id
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(id) DO UPDATE SET
                    updated_at=excluded.updated_at,
                    action=excluded.action,
                    final_action=excluded.final_action,
                    effective_risk=excluded.effective_risk,
                    risk_raw=excluded.risk_raw,
                    domain=excluded.domain,
                    queue=excluded.queue,
                    response_json=excluded.response_json,
                    workspace_id=excluded.workspace_id
                """,
                (
                    response.id,
                    now,
                    now,
                    self._display_filename(filename),
                    Path(filename).suffix.lower()[:16],
                    image_path,
                    action,
                    action,
                    effective,
                    risk_raw,
                    domain,
                    queue,
                    manual_status,
                    "",
                    "",
                    response.model_dump_json(),
                    str(workspace_id or "default")[:64],
                ),
            )

    def record_feedback(
        self,
        record_id: str,
        *,
        resolution: str,
        final_action: str,
        adjustment_direction: str = "",
        reason: str,
    ) -> dict[str, Any] | None:
        reason = reason.strip()
        if final_action not in ACTION_SEVERITY:
            raise ValueError("invalid_final_action")
        if resolution == "reject":
            if not reason:
                raise ValueError("feedback_reason_required")
            final_action = "review"
        elif resolution == "adjust":
            if not reason:
                raise ValueError("feedback_reason_required")
            if adjustment_direction not in {"downgrade", "upgrade"}:
                raise ValueError("adjustment_direction_required")
        elif resolution == "confirm":
            if final_action == "review":
                raise ValueError("confirm_requires_action")
            reason = reason or "确认系统建议处置"
        else:
            raise ValueError("invalid_resolution")
        now = self._now()
        with self._connect() as connection:
            row = connection.execute(
                "SELECT id, final_action FROM review_records WHERE id = ?",
                (record_id,),
            ).fetchone()
            if row is None:
                return None
            if resolution == "adjust":
                baseline = ACTION_SEVERITY.get(row["final_action"])
                target = ACTION_SEVERITY[final_action]
                if baseline is None or target == baseline:
                    raise ValueError("adjustment_requires_action_change")
                if adjustment_direction == "downgrade" and target > baseline:
                    raise ValueError("adjustment_direction_mismatch")
                if adjustment_direction == "upgrade" and target < baseline:
                    raise ValueError("adjustment_direction_mismatch")
            connection.execute(
                """
                UPDATE review_records
                SET updated_at = ?, final_action = ?, queue = 'human',
                    manual_status = ?, adjustment_direction = ?,
                    feedback_reason = ?
                WHERE id = ?
                """,
                (
                    now,
                    final_action,
                    {
                        "confirm": "confirmed",
                        "adjust": "adjusted",
                        "reject": "rejected",
                    }[resolution],
                    adjustment_direction if resolution == "adjust" else "",
                    reason[:1000],
                    record_id,
                ),
            )
        return self.get_record(record_id)

    @staticmethod
    def _record(row: sqlite3.Row) -> dict[str, Any]:
        response = json.loads(row["response_json"])
        return {
            "id": row["id"],
            "workspace_id": row["workspace_id"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
            "filename": row["filename"],
            "file_extension": row["file_extension"],
            "action": row["action"],
            "action_label": ACTION_LABELS.get(row["action"], row["action"]),
            "final_action": row["final_action"],
            "final_action_label": ACTION_LABELS.get(
                row["final_action"], row["final_action"]
            ),
            "current_level": ACTION_LEVELS.get(
                row["final_action"], "X"
            ),
            "current_level_label": (
                f"{ACTION_LEVELS[row['final_action']]} · "
                f"{ACTION_LABELS[row['final_action']]}"
                if row["final_action"] in ACTION_LEVELS
                else row["final_action"]
            ),
            "effective_risk": row["effective_risk"],
            "risk_raw": row["risk_raw"],
            "domain": row["domain"],
            "queue": row["queue"],
            "manual_status": row["manual_status"],
            "adjustment_direction": row["adjustment_direction"],
            "feedback_reason": row["feedback_reason"],
            "has_image": bool(row["image_path"]),
            "image_url": f"/api/records/{row['id']}/image",
            "response": response,
        }

    def list_records(
        self,
        *,
        domain: str = "",
        action: str = "",
        risk: str = "",
        workspace_id: str = "",
        limit: int = 200,
    ) -> list[dict[str, Any]]:
        clauses = []
        values: list[Any] = []
        if domain:
            clauses.append("domain = ?")
            values.append(domain)
        if action:
            clauses.append("final_action = ?")
            values.append(action)
        if risk:
            clauses.append("effective_risk = ?")
            values.append(risk)
        if workspace_id:
            clauses.append("workspace_id = ?")
            values.append(workspace_id)
        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        values.append(max(1, min(int(limit), 500)))
        with self._connect() as connection:
            rows = connection.execute(
                f"""
                SELECT * FROM review_records
                {where}
                ORDER BY updated_at DESC
                LIMIT ?
                """,
                values,
            ).fetchall()
        return [self._record(row) for row in rows]

    def count_records(self, *, workspace_id: str = "") -> int:
        with self._connect() as connection:
            if workspace_id:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM review_records "
                    "WHERE workspace_id = ?",
                    (workspace_id,),
                ).fetchone()
            else:
                row = connection.execute(
                    "SELECT COUNT(*) AS total FROM review_records"
                ).fetchone()
        return int(row["total"] if row else 0)

    def get_record(self, record_id: str) -> dict[str, Any] | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM review_records WHERE id = ?",
                (record_id,),
            ).fetchone()
        return self._record(row) if row else None

    def image_path(self, record_id: str) -> Path | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT image_path FROM review_records WHERE id = ?",
                (record_id,),
            ).fetchone()
        if row is None or not row["image_path"]:
            return None
        path = Path(row["image_path"])
        return path if path.is_file() else None

    def clear_records(self, *, workspace_id: str) -> tuple[int, list[Path]]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT image_path FROM review_records WHERE workspace_id = ?",
                (workspace_id,),
            ).fetchall()
            cursor = connection.execute(
                "DELETE FROM review_records WHERE workspace_id = ?",
                (workspace_id,),
            )
        paths = [
            Path(row["image_path"])
            for row in rows
            if row["image_path"]
        ]
        return cursor.rowcount, paths

    def export_csv(self, *, workspace_id: str = "") -> str:
        rows = self.list_records(workspace_id=workspace_id, limit=500)
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow([
            "id", "workspace_id", "created_at", "updated_at", "filename", "domain",
            "queue", "manual_status", "original_action", "final_action",
            "adjustment_direction", "effective_risk", "risk_raw",
            "feedback_reason",
        ])
        for row in rows:
            writer.writerow([
                row["id"], row["workspace_id"], row["created_at"], row["updated_at"],
                row["filename"], row["domain"], row["queue"],
                row["manual_status"], row["action"], row["final_action"],
                row["adjustment_direction"], row["effective_risk"],
                row["risk_raw"] or "",
                row["feedback_reason"],
            ])
        return "\ufeff" + output.getvalue()
