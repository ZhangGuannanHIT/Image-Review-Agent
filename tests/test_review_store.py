# -*- coding: utf-8 -*-

import tempfile
import unittest
from pathlib import Path

from backend.review_store import ReviewStore
from backend.schemas import ReviewResponse, ReviewResult


def response(
    request_id="review-1",
    action="allow",
    effective="L0_安全",
    domain="real",
):
    return ReviewResponse(
        id=request_id,
        suggestion="test",
        schema_version="test",
        result=ReviewResult(
            action=action,
            action_label=action,
            tone="neutral",
            effective_risk=effective,
            risk_raw=effective,
            domain=domain,
            domain_label=domain,
            domain_source="test",
            reason="test",
            conflict_type="none",
            uncertain=False,
            suggestion_source="program_template",
            degraded=False,
            recommended_action=action,
            recommended_action_label=action,
            review_hint=False,
            review_reason="",
        ),
        timing={"total_ms": 1, "stages": []},
    )


class ReviewStoreTests(unittest.TestCase):
    def test_save_filter_feedback_and_export(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(Path(directory) / "reviews.sqlite3")
            store.save_response(
                response(),
                filename="safe.png",
                image_path="",
            )
            store.save_response(
                response("review-2", "review", "L2_中度违规", "virtual"),
                filename="review.webp",
                image_path="",
            )

            all_rows = store.list_records()
            self.assertEqual([row["id"] for row in all_rows], ["review-2", "review-1"])
            self.assertEqual(all_rows[0]["queue"], "human")
            filtered = store.list_records(domain="real", action="allow")
            self.assertEqual([row["id"] for row in filtered], ["review-1"])

            updated = store.record_feedback(
                "review-1",
                resolution="adjust",
                final_action="limit",
                adjustment_direction="upgrade",
                reason="人工确认需要限流",
            )
            self.assertEqual(updated["queue"], "human")
            self.assertEqual(updated["manual_status"], "adjusted")
            self.assertEqual(updated["final_action"], "limit")
            self.assertEqual(updated["current_level"], "L1")

            rejected = store.record_feedback(
                "review-2",
                resolution="reject",
                final_action="allow",
                reason="无法确认自动结果",
            )
            self.assertEqual(rejected["final_action"], "review")
            self.assertEqual(rejected["manual_status"], "rejected")

            confirmed = store.record_feedback(
                "review-2",
                resolution="confirm",
                final_action="limit",
                reason="",
            )
            self.assertEqual(confirmed["manual_status"], "confirmed")
            self.assertEqual(confirmed["feedback_reason"], "确认系统建议处置")

            pending = store.list_records(domain="virtual")[0]
            self.assertEqual(pending["manual_status"], "confirmed")
            exported = store.export_csv()
            self.assertIn("review-1", exported)
            self.assertIn("人工确认需要限流", exported)

    def test_feedback_requires_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(Path(directory) / "reviews.sqlite3")
            store.save_response(response(), filename="safe.png")
            with self.assertRaisesRegex(ValueError, "feedback_reason_required"):
                store.record_feedback(
                    "review-1",
                    resolution="adjust",
                    final_action="limit",
                    adjustment_direction="upgrade",
                    reason=" ",
                )

    def test_adjustment_direction_must_match_action_change(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(Path(directory) / "reviews.sqlite3")
            store.save_response(response(), filename="safe.png")
            with self.assertRaisesRegex(ValueError, "adjustment_direction_mismatch"):
                store.record_feedback(
                    "review-1",
                    resolution="adjust",
                    final_action="limit",
                    adjustment_direction="downgrade",
                    reason="错误地标成降级",
                )
            with self.assertRaisesRegex(ValueError, "adjustment_requires_action_change"):
                store.record_feedback(
                    "review-1",
                    resolution="adjust",
                    final_action="allow",
                    adjustment_direction="upgrade",
                    reason="动作没有变化",
                )

    def test_review_record_remains_pending_until_explicit_resolution(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(Path(directory) / "reviews.sqlite3")
            store.save_response(
                response("review-pending", "review", "L2_中度违规"),
                filename="pending.png",
            )
            record = store.get_record("review-pending")
            self.assertEqual(record["queue"], "human")
            self.assertEqual(record["manual_status"], "pending")
            self.assertEqual(record["final_action"], "review")

    def test_repeated_adjustment_uses_current_manual_action_as_baseline(self):
        with tempfile.TemporaryDirectory() as directory:
            store = ReviewStore(Path(directory) / "reviews.sqlite3")
            store.save_response(
                response("review-repeat", "review", "L2_中度违规"),
                filename="repeat.png",
            )
            first = store.record_feedback(
                "review-repeat",
                resolution="adjust",
                final_action="limit",
                adjustment_direction="downgrade",
                reason="先降为限流",
            )
            self.assertEqual(first["current_level"], "L1")
            second = store.record_feedback(
                "review-repeat",
                resolution="adjust",
                final_action="block",
                adjustment_direction="upgrade",
                reason="再升级为下架",
            )
            self.assertEqual(second["current_level"], "L3")
            with self.assertRaisesRegex(ValueError, "adjustment_direction_mismatch"):
                store.record_feedback(
                    "review-repeat",
                    resolution="adjust",
                    final_action="review",
                    adjustment_direction="upgrade",
                    reason="L3 不能继续升级",
                )
            third = store.record_feedback(
                "review-repeat",
                resolution="adjust",
                final_action="allow",
                adjustment_direction="downgrade",
                reason="从下架直接降为放行",
            )
            self.assertEqual(third["queue"], "human")
            self.assertEqual(third["manual_status"], "adjusted")
            self.assertEqual(third["final_action"], "allow")
            self.assertEqual(third["current_level"], "L0")

    def test_records_are_scoped_and_cleared_by_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            store = ReviewStore(root / "reviews.sqlite3")
            image_a = root / "a.png"
            image_b = root / "b.png"
            image_a.write_bytes(b"a")
            image_b.write_bytes(b"b")
            store.save_response(
                response("review-a", "allow", "L0_安全"),
                filename="a.png",
                image_path=str(image_a),
                workspace_id="workspace-a",
            )
            store.save_response(
                response("review-b", "review", "L2_中度违规"),
                filename="b.png",
                image_path=str(image_b),
                workspace_id="workspace-b",
            )

            rows_a = store.list_records(workspace_id="workspace-a")
            rows_b = store.list_records(workspace_id="workspace-b")
            self.assertEqual([row["id"] for row in rows_a], ["review-a"])
            self.assertEqual([row["id"] for row in rows_b], ["review-b"])
            self.assertEqual(store.count_records(workspace_id="workspace-a"), 1)

            cleared, paths = store.clear_records(workspace_id="workspace-a")
            self.assertEqual(cleared, 1)
            self.assertEqual(paths, [image_a])
            self.assertEqual(store.count_records(workspace_id="workspace-a"), 0)
            self.assertEqual(store.count_records(workspace_id="workspace-b"), 1)


if __name__ == "__main__":
    unittest.main()
