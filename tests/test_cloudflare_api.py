from __future__ import annotations

import unittest

from deployments.cloudflare.api import handle_decision, health_payload


class CloudflareDecisionApiTests(unittest.TestCase):
    def test_health_payload(self) -> None:
        self.assertEqual(health_payload()["status"], "ok")
        self.assertEqual(health_payload()["service"], "serpentos")

    def test_transient_server_error_retries(self) -> None:
        status, body = handle_decision(
            {"attempt": 1, "status_code": 503, "latency_ms": 120}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["decision"]["action"], "retry")
        self.assertEqual(body["decision"]["metadata"]["rule"], "transient-server-error")
        self.assertEqual(body["audit"]["context"]["values"]["status_code"], 503)

    def test_rate_limit_waits(self) -> None:
        status, body = handle_decision(
            {"attempt": 1, "status_code": 429, "latency_ms": 50}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["decision"]["action"], "wait")

    def test_exhausted_attempts_fail(self) -> None:
        status, body = handle_decision(
            {"attempt": 3, "status_code": 503, "latency_ms": 50}
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["decision"]["action"], "fail")

    def test_rejects_non_object_payload(self) -> None:
        status, body = handle_decision([])
        self.assertEqual(status, 400)
        self.assertIn("JSON object", body["error"])

    def test_rejects_boolean_where_integer_required(self) -> None:
        status, body = handle_decision(
            {"attempt": True, "status_code": 503, "latency_ms": 50}
        )
        self.assertEqual(status, 400)
        self.assertEqual(body["error"], "attempt must be an integer")

    def test_rejects_out_of_range_status_code(self) -> None:
        status, body = handle_decision(
            {"attempt": 1, "status_code": 999, "latency_ms": 50}
        )
        self.assertEqual(status, 400)
        self.assertIn("status_code", body["error"])

    def test_request_id_is_preserved_in_audit(self) -> None:
        status, body = handle_decision(
            {
                "attempt": 1,
                "status_code": 503,
                "latency_ms": 50,
                "request_id": "req-123",
            }
        )
        self.assertEqual(status, 200)
        self.assertEqual(body["audit"]["context"]["request_id"], "req-123")


if __name__ == "__main__":
    unittest.main()
