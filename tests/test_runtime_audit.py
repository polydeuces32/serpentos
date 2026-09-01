"""Tests for audit records and sinks."""

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.runtime.audit import (
    AUDIT_SCHEMA_VERSION,
    REDACTED,
    SUPPORTED_AUDIT_SCHEMA_VERSIONS,
    AuditRecord,
    InMemoryAuditLog,
    JsonlAuditLog,
    NullAuditSink,
    read_jsonl,
)
from serpentos.runtime.engine import DecisionEngine
from serpentos.runtime.errors import AuditError, ConfigurationError
from serpentos.runtime.models import Decision, DecisionContext
from serpentos.runtime.policy import BasePolicy
from serpentos.runtime.validation import ValidationResult


class EchoPolicy(BasePolicy):
    def __init__(self):
        super().__init__("echo", "1.0")

    def decide(self, context):
        return self.decision("go", {"token": context.get("token"), "safe": 1})


def sample_record(**overrides):
    payload = dict(
        decision_id="d-1",
        timestamp="2024-01-01T00:00:00+00:00",
        policy_name="p",
        policy_version="1.0",
        action="go",
        context=DecisionContext({"a": 1}, request_id="r-1"),
        decision_metadata={"why": "because"},
        validation_result=ValidationResult.accepted("v"),
        request_id="r-1",
    )
    payload.update(overrides)
    return AuditRecord(**payload)


class AuditRecordTest(unittest.TestCase):
    def test_roundtrip(self):
        record = sample_record()
        self.assertEqual(AuditRecord.from_dict(record.to_dict()), record)

    def test_json_is_canonical(self):
        first = sample_record(decision_metadata={"b": 1, "a": 2}).to_json()
        second = sample_record(decision_metadata={"a": 2, "b": 1}).to_json()
        self.assertEqual(first, second)

    def test_build_from_a_decision(self):
        record = AuditRecord.build(
            decision=Decision("go", "p", "1.0", {"why": "x"}, "d-1"),
            context=DecisionContext({"a": 1}, request_id="r-1"),
            decision_id="d-1",
            timestamp="t",
            validation_result=ValidationResult.accepted("v"),
        )
        self.assertEqual(record.action, "go")
        self.assertEqual(record.request_id, "r-1")

    def test_from_dict_rejects_malformed_records(self):
        bad = [
            "not an object",
            {},
            {"decision_id": "", "timestamp": "t", "policy_name": "p", "policy_version": "1", "action": "a"},
            {"decision_id": "d", "timestamp": "t", "policy_name": "p", "policy_version": "1", "action": 3},
        ]
        for payload in bad:
            with self.assertRaises(AuditError):
                AuditRecord.from_dict(payload)

    def test_from_dict_rejects_a_broken_context(self):
        payload = sample_record().to_dict()
        payload["context"] = {"values": "not a mapping"}
        with self.assertRaises(AuditError):
            AuditRecord.from_dict(payload)

    def test_from_json_rejects_garbage(self):
        with self.assertRaises(AuditError):
            AuditRecord.from_json("{not json")

    def test_parsing_never_evaluates_the_payload(self):
        # A record whose fields look like code is data and stays data.
        payload = sample_record().to_dict()
        payload["decision_metadata"] = {"expr": "__import__('os').system('true')"}
        record = AuditRecord.from_dict(payload)
        self.assertEqual(record.decision_metadata["expr"], "__import__('os').system('true')")


class SchemaVersionTest(unittest.TestCase):
    """The on-disk format says which format it is, and readers act on that."""

    def test_every_persisted_record_declares_its_schema(self):
        payload = sample_record().to_dict()
        self.assertEqual(payload["schema_version"], AUDIT_SCHEMA_VERSION)
        self.assertEqual(json.loads(sample_record().to_json())["schema_version"], 1)

    def test_the_current_version_is_accepted(self):
        record = sample_record()
        payload = record.to_dict()
        self.assertEqual(AuditRecord.from_dict(payload), record)

    def test_a_record_written_before_the_field_existed_is_read_as_version_one(self):
        # Legacy records have exactly the v1 shape, so reading them is correct
        # rather than merely tolerated.
        record = sample_record()
        legacy = record.to_dict()
        del legacy["schema_version"]
        self.assertEqual(AuditRecord.from_dict(legacy), record)

    def test_a_future_version_is_refused_rather_than_guessed_at(self):
        payload = sample_record().to_dict()
        payload["schema_version"] = AUDIT_SCHEMA_VERSION + 1
        with self.assertRaises(AuditError) as caught:
            AuditRecord.from_dict(payload)
        message = str(caught.exception)
        self.assertIn("newer SerpentOS", message)
        self.assertIn(str(AUDIT_SCHEMA_VERSION + 1), message)

    def test_a_far_future_version_is_also_refused(self):
        payload = sample_record().to_dict()
        payload["schema_version"] = 9999
        with self.assertRaises(AuditError):
            AuditRecord.from_dict(payload)

    def test_a_nonsense_version_is_refused(self):
        for version in ("1", 1.5, True, [1], {"v": 1}, 0, -3):
            payload = sample_record().to_dict()
            payload["schema_version"] = version
            with self.assertRaises(AuditError):
                AuditRecord.from_dict(payload)

    def test_a_future_record_in_a_file_names_its_line(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "audit.jsonl")
        good = sample_record().to_json()
        future = json.loads(good)
        future["schema_version"] = 2
        with open(path, "w", encoding="utf-8") as handle:
            handle.write(good + "\n")
            handle.write(json.dumps(future) + "\n")

        with self.assertRaises(AuditError) as caught:
            list(read_jsonl(path))
        self.assertIn(":2:", str(caught.exception))
        # The reader can be told to press on past records it cannot understand.
        self.assertEqual(len(list(read_jsonl(path, skip_invalid=True))), 1)

    def test_the_supported_set_is_consistent_with_the_current_version(self):
        self.assertIn(AUDIT_SCHEMA_VERSION, SUPPORTED_AUDIT_SCHEMA_VERSIONS)
        self.assertEqual(max(SUPPORTED_AUDIT_SCHEMA_VERSIONS), AUDIT_SCHEMA_VERSION)

    def test_round_tripping_through_a_file_preserves_the_version(self):
        directory = tempfile.mkdtemp()
        path = os.path.join(directory, "audit.jsonl")
        sink = JsonlAuditLog(path)
        sink.record(sample_record())
        sink.close()
        with open(path, encoding="utf-8") as handle:
            self.assertEqual(json.loads(handle.read())["schema_version"], 1)


class RedactionTest(unittest.TestCase):
    def test_named_keys_are_masked_at_any_depth(self):
        record = sample_record(
            context=DecisionContext({"token": "secret", "user": {"token": "also-secret"}})
        )
        redacted = record.redacted(["token"])
        values = redacted.context.to_dict()["values"]
        self.assertEqual(values["token"], REDACTED)
        self.assertEqual(values["user"]["token"], REDACTED)

    def test_metadata_is_redacted_too(self):
        record = sample_record(decision_metadata={"token": "secret", "safe": 1})
        redacted = record.redacted(["token"])
        self.assertEqual(redacted.decision_metadata["token"], REDACTED)
        self.assertEqual(redacted.decision_metadata["safe"], 1)

    def test_context_can_be_dropped_entirely(self):
        redacted = sample_record().redacted(include_context=False)
        self.assertIsNone(redacted.context)
        self.assertEqual(redacted.to_dict()["context"], None)

    def test_redaction_does_not_touch_the_original(self):
        record = sample_record(context=DecisionContext({"token": "secret"}))
        record.redacted(["token"])
        self.assertEqual(record.context["token"], "secret")

    def test_sink_level_redaction_applies_to_engine_output(self):
        audit = InMemoryAuditLog(redact=["token"])
        engine = DecisionEngine(EchoPolicy(), audit_sink=audit)
        engine.decide(DecisionContext({"token": "hunter2", "keep": 1}))
        values = audit.records[0].context.to_dict()["values"]
        self.assertEqual(values["token"], REDACTED)
        self.assertEqual(values["keep"], 1)
        self.assertEqual(audit.records[0].decision_metadata["token"], REDACTED)

    def test_redact_rejects_a_bare_string(self):
        with self.assertRaises(ConfigurationError):
            InMemoryAuditLog(redact="token")


class InMemoryAuditLogTest(unittest.TestCase):
    def test_records_accumulate_in_order(self):
        log = InMemoryAuditLog()
        log.record(sample_record(decision_id="a"))
        log.record(sample_record(decision_id="b"))
        self.assertEqual([r.decision_id for r in log.records], ["a", "b"])
        self.assertEqual(len(log), 2)

    def test_oldest_records_are_dropped_at_the_cap(self):
        log = InMemoryAuditLog(max_records=2)
        for index in range(4):
            log.record(sample_record(decision_id=str(index)))
        self.assertEqual([r.decision_id for r in log.records], ["2", "3"])

    def test_records_property_is_a_snapshot(self):
        log = InMemoryAuditLog()
        log.record(sample_record())
        snapshot = log.records
        log.clear()
        self.assertEqual(len(snapshot), 1)
        self.assertEqual(len(log), 0)

    def test_cap_must_be_positive(self):
        with self.assertRaises(ConfigurationError):
            InMemoryAuditLog(max_records=0)


class NullAuditSinkTest(unittest.TestCase):
    def test_discards_everything(self):
        sink = NullAuditSink()
        sink.record(sample_record())
        engine = DecisionEngine(EchoPolicy(), audit_sink=sink)
        self.assertEqual(engine.decide(DecisionContext()).action, "go")


class JsonlAuditLogTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "audit.jsonl")

    def test_writes_one_json_object_per_line(self):
        with JsonlAuditLog(self.path) as log:
            log.record(sample_record(decision_id="a"))
            log.record(sample_record(decision_id="b"))
        lines = Path(self.path).read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(lines), 2)
        self.assertEqual(json.loads(lines[0])["decision_id"], "a")

    def test_records_round_trip_through_the_file(self):
        record = sample_record()
        with JsonlAuditLog(self.path) as log:
            log.record(record)
        self.assertEqual(list(read_jsonl(self.path)), [record])

    def test_creates_missing_directories(self):
        nested = os.path.join(self.dir, "deep", "deeper", "audit.jsonl")
        with JsonlAuditLog(nested) as log:
            log.record(sample_record())
        self.assertTrue(os.path.exists(nested))

    def test_appends_across_reopen(self):
        JsonlAuditLog(self.path).record(sample_record(decision_id="a"))
        JsonlAuditLog(self.path).record(sample_record(decision_id="b"))
        self.assertEqual([r.decision_id for r in read_jsonl(self.path)], ["a", "b"])

    def test_rotates_instead_of_growing_without_bound(self):
        log = JsonlAuditLog(self.path, max_bytes=200)
        for index in range(20):
            log.record(sample_record(decision_id=f"d-{index}"))
        log.close()
        self.assertTrue(os.path.exists(self.path + ".1"))
        self.assertLess(os.path.getsize(self.path), 2000)

    def test_redaction_reaches_disk(self):
        log = JsonlAuditLog(self.path, redact=["token"])
        log.record(sample_record(context=DecisionContext({"token": "secret"})))
        log.close()
        text = Path(self.path).read_text(encoding="utf-8")
        self.assertNotIn("secret", text)
        self.assertIn(REDACTED, text)

    def test_context_can_be_left_off_disk(self):
        log = JsonlAuditLog(self.path, include_context=False)
        log.record(sample_record(context=DecisionContext({"token": "secret"})))
        log.close()
        payload = json.loads(Path(self.path).read_text(encoding="utf-8"))
        self.assertIsNone(payload["context"])

    def test_bad_configuration_is_refused(self):
        with self.assertRaises(ConfigurationError):
            JsonlAuditLog("")
        with self.assertRaises(ConfigurationError):
            JsonlAuditLog(self.path, max_bytes=-1)

    def test_unwritable_path_raises_audit_error(self):
        log = JsonlAuditLog(os.path.join(self.dir, "audit.jsonl", "nope.jsonl"))
        Path(self.path).write_text("", encoding="utf-8")
        with self.assertRaises(AuditError):
            log.record(sample_record())


class ReadJsonlTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        self.path = os.path.join(self.dir, "audit.jsonl")

    def write(self, text):
        Path(self.path).write_text(text, encoding="utf-8")

    def test_blank_lines_are_ignored(self):
        self.write(sample_record().to_json() + "\n\n\n")
        self.assertEqual(len(list(read_jsonl(self.path))), 1)

    def test_a_truncated_final_line_fails_loudly_by_default(self):
        self.write(sample_record().to_json() + "\n{\"decision_id\": \"partial\"")
        with self.assertRaises(AuditError) as caught:
            list(read_jsonl(self.path))
        self.assertIn(":2:", str(caught.exception))

    def test_a_truncated_final_line_can_be_skipped(self):
        self.write(sample_record().to_json() + "\n{\"decision_id\": \"partial\"")
        self.assertEqual(len(list(read_jsonl(self.path, skip_invalid=True))), 1)

    def test_missing_file_raises_audit_error(self):
        with self.assertRaises(AuditError):
            list(read_jsonl(os.path.join(self.dir, "nope.jsonl")))

    def test_absurdly_long_lines_are_refused(self):
        self.write("x" * (1024 * 1024 + 10))
        with self.assertRaises(AuditError):
            list(read_jsonl(self.path))


if __name__ == "__main__":
    unittest.main()
