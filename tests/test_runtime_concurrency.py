"""The concurrency contract, tested rather than asserted in a docstring.

A claim of thread safety that nobody exercised is a claim you find out about in
production. These tests hammer the built-in sinks and a shared engine from many
threads and check the things that actually break under contention: lost records,
interleaved half-lines in a file, and a rotation that races a write.

They are necessarily probabilistic — a passing run does not prove the absence of
a race.

Worth knowing which of these are load-bearing. Under CPython's GIL a single
``list.append`` or ``os.write`` is already atomic, so most of the tests below
pass with the locks removed; they guard the contract rather than a live bug, and
they matter for free-threaded builds where that GIL guarantee is gone.

The rotation test is different. ``JsonlAuditLog`` checks the file size, closes
its descriptor, renames the file and reopens — a compound sequence with real
interleaving windows. Removing the lock and running
``test_rotation_under_contention_is_not_a_race`` fails in most attempts with
`Bad file descriptor`, or a write against a descriptor another thread has
already set to ``None``. That one is protecting against something.
"""

import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos.runtime.audit import (
    AuditRecord,
    InMemoryAuditLog,
    JsonlAuditLog,
    read_jsonl,
)
from serpentos.runtime.engine import DecisionEngine
from serpentos.runtime.models import DecisionContext
from serpentos.runtime.policy import BasePolicy
from serpentos.runtime.validation import ActionValidator

THREADS = 8
PER_THREAD = 60


class CountingPolicy(BasePolicy):
    """Stateless, as the purity contract requires, so it is safe to share."""

    def __init__(self):
        super().__init__("counter", "1.0")

    def decide(self, context):
        return self.decision(
            "retry" if context["n"] % 2 else "fail", {"n": context["n"]}
        )


def run_in_threads(work, count=THREADS):
    """Run ``work(index)`` on ``count`` threads, re-raising anything it throws."""
    errors = []
    start = threading.Barrier(count)

    def target(index):
        try:
            start.wait(timeout=30)
            work(index)
        except Exception as exc:  # noqa: BLE001 - reported to the main thread
            errors.append(exc)

    threads = [threading.Thread(target=target, args=(index,)) for index in range(count)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=60)
    assert not any(thread.is_alive() for thread in threads), "a worker thread hung"
    return errors


def a_record(index):
    return AuditRecord(
        decision_id=f"id-{index:05d}",
        timestamp="2026-01-01T00:00:00+00:00",
        policy_name="p",
        policy_version="1.0",
        action="retry",
        context=DecisionContext({"n": index}),
    )


class InMemorySinkConcurrencyTest(unittest.TestCase):
    def test_concurrent_writers_lose_nothing(self):
        sink = InMemoryAuditLog(max_records=THREADS * PER_THREAD)

        def work(thread_index):
            for step in range(PER_THREAD):
                sink.record(a_record(thread_index * PER_THREAD + step))

        self.assertEqual(run_in_threads(work), [])
        self.assertEqual(len(sink), THREADS * PER_THREAD)
        ids = {record.decision_id for record in sink.records}
        self.assertEqual(len(ids), THREADS * PER_THREAD)

    def test_the_cap_holds_under_contention(self):
        # Nothing may exceed the bound, however many threads push at once.
        sink = InMemoryAuditLog(max_records=50)

        def work(thread_index):
            for step in range(PER_THREAD):
                sink.record(a_record(thread_index * PER_THREAD + step))

        self.assertEqual(run_in_threads(work), [])
        self.assertEqual(len(sink), 50)

    def test_reading_a_snapshot_while_writing_never_tears(self):
        sink = InMemoryAuditLog()
        stop = threading.Event()
        seen = []

        def reader():
            while not stop.is_set():
                snapshot = sink.records
                # A snapshot is a tuple: it cannot change under iteration.
                seen.append(sum(1 for _ in snapshot))

        thread = threading.Thread(target=reader)
        thread.start()
        try:
            for index in range(500):
                sink.record(a_record(index))
        finally:
            stop.set()
            thread.join(timeout=30)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(sink), 500)

    def test_clear_races_safely_against_writers(self):
        sink = InMemoryAuditLog()

        def work(thread_index):
            for step in range(PER_THREAD):
                sink.record(a_record(step))
                if thread_index == 0 and step % 10 == 0:
                    sink.clear()

        self.assertEqual(run_in_threads(work), [])
        # The exact count is timing-dependent; not crashing is the contract.
        self.assertLessEqual(len(sink), THREADS * PER_THREAD)


class JsonlSinkConcurrencyTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.mkdtemp()
        self.path = os.path.join(self.directory, "audit.jsonl")
        self.addCleanup(shutil.rmtree, self.directory, ignore_errors=True)

    def test_every_line_is_whole_and_nothing_is_lost(self):
        sink = JsonlAuditLog(self.path)
        self.addCleanup(sink.close)

        def work(thread_index):
            for step in range(PER_THREAD):
                sink.record(a_record(thread_index * PER_THREAD + step))

        self.assertEqual(run_in_threads(work), [])
        sink.close()

        with open(self.path, encoding="utf-8") as handle:
            lines = [line for line in handle if line.strip()]
        self.assertEqual(len(lines), THREADS * PER_THREAD)
        # Every line must be independently parseable: a torn write shows up here.
        for line in lines:
            json.loads(line)
        records = list(read_jsonl(self.path))
        self.assertEqual(len(records), THREADS * PER_THREAD)
        self.assertEqual(
            len({record.decision_id for record in records}), THREADS * PER_THREAD
        )

    def test_rotation_under_contention_is_not_a_race(self):
        """The one test here that catches a real bug rather than guarding a rule.

        Rotation is check-size, close, rename, reopen — four steps that must not
        interleave. A tiny cap plus a short switch interval forces the window
        open thousands of times. Without the lock this raises ``Bad file
        descriptor`` in most runs.
        """
        original_interval = sys.getswitchinterval()
        sys.setswitchinterval(1e-6)
        self.addCleanup(sys.setswitchinterval, original_interval)

        sink = JsonlAuditLog(self.path, max_bytes=700)
        self.addCleanup(sink.close)
        failures = []

        def work(thread_index):
            for step in range(200):
                try:
                    sink.record(a_record(thread_index * 200 + step))
                except Exception as exc:  # noqa: BLE001 - the thing under test
                    failures.append(repr(exc))

        self.assertEqual(run_in_threads(work, count=12), [])
        sink.close()
        self.assertEqual(failures, [], "rotation raced with a concurrent write")

        # Rotation legitimately discards the previous .1, so records are lost by
        # design. What must hold is that every surviving line is whole.
        total = 0
        for candidate in (self.path, self.path + ".1"):
            if os.path.exists(candidate):
                with open(candidate, encoding="utf-8") as handle:
                    for line in handle:
                        if line.strip():
                            json.loads(line)
                            total += 1
        self.assertGreater(total, 0)

    def test_closing_while_others_write_is_safe(self):
        sink = JsonlAuditLog(self.path)
        self.addCleanup(sink.close)

        def work(thread_index):
            for step in range(PER_THREAD):
                sink.record(a_record(step))
                if thread_index == 0 and step % 15 == 0:
                    sink.close()  # reopened lazily by the next write

        self.assertEqual(run_in_threads(work), [])
        sink.close()
        with open(self.path, encoding="utf-8") as handle:
            for line in handle:
                if line.strip():
                    json.loads(line)


class SharedEngineConcurrencyTest(unittest.TestCase):
    def test_one_engine_serves_many_threads(self):
        sink = InMemoryAuditLog()
        engine = DecisionEngine(
            CountingPolicy(),
            validator=ActionValidator({"retry", "fail"}),
            audit_sink=sink,
        )
        results = {}
        lock = threading.Lock()

        def work(thread_index):
            local = []
            for step in range(PER_THREAD):
                number = thread_index * PER_THREAD + step
                decision = engine.decide(DecisionContext({"n": number}))
                local.append((number, decision.action, decision.decision_id))
            with lock:
                results[thread_index] = local

        self.assertEqual(run_in_threads(work), [])

        flat = [item for local in results.values() for item in local]
        self.assertEqual(len(flat), THREADS * PER_THREAD)
        # Every decision is correct for its own context: no cross-thread bleed.
        for number, action, _ in flat:
            self.assertEqual(action, "retry" if number % 2 else "fail")
        # Identifiers are unique, and every decision was audited.
        self.assertEqual(len({item[2] for item in flat}), THREADS * PER_THREAD)
        self.assertEqual(len(sink), THREADS * PER_THREAD)

    def test_a_shared_context_is_not_corrupted_by_concurrent_readers(self):
        context = DecisionContext({"n": 7, "nested": {"a": [1, 2, 3]}})
        before = context.to_json()
        engine = DecisionEngine(CountingPolicy())

        def work(thread_index):
            for _ in range(PER_THREAD):
                self.assertEqual(engine.decide(context).action, "retry")

        self.assertEqual(run_in_threads(work), [])
        self.assertEqual(context.to_json(), before)


if __name__ == "__main__":
    unittest.main()
