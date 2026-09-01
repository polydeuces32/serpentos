"""Audit records: what was decided, by whom, on what evidence.

An audit record is the durable form of one engine decision. It is plain JSON —
no pickles, no class references, nothing that executes on load — so a record
written today can be read by a different program, on a different machine, years
later, and replayed.

The runtime does not assume the context is safe to keep. Applications routinely
put tokens, card numbers and personal data in the values a policy reads, so both
sinks accept a redaction list and can drop the context entirely. Nothing is
persisted unless you attach a sink.
"""

from __future__ import annotations

import contextlib
import json
import os
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple

try:  # pragma: no cover - the fallback only runs on Python < 3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

    def runtime_checkable(cls):  # type: ignore[misc]
        return cls


from .errors import AuditError, ConfigurationError
from .models import Decision, DecisionContext, to_canonical_json
from .validation import ValidationResult

__all__ = [
    "AUDIT_SCHEMA_VERSION",
    "SUPPORTED_AUDIT_SCHEMA_VERSIONS",
    "AuditRecord",
    "AuditSink",
    "InMemoryAuditLog",
    "JsonlAuditLog",
    "NullAuditSink",
    "REDACTED",
    "read_jsonl",
]

#: Schema version stamped onto every persisted record. Bumped only when the
#: on-disk shape changes in a way an older reader would misinterpret.
AUDIT_SCHEMA_VERSION = 1

#: Versions this build can read. A record from the future is refused rather than
#: guessed at: silently misreading an audit log is worse than failing to read it.
SUPPORTED_AUDIT_SCHEMA_VERSIONS = frozenset({1})

#: Placeholder substituted for redacted values.
REDACTED = "[REDACTED]"

#: Default rotation threshold for JSONL logs. A long-running service must not be
#: able to fill a disk with audit data.
DEFAULT_MAX_BYTES = 5 * 1024 * 1024

#: Refuse to parse absurdly long lines rather than buffering them into memory.
MAX_LINE_BYTES = 1024 * 1024


def utc_now() -> str:
    """Current UTC time as an ISO-8601 string with second resolution."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _redact(value: Any, keys: frozenset) -> Any:
    """Recursively replace values whose key is in ``keys``.

    Matching is by key name at any depth, so ``{"user": {"token": "..."}}`` is
    covered by ``redact=("token",)``.
    """
    if isinstance(value, Mapping):
        return {
            key: REDACTED if key in keys else _redact(item, keys)
            for key, item in value.items()
        }
    if isinstance(value, (list, tuple)):
        return [_redact(item, keys) for item in value]
    return value


@dataclass(frozen=True)
class AuditRecord:
    """One decision, as it will be written down.

    ``context`` is ``None`` when the sink was configured not to persist it. That
    is a deliberate signal: replay refuses to run against a record with no
    context rather than inventing an empty one.
    """

    decision_id: str
    timestamp: str
    policy_name: str
    policy_version: str
    action: str
    context: Optional[DecisionContext] = None
    decision_metadata: Mapping[str, Any] = None  # type: ignore[assignment]
    validation_result: Optional[ValidationResult] = None
    request_id: Optional[str] = None

    def __post_init__(self) -> None:
        if self.decision_metadata is None:
            object.__setattr__(self, "decision_metadata", {})

    # -- construction ------------------------------------------------
    @classmethod
    def build(
        cls,
        *,
        decision: Decision,
        context: DecisionContext,
        decision_id: str,
        timestamp: str,
        validation_result: Optional[ValidationResult] = None,
    ) -> "AuditRecord":
        """Assemble a record from an engine decision."""
        return cls(
            decision_id=decision_id,
            timestamp=timestamp,
            policy_name=decision.policy_name,
            policy_version=decision.policy_version,
            action=decision.action,
            context=context,
            decision_metadata=decision.metadata,
            validation_result=validation_result,
            request_id=context.request_id,
        )

    # -- privacy -----------------------------------------------------
    def redacted(
        self, keys: Iterable[str] = (), *, include_context: bool = True
    ) -> "AuditRecord":
        """A copy with sensitive values masked or the context removed entirely.

        Redaction applies to both the context values and the decision metadata,
        since a policy can echo an input straight into its explanation.
        """
        keyset = frozenset(keys)
        context: Optional[DecisionContext] = self.context if include_context else None
        if context is not None and keyset:
            context = DecisionContext(
                _redact(context.to_dict()["values"], keyset), context.request_id
            )
        metadata = (
            _redact(dict(self.decision_metadata), keyset) if keyset else self.decision_metadata
        )
        return AuditRecord(
            decision_id=self.decision_id,
            timestamp=self.timestamp,
            policy_name=self.policy_name,
            policy_version=self.policy_version,
            action=self.action,
            context=context,
            decision_metadata=metadata,
            validation_result=self.validation_result,
            request_id=self.request_id,
        )

    # -- serialisation -----------------------------------------------
    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        from .models import thaw_value

        return {
            "schema_version": AUDIT_SCHEMA_VERSION,
            "decision_id": self.decision_id,
            "timestamp": self.timestamp,
            "policy_name": self.policy_name,
            "policy_version": self.policy_version,
            "action": self.action,
            "context": self.context.to_dict() if self.context is not None else None,
            "decision_metadata": thaw_value(self.decision_metadata),
            "validation_result": (
                self.validation_result.to_dict() if self.validation_result is not None else None
            ),
            "request_id": self.request_id,
        }

    def to_json(self) -> str:
        """Canonical JSON. Equal records always serialise identically."""
        return to_canonical_json(self.to_dict())

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "AuditRecord":
        """Rebuild from :meth:`to_dict` output.

        :raises AuditError: if the payload is not a well-formed record, or was
            written by a SerpentOS whose schema this build does not understand.
            Parsing never constructs arbitrary types — every field is checked.
        """
        if not isinstance(payload, Mapping):
            raise AuditError(f"audit record must be a JSON object, got {type(payload).__name__}")
        cls._check_schema_version(payload.get("schema_version"))
        required = ("decision_id", "timestamp", "policy_name", "policy_version", "action")
        for name in required:
            if not isinstance(payload.get(name), str) or not payload[name]:
                raise AuditError(f"audit record field {name!r} must be a non-empty string")

        raw_context = payload.get("context")
        try:
            context = DecisionContext.from_dict(raw_context) if raw_context is not None else None
        except ConfigurationError as exc:
            raise AuditError(f"audit record has an unusable context: {exc}") from exc

        metadata = payload.get("decision_metadata") or {}
        if not isinstance(metadata, Mapping):
            raise AuditError("audit record decision_metadata must be a JSON object")

        raw_validation = payload.get("validation_result")
        try:
            validation = (
                ValidationResult.from_dict(raw_validation) if raw_validation is not None else None
            )
        except ConfigurationError as exc:
            raise AuditError(f"audit record has an unusable validation_result: {exc}") from exc

        request_id = payload.get("request_id")
        if request_id is not None and not isinstance(request_id, str):
            raise AuditError("audit record request_id must be a string or null")

        return cls(
            decision_id=payload["decision_id"],
            timestamp=payload["timestamp"],
            policy_name=payload["policy_name"],
            policy_version=payload["policy_version"],
            action=payload["action"],
            context=context,
            decision_metadata=dict(metadata),
            validation_result=validation,
            request_id=request_id,
        )

    @staticmethod
    def _check_schema_version(raw: Any) -> None:
        """Refuse a record this build cannot read correctly.

        A missing ``schema_version`` means version 1. Records written before the
        field existed have exactly the version-1 shape, so reading them is
        correct rather than merely tolerated — but every record written from now
        on carries the field explicitly.
        """
        if raw is None:
            return
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise AuditError(
                f"audit record schema_version must be an integer, got {type(raw).__name__}"
            )
        if raw in SUPPORTED_AUDIT_SCHEMA_VERSIONS:
            return
        known = ", ".join(str(version) for version in sorted(SUPPORTED_AUDIT_SCHEMA_VERSIONS))
        if raw > AUDIT_SCHEMA_VERSION:
            raise AuditError(
                f"audit record schema_version {raw} was written by a newer SerpentOS; "
                f"this build reads {known}. Upgrade rather than risk misreading it."
            )
        raise AuditError(
            f"audit record schema_version {raw} is not a version this build reads ({known})"
        )

    @classmethod
    def from_json(cls, line: str) -> "AuditRecord":
        """Parse one JSON line into a record.

        :raises AuditError: on malformed JSON or a malformed record.
        """
        try:
            payload = json.loads(line)
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise AuditError(f"audit record is not valid JSON: {exc}") from exc
        return cls.from_dict(payload)


@runtime_checkable
class AuditSink(Protocol):
    """Somewhere audit records go."""

    def record(self, record: AuditRecord) -> None:
        """Persist or accumulate ``record``."""
        ...


class _RedactingSink:
    """Shared redaction handling for the built-in sinks."""

    def __init__(self, redact: Iterable[str] = (), *, include_context: bool = True) -> None:
        if isinstance(redact, str):
            raise ConfigurationError(
                "redact must be a collection of key names, not a single string"
            )
        self._redact_keys: Tuple[str, ...] = tuple(redact)
        for key in self._redact_keys:
            if not isinstance(key, str):
                raise ConfigurationError("redact keys must be strings")
        self._include_context = bool(include_context)

    def _prepare(self, record: AuditRecord) -> AuditRecord:
        if not self._redact_keys and self._include_context:
            return record
        return record.redacted(self._redact_keys, include_context=self._include_context)


class NullAuditSink:
    """Discards everything. The explicit way to say "do not persist decisions"."""

    def record(self, record: AuditRecord) -> None:
        """Do nothing."""


class InMemoryAuditLog(_RedactingSink):
    """Keeps records in a bounded list. Useful in tests and short-lived jobs.

    ``max_records`` caps memory use; the oldest records are dropped first.

    **Thread-safe.** One instance may be shared by any number of application
    threads: appends are serialised by a lock, and :attr:`records` returns a
    snapshot rather than a live view, so iterating it cannot race with a writer.
    """

    def __init__(
        self,
        *,
        max_records: int = 10_000,
        redact: Iterable[str] = (),
        include_context: bool = True,
    ) -> None:
        super().__init__(redact, include_context=include_context)
        if max_records <= 0:
            raise ConfigurationError("max_records must be positive")
        self._max_records = max_records
        self._records: List[AuditRecord] = []
        self._lock = threading.Lock()

    def record(self, record: AuditRecord) -> None:
        """Append ``record``, dropping the oldest if the cap is reached."""
        # Redaction is pure, so it can happen outside the lock.
        prepared = self._prepare(record)
        with self._lock:
            self._records.append(prepared)
            if len(self._records) > self._max_records:
                del self._records[: len(self._records) - self._max_records]

    @property
    def records(self) -> Sequence[AuditRecord]:
        """The records held, oldest first. A snapshot, safe to iterate."""
        with self._lock:
            return tuple(self._records)

    def clear(self) -> None:
        """Drop everything held."""
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


class JsonlAuditLog(_RedactingSink):
    """Appends one JSON object per line to a file, with size-based rotation.

    Each record is written with a single ``write`` syscall to a file opened
    ``O_APPEND``, so concurrent writers interleave whole lines rather than
    corrupting each other. Durability is a separate question: pass
    ``fsync=True`` to flush to disk on every record, at a substantial cost in
    throughput.

    **Thread-safe within one process.** A lock serialises the
    check-size-then-rotate-then-write sequence, which is what actually races —
    without it two threads can both decide to rotate and one loses its file
    descriptor mid-write.

    *Across* processes, `O_APPEND` keeps individual lines intact, but rotation
    is not coordinated: two processes rotating the same path at the same moment
    can lose records. Give each process its own file if that matters.

    ``path`` is trusted caller configuration. Never build it from context data —
    that is how a policy input turns into a path traversal.
    """

    def __init__(
        self,
        path: str,
        *,
        max_bytes: int = DEFAULT_MAX_BYTES,
        redact: Iterable[str] = (),
        include_context: bool = True,
        fsync: bool = False,
    ) -> None:
        super().__init__(redact, include_context=include_context)
        if not isinstance(path, str) or not path:
            raise ConfigurationError("path must be a non-empty string")
        if max_bytes < 0:
            raise ConfigurationError("max_bytes must be zero (unbounded) or positive")
        self.path = os.path.abspath(os.path.expanduser(path))
        self.max_bytes = max_bytes
        self._fsync = bool(fsync)
        self._fd: Optional[int] = None
        self._lock = threading.Lock()

    # -- lifecycle ---------------------------------------------------
    def _open(self) -> int:
        if self._fd is None:
            directory = os.path.dirname(self.path) or "."
            try:
                os.makedirs(directory, exist_ok=True)
                self._fd = os.open(
                    self.path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600
                )
            except OSError as exc:
                raise AuditError(f"cannot open audit log {self.path}: {exc}") from exc
        return self._fd

    def _close_locked(self) -> None:
        if self._fd is not None:
            with contextlib.suppress(OSError):
                os.close(self._fd)
            self._fd = None

    def close(self) -> None:
        """Close the underlying file. Further writes reopen it."""
        with self._lock:
            self._close_locked()

    def __enter__(self) -> "JsonlAuditLog":
        with self._lock:
            self._open()
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    # -- writing -----------------------------------------------------
    def _rotate_if_needed(self) -> None:
        if self.max_bytes <= 0:
            return
        try:
            size = os.path.getsize(self.path) if os.path.exists(self.path) else 0
        except OSError:
            return
        if size < self.max_bytes:
            return
        self._close_locked()
        with contextlib.suppress(OSError):
            os.replace(self.path, self.path + ".1")

    def record(self, record: AuditRecord) -> None:
        """Append ``record`` as one JSON line.

        :raises AuditError: if the record cannot be serialised or written.
        """
        # Redaction and serialisation are pure; only the file needs the lock.
        prepared = self._prepare(record)
        try:
            line = (prepared.to_json() + "\n").encode("utf-8")
        except (TypeError, ValueError) as exc:
            raise AuditError(f"cannot serialise audit record: {exc}") from exc

        with self._lock:
            self._rotate_if_needed()
            fd = self._open()
            try:
                written = os.write(fd, line)
                if written != len(line):  # pragma: no cover - short writes are rare
                    raise AuditError(
                        f"short write to {self.path}: {written} of {len(line)} bytes"
                    )
                if self._fsync:
                    os.fsync(fd)
            except OSError as exc:
                raise AuditError(f"cannot write to audit log {self.path}: {exc}") from exc


def read_jsonl(path: str, *, skip_invalid: bool = False) -> Iterator[AuditRecord]:
    """Yield the audit records in a JSONL file, oldest first.

    Blank lines are ignored. A malformed line raises
    :class:`~serpentos.runtime.errors.AuditError` naming the line number, unless
    ``skip_invalid`` is set — a half-written final line from a crashed process is
    the usual cause.

    :raises AuditError: if the file cannot be read, or contains a bad record.
    """
    try:
        handle = open(path, "r", encoding="utf-8")
    except OSError as exc:
        raise AuditError(f"cannot read audit log {path}: {exc}") from exc
    with handle:
        for number, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            if len(stripped) > MAX_LINE_BYTES:
                if skip_invalid:
                    continue
                raise AuditError(f"{path}:{number}: line exceeds {MAX_LINE_BYTES} bytes")
            try:
                yield AuditRecord.from_json(stripped)
            except AuditError as exc:
                if skip_invalid:
                    continue
                raise AuditError(f"{path}:{number}: {exc}") from exc
