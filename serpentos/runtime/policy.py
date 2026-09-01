"""The policy interface.

A policy is a pure function from a :class:`~serpentos.runtime.models.DecisionContext`
to a :class:`~serpentos.runtime.models.Decision`. It answers "what should we do?"
and nothing else.

Policies must not perform side effects. No network calls, no writes, no clock
reads, no mutation of the context. This is not a style preference — it is the
property that makes replay, comparison and audit meaningful. A policy that
talks to the outside world cannot be re-run against a recorded context and
cannot be compared fairly against another policy offline.

The host application executes the returned action. The runtime never does.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Mapping, Optional, Tuple

try:  # pragma: no cover - the fallback only runs on Python < 3.8
    from typing import Protocol, runtime_checkable
except ImportError:  # pragma: no cover
    Protocol = object  # type: ignore[assignment]

    def runtime_checkable(cls):  # type: ignore[misc]
        return cls


from .errors import ConfigurationError
from .models import Decision, DecisionContext

__all__ = ["Policy", "BasePolicy", "ensure_policy", "is_deterministic", "policy_identity"]


@runtime_checkable
class Policy(Protocol):
    """Structural interface every policy satisfies.

    Implementations need three things: a ``name``, a ``version`` and a
    :meth:`decide` method. Inheriting from :class:`BasePolicy` is convenient but
    not required — anything with the right shape works, including plain objects
    and third-party classes you cannot subclass.

    An optional ``deterministic`` attribute (default ``True`` when absent) tells
    the runtime whether the same context always yields the same decision. Set it
    to ``False`` if your policy consults an unseeded random source; replay will
    then refuse to certify a match as guaranteed.
    """

    name: str
    version: str

    def decide(self, context: DecisionContext) -> Decision:
        """Propose an action for ``context``. Must not mutate it."""
        ...


class BasePolicy(ABC):
    """Convenience base class handling identity and Decision construction.

    Subclasses implement :meth:`decide`. Use :meth:`decision` to build the
    return value so the policy's name and version are always stamped correctly.
    """

    #: Whether the same context always produces the same decision.
    deterministic: bool = True

    def __init__(self, name: str, version: str) -> None:
        if not isinstance(name, str) or not name:
            raise ConfigurationError("policy name must be a non-empty string")
        if not isinstance(version, str) or not version:
            raise ConfigurationError("policy version must be a non-empty string")
        self.name = name
        self.version = version

    @abstractmethod
    def decide(self, context: DecisionContext) -> Decision:
        """Propose an action for ``context``.

        :raises PolicyError: implementations should raise this (or let the
            engine wrap whatever they raise) rather than returning ``None``.
        """

    def decision(self, action: str, metadata: Optional[Mapping[str, Any]] = None) -> Decision:
        """Build a :class:`Decision` stamped with this policy's identity."""
        return Decision(
            action=action,
            policy_name=self.name,
            policy_version=self.version,
            metadata=metadata or {},
        )

    def __repr__(self) -> str:
        return f"{type(self).__name__}(name={self.name!r}, version={self.version!r})"


def ensure_policy(candidate: Any) -> "Policy":
    """Check ``candidate`` satisfies the policy interface and return it.

    Raises early with a readable message instead of failing later inside the
    engine with an ``AttributeError``.

    :raises ConfigurationError: if the object is not usable as a policy.
    """
    for attribute in ("name", "version"):
        value = getattr(candidate, attribute, None)
        if not isinstance(value, str) or not value:
            raise ConfigurationError(
                f"{type(candidate).__name__} is not a policy: "
                f"missing a non-empty string {attribute!r}"
            )
    if not callable(getattr(candidate, "decide", None)):
        raise ConfigurationError(
            f"{type(candidate).__name__} is not a policy: no callable decide()"
        )
    return candidate


def is_deterministic(policy: Any) -> bool:
    """Whether ``policy`` claims to be deterministic.

    Policies opt out by setting ``deterministic = False``. Absence of the
    attribute is read as ``True``, which matches every policy shipped here
    except an exploring Q-learning agent.
    """
    return bool(getattr(policy, "deterministic", True))


def policy_identity(policy: Any) -> Tuple[str, str]:
    """``(name, version)`` for ``policy``."""
    return getattr(policy, "name", ""), getattr(policy, "version", "")
