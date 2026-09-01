"""Typed failures for the SerpentOS policy runtime.

Every failure mode in the runtime raises one of these. Nothing is swallowed and
nothing returns an ambiguous ``None`` in place of an error, so a caller can
distinguish "the policy blew up" from "the policy proposed something we refuse
to allow" without parsing strings.
"""

from __future__ import annotations

__all__ = [
    "SerpentOSError",
    "ConfigurationError",
    "PolicyError",
    "DecisionValidationError",
    "ReplayError",
    "AuditError",
]


class SerpentOSError(Exception):
    """Base class for every error the runtime raises deliberately."""


class ConfigurationError(SerpentOSError):
    """A component was constructed with unusable arguments.

    Also raised when a value handed to a model cannot be represented as JSON,
    because that makes the object unserialisable and therefore unreplayable.
    """


class PolicyError(SerpentOSError):
    """A policy failed, or returned something that is not a valid Decision.

    Exceptions raised inside :meth:`Policy.decide` are wrapped in this so the
    engine's contract stays narrow: callers catch ``PolicyError`` rather than
    "whatever the policy author happened to raise". The original exception is
    kept on ``__cause__``.
    """


class DecisionValidationError(SerpentOSError):
    """A validator rejected a proposed decision.

    ``result`` carries the :class:`~serpentos.runtime.validation.ValidationResult`
    that caused the rejection, so callers can report the reason without
    re-running validation.
    """

    def __init__(self, message: str, result: object = None) -> None:
        super().__init__(message)
        self.result = result


class ReplayError(SerpentOSError):
    """A recorded decision could not be replayed against the supplied policy."""


class AuditError(SerpentOSError):
    """An audit record could not be written, read or parsed."""
