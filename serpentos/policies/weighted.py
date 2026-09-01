"""Score every candidate action and take the highest.

Where :class:`~serpentos.policies.rules.RulePolicy` answers "which condition
fired?", this answers "which option scored best?". It exists partly because
scoring is a genuinely different shape of decision — ranking, routing, bin
packing — and partly to demonstrate that the :class:`~serpentos.runtime.policy.Policy`
interface is not secretly shaped around Q-learning.

Two ways to supply the scores:

* :class:`WeightedPolicy` takes Python callables, one per action. Maximum
  flexibility, in-process only.
* :meth:`WeightedPolicy.from_linear` builds those callables from pure data —
  ``score = bias + sum(weight * value)`` — so a scoring model can live in a
  config file. There is no expression parser here and there will not be one:
  the whole point is that untrusted configuration cannot execute.

Ties are broken by declaration order, not arbitrarily, so the policy is
deterministic and replayable.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, Mapping, Optional, Tuple

from ..runtime.errors import ConfigurationError, PolicyError
from ..runtime.models import Decision, DecisionContext
from ..runtime.policy import BasePolicy

__all__ = ["WeightedPolicy", "LinearScorer"]

#: Signature of an action scorer: context values in, a finite number out.
Scorer = Callable[[Mapping[str, Any]], float]


class LinearScorer:
    """``bias + sum(weight * value)`` over a fixed set of keys.

    Pure data, so a whole scoring model round-trips through JSON without any
    code being loaded. Missing keys contribute nothing; non-numeric values are a
    hard error rather than a silent zero, because quietly scoring a string as 0
    is how a routing bug hides for six months.

    ``source`` names a nested object to read the factors from, which is what you
    want when each candidate carries its own attributes:

    >>> quote = {"ups": {"cost": 12.5, "days": 2}, "fedex": {"cost": 18.0, "days": 1}}
    >>> scorer = LinearScorer({"cost": -1.0, "days": -3.0}, source="ups")
    >>> scorer(quote)
    -18.5
    >>> scorer.explain(quote)["days"]
    -6.0
    """

    def __init__(
        self,
        weights: Mapping[str, float],
        bias: float = 0.0,
        *,
        source: Optional[str] = None,
    ) -> None:
        if not isinstance(weights, Mapping):
            raise ConfigurationError(
                f"weights must be a mapping, got {type(weights).__name__}"
            )
        cleaned: Dict[str, float] = {}
        for key, weight in weights.items():
            if not isinstance(key, str) or not key:
                raise ConfigurationError("weight keys must be non-empty strings")
            if isinstance(weight, bool) or not isinstance(weight, (int, float)):
                raise ConfigurationError(
                    f"weight for {key!r} must be a number, got {type(weight).__name__}"
                )
            if not math.isfinite(weight):
                raise ConfigurationError(f"weight for {key!r} must be finite")
            cleaned[key] = float(weight)
        if isinstance(bias, bool) or not isinstance(bias, (int, float)) or not math.isfinite(bias):
            raise ConfigurationError("bias must be a finite number")
        if source is not None and (not isinstance(source, str) or not source):
            raise ConfigurationError("source must be a non-empty string or None")
        self.weights: Mapping[str, float] = dict(cleaned)
        self.bias = float(bias)
        self.source = source

    def _factors(self, values: Mapping[str, Any]) -> Mapping[str, Any]:
        if self.source is None:
            return values
        nested = values.get(self.source)
        if nested is None:
            return {}
        if not isinstance(nested, Mapping):
            raise PolicyError(
                f"cannot score {self.source!r}: expected an object of factors, "
                f"got {type(nested).__name__}"
            )
        return nested

    def explain(self, values: Mapping[str, Any]) -> Dict[str, float]:
        """Each factor's contribution to the score, including the bias.

        The contributions always sum to :meth:`__call__`'s result — the
        explanation is where the score comes from, not a second calculation that
        might drift away from it.

        :raises PolicyError: if a weighted key holds a non-numeric value.
        """
        factors = self._factors(values)
        contributions: Dict[str, float] = {}
        if self.bias:
            contributions["bias"] = self.bias
        for key, weight in self.weights.items():
            raw = factors.get(key)
            if raw is None:
                continue
            if isinstance(raw, bool):
                raw = int(raw)
            elif not isinstance(raw, (int, float)):
                raise PolicyError(
                    f"cannot score {key!r}: expected a number, got {type(raw).__name__}"
                )
            contributions[key] = weight * float(raw)
        return contributions

    def __call__(self, values: Mapping[str, Any]) -> float:
        """Score ``values``.

        :raises PolicyError: if a weighted key holds a non-numeric value.
        """
        return self.bias + sum(
            value for key, value in self.explain(values).items() if key != "bias"
        )

    def to_dict(self) -> Dict[str, Any]:
        """A JSON-serialisable form of this scorer."""
        payload: Dict[str, Any] = {
            "bias": self.bias,
            "weights": dict(sorted(self.weights.items())),
        }
        if self.source is not None:
            payload["source"] = self.source
        return payload

    def __repr__(self) -> str:
        return (
            f"LinearScorer(weights={self.weights!r}, bias={self.bias!r}, "
            f"source={self.source!r})"
        )


class WeightedPolicy(BasePolicy):
    """Picks the highest-scoring action.

    >>> policy = WeightedPolicy(
    ...     name="router",
    ...     version="1.0",
    ...     scorers={
    ...         "primary": LinearScorer({"primary_healthy": 10.0}),
    ...         "secondary": LinearScorer({}, bias=1.0),
    ...     },
    ... )
    >>> policy.decide(DecisionContext({"primary_healthy": 1})).action
    'primary'
    >>> policy.decide(DecisionContext({"primary_healthy": 0})).action
    'secondary'

    :param name: policy identity, recorded on every decision.
    :param version: policy version, recorded on every decision.
    :param scorers: one callable per candidate action. Insertion order is the
        tie-break order, so put the action you would rather take first.
    :param minimum_score: actions scoring below this are not eligible. Without a
        ``default_action`` an empty eligible set is an error, which is usually
        what you want: it means the model has nothing to say.
    :param default_action: proposed when no action is eligible.

    :raises ConfigurationError: if the scorers or thresholds are unusable.

    The decision metadata carries every score, so an audit record explains not
    just what was chosen but what it beat.
    """

    deterministic = True

    def __init__(
        self,
        name: str,
        version: str,
        scorers: Mapping[str, Scorer],
        *,
        minimum_score: Optional[float] = None,
        default_action: Optional[str] = None,
    ) -> None:
        super().__init__(name, version)
        if not isinstance(scorers, Mapping) or not scorers:
            raise ConfigurationError("scorers must be a non-empty mapping of action -> callable")
        for action, scorer in scorers.items():
            if not isinstance(action, str) or not action:
                raise ConfigurationError("scorer actions must be non-empty strings")
            if not callable(scorer):
                raise ConfigurationError(f"scorer for {action!r} is not callable")
        if minimum_score is not None:
            if isinstance(minimum_score, bool) or not isinstance(minimum_score, (int, float)):
                raise ConfigurationError("minimum_score must be a number or None")
            if not math.isfinite(minimum_score):
                raise ConfigurationError("minimum_score must be finite")
            minimum_score = float(minimum_score)
        if default_action is not None and (
            not isinstance(default_action, str) or not default_action
        ):
            raise ConfigurationError("default_action must be a non-empty string or None")
        # dict() preserves insertion order, which is the documented tie-break.
        self._scorers: Dict[str, Scorer] = dict(scorers)
        self.minimum_score = minimum_score
        self.default_action = default_action

    @classmethod
    def from_linear(
        cls,
        name: str,
        version: str,
        weights: Mapping[str, Mapping[str, Any]],
        *,
        minimum_score: Optional[float] = None,
        default_action: Optional[str] = None,
    ) -> "WeightedPolicy":
        """Build a policy from pure data: ``{action: {"bias": .., "weights": {..}}}``.

        The resulting policy is equivalent to passing :class:`LinearScorer`
        instances, but the configuration can come from a JSON file with no code
        loading anywhere in the path.

        :raises ConfigurationError: if the weight specification is malformed.
        """
        if not isinstance(weights, Mapping) or not weights:
            raise ConfigurationError("weights must be a non-empty mapping of action -> spec")
        scorers: Dict[str, Scorer] = {}
        for action, spec in weights.items():
            if not isinstance(spec, Mapping):
                raise ConfigurationError(
                    f"weight spec for {action!r} must be a mapping, got {type(spec).__name__}"
                )
            scorers[action] = LinearScorer(
                spec.get("weights", {}),
                spec.get("bias", 0.0),
                source=spec.get("source"),
            )
        return cls(
            name,
            version,
            scorers,
            minimum_score=minimum_score,
            default_action=default_action,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialise back to the form :meth:`from_linear` accepts.

        :raises ConfigurationError: if any scorer is a plain callable. Arbitrary
            Python cannot be written to disk as data, and pretending otherwise
            would mean either dropping logic silently or serialising code.
        """
        scorers: Dict[str, Any] = {}
        for action, scorer in self._scorers.items():
            serialise = getattr(scorer, "to_dict", None)
            if not callable(serialise):
                raise ConfigurationError(
                    f"scorer for {action!r} is a plain callable and cannot be "
                    "serialised; use LinearScorer for configuration you intend to store"
                )
            scorers[action] = serialise()
        return {
            "name": self.name,
            "version": self.version,
            "scorers": scorers,
            "minimum_score": self.minimum_score,
            "default_action": self.default_action,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "WeightedPolicy":
        """Rebuild a policy from :meth:`to_dict` output.

        :raises ConfigurationError: if the payload is malformed.
        """
        if not isinstance(payload, Mapping):
            raise ConfigurationError(
                f"policy payload must be a mapping, got {type(payload).__name__}"
            )
        for required in ("name", "version", "scorers"):
            if required not in payload:
                raise ConfigurationError(f"policy payload is missing {required!r}")
        return cls.from_linear(
            payload["name"],
            payload["version"],
            payload["scorers"],
            minimum_score=payload.get("minimum_score"),
            default_action=payload.get("default_action"),
        )

    @property
    def actions(self) -> Tuple[str, ...]:
        """Every action this policy can propose, sorted."""
        candidates = set(self._scorers)
        if self.default_action:
            candidates.add(self.default_action)
        return tuple(sorted(candidates))

    def decide(self, context: DecisionContext) -> Decision:
        """Score every candidate and propose the winner.

        :raises PolicyError: if a scorer raises, returns a non-number or returns
            a non-finite number, or if nothing is eligible and no default is set.
        """
        values = context.values
        scores: Dict[str, float] = {}
        factors: Dict[str, Dict[str, float]] = {}
        for action, scorer in self._scorers.items():
            explain = getattr(scorer, "explain", None)
            try:
                if callable(explain):
                    # Derive the score from the explanation so the two cannot
                    # disagree. An explanation that does not add up to the score
                    # is worse than no explanation at all.
                    breakdown = explain(values)
                    if not isinstance(breakdown, Mapping):
                        raise PolicyError(
                            f"scorer for {action!r} explained itself as "
                            f"{type(breakdown).__name__}, expected a mapping"
                        )
                    factors[action] = {
                        str(name): round(float(value), 6)
                        for name, value in breakdown.items()
                    }
                    raw = sum(breakdown.values())
                else:
                    raw = scorer(values)
            except PolicyError:
                raise
            except Exception as exc:  # noqa: BLE001 - re-typed for the caller
                raise PolicyError(
                    f"scorer for {action!r} raised {type(exc).__name__}: {exc}"
                ) from exc
            if isinstance(raw, bool) or not isinstance(raw, (int, float)):
                raise PolicyError(
                    f"scorer for {action!r} returned {type(raw).__name__}, expected a number"
                )
            score = float(raw)
            if not math.isfinite(score):
                raise PolicyError(f"scorer for {action!r} returned a non-finite score")
            scores[action] = score

        eligible = [
            (action, score)
            for action, score in scores.items()
            if self.minimum_score is None or score >= self.minimum_score
        ]

        explanation: Dict[str, Any] = {
            "scores": {action: round(score, 6) for action, score in scores.items()}
        }
        if factors:
            explanation["factors"] = factors

        if not eligible:
            if self.default_action is None:
                raise PolicyError(
                    f"no action scored at or above minimum_score={self.minimum_score}; "
                    "configure default_action if that should be tolerated"
                )
            explanation.update(chosen_score=None, reason="below-minimum")
            return self.decision(self.default_action, explanation)

        # max() over the insertion-ordered list keeps the first of any tie.
        best_action, best_score = max(eligible, key=lambda item: item[1])
        explanation.update(chosen_score=round(best_score, 6), reason="highest-score")
        return self.decision(best_action, explanation)
