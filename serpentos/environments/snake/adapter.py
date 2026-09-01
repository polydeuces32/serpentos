"""Snake, expressed in the runtime's vocabulary.

This module is the worked example the rest of the documentation points at. It
does the three things every real integration has to do:

1. **Build a context.** :func:`context_from_state` turns the agent-visible game
   state into plain JSON. Note that it publishes *derived* features —
   ``food_left``, ``food_ahead`` — alongside the raw ones. Deciding what a
   policy is allowed to see, and pre-computing anything awkward, is the host
   application's job, not the policy's.

2. **Execute the action.** :func:`action_index` maps ``"left"`` back onto the
   integer ``SnakeEnv.step`` wants. The runtime never touches the environment.

3. **Report an outcome.** :func:`outcome_from_episode` says what "success" meant
   here — surviving to the step limit rather than crashing. That is this
   environment's definition, chosen by this environment. The runtime has no
   opinion.

The payoff is :func:`survival_policy`: eight hand-written rules, no training, no
Q-table, playing the same game through the same interface as the learned agent.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Tuple

from ... import core
from ...policies.rules import AllOf, Rule, RulePolicy, when
from ...runtime.engine import DecisionEngine
from ...runtime.errors import ConfigurationError
from ...runtime.models import DecisionContext, Outcome
from ...runtime.policy import Policy

__all__ = [
    "SNAKE_ACTIONS",
    "EpisodeResult",
    "action_index",
    "context_from_state",
    "evaluate_policy",
    "outcome_from_episode",
    "run_policy_episode",
    "survival_policy",
]

#: Action names in Q-table column order: index 0 is straight, 1 left, 2 right.
SNAKE_ACTIONS: Tuple[str, str, str] = ("straight", "left", "right")

_ACTION_INDEX: Mapping[str, int] = {name: i for i, name in enumerate(SNAKE_ACTIONS)}


def action_index(action: str) -> int:
    """Q-table column / :meth:`SnakeEnv.step` argument for ``action``.

    :raises ConfigurationError: if the action is not one of :data:`SNAKE_ACTIONS`.
    """
    try:
        return _ACTION_INDEX[action]
    except (KeyError, TypeError) as exc:
        raise ConfigurationError(
            f"{action!r} is not a snake action; expected one of {SNAKE_ACTIONS}"
        ) from exc


def context_from_state(
    state: "core.State", *, request_id: Optional[str] = None
) -> DecisionContext:
    """Turn a :class:`serpentos.core.State` into a decision context.

    The context carries three groups of values:

    * ``state_key`` — the exact Q-table key, so
      :class:`~serpentos.policies.qlearning.QLearningPolicy` can look the state
      up without knowing anything about Snake.
    * the raw observation — ``dx``, ``dy``, the three danger bits, ``direction``,
      ``wall_dist``, ``length_bucket``.
    * derived relative-food flags — ``food_ahead``, ``food_left``, ``food_right``
      — computed from ``direction`` and the food offset so that a rule policy can
      steer without doing trigonometry in its conditions.

    All values are JSON scalars, so a context can be logged, replayed and diffed.
    """
    ahead_y, ahead_x = core.VEC[state.direction]
    left_y, left_x = core.VEC[core.turn_left(state.direction)]
    right_y, right_x = core.VEC[core.turn_right(state.direction)]

    return DecisionContext(
        {
            "state_key": state.key,
            "dx": state.dx,
            "dy": state.dy,
            "danger_ahead": state.danger_ahead,
            "danger_left": state.danger_left,
            "danger_right": state.danger_right,
            "direction": state.direction,
            "wall_dist": state.wall_dist,
            "length_bucket": state.length_bucket,
            "food_ahead": state.dy * ahead_y + state.dx * ahead_x > 0,
            "food_left": state.dy * left_y + state.dx * left_x > 0,
            "food_right": state.dy * right_y + state.dx * right_x > 0,
        },
        request_id,
    )


# =========================
# A HAND-WRITTEN SNAKE POLICY
# =========================
def survival_policy(name: str = "snake-survival", version: str = "1.0") -> RulePolicy:
    """Eight rules that play Snake without any learning at all.

    Avoid the wall in front of you; prefer the turn that also heads towards the
    food; otherwise steer towards the food when the way is clear; otherwise keep
    going. First match wins, so the danger rules dominate.

    It is not as good as a well-trained Q-table and it is not meant to be. It is
    here to show that the interface is not shaped around reinforcement learning:
    the same engine, the same validator and the same audit log serve a lookup
    table and a handful of if-statements identically.
    """
    blocked_ahead = when("danger_ahead", "eq", 1)
    return RulePolicy(
        name=name,
        version=version,
        rules=[
            Rule(
                "left",
                AllOf(blocked_ahead, when("danger_left", "eq", 0), when("food_left", "eq", True)),
                name="dodge-left-towards-food",
            ),
            Rule(
                "right",
                AllOf(blocked_ahead, when("danger_right", "eq", 0), when("food_right", "eq", True)),
                name="dodge-right-towards-food",
            ),
            Rule(
                "left",
                AllOf(blocked_ahead, when("danger_left", "eq", 0)),
                name="dodge-left",
            ),
            Rule(
                "right",
                AllOf(blocked_ahead, when("danger_right", "eq", 0)),
                name="dodge-right",
            ),
            Rule("straight", blocked_ahead, name="trapped"),
            Rule(
                "straight",
                AllOf(when("food_ahead", "eq", True), when("danger_ahead", "eq", 0)),
                name="chase-ahead",
            ),
            Rule(
                "left",
                AllOf(when("food_left", "eq", True), when("danger_left", "eq", 0)),
                name="chase-left",
            ),
            Rule(
                "right",
                AllOf(when("food_right", "eq", True), when("danger_right", "eq", 0)),
                name="chase-right",
            ),
        ],
        default_action="straight",
    )


# =========================
# RUNNING AN EPISODE
# =========================
@dataclass(frozen=True)
class EpisodeResult:
    """What one policy-driven episode achieved."""

    score: int
    steps: int
    reason: str
    total_reward: float
    decisions: int

    def to_dict(self) -> Dict[str, Any]:
        """A mutable, JSON-ready copy."""
        return {
            "score": self.score,
            "steps": self.steps,
            "reason": self.reason,
            "total_reward": round(self.total_reward, 4),
            "decisions": self.decisions,
        }


def outcome_from_episode(result: EpisodeResult) -> Outcome:
    """Report an episode as a runtime :class:`~serpentos.runtime.models.Outcome`.

    Success here means the snake did not crash — it ran out of steps or filled
    the board. That is a choice this environment makes; a different application
    would choose differently, and the runtime would not care either way.
    """
    return Outcome(
        success=result.reason not in ("wall", "body"),
        metrics={
            "score": float(result.score),
            "steps": float(result.steps),
            "total_reward": result.total_reward,
        },
        metadata={"reason": result.reason},
    )


def run_policy_episode(
    engine: DecisionEngine,
    env: Optional["core.SnakeEnv"] = None,
    *,
    rows: int = 22,
    cols: int = 78,
    seed: Optional[int] = None,
    max_steps: int = 1800,
    request_id: Optional[str] = None,
) -> EpisodeResult:
    """Play one episode, taking every action from ``engine``.

    The loop is the whole integration: observe, build a context, ask the engine,
    execute the action it returns. Note that the engine's decision is *proposed*
    — this function is the host application, and executing it is this function's
    responsibility, not the runtime's.

    :param engine: the configured engine. Any validator and audit sink attached
        to it apply to every step, so a full episode is auditable.
    :param env: an existing environment, or ``None`` to build one from ``rows``,
        ``cols``, ``seed`` and ``max_steps``.

    :raises DecisionValidationError: if the policy proposes an action the
        engine's validator refuses. Nothing is executed in that case.
    :raises ConfigurationError: if the policy proposes an action Snake does not
        have.
    """
    if env is None:
        env = core.SnakeEnv(
            rows,
            cols,
            shaping=False,
            max_steps=max_steps,
            rng=random.Random(seed),
        )

    state = env.reset()
    total_reward = 0.0
    decisions = 0
    while True:
        decision = engine.decide(context_from_state(state, request_id=request_id))
        decisions += 1
        state, reward, done, info = env.step(action_index(decision.action))
        total_reward += reward
        if done:
            return EpisodeResult(
                score=info.score,
                steps=info.steps,
                reason=info.reason or "done",
                total_reward=total_reward,
                decisions=decisions,
            )


def evaluate_policy(
    policy: Policy,
    *,
    episodes: int = 20,
    rows: int = 22,
    cols: int = 78,
    seed_base: int = 1000,
    max_steps: int = 1800,
    engine_factory=None,
) -> Dict[str, Any]:
    """Play ``episodes`` seeded episodes and summarise the scores.

    Every episode gets its own seeded RNG, so the result depends only on the
    policy. This is a convenience for comparing policies on the reference
    environment; the frozen, citable benchmark for Q-tables remains
    :func:`serpentos.core.run_benchmark` and is untouched by any of this.
    """
    if episodes <= 0:
        raise ConfigurationError("episodes must be positive")
    engine = (engine_factory or DecisionEngine)(policy)
    scores: List[int] = []
    steps: List[int] = []
    survived = 0
    for index in range(episodes):
        env = core.SnakeEnv(
            rows,
            cols,
            shaping=False,
            max_steps=max_steps,
            rng=random.Random(seed_base + index),
        )
        result = run_policy_episode(engine, env)
        scores.append(result.score)
        steps.append(result.steps)
        survived += 1 if result.reason not in ("wall", "body") else 0
    return {
        "policy": policy.name,
        "version": policy.version,
        "episodes": episodes,
        "mean_score": round(sum(scores) / len(scores), 4),
        "best_score": max(scores),
        "worst_score": min(scores),
        "mean_steps": round(sum(steps) / len(steps), 2),
        "survival_rate": round(survived / episodes, 4),
    }
