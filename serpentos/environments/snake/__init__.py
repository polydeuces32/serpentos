"""Snake: the built-in reference environment.

Snake is where SerpentOS started and it is still the most complete worked
example in the project — a real environment with state, actions, rewards,
persistence, a trained policy and a frozen reproducible benchmark. What changed
in 2.0 is its status: it is a *consumer* of the runtime, not the thing the
runtime is built around.

The game itself has not moved. :mod:`serpentos.core` still holds the rules, the
tabular agent and the on-disk formats; :mod:`serpentos.serpentos` still draws
the terminal UI and :mod:`serpentos.bot` still runs it headlessly. This package
re-exports those pieces under the environment namespace and adds the glue that
connects them to the runtime:

>>> from serpentos import ActionValidator, DecisionEngine
>>> from serpentos.environments.snake import SNAKE_ACTIONS, run_policy_episode, survival_policy
>>> engine = DecisionEngine(survival_policy(), validator=ActionValidator(SNAKE_ACTIONS))
>>> result = run_policy_episode(engine, rows=12, cols=20, seed=7)
>>> result.score >= 0
True
"""

from __future__ import annotations

from ...core import (
    BENCHMARK_SPEC,
    DIFFICULTY,
    PRESETS,
    QAgent,
    SnakeEnv,
    State,
    StepInfo,
    Storage,
    Transition,
    export_policy,
    import_policy,
    load_agent,
    policy_fingerprint,
    run_benchmark,
    run_episode,
)
from .adapter import (
    SNAKE_ACTIONS,
    EpisodeResult,
    action_index,
    context_from_state,
    evaluate_policy,
    outcome_from_episode,
    run_policy_episode,
    survival_policy,
)

__all__ = [
    # the game, re-exported from serpentos.core
    "SnakeEnv",
    "State",
    "StepInfo",
    "Transition",
    "QAgent",
    "Storage",
    "run_episode",
    "load_agent",
    "run_benchmark",
    "policy_fingerprint",
    "export_policy",
    "import_policy",
    "PRESETS",
    "DIFFICULTY",
    "BENCHMARK_SPEC",
    # runtime glue
    "SNAKE_ACTIONS",
    "EpisodeResult",
    "action_index",
    "context_from_state",
    "evaluate_policy",
    "outcome_from_episode",
    "run_policy_episode",
    "survival_policy",
]
