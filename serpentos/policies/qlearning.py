"""Adapter presenting the existing Q-learning agent as a runtime policy.

The tabular agent in :mod:`serpentos.core` predates the runtime and is not
changed by it. This is a read-only wrapper: it looks up a state in the Q-table,
picks the best action and reports it as a :class:`~serpentos.runtime.models.Decision`.
Training, checkpointing, benchmarking and policy import/export continue to go
through :mod:`serpentos.core` exactly as before.

Two properties matter and are enforced here:

* **The table is never modified.** Lookups go through ``QAgent.peek``, which
  returns a shared zero row for unseen states instead of inserting one. Running
  a million decisions through this adapter leaves the Q-table byte-identical, so
  the fingerprint you benchmarked is the fingerprint you deployed.
* **Determinism is honest.** Greedy mode breaks ties by action order and is
  fully replayable. Exploring mode draws from the agent's RNG, so the adapter
  reports ``deterministic = False`` and replay refuses to certify its matches.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence, Tuple

from .. import core
from ..runtime.errors import ConfigurationError, PolicyError
from ..runtime.models import Decision, DecisionContext
from ..runtime.policy import BasePolicy

__all__ = ["QLearningPolicy", "SNAKE_ACTIONS", "STATE_KEY_FIELD"]

#: Action names for the three relative moves, in Q-table column order.
SNAKE_ACTIONS: Tuple[str, str, str] = ("straight", "left", "right")

#: Context field the adapter reads the Q-table key from.
STATE_KEY_FIELD = "state_key"


class QLearningPolicy(BasePolicy):
    """Serves decisions from a trained Q-table.

    >>> agent = core.QAgent(q={"0|0|0|0|0|L|3|0": [1.0, 5.0, 2.0]})
    >>> policy = QLearningPolicy(agent, version="test")
    >>> policy.decide(DecisionContext({"state_key": "0|0|0|0|0|L|3|0"})).action
    'left'

    :param agent: a :class:`serpentos.core.QAgent`. Held by reference and never
        written to.
    :param actions: action names in Q-table column order. Must have exactly
        three entries to match the agent's action space.
    :param state_field: the context key holding the Q-table state key. Use
        :func:`serpentos.environments.snake.context_from_state` to build a
        context with the right shape.
    :param explore: when true, follow the agent's epsilon-greedy behaviour
        including its random exploration. Off by default: the runtime's job is
        to serve a trained policy, not to keep training it.
    :param name: policy identity.
    :param version: policy version. Defaults to a prefix of the Q-table
        fingerprint, so a table that changes gets a new version and strict
        replay notices.

    :raises ConfigurationError: if the agent or action names are unusable.
    """

    def __init__(
        self,
        agent: "core.QAgent",
        *,
        actions: Sequence[str] = SNAKE_ACTIONS,
        state_field: str = STATE_KEY_FIELD,
        explore: bool = False,
        name: str = "qlearning",
        version: Optional[str] = None,
    ) -> None:
        if not hasattr(agent, "peek") or not hasattr(agent, "q"):
            raise ConfigurationError(
                f"agent must be a QAgent-like object, got {type(agent).__name__}"
            )
        names = tuple(actions)
        if len(names) != core.N_ACTIONS:
            raise ConfigurationError(
                f"actions must name all {core.N_ACTIONS} Q-table columns, got {len(names)}"
            )
        for action in names:
            if not isinstance(action, str) or not action:
                raise ConfigurationError("action names must be non-empty strings")
        if len(set(names)) != len(names):
            raise ConfigurationError("action names must be unique")
        if not isinstance(state_field, str) or not state_field:
            raise ConfigurationError("state_field must be a non-empty string")

        self._agent = agent
        self._actions = names
        self._state_field = state_field
        self._explore = bool(explore)
        self._fingerprint = core.policy_fingerprint(agent.q)
        super().__init__(name, version or self._fingerprint[:16])
        # Exploration consults the agent's RNG, which replay cannot reproduce.
        self.deterministic = not self._explore

    # -- alternative constructors ------------------------------------
    @classmethod
    def from_policy_file(cls, path: str, **kwargs: Any) -> "QLearningPolicy":
        """Load an exported ``serpentos.policy/1`` file and wrap it.

        The file is pure data and is validated by
        :func:`serpentos.core.import_policy`; nothing in it is ever executed.

        :raises ConfigurationError: if the file is not a usable policy.
        """
        try:
            q, _payload = core.import_policy(path)
        except (OSError, ValueError) as exc:
            raise ConfigurationError(f"cannot load policy from {path}: {exc}") from exc
        return cls(core.QAgent(q=q), **kwargs)

    @classmethod
    def from_data_dir(cls, directory: Optional[str] = None, **kwargs: Any) -> "QLearningPolicy":
        """Wrap the Q-table checkpointed in a SerpentOS data directory.

        Reads the checkpoint the same way the game and the bot do, so a policy
        served here is the one the agent actually trained.
        """
        storage = core.Storage(directory)
        q, _meta = storage.load_checkpoint()
        return cls(core.QAgent(q=q), **kwargs)

    # -- identity ----------------------------------------------------
    @property
    def agent(self) -> "core.QAgent":
        """The wrapped agent. Treat as read-only."""
        return self._agent

    @property
    def fingerprint(self) -> str:
        """Content hash of the Q-table as it was when this policy was built."""
        return self._fingerprint

    @property
    def actions(self) -> Tuple[str, ...]:
        """The action names this policy can propose, sorted."""
        return tuple(sorted(self._actions))

    @property
    def states(self) -> int:
        """How many states the wrapped table knows."""
        return len(self._agent.q)

    # -- decisions ---------------------------------------------------
    def decide(self, context: DecisionContext) -> Decision:
        """Propose the best-known action for the state named in ``context``.

        An unknown state yields an all-zero row and therefore the first action.
        The decision metadata says ``known_state: false`` in that case, so an
        audit log distinguishes "the agent chose this" from "the agent has never
        seen this and defaulted".

        :raises PolicyError: if the context has no usable state key.
        """
        key = context.get(self._state_field)
        if not isinstance(key, str) or not key:
            raise PolicyError(
                f"context has no usable {self._state_field!r}: expected the Q-table "
                f"state key as a non-empty string, got {key!r}"
            )

        row = self._agent.peek(key)
        known = key in self._agent.q

        if self._explore and self._agent.rng.random() < self._agent.epsilon:
            index = self._agent.rng.randrange(core.N_ACTIONS)
            reason = "explore"
        else:
            best = max(row)
            # First index wins. QAgent.act breaks ties with its RNG; doing that
            # here would make greedy decisions unreplayable for no benefit.
            index = next(i for i, value in enumerate(row) if value == best)
            reason = "greedy"

        return self.decision(
            self._actions[index],
            {
                "state_key": key,
                "q_values": [round(float(value), 6) for value in row],
                "known_state": known,
                "reason": reason,
            },
        )

    def action_index(self, action: str) -> int:
        """Q-table column for ``action``, for feeding back into ``SnakeEnv.step``.

        :raises ConfigurationError: if the action is not one of this policy's.
        """
        try:
            return self._actions.index(action)
        except ValueError as exc:
            raise ConfigurationError(
                f"{action!r} is not one of {self._actions}"
            ) from exc

    def __repr__(self) -> str:
        return (
            f"QLearningPolicy(name={self.name!r}, version={self.version!r}, "
            f"states={self.states}, explore={self._explore})"
        )
