"""Snake as a reference consumer of the runtime.

These tests are the proof that the abstraction is real: the same engine,
validator and audit log drive a learned Q-table and a hand-written rule set
through an actual game.
"""

import random
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos import core
from serpentos.environments.snake import (
    SNAKE_ACTIONS,
    action_index,
    context_from_state,
    evaluate_policy,
    outcome_from_episode,
    run_policy_episode,
    survival_policy,
)
from serpentos.policies.qlearning import QLearningPolicy
from serpentos.runtime.audit import InMemoryAuditLog
from serpentos.runtime.comparison import compare
from serpentos.runtime.engine import DecisionEngine
from serpentos.runtime.errors import ConfigurationError, DecisionValidationError
from serpentos.runtime.models import DecisionContext
from serpentos.runtime.policy import BasePolicy
from serpentos.runtime.replay import replay_all
from serpentos.runtime.validation import ActionValidator


def fresh_env(seed=1, rows=12, cols=20):
    return core.SnakeEnv(rows, cols, shaping=False, max_steps=400, rng=random.Random(seed))


class ContextTest(unittest.TestCase):
    def test_carries_the_exact_qtable_key(self):
        state = fresh_env().reset()
        self.assertEqual(context_from_state(state)["state_key"], state.key)

    def test_publishes_the_raw_observation(self):
        state = fresh_env().reset()
        context = context_from_state(state)
        for field in ("dx", "dy", "danger_ahead", "direction", "wall_dist", "length_bucket"):
            self.assertEqual(context[field], getattr(state, field))

    def test_derived_food_flags_agree_with_the_geometry(self):
        # Heading left with food directly to the west: straight ahead.
        state = core.State(dx=-1, dy=0, danger_ahead=0, danger_left=0, danger_right=0,
                           direction="L", wall_dist=4, length_bucket=0)
        context = context_from_state(state)
        self.assertTrue(context["food_ahead"])
        self.assertFalse(context["food_left"])
        self.assertFalse(context["food_right"])

        # Heading left with food to the north: that is a right turn.
        state = state._replace(dx=0, dy=-1)
        context = context_from_state(state)
        self.assertFalse(context["food_ahead"])
        self.assertTrue(context["food_right"])
        self.assertFalse(context["food_left"])

    def test_context_is_json_serialisable(self):
        context = context_from_state(fresh_env().reset(), request_id="ep-1")
        self.assertEqual(DecisionContext.from_dict(context.to_dict()), context)
        self.assertEqual(context.request_id, "ep-1")

    def test_action_names_map_to_environment_actions(self):
        self.assertEqual([action_index(name) for name in SNAKE_ACTIONS], [0, 1, 2])
        with self.assertRaises(ConfigurationError):
            action_index("diagonal")


class SurvivalPolicyTest(unittest.TestCase):
    def setUp(self):
        self.policy = survival_policy()

    def test_only_proposes_real_snake_actions(self):
        self.assertEqual(set(self.policy.actions) - set(SNAKE_ACTIONS), set())

    def test_turns_away_from_a_wall_ahead(self):
        state = core.State(0, 0, danger_ahead=1, danger_left=0, danger_right=1,
                           direction="U", wall_dist=0, length_bucket=0)
        self.assertEqual(self.policy.decide(context_from_state(state)).action, "left")

    def test_turns_the_other_way_when_left_is_blocked_too(self):
        state = core.State(0, 0, danger_ahead=1, danger_left=1, danger_right=0,
                           direction="U", wall_dist=0, length_bucket=0)
        self.assertEqual(self.policy.decide(context_from_state(state)).action, "right")

    def test_boxed_in_it_goes_straight_and_says_so(self):
        state = core.State(0, 0, danger_ahead=1, danger_left=1, danger_right=1,
                           direction="U", wall_dist=0, length_bucket=0)
        decision = self.policy.decide(context_from_state(state))
        self.assertEqual(decision.action, "straight")
        self.assertEqual(decision.metadata["rule"], "trapped")

    def test_steers_towards_food(self):
        state = core.State(dx=0, dy=-1, danger_ahead=0, danger_left=0, danger_right=0,
                           direction="L", wall_dist=4, length_bucket=0)
        decision = self.policy.decide(context_from_state(state))
        self.assertEqual(decision.action, "right")
        self.assertEqual(decision.metadata["rule"], "chase-right")

    def test_it_actually_plays_the_game(self):
        # No training, no Q-table: eight rules feeding the same engine.
        summary = evaluate_policy(self.policy, episodes=10, rows=22, cols=78)
        self.assertGreater(summary["mean_score"], 5)

    def test_it_beats_an_untrained_qtable_by_a_wide_margin(self):
        untrained = QLearningPolicy(core.QAgent(q={}), version="empty")
        rules = evaluate_policy(self.policy, episodes=10, rows=22, cols=78)
        blank = evaluate_policy(untrained, episodes=10, rows=22, cols=78)
        self.assertGreater(rules["mean_score"], blank["mean_score"] + 5)


class EpisodeTest(unittest.TestCase):
    def setUp(self):
        self.engine = DecisionEngine(
            survival_policy(), validator=ActionValidator(SNAKE_ACTIONS)
        )

    def test_runs_an_episode_to_completion(self):
        result = run_policy_episode(self.engine, fresh_env(seed=5))
        self.assertGreaterEqual(result.score, 0)
        self.assertGreater(result.steps, 0)
        self.assertEqual(result.decisions, result.steps)
        self.assertIn(result.reason, ("wall", "body", "truncated", "won"))

    def test_the_same_seed_reproduces_the_same_episode(self):
        first = run_policy_episode(self.engine, fresh_env(seed=11))
        second = run_policy_episode(self.engine, fresh_env(seed=11))
        self.assertEqual(first, second)

    def test_builds_its_own_environment_when_not_given_one(self):
        result = run_policy_episode(self.engine, rows=12, cols=20, seed=3, max_steps=100)
        self.assertLessEqual(result.steps, 100)

    def test_every_step_is_audited(self):
        audit = InMemoryAuditLog()
        engine = DecisionEngine(
            survival_policy(), validator=ActionValidator(SNAKE_ACTIONS), audit_sink=audit
        )
        result = run_policy_episode(engine, fresh_env(seed=2))
        self.assertEqual(len(audit), result.steps)
        self.assertTrue(all(record.validation_result.valid for record in audit.records))

    def test_a_recorded_episode_replays_exactly(self):
        audit = InMemoryAuditLog()
        policy = survival_policy()
        engine = DecisionEngine(policy, audit_sink=audit)
        run_policy_episode(engine, fresh_env(seed=4))
        report = replay_all(policy, audit.records)
        self.assertGreater(report.total, 10)
        self.assertEqual(report.mismatched, 0)
        self.assertTrue(report.guaranteed)

    def test_a_rogue_action_is_blocked_before_the_game_sees_it(self):
        class RoguePolicy(BasePolicy):
            def __init__(self):
                super().__init__("rogue", "1.0")

            def decide(self, context):
                return self.decision("teleport")

        engine = DecisionEngine(RoguePolicy(), validator=ActionValidator(SNAKE_ACTIONS))
        env = fresh_env()
        with self.assertRaises(DecisionValidationError):
            run_policy_episode(engine, env)
        self.assertEqual(env.steps, 0)

    def test_outcomes_describe_the_episode(self):
        result = run_policy_episode(self.engine, fresh_env(seed=6))
        outcome = outcome_from_episode(result)
        self.assertEqual(outcome.metrics["score"], float(result.score))
        self.assertEqual(outcome.success, result.reason not in ("wall", "body"))
        self.assertEqual(outcome.metadata["reason"], result.reason)


class CrossPolicyComparisonTest(unittest.TestCase):
    def test_two_unrelated_policies_are_comparable_on_the_same_contexts(self):
        env = fresh_env(seed=9)
        state = env.reset()
        cases = []
        engine = DecisionEngine(survival_policy())
        for _ in range(60):
            context = context_from_state(state)
            cases.append(context)
            state, _reward, done, _info = env.step(
                action_index(engine.decide(context).action)
            )
            if done:
                break

        report = compare(
            [survival_policy(), QLearningPolicy(core.QAgent(q={}), version="empty")],
            cases,
            validator=ActionValidator(SNAKE_ACTIONS),
        )
        self.assertEqual(len(report.reports), 2)
        for policy_report in report.reports:
            self.assertEqual(policy_report.errors, 0)
            self.assertEqual(policy_report.validation_failures, 0)
            self.assertEqual(policy_report.decisions, len(cases))
            self.assertTrue(set(policy_report.action_counts) <= set(SNAKE_ACTIONS))


class EvaluateArgumentTest(unittest.TestCase):
    def test_episode_count_must_be_positive(self):
        with self.assertRaises(ConfigurationError):
            evaluate_policy(survival_policy(), episodes=0)


if __name__ == "__main__":
    unittest.main()
