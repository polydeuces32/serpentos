"""Tests for the Q-learning policy adapter.

The adapter must be strictly read-only. Existing training, checkpointing,
benchmarking and export behaviour is covered by ``test_core.py``; what matters
here is that serving decisions through the runtime never disturbs any of it.
"""

import copy
import json
import os
import random
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from serpentos import core
from serpentos.environments.snake import context_from_state
from serpentos.policies.qlearning import QLearningPolicy
from serpentos.runtime.errors import ConfigurationError, PolicyError
from serpentos.runtime.models import DecisionContext

KEY = "0|0|0|0|0|L|3|0"


def trained_agent(episodes=200, seed=7):
    agent = core.QAgent(core.PRESETS["BOLD"], rng=random.Random(seed))
    for index in range(episodes):
        env = core.SnakeEnv(12, 20, rng=random.Random(index))
        core.run_episode(env, agent)
    return agent


class DecisionTest(unittest.TestCase):
    def setUp(self):
        self.agent = core.QAgent(q={KEY: [1.0, 5.0, 2.0]})
        self.policy = QLearningPolicy(self.agent, version="test")

    def test_picks_the_highest_valued_action(self):
        decision = self.policy.decide(DecisionContext({"state_key": KEY}))
        self.assertEqual(decision.action, "left")

    def test_metadata_explains_the_lookup(self):
        decision = self.policy.decide(DecisionContext({"state_key": KEY}))
        self.assertEqual(decision.metadata["state_key"], KEY)
        self.assertEqual(list(decision.metadata["q_values"]), [1.0, 5.0, 2.0])
        self.assertTrue(decision.metadata["known_state"])
        self.assertEqual(decision.metadata["reason"], "greedy")

    def test_an_unknown_state_is_flagged_as_such(self):
        decision = self.policy.decide(DecisionContext({"state_key": "1|1|0|0|0|U|2|0"}))
        self.assertFalse(decision.metadata["known_state"])
        self.assertEqual(list(decision.metadata["q_values"]), [0.0, 0.0, 0.0])
        self.assertEqual(decision.action, "straight")

    def test_ties_break_towards_the_first_action(self):
        policy = QLearningPolicy(core.QAgent(q={KEY: [3.0, 3.0, 1.0]}), version="t")
        self.assertEqual(policy.decide(DecisionContext({"state_key": KEY})).action, "straight")

    def test_repeated_decisions_are_identical(self):
        context = DecisionContext({"state_key": KEY})
        self.assertEqual(
            self.policy.decide(context).to_json(), self.policy.decide(context).to_json()
        )

    def test_a_missing_state_key_is_a_policy_error(self):
        for values in ({}, {"state_key": ""}, {"state_key": 7}):
            with self.assertRaises(PolicyError):
                self.policy.decide(DecisionContext(values))

    def test_action_index_maps_back_to_the_environment(self):
        self.assertEqual(self.policy.action_index("straight"), 0)
        self.assertEqual(self.policy.action_index("left"), 1)
        self.assertEqual(self.policy.action_index("right"), 2)
        with self.assertRaises(ConfigurationError):
            self.policy.action_index("sideways")


class ReadOnlyTest(unittest.TestCase):
    def test_serving_decisions_never_grows_the_table(self):
        agent = core.QAgent(q={KEY: [1.0, 5.0, 2.0]})
        policy = QLearningPolicy(agent, version="t")
        before = copy.deepcopy(agent.q)
        for index in range(50):
            policy.decide(DecisionContext({"state_key": f"{index}|0|0|0|0|L|3|0"}))
        self.assertEqual(agent.q, before)
        self.assertEqual(len(agent.q), 1)

    def test_the_fingerprint_survives_a_full_episode_of_decisions(self):
        agent = trained_agent()
        fingerprint = core.policy_fingerprint(agent.q)
        policy = QLearningPolicy(agent)
        env = core.SnakeEnv(12, 20, rng=random.Random(3))
        state = env.reset()
        for _ in range(200):
            decision = policy.decide(context_from_state(state))
            state, _reward, done, _info = env.step(policy.action_index(decision.action))
            if done:
                break
        self.assertEqual(core.policy_fingerprint(agent.q), fingerprint)

    def test_the_benchmark_is_unchanged_by_serving(self):
        agent = trained_agent()
        before = core.run_benchmark(agent.q)
        policy = QLearningPolicy(agent)
        for index in range(100):
            policy.decide(DecisionContext({"state_key": f"{index}|1|0|0|0|U|2|1"}))
        self.assertEqual(core.run_benchmark(agent.q), before)

    def test_does_not_disturb_the_agents_live_telemetry(self):
        agent = core.QAgent(q={KEY: [1.0, 5.0, 2.0]})
        agent.last_qvals = [9.0, 9.0, 9.0]
        agent.last_action = 2
        QLearningPolicy(agent, version="t").decide(DecisionContext({"state_key": KEY}))
        self.assertEqual(agent.last_qvals, [9.0, 9.0, 9.0])
        self.assertEqual(agent.last_action, 2)


class DeterminismTest(unittest.TestCase):
    def test_greedy_mode_is_declared_deterministic(self):
        self.assertTrue(QLearningPolicy(core.QAgent(q={}), version="t").deterministic)

    def test_exploring_mode_is_declared_nondeterministic(self):
        policy = QLearningPolicy(core.QAgent(q={}), version="t", explore=True)
        self.assertFalse(policy.deterministic)

    def test_exploring_mode_can_pick_a_non_greedy_action(self):
        agent = core.QAgent(q={KEY: [10.0, 0.0, 0.0]}, rng=random.Random(1))
        agent.epsilon = 1.0
        policy = QLearningPolicy(agent, version="t", explore=True)
        reasons = {
            policy.decide(DecisionContext({"state_key": KEY})).metadata["reason"]
            for _ in range(20)
        }
        self.assertEqual(reasons, {"explore"})

    def test_greedy_agrees_with_the_agent_when_there_is_a_clear_winner(self):
        agent = trained_agent()
        policy = QLearningPolicy(agent)
        agent.epsilon = 0.0
        checked = 0
        for key, row in list(agent.q.items())[:200]:
            if row.count(max(row)) != 1:
                continue  # QAgent breaks ties with its RNG; the adapter does not.
            expected = row.index(max(row))
            actual = policy.action_index(
                policy.decide(DecisionContext({"state_key": key})).action
            )
            self.assertEqual(actual, expected)
            checked += 1
        self.assertGreater(checked, 0)


class ConstructionTest(unittest.TestCase):
    def test_version_defaults_to_the_table_fingerprint(self):
        agent = core.QAgent(q={KEY: [1.0, 0.0, 0.0]})
        policy = QLearningPolicy(agent)
        self.assertEqual(policy.version, core.policy_fingerprint(agent.q)[:16])

    def test_a_changed_table_produces_a_different_version(self):
        first = QLearningPolicy(core.QAgent(q={KEY: [1.0, 0.0, 0.0]}))
        second = QLearningPolicy(core.QAgent(q={KEY: [2.0, 0.0, 0.0]}))
        self.assertNotEqual(first.version, second.version)

    def test_reports_the_table_size(self):
        self.assertEqual(QLearningPolicy(core.QAgent(q={KEY: [0.0] * 3})).states, 1)

    def test_construction_is_validated(self):
        with self.assertRaises(ConfigurationError):
            QLearningPolicy(object())
        with self.assertRaises(ConfigurationError):
            QLearningPolicy(core.QAgent(q={}), actions=("a", "b"))
        with self.assertRaises(ConfigurationError):
            QLearningPolicy(core.QAgent(q={}), actions=("a", "a", "b"))
        with self.assertRaises(ConfigurationError):
            QLearningPolicy(core.QAgent(q={}), state_field="")

    def test_custom_action_names(self):
        policy = QLearningPolicy(
            core.QAgent(q={KEY: [0.0, 1.0, 0.0]}), actions=("hold", "port", "starboard"),
            version="t",
        )
        self.assertEqual(policy.decide(DecisionContext({"state_key": KEY})).action, "port")


class LoadingTest(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()

    def test_loads_an_exported_policy_file(self):
        path = os.path.join(self.dir, "policy.json")
        core.export_policy(path, {KEY: [1.0, 5.0, 2.0]}, name="test")
        policy = QLearningPolicy.from_policy_file(path)
        self.assertEqual(policy.decide(DecisionContext({"state_key": KEY})).action, "left")

    def test_refuses_a_file_that_is_not_a_policy(self):
        path = os.path.join(self.dir, "not-a-policy.json")
        Path(path).write_text(json.dumps({"format": "something/else"}), encoding="utf-8")
        with self.assertRaises(ConfigurationError):
            QLearningPolicy.from_policy_file(path)

    def test_refuses_a_missing_file(self):
        with self.assertRaises(ConfigurationError):
            QLearningPolicy.from_policy_file(os.path.join(self.dir, "nope.json"))

    def test_loads_a_checkpoint_from_a_data_directory(self):
        storage = core.Storage(self.dir)
        storage.save_checkpoint({KEY: [0.0, 0.0, 9.0]}, {"episodes": 1})
        policy = QLearningPolicy.from_data_dir(self.dir)
        self.assertEqual(policy.decide(DecisionContext({"state_key": KEY})).action, "right")

    def test_an_empty_data_directory_yields_an_empty_policy(self):
        self.assertEqual(QLearningPolicy.from_data_dir(self.dir).states, 0)


if __name__ == "__main__":
    unittest.main()
