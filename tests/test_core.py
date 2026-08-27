"""Tests for the headless SerpentOS core."""

import json
import os
import random
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from serpentos import core  # noqa: E402


class EnvTest(unittest.TestCase):
    def env(self, rows=10, cols=20, **kwargs):
        kwargs.setdefault("rng", random.Random(1))
        return core.SnakeEnv(rows, cols, **kwargs)

    def test_rejects_undersized_grid(self):
        with self.assertRaises(ValueError):
            core.SnakeEnv(2, 2)

    def test_reset_places_snake_inside_grid(self):
        env = self.env()
        for y, x in env.snake:
            self.assertTrue(env.in_bounds(y, x))
        self.assertEqual(env.score, 0)
        self.assertEqual(env.direction, "L")

    def test_food_never_spawns_on_the_snake(self):
        env = self.env()
        for _ in range(300):
            env.reset()
            self.assertNotIn(env.food, set(env.snake))

    def test_food_spawns_on_a_full_board_or_ends_the_episode(self):
        env = self.env(rows=6, cols=10)
        env.snake = [(y, x) for y in range(6) for x in range(10)][:59]
        env._occupied = set(env.snake)
        food = env._spawn_food()
        self.assertNotIn(food, env._occupied)

    def test_wall_collision_terminates(self):
        env = self.env(rows=6, cols=10)
        info = None
        for _ in range(20):
            _, reward, done, info = env.step_dir("L")
            if done:
                break
        self.assertTrue(info.done)
        self.assertEqual(info.reason, "wall")
        self.assertEqual(reward, env.death_penalty)

    def test_body_collision_terminates(self):
        env = self.env()
        # Head at (5,5) facing up; (4,5) is a mid-body segment, not the tail.
        env.snake = [(5, 5), (5, 6), (4, 6), (4, 5), (3, 5)]
        env._occupied = set(env.snake)
        env.direction = "U"
        env.food = (0, 0)
        env.done = False
        _, reward, done, info = env.step_dir("U")
        self.assertTrue(done)
        self.assertEqual(info.reason, "body")
        self.assertEqual(reward, env.death_penalty)

    def test_moving_into_the_vacating_tail_is_legal(self):
        env = self.env()
        env.snake = [(5, 5), (5, 6), (4, 6), (4, 5)]
        env._occupied = set(env.snake)
        env.direction = "U"
        env.food = (0, 0)
        env.done = False
        _, _, done, _ = env.step_dir("L")  # into (4, 5), the tail cell
        self.assertFalse(done)

    def test_eating_grows_the_snake_and_pays_out(self):
        env = self.env(shaping=False)
        env.snake = [(5, 5), (5, 6), (5, 7)]
        env._occupied = set(env.snake)
        env.direction = "L"
        env.food = (5, 4)
        env.done = False
        length = len(env.snake)
        _, reward, done, info = env.step_dir("L")
        self.assertFalse(done)
        self.assertEqual(reward, env.food_reward)
        self.assertEqual(info.score, 1)
        self.assertEqual(len(env.snake), length + 1)

    def test_reversal_is_ignored(self):
        env = self.env()
        head = env.snake[0]
        env.step_dir("R")  # snake faces L; a reversal must not happen
        self.assertEqual(env.direction, "L")
        self.assertEqual(env.snake[0], (head[0], head[1] - 1))

    def test_truncation_is_flagged_not_fatal(self):
        env = self.env(max_steps=3, shaping=False)
        info = None
        for _ in range(3):
            _, _, done, info = env.step_dir(env.direction)
            if done:
                break
        self.assertTrue(info.truncated)
        self.assertEqual(info.reason, "truncated")

    def test_stepping_a_finished_episode_raises(self):
        env = self.env(rows=6, cols=10)
        while not env.done:
            env.step_dir("L")
        with self.assertRaises(RuntimeError):
            env.step_dir("L")

    def test_shaping_skips_the_eating_step(self):
        env = self.env(shaping=True)
        env.snake = [(5, 5), (5, 6), (5, 7)]
        env._occupied = set(env.snake)
        env.direction = "L"
        env.food = (5, 4)
        env.done = False
        _, reward, _, _ = env.step_dir("L")
        self.assertEqual(reward, env.food_reward)

    def test_same_seed_reproduces_the_same_episode(self):
        def rollout(seed):
            env = core.SnakeEnv(10, 20, rng=random.Random(seed))
            agent = core.QAgent(rng=random.Random(seed))
            return [core.run_episode(env, agent).score for _ in range(20)]

        self.assertEqual(rollout(7), rollout(7))


class StateTest(unittest.TestCase):
    def test_key_is_v1_wire_compatible(self):
        state = core.State(-1, 1, 0, 1, 0, "L", 3, 2)
        self.assertEqual(state.key, "-1|1|0|1|0|L|3|2")

    def test_dangers_reflect_walls(self):
        env = core.SnakeEnv(6, 10, rng=random.Random(0))
        env.snake = [(0, 5), (1, 5)]
        env._occupied = set(env.snake)
        env.direction = "U"
        env.food = (3, 3)
        state = env.state()
        self.assertEqual(state.danger_ahead, 1)  # wall above
        self.assertEqual(state.wall_dist, 0)

    def test_wall_distance_clamps_at_four(self):
        env = core.SnakeEnv(20, 20, rng=random.Random(0))
        env.snake = [(19, 10)]
        env._occupied = set(env.snake)
        env.direction = "U"
        env.food = (0, 0)
        self.assertEqual(env.state().wall_dist, 4)

    def test_ui_and_bot_share_one_keyspace(self):
        """Regression: turbo training used to write 'K|<int>' keys that live
        play could never read, so training transferred nothing."""
        env = core.SnakeEnv(12, 24, rng=random.Random(3))
        agent = core.QAgent(rng=random.Random(3))
        core.run_episode(env, agent)
        self.assertTrue(agent.q)
        for key in agent.q:
            self.assertEqual(len(key.split("|")), 8)
            self.assertFalse(key.startswith("K|"))


class AgentTest(unittest.TestCase):
    def test_learn_moves_q_towards_the_target(self):
        agent = core.QAgent(rng=random.Random(0))
        state = core.State(0, 0, 0, 0, 0, "L", 4, 0)
        nxt = core.State(1, 0, 0, 0, 0, "L", 4, 0)
        agent.learn(state, 0, 10.0, nxt, terminal=True)
        self.assertAlmostEqual(agent.qvals(state.key)[0], 0.15 * 10.0)

    def test_terminal_updates_do_not_bootstrap(self):
        agent = core.QAgent(rng=random.Random(0))
        state = core.State(0, 0, 0, 0, 0, "L", 4, 0)
        nxt = core.State(1, 0, 0, 0, 0, "R", 4, 0)
        agent.q[nxt.key] = [100.0, 100.0, 100.0]
        agent.learn(state, 1, 0.0, nxt, terminal=True)
        self.assertEqual(agent.qvals(state.key)[1], 0.0)

    def test_epsilon_decays_to_the_floor(self):
        agent = core.QAgent(core.PRESETS["BOLD"], rng=random.Random(0))
        for _ in range(5000):
            agent.end_episode(0)
        self.assertAlmostEqual(agent.epsilon, core.PRESETS["BOLD"]["eps_min"])
        self.assertEqual(agent.episodes, 5000)

    def test_greedy_action_ignores_epsilon(self):
        agent = core.QAgent(rng=random.Random(0))
        agent.epsilon = 1.0
        state = core.State(0, 0, 0, 0, 0, "L", 4, 0)
        agent.q[state.key] = [0.0, 5.0, 0.0]
        self.assertEqual({agent.act(state, explore=False) for _ in range(50)}, {1})

    def test_end_episode_does_not_touch_disk(self):
        agent = core.QAgent(rng=random.Random(0))
        with tempfile.TemporaryDirectory() as tmp:
            storage = core.Storage(tmp)
            agent.end_episode(3)
            self.assertFalse(os.path.exists(storage.checkpoint_path))

    def test_apply_preset_keeps_learned_values(self):
        agent = core.QAgent(rng=random.Random(0))
        agent.q["a|b"] = [1.0, 2.0, 3.0]
        agent.apply_preset("CAREFUL")
        self.assertEqual(agent.alpha, core.PRESETS["CAREFUL"]["alpha"])
        self.assertEqual(agent.q["a|b"], [1.0, 2.0, 3.0])


class StorageTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.storage = core.Storage(self._tmp.name)

    def test_checkpoint_roundtrip(self):
        self.storage.save_checkpoint({"1|0|0|0|0|L|4|0": [1.0, 2.0, 3.0]}, {"episodes": 7, "preset": "BOLD"})
        q, meta = self.storage.load_checkpoint()
        self.assertEqual(q["1|0|0|0|0|L|4|0"], [1.0, 2.0, 3.0])
        self.assertEqual(meta["episodes"], 7)

    def test_reads_v1_bare_qtable(self):
        self.storage.ensure()
        with open(self.storage.checkpoint_path, "w", encoding="utf-8") as f:
            json.dump({"1|0|0|0|0|L|4|0": [0.5, 0.0, 0.0]}, f)
        q, meta = self.storage.load_checkpoint()
        self.assertEqual(q["1|0|0|0|0|L|4|0"], [0.5, 0.0, 0.0])
        self.assertEqual(meta, {})

    def test_drops_orphaned_turbo_keys(self):
        self.storage.save_checkpoint({"K|137": [1.0, 1.0, 1.0], "1|0|0|0|0|L|4|0": [2.0, 0.0, 0.0]}, {})
        q, _ = self.storage.load_checkpoint()
        self.assertEqual(list(q), ["1|0|0|0|0|L|4|0"])

    def test_corrupt_checkpoint_is_quarantined_not_deleted(self):
        self.storage.ensure()
        with open(self.storage.checkpoint_path, "w", encoding="utf-8") as f:
            f.write("{not json")
        q, meta = self.storage.load_checkpoint()
        self.assertEqual(q, {})
        self.assertEqual(meta, {})
        quarantined = [n for n in os.listdir(self.storage.dir) if ".corrupt-" in n]
        self.assertEqual(len(quarantined), 1)

    def test_save_is_atomic_and_leaves_no_temp_files(self):
        self.storage.save_checkpoint({"a|b": [0.0, 0.0, 0.0]}, {})
        leftovers = [n for n in os.listdir(self.storage.dir) if n.startswith(".serpentos-")]
        self.assertEqual(leftovers, [])

    def test_leaderboard_keeps_top_ten_sorted(self):
        for score in range(15):
            self.storage.add_score(score, "AI(Q)", "NORMAL")
        items = self.storage.load_leaderboard()
        self.assertEqual(len(items), 10)
        self.assertEqual([i["score"] for i in items], sorted([i["score"] for i in items], reverse=True))
        self.assertEqual(items[0]["score"], 14)

    def test_training_log_writes_header_once(self):
        for _ in range(2):
            with self.storage.training_log() as log:
                log.write(1, 2, 3.0, 0.1, "bot")
        with open(self.storage.training_log_path, encoding="utf-8") as f:
            lines = f.read().strip().splitlines()
        self.assertEqual(lines[0], "episode,score,avg50,epsilon,mode")
        self.assertEqual(len(lines), 3)

    def test_training_log_rotates_when_large(self):
        with self.storage.training_log() as log:
            log.write(1, 1, 1.0, 0.1, "bot")
        with self.storage.training_log(max_bytes=1) as log:
            log.write(2, 2, 2.0, 0.1, "bot")
        self.assertTrue(os.path.exists(self.storage.training_log_path + ".1"))

    def test_lock_is_exclusive(self):
        with self.storage.lock():
            other = core.Storage(self._tmp.name)
            with self.assertRaises(core.LockError):
                with other.lock():
                    pass
        with self.storage.lock():  # released on exit
            pass

    def test_stale_lock_is_reclaimed(self):
        self.storage.ensure()
        with open(self.storage.lock_path, "w", encoding="utf-8") as f:
            json.dump({"pid": 2 ** 22, "owner": "ghost"}, f)
        if sys.platform == "win32":
            self.skipTest("PID liveness cannot be probed safely on Windows")
        with self.storage.lock():
            pass
        self.assertFalse(os.path.exists(self.storage.lock_path))

    def test_load_agent_resumes_progress(self):
        agent = core.QAgent(core.PRESETS["BOLD"], rng=random.Random(0))
        for _ in range(10):
            agent.end_episode(1)
        agent.q["1|0|0|0|0|L|4|0"] = [1.0, 0.0, 0.0]
        self.storage.save_checkpoint(agent.q, agent.meta("BOLD"))

        resumed = core.load_agent(self.storage, "BOLD")
        self.assertEqual(resumed.episodes, 10)
        self.assertEqual(resumed.total_food, 10)
        self.assertAlmostEqual(resumed.epsilon, agent.epsilon, places=5)
        self.assertEqual(resumed.q["1|0|0|0|0|L|4|0"], [1.0, 0.0, 0.0])


class RunEpisodeTest(unittest.TestCase):
    def test_on_step_can_abort(self):
        env = core.SnakeEnv(10, 20, rng=random.Random(0))
        agent = core.QAgent(rng=random.Random(0))
        info = core.run_episode(env, agent, on_step=lambda t: t.info.steps < 3)
        self.assertEqual(info.reason, "aborted")
        self.assertEqual(agent.episodes, 1)

    def test_evaluation_does_not_change_the_table(self):
        env = core.SnakeEnv(10, 20, rng=random.Random(0))
        agent = core.QAgent(rng=random.Random(0))
        core.run_episode(env, agent, train=True)
        before = {k: v[:] for k, v in agent.q.items()}
        episodes_before = agent.episodes
        core.run_episode(env, agent, train=False)
        self.assertEqual({k: v for k, v in agent.q.items() if k in before}, before)
        self.assertEqual(agent.episodes, episodes_before)

    def test_learning_beats_a_random_policy(self):
        """End-to-end signal check: shaped Q-learning should out-eat chance."""
        rng = random.Random(11)
        env = core.SnakeEnv(12, 24, rng=rng)
        agent = core.QAgent(core.PRESETS["BOLD"], rng=random.Random(11))
        for _ in range(1500):
            core.run_episode(env, agent, train=True)
        agent.epsilon = 0.0
        trained = sum(core.run_episode(env, agent, train=False).score for _ in range(50)) / 50

        random_agent = core.QAgent(rng=random.Random(5))
        random_agent.epsilon = 1.0
        baseline = sum(core.run_episode(env, random_agent, train=False).score for _ in range(50)) / 50

        self.assertGreater(trained, baseline)


class PolicyExchangeTest(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.path = os.path.join(self._tmp.name, "policy.json")

    def trained_table(self, seed=13, episodes=300):
        env = core.SnakeEnv(12, 24, rng=random.Random(seed))
        agent = core.QAgent(core.PRESETS["BOLD"], rng=random.Random(seed))
        for _ in range(episodes):
            core.run_episode(env, agent, train=True)
        return agent.q

    def test_fingerprint_is_order_independent(self):
        a = {"1|0|0|0|0|L|4|0": [1.0, 2.0, 3.0], "0|1|0|0|0|R|4|0": [4.0, 5.0, 6.0]}
        b = dict(reversed(list(a.items())))
        self.assertEqual(core.policy_fingerprint(a), core.policy_fingerprint(b))

    def test_fingerprint_changes_with_values(self):
        a = {"1|0|0|0|0|L|4|0": [1.0, 2.0, 3.0]}
        b = {"1|0|0|0|0|L|4|0": [1.0, 2.0, 3.5]}
        self.assertNotEqual(core.policy_fingerprint(a), core.policy_fingerprint(b))

    def test_benchmark_is_reproducible(self):
        q = self.trained_table()
        first = core.run_benchmark(q)
        second = core.run_benchmark(q)
        self.assertEqual(first, second)
        self.assertEqual(first["episodes"], core.BENCHMARK_SPEC["episodes"])

    def test_benchmark_does_not_mutate_the_policy(self):
        q = self.trained_table()
        before = core.policy_fingerprint(q)
        core.run_benchmark(q)
        self.assertEqual(core.policy_fingerprint(q), before)

    def test_benchmark_rewards_a_trained_policy(self):
        self.assertGreater(core.run_benchmark(self.trained_table(episodes=1200))["mean_score"],
                           core.run_benchmark({})["mean_score"])

    def test_policy_roundtrip(self):
        q = self.trained_table(episodes=100)
        core.export_policy(self.path, q, {"episodes": 100}, benchmark={"mean_score": 1.0}, name="demo")
        loaded, payload = core.import_policy(self.path)
        self.assertEqual(loaded, q)
        self.assertEqual(payload["name"], "demo")
        self.assertEqual(payload["fingerprint"], core.policy_fingerprint(q))

    def test_import_rejects_foreign_files(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"format": "something/else", "q": {}}, f)
        with self.assertRaises(ValueError):
            core.import_policy(self.path)

    def test_import_rejects_empty_table(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump({"format": core.POLICY_FORMAT, "q": {}}, f)
        with self.assertRaises(ValueError):
            core.import_policy(self.path)

    def test_import_drops_malformed_rows(self):
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(
                {
                    "format": core.POLICY_FORMAT,
                    "q": {
                        "1|0|0|0|0|L|4|0": [1.0, 2.0, 3.0],
                        "bad-row": [1.0],
                        "injected": "rm -rf /",
                        "K|9": [1.0, 2.0, 3.0],
                    },
                },
                f,
            )
        q, _ = core.import_policy(self.path)
        self.assertEqual(list(q), ["1|0|0|0|0|L|4|0"])

    def test_imported_policy_is_data_not_code(self):
        """A downloaded policy must never be able to smuggle in behaviour."""
        q = {"1|0|0|0|0|L|4|0": [1.0, 2.0, 3.0]}
        core.export_policy(self.path, q)
        with open(self.path, encoding="utf-8") as f:
            payload = json.load(f)
        for value in payload["q"].values():
            self.assertTrue(all(isinstance(v, float) for v in value))


if __name__ == "__main__":
    unittest.main()
