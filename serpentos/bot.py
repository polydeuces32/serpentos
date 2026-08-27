"""Autonomous SerpentOS agent.

Runs the same environment and Q-learning agent as the terminal UI, but with no
curses, no keyboard and no screen — so it can be supervised by systemd, a
container, or a CI job. It trains until told to stop, checkpoints atomically as
it goes, and shuts down cleanly on SIGINT/SIGTERM.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import random
import signal
import sys
import time
from collections import deque
from typing import Callable, Optional

try:
    from . import core
except ImportError:  # executed as a plain script: python serpentos/bot.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import core  # type: ignore[no-redef]

log = logging.getLogger("serpentos.bot")

DEFAULT_ROWS = 22
DEFAULT_COLS = 78
EXIT_OK = 0
EXIT_ERROR = 1
EXIT_LOCKED = 2
EXIT_INTERRUPTED = 130


class BotRunner:
    """Drives episodes and owns the checkpoint / metrics cadence."""

    def __init__(
        self,
        env: core.SnakeEnv,
        agent: core.QAgent,
        storage: core.Storage,
        *,
        preset: str = "DEFAULT",
        train: bool = True,
        checkpoint_every: int = 100,
        log_every: int = 100,
        metrics_path: Optional[str] = None,
        mode_label: str = "bot",
        should_stop: Optional[Callable[[], bool]] = None,
    ) -> None:
        self.env = env
        self.agent = agent
        self.storage = storage
        self.preset = preset
        self.train = train
        self.checkpoint_every = max(0, checkpoint_every)
        self.log_every = max(0, log_every)
        self.metrics_path = metrics_path
        self.mode_label = mode_label
        self.should_stop = should_stop or (lambda: False)

        self.window: deque = deque(maxlen=50)
        self.best = 0
        self.episodes_run = 0
        self.total_score = 0
        self.total_steps = 0
        self.deaths = {"wall": 0, "body": 0, "truncated": 0, "won": 0}

    # -- main loop ---------------------------------------------------
    def run(self, episodes: Optional[int] = None, max_seconds: Optional[float] = None) -> dict:
        started = time.monotonic()
        stopped_by = "completed"
        metrics = self._open_metrics()
        try:
            with self.storage.training_log() as training_log:
                while True:
                    if episodes is not None and self.episodes_run >= episodes:
                        break
                    if max_seconds is not None and time.monotonic() - started >= max_seconds:
                        stopped_by = "time-budget"
                        break
                    if self.should_stop():
                        stopped_by = "signal"
                        break

                    info = core.run_episode(self.env, self.agent, train=self.train)
                    self._record(info, training_log, metrics)

                    if (
                        self.train
                        and self.checkpoint_every
                        and self.episodes_run % self.checkpoint_every == 0
                    ):
                        self.checkpoint()
                training_log.flush()
        finally:
            if metrics is not None:
                metrics.close()

        if self.train:
            self.checkpoint()

        elapsed = time.monotonic() - started
        summary = self.summary(elapsed, stopped_by)
        log.info(
            "%s: %d episodes in %.1fs (%.0f eps/s) avg=%.2f best=%d states=%d",
            "training" if self.train else "evaluation",
            self.episodes_run,
            elapsed,
            self.episodes_run / elapsed if elapsed > 0 else 0.0,
            summary["avg_score"],
            self.best,
            len(self.agent.q),
        )
        return summary

    def _record(self, info: core.StepInfo, training_log: core.TrainingLog, metrics) -> None:
        self.episodes_run += 1
        self.total_score += info.score
        self.total_steps += info.steps
        self.window.append(info.score)
        self.best = max(self.best, info.score)
        reason = info.reason if info.reason in self.deaths else "wall"
        self.deaths[reason] = self.deaths.get(reason, 0) + 1

        avg = core.moving_average(self.window)
        training_log.write(self.agent.episodes, info.score, avg, self.agent.epsilon, self.mode_label)

        if metrics is not None:
            metrics.write(
                json.dumps(
                    {
                        "ts": time.time(),
                        "episode": self.agent.episodes,
                        "score": info.score,
                        "steps": info.steps,
                        "avg50": round(avg, 3),
                        "epsilon": round(self.agent.epsilon, 5),
                        "reason": info.reason or "alive",
                        "states": len(self.agent.q),
                        "mode": self.mode_label,
                    }
                )
                + "\n"
            )
            metrics.flush()

        if self.log_every and self.episodes_run % self.log_every == 0:
            training_log.flush()
            log.info(
                "episode %d | score %d | avg50 %.2f | best %d | eps %.3f | states %d",
                self.agent.episodes,
                info.score,
                avg,
                self.best,
                self.agent.epsilon,
                len(self.agent.q),
            )

    def checkpoint(self) -> None:
        self.storage.save_checkpoint(self.agent.q, self.agent.meta(self.preset))

    def summary(self, elapsed: float, stopped_by: str) -> dict:
        return {
            "mode": self.mode_label,
            "preset": self.preset,
            "episodes": self.episodes_run,
            "lifetime_episodes": self.agent.episodes,
            "avg_score": round(self.total_score / self.episodes_run, 3) if self.episodes_run else 0.0,
            "avg50": round(core.moving_average(self.window), 3),
            "best_score": self.best,
            "total_steps": self.total_steps,
            "epsilon": round(self.agent.epsilon, 5),
            "states": len(self.agent.q),
            "endings": dict(self.deaths),
            "elapsed_seconds": round(elapsed, 3),
            "episodes_per_second": round(self.episodes_run / elapsed, 2) if elapsed > 0 else 0.0,
            "stopped_by": stopped_by,
            "data_dir": self.storage.dir,
        }

    def _open_metrics(self):
        if not self.metrics_path:
            return None
        directory = os.path.dirname(os.path.abspath(self.metrics_path))
        os.makedirs(directory, exist_ok=True)
        return open(self.metrics_path, "a", encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="serpentos bot",
        description="Run the SerpentOS Q-learning agent headlessly.",
    )
    duration = parser.add_mutually_exclusive_group()
    duration.add_argument("--episodes", type=int, default=1000, help="episodes to run (default: 1000)")
    duration.add_argument("--forever", action="store_true", help="run until stopped or --max-seconds elapses")
    parser.add_argument("--max-seconds", type=float, default=None, help="wall-clock budget")
    parser.add_argument("--eval", type=int, metavar="N", default=None,
                        help="run N greedy episodes without learning or checkpointing")
    parser.add_argument("--preset", choices=core.PRESET_NAMES, default="DEFAULT")
    parser.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    parser.add_argument("--cols", type=int, default=DEFAULT_COLS)
    parser.add_argument("--max-steps", type=int, default=1800, help="step cap per episode")
    parser.add_argument("--shaping", dest="shaping", action="store_true", default=None,
                        help="force distance shaping on")
    parser.add_argument("--no-shaping", dest="shaping", action="store_false",
                        help="force distance shaping off")
    parser.add_argument("--seed", type=int, default=None, help="seed for reproducible runs")
    parser.add_argument("--data-dir", default=None, help=f"state directory (default: {core.DEFAULT_DATA_DIR})")
    parser.add_argument("--fresh", action="store_true", help="ignore any saved Q-table and start empty")
    parser.add_argument("--checkpoint-every", type=int, default=100, help="episodes between saves")
    parser.add_argument("--log-every", type=int, default=100, help="episodes between progress lines")
    parser.add_argument("--metrics", default=None, help="append per-episode JSON lines to this file")
    parser.add_argument("--json", action="store_true", help="print the run summary as JSON on stdout")
    parser.add_argument("--quiet", action="store_true", help="only log warnings and errors")

    exchange = parser.add_argument_group("policy exchange")
    exchange.add_argument("--bench", action="store_true",
                          help="score the policy on the frozen benchmark and exit")
    exchange.add_argument("--import-policy", metavar="PATH", default=None,
                          help="load a shared policy file instead of the saved checkpoint")
    exchange.add_argument("--export-policy", metavar="PATH", default=None,
                          help="write the resulting policy to a shareable file")
    exchange.add_argument("--name", default=None, help="name recorded in the exported policy")
    return parser


def _install_signal_handlers(state: dict) -> None:
    def handler(signum, _frame):
        if state["stop"]:
            log.warning("second signal received; exiting now")
            raise SystemExit(EXIT_INTERRUPTED)
        state["stop"] = True
        state["signal"] = signum
        log.info("signal %s received; finishing episode and checkpointing", signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        try:
            signal.signal(sig, handler)
        except (ValueError, OSError, AttributeError):
            # Not the main thread, or the platform lacks the signal.
            log.debug("could not install handler for %s", sig)


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)

    logging.basicConfig(
        level=logging.WARNING if args.quiet else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
        stream=sys.stderr,
    )

    evaluating = args.eval is not None
    if evaluating and args.eval <= 0:
        log.error("--eval needs a positive episode count")
        return EXIT_ERROR

    rng = random.Random(args.seed)
    shaping = core.PRESETS[args.preset]["use_shaping"] if args.shaping is None else args.shaping

    storage = core.Storage(args.data_dir)
    try:
        env = core.SnakeEnv(
            args.rows,
            args.cols,
            shaping=shaping,
            max_steps=args.max_steps,
            rng=rng,
        )
    except ValueError as exc:
        log.error("%s", exc)
        return EXIT_ERROR

    stop_state = {"stop": False, "signal": None}
    _install_signal_handlers(stop_state)

    try:
        with storage.lock(owner="bot"):
            if args.import_policy:
                q, payload = core.import_policy(args.import_policy)
                agent = core.QAgent(core.PRESETS[args.preset], q, rng=rng)
                log.info("imported policy %r with %d states from %s",
                         payload.get("name", "unnamed"), len(q), args.import_policy)
            else:
                agent = core.load_agent(storage, args.preset, rng=rng, resume=not args.fresh)
                if args.fresh:
                    agent.reset_knowledge()
                agent.apply_preset(args.preset, reset_epsilon=args.fresh or not agent.q)
            if evaluating:
                agent.epsilon = 0.0

            if args.bench:
                if not agent.q:
                    log.error("nothing to benchmark: train first, or pass --import-policy")
                    return EXIT_ERROR
                result = core.run_benchmark(agent.q)
                log.info("benchmark v%d: mean %.2f over %d episodes (best %d, %d states)",
                         result["benchmark_version"], result["mean_score"],
                         result["episodes"], result["best_score"], result["states"])
                if args.export_policy:
                    core.export_policy(args.export_policy, agent.q, agent.meta(args.preset),
                                       benchmark=result, name=args.name)
                    log.info("wrote %s", args.export_policy)
                print(json.dumps(result, indent=2))
                return EXIT_OK

            runner = BotRunner(
                env,
                agent,
                storage,
                preset=args.preset,
                train=not evaluating,
                checkpoint_every=args.checkpoint_every,
                log_every=args.log_every,
                metrics_path=args.metrics,
                mode_label="eval" if evaluating else "bot",
                should_stop=lambda: stop_state["stop"],
            )

            log.info(
                "starting %s: preset=%s grid=%dx%d shaping=%s states=%d lifetime_episodes=%d data=%s",
                runner.mode_label,
                args.preset,
                args.rows,
                args.cols,
                "on" if shaping else "off",
                len(agent.q),
                agent.episodes,
                storage.dir,
            )

            episodes = args.eval if evaluating else (None if args.forever else args.episodes)
            summary = runner.run(episodes=episodes, max_seconds=args.max_seconds)

            if args.export_policy:
                core.export_policy(
                    args.export_policy, agent.q, agent.meta(args.preset),
                    benchmark=core.run_benchmark(agent.q) if agent.q else None,
                    name=args.name,
                )
                summary["exported_policy"] = args.export_policy
                log.info("wrote %s", args.export_policy)
    except core.LockError as exc:
        log.error("%s", exc)
        return EXIT_LOCKED
    except ValueError as exc:
        log.error("invalid policy file: %s", exc)
        return EXIT_ERROR
    except OSError as exc:
        log.error("storage failure in %s: %s", storage.dir, exc)
        return EXIT_ERROR

    if args.json:
        print(json.dumps(summary, indent=2))
    return EXIT_OK


if __name__ == "__main__":
    raise SystemExit(main())
