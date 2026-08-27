"""Headless core for SerpentOS.

Game rules, the tabular Q-learning agent and on-disk persistence live here.
Nothing in this module imports curses, so the terminal UI and the autonomous
bot share one implementation of the environment and — critically — one state
encoding, which is what makes learning transfer between them.
"""

from __future__ import annotations

import contextlib
import json
import logging
import os
import random
import sys
import tempfile
from datetime import datetime, timezone
from typing import Callable, Dict, Iterator, List, NamedTuple, Optional, Sequence

log = logging.getLogger("serpentos")

CHECKPOINT_VERSION = 2
DEFAULT_DATA_DIR = os.path.join(os.path.expanduser("~"), ".serpentos")

# Training logs are appended to forever by a daemonised bot, so they rotate.
TRAINING_LOG_MAX_BYTES = 5 * 1024 * 1024


# =========================
# CONFIG
# =========================
DIFFICULTY = {
    "easy":   {"speed": 0.13, "name": "EASY"},
    "normal": {"speed": 0.10, "name": "NORMAL"},
    "hard":   {"speed": 0.07, "name": "HARD"},
}

PRESETS = {
    "DEFAULT": {
        "alpha": 0.15, "gamma": 0.95, "epsilon": 0.25,
        "eps_min": 0.05, "eps_decay": 0.995, "use_shaping": True,
        "desc": "Balanced — good starting point",
    },
    "BOLD": {
        "alpha": 0.30, "gamma": 0.90, "epsilon": 0.40,
        "eps_min": 0.05, "eps_decay": 0.990, "use_shaping": True,
        "desc": "High alpha/eps — learns fast, less stable",
    },
    "CAREFUL": {
        "alpha": 0.08, "gamma": 0.97, "epsilon": 0.20,
        "eps_min": 0.02, "eps_decay": 0.998, "use_shaping": True,
        "desc": "Low alpha/eps — slow but steady",
    },
    "SPARSE": {
        "alpha": 0.15, "gamma": 0.95, "epsilon": 0.25,
        "eps_min": 0.05, "eps_decay": 0.995, "use_shaping": False,
        "desc": "No distance shaping — sparse reward only",
    },
}
PRESET_NAMES = list(PRESETS.keys())


# =========================
# GEOMETRY
# =========================
DIRS = ("U", "R", "D", "L")
VEC = {"U": (-1, 0), "R": (0, 1), "D": (1, 0), "L": (0, -1)}
OPPOSITE = {"U": "D", "D": "U", "L": "R", "R": "L"}

ACTION_STRAIGHT, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2
N_ACTIONS = 3
_ZERO_ROW = (0.0,) * N_ACTIONS  # shared, immutable: never hand this out for writing


def turn_left(direction: str) -> str:
    return DIRS[(DIRS.index(direction) - 1) % 4]


def turn_right(direction: str) -> str:
    return DIRS[(DIRS.index(direction) + 1) % 4]


def rel_to_abs(direction: str, action: int) -> str:
    if action == ACTION_LEFT:
        return turn_left(direction)
    if action == ACTION_RIGHT:
        return turn_right(direction)
    return direction


def sign(v: int) -> int:
    return -1 if v < 0 else (1 if v > 0 else 0)


# =========================
# STATE
# =========================
class State(NamedTuple):
    """Agent-visible observation.

    ``key`` is wire-compatible with the v1 on-disk Q-table so existing
    ``~/.serpentos/qtable.json`` files keep their learned values.
    """

    dx: int
    dy: int
    danger_ahead: int
    danger_left: int
    danger_right: int
    direction: str
    wall_dist: int
    length_bucket: int

    @property
    def key(self) -> str:
        return (
            f"{self.dx}|{self.dy}|{self.danger_ahead}|{self.danger_left}"
            f"|{self.danger_right}|{self.direction}|{self.wall_dist}|{self.length_bucket}"
        )


class StepInfo(NamedTuple):
    score: int
    steps: int
    done: bool
    truncated: bool
    reason: str


class Transition(NamedTuple):
    state: State
    action: int
    reward: float
    next_state: State
    info: StepInfo


# =========================
# ENVIRONMENT
# =========================
class SnakeEnv:
    """Snake on a ``rows x cols`` grid with 0-indexed interior coordinates.

    Walls are outside the grid; the renderer is responsible for offsetting
    coordinates into whatever window it draws in.
    """

    MIN_ROWS = 6
    MIN_COLS = 10

    def __init__(
        self,
        rows: int,
        cols: int,
        *,
        shaping: bool = True,
        max_steps: int = 1800,
        rng: Optional[random.Random] = None,
        step_penalty: float = -0.02,
        food_reward: float = 10.0,
        death_penalty: float = -10.0,
        closer_reward: float = 0.05,
        farther_penalty: float = -0.03,
    ) -> None:
        if rows < self.MIN_ROWS or cols < self.MIN_COLS:
            raise ValueError(
                f"grid too small: need at least {self.MIN_ROWS}x{self.MIN_COLS}, got {rows}x{cols}"
            )
        self.rows = rows
        self.cols = cols
        self.shaping = shaping
        self.max_steps = max_steps
        self.rng = rng or random.Random()
        self.step_penalty = step_penalty
        self.food_reward = food_reward
        self.death_penalty = death_penalty
        self.closer_reward = closer_reward
        self.farther_penalty = farther_penalty

        self.snake: List[tuple] = []
        self._occupied: set = set()
        self.food: Optional[tuple] = None
        self.direction = "L"
        self.score = 0
        self.steps = 0
        self.done = True
        self._prev_dist = 0
        self.reset()

    # -- lifecycle ---------------------------------------------------
    def reset(self) -> State:
        row = self.rows // 2
        col = self.cols // 2
        length = 4 if col + 4 <= self.cols else 2
        self.snake = [(row, col + i) for i in range(length)]
        self._occupied = set(self.snake)
        self.direction = "L"
        self.score = 0
        self.steps = 0
        self.done = False
        self.food = self._spawn_food()
        self._prev_dist = self._food_distance()
        return self.state()

    def _spawn_food(self) -> Optional[tuple]:
        remaining = self.rows * self.cols - len(self._occupied)
        if remaining <= 0:
            return None
        for _ in range(30):
            cell = (self.rng.randrange(self.rows), self.rng.randrange(self.cols))
            if cell not in self._occupied:
                return cell
        # Dense board: fall back to an exact draw so food never lands on the snake.
        free = [
            (y, x)
            for y in range(self.rows)
            for x in range(self.cols)
            if (y, x) not in self._occupied
        ]
        return self.rng.choice(free) if free else None

    def _food_distance(self) -> int:
        if self.food is None:
            return 0
        hy, hx = self.snake[0]
        return abs(self.food[0] - hy) + abs(self.food[1] - hx)

    # -- observation -------------------------------------------------
    def in_bounds(self, y: int, x: int) -> bool:
        return 0 <= y < self.rows and 0 <= x < self.cols

    def _blocked(self, y: int, x: int) -> int:
        if not self.in_bounds(y, x):
            return 1
        return 1 if (y, x) in self._occupied else 0

    def state(self) -> State:
        hy, hx = self.snake[0]
        fy, fx = self.food if self.food is not None else (hy, hx)

        ay, ax = VEC[self.direction]
        ly, lx = VEC[turn_left(self.direction)]
        ry, rx = VEC[turn_right(self.direction)]

        wall_dist = 0
        y, x = hy, hx
        for _ in range(4):
            y += ay
            x += ax
            if not self.in_bounds(y, x):
                break
            wall_dist += 1

        length = len(self.snake)
        bucket = 0 if length < 6 else (1 if length < 16 else 2)

        return State(
            dx=sign(fx - hx),
            dy=sign(fy - hy),
            danger_ahead=self._blocked(hy + ay, hx + ax),
            danger_left=self._blocked(hy + ly, hx + lx),
            danger_right=self._blocked(hy + ry, hx + rx),
            direction=self.direction,
            wall_dist=wall_dist,
            length_bucket=bucket,
        )

    # -- transition --------------------------------------------------
    def step(self, action: int):
        """Advance using a relative action (0 straight, 1 left, 2 right)."""
        return self.step_dir(rel_to_abs(self.direction, action))

    def step_dir(self, direction: str):
        """Advance using an absolute direction. Reversals are ignored."""
        if self.done:
            raise RuntimeError("step() called on a finished episode; call reset() first")
        if direction not in VEC:
            raise ValueError(f"unknown direction: {direction!r}")
        if len(self.snake) > 1 and direction == OPPOSITE[self.direction]:
            direction = self.direction

        self.direction = direction
        self.steps += 1

        dy, dx = VEC[direction]
        hy, hx = self.snake[0]
        ny, nx = hy + dy, hx + dx
        tail = self.snake[-1]

        if not self.in_bounds(ny, nx):
            return self._terminate(self.death_penalty, "wall")
        # The tail vacates this tick unless we grow into it, which only
        # happens when food sits on the tail cell.
        if (ny, nx) in self._occupied and ((ny, nx) != tail or (ny, nx) == self.food):
            return self._terminate(self.death_penalty, "body")

        head = (ny, nx)
        self.snake.insert(0, head)
        self._occupied.add(head)

        reward = self.step_penalty
        ate = head == self.food
        if ate:
            self.score += 1
            reward = self.food_reward
            self.food = self._spawn_food()
            if self.food is None:
                return self._terminate(reward, "won")
        else:
            self.snake.pop()
            self._occupied.discard(tail)

        dist = self._food_distance()
        if self.shaping and not ate:
            # Skip shaping on the eating step: the food teleports, so the
            # distance delta across that step carries no signal.
            if dist < self._prev_dist:
                reward += self.closer_reward
            elif dist > self._prev_dist:
                reward += self.farther_penalty
        self._prev_dist = dist

        if self.steps >= self.max_steps:
            self.done = True
            return self.state(), reward, True, StepInfo(self.score, self.steps, True, True, "truncated")

        return self.state(), reward, False, StepInfo(self.score, self.steps, False, False, "")

    def _terminate(self, reward: float, reason: str):
        self.done = True
        return self.state(), reward, True, StepInfo(self.score, self.steps, True, False, reason)


# =========================
# AGENT
# =========================
class QAgent:
    """Tabular Q-learning over the three relative actions."""

    def __init__(
        self,
        hparams: Optional[dict] = None,
        q: Optional[Dict[str, List[float]]] = None,
        *,
        rng: Optional[random.Random] = None,
        episodes: int = 0,
        total_food: int = 0,
        epsilon: Optional[float] = None,
    ) -> None:
        p = hparams or PRESETS["DEFAULT"]
        self.q: Dict[str, List[float]] = q if q is not None else {}
        self.rng = rng or random.Random()
        self.alpha = p["alpha"]
        self.gamma = p["gamma"]
        self.eps_min = p["eps_min"]
        self.eps_decay = p["eps_decay"]
        self.epsilon = p["epsilon"] if epsilon is None else epsilon

        self.episodes = episodes
        self.total_food = total_food

        self.last_action: Optional[int] = None
        self.last_reward = 0.0
        self.last_qvals = [0.0, 0.0, 0.0]

    def apply_preset(self, name: str, reset_epsilon: bool = True) -> None:
        p = PRESETS[name]
        self.alpha = p["alpha"]
        self.gamma = p["gamma"]
        self.eps_min = p["eps_min"]
        self.eps_decay = p["eps_decay"]
        if reset_epsilon:
            self.epsilon = p["epsilon"]

    def qvals(self, key: str) -> List[float]:
        """Row for ``key``, creating it. Only for values we are about to write."""
        row = self.q.get(key)
        if row is None:
            row = [0.0] * N_ACTIONS
            self.q[key] = row
        return row

    def peek(self, key: str):
        """Read-only row lookup: never grows the table."""
        return self.q.get(key, _ZERO_ROW)

    def act(self, state: State, explore: bool = True) -> int:
        qvals = self.peek(state.key)
        self.last_qvals = list(qvals)
        if explore and self.rng.random() < self.epsilon:
            action = self.rng.randrange(N_ACTIONS)
        else:
            best = max(qvals)
            action = self.rng.choice([i for i, v in enumerate(qvals) if v == best])
        self.last_action = action
        return action

    def learn(self, state: State, action: int, reward: float, next_state: State, terminal: bool) -> None:
        qvals = self.qvals(state.key)
        target = reward
        if not terminal:
            target += self.gamma * max(self.peek(next_state.key))
        qvals[action] = (1 - self.alpha) * qvals[action] + self.alpha * target
        self.last_reward = reward

    def end_episode(self, food_eaten: int) -> None:
        """Close out an episode. Deliberately does no disk I/O — the caller
        decides how often to checkpoint."""
        self.episodes += 1
        self.total_food += food_eaten
        if self.epsilon > self.eps_min:
            self.epsilon = max(self.eps_min, self.epsilon * self.eps_decay)
        self.last_action = None
        self.last_reward = 0.0
        self.last_qvals = [0.0, 0.0, 0.0]

    def reset_knowledge(self) -> None:
        self.q = {}
        self.episodes = 0
        self.total_food = 0

    def meta(self, preset: Optional[str] = None) -> dict:
        return {
            "episodes": self.episodes,
            "total_food": self.total_food,
            "epsilon": round(self.epsilon, 6),
            "alpha": self.alpha,
            "gamma": self.gamma,
            "states": len(self.q),
            "preset": preset,
            "updated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        }


# =========================
# EPISODE RUNNER
# =========================
def run_episode(
    env: SnakeEnv,
    agent: QAgent,
    *,
    train: bool = True,
    on_step: Optional[Callable[[Transition], bool]] = None,
) -> StepInfo:
    """Run one episode. ``on_step`` may return False to abort early.

    Truncated episodes bootstrap from the next state instead of being treated
    as terminal, which keeps the value estimates unbiased for long runs.
    """
    state = env.reset()
    while True:
        action = agent.act(state, explore=train)
        next_state, reward, done, info = env.step(action)
        if train:
            agent.learn(state, action, reward, next_state, done and not info.truncated)
        if on_step is not None and on_step(Transition(state, action, reward, next_state, info)) is False:
            if train:
                agent.end_episode(env.score)
            return info._replace(done=True, reason="aborted")
        state = next_state
        if done:
            if train:
                agent.end_episode(env.score)
            return info


# =========================
# PERSISTENCE
# =========================
class LockError(RuntimeError):
    """Raised when another SerpentOS process owns the data directory."""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    if sys.platform == "win32":
        # os.kill(pid, 0) terminates the target on Windows; never probe there.
        return True
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return True
    return True


def _atomic_write_json(path: str, payload) -> None:
    directory = os.path.dirname(path) or "."
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory, prefix=".serpentos-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(payload, f, separators=(",", ":"))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


class Storage:
    """Everything SerpentOS keeps on disk, in one place."""

    def __init__(self, directory: Optional[str] = None) -> None:
        self.dir = os.path.abspath(os.path.expanduser(directory or DEFAULT_DATA_DIR))
        self.checkpoint_path = os.path.join(self.dir, "qtable.json")
        self.leaderboard_path = os.path.join(self.dir, "leaderboard.json")
        self.training_log_path = os.path.join(self.dir, "training_log.csv")
        self.lock_path = os.path.join(self.dir, "serpentos.lock")

    def ensure(self) -> None:
        os.makedirs(self.dir, exist_ok=True)

    # -- checkpoint --------------------------------------------------
    def load_checkpoint(self):
        """Return ``(qtable, meta)``. Handles v1 (bare dict) and v2 files."""
        if not os.path.exists(self.checkpoint_path):
            return {}, {}
        try:
            with open(self.checkpoint_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            quarantine = self._quarantine(self.checkpoint_path)
            log.warning("unreadable checkpoint (%s); moved to %s and starting fresh", exc, quarantine)
            return {}, {}

        if isinstance(data, dict) and "q" in data and isinstance(data.get("q"), dict):
            raw, meta = data["q"], data.get("meta", {}) or {}
        elif isinstance(data, dict):
            raw, meta = data, {}  # v1: bare {state_key: [q0, q1, q2]}
        else:
            log.warning("checkpoint has unexpected shape (%s); starting fresh", type(data).__name__)
            return {}, {}

        q = {}
        dropped = 0
        for key, row in raw.items():
            # "K|<int>" keys came from the old turbo encoder, which wrote into a
            # keyspace no other mode could ever read. They are dead weight.
            if not isinstance(key, str) or key.startswith("K|"):
                dropped += 1
                continue
            if isinstance(row, list) and len(row) == N_ACTIONS:
                try:
                    q[key] = [float(v) for v in row]
                except (TypeError, ValueError):
                    dropped += 1
            else:
                dropped += 1
        if dropped:
            log.info("dropped %d unusable Q-table entries during load", dropped)
        return q, meta

    def save_checkpoint(self, q: Dict[str, List[float]], meta: Optional[dict] = None) -> None:
        self.ensure()
        _atomic_write_json(
            self.checkpoint_path,
            {"version": CHECKPOINT_VERSION, "meta": meta or {}, "q": q},
        )

    def _quarantine(self, path: str) -> str:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        target = f"{path}.corrupt-{stamp}"
        with contextlib.suppress(OSError):
            os.replace(path, target)
        return target

    # -- leaderboard -------------------------------------------------
    def load_leaderboard(self) -> List[dict]:
        if not os.path.exists(self.leaderboard_path):
            return []
        try:
            with open(self.leaderboard_path, "r", encoding="utf-8") as f:
                data = json.load(f)
        except (json.JSONDecodeError, UnicodeDecodeError, OSError) as exc:
            log.warning("unreadable leaderboard (%s); starting a new one", exc)
            return []
        return data if isinstance(data, list) else []

    def add_score(self, score: int, mode: str, difficulty: str, limit: int = 10) -> List[dict]:
        items = self.load_leaderboard()
        items.append(
            {
                "score": int(score),
                "mode": mode,
                "diff": difficulty,
                "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            }
        )
        items.sort(key=lambda item: item.get("score", 0), reverse=True)
        items = items[:limit]
        self.ensure()
        _atomic_write_json(self.leaderboard_path, items)
        return items

    # -- training log ------------------------------------------------
    @contextlib.contextmanager
    def training_log(self, max_bytes: int = TRAINING_LOG_MAX_BYTES) -> Iterator["TrainingLog"]:
        self.ensure()
        if max_bytes > 0 and os.path.exists(self.training_log_path):
            with contextlib.suppress(OSError):
                if os.path.getsize(self.training_log_path) >= max_bytes:
                    os.replace(self.training_log_path, self.training_log_path + ".1")
        is_new = not os.path.exists(self.training_log_path)
        handle = open(self.training_log_path, "a", encoding="utf-8", newline="")
        try:
            if is_new:
                handle.write("episode,score,avg50,epsilon,mode\n")
                handle.flush()
            yield TrainingLog(handle)
        finally:
            handle.close()

    # -- lock --------------------------------------------------------
    @contextlib.contextmanager
    def lock(self, owner: str = "serpentos"):
        """Guard the data directory so two writers cannot clobber the Q-table."""
        self.ensure()
        acquired = False
        try:
            fd = self._try_lock(owner)
            if fd is None:
                holder = self._lock_holder()
                raise LockError(
                    f"{self.dir} is in use by {holder}. Stop it, or pass a different data directory."
                )
            acquired = True
            os.close(fd)
            yield self
        finally:
            if acquired:
                with contextlib.suppress(OSError):
                    os.unlink(self.lock_path)

    def _try_lock(self, owner: str):
        for _ in range(2):
            try:
                fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644)
            except FileExistsError:
                info = self._read_lock()
                if info and _pid_alive(int(info.get("pid", -1))):
                    return None
                log.info("removing stale lock at %s", self.lock_path)
                with contextlib.suppress(OSError):
                    os.unlink(self.lock_path)
                continue
            os.write(
                fd,
                json.dumps({"pid": os.getpid(), "owner": owner, "since": datetime.now(timezone.utc).isoformat()}).encode(),
            )
            return fd
        return None

    def _read_lock(self) -> Optional[dict]:
        try:
            with open(self.lock_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return data if isinstance(data, dict) else None
        except (OSError, json.JSONDecodeError):
            return None

    def _lock_holder(self) -> str:
        info = self._read_lock()
        if not info:
            return "another process"
        return f"{info.get('owner', 'a process')} (pid {info.get('pid', '?')})"


class TrainingLog:
    """CSV sink for per-episode results."""

    def __init__(self, handle) -> None:
        self._handle = handle

    def write(self, episode: int, score: int, avg: float, epsilon: float, mode: str) -> None:
        self._handle.write(f"{episode},{score},{avg:.2f},{epsilon:.4f},{mode}\n")

    def flush(self) -> None:
        self._handle.flush()


def load_agent(storage: Storage, preset: str = "DEFAULT", *, rng: Optional[random.Random] = None,
               resume: bool = True) -> QAgent:
    """Build an agent, resuming episode count and exploration rate when present."""
    q, meta = storage.load_checkpoint()
    hparams = PRESETS[preset]
    epsilon = None
    episodes = 0
    total_food = 0
    if resume and meta:
        if meta.get("preset") == preset and isinstance(meta.get("epsilon"), (int, float)):
            epsilon = float(meta["epsilon"])
        episodes = int(meta.get("episodes", 0) or 0)
        total_food = int(meta.get("total_food", 0) or 0)
    return QAgent(hparams, q, rng=rng, episodes=episodes, total_food=total_food, epsilon=epsilon)


def moving_average(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


# =========================
# PORTABLE POLICIES + BENCHMARK
# =========================
POLICY_FORMAT = "serpentos.policy/1"

# Frozen so that a score produced on one machine means the same thing on
# another. Changing any of it requires a new BENCHMARK_VERSION.
BENCHMARK_VERSION = 1
BENCHMARK_SPEC = {
    "version": BENCHMARK_VERSION,
    "rows": 22,
    "cols": 78,
    "episodes": 100,
    "max_steps": 1800,
    "seed_base": 1000,
}


def policy_fingerprint(q: Dict[str, List[float]]) -> str:
    """Stable content hash of a Q-table, for citing exactly which brain ran."""
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(q):
        digest.update(key.encode("utf-8"))
        digest.update(b":")
        digest.update(",".join(f"{v:.6f}" for v in q[key]).encode("utf-8"))
        digest.update(b";")
    return digest.hexdigest()


def run_benchmark(q: Dict[str, List[float]], spec: Optional[dict] = None) -> dict:
    """Score a Q-table on the frozen benchmark.

    Every episode gets its own seeded RNG, so the result depends only on the
    Q-table — not on episode order, wall-clock time or machine.
    """
    spec = spec or BENCHMARK_SPEC
    agent = QAgent(q=q)
    agent.epsilon = 0.0

    scores = []
    steps = []
    for i in range(spec["episodes"]):
        seed = spec["seed_base"] + i
        env = SnakeEnv(
            spec["rows"], spec["cols"],
            shaping=False, max_steps=spec["max_steps"], rng=random.Random(seed),
        )
        agent.rng = random.Random(seed)
        info = run_episode(env, agent, train=False)
        scores.append(info.score)
        steps.append(info.steps)

    scores_sorted = sorted(scores)
    mid = len(scores_sorted) // 2
    return {
        "benchmark_version": spec["version"],
        "episodes": spec["episodes"],
        "mean_score": round(sum(scores) / len(scores), 4),
        "median_score": scores_sorted[mid],
        "best_score": max(scores),
        "worst_score": min(scores),
        "mean_steps": round(sum(steps) / len(steps), 2),
        "states": len(q),
        "fingerprint": policy_fingerprint(q),
    }


def export_policy(path: str, q: Dict[str, List[float]], meta: Optional[dict] = None,
                  benchmark: Optional[dict] = None, name: Optional[str] = None) -> dict:
    """Write a shareable policy file.

    The artifact is pure data — a state->values table plus provenance. Importing
    one never executes anything from the author, which is what makes swapping
    policies with strangers safe.
    """
    payload = {
        "format": POLICY_FORMAT,
        "name": name or "unnamed",
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "meta": meta or {},
        "benchmark": benchmark,
        "fingerprint": policy_fingerprint(q),
        "q": q,
    }
    _atomic_write_json(path, payload)
    return payload


def import_policy(path: str):
    """Read a policy file and return ``(qtable, payload)``.

    Raises ValueError when the file is not a well-formed policy, so a bad
    download fails loudly instead of silently training from garbage.
    """
    with open(path, "r", encoding="utf-8") as f:
        payload = json.load(f)
    if not isinstance(payload, dict) or payload.get("format") != POLICY_FORMAT:
        raise ValueError(f"{path} is not a {POLICY_FORMAT} file")
    raw = payload.get("q")
    if not isinstance(raw, dict):
        raise ValueError(f"{path} has no Q-table")

    q = {}
    for key, row in raw.items():
        if (
            isinstance(key, str)
            and not key.startswith("K|")
            and isinstance(row, list)
            and len(row) == N_ACTIONS
            and all(isinstance(v, (int, float)) for v in row)
        ):
            q[key] = [float(v) for v in row]
    if not q:
        raise ValueError(f"{path} contains no usable entries")

    claimed = payload.get("fingerprint")
    actual = policy_fingerprint(q)
    if claimed and claimed != actual:
        log.warning("policy fingerprint mismatch: file claims %s, content hashes to %s", claimed, actual)
    return q, payload
