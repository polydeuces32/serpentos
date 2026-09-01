# SerpentOS

**A policy runtime you can embed in a Python project, and a terminal snake game that demonstrates it.**

Two things live in this repository, and it is worth knowing which one you came for.

**[The runtime](#using-serpentos-as-a-decision-engine)** is a small standard-library kernel for decision logic: define a policy, run it under guardrails, record what it decided, and prove later that it still decides the same way. No dependencies, no services, no machine learning required.

**The game** is the reference environment — a complete, honest example of an application with state, actions, a learned policy, persistence and a reproducible benchmark. Play it, train it, or read it as a worked example.

---

## Who this is for

**Engineers with decision logic scattered through their codebase.** Retry strategies, queue prioritisation, routing rules, feature rollout — the ones that are three nested `if` statements today, untested, unlogged and impossible to change safely. The runtime gives them a shape: a policy that proposes, a validator that gates, an audit record that explains, and replay that tells you whether your change would have altered anything. See **[docs/DECISION_ENGINE.md](docs/DECISION_ENGINE.md)**.

**Anyone learning reinforcement learning by watching it happen.** The AI HUD shows the agent's actual state, its three Q-values and the reward for every step, so "exploration versus exploitation" stops being a phrase and becomes something on screen. There is no PyTorch, no GPU, no CUDA and no account — clone it and it runs.

**People who want to tinker with hyperparameters and see the consequence.** Four presets, a shaping toggle, a live score sparkline and a CSV of every episode. Change `alpha`, run 2,000 episodes, compare the curves.

**Terminal-dwellers who want a good-looking snake game.** It plays fine as a game and never leaves the shell.

**Anyone who wants a small, complete codebase to read.** Standard-library Python throughout: an environment, a tabular agent, atomic persistence, a headless runner, a reproducible benchmark, a policy runtime and a test suite. Small enough to read in one sitting, structured enough to be worth reading.

It is **not** a serious RL research tool. Tabular Q-learning over eight state features has a hard ceiling, and [docs/ECOSYSTEM.md](docs/ECOSYSTEM.md) is candid about where that ceiling is and why.

---

## Using SerpentOS as a Decision Engine

Install it, import it, decide something:

```python
from serpentos import ActionValidator, DecisionContext, DecisionEngine
from serpentos.policies import Rule, RulePolicy, when

policy = RulePolicy(
    name="retry-policy",
    version="1.0",
    rules=[
        Rule("fail",  when("attempts", "ge", 3),               name="give-up"),
        Rule("retry", when("status_code", "in", [503, 504]),   name="retry-5xx"),
    ],
    default_action="fail",
)

engine = DecisionEngine(
    policy=policy,
    validator=ActionValidator(allowed_actions={"retry", "fail"}),
)

decision = engine.decide(DecisionContext(values={"attempts": 2, "status_code": 503}))
print(decision.action)  # retry
```

Four concepts, and that is the whole model:

**A context** is what the policy is allowed to see — a plain JSON object you build from whatever your application knows. It is deeply immutable, so handing the same context to five policies cannot let one corrupt it for the others.

**A policy** is a pure function from a context to a decision. It proposes; it never acts. That restriction is what makes everything below it possible — a policy that talks to the network cannot be replayed, compared offline or trusted to behave the same twice.

**A decision** is what the policy proposed, stamped with the policy's name and version and carrying whatever metadata explains the choice. Your application decides whether to execute it. The runtime never does.

**Validation** is the allow-list standing between the proposal and your application. It lives with you, not with the policy, so a policy cannot widen its own permissions. A rejected action raises `DecisionValidationError` rather than being quietly swapped for something safe — silently rewriting a decision is how you get an audit log that does not describe what happened. If you want a substitute, configure a `fallback_policy` explicitly and both attempts are recorded.

**Audit and replay** are the payoff. Attach a sink and every decision becomes a JSON record: who decided, on what evidence, what the guardrails said. Feed those records back through `replay()` and you get a straight answer to "would my change have decided anything differently?"

```python
from serpentos import InMemoryAuditLog, replay_all

audit = InMemoryAuditLog(redact=["authorization"])          # never log the token
engine = DecisionEngine(policy, audit_sink=audit)
...
report = replay_all(revised_policy, audit.records, strict=False)
print(report.matched, report.mismatched)
```

**Comparison** answers the other question: how do two candidate policies actually differ? Not just their action distributions — two retry policies can each retry exactly half the time and never once agree on *which* half — so `compare()` reports disagreement directly, overall and pairwise, with worked examples of the cases where they parted company. It declares no winner. Whether retrying more is better depends on what a retry costs you, and only you know that.

Three policy implementations ship with it. `RulePolicy` is ordered conditions over a closed set of operators, so a rule set loaded from a JSON file is data and cannot execute. `WeightedPolicy` scores every candidate action and takes the highest, and explains each score factor by factor — not "UPS scored 8.6" but "−12.50 on cost, −8.00 for two days in transit, +29.10 on reliability". `QLearningPolicy` is a read-only adapter over the snake agent — proof that a learned policy and a hand-written one can sit behind the same interface.

**Try it without cloning your imagination first:**

```bash
python examples/retry_policy.py
```

Six failed requests, decided, audited, replayed against a tightened retry budget to show exactly which two decisions would change, then compared against a scoring model. No network, no game, about a tenth of a second.

**How Snake relates to all this.** It is the reference environment, not the centre. `serpentos.environments.snake` shows the full integration: turning game state into a context, executing the action the engine returns, reporting the outcome. The most useful thing in it is `survival_policy()` — eight hand-written rules that play the game through the same engine as the trained agent, and average **36.9** food per episode against **0.05** for an untrained Q-table. Nothing about the runtime is shaped around machine learning.

A shared `DecisionEngine` is safe to use from multiple threads, and so are both built-in audit sinks. Your own policy or sink is safe only if you made it so — the table in [docs/DECISION_ENGINE.md](docs/DECISION_ENGINE.md#can-i-share-one-engine-across-threads) is explicit about where the boundary falls.

Full architecture, guarantees, security model and limitations: **[docs/DECISION_ENGINE.md](docs/DECISION_ENGINE.md)**. What counts as public API and what may change: **[docs/COMPATIBILITY.md](docs/COMPATIBILITY.md)**.

---

## Features

- **Policy runtime** — context → policy → decision → outcome, with validation, audit, replay, and comparison that reports where policies disagree
- **Three policy types** — ordered rules, weighted scoring, and the Q-learning adapter, all behind one interface
- **Human mode** — classic snake with arrow keys or WASD
- **AI mode** — tabular Q-learning agent with a live thinking HUD
- **Headless agent** — trains itself with no screen, survives SIGTERM, checkpoints as it goes
- **Fast & Turbo training** — run hundreds or thousands of episodes without rendering
- **Hyperparameter presets** — swap learning profiles mid-session (DEFAULT / BOLD / CAREFUL / SPARSE)
- **Reward shaping toggle** — compare sparse vs. dense reward learning
- **Live sparkline** — ASCII score graph updates during training
- **Reproducible benchmark** — score any agent on a frozen course and get the same number anywhere
- **Portable policies** — export a trained brain to a file and hand it to someone else
- **CSV + JSONL logs** — every episode recorded for offline plotting
- **Persistent Q-table** — the agent remembers what it learned between sessions
- **Leaderboard** — top 10 scores saved locally
- **Difficulty levels** — Easy / Normal / Hard

---

## Quick start

```bash
git clone https://github.com/polydeuces32/serpentos.git
cd serpentos

python3 -m serpentos run                    # play
python3 -m serpentos bot --episodes 5000    # train with no UI
```

Python 3.9 or newer. No packages to install on macOS or Linux.

## Commands

| Command | What it does |
|---------|-------------|
| `serpentos run` | Play in the terminal. The default when no command is given. |
| `serpentos bot` | Train, evaluate or export a policy with no terminal at all. |
| `serpentos bench` | Score the current policy on the frozen benchmark. |
| `serpentos help` | List the commands. |

Every command takes `--help`. Without installing, put `python3 -m ` in front of any of them. The single-file invocation `python3 serpentos/serpentos.py` still works too.

---

## Installation

### macOS

**1. Check if Python is installed**
```bash
python3 --version
```
If you see `Python 3.9` or higher you're good. If not, install it:
- Download from [python.org/downloads](https://www.python.org/downloads/) and run the installer, **or**
- Install via Homebrew: `brew install python3`

**2. Download SerpentOS**

Option A — clone with Git:
```bash
git clone https://github.com/polydeuces32/serpentos.git
cd serpentos
```

Option B — download the ZIP:
1. Click the green **Code** button on this page → **Download ZIP**
2. Unzip it and open Terminal in that folder

**3. Run**
```bash
python3 -m serpentos run
```

---

### Windows

> The game uses Python's `curses` library for terminal rendering. This is not available in standard Windows Command Prompt or PowerShell, so you have two options below.
>
> The **headless agent needs none of this** — `python -m serpentos bot` runs on stock Windows Python.

#### Option A — WSL (recommended, one-time setup)

WSL (Windows Subsystem for Linux) lets you run a real Linux terminal on Windows. It's free and built into Windows 10/11.

**1. Install WSL** — open PowerShell as Administrator and run:
```powershell
wsl --install
```
Restart your PC when prompted. This installs Ubuntu by default.

**2. Open Ubuntu** from the Start menu.

**3. Install Python** (if not already present):
```bash
sudo apt update && sudo apt install python3 -y
```

**4. Download SerpentOS** inside WSL:
```bash
git clone https://github.com/polydeuces32/serpentos.git
cd serpentos
```

Or copy the project folder into WSL from Windows Explorer — paste it into `\\wsl$\Ubuntu\home\<your-username>\`.

**5. Run**
```bash
python3 serpentos/serpentos.py
```

---

#### Option B — windows-curses (no WSL needed)

**1.** Download and install Python from [python.org/downloads](https://www.python.org/downloads/)
- During install, check **"Add Python to PATH"**

**2.** Open Command Prompt and install the curses package:
```cmd
pip install windows-curses
```

**3.** Download SerpentOS — click **Code → Download ZIP** on this page, unzip it.

**4.** In Command Prompt, navigate to the folder and run:
```cmd
python serpentos\serpentos.py
```

> **Tip:** Windows Terminal (available free from the Microsoft Store) gives a much better experience than the default Command Prompt.

---

### Install as a command (optional)

```bash
pip install .
serpentos            # the game
serpentos bot --help # the agent
```

---

## Controls

| Key | Action |
|-----|--------|
| Arrow keys / WASD | Move (human mode) |
| `H` | Toggle AI thinking HUD |
| `Q` | Quit current game |
| `Y` / `N` | Play again / return to menu |

The terminal must be at least **40x14**. Below that SerpentOS shows a resize prompt instead of starting.

---

## Colours

SerpentOS picks a palette from what your terminal reports and never assumes more than it has:

| Terminal | What you get |
|----------|-------------|
| 256-colour (`xterm-256color`, most modern terminals) | Full palette: the snake is a green gradient from a bright `@` head to a darker tail, food is red, the border is dim grey, and HUD labels, values and warnings each get their own colour |
| 8-colour (`xterm`, `linux`) | The same layout in the eight ANSI colours, using bold for the bright variants |
| Monochrome (`vt100`, `TERM=dumb`) | No colour at all — bold, dim and reverse video keep the screen readable |

Turn it off with `serpentos run --no-color`, or by setting [`NO_COLOR`](https://no-color.org) in your environment. That means *no* colour: the screen is drawn entirely with your terminal's own foreground and background, so a light-background theme stays light.

In the AI HUD the colours carry information rather than decoration: each danger bit is green when that turn is safe and red when it kills, the reward is green when positive and red on a death, and the action the agent rates highest is the highlighted one.

---

## AI training

### Presets
Press **C** in the AI submenu to open the config screen. Use the left/right arrow keys to cycle through presets:

| Preset | What it does |
|--------|-------------|
| DEFAULT | Balanced — good starting point |
| BOLD | High learning rate + exploration — learns fast, less stable |
| CAREFUL | Low learning rate — slow but steady convergence |
| SPARSE | No distance shaping — only food/death rewards |

Switching presets applies the new hyperparameters and resets exploration, but **keeps** what the agent has learned. Press **R** on that screen to wipe the Q-table and start from nothing.

### Training modes
- **Fast** — runs episodes without rendering, shows a live sparkline of scores
- **Turbo** — maximum speed, minimal UI updates, best for large episode counts

Both feed the same Q-table the AI plays with, so training always transfers to play.

---

## The self-running agent

The learning core has no curses dependency, so the agent can run with no terminal, no keyboard and no screen:

```bash
python -m serpentos bot --episodes 20000 --preset BOLD   # train
python -m serpentos bot --forever                        # train until stopped
python -m serpentos bot --eval 200 --json                # score it, no learning
python -m serpentos bot --bench                          # reproducible benchmark
```

It checkpoints every 100 episodes, and on `SIGINT`/`SIGTERM` it finishes the current episode, saves, and exits cleanly — so `docker stop` or `systemctl restart` never costs more than one episode.

Roughly 800 episodes/second on a modern laptop core.

See **[docs/AGENT.md](docs/AGENT.md)** for running it under systemd, Docker, cron and CI, plus the full flag reference.

### Sharing a trained brain

A policy is pure data — a table of state keys to three numbers. Importing one never executes anything the author wrote.

```bash
python -m serpentos bot --bench --export-policy my-policy.json --name "your-handle"
python -m serpentos bot --bench --import-policy someone-elses-policy.json
```

The benchmark runs 100 fixed episodes on a fixed grid with fixed seeds, so the same policy scores the same on every machine. See **[docs/ECOSYSTEM.md](docs/ECOSYSTEM.md)**.

---

## Where your data lives

Everything is under `~/.serpentos/` (override with `--data-dir`):

| File | What it is |
|------|-----------|
| `qtable.json` | The learned Q-table plus episode count and exploration rate |
| `leaderboard.json` | Top 10 scores |
| `training_log.csv` | `episode,score,avg50,epsilon,mode` — rotates at 5 MB |
| `serpentos.lock` | Held while a game or agent is running |

Checkpoints are written atomically, so killing the process mid-save cannot corrupt them. If a checkpoint is ever unreadable, it is moved aside as `qtable.json.corrupt-<timestamp>` rather than deleted.

---

## Troubleshooting

| Symptom | Fix |
|---------|-----|
| `ModuleNotFoundError: No module named '_curses'` | Windows: `pip install windows-curses`, or use WSL, or run the headless agent |
| `TERMINAL TOO SMALL` | Resize to at least 40x14 |
| `... is in use by bot (pid N)` | Another agent or game owns the data directory. Stop it, or pass `--data-dir` |
| No colours | Check `echo $TERM` and `echo $NO_COLOR`. `TERM=dumb` or any non-empty `NO_COLOR` disables colour by design |
| Colours look wrong or the box is broken | Set `TERM=xterm-256color`; on Windows use Windows Terminal, not `cmd.exe` |
| The AI plays badly after training | Check `python -m serpentos bot --bench`. A fresh table scores ~0; 3,000 episodes on `BOLD` reaches a mean around 10, and 8,000 around 12 |
| Training feels slow | Use the headless agent, which does not render at all |

---

## Development

```bash
python -m unittest discover -s tests -v
```

493 tests, standard library only, no third-party dependencies. See [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Requirements

- Python 3.9 or higher
- macOS or Linux: no extra packages needed
- Windows: WSL **or** `pip install windows-curses` (UI only — the agent needs neither)
