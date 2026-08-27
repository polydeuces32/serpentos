# Can SerpentOS become a community ecosystem?

An honest assessment, plus the parts that now exist and the parts that do not.

---

## Short answer

Yes, but not as a game. Games with a leaderboard need constant new players to stay alive. What SerpentOS has that most hobby snake clones do not is a **verifiable claim**: a trained agent is a file, and its score is a deterministic function of that file. Anyone can recompute anyone else's number and get the same digits. That is the seed a small technical community can grow around, because it removes the thing that normally kills open leaderboards — trust.

The realistic ceiling is a focused niche: people learning reinforcement learning who want something that runs in one command with no GPU, no PyTorch and no account. That is a real audience, and it is one this project can serve better than heavier frameworks.

---

## Why it is plausible

**The barrier to entry is nearly zero.** Standard library only, Python 3.9+, one file to clone. No install step, no API key, no CUDA. Most RL tutorials lose people at the environment setup; this one starts in five seconds.

**The artifact is small and portable.** A trained Q-table for the default grid is roughly 1,600 states — about 90 KB of JSON. It fits in a gist, a PR diff, or a GitHub release asset. Compare a neural policy, where sharing means hosting weights.

**A policy is data, not code.** An exported policy is a map from string keys to three floats. `import_policy` validates the format, drops any row that is not three numbers, and never evaluates anything. Downloading a stranger's brain cannot run their code. This one property is what makes an open exchange safe by construction rather than by moderation.

**Scores are cheap to verify.** The benchmark is 100 episodes on a fixed 22x78 grid with fixed per-episode seeds and greedy action selection. Scoring a policy takes well under a second. A CI job can re-score every submission on every push.

---

## What now exists

| Primitive | Status |
|-----------|--------|
| Headless agent that runs unattended | `python -m serpentos bot --forever` |
| Deterministic, versioned benchmark | `--bench`, `BENCHMARK_VERSION = 1` |
| Content-addressed policy identity | SHA-256 fingerprint over the sorted table |
| Portable, safe policy format | `serpentos.policy/1`, data only |
| Reproducible runs | `--seed` fixes the environment and the agent |
| Crash-safe, resumable state | Atomic checkpoints, lock, quarantine on corruption |
| CI that enforces reproducibility | Trains, scores twice, fails on any difference |
| Test suite | 62 cases, standard library only |

A worked example, all measured on this branch: 8,000 episodes with `--preset BOLD --seed 21` produces a 1,571-state policy that benchmarks at **mean 12.58, median 13, best 27**, with fingerprint `c965d4e6…`. Re-scoring the exported file in a different process and a different data directory returns exactly those numbers.

That is the whole submission protocol already working. What is missing is a place to put it.

---

## What is still missing

Ordered by how much each one unblocks.

### 1. Somewhere to submit

A leaderboard needs a location. The lowest-maintenance version is a second repository (or a `policies/` directory here) where a submission is a pull request adding one JSON file. A CI job scores it, writes the result into a generated table, and fails the PR if the file is malformed. No server, no database, no hosting bill, and the git history becomes the audit log.

The rule that makes this work: **the score in the table is always the one CI computed, never the one the submitter claimed.**

### 2. Headroom

This is the real risk, and it is worth being blunt about. Tabular Q-learning over eight state fields has a hard ceiling. The agent sees danger in three directions, the food's quadrant, its distance to the wall ahead and a coarse length bucket. It cannot see the shape of its own body. Once the snake is long enough to trap itself, no amount of training fixes that, because the information needed is not in the state.

A leaderboard where everyone converges to the same number within a week is a dead leaderboard. Two ways to create headroom, and the project probably needs both:

- **Open the state.** Let a submission declare which features it uses. A richer state is a bigger table, which is a fair trade to compete on.
- **Open the algorithm.** Define the benchmark against the *environment* rather than the agent, so anything that maps an observation to one of three actions can enter: DQN, Monte Carlo tree search, a hand-written Hamiltonian cycle solver. This needs a stable `Agent` protocol (`act(state) -> int`) and a documented observation, neither of which exists yet.

Opening the algorithm means running submitted code, which forfeits the safety property above. The clean resolution is two tracks:

| Track | Artifact | Verification |
|-------|----------|--------------|
| **Policy** | A `serpentos.policy/1` file | Scored directly. Nothing executes. Anyone can verify locally. |
| **Agent** | A Python file implementing the protocol | Scored in a sandboxed CI job with no network and a time limit. |

Keep the policy track as the default and the front door. It is the one that is safe for a newcomer to try.

### 3. A reason to come back

Static benchmarks get solved and abandoned. Cheap sources of recurring novelty:

- **Seasonal seeds.** Rotate `seed_base` monthly. Same rules, new course, new leaderboard, no code changes — only `BENCHMARK_SPEC`.
- **Constrained divisions.** Best score under 500 states. Best score with `SPARSE` rewards. Best score in under 10,000 training episodes. These reward cleverness rather than compute, which keeps it fair for someone on a laptop.
- **Head-to-head.** Two policies on one grid, most food wins. Needs a two-snake environment, which is a genuine but bounded piece of work.

### 4. Ordinary project hygiene

Nothing exotic, but a project without these reads as abandoned regardless of code quality: a PyPI release so the install is `pipx run serpentos`; issue and PR templates; a code of conduct; a `good first issue` label with three or four real ones; semantic versioning with `BENCHMARK_VERSION` bumped separately, since a benchmark change invalidates every published score.

### 5. Something to look at

Text-only projects are hard to share. A short asciinema recording of the HUD while the agent plays, and a plot of score against episode from `training_log.csv`, are each an afternoon's work and do more for adoption than any feature.

---

## The flywheel

The order matters. Each step should pay for itself before the next one starts.

1. Someone runs one command and watches a snake teach itself. That is the hook.
2. They leave it training and come back to a better score. That is the retention.
3. They export a policy and submit a PR. That is the contribution — and it is a one-file PR, which is the easiest first contribution in open source.
4. CI verifies and publishes it. That is the reward, and it is automatic, so it does not depend on a maintainer being awake.
5. They try to beat their own entry, hit the tabular ceiling, and write a better agent. That is where a user becomes a contributor.

Step 5 is where the project either grows or stalls, which is why the headroom question in section 2 is the one that actually decides the answer.

---

## Honest limits

**This is a niche.** Snake with tabular Q-learning is a teaching problem, not a research frontier. The ceiling on a community is probably tens of contributors, not thousands. That is a perfectly good outcome — but a strategy that assumes otherwise will read as overreach.

**Autonomy is not self-sustaining.** The agent runs itself; the project does not. Every ecosystem here still needs someone to review PRs, keep CI green and adjudicate rule disputes. Automation lowers that cost, it does not remove it. The single most effective thing for longevity is a second maintainer.

**Determinism is a commitment.** The moment a published score cannot be reproduced, the entire premise is gone. That means the benchmark spec is frozen, changes go through `BENCHMARK_VERSION`, and reproducibility is enforced in CI — which is why that job exists on this branch rather than being left as a convention.

**Compute is not the differentiator, and that is the good news.** Anyone can run 800 episodes/second on one core. A student with a laptop competes on equal footing with anyone else. Protect that: resist any division that rewards spending money.

---

## Suggested order of work

Grouped by scope, not schedule.

**Foundation — small, mostly configuration**
Publish to PyPI. Add issue/PR templates and a code of conduct. Record an asciinema demo. Open a handful of genuinely small issues.

**Exchange — one new CI job and a directory convention**
Add `policies/` with a submission workflow that scores every file, regenerates the leaderboard table and fails on malformed input. Document the submission process in one page.

**Headroom — the invasive one**
Define and freeze an `Agent` protocol and a documented observation. Split the benchmark into policy and agent tracks. Sandbox the agent track: no network, hard time limit, resource caps. This touches the core's public interface, so do it before the policy format has many users rather than after.

**Novelty — depends on the two above**
Seasonal seeds and constrained divisions are configuration once the exchange exists. Head-to-head needs a real multi-agent environment and should be scoped separately.
