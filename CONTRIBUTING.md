# Contributing to SerpentOS

Thanks for taking a look. This project has no dependencies and no build step, so getting set up should take under a minute.

```bash
git clone https://github.com/polydeuces32/serpentos.git
cd serpentos
python -m unittest discover -s tests -v
```

Python 3.9 or newer. Nothing to install.

---

## Layout

| Path | What it holds |
|------|---------------|
| `serpentos/core.py` | Game rules, the Q-learning agent, persistence, benchmark. No curses. |
| `serpentos/bot.py` | The headless agent and its CLI. |
| `serpentos/serpentos.py` | The terminal UI. Drawing only — no rules. |
| `serpentos/theme.py` | Colour palettes and the 256/8/monochrome fallback. |
| `serpentos/__main__.py` | Command dispatch for `run`, `bot` and `bench`. |
| `tests/` | Standard library `unittest`. |

The one architectural rule: **`core.py` must never import curses, and the UI must never contain game rules.** That split is what lets the same learning run in a terminal, in CI and in a container, and it is what makes the tests possible.

---

## Before opening a pull request

- `python -m unittest discover -s tests` passes.
- New behaviour has a test. Bug fixes have a regression test that fails without the fix.
- No new dependencies. The standard library has been enough so far; if you think something is genuinely needed, open an issue first.
- Match the surrounding style: 4-space indent, `snake_case`, comments only where the code cannot explain itself.

If you touch the environment, the state encoding or the benchmark, say so explicitly in the PR — those change published scores.

If you touch the UI, never hard-code a colour. Add or reuse a role in `serpentos/theme.py` so the screen still works on an 8-colour terminal and in monochrome, and check it with `TERM=vt100 serpentos run`.

---

## Changing the benchmark

`BENCHMARK_SPEC` in `core.py` is frozen on purpose. Every published score is only meaningful because the spec has not moved. If you need to change it:

1. Bump `BENCHMARK_VERSION`.
2. Say in the PR that previously published scores are invalidated.
3. Do not change the spec and the scoring code in the same PR.

The same applies to the state encoding. Q-table keys are the on-disk format; changing them silently invalidates every saved agent and every shared policy.

---

## Submitting a trained policy

Policies are data, never code:

```bash
python -m serpentos bot --episodes 50000 --preset BOLD --data-dir runs/mine
python -m serpentos bot --bench --data-dir runs/mine \
    --export-policy my-policy.json --name "your-handle"
```

Include the benchmark output in the PR. Do not edit the score by hand — it is recomputed from the file, and a mismatch between the claimed and computed value will fail review.

---

## Good first contributions

- A plotting script for `training_log.csv`.
- An asciinema recording of the agent playing with the HUD on.
- Additional presets with a note on what they are for.
- Documentation fixes — especially anything in the README that did not match what actually happened when you ran it.

---

## Reporting bugs

Include your OS, Python version (`python3 --version`), terminal size (`stty size`), how you started it, and the full traceback. If it involves a trained agent, `python -m serpentos bot --bench` output identifies the policy exactly.
