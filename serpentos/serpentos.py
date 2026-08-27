#!/usr/bin/env python3
"""SerpentOS terminal UI.

All game rules, learning and persistence live in :mod:`serpentos.core`; this
module only draws them. Run it directly (``python3 serpentos/serpentos.py``),
as a module (``python -m serpentos``), or skip it entirely and use the headless
agent (``python -m serpentos bot``).
"""

from __future__ import annotations

import argparse
import math
import os
import sys
import time
from typing import List, Optional

try:
    import curses
except ImportError:  # pragma: no cover - platform specific
    sys.stderr.write(
        "SerpentOS needs the curses module for its terminal UI.\n"
        "  Windows : pip install windows-curses   (or run it under WSL)\n"
        "  Anywhere: python -m serpentos bot --episodes 1000   (no curses needed)\n"
    )
    raise SystemExit(1)

try:
    from . import core
except ImportError:  # executed as a plain script: python serpentos/serpentos.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import core  # type: ignore[no-redef]

DIFFICULTY = core.DIFFICULTY
PRESETS = core.PRESETS
PRESET_NAMES = core.PRESET_NAMES

# Enough room for the border, the six-line HUD and a playable grid.
MIN_HEIGHT = core.SnakeEnv.MIN_ROWS + 8
MIN_WIDTH = max(core.SnakeEnv.MIN_COLS + 4, 40)

CHECKPOINT_EVERY = 50  # episodes; the Q-table is written atomically, not per step


# =========================
# SAFE CURSES HELPERS
# =========================
def safe_addch(stdscr, y, x, ch, attr=0):
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h or x < 0 or x >= w:
        return
    try:
        stdscr.addch(y, x, ch, attr)
    except curses.error:
        pass


def safe_addstr(stdscr, y, x, s, attr=0):
    h, w = stdscr.getmaxyx()
    if y < 0 or y >= h or x >= w:
        return
    if x < 0:
        s = s[-x:]
        x = 0
    if not s:
        return
    try:
        stdscr.addstr(y, x, s[: max(0, w - x - 1)], attr)
    except curses.error:
        pass


def init_colors():
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)


def draw_center(stdscr, y, text, attr=0):
    h, w = stdscr.getmaxyx()
    x = max(0, (w // 2) - (len(text) // 2))
    safe_addstr(stdscr, y, x, text, attr)


def terminal_too_small(stdscr) -> bool:
    h, w = stdscr.getmaxyx()
    return h < MIN_HEIGHT or w < MIN_WIDTH


def size_warning(stdscr) -> bool:
    """Block until the terminal is big enough. Returns False if the user quits."""
    stdscr.nodelay(False)
    while terminal_too_small(stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.clear()
        safe_addstr(stdscr, 0, 0, "TERMINAL TOO SMALL")
        safe_addstr(stdscr, 1, 0, f"have {w}x{h}, need {MIN_WIDTH}x{MIN_HEIGHT}")
        safe_addstr(stdscr, 2, 0, "resize, or Q to quit")
        stdscr.refresh()
        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            return False
    return True


# =========================
# BOOT ANIMATION
# =========================
def boot_animation(stdscr):
    curses.curs_set(0)
    init_colors()

    h, w = stdscr.getmaxyx()
    cy, cx = h // 2, w // 2
    r = max(3, min(h, w) // 6)

    dots = 28
    trail = 10
    frames = 70
    chars = [".", ":", "o", "O", "0"]

    pts = []
    for i in range(dots):
        a = 2 * math.pi * i / dots
        pts.append((int(cy + math.sin(a) * r), int(cx + math.cos(a) * r * 2)))

    stdscr.nodelay(True)
    for t in range(frames):
        if stdscr.getch() != -1:  # any key skips the intro
            break
        stdscr.clear()
        head = t % dots
        for i in range(dots):
            d = (head - i) % dots
            if d > trail:
                continue
            y, x = pts[i]
            ch = chars[max(0, len(chars) - 1 - d)]
            attr = curses.color_pair(1)
            if d == 0:
                attr |= curses.A_BOLD
            elif d >= trail - 2:
                attr |= curses.A_DIM
            safe_addch(stdscr, y, x, ch, attr)

        title = "SERPENTOS CORE"
        sub = "BOOTING TERMINAL SIM"
        safe_addstr(stdscr, cy + r + 2, cx - len(title) // 2, title, curses.color_pair(1) | curses.A_BOLD)
        safe_addstr(stdscr, cy + r + 3, cx - len(sub) // 2, sub, curses.color_pair(1))
        stdscr.refresh()
        time.sleep(0.04)

    stdscr.nodelay(False)
    stdscr.clear()
    stdscr.refresh()


# =========================
# MENUS
# =========================
def menu(stdscr, lines, accept, bold_rows=()):
    stdscr.nodelay(False)
    stdscr.clear()
    init_colors()
    h, _ = stdscr.getmaxyx()
    y0 = max(0, h // 2 - len(lines) // 2)
    for i, line in enumerate(lines):
        bold = curses.A_BOLD if i in bold_rows else 0
        draw_center(stdscr, y0 + i, line, curses.color_pair(1) | bold)
    stdscr.refresh()

    while True:
        k = stdscr.getch()
        if 0 <= k < 256 and chr(k).lower() in accept:
            return chr(k).lower()


def main_menu(stdscr):
    return menu(
        stdscr,
        ["1  HUMAN SNAKE", "2  AI SNAKE (Q-LEARNING)", "3  LEADERBOARD", "4  AGENT STATUS", "", "Q  QUIT"],
        {"1", "2", "3", "4", "q"},
        bold_rows={0, 1, 2, 3, 5},
    )


def ai_submenu(stdscr, current_preset):
    return menu(
        stdscr,
        [
            "AI SNAKE (Q-LEARNING)",
            "",
            "1  PLAY (LIVE HUD)",
            "2  TRAIN FAST (NO RENDER) THEN PLAY",
            "3  TRAIN TURBO (MAX SPEED) THEN PLAY",
            "",
            f"C  CONFIG (preset: {current_preset})",
            "B  BACK",
        ],
        {"1", "2", "3", "c", "b"},
        bold_rows={0, 2, 3, 4, 6, 7},
    )


def difficulty_menu(stdscr):
    choice = menu(
        stdscr,
        ["SELECT DIFFICULTY", "", "1  EASY", "2  NORMAL", "3  HARD"],
        {"1", "2", "3"},
        bold_rows={0, 2, 3, 4},
    )
    return ["easy", "normal", "hard"][int(choice) - 1]


def training_menu(stdscr, turbo=False):
    counts = (1000, 2000, 5000, 10000) if turbo else (200, 500, 1000, 2000)
    title = "TRAINING EPISODES (TURBO)" if turbo else "TRAINING EPISODES (FAST)"
    lines = [title, ""] + [f"{i + 1}  {n}" for i, n in enumerate(counts)] + ["", "B  BACK"]
    choice = menu(stdscr, lines, {"1", "2", "3", "4", "b"}, bold_rows={0, 2, 3, 4, 5, 7})
    if choice == "b":
        return None
    return counts[int(choice) - 1]


def ai_config_menu(stdscr, current_preset):
    """Returns ``(preset, reset_requested)``."""
    stdscr.nodelay(False)
    init_colors()
    idx = PRESET_NAMES.index(current_preset)

    while True:
        stdscr.clear()
        stdscr.box()
        safe_addstr(stdscr, 0, 2, " AI CONFIG ", curses.color_pair(1) | curses.A_BOLD)
        p = PRESETS[PRESET_NAMES[idx]]
        lines = [
            f"PRESET: < {PRESET_NAMES[idx]} >  (LEFT/RIGHT to change)",
            f"  {p['desc']}",
            "",
            f"  alpha     = {p['alpha']:.2f}   (learning rate)",
            f"  gamma     = {p['gamma']:.2f}   (discount factor)",
            f"  epsilon   = {p['epsilon']:.2f}   (initial exploration)",
            f"  eps_min   = {p['eps_min']:.2f}   (min exploration)",
            f"  eps_decay = {p['eps_decay']:.3f} (per-episode decay)",
            f"  shaping   = {'ON  (distance reward)' if p['use_shaping'] else 'OFF (sparse only)'}",
            "",
            "ENTER confirm   R wipe learned Q-table   B back",
        ]
        h, _ = stdscr.getmaxyx()
        y0 = max(2, h // 2 - len(lines) // 2)
        for i, line in enumerate(lines):
            attr = curses.color_pair(1) | (curses.A_BOLD if i == 0 else 0)
            safe_addstr(stdscr, y0 + i, 4, line, attr)
        stdscr.refresh()

        k = stdscr.getch()
        if k == curses.KEY_LEFT:
            idx = (idx - 1) % len(PRESET_NAMES)
        elif k == curses.KEY_RIGHT:
            idx = (idx + 1) % len(PRESET_NAMES)
        elif k in (10, 13, ord(" ")):
            return PRESET_NAMES[idx], False
        elif k in (ord("r"), ord("R")):
            if confirm(stdscr, "WIPE THE LEARNED Q-TABLE? (Y/N)"):
                return PRESET_NAMES[idx], True
        elif k in (ord("b"), ord("B"), 27):
            return current_preset, False


def confirm(stdscr, prompt) -> bool:
    h, _ = stdscr.getmaxyx()
    draw_center(stdscr, h - 3, prompt, curses.color_pair(1) | curses.A_BOLD)
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k in (ord("y"), ord("Y")):
            return True
        if k in (ord("n"), ord("N"), 27):
            return False


def leaderboard_view(stdscr, storage):
    stdscr.nodelay(False)
    init_colors()
    items = storage.load_leaderboard()

    while True:
        stdscr.clear()
        stdscr.box()
        safe_addstr(stdscr, 0, 2, " LEADERBOARD (TOP 10) ", curses.color_pair(1) | curses.A_BOLD)
        if not items:
            draw_center(stdscr, 3, "NO SCORES YET", curses.color_pair(1))
        else:
            for i, item in enumerate(items, start=1):
                line = (
                    f"{i:>2}. {item.get('score', 0):>4}  {item.get('mode', '?'):<10}  "
                    f"{item.get('diff', '?'):<6}  {item.get('ts', '')}"
                )
                safe_addstr(stdscr, 2 + i, 2, line, curses.color_pair(1))
        draw_center(stdscr, stdscr.getmaxyx()[0] - 2, "PRESS B TO GO BACK", curses.color_pair(1))
        stdscr.refresh()
        if stdscr.getch() in (ord("b"), ord("B"), 27):
            return


def agent_status_view(stdscr, agent, storage, preset):
    stdscr.nodelay(False)
    init_colors()
    stdscr.clear()
    stdscr.box()
    safe_addstr(stdscr, 0, 2, " AGENT STATUS ", curses.color_pair(1) | curses.A_BOLD)
    avg_food = agent.total_food / agent.episodes if agent.episodes else 0.0
    lines = [
        f"PRESET          : {preset}",
        f"EPISODES LIVED  : {agent.episodes}",
        f"FOOD EATEN      : {agent.total_food}  (avg {avg_food:.2f}/episode)",
        f"KNOWN STATES    : {len(agent.q)}",
        f"EPSILON         : {agent.epsilon:.4f}",
        f"DATA DIR        : {storage.dir}",
        "",
        "Headless training:",
        "  python -m serpentos bot --episodes 20000 --preset BOLD",
        "",
        "PRESS B TO GO BACK",
    ]
    for i, line in enumerate(lines):
        safe_addstr(stdscr, 2 + i, 2, line, curses.color_pair(1))
    stdscr.refresh()
    while True:
        if stdscr.getch() in (ord("b"), ord("B"), 27):
            return


# =========================
# RENDERING
# =========================
def sparkline(values, width=30):
    if not values:
        return " " * width
    bars = " \u2581\u2582\u2583\u2584\u2585\u2586\u2587\u2588"
    lo, hi = min(values), max(values)
    if hi == lo:
        return ("\u2584" * min(len(values), width)).ljust(width)
    out = ""
    for v in list(values)[-width:]:
        out += bars[int((v - lo) / (hi - lo) * (len(bars) - 1))]
    return out.ljust(width)


def env_for(stdscr, shaping: bool, rng=None) -> core.SnakeEnv:
    """Fit an environment to the window interior (offset by the drawn border)."""
    h, w = stdscr.getmaxyx()
    return core.SnakeEnv(max(core.SnakeEnv.MIN_ROWS, h - 2),
                         max(core.SnakeEnv.MIN_COLS, w - 2),
                         shaping=shaping, rng=rng)


def render_board(stdscr, env, header):
    stdscr.clear()
    stdscr.box()
    safe_addstr(stdscr, 0, 2, header, curses.color_pair(1) | curses.A_BOLD)
    for y, x in env.snake:
        safe_addch(stdscr, y + 1, x + 1, "#", curses.color_pair(1))
    if env.food is not None:
        safe_addch(stdscr, env.food[0] + 1, env.food[1] + 1, "*", curses.color_pair(2))


def draw_ai_hud(stdscr, agent, state, abs_dir, score, steps, diff_name):
    q0, q1, q2 = agent.last_qvals
    bucket = ("short", "medium", "long")[state.length_bucket]
    lines = [
        f"MODE: AI(Q)   DIFF: {diff_name}   SCORE: {score}   STEPS: {steps}",
        f"EPS: {agent.epsilon:.3f}  ALPHA: {agent.alpha:.2f}  GAMMA: {agent.gamma:.2f}  EPISODES: {agent.episodes}",
        f"STATE: dx={state.dx} dy={state.dy}  danger(a,l,r)="
        f"({state.danger_ahead},{state.danger_left},{state.danger_right})  "
        f"dir={state.direction}  wall={state.wall_dist}  len={bucket}",
        f"ACTION: rel={agent.last_action} abs={abs_dir}  REWARD: {agent.last_reward:+.2f}",
        f"Q: straight={q0:+.2f}  left={q1:+.2f}  right={q2:+.2f}",
        "HUD: H toggle   Q quit",
    ]
    for i, line in enumerate(lines):
        safe_addstr(stdscr, 1 + i, 2, line, curses.color_pair(1) | (curses.A_BOLD if i == 0 else 0))


# =========================
# GAME LOOPS
# =========================
def play_human(stdscr, speed, diff_name):
    init_colors()
    env = env_for(stdscr, shaping=False)
    stdscr.nodelay(True)

    key_map = {
        curses.KEY_UP: "U", curses.KEY_DOWN: "D", curses.KEY_LEFT: "L", curses.KEY_RIGHT: "R",
        ord("w"): "U", ord("s"): "D", ord("a"): "L", ord("d"): "R",
        ord("W"): "U", ord("S"): "D", ord("A"): "L", ord("D"): "R",
    }

    while True:
        render_board(stdscr, env, f" HUMAN  DIFF: {diff_name}  SCORE: {env.score} ")
        stdscr.refresh()

        direction = env.direction
        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            return env.score, True
        if k in key_map:
            direction = key_map[k]

        _, _, done, info = env.step_dir(direction)
        if done:
            render_board(stdscr, env, f" HUMAN  DIFF: {diff_name}  SCORE: {env.score} ")
            stdscr.refresh()
            return info.score, False
        time.sleep(speed)


def play_ai(stdscr, speed, diff_name, agent, shaping):
    init_colors()
    env = env_for(stdscr, shaping=shaping)
    stdscr.nodelay(True)
    hud = {"on": True}

    def on_step(t: core.Transition) -> bool:
        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            return False
        if k in (ord("h"), ord("H")):
            hud["on"] = not hud["on"]
        render_board(stdscr, env, f" AI(Q)  DIFF: {diff_name}  SCORE: {env.score} ")
        if hud["on"]:
            abs_dir = core.rel_to_abs(t.state.direction, t.action)
            draw_ai_hud(stdscr, agent, t.state, abs_dir, env.score, t.info.steps, diff_name)
        stdscr.refresh()
        time.sleep(speed)
        return True

    info = core.run_episode(env, agent, train=True, on_step=on_step)
    return info.score, info.reason == "aborted"


def train(stdscr, agent, storage, episodes, diff_name, shaping, preset, turbo=False):
    """Headless-speed training with a periodic progress panel."""
    init_colors()
    stdscr.nodelay(True)
    env = env_for(stdscr, shaping=shaping)

    label = "TURBO" if turbo else "FAST"
    ui_every = 50 if turbo else 10
    window: List[int] = []
    best = 0
    aborted = False

    with storage.training_log() as log:
        for ep in range(1, episodes + 1):
            k = stdscr.getch()
            if k in (ord("q"), ord("Q")):
                aborted = True
                break

            info = core.run_episode(env, agent, train=True)
            window.append(info.score)
            if len(window) > 50:
                window.pop(0)
            avg = core.moving_average(window)
            best = max(best, info.score)
            log.write(agent.episodes, info.score, avg, agent.epsilon, label.lower())

            if ep % CHECKPOINT_EVERY == 0:
                save_checkpoint(storage, agent, stdscr, preset)

            if ep == 1 or ep % ui_every == 0 or ep == episodes:
                log.flush()
                stdscr.clear()
                stdscr.box()
                safe_addstr(stdscr, 0, 2, f" TRAINING ({label}) ", curses.color_pair(1) | curses.A_BOLD)
                safe_addstr(stdscr, 2, 2, f"DIFF: {diff_name}  SHAPING: {'ON' if shaping else 'OFF'}", curses.color_pair(1))
                safe_addstr(stdscr, 3, 2, f"EPISODE: {ep}/{episodes}", curses.color_pair(1) | curses.A_BOLD)
                safe_addstr(stdscr, 4, 2, f"AVG (last 50): {avg:.2f}   BEST: {best}", curses.color_pair(1))
                safe_addstr(stdscr, 5, 2, f"EPS: {agent.epsilon:.3f}   STATES: {len(agent.q)}", curses.color_pair(1))
                safe_addstr(stdscr, 6, 2, f"SCORES: {sparkline(window)}", curses.color_pair(1))
                safe_addstr(stdscr, 8, 2, "PRESS Q TO ABORT", curses.color_pair(1))
                stdscr.refresh()

    save_checkpoint(storage, agent, stdscr, preset)
    stdscr.nodelay(False)
    return best, aborted


def save_checkpoint(storage, agent, stdscr=None, preset=None):
    try:
        storage.save_checkpoint(agent.q, agent.meta(preset))
    except OSError as exc:
        if stdscr is not None:
            h, _ = stdscr.getmaxyx()
            safe_addstr(stdscr, h - 2, 2, f"WARNING: could not save progress: {exc}", curses.A_BOLD)
            stdscr.refresh()


# =========================
# END SCREENS
# =========================
def game_over_screen(stdscr, score):
    stdscr.nodelay(False)
    h, _ = stdscr.getmaxyx()
    stdscr.clear()
    stdscr.box()
    draw_center(stdscr, h // 2 - 2, "GAME OVER", curses.A_BOLD)
    draw_center(stdscr, h // 2, f"SCORE: {score}")
    draw_center(stdscr, h // 2 + 2, "PLAY AGAIN? (Y/N)")
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k in (ord("y"), ord("Y")):
            return True
        if k in (ord("n"), ord("N"), 27):
            return False


def post_training_screen(stdscr, storage, best, aborted, turbo=False):
    init_colors()
    stdscr.nodelay(False)
    stdscr.clear()
    stdscr.box()
    title = " TRAINING COMPLETE (TURBO) " if turbo else " TRAINING COMPLETE (FAST) "
    safe_addstr(stdscr, 0, 2, title, curses.color_pair(1) | curses.A_BOLD)
    safe_addstr(stdscr, 3, 2, f"BEST SCORE: {best}", curses.color_pair(1))
    if aborted:
        safe_addstr(stdscr, 4, 2, "STATUS: ABORTED BY USER", curses.color_pair(1))
    safe_addstr(stdscr, 5, 2, f"LOG: {storage.training_log_path}", curses.color_pair(1))
    safe_addstr(stdscr, 7, 2, "1  PLAY TRAINED AI NOW", curses.color_pair(1) | curses.A_BOLD)
    safe_addstr(stdscr, 8, 2, "B  BACK", curses.color_pair(1) | curses.A_BOLD)
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k == ord("1"):
            return "play"
        if k in (ord("b"), ord("B"), 27):
            return "back"


# =========================
# APP
# =========================
def run_ui(stdscr, storage):
    if not size_warning(stdscr):
        return
    boot_animation(stdscr)

    preset = "DEFAULT"
    agent = core.load_agent(storage, preset)

    while True:
        if not size_warning(stdscr):
            return
        choice = main_menu(stdscr)

        if choice == "q":
            save_checkpoint(storage, agent, stdscr, preset)
            return
        if choice == "3":
            leaderboard_view(stdscr, storage)
            continue
        if choice == "4":
            agent_status_view(stdscr, agent, storage, preset)
            continue

        diff_key = difficulty_menu(stdscr)
        speed = DIFFICULTY[diff_key]["speed"]
        diff_name = DIFFICULTY[diff_key]["name"]

        if choice == "1":
            score, quit_early = play_human(stdscr, speed, diff_name)
            storage.add_score(score, "HUMAN", diff_name)
            if quit_early:
                continue
            if not game_over_screen(stdscr, score):
                return
            continue

        # AI branch
        while True:
            sub = ai_submenu(stdscr, preset)
            if sub == "b":
                break
            if sub == "c":
                new_preset, wipe = ai_config_menu(stdscr, preset)
                if wipe:
                    agent.reset_knowledge()
                if new_preset != preset or wipe:
                    preset = new_preset
                    agent.apply_preset(preset, reset_epsilon=True)
                save_checkpoint(storage, agent, stdscr, preset)
                continue

            shaping = PRESETS[preset]["use_shaping"]

            if sub in ("2", "3"):
                turbo = sub == "3"
                episodes = training_menu(stdscr, turbo=turbo)
                if episodes is None:
                    continue
                best, aborted = train(stdscr, agent, storage, episodes, diff_name, shaping, preset, turbo=turbo)
                if post_training_screen(stdscr, storage, best, aborted, turbo=turbo) != "play":
                    continue

            score, quit_early = play_ai(stdscr, speed, diff_name, agent, shaping)
            save_checkpoint(storage, agent, stdscr, preset)
            storage.add_score(score, "AI(Q)", diff_name)
            if quit_early:
                break
            if not game_over_screen(stdscr, score):
                return


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(prog="serpentos", description="SerpentOS terminal UI")
    parser.add_argument("--data-dir", default=None,
                        help=f"state directory (default: {core.DEFAULT_DATA_DIR})")
    args = parser.parse_args(argv)

    storage = core.Storage(args.data_dir)
    try:
        with storage.lock(owner="ui"):
            curses.wrapper(run_ui, storage)
    except core.LockError as exc:
        sys.stderr.write(f"{exc}\n")
        return 2
    except OSError as exc:
        sys.stderr.write(f"SerpentOS could not use {storage.dir}: {exc}\n")
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
