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
    from .theme import Theme
except ImportError:  # executed as a plain script: python serpentos/serpentos.py
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import core  # type: ignore[no-redef]
    from theme import Theme  # type: ignore[no-redef]

DIFFICULTY = core.DIFFICULTY
PRESETS = core.PRESETS
PRESET_NAMES = core.PRESET_NAMES

# Enough room for the border, the six-line HUD and a playable grid.
MIN_HEIGHT = core.SnakeEnv.MIN_ROWS + 8
MIN_WIDTH = max(core.SnakeEnv.MIN_COLS + 4, 40)

CHECKPOINT_EVERY = 50  # episodes; the Q-table is written atomically, not per step

HEAD_CH = "@"
BODY_CH = "#"
FOOD_CH = "*"

_THEME: Optional[Theme] = None


def T(role: str, extra: int = 0) -> int:
    """Attribute for a theme role. Falls back to plain text before setup."""
    return _THEME(role, extra) if _THEME is not None else extra


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


def hide_cursor():
    """Not every terminal can hide the cursor; a VT100 cannot."""
    try:
        curses.curs_set(0)
    except curses.error:
        pass


def draw_center(stdscr, y, text, attr=0):
    h, w = stdscr.getmaxyx()
    x = max(0, (w // 2) - (len(text) // 2))
    safe_addstr(stdscr, y, x, text, attr)


def draw_frame(stdscr, title=None, title_role="title"):
    """Clear the screen and draw the border, tinted, with an optional title."""
    stdscr.clear()
    stdscr.attrset(T("border"))
    stdscr.box()
    stdscr.attrset(0)
    if title:
        safe_addstr(stdscr, 0, 2, title, T(title_role))


def terminal_too_small(stdscr) -> bool:
    h, w = stdscr.getmaxyx()
    return h < MIN_HEIGHT or w < MIN_WIDTH


def size_warning(stdscr) -> bool:
    """Block until the terminal is big enough. Returns False if the user quits."""
    stdscr.nodelay(False)
    while terminal_too_small(stdscr):
        h, w = stdscr.getmaxyx()
        stdscr.clear()
        safe_addstr(stdscr, 0, 0, "TERMINAL TOO SMALL", T("bad", curses.A_BOLD))
        safe_addstr(stdscr, 1, 0, f"have {w}x{h}, need {MIN_WIDTH}x{MIN_HEIGHT}", T("text"))
        safe_addstr(stdscr, 2, 0, "resize, or Q to quit", T("dim"))
        stdscr.refresh()
        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            return False
    return True


# =========================
# BOOT ANIMATION
# =========================
def boot_animation(stdscr):
    hide_cursor()

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
            if d == 0:
                role = "glow_hot"
            elif d >= trail - 3:
                role = "glow_cool"
            else:
                role = "glow_warm"
            safe_addch(stdscr, y, x, ch, T(role))

        title = "SERPENTOS CORE"
        sub = "BOOTING TERMINAL SIM"
        safe_addstr(stdscr, cy + r + 2, cx - len(title) // 2, title, T("title"))
        safe_addstr(stdscr, cy + r + 3, cx - len(sub) // 2, sub, T("accent"))
        stdscr.refresh()
        time.sleep(0.04)

    stdscr.nodelay(False)
    stdscr.clear()
    stdscr.refresh()


# =========================
# MENUS
# =========================
def draw_menu_line(stdscr, y, line, selectable, x=None):
    """Draw ``K  LABEL`` with the shortcut key picked out from the label."""
    if x is None:
        _, w = stdscr.getmaxyx()
        x = max(0, (w // 2) - (len(line) // 2))
    if selectable and len(line) > 3 and line[1:3] == "  ":
        safe_addstr(stdscr, y, x, line[0], T("value", curses.A_BOLD))
        safe_addstr(stdscr, y, x + 3, line[3:], T("accent"))
    else:
        safe_addstr(stdscr, y, x, line, T("title" if selectable else "dim"))


def menu(stdscr, lines, accept, bold_rows=()):
    stdscr.nodelay(False)
    stdscr.clear()
    h, _ = stdscr.getmaxyx()
    y0 = max(0, h // 2 - len(lines) // 2)
    for i, line in enumerate(lines):
        draw_menu_line(stdscr, y0 + i, line, i in bold_rows)
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
    idx = PRESET_NAMES.index(current_preset)

    while True:
        draw_frame(stdscr, " AI CONFIG ")
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
            if i == 0:
                role = "title"
            elif i == 1:
                role = "dim"
            elif i == len(lines) - 1:
                role = "accent"
            else:
                role = "text"
            safe_addstr(stdscr, y0 + i, 4, line, T(role))
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
    draw_center(stdscr, h - 3, prompt, T("bad", curses.A_BOLD))
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k in (ord("y"), ord("Y")):
            return True
        if k in (ord("n"), ord("N"), 27):
            return False


def draw_field(stdscr, y, x, label, value, value_role="value"):
    """A ``LABEL : value`` row, with the two halves coloured differently."""
    safe_addstr(stdscr, y, x, label, T("label"))
    safe_addstr(stdscr, y, x + len(label), value, T(value_role))


def leaderboard_view(stdscr, storage):
    stdscr.nodelay(False)
    items = storage.load_leaderboard()

    while True:
        draw_frame(stdscr, " LEADERBOARD (TOP 10) ")
        if not items:
            draw_center(stdscr, 3, "NO SCORES YET", T("dim"))
        else:
            safe_addstr(stdscr, 2, 2, f"{'#':>2}  {'SCORE':>5}  {'MODE':<10}  {'DIFF':<6}  WHEN", T("label"))
            for i, item in enumerate(items, start=1):
                # The podium gets picked out; everything else is plain text.
                role = "good" if i == 1 else ("accent" if i <= 3 else "text")
                safe_addstr(stdscr, 2 + i, 2, f"{i:>2}.", T("dim"))
                safe_addstr(stdscr, 2 + i, 6, f"{item.get('score', 0):>5}", T(role, curses.A_BOLD))
                safe_addstr(
                    stdscr, 2 + i, 13,
                    f"{item.get('mode', '?'):<10}  {item.get('diff', '?'):<6}  {item.get('ts', '')}",
                    T("text" if i <= 3 else "dim"),
                )
        draw_center(stdscr, stdscr.getmaxyx()[0] - 2, "PRESS B TO GO BACK", T("dim"))
        stdscr.refresh()
        if stdscr.getch() in (ord("b"), ord("B"), 27):
            return


def agent_status_view(stdscr, agent, storage, preset):
    stdscr.nodelay(False)
    draw_frame(stdscr, " AGENT STATUS ")
    avg_food = agent.total_food / agent.episodes if agent.episodes else 0.0
    fields = [
        ("PRESET          : ", preset, "accent"),
        ("EPISODES LIVED  : ", f"{agent.episodes}", "value"),
        ("FOOD EATEN      : ", f"{agent.total_food}  (avg {avg_food:.2f}/episode)", "value"),
        ("KNOWN STATES    : ", f"{len(agent.q)}", "value"),
        ("EPSILON         : ", f"{agent.epsilon:.4f}", "value"),
        ("DATA DIR        : ", storage.dir, "text"),
    ]
    for i, (label, value, role) in enumerate(fields):
        draw_field(stdscr, 2 + i, 2, label, value, role)

    y = 2 + len(fields) + 1
    safe_addstr(stdscr, y, 2, "Headless training:", T("label"))
    safe_addstr(stdscr, y + 1, 2, "  python -m serpentos bot --episodes 20000 --preset BOLD", T("accent"))
    safe_addstr(stdscr, y + 3, 2, "PRESS B TO GO BACK", T("dim"))
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


def draw_progress_bar(stdscr, y, x, fraction, width=30):
    fraction = max(0.0, min(1.0, fraction))
    filled = int(round(fraction * width))
    safe_addstr(stdscr, y, x, "[", T("dim"))
    safe_addstr(stdscr, y, x + 1, "=" * filled, T("good"))
    safe_addstr(stdscr, y, x + 1 + filled, "." * (width - filled), T("dim"))
    safe_addstr(stdscr, y, x + 1 + width, f"] {fraction * 100:3.0f}%", T("label"))


def env_for(stdscr, shaping: bool, rng=None) -> core.SnakeEnv:
    """Fit an environment to the window interior (offset by the drawn border)."""
    h, w = stdscr.getmaxyx()
    return core.SnakeEnv(max(core.SnakeEnv.MIN_ROWS, h - 2),
                         max(core.SnakeEnv.MIN_COLS, w - 2),
                         shaping=shaping, rng=rng)


def render_board(stdscr, env, mode_label, diff_name, score):
    draw_frame(stdscr)
    safe_addstr(stdscr, 0, 2, f" {mode_label} ", T("title"))
    x = 4 + len(mode_label)
    draw_field(stdscr, 0, x, " DIFF: ", f"{diff_name} ", "accent")
    draw_field(stdscr, 0, x + 8 + len(diff_name), " SCORE: ", f"{score} ", "good")

    length = len(env.snake)
    theme = _THEME
    for i, (y, x) in enumerate(env.snake):
        if i == 0:
            safe_addch(stdscr, y + 1, x + 1, HEAD_CH, T("head"))
        else:
            role = theme.body_role(i, length) if theme is not None else "body"
            safe_addch(stdscr, y + 1, x + 1, BODY_CH, T(role))
    if env.food is not None:
        safe_addch(stdscr, env.food[0] + 1, env.food[1] + 1, FOOD_CH, T("food"))


def draw_ai_hud(stdscr, agent, state, abs_dir, steps, diff_name):
    q0, q1, q2 = agent.last_qvals
    bucket = ("short", "medium", "long")[state.length_bucket]
    danger = (state.danger_ahead, state.danger_left, state.danger_right)

    draw_field(stdscr, 1, 2, "STEPS: ", f"{steps}")
    draw_field(stdscr, 1, 18, "EPISODES: ", f"{agent.episodes}")
    draw_field(stdscr, 1, 40, "EPS: ", f"{agent.epsilon:.3f}")
    draw_field(stdscr, 1, 54, "a/g: ", f"{agent.alpha:.2f}/{agent.gamma:.2f}")

    draw_field(stdscr, 2, 2, "FOOD: ", f"dx={state.dx:+d} dy={state.dy:+d}")
    safe_addstr(stdscr, 2, 22, "DANGER: ", T("label"))
    # Each danger bit is coloured on its own: red means that turn kills.
    for i, (name, bit) in enumerate(zip(("a", "l", "r"), danger)):
        safe_addstr(stdscr, 2, 30 + i * 4, f"{name}{bit}", T("bad" if bit else "good"))
    draw_field(stdscr, 2, 44, "dir: ", state.direction)
    draw_field(stdscr, 2, 54, "wall: ", str(state.wall_dist))
    draw_field(stdscr, 2, 64, "len: ", bucket)

    draw_field(stdscr, 3, 2, "ACTION: ", f"rel={agent.last_action} abs={abs_dir}")
    draw_field(stdscr, 3, 26, "REWARD: ", f"{agent.last_reward:+.2f}",
               "good" if agent.last_reward > 0 else ("bad" if agent.last_reward <= -1 else "dim"))

    # Highlight whichever action the agent rates highest.
    qvals = (q0, q1, q2)
    best = max(qvals)
    safe_addstr(stdscr, 4, 2, "Q: ", T("label"))
    for i, (name, value) in enumerate(zip(("straight", "left", "right"), qvals)):
        chosen = value == best
        safe_addstr(stdscr, 4, 6 + i * 20, f"{name}=", T("label"))
        safe_addstr(stdscr, 4, 6 + i * 20 + len(name) + 1, f"{value:+.2f}",
                    T("good" if chosen else "dim", curses.A_BOLD if chosen else 0))

    safe_addstr(stdscr, 5, 2, "H toggle HUD   Q quit", T("dim"))


# =========================
# GAME LOOPS
# =========================
def play_human(stdscr, speed, diff_name):
    env = env_for(stdscr, shaping=False)
    stdscr.nodelay(True)

    key_map = {
        curses.KEY_UP: "U", curses.KEY_DOWN: "D", curses.KEY_LEFT: "L", curses.KEY_RIGHT: "R",
        ord("w"): "U", ord("s"): "D", ord("a"): "L", ord("d"): "R",
        ord("W"): "U", ord("S"): "D", ord("A"): "L", ord("D"): "R",
    }

    while True:
        render_board(stdscr, env, "HUMAN", diff_name, env.score)
        stdscr.refresh()

        direction = env.direction
        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            return env.score, True
        if k in key_map:
            direction = key_map[k]

        _, _, done, info = env.step_dir(direction)
        if done:
            render_board(stdscr, env, "HUMAN", diff_name, env.score)
            stdscr.refresh()
            return info.score, False
        time.sleep(speed)


def play_ai(stdscr, speed, diff_name, agent, shaping):
    env = env_for(stdscr, shaping=shaping)
    stdscr.nodelay(True)
    hud = {"on": True}

    def on_step(t: core.Transition) -> bool:
        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            return False
        if k in (ord("h"), ord("H")):
            hud["on"] = not hud["on"]
        render_board(stdscr, env, "AI(Q)", diff_name, env.score)
        if hud["on"]:
            abs_dir = core.rel_to_abs(t.state.direction, t.action)
            draw_ai_hud(stdscr, agent, t.state, abs_dir, t.info.steps, diff_name)
        stdscr.refresh()
        time.sleep(speed)
        return True

    info = core.run_episode(env, agent, train=True, on_step=on_step)
    return info.score, info.reason == "aborted"


def train(stdscr, agent, storage, episodes, diff_name, shaping, preset, turbo=False):
    """Headless-speed training with a periodic progress panel."""
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
                draw_frame(stdscr, f" TRAINING ({label}) ")
                draw_field(stdscr, 2, 2, "DIFF: ", diff_name, "accent")
                draw_field(stdscr, 2, 20, "SHAPING: ", "ON" if shaping else "OFF",
                           "good" if shaping else "dim")
                draw_field(stdscr, 3, 2, "EPISODE: ", f"{ep}/{episodes}")
                draw_progress_bar(stdscr, 3, 26, ep / episodes, width=30)
                draw_field(stdscr, 4, 2, "AVG (last 50): ", f"{avg:.2f}")
                draw_field(stdscr, 4, 26, "BEST: ", f"{best}", "good")
                draw_field(stdscr, 5, 2, "EPS: ", f"{agent.epsilon:.3f}")
                draw_field(stdscr, 5, 26, "STATES: ", f"{len(agent.q)}")
                safe_addstr(stdscr, 6, 2, "SCORES: ", T("label"))
                safe_addstr(stdscr, 6, 10, sparkline(window), T("spark"))
                safe_addstr(stdscr, 8, 2, "PRESS Q TO ABORT", T("dim"))
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
            safe_addstr(stdscr, h - 2, 2, f"WARNING: could not save progress: {exc}", T("bad", curses.A_BOLD))
            stdscr.refresh()


# =========================
# END SCREENS
# =========================
def game_over_screen(stdscr, score):
    stdscr.nodelay(False)
    h, _ = stdscr.getmaxyx()
    draw_frame(stdscr)
    draw_center(stdscr, h // 2 - 2, "GAME OVER", T("bad", curses.A_BOLD))
    draw_center(stdscr, h // 2, f"SCORE: {score}", T("good", curses.A_BOLD))
    draw_center(stdscr, h // 2 + 2, "PLAY AGAIN? (Y/N)", T("accent"))
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k in (ord("y"), ord("Y")):
            return True
        if k in (ord("n"), ord("N"), 27):
            return False


def post_training_screen(stdscr, storage, best, aborted, turbo=False):
    stdscr.nodelay(False)
    title = " TRAINING COMPLETE (TURBO) " if turbo else " TRAINING COMPLETE (FAST) "
    draw_frame(stdscr, title)
    draw_field(stdscr, 3, 2, "BEST SCORE: ", str(best), "good")
    if aborted:
        draw_field(stdscr, 4, 2, "STATUS: ", "ABORTED BY USER", "bad")
    draw_field(stdscr, 5, 2, "LOG: ", storage.training_log_path, "text")
    draw_menu_line(stdscr, 7, "1  PLAY TRAINED AI NOW", True, x=2)
    draw_menu_line(stdscr, 8, "B  BACK", True, x=2)
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
def run_ui(stdscr, storage, color=True):
    global _THEME
    _THEME = Theme(enabled=color)

    hide_cursor()
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
    parser = argparse.ArgumentParser(
        prog="serpentos run", description="Play SerpentOS in the terminal."
    )
    parser.add_argument("--data-dir", default=None,
                        help=f"state directory (default: {core.DEFAULT_DATA_DIR})")
    parser.add_argument("--no-color", dest="color", action="store_false",
                        help="disable colour (also honours the NO_COLOR environment variable)")
    args = parser.parse_args(argv)

    storage = core.Storage(args.data_dir)
    try:
        with storage.lock(owner="ui"):
            curses.wrapper(run_ui, storage, args.color)
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
