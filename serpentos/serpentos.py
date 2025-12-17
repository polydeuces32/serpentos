#!/usr/bin/env python3
import curses
import time
import random
import math
import json
import os
from datetime import datetime

# =========================
# PATHS (persistent data)
# =========================
DATA_DIR = os.path.join(os.path.expanduser("~"), ".serpentos")
QTABLE_PATH = os.path.join(DATA_DIR, "qtable.json")
LEADERBOARD_PATH = os.path.join(DATA_DIR, "leaderboard.json")


# =========================
# DIFFICULTY
# =========================
DIFFICULTY = {
    "easy":   {"speed": 0.13, "name": "EASY"},
    "normal": {"speed": 0.10, "name": "NORMAL"},
    "hard":   {"speed": 0.07, "name": "HARD"},
}


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


# =========================
# UTIL: data dir
# =========================
def ensure_data_dir():
    try:
        os.makedirs(DATA_DIR, exist_ok=True)
    except Exception:
        pass


# =========================
# PERSISTENCE: QTABLE
# =========================
def load_qtable():
    ensure_data_dir()
    if not os.path.exists(QTABLE_PATH):
        return {}
    try:
        with open(QTABLE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}

def save_qtable(q):
    ensure_data_dir()
    try:
        with open(QTABLE_PATH, "w", encoding="utf-8") as f:
            json.dump(q, f)
    except Exception:
        pass


# =========================
# PERSISTENCE: LEADERBOARD
# =========================
def load_leaderboard():
    ensure_data_dir()
    if not os.path.exists(LEADERBOARD_PATH):
        return []
    try:
        with open(LEADERBOARD_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
            if isinstance(data, list):
                return data
            return []
    except Exception:
        return []

def save_leaderboard(items):
    ensure_data_dir()
    try:
        with open(LEADERBOARD_PATH, "w", encoding="utf-8") as f:
            json.dump(items, f)
    except Exception:
        pass

def add_score_to_leaderboard(score, mode, diff_name):
    items = load_leaderboard()
    items.append({
        "score": int(score),
        "mode": mode,
        "diff": diff_name,
        "ts": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    })
    items.sort(key=lambda x: x["score"], reverse=True)
    items = items[:10]
    save_leaderboard(items)


# =========================
# BOOT ANIMATION (yellow dotted ring)
# =========================
def boot_animation(stdscr):
    curses.curs_set(0)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)

    h, w = stdscr.getmaxyx()
    cy, cx = h // 2, w // 2
    r = max(6, min(h, w) // 6)

    dots = 28
    trail = 10
    frames = 70
    chars = [".", ":", "o", "O", "0"]

    pts = []
    for i in range(dots):
        a = 2 * math.pi * i / dots
        y = int(cy + math.sin(a) * r)
        x = int(cx + math.cos(a) * r * 2)
        pts.append((y, x))

    for t in range(frames):
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
        safe_addstr(stdscr, cy + r + 2, cx - len(title)//2, title, curses.color_pair(1) | curses.A_BOLD)
        safe_addstr(stdscr, cy + r + 3, cx - len(sub)//2, sub, curses.color_pair(1))
        stdscr.refresh()
        time.sleep(0.04)

    stdscr.clear()
    stdscr.refresh()


# =========================
# MENU HELPERS
# =========================
def draw_center(stdscr, y, text, attr=0):
    h, w = stdscr.getmaxyx()
    x = max(0, (w // 2) - (len(text) // 2))
    safe_addstr(stdscr, y, x, text, attr)

def main_menu(stdscr):
    stdscr.nodelay(False)
    stdscr.clear()
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)

    lines = [
        "1  HUMAN SNAKE",
        "2  AI SNAKE (Q-LEARNING)",
        "3  LEADERBOARD",
        "",
        "Q  QUIT",
    ]
    h, _ = stdscr.getmaxyx()
    y0 = h // 2 - len(lines) // 2
    for i, ln in enumerate(lines):
        bold = curses.A_BOLD if ln and (ln[0].isdigit() or ln[0] in "Q") else 0
        draw_center(stdscr, y0 + i, ln, curses.color_pair(1) | bold)
    stdscr.refresh()

    while True:
        k = stdscr.getch()
        if k in (ord("1"), ord("2"), ord("3"), ord("q"), ord("Q")):
            return chr(k).lower()

def ai_submenu(stdscr):
    stdscr.nodelay(False)
    stdscr.clear()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)

    lines = [
        "AI SNAKE (Q-LEARNING)",
        "",
        "1  PLAY (LIVE HUD)",
        "2  TRAIN FAST (NO RENDER) THEN PLAY",
        "3  TRAIN TURBO (MAX SPEED) THEN PLAY",
        "",
        "B  BACK",
    ]
    h, _ = stdscr.getmaxyx()
    y0 = h // 2 - len(lines) // 2
    for i, ln in enumerate(lines):
        bold = curses.A_BOLD if i in (0, 2, 3, 4, 6) else 0
        draw_center(stdscr, y0 + i, ln, curses.color_pair(1) | bold)
    stdscr.refresh()

    while True:
        k = stdscr.getch()
        if k in (ord("1"), ord("2"), ord("3"), ord("b"), ord("B")):
            return chr(k).lower()

def training_menu(stdscr, turbo=False):
    stdscr.nodelay(False)
    stdscr.clear()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)

    if not turbo:
        lines = [
            "TRAINING EPISODES (FAST)",
            "",
            "1  200",
            "2  500",
            "3  1000",
            "4  2000",
            "",
            "B  BACK",
        ]
        mapping = {"1": 200, "2": 500, "3": 1000, "4": 2000}
    else:
        lines = [
            "TRAINING EPISODES (TURBO)",
            "",
            "1  1000",
            "2  2000",
            "3  5000",
            "4  10000",
            "",
            "B  BACK",
        ]
        mapping = {"1": 1000, "2": 2000, "3": 5000, "4": 10000}

    h, _ = stdscr.getmaxyx()
    y0 = h // 2 - len(lines) // 2
    for i, ln in enumerate(lines):
        bold = curses.A_BOLD if i in (0, 2, 3, 4, 5, 7) else 0
        draw_center(stdscr, y0 + i, ln, curses.color_pair(1) | bold)
    stdscr.refresh()

    while True:
        k = stdscr.getch()
        if k in (ord("b"), ord("B")):
            return None
        ch = chr(k) if 0 <= k < 256 else ""
        if ch in mapping:
            return mapping[ch]

def difficulty_menu(stdscr):
    stdscr.clear()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    lines = [
        "SELECT DIFFICULTY",
        "",
        "1  EASY",
        "2  NORMAL",
        "3  HARD",
    ]
    h, _ = stdscr.getmaxyx()
    y0 = h // 2 - len(lines) // 2
    for i, ln in enumerate(lines):
        bold = curses.A_BOLD if i in (0, 2, 3, 4) else 0
        draw_center(stdscr, y0 + i, ln, curses.color_pair(1) | bold)
    stdscr.refresh()

    while True:
        k = stdscr.getch()
        if k in (ord("1"), ord("2"), ord("3")):
            idx = int(chr(k)) - 1
            return ["easy", "normal", "hard"][idx]

def leaderboard_view(stdscr):
    stdscr.nodelay(False)
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)

    items = load_leaderboard()

    while True:
        stdscr.clear()
        stdscr.box()
        safe_addstr(stdscr, 0, 2, " LEADERBOARD (TOP 10) ", curses.color_pair(1) | curses.A_BOLD)

        if not items:
            draw_center(stdscr, 3, "NO SCORES YET", curses.color_pair(1))
        else:
            y = 2
            for i, it in enumerate(items, start=1):
                line = f"{i:>2}. {it['score']:>4}  {it['mode']:<10}  {it['diff']:<6}  {it['ts']}"
                safe_addstr(stdscr, y + i, 2, line, curses.color_pair(1))

        draw_center(stdscr, stdscr.getmaxyx()[0]-2, "PRESS B TO GO BACK", curses.color_pair(1))
        stdscr.refresh()

        k = stdscr.getch()
        if k in (ord("b"), ord("B"), 27):
            return


# =========================
# Q-LEARNING AGENT (TABULAR)
# =========================
class QAgent:
    def __init__(self):
        self.q = load_qtable()  # dict: key -> [q0,q1,q2]
        self.alpha = 0.15
        self.gamma = 0.95
        self.epsilon = 0.25
        self.eps_min = 0.05
        self.eps_decay = 0.995

        self.episodes = 0
        self.total_food = 0

        self.last_action = None
        self.last_reward = 0.0
        self.last_qvals = [0.0, 0.0, 0.0]

    def _key_str(self, state_tuple):
        return "|".join(map(str, state_tuple))

    def get_qvals(self, state_tuple):
        k = self._key_str(state_tuple)
        if k not in self.q:
            self.q[k] = [0.0, 0.0, 0.0]
        return self.q[k]

    def pick_action(self, state_tuple):
        qvals = self.get_qvals(state_tuple)
        self.last_qvals = qvals[:]

        if random.random() < self.epsilon:
            a = random.randint(0, 2)
        else:
            m = max(qvals)
            best = [i for i, v in enumerate(qvals) if v == m]
            a = random.choice(best)

        self.last_action = a
        return a

    def update_from_transition(self, s, a, r, s2, done):
        qvals = self.get_qvals(s)
        next_q = self.get_qvals(s2)
        target = r
        if not done:
            target += self.gamma * max(next_q)
        qvals[a] = (1 - self.alpha) * qvals[a] + self.alpha * target

    def end_episode(self, food_eaten):
        self.episodes += 1
        self.total_food += food_eaten

        if self.epsilon > self.eps_min:
            self.epsilon *= self.eps_decay
            if self.epsilon < self.eps_min:
                self.epsilon = self.eps_min

        save_qtable(self.q)

        self.last_action = None
        self.last_reward = 0.0
        self.last_qvals = [0.0, 0.0, 0.0]


# =========================
# GAME HELPERS
# =========================
DIRS = ["U", "R", "D", "L"]
VEC = {"U": (-1, 0), "D": (1, 0), "L": (0, -1), "R": (0, 1)}
DIR_IDX = {"U": 0, "R": 1, "D": 2, "L": 3}

def turn_left(d):
    return DIRS[(DIRS.index(d) - 1) % 4]

def turn_right(d):
    return DIRS[(DIRS.index(d) + 1) % 4]

def rel_action_to_abs_dir(cur_dir, rel_action):
    if rel_action == 0:
        return cur_dir
    if rel_action == 1:
        return turn_left(cur_dir)
    return turn_right(cur_dir)

def sign(v):
    return -1 if v < 0 else (1 if v > 0 else 0)

def is_collision(pos, snake, box):
    y, x = pos
    if y <= box[0][0] or y >= box[1][0] or x <= box[0][1] or x >= box[1][1]:
        return True
    if [y, x] in snake:
        return True
    return False

def build_state(snake, food, box, cur_dir):
    head = snake[0]
    dy = food[0] - head[0]
    dx = food[1] - head[1]

    ahead_dir = cur_dir
    left_dir = turn_left(cur_dir)
    right_dir = turn_right(cur_dir)

    ay, ax = VEC[ahead_dir]
    ly, lx = VEC[left_dir]
    ry, rx = VEC[right_dir]

    danger_a = 1 if is_collision([head[0] + ay, head[1] + ax], snake, box) else 0
    danger_l = 1 if is_collision([head[0] + ly, head[1] + lx], snake, box) else 0
    danger_r = 1 if is_collision([head[0] + ry, head[1] + rx], snake, box) else 0

    return (sign(dx), sign(dy), danger_a, danger_l, danger_r, cur_dir)

# TURBO: compact integer state key (same info, faster hashing)
# dx,dy in {-1,0,1} -> map to {0,1,2}
# danger bits -> 3 bits
# dir -> 2 bits
def build_state_int(head_y, head_x, food_y, food_x, snake_set, box_y1, box_x1, box_y2, box_x2, cur_dir):
    dx = food_x - head_x
    dy = food_y - head_y
    sdx = 0 if dx < 0 else (2 if dx > 0 else 1)
    sdy = 0 if dy < 0 else (2 if dy > 0 else 1)

    d = DIR_IDX[cur_dir]
    # ahead / left / right
    if d == 0:   # U
        ay, ax = -1, 0;  ly, lx = 0, -1;  ry, rx = 0, 1
    elif d == 1: # R
        ay, ax = 0, 1;   ly, lx = -1, 0;  ry, rx = 1, 0
    elif d == 2: # D
        ay, ax = 1, 0;   ly, lx = 0, 1;   ry, rx = 0, -1
    else:        # L
        ay, ax = 0, -1;  ly, lx = 1, 0;   ry, rx = -1, 0

    def coll(ny, nx):
        if ny <= box_y1 or ny >= box_y2 or nx <= box_x1 or nx >= box_x2:
            return 1
        if (ny, nx) in snake_set:
            return 1
        return 0

    da = coll(head_y + ay, head_x + ax)
    dl = coll(head_y + ly, head_x + lx)
    dr = coll(head_y + ry, head_x + rx)

    # pack bits: sdx(2) sdy(2) danger(3) dir(2) -> 9 bits total
    # layout: [sdx:2][sdy:2][da:1][dl:1][dr:1][dir:2]
    key = (sdx << 7) | (sdy << 5) | (da << 4) | (dl << 3) | (dr << 2) | (d & 3)
    return key


# =========================
# HUD (AI THINKING)
# =========================
def draw_ai_hud(stdscr, agent, state_tuple, abs_dir, score, steps, diff_name, show=True):
    if not show:
        return
    sx, sy, da, dl, dr, cd = state_tuple
    q0, q1, q2 = agent.last_qvals if agent.last_qvals else [0.0, 0.0, 0.0]

    lines = [
        f"MODE: AI(Q)   DIFF: {diff_name}   SCORE: {score}   STEPS: {steps}",
        f"EPS: {agent.epsilon:.3f}  ALPHA: {agent.alpha:.2f}  GAMMA: {agent.gamma:.2f}  EPISODES: {agent.episodes}",
        f"STATE: dx={sx} dy={sy}  danger(a,l,r)=({da},{dl},{dr})  dir={cd}",
        f"ACTION: rel={agent.last_action} abs={abs_dir}  REWARD: {agent.last_reward:+.2f}",
        f"Q: straight={q0:+.2f}  left={q1:+.2f}  right={q2:+.2f}",
        "HUD: H toggle   Q quit",
    ]
    y = 1
    x = 2
    for i, ln in enumerate(lines):
        safe_addstr(stdscr, y + i, x, ln, curses.color_pair(1) | (curses.A_BOLD if i == 0 else 0))


# =========================
# TRAINING (FAST, NO RENDER)
# =========================
def train_fast(stdscr, agent: QAgent, episodes: int, h: int, w: int, diff_name: str):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)

    stdscr.nodelay(True)
    box = [[1, 1], [h - 2, w - 2]]

    recent = []
    best = 0
    aborted = False

    for ep in range(1, episodes + 1):
        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            aborted = True
            break

        snake = [[h // 2, w // 2 + i] for i in range(4)]
        cur_dir = "L"
        score = 0
        steps = 0

        food = [random.randint(2, h - 3), random.randint(2, w - 3)]
        prev_dist = abs(food[0] - snake[0][0]) + abs(food[1] - snake[0][1])

        max_steps = 1800

        while steps < max_steps:
            steps += 1

            s = build_state(snake, food, box, cur_dir)
            a_rel = agent.pick_action(s)
            abs_dir = rel_action_to_abs_dir(cur_dir, a_rel)

            dy, dx = VEC[abs_dir]
            head = [snake[0][0] + dy, snake[0][1] + dx]

            r = -0.02
            done = False

            if is_collision(head, snake, box):
                r = -10.0
                done = True
                s2 = s
            else:
                snake.insert(0, head)

                if head == food:
                    score += 1
                    r = 10.0
                    food = [random.randint(2, h - 3), random.randint(2, w - 3)]
                else:
                    snake.pop()

                dist = abs(food[0] - snake[0][0]) + abs(food[1] - snake[0][1])
                if dist < prev_dist:
                    r += 0.05
                elif dist > prev_dist:
                    r -= 0.03
                prev_dist = dist

                s2 = build_state(snake, food, box, abs_dir)

            agent.last_reward = r
            agent.update_from_transition(s, a_rel, r, s2, done)
            cur_dir = abs_dir

            if done:
                break

        agent.end_episode(score)

        recent.append(score)
        if len(recent) > 50:
            recent.pop(0)
        avg = sum(recent) / max(1, len(recent))
        best = max(best, score)

        if ep == 1 or ep % 10 == 0 or ep == episodes:
            stdscr.clear()
            stdscr.box()
            safe_addstr(stdscr, 0, 2, " TRAINING (FAST) ", curses.color_pair(1) | curses.A_BOLD)
            safe_addstr(stdscr, 2, 2, f"DIFF: {diff_name}", curses.color_pair(1))
            safe_addstr(stdscr, 3, 2, f"EPISODE: {ep}/{episodes}", curses.color_pair(1) | curses.A_BOLD)
            safe_addstr(stdscr, 4, 2, f"AVG (last 50): {avg:.2f}", curses.color_pair(1))
            safe_addstr(stdscr, 5, 2, f"BEST: {best}", curses.color_pair(1))
            safe_addstr(stdscr, 6, 2, f"EPS: {agent.epsilon:.3f}", curses.color_pair(1))
            safe_addstr(stdscr, 8, 2, "PRESS Q TO ABORT", curses.color_pair(1))
            stdscr.refresh()

    stdscr.nodelay(False)
    return best, aborted


# =========================
# TRAINING TURBO (MAX SPEED, NO CURSES IN STEP LOOP)
# =========================
def train_turbo(stdscr, agent: QAgent, episodes: int, h: int, w: int, diff_name: str):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)

    stdscr.nodelay(True)

    box_y1, box_x1 = 1, 1
    box_y2, box_x2 = h - 2, w - 2

    recent = []
    best = 0
    aborted = False

    # Turbo usually benefits from a bit more exploration early
    # (keeps your existing agent settings; we just let it run much faster)
    ui_every = 50  # reduce curses overhead

    for ep in range(1, episodes + 1):
        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            aborted = True
            break

        # local vars for speed
        head_y = h // 2
        head_x = w // 2
        # snake as list + set for O(1) collision
        snake = [(head_y, head_x), (head_y, head_x + 1), (head_y, head_x + 2), (head_y, head_x + 3)]
        snake_set = set(snake)

        cur_dir = "L"
        d = cur_dir

        score = 0
        steps = 0
        max_steps = 1600  # slightly tighter; turbo wants more episodes, shorter runs

        food_y = random.randint(2, h - 3)
        food_x = random.randint(2, w - 3)

        prev_dist = abs(food_y - head_y) + abs(food_x - head_x)

        done = False

        while steps < max_steps and not done:
            steps += 1

            s_key = build_state_int(head_y, head_x, food_y, food_x, snake_set,
                                    box_y1, box_x1, box_y2, box_x2, d)

            # Use a small wrapper state so we reuse your QAgent unchanged.
            # We store int keys as strings inside the same JSON qtable.
            # This preserves your architecture and persistence format.
            s = ("K", s_key)  # 2-tuple is enough to namespace keys

            a_rel = agent.pick_action(s)

            # compute abs dir without list lookups
            if a_rel == 0:
                nd = d
            elif a_rel == 1:
                # left
                if d == "U": nd = "L"
                elif d == "L": nd = "D"
                elif d == "D": nd = "R"
                else: nd = "U"
            else:
                # right
                if d == "U": nd = "R"
                elif d == "R": nd = "D"
                elif d == "D": nd = "L"
                else: nd = "U"

            if nd == "U":
                ny, nx = head_y - 1, head_x
            elif nd == "D":
                ny, nx = head_y + 1, head_x
            elif nd == "L":
                ny, nx = head_y, head_x - 1
            else:
                ny, nx = head_y, head_x + 1

            # collision check (walls + body)
            if ny <= box_y1 or ny >= box_y2 or nx <= box_x1 or nx >= box_x2 or (ny, nx) in snake_set:
                r = -10.0
                done = True
                s2 = s
                agent.last_reward = r
                agent.update_from_transition(s, a_rel, r, s2, True)
                break

            # move snake
            new_head = (ny, nx)
            snake.insert(0, new_head)
            snake_set.add(new_head)

            r = -0.02

            if ny == food_y and nx == food_x:
                score += 1
                r = 10.0
                # respawn food (allow overlap avoidance lightly)
                for _ in range(12):
                    fy = random.randint(2, h - 3)
                    fx = random.randint(2, w - 3)
                    if (fy, fx) not in snake_set:
                        food_y, food_x = fy, fx
                        break
                else:
                    food_y, food_x = random.randint(2, h - 3), random.randint(2, w - 3)
            else:
                tail = snake.pop()
                snake_set.discard(tail)

            # distance shaping
            dist = abs(food_y - ny) + abs(food_x - nx)
            if dist < prev_dist:
                r += 0.05
            elif dist > prev_dist:
                r -= 0.03
            prev_dist = dist

            # next state
            d = nd
            head_y, head_x = ny, nx

            s2_key = build_state_int(head_y, head_x, food_y, food_x, snake_set,
                                     box_y1, box_x1, box_y2, box_x2, d)
            s2 = ("K", s2_key)

            agent.last_reward = r
            agent.update_from_transition(s, a_rel, r, s2, False)

        agent.end_episode(score)

        recent.append(score)
        if len(recent) > 50:
            recent.pop(0)
        avg = sum(recent) / max(1, len(recent))
        if score > best:
            best = score

        if ep == 1 or ep % ui_every == 0 or ep == episodes:
            stdscr.clear()
            stdscr.box()
            safe_addstr(stdscr, 0, 2, " TRAINING (TURBO) ", curses.color_pair(1) | curses.A_BOLD)
            safe_addstr(stdscr, 2, 2, f"DIFF: {diff_name}", curses.color_pair(1))
            safe_addstr(stdscr, 3, 2, f"EPISODE: {ep}/{episodes}", curses.color_pair(1) | curses.A_BOLD)
            safe_addstr(stdscr, 4, 2, f"AVG (last 50): {avg:.2f}", curses.color_pair(1))
            safe_addstr(stdscr, 5, 2, f"BEST: {best}", curses.color_pair(1))
            safe_addstr(stdscr, 6, 2, f"EPS: {agent.epsilon:.3f}", curses.color_pair(1))
            safe_addstr(stdscr, 8, 2, "PRESS Q TO ABORT", curses.color_pair(1))
            stdscr.refresh()

    stdscr.nodelay(False)
    return best, aborted


# =========================
# GAME LOOPS
# =========================
def play_human(stdscr, speed, diff_name):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)

    h, w = stdscr.getmaxyx()
    box = [[1, 1], [h - 2, w - 2]]

    snake = [[h // 2, w // 2 + i] for i in range(4)]
    direction = "L"
    score = 0

    food = [random.randint(2, h - 3), random.randint(2, w - 3)]
    stdscr.nodelay(True)

    key_map = {
        curses.KEY_UP: "U",
        curses.KEY_DOWN: "D",
        curses.KEY_LEFT: "L",
        curses.KEY_RIGHT: "R",
        ord("w"): "U",
        ord("s"): "D",
        ord("a"): "L",
        ord("d"): "R",
    }

    while True:
        stdscr.clear()
        stdscr.box()
        safe_addstr(stdscr, 0, 2, f" HUMAN  DIFF: {diff_name}  SCORE: {score} ", curses.color_pair(1) | curses.A_BOLD)

        for y, x in snake:
            safe_addch(stdscr, y, x, "#", curses.color_pair(1))
        safe_addch(stdscr, food[0], food[1], "*", curses.color_pair(2))

        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            return score, True
        if k in key_map:
            direction = key_map[k]

        dy, dx = VEC[direction]
        head = [snake[0][0] + dy, snake[0][1] + dx]

        if is_collision(head, snake, box):
            return score, False

        snake.insert(0, head)
        if head == food:
            score += 1
            food = [random.randint(2, h - 3), random.randint(2, w - 3)]
        else:
            snake.pop()

        stdscr.refresh()
        time.sleep(speed)

def play_ai_qlearn(stdscr, speed, diff_name, agent: QAgent):
    curses.start_color()
    curses.use_default_colors()
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    curses.init_pair(2, curses.COLOR_RED, -1)

    h, w = stdscr.getmaxyx()
    box = [[1, 1], [h - 2, w - 2]]

    snake = [[h // 2, w // 2 + i] for i in range(4)]
    cur_dir = "L"
    score = 0
    steps = 0
    hud_on = True

    food = [random.randint(2, h - 3), random.randint(2, w - 3)]
    stdscr.nodelay(True)

    prev_dist = abs(food[0] - snake[0][0]) + abs(food[1] - snake[0][1])

    while True:
        steps += 1
        stdscr.clear()
        stdscr.box()

        state = build_state(snake, food, box, cur_dir)
        rel_action = agent.pick_action(state)
        abs_dir = rel_action_to_abs_dir(cur_dir, rel_action)

        k = stdscr.getch()
        if k in (ord("q"), ord("Q")):
            agent.end_episode(score)
            return score, True
        if k in (ord("h"), ord("H")):
            hud_on = not hud_on

        dy, dx = VEC[abs_dir]
        head = [snake[0][0] + dy, snake[0][1] + dx]

        reward = -0.02
        done = False

        if is_collision(head, snake, box):
            reward = -10.0
            done = True
            new_state = state
        else:
            snake.insert(0, head)

            if head == food:
                score += 1
                reward = 10.0
                food = [random.randint(2, h - 3), random.randint(2, w - 3)]
            else:
                snake.pop()

            dist = abs(food[0] - snake[0][0]) + abs(food[1] - snake[0][1])
            if dist < prev_dist:
                reward += 0.05
            elif dist > prev_dist:
                reward -= 0.03
            prev_dist = dist

            new_state = build_state(snake, food, box, abs_dir)

        agent.last_reward = reward
        agent.update_from_transition(state, rel_action, reward, new_state, done)

        for y, x in snake:
            safe_addch(stdscr, y, x, "#", curses.color_pair(1))
        safe_addch(stdscr, food[0], food[1], "*", curses.color_pair(2))

        draw_ai_hud(stdscr, agent, state, abs_dir, score, steps, diff_name, show=hud_on)

        stdscr.refresh()
        time.sleep(speed)

        cur_dir = abs_dir

        if done:
            agent.end_episode(score)
            return score, False


# =========================
# GAME OVER + REPLAY
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

def post_training_screen(stdscr, best, aborted, turbo=False):
    curses.init_pair(1, curses.COLOR_YELLOW, -1)
    stdscr.nodelay(False)
    stdscr.clear()
    stdscr.box()
    title = " TRAINING COMPLETE (TURBO) " if turbo else " TRAINING COMPLETE (FAST) "
    safe_addstr(stdscr, 0, 2, title, curses.color_pair(1) | curses.A_BOLD)
    safe_addstr(stdscr, 3, 2, f"BEST SCORE: {best}", curses.color_pair(1))
    if aborted:
        safe_addstr(stdscr, 4, 2, "STATUS: ABORTED BY USER", curses.color_pair(1))
    safe_addstr(stdscr, 6, 2, "1  PLAY TRAINED AI NOW", curses.color_pair(1) | curses.A_BOLD)
    safe_addstr(stdscr, 7, 2, "B  BACK", curses.color_pair(1) | curses.A_BOLD)
    stdscr.refresh()
    while True:
        k = stdscr.getch()
        if k == ord("1"):
            return "play"
        if k in (ord("b"), ord("B"), 27):
            return "back"


# =========================
# MAIN
# =========================
def main(stdscr):
    boot_animation(stdscr)
    agent = QAgent()

    while True:
        choice = main_menu(stdscr)

        if choice == "q":
            break

        if choice == "3":
            leaderboard_view(stdscr)
            continue

        diff_key = difficulty_menu(stdscr)
        speed = DIFFICULTY[diff_key]["speed"]
        diff_name = DIFFICULTY[diff_key]["name"]

        if choice == "1":
            score, quit_early = play_human(stdscr, speed, diff_name)
            add_score_to_leaderboard(score, "HUMAN", diff_name)
            if quit_early:
                continue
        else:
            sub = ai_submenu(stdscr)
            if sub == "b":
                continue

            if sub == "1":
                score, quit_early = play_ai_qlearn(stdscr, speed, diff_name, agent)
                add_score_to_leaderboard(score, "AI(Q)", diff_name)
                if quit_early:
                    continue

            elif sub == "2":
                eps = training_menu(stdscr, turbo=False)
                if eps is None:
                    continue
                h, w = stdscr.getmaxyx()
                best, aborted = train_fast(stdscr, agent, eps, h, w, diff_name)
                nxt = post_training_screen(stdscr, best, aborted, turbo=False)
                if nxt == "play":
                    score, quit_early = play_ai_qlearn(stdscr, speed, diff_name, agent)
                    add_score_to_leaderboard(score, "AI(Q)", diff_name)
                    if quit_early:
                        continue
                else:
                    continue

            elif sub == "3":
                eps = training_menu(stdscr, turbo=True)
                if eps is None:
                    continue
                h, w = stdscr.getmaxyx()
                best, aborted = train_turbo(stdscr, agent, eps, h, w, diff_name)
                nxt = post_training_screen(stdscr, best, aborted, turbo=True)
                if nxt == "play":
                    score, quit_early = play_ai_qlearn(stdscr, speed, diff_name, agent)
                    add_score_to_leaderboard(score, "AI(Q)", diff_name)
                    if quit_early:
                        continue
                else:
                    continue

        if not game_over_screen(stdscr, score):
            break


if __name__ == "__main__":
    curses.wrapper(main)

