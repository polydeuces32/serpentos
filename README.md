# SerpentOS

A terminal snake game with a built-in Q-learning AI. Watch the agent learn in real time, train it through thousands of episodes, and compare different learning strategies — all from your terminal with zero dependencies.

---

## Features

- **Human mode** — classic snake with arrow keys or WASD
- **AI mode** — tabular Q-learning agent with a live thinking HUD
- **Fast & Turbo training** — run hundreds or thousands of episodes without rendering
- **Hyperparameter presets** — swap learning profiles mid-session (DEFAULT / BOLD / CAREFUL / SPARSE)
- **Reward shaping toggle** — compare sparse vs. dense reward learning
- **Live sparkline** — ASCII score graph updates during training
- **CSV training log** — export every episode to `~/.serpentos/training_log.csv` for offline plotting
- **Persistent Q-table** — the agent remembers what it learned between sessions
- **Leaderboard** — top 10 scores saved locally
- **Difficulty levels** — Easy / Normal / Hard

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
python3 serpentos/serpentos.py
```

---

### Windows

> The game uses Python's `curses` library for terminal rendering. This is not available in standard Windows Command Prompt or PowerShell, so you have two options below.

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

If you'd rather stay in Windows natively, install the `windows-curses` package which adds curses support to standard Python on Windows.

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

## Controls

| Key | Action |
|-----|--------|
| Arrow keys / WASD | Move (human mode) |
| `H` | Toggle AI thinking HUD |
| `Q` | Quit current game |
| `Y` / `N` | Play again / return to menu |

---

## AI Training

### Presets
Press **C** in the AI submenu to open the config screen. Use the left/right arrow keys to cycle through presets:

| Preset | What it does |
|--------|-------------|
| DEFAULT | Balanced — good starting point |
| BOLD | High learning rate + exploration — learns fast, less stable |
| CAREFUL | Low learning rate — slow but steady convergence |
| SPARSE | No distance shaping — only food/death rewards |

Switching presets starts the agent fresh with the new hyperparameters.

### Training modes
- **Fast** — runs episodes without rendering, shows a live sparkline of scores
- **Turbo** — maximum speed, minimal UI updates, best for large episode counts

### Training log
Every episode is appended to `~/.serpentos/training_log.csv`:
```
episode, score, avg50, epsilon, mode
```
Open it in Excel, Google Sheets, or plot it with Python to visualise how the agent learns over time.

---

## Requirements

- Python 3.9 or higher
- macOS or Linux: no extra packages needed
- Windows: WSL **or** `pip install windows-curses`
