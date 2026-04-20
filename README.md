# SerpentOS

A terminal-native snake simulation featuring a real Q-learning agent with live AI introspection.

SerpentOS is built for:
- CLI-first environments
- Nano-based workflows
- Deterministic reinforcement learning
- Zero external dependencies

## Features

- Human-controlled snake mode
- AI snake using true tabular Q-learning
- Training Turbo (10,000+ episodes without rendering)
- Live AI "thinking" HUD
- Persistent learning (Q-table saved locally)
- Difficulty levels (Easy / Normal / Hard)
- Terminal leaderboard
- Futuristic ASCII boot animation

## Controls

### Human Mode
- Arrow keys or WASD — move
- Q — quit

### AI Mode
- H — toggle AI HUD
- Q — quit

## Requirements

- Python 3.9+
- Unix terminal (Linux / macOS / WSL)
- No external libraries required

## Run

```bash
python3 serpentos.py

