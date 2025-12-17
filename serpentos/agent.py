import json
import os
from collections import defaultdict

MEMORY_PATH = os.path.expanduser("~/.serpentos_memory.json")


class LearningAgent:
    def __init__(self):
        self.episodes = 0
        self.total_steps = 0
        self.total_food = 0

        # memory[(dy, dx)] = [attempts, deaths]
        self.memory = defaultdict(lambda: [0, 0])

        self.load()

    def record_move(self, direction):
        self.memory[direction][0] += 1

    def record_death(self, direction):
        self.memory[direction][1] += 1

    def finish_episode(self, steps, food):
        self.episodes += 1
        self.total_steps += steps
        self.total_food += food
        self.save()

    def risk(self, direction):
        attempts, deaths = self.memory[direction]
        if attempts == 0:
            return 0.0
        return deaths / attempts

    def save(self):
        data = {
            "episodes": self.episodes,
            "total_steps": self.total_steps,
            "total_food": self.total_food,
            "memory": {str(k): v for k, v in self.memory.items()},
        }
        with open(MEMORY_PATH, "w") as f:
            json.dump(data, f)

    def load(self):
        if not os.path.exists(MEMORY_PATH):
            return

        with open(MEMORY_PATH) as f:
            data = json.load(f)

        self.episodes = data.get("episodes", 0)
        self.total_steps = data.get("total_steps", 0)
        self.total_food = data.get("total_food", 0)

        for k, v in data.get("memory", {}).items():
            self.memory[eval(k)] = v

