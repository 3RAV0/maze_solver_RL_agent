import numpy as np
import gymnasium as gym
from gymnasium import spaces


DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


class MazeEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, walls: np.ndarray, start=(0, 0), goal=None):

        super().__init__()
        self.walls = walls
        self.height, self.width = walls.shape[:2]
        self.start_cell = start
        self.goal_cell = goal if goal is not None else (self.height - 1, self.width - 1)

        
        self.observation_space = spaces.Discrete(self.height * self.width)
        self.action_space = spaces.Discrete(4)

        self.max_steps = self.height * self.width * 4

        self._agent_pos = None  
        self._step_count = 0

    def _obs(self) -> int:
        
        y, x = self._agent_pos
        return y * self.width + x

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self._agent_pos = list(self.start_cell)
        self._step_count = 0
        return self._obs(), {}

    def step(self, action: int):
        y, x = self._agent_pos

        if self.walls[y, x, action] == 0:
            dy, dx = DIRECTIONS[action]
            ny, nx = y + dy, x + dx
            self._agent_pos = [ny, nx]

        self._step_count += 1

        terminated = (tuple(self._agent_pos) == self.goal_cell)
        truncated = (self._step_count >= self.max_steps)

        reward = -1.0
        if terminated:
            reward = 100.0

        return self._obs(), reward, terminated, truncated, {}
