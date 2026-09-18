from enum import IntEnum
from typing import Optional
import numpy as np
import gymnasium as gym


class Action(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    INTERACT = 4
    WAIT = 5

class OfficeEnv(gym.Env):

    def __init__(self, size: int = 5):
        self.size = size # 5x5 grid by default

        # Initialize positions, will be randmsly set in reset (-1 is unitialized state for now)
        self._agent_location = np.array([-1,-1], dtype=np.int32)
        self._target_location = np.array([-1,-1], dtype=np.int32)

        # Defining what agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int)
            }
        )

        self.action_space = gym.spaces.Discrete(len(Action))
