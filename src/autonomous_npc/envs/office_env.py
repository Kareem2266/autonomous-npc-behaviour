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
        super().__init__()
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

        self.action_to_direction = {
            Action.NORTH : np.array([0,-1]),
            Action.EAST : np.array([1,0]),
            Action.SOUTH : np.array([0,1]),
            Action.WEST : np.array([-1,0])
        }

    # Using this to get environement internal state
    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    # Find distance between agent and target
    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }
