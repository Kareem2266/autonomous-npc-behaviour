import numpy as np
import pytest

from autonomous_npc.envs.office_env import Action, OfficeEnv
from scripts.run_random_agent import choose_action


@pytest.mark.parametrize(
    ("agent", "target", "expected_action"),
    [
        ([2, 2], [3, 2], Action.EAST),
        ([2, 2], [1, 2], Action.WEST),
        ([2, 2], [2, 1], Action.NORTH),
        ([2, 2], [2, 3], Action.SOUTH),
        ([2, 2], [3, 3], Action.EAST),
    ],
    ids=["east", "west", "north", "south", "x-first-on-diagonal"],
)
def test_choose_action_moves_toward_target(agent, target, expected_action):
    environment = OfficeEnv(size=5)
    observation = {
        "agent": np.array(agent, dtype=np.int32),
        "target": np.array(target, dtype=np.int32),
    }

    action = choose_action(observation, environment)

    assert action == expected_action
    assert environment.action_space.contains(action)
