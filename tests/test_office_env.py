import numpy as np
import pytest

from autonomous_npc.envs.office_env import OfficeEnv

# Running tests using: python -m pytest -v

@pytest.mark.parametrize("invalid_size", [1, 0, -1])
def test_rejects_grid_sizes_smaller_than_two(invalid_size):
    with pytest.raises(ValueError, match="grid less than 2x2"):
        OfficeEnv(size=invalid_size)


@pytest.mark.parametrize("valid_size", [2, 5, 8])
def test_stores_valid_grid_size(valid_size):
    environment = OfficeEnv(size=valid_size)

    assert environment.size == valid_size


def test_reset_returns_observation_inside_declared_space():
    environment = OfficeEnv(size=5)

    observation, _ = environment.reset(seed=42)

    assert environment.observation_space.contains(observation)


@pytest.mark.parametrize("seed", range(10))
def test_reset_places_agent_and_target_in_different_cells(seed):
    environment = OfficeEnv(size=5)

    observation, _ = environment.reset(seed=seed)

    assert not np.array_equal(observation["agent"], observation["target"])


def test_reset_with_same_seed_reproduces_starting_positions():
    environment = OfficeEnv(size=5)

    first_observation, _ = environment.reset(seed=42)
    second_observation, _ = environment.reset(seed=42)

    assert np.array_equal(
        first_observation["agent"], second_observation["agent"]
    )
    assert np.array_equal(
        first_observation["target"], second_observation["target"]
    )


def test_reset_info_reports_manhattan_distance():
    environment = OfficeEnv(size=5)

    observation, info = environment.reset(seed=42)
    coordinate_difference = np.abs(
        observation["agent"] - observation["target"]
    )
    expected_distance = coordinate_difference.sum()

    assert info["distance"] == expected_distance
