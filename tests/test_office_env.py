import numpy as np
import pytest

from autonomous_npc.envs.office_env import Action, OfficeEnv

# Running tests using: python -m pytest -v

@pytest.mark.parametrize(
    "invalid_size", [1, 0, -1], ids=["size=1", "size=0", "size=-1"]
)
def test_rejects_grid_sizes_smaller_than_two(invalid_size):
    with pytest.raises(ValueError, match="grid less than 2x2"):
        OfficeEnv(size=invalid_size)


@pytest.mark.parametrize(
    "valid_size", [2, 5, 8], ids=["size=2", "size=5", "size=8"]
)
def test_stores_valid_grid_size(valid_size):
    environment = OfficeEnv(size=valid_size)

    assert environment.size == valid_size


def test_reset_returns_observation_inside_declared_space():
    environment = OfficeEnv(size=5)

    observation, _ = environment.reset(seed=42)

    assert environment.observation_space.contains(observation)


@pytest.mark.parametrize("seed", range(10), ids=lambda seed: f"seed={seed}")
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


def test_reset_info_reports_shortest_walkable_distance():
    environment = OfficeEnv(size=5)

    observation, info = environment.reset(seed=42)
    coordinate_difference = np.abs(
        observation["agent"] - observation["target"]
    )
    expected_distance = coordinate_difference.sum()

    assert info["distance"] == expected_distance


@pytest.mark.parametrize("seed", range(20), ids=lambda seed: f"seed={seed}")
def test_reset_never_places_agent_or_target_on_obstacle(seed):
    environment = OfficeEnv(size=5)

    observation, _ = environment.reset(seed=seed)

    assert not np.array_equal(observation["agent"], observation["obstacle"])
    assert not np.array_equal(observation["target"], observation["obstacle"])


def test_path_distance_counts_route_around_obstacle():
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array([1, 2], dtype=np.int32)
    environment._target_location = np.array([3, 2], dtype=np.int32)

    assert environment._get_info()["distance"] == 4


def test_move_into_obstacle_is_blocked():
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array([1, 2], dtype=np.int32)
    environment._target_location = np.array([3, 2], dtype=np.int32)

    observation, reward, terminated, truncated, info = environment.step(Action.EAST)

    np.testing.assert_array_equal(observation["agent"], [1, 2])
    assert reward == pytest.approx(-0.01)
    assert info["distance"] == 4
    assert terminated is False
    assert truncated is False


@pytest.mark.parametrize(
    ("action", "expected_reward"),
    [(Action.EAST, 0.09), (Action.NORTH, -0.11)],
    ids=["moves-closer", "moves-farther"],
)
def test_reward_reflects_progress_toward_target(action, expected_reward):
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array([2, 2], dtype=np.int32)
    environment._target_location = np.array([4, 4], dtype=np.int32)

    _, reward, _, _, _ = environment.step(action)

    assert reward == pytest.approx(expected_reward)


@pytest.mark.parametrize(
    ("action", "expected_location", "expected_reward"),
    [
        (Action.NORTH, [2, 1], -0.11),
        (Action.EAST, [3, 2], 0.09),
        (Action.SOUTH, [2, 3], 0.09),
        (Action.WEST, [1, 2], -0.11),
    ],
    ids=["north", "east", "south", "west"],
)
def test_movement_action_moves_agent_and_preserves_target(
    action, expected_location, expected_reward
):
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array([2, 2], dtype=np.int32)
    environment._target_location = np.array([4, 4], dtype=np.int32)

    observation, reward, terminated, truncated, info = environment.step(action)

    np.testing.assert_array_equal(observation["agent"], expected_location)
    np.testing.assert_array_equal(observation["target"], [4, 4])
    assert reward == pytest.approx(expected_reward)
    assert terminated is False
    assert truncated is False
    assert info["distance"] == 8 - sum(expected_location)


@pytest.mark.parametrize(
    ("action", "start_location"),
    [
        (Action.NORTH, [2, 0]),
        (Action.EAST, [4, 2]),
        (Action.SOUTH, [2, 4]),
        (Action.WEST, [0, 2]),
    ],
    ids=["north-edge", "east-edge", "south-edge", "west-edge"],
)
def test_movement_stays_within_grid_boundaries(action, start_location):
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array(start_location, dtype=np.int32)
    environment._target_location = np.array([4, 4], dtype=np.int32)

    observation, reward, _, _, _ = environment.step(action)

    np.testing.assert_array_equal(observation["agent"], start_location)
    assert reward == pytest.approx(-0.01)


@pytest.mark.parametrize(
    "action", [Action.INTERACT, Action.WAIT], ids=["interact", "wait"]
)
def test_non_movement_actions_leave_agent_in_place(action):
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array([2, 2], dtype=np.int32)
    environment._target_location = np.array([4, 4], dtype=np.int32)

    observation, reward, terminated, truncated, _ = environment.step(action)

    np.testing.assert_array_equal(observation["agent"], [2, 2])
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False


def test_reaching_target_gives_reward_and_terminates_episode():
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array([2, 2], dtype=np.int32)
    environment._target_location = np.array([3, 2], dtype=np.int32)

    observation, reward, terminated, truncated, _ = environment.step(Action.EAST)

    np.testing.assert_array_equal(observation["agent"], [3, 2])
    assert reward == pytest.approx(1.09)
    assert terminated is True
    assert truncated is False


def test_rgb_array_render_uses_xy_coordinates_and_returns_height_width_channels():
    environment = OfficeEnv(size=5, render_mode="rgb_array", window_size=100)
    environment._agent_location = np.array([1, 3], dtype=np.int32)
    environment._target_location = np.array([3, 1], dtype=np.int32)

    frame = environment.render()

    assert frame.shape == (100, 100, 3)
    assert frame.dtype == np.uint8
    np.testing.assert_array_equal(frame[30, 70], [255, 0, 0])
    np.testing.assert_array_equal(frame[70, 30], [0, 0, 255])


def test_render_returns_none_when_rendering_is_disabled():
    environment = OfficeEnv(size=5)

    assert environment.render() is None


def test_rejects_unsupported_render_mode():
    with pytest.raises(ValueError, match="Unsupported render mode"):
        OfficeEnv(size=5, render_mode="terminal")
