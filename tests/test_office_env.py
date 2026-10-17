import numpy as np
import pytest

from autonomous_npc.envs.office_env import Action, AgentConfig, OfficeEnv

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
    assert reward == pytest.approx(-0.21)
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


def test_moving_back_to_previous_cell_gets_reversal_penalty():
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array([1, 1], dtype=np.int32)
    environment._previous_agent_location = np.array([2, 1], dtype=np.int32)
    environment._target_location = np.array([4, 4], dtype=np.int32)

    observation, reward, _, _, _ = environment.step(Action.EAST)

    np.testing.assert_array_equal(observation["agent"], [2, 1])
    np.testing.assert_array_equal(observation["previous_agent"], [1, 1])
    assert reward == pytest.approx(0.04)


def test_wait_does_not_get_reversal_penalty_when_previous_cell_is_current_cell():
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array([1, 1], dtype=np.int32)
    environment._previous_agent_location = environment._agent_location.copy()
    environment._target_location = np.array([4, 4], dtype=np.int32)

    observation, reward, _, _, _ = environment.step(Action.WAIT)

    np.testing.assert_array_equal(observation["agent"], [1, 1])
    assert reward == pytest.approx(-0.01)


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
    environment._obstacle_locations = np.array(
        [[0, 0], [0, 4]], dtype=np.int32
    )

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
    environment._agent_location = np.array([1, 1], dtype=np.int32)
    environment._target_location = np.array([4, 4], dtype=np.int32)

    observation, reward, terminated, truncated, _ = environment.step(action)

    np.testing.assert_array_equal(observation["agent"], [1, 1])
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False


def test_reaching_target_then_interacting_terminates_episode():
    environment = OfficeEnv(size=5)
    environment._agent_location = np.array([1, 1], dtype=np.int32)
    environment._target_location = np.array([2, 1], dtype=np.int32)
    environment._carrying_document = True
    environment._task_route = [environment._target_location.copy()]
    environment._route_index = 0

    observation, reward, terminated, truncated, _ = environment.step(Action.EAST)

    np.testing.assert_array_equal(observation["agent"], [2, 1])
    np.testing.assert_array_equal(observation["target"], [2, 1])
    assert reward == pytest.approx(0.09)
    assert terminated is False
    assert truncated is False

    observation, reward, terminated, truncated, _ = environment.step(Action.INTERACT)

    np.testing.assert_array_equal(observation["agent"], [2, 1])
    assert reward == pytest.approx(0.99)
    assert terminated is True
    assert truncated is False


def test_interacting_at_desk_targets_printer_approach_cell():
    environment = OfficeEnv(size=10)
    desk_approach = environment._station_approach_locations[
        tuple(environment._desk_location)
    ]
    printer_approach = environment._station_approach_locations[
        tuple(environment._printer_location)
    ]
    environment._agent_location = desk_approach.copy()
    environment._target_location = desk_approach.copy()
    environment._current_station_location = environment._desk_location.copy()
    environment._task_route = [environment._printer_location.copy()]
    environment._route_index = 0

    observation, reward, terminated, truncated, info = environment.step(Action.INTERACT)

    np.testing.assert_array_equal(observation["agent"], desk_approach)
    np.testing.assert_array_equal(observation["target"], printer_approach)
    assert environment._carrying_document is True
    assert reward == pytest.approx(-0.01)
    assert terminated is False
    assert truncated is False
    assert info["distance"] > 0


def test_reaching_printer_with_document_ends_episode():
    environment = OfficeEnv(size=5)
    environment._agent_location = environment._printer_location.copy()
    environment._target_location = environment._printer_location.copy()
    environment._carrying_document = True
    environment._task_route = [environment._printer_location.copy()]
    environment._route_index = 0

    observation, reward, terminated, truncated, info = environment.step(Action.INTERACT)

    np.testing.assert_array_equal(observation["agent"], environment._printer_location)
    assert reward == pytest.approx(0.99)
    assert terminated is True
    assert truncated is False
    assert info["distance"] == 0


def test_reaching_intermediate_task_stop_updates_target_without_terminating():
    environment = OfficeEnv(size=10)
    environment._agent_location = environment._station_approach_locations[
        tuple(environment._snack_location)
    ].copy()
    environment._target_location = environment._station_approach_locations[
        tuple(environment._snack_location)
    ].copy()
    environment._carrying_document = True
    environment._current_station_location = environment._snack_location.copy()
    environment._task_route = [
        environment._snack_location.copy(),
        environment._coffee_location.copy(),
        environment._desk_location.copy(),
    ]
    environment._route_index = 0

    observation, _, terminated, truncated, _ = environment.step(Action.INTERACT)

    np.testing.assert_array_equal(
        observation["target"],
        environment._station_approach_locations[tuple(environment._coffee_location)],
    )
    np.testing.assert_array_equal(
        environment._current_station_location,
        environment._coffee_location,
    )
    assert environment._route_index == 1
    assert terminated is False
    assert truncated is False


def test_station_tiles_are_blocked_and_approach_tiles_are_walkable():
    environment = OfficeEnv(size=10)
    snack = environment._snack_location
    snack_approach = environment._station_approach_locations[tuple(snack)]
    environment._agent_location = snack_approach.copy()
    environment._target_location = np.array([9, 9], dtype=np.int32)

    assert not environment.action_masks()[Action.EAST]

    observation, reward, terminated, truncated, _ = environment.step(Action.EAST)

    np.testing.assert_array_equal(observation["agent"], snack_approach)
    assert reward == pytest.approx(-0.21)
    assert terminated is False
    assert truncated is False

    environment._agent_location = np.array([1, 2], dtype=np.int32)
    environment._target_location = snack_approach.copy()
    assert environment.action_masks()[Action.EAST]


def test_reset_chooses_a_valid_route_and_keeps_obstacles_off_stations_and_approaches():
    environment = OfficeEnv(size=10)

    observation, _ = environment.reset(seed=42)

    valid_routes = [
        [tuple(environment._printer_location), tuple(environment._desk_location)],
        [tuple(environment._bathroom_location), tuple(environment._desk_location)],
        [
            tuple(environment._snack_location),
            tuple(environment._coffee_location),
            tuple(environment._desk_location),
        ],
        [tuple(environment._snack_location), tuple(environment._desk_location)],
        [tuple(environment._coffee_location), tuple(environment._desk_location)],
    ]
    selected_route = [tuple(location) for location in environment._task_route]
    stations = {
        tuple(environment._desk_location),
        tuple(environment._printer_location),
        tuple(environment._bathroom_location),
        tuple(environment._snack_location),
        tuple(environment._coffee_location),
    }
    approach_cells = {
        tuple(location)
        for location in environment._station_approach_locations.values()
    }
    obstacle_cells = {tuple(location) for location in environment._obstacle_locations}

    assert selected_route in valid_routes
    assert environment._route_index == 0
    assert tuple(observation["target"]) == tuple(
        environment._station_approach_locations[tuple(environment._desk_location)]
    )
    assert obstacle_cells.isdisjoint(stations | approach_cells)
    assert tuple(observation["agent"]) not in stations | approach_cells


def test_multi_agent_reset_assigns_each_agent_a_distinct_reserved_desk():
    environment = OfficeEnv(
        size=10,
        agent_configs=(
            AgentConfig("worker_1", ("printer",)),
            AgentConfig("worker_2", ("coffee",)),
        ),
    )

    environment.reset(seed=42)

    desks = [
        environment._desk_locations[agent.desk_name]
        for agent in environment._agent_states
    ]
    desk_cells = {tuple(desk) for desk in desks}
    desk_approaches = {
        tuple(environment._station_approach_locations[tuple(desk)])
        for desk in desks
    }
    obstacle_cells = {
        tuple(obstacle) for obstacle in environment._obstacle_locations
    }

    assert len(environment._agent_states) == 2
    assert len(desk_cells) == 2
    assert desk_cells.isdisjoint(desk_approaches)
    assert obstacle_cells.isdisjoint(desk_cells | desk_approaches)

    for agent in environment._agent_states:
        assigned_desk = environment._desk_locations[agent.desk_name]
        np.testing.assert_array_equal(agent.current_station_location, assigned_desk)
        np.testing.assert_array_equal(agent.task_route[-1], assigned_desk)


def test_agent_cannot_be_configured_to_return_to_another_agents_desk():
    with pytest.raises(ValueError, match="only use its assigned desk"):
        OfficeEnv(
            size=10,
            agent_configs=(
                AgentConfig("worker_1", ("desk_2",)),
                AgentConfig("worker_2", ("printer",)),
            ),
        )


def test_multi_agent_completion_info_reports_each_agents_status():
    environment = OfficeEnv(
        size=10,
        agent_configs=(
            AgentConfig("worker_1", ("printer",)),
            AgentConfig("worker_2", ("coffee",)),
        ),
    )
    environment.reset(seed=42)

    first_agent = environment._agent_states[0]
    first_desk = environment._desk_locations[first_agent.desk_name]
    first_desk_approach = environment._approach_for_station(first_desk)
    first_agent.location = first_desk_approach.copy()
    first_agent.target_location = first_desk_approach.copy()
    first_agent.current_station_location = first_desk.copy()
    first_agent.task_route = [first_desk.copy()]
    first_agent.route_index = 0
    first_agent.carrying_document = True

    _, _, terminated, _, info = environment.step(Action.INTERACT, agent_index=0)

    assert terminated is False
    assert info["agents_finished"] == {
        "worker_1": True,
        "worker_2": False,
    }


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
