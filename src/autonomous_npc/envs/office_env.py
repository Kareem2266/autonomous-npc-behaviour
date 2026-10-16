from enum import IntEnum
from typing import Optional
from collections import deque
from dataclasses import dataclass
import numpy as np
import gymnasium as gym
import pygame


class Action(IntEnum):
    NORTH = 0
    EAST = 1
    SOUTH = 2
    WEST = 3
    INTERACT = 4
    WAIT = 5

@dataclass
class AgentState:
    agent_id: str
    desk_name: str
    location: np.ndarray
    previous_location: np.ndarray
    target_location: np.ndarray
    current_station_location: np.ndarray
    task_route: list[np.ndarray]
    route_index: int = 0
    carrying_document: bool = False
    finished: bool = False

@dataclass(frozen=True)
class AgentConfig:
    agent_id: str
    task_route: tuple[str,...]
    desk_name: str | None = None

DEFAULT_AGENT_CONFIGS = (
    AgentConfig("worker_1", ("printer", "snack", "desk_1"), "desk_1"),
    AgentConfig("worker_2", ("coffee", "bathroom", "desk_2"), "desk_2"),
)

class OfficeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size: int = 10,
        render_mode: Optional[str] = None,
        window_size: int = 512,
        agent_configs: tuple[AgentConfig,...] | None = None,
    ):
        super().__init__()
        if size < 2:
            raise ValueError("You cannot have a grid less than 2x2")
        if render_mode not in (None, *self.metadata["render_modes"]):
            raise ValueError(f"Unsupported render mode: {render_mode}")
        if window_size < 1:
            raise ValueError("window_size must be positive")

        self.size = size
        self.render_mode = render_mode
        self.window_size = window_size
        self.window = None
        self.clock = None

        self._obstacle_locations = np.array(
            [[size // 2, size // 2], [size // 2, (size // 2 + 1) % size]],
            dtype=np.int32,
        )

        self._agent_configs = (
            None if agent_configs is None else tuple(agent_configs)
        )
        if self._agent_configs is not None and not self._agent_configs:
            raise ValueError("At least one agent config is required.")
        agent_count = 1 if self._agent_configs is None else len(self._agent_configs)
        self._desk_names = (
            ["desk_1"]
            if self._agent_configs is None
            else [
                config.desk_name or f"desk_{index + 1}"
                for index, config in enumerate(self._agent_configs)
            ]
        )
        if len(self._desk_names) != len(set(self._desk_names)):
            raise ValueError("Each agent must have a unique desk name.")
        
        if size == 10:
            self._printer_location = np.array([8, 2], dtype=np.int32)
            self._bathroom_location = np.array([8, 8], dtype=np.int32)
            self._snack_location = np.array([3, 2], dtype=np.int32)
            self._coffee_location = np.array([3, 8], dtype=np.int32)

            self._station_approach_locations = {
                tuple(self._printer_location): np.array([7, 2], dtype=np.int32),
                tuple(self._bathroom_location): np.array([7, 8], dtype=np.int32),
                tuple(self._snack_location): np.array([2, 2], dtype=np.int32),
                tuple(self._coffee_location): np.array([2, 8], dtype=np.int32),
            }

            # Keep the original desk position for one worker. With more
            # workers, spread desks along the left side and reserve the
            # neighboring cell as each desk's approach tile.
            desk_rows = [5] if agent_count == 1 else [4, 6, 3, 7, 1, 9, 0, 5]
            if agent_count > len(desk_rows):
                raise ValueError(
                    f"A {size}x{size} office supports at most "
                    f"{len(desk_rows)} desks with the current station layout."
                )
            desk_rows = desk_rows[:agent_count]
            desk_positions = [
                np.array([1, row], dtype=np.int32) for row in desk_rows
            ]
            for desk_name, desk_position in zip(self._desk_names, desk_positions):
                self._station_approach_locations[tuple(desk_position)] = np.array(
                    [2, int(desk_position[1])], dtype=np.int32
                )

        else:
            self._printer_location = np.array([size - 1, (size -1) // 2], dtype=np.int32)
            self._station_approach_locations = {}
            if agent_count > size:
                raise ValueError(
                    f"A {size}x{size} office supports at most {size} desks."
                )
            desk_rows = list(range(size))
            desk_positions = [
                np.array([0, row], dtype=np.int32) for row in desk_rows[:agent_count]
            ]

        self._desk_locations = {
            desk_name: position
            for desk_name, position in zip(self._desk_names, desk_positions)
        }
        self._desk_location = self._desk_locations[self._desk_names[0]]

        self._station_locations = {
            **self._desk_locations,
            "desk": self._desk_location,
            "printer": self._printer_location
        }

        if size == 10:
            self._station_locations.update({
                "bathroom": self._bathroom_location,
                "snack": self._snack_location,
                "coffee": self._coffee_location,
            })

        if self._agent_configs is not None:
            agent_ids = [
                agent_config.agent_id for agent_config in self._agent_configs
            ]
            if len(agent_ids) != len(set(agent_ids)):
                raise ValueError("Agent IDs must be unique.")

            self._agent_route_names = []
            for agent_config, desk_name in zip(
                self._agent_configs, self._desk_names
            ):
                route_names = tuple(
                    desk_name if station_name == "desk" else station_name
                    for station_name in agent_config.task_route
                )
                desk_stops = {
                    station_name
                    for station_name in route_names
                    if station_name in self._desk_locations
                }
                if desk_stops - {desk_name}:
                    raise ValueError(
                        f"{agent_config.agent_id} can only use its assigned "
                        f"desk {desk_name!r} in its route."
                    )
                if not route_names or route_names[-1] != desk_name:
                    route_names = (*route_names, desk_name)

                unknown_stations = (
                    set(route_names) - set(self._station_locations)
                )
                if unknown_stations:
                    raise ValueError(
                        f"Unknown stations for {agent_config.agent_id}: "
                        f"{sorted(unknown_stations)}"
                    )
                self._agent_route_names.append(route_names)

        state_configs = self._agent_configs
        if state_configs is None:
            state_configs = (AgentConfig("worker_1", ()),)

        self._agent_states = []
        for agent_index, agent_config in enumerate(state_configs):
            desk_name = self._desk_names[agent_index]
            route_names = (
                self._agent_route_names[agent_index]
                if self._agent_configs is not None
                else agent_config.task_route
            )
            self._agent_states.append(
                AgentState(
                    agent_id=agent_config.agent_id,
                    desk_name=desk_name,
                    location=np.array([-1, -1], dtype=np.int32),
                    previous_location=np.array([-1, -1], dtype=np.int32),
                    target_location=np.array([-1, -1], dtype=np.int32),
                    current_station_location=np.array([-1, -1], dtype=np.int32),
                    task_route=self._location_for_route(route_names),
                )
            )

        self._active_agent_index = 0

        # Defining what agent can observe
        self.observation_space = gym.spaces.Dict(
            {
                "agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "previous_agent": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "target": gym.spaces.Box(0, size - 1, shape=(2,), dtype=int),
                "obstacle": gym.spaces.Box(0, size - 1, shape=(2,2), dtype=int)
            }
        )

        self.action_space = gym.spaces.Discrete(len(Action))

        self._action_to_direction = {
            Action.NORTH : np.array([0,-1]),
            Action.EAST : np.array([1,0]),
            Action.SOUTH : np.array([0,1]),
            Action.WEST : np.array([-1,0])
        }

    @property
    def _agent_location(self):
        return self._agent_states[self._active_agent_index].location

    @_agent_location.setter
    def _agent_location(self, value):
        self._agent_states[self._active_agent_index].location = value

    @property
    def _previous_agent_location(self):
        return self._agent_states[self._active_agent_index].previous_location

    @_previous_agent_location.setter
    def _previous_agent_location(self, value):
        self._agent_states[self._active_agent_index].previous_location = value

    @property
    def _target_location(self):
        return self._agent_states[self._active_agent_index].target_location

    @_target_location.setter
    def _target_location(self, value):
        self._agent_states[self._active_agent_index].target_location = value

    @property
    def _current_station_location(self):
        return self._agent_states[self._active_agent_index].current_station_location

    @_current_station_location.setter
    def _current_station_location(self, value):
        self._agent_states[self._active_agent_index].current_station_location = value

    @property
    def _task_route(self):
        return self._agent_states[self._active_agent_index].task_route

    @_task_route.setter
    def _task_route(self, value):
        self._agent_states[self._active_agent_index].task_route = value

    @property
    def _route_index(self):
        return self._agent_states[self._active_agent_index].route_index

    @_route_index.setter
    def _route_index(self, value):
        self._agent_states[self._active_agent_index].route_index = value

    @property
    def _carrying_document(self):
        return self._agent_states[self._active_agent_index].carrying_document

    @_carrying_document.setter
    def _carrying_document(self, value):
        self._agent_states[self._active_agent_index].carrying_document = value

    def _approach_for_station(self, station):
        approach = self._station_approach_locations.get(tuple(station),station)
        return approach.copy()

    def _location_for_route(self, route):
        locations = []
        for name in route:
            locations.append(self._station_locations[name].copy())
        return locations

    # Using this to get environement internal state
    def _get_obs(self):
        return {"agent": self._agent_location,
                "previous_agent": self._previous_agent_location,
                "target": self._target_location,
                "obstacle": self._obstacle_locations}

    def _select_agent(self, agent_index):
        if agent_index < 0 or agent_index >= len(self._agent_states):
            raise IndexError(f"Invalid agent index: {agent_index}")
        self._active_agent_index = agent_index

    def get_agent_observation(self, agent_index):
        self._select_agent(agent_index)
        return self._get_obs()



    def _get_path_distance(self, start_location=None):
        start = tuple(
            self._agent_location if start_location is None else start_location
        )

        target = tuple(self._target_location)
        blocked_cells = {tuple(position) for position in self._obstacle_locations}
        blocked_cells.update(self._station_approach_locations.keys())
        queue = deque([(start,0)])
        visited = {start}

        while queue:
            current, distance = queue.popleft()
            if current == target:
                return distance

            for direction in self._action_to_direction.values():
                next_position = np.array(current) + direction
                if np.any(next_position < 0) or np.any(next_position >= self.size):
                    continue

                next_cell = tuple(next_position)
                if next_cell in blocked_cells or next_cell in visited:
                    continue

                visited.add(next_cell)
                queue.append((next_cell, distance + 1))
        return float("inf")

    # Find distance between agent and target
    def _get_info(self):
        return {"distance": self._get_path_distance()}

    def reset(self, seed=None, options=None):
        # seed self.np_random
        super().reset(seed=seed)

        self._active_agent_index = 0

        if self._agent_configs is None:
            if self.size == 10:
                task_routes = [
                    [self._printer_location, self._desk_location],
                    [self._bathroom_location, self._desk_location],
                    [self._snack_location, self._coffee_location, self._desk_location],
                    [self._snack_location, self._desk_location],
                    [self._coffee_location, self._desk_location],
                  ]
            else:
                task_routes = [[self._printer_location]]

            self._agent_states[0].task_route = task_routes[
                self.np_random.integers(len(task_routes))
            ]
        else:
            for agent_index, agent_state in enumerate(self._agent_states):
                agent_state.task_route = self._location_for_route(
                    self._agent_route_names[agent_index]
                )
        for agent_state in self._agent_states:
            agent_state.carrying_document = False
            agent_state.finished = False
            agent_state.route_index = 0
            agent_desk = self._desk_locations[agent_state.desk_name]
            agent_state.current_station_location = agent_desk.copy()
            agent_state.target_location = self._approach_for_station(
                agent_desk
            )

        all_cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
        ]
        reserved_cells = {
            tuple(location) for location in self._station_locations.values()
        }

        reserved_cells.update(
            tuple(approach)
            for approach in self._station_approach_locations.values()
        )


        obstacle_candidates = [
            cell for cell in all_cells if cell not in reserved_cells
        ]

        layout_found = False
        for _ in range (1000):
            indices = self.np_random.choice(
                len(obstacle_candidates), size = 2, replace=False
            )
            obstacles = np.asarray(
                [obstacle_candidates[index] for index in indices],
                dtype=np.int32,
            )
            self._obstacle_locations = obstacles

            routes_reachable = True
            for agent_index, agent_state in enumerate(self._agent_states):
                self._active_agent_index = agent_index
                agent_desk = self._desk_locations[agent_state.desk_name]
                route_stops = [agent_desk, *agent_state.task_route]
                approach_stops = [
                    self._approach_for_station(stop) for stop in route_stops
                ]

                for start, target in zip(approach_stops, approach_stops[1:]):
                    self._agent_location = start.copy()
                    self._target_location = target.copy()
                    if not np.isfinite(self._get_path_distance()):
                        routes_reachable = False
                        break

                if not routes_reachable:
                    break

            if not routes_reachable:
                continue

            self._active_agent_index = 0
            first_desk = self._desk_locations[
                self._agent_states[0].desk_name
            ]
            self._target_location = self._approach_for_station(first_desk)
            blocked_cells = {tuple(cell) for cell in obstacles}
            reachable_starts = []

            for cell in all_cells:
                if cell in blocked_cells or cell in reserved_cells:
                    continue

                self._agent_location = np.asarray(cell, dtype=np.int32)
                if np.isfinite(self._get_path_distance()):
                    reachable_starts.append(cell)

            if len(reachable_starts) < len(self._agent_states):
                continue

            start_indices = self.np_random.choice(
                len(reachable_starts),
                size=len(self._agent_states),
                replace=False,
            )

            for agent_index, start_index in enumerate(start_indices):
                agent_state = self._agent_states[agent_index]
                start = np.asarray(reachable_starts[start_index], dtype=np.int32)
                agent_state.location = start
                agent_state.previous_location = start.copy()
                agent_desk = self._desk_locations[agent_state.desk_name]
                agent_state.current_station_location = agent_desk.copy()
                agent_state.target_location = self._approach_for_station(
                    agent_desk
                )

            self._active_agent_index = 0
            layout_found = True
            break

        if not layout_found:
            raise ValueError("Could not create a reachable layout for all agents.")

        self._previous_agent_location = self._agent_location.copy()
        first_desk = self._desk_locations[
            self._agent_states[0].desk_name
        ]
        self._current_station_location = first_desk.copy()
        self._target_location = self._approach_for_station(first_desk)

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def action_masks(self, agent_index=None):
        if agent_index is not None:
            self._select_agent(agent_index)

        mask = np.ones(self.action_space.n, dtype=bool)

        if self._agent_states[self._active_agent_index].finished:
            mask[:] = False
            mask[Action.WAIT] = True
            return mask

        other_agent_cells = {
            tuple(agent_state.location)
            for index, agent_state in enumerate(self._agent_states)
            if index != self._active_agent_index
            and np.all(agent_state.location >= 0)
        }

        for action, direction in self._action_to_direction.items():
            next_location = self._agent_location + direction
            outside_grid = np.any(next_location < 0) or np.any(next_location >= self.size)
            hits_obstacle = any(
                np.array_equal(next_location, obstacle)
                for obstacle in self._obstacle_locations
            )
            hits_station = tuple(next_location) in self._station_approach_locations
            hits_agent = tuple(next_location) in other_agent_cells
            if outside_grid or hits_obstacle or hits_station or hits_agent:
                mask[action] = False
            elif not outside_grid and not hits_obstacle and not hits_station and not hits_agent:
                current_distance = self._get_path_distance()
                next_distance = self._get_path_distance(next_location)
                if next_distance >= current_distance:
                    mask[action] = False

        at_target = np.array_equal(self._agent_location, self._target_location)
        if at_target:
            # Stop on each target and require an explicit interaction before
            # the route can continue.
            for action in self._action_to_direction:
                mask[action] = False

        if self._carrying_document:
            mask[Action.INTERACT] = at_target
        else:
            mask[Action.INTERACT] = (
                at_target
                and np.array_equal(
                    self._current_station_location,
                    self._desk_locations[
                        self._agent_states[self._active_agent_index].desk_name
                    ],
                )
            )

        mask[Action.WAIT] = False
        if not np.any(mask):
            mask[Action.WAIT] = True


        return mask

    def step_simultaneous(self, actions):
        """Apply one action per agent from the same pre-move state."""
        actions = np.asarray(actions, dtype=np.int64).reshape(-1)
        agent_count = len(self._agent_states)
        if actions.shape != (agent_count,):
            raise ValueError(
                f"Expected {agent_count} actions, received {len(actions)}."
            )
        if np.any(actions < 0) or np.any(actions >= self.action_space.n):
            raise ValueError("Every action must be a valid Action value.")

        previous_active_agent_index = self._active_agent_index
        old_locations = [agent.location.copy() for agent in self._agent_states]
        old_previous_locations = [
            agent.previous_location.copy() for agent in self._agent_states
        ]
        old_finished = [agent.finished for agent in self._agent_states]
        active_agent_count = sum(not agent.finished for agent in self._agent_states)
        old_distances = []
        action_masks = []
        for agent_index in range(agent_count):
            self._active_agent_index = agent_index
            old_distances.append(self._get_path_distance())
            action_masks.append(self.action_masks())

        proposed_locations = [location.copy() for location in old_locations]
        hit_obstacles = [False] * agent_count
        valid_actions = [False] * agent_count

        for agent_index, action_value in enumerate(actions):
            agent = self._agent_states[agent_index]
            action = Action(int(action_value))
            valid_actions[agent_index] = action_masks[agent_index][action]

            if agent.finished or action == Action.WAIT:
                continue
            if action in self._action_to_direction:
                if valid_actions[agent_index]:
                    proposed_locations[agent_index] = np.clip(
                        old_locations[agent_index]
                        + self._action_to_direction[action],
                        0,
                        self.size - 1,
                    )
                else:
                    hit_obstacles[agent_index] = True

        destination_agents = {}
        for agent_index, (old_location, proposed_location) in enumerate(
            zip(old_locations, proposed_locations)
        ):
            if not np.array_equal(old_location, proposed_location):
                destination_agents.setdefault(tuple(proposed_location), []).append(
                    agent_index
                )
        for colliding_agents in destination_agents.values():
            if len(colliding_agents) > 1:
                for agent_index in colliding_agents:
                    proposed_locations[agent_index] = old_locations[agent_index]
                    hit_obstacles[agent_index] = True

        for agent_index, agent in enumerate(self._agent_states):
            if not np.array_equal(old_locations[agent_index], proposed_locations[agent_index]):
                agent.previous_location = old_locations[agent_index]
                agent.location = proposed_locations[agent_index]

        completed_this_step = [False] * agent_count
        for agent_index, agent in enumerate(self._agent_states):
            if (
                agent.finished
                or Action(int(actions[agent_index])) != Action.INTERACT
                or not valid_actions[agent_index]
                or not np.array_equal(agent.location, agent.target_location)
            ):
                continue

            agent_desk = self._desk_locations[agent.desk_name]
            if (
                not agent.carrying_document
                and np.array_equal(agent.current_station_location, agent_desk)
            ):
                agent.carrying_document = True
                agent.current_station_location = agent.task_route[
                    agent.route_index
                ].copy()
                agent.target_location = self._approach_for_station(
                    agent.current_station_location
                )
            elif agent.carrying_document:
                agent.route_index += 1
                if agent.route_index == len(agent.task_route):
                    agent.finished = True
                    completed_this_step[agent_index] = True
                else:
                    agent.current_station_location = agent.task_route[
                        agent.route_index
                    ].copy()
                    agent.target_location = self._approach_for_station(
                        agent.current_station_location
                    )

        rewards = []
        for agent_index, agent in enumerate(self._agent_states):
            if old_finished[agent_index]:
                rewards.append(0.0)
                continue
            self._active_agent_index = agent_index
            new_distance = self._get_path_distance()
            difference = old_distances[agent_index] - new_distance
            reward = 0.1 * difference - 0.01
            if (
                not np.array_equal(old_locations[agent_index], agent.location)
                and np.array_equal(
                    agent.location,
                    old_previous_locations[agent_index],
                )
            ):
                reward -= 0.05
            if hit_obstacles[agent_index]:
                reward -= 0.2
            if completed_this_step[agent_index]:
                reward += 1.0
            rewards.append(reward)

        self._active_agent_index = previous_active_agent_index
        terminated = all(agent.finished for agent in self._agent_states)
        info = {
            "distance": float(np.mean([
                self._path_distance_for_agent(index)
                for index in range(agent_count)
                if not self._agent_states[index].finished
            ])) if not terminated else 0.0,
            "agents_finished": {
                agent.agent_id: agent.finished
                for agent in self._agent_states
            },
            "agent_rewards": {
                agent.agent_id: rewards[index]
                for index, agent in enumerate(self._agent_states)
            },
        }

        if self.render_mode == "human":
            self.render()

        team_reward = sum(rewards) / max(1, active_agent_count)
        return None, float(team_reward), terminated, False, info

    def _path_distance_for_agent(self, agent_index):
        previous_active_agent_index = self._active_agent_index
        self._active_agent_index = agent_index
        distance = self._get_path_distance()
        self._active_agent_index = previous_active_agent_index
        return distance

    def step(self, action, agent_index=None):

        if agent_index is not None:
            self._select_agent(agent_index)

        active_agent = self._agent_states[self._active_agent_index]
        if active_agent.finished:
            observation = self._get_obs()
            info = self._get_info()
            info["agents_finished"] = {
                agent.agent_id: agent.finished
                for agent in self._agent_states
            }
            return (
                observation,
                0.0,
                all(agent.finished for agent in self._agent_states),
                False,
                info,
            )

        completed_this_step = False


        # rewards progress
        old_distance = self._get_info()["distance"]

        # Map tthe action element to the direction
        if action not in self._action_to_direction:
             direction = 0
        else:
            direction = self._action_to_direction[action]

        # use np.clip to stay in grid
        proposed_location = np.clip(
             self._agent_location + direction, 0, self.size - 1
        )

        reversed_step = not np.array_equal(proposed_location, self._agent_location) and np.array_equal(proposed_location, self._previous_agent_location)

        other_agent_cells = {
            tuple(agent_state.location)
            for index, agent_state in enumerate(self._agent_states)
            if index != self._active_agent_index
            and np.all(agent_state.location >= 0)
        }


        hit_obstacle = (
            any(
                np.array_equal(proposed_location, obstacle)
                for obstacle in self._obstacle_locations
            )
            or tuple(proposed_location) in self._station_approach_locations
            or tuple(proposed_location) in other_agent_cells
        )

        if not hit_obstacle and not np.array_equal(proposed_location, self._agent_location):
            self._previous_agent_location = self._agent_location.copy()
            self._agent_location = proposed_location

        new_distance = self._get_info()["distance"]

        # Reaching a target only gets the agent there. The route advances on
        # INTERACT, so station visits are represented in the environment.
        terminated = False

        if (
            action == Action.INTERACT
            and np.array_equal(self._agent_location, self._target_location)
        ):
            if (
                not self._carrying_document
                and np.array_equal(
                   self._current_station_location,
                    self._desk_locations[active_agent.desk_name],
                )
            ):
                self._carrying_document = True
                self._current_station_location = self._task_route[self._route_index].copy()
                self._target_location = self._approach_for_station(self._current_station_location)
            elif self._carrying_document:
                self._route_index += 1
                if self._route_index == len(self._task_route):
                    active_agent.finished = True
                    completed_this_step = True
                else:
                    self._current_station_location = self._task_route[self._route_index].copy()
                    self._target_location = self._approach_for_station(self._current_station_location)

        terminated = all(agent.finished for agent in self._agent_states)

        difference = old_distance - new_distance

        # Per step feedback
        reward = 0.1 * difference - 0.01

        if reversed_step:
            reward -= 0.05

        if hit_obstacle:
            reward -= 0.2

        if completed_this_step:
            reward += 1


        observation = self._get_obs()
        info = self._get_info()
        info["agents_finished"] = {
            agent.agent_id: agent.finished
            for agent in self._agent_states
        }

        if self.render_mode == "human":
            self.render()

        return observation, reward, terminated, False, info

    # Rendering
    def render(self):
        if self.render_mode is None:
            return None

        frame = self._render_frame()
        if self.render_mode == "rgb_array":
            return frame
        return None

    def _render_frame(self):
        if self.render_mode == "human" and self.window is None:
            pygame.init()
            pygame.display.init()
            self.window = pygame.display.set_mode(
                (self.window_size, self.window_size)
            )
        if self.render_mode == "human" and self.clock is None:
            self.clock = pygame.time.Clock()

        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))

        # Convert grid coordinates [x, y] to pixel coordinates without reversing axes.
        pix_square_size = self.window_size / self.size
        target_x, target_y = self._target_location
        left = round(target_x * pix_square_size)
        top = round(target_y * pix_square_size)
        right = round((target_x + 1) * pix_square_size)
        bottom = round((target_y + 1) * pix_square_size)
        pygame.draw.rect(
            canvas,
            (255, 0, 0),
            pygame.Rect(left, top, right - left, bottom - top),
        )

        for obstacle_x, obstacle_y in self._obstacle_locations:
            left = round(obstacle_x * pix_square_size)
            top = round(obstacle_y * pix_square_size)
            right = round((obstacle_x + 1) * pix_square_size)
            bottom = round((obstacle_y + 1) * pix_square_size)
            pygame.draw.rect(
                canvas,
                (80, 80, 80),
                pygame.Rect(left, top, right - left, bottom - top),
            )

        # Draw the agent at the center of its [x, y] grid cell.
        agent_x, agent_y = self._agent_location
        agent_center = (
            round((agent_x + 0.5) * pix_square_size),
            round((agent_y + 0.5) * pix_square_size),
        )
        pygame.draw.circle(
            canvas,
            (0, 0, 255),
            agent_center,
            max(1, int(pix_square_size * 0.3)),
        )

        # Draw horizontal and vertical gridlines.
        for grid_index in range(self.size + 1):
            pixel = round(grid_index * pix_square_size)
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (0, pixel),
                (self.window_size, pixel),
                width=1,
            )
            pygame.draw.line(
                canvas,
                (0, 0, 0),
                (pixel, 0),
                (pixel, self.window_size),
                width=1,
            )

        if self.render_mode == "human":
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return None

            self.window.blit(canvas, (0, 0))
            pygame.display.flip()
            self.clock.tick(self.metadata["render_fps"])
            return None

        # Pygame arrays are indexed [x, y, channel]; Gymnasium expects [y, x, channel].
        return pygame.surfarray.array3d(canvas).transpose(1, 0, 2).copy()

    def close(self):
        if self.window is not None:
            pygame.display.quit()
            self.window = None
            self.clock = None

            
