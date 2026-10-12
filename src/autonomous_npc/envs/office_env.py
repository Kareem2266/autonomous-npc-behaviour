from enum import IntEnum
from typing import Optional
from collections import deque
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

class OfficeEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(
        self,
        size: int = 10,
        render_mode: Optional[str] = None,
        window_size: int = 512,
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

        # Initialize positions, will be randmsly set in reset (-1 is unitialized state for now)
        self._agent_location = np.array([-1,-1], dtype=np.int32)
        self._target_location = np.array([-1,-1], dtype=np.int32)
        self._obstacle_locations = np.array(
            [[size // 2, size // 2], [size // 2, (size // 2 + 1) % size]],
            dtype=np.int32)
        self._previous_agent_location = np.array([-1, -1], dtype=np.int32)
        
        if size == 10:
            self._desk_location = np.array([1, 5], dtype=np.int32)
            self._printer_location = np.array([8, 2], dtype=np.int32)
            self._bathroom_location = np.array([8, 8], dtype=np.int32)
            self._snack_location = np.array([3, 2], dtype=np.int32)
            self._coffee_location = np.array([3, 8], dtype=np.int32)
        else:
            self._desk_location = np.array([0, size // 2], dtype=np.int32)
            self._printer_location = np.array([size - 1, (size -1) // 2], dtype=np.int32)
        self._carrying_document = False
        self._task_route = []
        self._route_index = 0

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

    # Using this to get environement internal state
    def _get_obs(self):
        return {"agent": self._agent_location,
                "previous_agent": self._previous_agent_location,
                "target": self._target_location,
                "obstacle": self._obstacle_locations}

    def _get_path_distance(self):
        start = tuple(self._agent_location)
        target = tuple(self._target_location)
        obstacles = {tuple(position) for position in self._obstacle_locations}
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
                if next_cell in obstacles or next_cell in visited:
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
        self._carrying_document = False

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

        self._task_route = task_routes[self.np_random.integers(len(task_routes))]
        self._route_index = 0

        all_cells = [
            (x, y)
            for x in range(self.size)
            for y in range(self.size)
        ]
        if self.size == 10:
            reserved_cells = {
                tuple(self._desk_location),
                tuple(self._printer_location),
                tuple(self._bathroom_location),
                tuple(self._snack_location),
                tuple(self._coffee_location),
            }
        else:
            reserved_cells = {
                tuple(self._desk_location),
                tuple(self._printer_location),

            }
        obstacle_candidates = [
            cell for cell in all_cells if cell not in reserved_cells
        ]

        layout_found = False
        for _ in range(1000):
            indices = self.np_random.choice(
                len(obstacle_candidates), size=2, replace=False
            )
            obstacles = np.asarray(
                [obstacle_candidates[index] for index in indices],
                dtype=np.int32,
            )

            self._obstacle_locations = obstacles

            # Keep layouts where every leg of the selected task route is open.
            route_stops = [self._desk_location, *self._task_route]
            route_reachable = True

            for start, target in zip(route_stops, route_stops[1:]):
                self._agent_location = start.copy()
                self._target_location = target.copy()
                if not np.isfinite(self._get_path_distance()):
                    route_reachable = False
                    break

            if not route_reachable:
                continue

            blocked_cells = {tuple(cell) for cell in obstacles}
            reachable_starts = []
            for cell in all_cells:
                if cell in blocked_cells or cell == tuple(self._desk_location):
                    continue

                self._agent_location = np.asarray(cell, dtype=np.int32)
                self._target_location = self._desk_location.copy()
                if np.isfinite(self._get_path_distance()):
                    reachable_starts.append(cell)

            if reachable_starts:
                start_index = self.np_random.integers(len(reachable_starts))
                self._agent_location = np.asarray(
                    reachable_starts[start_index], dtype=np.int32
                )
                layout_found = True
                break

        if not layout_found:
            raise ValueError("Could not create a reachable layout with two obstacles.")


        self._previous_agent_location = self._agent_location.copy()
        self._target_location = self._desk_location.copy()

        observation = self._get_obs()
        info = self._get_info()

        if self.render_mode == "human":
            self.render()

        return observation, info

    def step(self, action):

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

        hit_obstacle = any(np.array_equal(proposed_location, obstacle) for obstacle in self._obstacle_locations)
        if not hit_obstacle and not np.array_equal(proposed_location, self._agent_location):
            self._previous_agent_location = self._agent_location.copy()
            self._agent_location = proposed_location

        new_distance = self._get_info()["distance"]

        # Check if it reached the target
        terminated = False

        if np.array_equal(self._agent_location, self._target_location):
            if not self._carrying_document and action == Action.INTERACT:
                self._carrying_document = True
                self._target_location = self._task_route[self._route_index].copy()
            elif self._carrying_document:
                self._route_index += 1
                if self._route_index == len(self._task_route):
                    terminated = True
                else:
                    self._target_location = self._task_route[self._route_index].copy()


        difference = old_distance - new_distance

        # Per step feedback
        reward = 0.1 * difference - 0.01

        if reversed_step:
            reward -= 0.05

        if hit_obstacle:
            reward -= 0.2

        if terminated:
            reward += 1


        observation = self._get_obs()
        info = self._get_info()

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

            
