import numpy as np

from autonomous_npc.envs.office_env import Action, AgentConfig, OfficeEnv
from autonomous_npc.envs.multi_agent_env import SimultaneousMultiAgentEnv


def make_two_worker_env(max_episode_steps=400):
    office_env = OfficeEnv(
        size=10,
        agent_configs=(
            AgentConfig("worker_1", ("printer",)),
            AgentConfig("worker_2", ("coffee",)),
        ),
    )
    return SimultaneousMultiAgentEnv(
        office_env,
        max_episode_steps=max_episode_steps,
    )


def test_joint_observation_and_action_mask_cover_both_workers():
    environment = make_two_worker_env()

    observation, _ = environment.reset(seed=42)

    assert environment.observation_space.contains(observation)
    assert observation["agents"].shape == (2, 2)
    assert observation["targets"].shape == (2, 2)
    assert environment.action_space.nvec.tolist() == [len(Action), len(Action)]
    assert environment.action_masks().shape == (2 * len(Action),)


def test_joint_step_moves_both_workers_from_the_same_snapshot():
    environment = make_two_worker_env()
    environment.reset(seed=42)
    workers = environment.office_env._agent_states
    environment.office_env._obstacle_locations = np.array(
        [[5, 5], [5, 6]], dtype=np.int32
    )

    workers[0].location = np.array([0, 0], dtype=np.int32)
    workers[0].previous_location = np.array([0, 1], dtype=np.int32)
    workers[0].target_location = np.array([1, 0], dtype=np.int32)
    workers[1].location = np.array([9, 9], dtype=np.int32)
    workers[1].previous_location = np.array([9, 8], dtype=np.int32)
    workers[1].target_location = np.array([8, 9], dtype=np.int32)

    observation, _, terminated, truncated, _ = environment.step(
        np.array([Action.EAST, Action.WEST])
    )

    np.testing.assert_array_equal(observation["agents"], [[1, 0], [8, 9]])
    assert terminated is False
    assert truncated is False


def test_workers_both_stop_when_their_simultaneous_moves_conflict():
    environment = make_two_worker_env()
    environment.reset(seed=42)
    workers = environment.office_env._agent_states
    environment.office_env._obstacle_locations = np.array(
        [[0, 9], [9, 0]], dtype=np.int32
    )

    workers[0].location = np.array([4, 5], dtype=np.int32)
    workers[0].previous_location = np.array([4, 6], dtype=np.int32)
    workers[0].target_location = np.array([5, 5], dtype=np.int32)
    workers[1].location = np.array([6, 5], dtype=np.int32)
    workers[1].previous_location = np.array([6, 6], dtype=np.int32)
    workers[1].target_location = np.array([5, 5], dtype=np.int32)

    observation, _, _, _, info = environment.step(
        np.array([Action.EAST, Action.WEST])
    )

    np.testing.assert_array_equal(observation["agents"], [[4, 5], [6, 5]])
    assert info["agent_rewards"]["worker_1"] < 0
    assert info["agent_rewards"]["worker_2"] < 0


def test_both_workers_can_finish_on_the_same_simultaneous_tick():
    environment = make_two_worker_env()
    environment.reset(seed=42)
    workers = environment.office_env._agent_states

    for worker in workers:
        desk = environment.office_env._desk_locations[worker.desk_name]
        approach = environment.office_env._approach_for_station(desk)
        worker.location = approach.copy()
        worker.previous_location = approach.copy()
        worker.target_location = approach.copy()
        worker.current_station_location = desk.copy()
        worker.task_route = [desk.copy()]
        worker.route_index = 0
        worker.carrying_document = True

    _, reward, terminated, truncated, info = environment.step(
        np.array([Action.INTERACT, Action.INTERACT])
    )

    assert reward > 0
    assert terminated is True
    assert truncated is False
    assert info["agents_finished"] == {
        "worker_1": True,
        "worker_2": True,
    }


def test_simultaneous_env_truncates_at_its_episode_step_limit():
    environment = make_two_worker_env(max_episode_steps=1)
    environment.reset(seed=42)

    actions = np.array([Action.WAIT, Action.WAIT])
    _, _, terminated, truncated, _ = environment.step(actions)

    assert terminated is False
    assert truncated is True
