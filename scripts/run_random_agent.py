from autonomous_npc.envs.office_env import OfficeEnv
from autonomous_npc.envs.office_env import Action
import numpy as np

def choose_action(observation):
    agent = observation["agent"]
    target = observation["target"]

    position = target-agent
    pos_x,pos_y = position[0], position[1]
    direction_x, direction_y = np.array([np.sign(pos_x),0]), np.array([0,np.sign(pos_y)])

    if pos_x != 0:
        direction = direction_x
    else:
        direction = direction_y

    return next(
        action
        for action, mapped_direction in env._action_to_direction.items()
        if np.array_equal(mapped_direction, direction)
    )

env = OfficeEnv(render_mode="human")
observation, info = env.reset()

print(f"Starting observation: {observation}")
episode_over = False
total_reward = 0

try:
    while not episode_over:
        action = choose_action(observation)
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
finally:
    print(f"Episode finished! Total reward: {total_reward}")
    env.close()

