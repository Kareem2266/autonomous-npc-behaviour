from autonomous_npc.envs.office_env import OfficeEnv


env = OfficeEnv(render_mode="human")
observation, info = env.reset()

print(f"Starting observation: {observation}")
episode_over = False
total_reward = 0

try:
    while not episode_over:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        episode_over = terminated or truncated
finally:
    print(f"Episode finished! Total reward: {total_reward}")
    env.close()