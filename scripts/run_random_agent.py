from autonomous_npc.envs.office_env import OfficeEnv


env = OfficeEnv(render_mode="human")
env.reset()

try:
    while True:
        action = env.action_space.sample()
        observation, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            env.close()
            break
finally:
    env.close()