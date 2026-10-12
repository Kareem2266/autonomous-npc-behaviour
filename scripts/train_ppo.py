from autonomous_npc.envs.office_env import OfficeEnv
from stable_baselines3 import PPO

def main():
    env = OfficeEnv()
    model = PPO("MultiInputPolicy",env, verbose=1)
    model.learn(total_timesteps=50_000)
    model.save("office_ppo_obstacle")
    observation, info = env.reset()

    print(f"Starting observation: {observation}")
    print(f"Starting info: {info}")

    env.close()

if __name__ == "__main__":
    main()