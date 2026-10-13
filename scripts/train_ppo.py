from autonomous_npc.envs.office_env import OfficeEnv
from sb3_contrib import MaskablePPO
# from stable_baselines3 import PPO

def main():
    env = OfficeEnv()
    model = MaskablePPO("MultiInputPolicy",env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save("office_10x10_masked_ppo")
    observation, info = env.reset()

    print(f"Starting observation: {observation}")
    print(f"Starting info: {info}")

    env.close()

if __name__ == "__main__":
    main()