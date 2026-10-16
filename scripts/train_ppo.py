from autonomous_npc.envs.office_env import DEFAULT_AGENT_CONFIGS, OfficeEnv
from autonomous_npc.envs.multi_agent_env import SimultaneousMultiAgentEnv
from sb3_contrib import MaskablePPO
# from stable_baselines3 import PPO

def main():
    office_env = OfficeEnv(agent_configs=DEFAULT_AGENT_CONFIGS)
    env = SimultaneousMultiAgentEnv(office_env)
    model = MaskablePPO("MultiInputPolicy",env, verbose=1)
    model.learn(total_timesteps=200_000)
    model.save("office_simultaneous_multi_agent_masked_ppo")
    observation, info = env.reset()

    print(f"Starting observation: {observation}")
    print(f"Starting info: {info}")

    env.close()

if __name__ == "__main__":
    main()
