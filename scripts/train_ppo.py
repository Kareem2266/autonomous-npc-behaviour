import argparse
from pathlib import Path

from autonomous_npc.envs.office_env import DEFAULT_AGENT_CONFIGS, OfficeEnv
from autonomous_npc.envs.multi_agent_env import SimultaneousMultiAgentEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from sb3_contrib import MaskablePPO
# from stable_baselines3 import PPO


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--total-timesteps", type=int, default=200_000)
    parser.add_argument("--checkpoint-freq", type=int, default=10_000)
    parser.add_argument(
        "--model-path",
        default="office_simultaneous_multi_agent_masked_ppo",
        help="Path for the final model; periodic checkpoints go in models/checkpoints.",
    )
    args = parser.parse_args()

    office_env = OfficeEnv(agent_configs=DEFAULT_AGENT_CONFIGS)
    env = SimultaneousMultiAgentEnv(office_env)
    model = MaskablePPO("MultiInputPolicy", env, verbose=1)
    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path="models/checkpoints",
        name_prefix=Path(args.model_path).name,
    )
    model.learn(
        total_timesteps=args.total_timesteps,
        callback=checkpoint_callback,
    )
    model.save(args.model_path)
    observation, info = env.reset()

    print(f"Starting observation: {observation}")
    print(f"Starting info: {info}")

    env.close()

if __name__ == "__main__":
    main()
