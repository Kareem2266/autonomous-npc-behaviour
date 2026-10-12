import argparse
import secrets
import socket
import time

from autonomous_npc.envs.office_env import OfficeEnv
from sb3_contrib import MaskablePPO
# from stable_baselines3 import PPO

def send_state(connection, observation):
    agent_x, agent_y = map(int, observation["agent"])
    target_x, target_y = map(int, observation["target"])
    obstacles = observation["obstacle"].astype(int).flatten()

    message = (
        f"{agent_x},{agent_y},{target_x},{target_y},"
        f"{obstacles[0]},{obstacles[1]},{obstacles[2]},{obstacles[3]}\n"
    )
    connection.sendall(message.encode("utf-8"))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="Use a fixed seed to replay a run; omit it for a random seed.",
    )
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(32)

    env = OfficeEnv()
    model = MaskablePPO.load("office_10x10_masked_ppo")
    observation, _ = env.reset(seed=seed)
    print(f"Episode seed: {seed}")

    # Unity must be in Play mode and listening before Python connects
    with socket.create_connection(("127.0.0.1", 5005)) as connection:
        send_state(connection, observation)
        time.sleep(1.0)

        for step in range(100):
            previous_target = tuple(observation["target"])
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            observation, reward, terminated, truncated, info = env.step(
                action.item()
            )

            send_state(connection, observation)
            print(f"Step {step + 1}: agent at {observation['agent']}")
            if tuple(observation["target"]) != previous_target:
                print("Document picked up; target switched to the printer.")
            # Give Unity enough time to finish a one-cell move before the next command.
            time.sleep(1.0)

            if terminated or truncated:
                print("Document delivered; episode finished.")
                break
    env.close()

if __name__ == "__main__":
    main()
