import argparse
import secrets
import socket
import time

from autonomous_npc.envs.office_env import OfficeEnv
from sb3_contrib import MaskablePPO
# from stable_baselines3 import PPO

STATION_NAMES = {
    (1, 5): "Desk",
    (8, 2): "Printer",
    (8, 8): "Bathroom",
    (3, 2): "Snack Area",
    (3, 8): "Coffee Area",
}

def send_state(connection, observation):
    agent_x, agent_y = map(int, observation["agent"])
    target_x, target_y = map(int, observation["target"])
    obstacles = observation["obstacle"].astype(int).flatten()

    message = (
        f"{agent_x},{agent_y},{target_x},{target_y},"
        f"{obstacles[0]},{obstacles[1]},{obstacles[2]},{obstacles[3]}\n"
    )
    connection.sendall(message.encode("utf-8"))

def send_route(connection, env):
    route = [env._desk_location, *env._task_route, env._desk_location]
    message = "ROUTE:" + ",".join(STATION_NAMES[tuple(stop)] for stop in route) + "\n"
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
        send_route(connection, env)
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
            reached_station = tuple(observation["target"]) != previous_target
            if reached_station:
                target_name = STATION_NAMES.get(tuple(observation["target"]), "unknown location")
                print(f"Target changed to {target_name}.")
            if reached_station or terminated:
                time.sleep(1.0)
            else:
                # At 2 Unity units per second, a one-cell move takes about 0.5 seconds.
                time.sleep(0.5)

            if terminated or truncated:
                print("Document delivered; episode finished.")
                break
    env.close()

if __name__ == "__main__":
    main()
