import argparse
import secrets
import socket
import time

import numpy as np
from sb3_contrib import MaskablePPO
from autonomous_npc.envs.office_env import DEFAULT_AGENT_CONFIGS, OfficeEnv
from autonomous_npc.envs.multi_agent_env import SimultaneousMultiAgentEnv

STATION_EVENTS = {
    (8, 2): "PRINTER",
    (3, 2): "SNACK",
    (3, 8): "COFFEE",
    (8, 8): "BATHROOM",
}


def send_state(connection, observation):
    """Send both workers, both goals, and the two obstacles as one TCP line."""
    values = ["MULTI"]
    for position, target in zip(observation["agents"], observation["targets"]):
        values.extend(map(str, (*map(int, position), *map(int, target))))
    values.extend(map(str, observation["obstacles"].astype(int).flatten()))
    connection.sendall((",".join(values) + "\n").encode("utf-8"))


def send_station_events(connection, before, office_env):
    for index, (previous, agent) in enumerate(zip(before, office_env._agent_states)):
        if previous["carrying"] and agent.route_index > previous["route_index"]:
            station = tuple(map(int, previous["station"]))
            event = STATION_EVENTS.get(station)
            if event is not None:
                connection.sendall(
                    f"EVENT,{index},{event}\n".encode("utf-8")
                )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seed",
        type=int,
        help="Use a fixed seed to replay a run; omit it for a random seed.",
    )
    parser.add_argument(
        "--model-path",
        default="models/multi_agent_fixed",
        help="Path to the trained simultaneous multi-agent MaskablePPO model.",
    )
    parser.add_argument("--port", type=int, default=5005)
    args = parser.parse_args()
    seed = args.seed if args.seed is not None else secrets.randbits(32)

    env = SimultaneousMultiAgentEnv(OfficeEnv(agent_configs=DEFAULT_AGENT_CONFIGS))
    model = MaskablePPO.load(args.model_path, env=env)
    observation, _ = env.reset(seed=seed)
    print(f"Episode seed: {seed}")

    # Unity must be in Play mode and listening before Python connects
    with socket.create_connection(("127.0.0.1", args.port)) as connection:
        send_state(connection, observation)
        time.sleep(1.0)

        for step in range(env.max_episode_steps):
            before = [
                {
                    "route_index": agent.route_index,
                    "carrying": agent.carrying_document,
                    "station": agent.current_station_location.copy(),
                }
                for agent in env.office_env._agent_states
            ]
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            observation, reward, terminated, truncated, info = env.step(
                np.asarray(action).reshape(-1)
            )

            send_state(connection, observation)
            send_station_events(connection, before, env.office_env)
            print(f"Tick {step + 1}: workers at {observation['agents'].tolist()}")
            # Unity moves one grid cell in roughly half a second at its default speed.
            time.sleep(0.5)

            if terminated or truncated:
                print(f"Episode finished; all workers completed: {terminated}")
                break
    env.close()

if __name__ == "__main__":
    main()
