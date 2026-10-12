import socket
import time

from autonomous_npc.envs.office_env import OfficeEnv
from stable_baselines3 import PPO

def send_state(connection, observation):
    agent_x, agent_y = map(int, observation["agent"])
    target_x, target_y = map(int, observation["target"])

    message = f"{agent_x},{agent_y},{target_x},{target_y}\n"
    connection.sendall(message.encode("utf-8"))

def main():
    env = OfficeEnv()
    model = PPO.load("office_ppo_obstacle")
    observation, _ = env.reset(seed=42)

    # Unity must be in Play mode and listening before Python connects
    with socket.create_connection(("127.0.0.1", 5005)) as connection:
        send_state(connection, observation)
        # Let the scene agent reach the episode's initial grid position before
        # the policy sends its first move.
        time.sleep(2.0)

        for step in range(100):
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, terminated, truncated, info = env.step(
                action.item()
            )

            send_state(connection, observation)
            print(f"Step {step + 1}: agent at {observation['agent']}")
            # Give Unity enough time to finish a one-cell move before the next command.
            time.sleep(1.0)

            if terminated or truncated:
                print("Episode finished.")
                break
    env.close()

if __name__ == "__main__":
    main()
