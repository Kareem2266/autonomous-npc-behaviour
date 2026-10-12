from autonomous_npc.envs.office_env import OfficeEnv
from stable_baselines3 import PPO

def main():
    env = OfficeEnv()
    model = PPO.load("office_task_ppo", env=env)
    
    successes = 0
    successful_steps = 0
    successful_shortest_steps = 0
    run_amount = 100

    for episode in range(run_amount):
        observation, info = env.reset(seed=episode)
        shortest_steps = int(info["distance"])
        steps = 0
        terminated = False
        truncated = False
        trajectory = []
        while not terminated and not truncated and steps < 100:
           
            before = tuple(observation["agent"])
            previous_target = tuple(observation["target"])
            action, _ = model.predict(observation, deterministic=True)
            action = action.item()
           
            observation, _, terminated, truncated, info = env.step(action)
            after = tuple(observation["agent"])
            trajectory.append((before, action, after))

            if tuple(observation["target"]) != previous_target:
                shortest_steps += 1 + int(info["distance"])

            steps += 1
        
        if terminated:
            successes += 1
            successful_steps += steps
            successful_shortest_steps += shortest_steps
        else:
            print(f"Failed seed {episode}; target: {tuple(observation['target'])}")
            print("Last 12 moves (before, action number, after):", trajectory[-12:])

        print(f"Episode {episode + 1}: reached target: {terminated}; steps: {steps}")
    print(f"Success rate: {successes}/{run_amount}")

    if successes:
        average_steps = successful_steps / successes
        average_shortest = successful_shortest_steps / successes
        print(f"Average steps: {average_steps:.1f}")
        print(f"Average shortest route: {average_shortest:.1f}")
        print(f"Average extra steps: {average_steps - average_shortest:.1f}")
    else:
        print("No successful episodes.")

    env.close()


if __name__ == "__main__":
    main()