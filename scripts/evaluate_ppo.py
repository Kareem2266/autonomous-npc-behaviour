from autonomous_npc.envs.office_env import OfficeEnv
from stable_baselines3 import PPO

def main():
    env = OfficeEnv()
    model = PPO.load("office_ppo", env=env)
    
    successes = 0
    successful_steps = 0

    for episode in range(20):
        observation, info = env.reset(seed=episode)
        steps = 0
        terminated = False
        truncated = False

        while not terminated and not truncated and steps < 100:
           action, _ = model.predict(observation, deterministic=True)
           action = action.item()
           observation, reward, terminated, truncated, info = env.step(action)
           steps += 1
        
        if terminated:
            successes += 1
            successful_steps += steps
        print(f"Episode {episode + 1}: reached target: {terminated}; steps: {steps}")
    print(f"Success rate: {successes}/20")
    
    if successes:
        print(f"Average steps on success: {successful_steps / successes:.1f}")

    env.close()


if __name__ == "__main__":
    main()