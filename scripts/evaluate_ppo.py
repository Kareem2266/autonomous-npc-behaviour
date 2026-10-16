from collections import deque

import numpy as np

from autonomous_npc.envs.office_env import Action, DEFAULT_AGENT_CONFIGS, OfficeEnv
from autonomous_npc.envs.multi_agent_env import SimultaneousMultiAgentEnv
from sb3_contrib import MaskablePPO
# from stable_baselines3 import PPO

def main():
    office_env = OfficeEnv(agent_configs=DEFAULT_AGENT_CONFIGS)
    env = SimultaneousMultiAgentEnv(office_env)
    model = MaskablePPO.load(
        "office_simultaneous_multi_agent_masked_ppo",
        env=env,
    )
    
    successes = 0
    successful_steps = 0
    run_amount = 100

    for episode in range(run_amount):
        observation, info = env.reset(seed=episode)
        steps = 0
        recent_steps = deque(maxlen=12)
        terminated = False
        truncated = False
        while not terminated and not truncated:
            before = [
                {
                    "location": tuple(agent.location),
                    "target": tuple(agent.target_location),
                    "station": tuple(agent.current_station_location),
                    "route_index": agent.route_index,
                    "carrying": agent.carrying_document,
                    "finished": agent.finished,
                }
                for agent in office_env._agent_states
            ]
            action, _ = model.predict(
                observation,
                deterministic=True,
                action_masks=env.action_masks(),
            )
            observation, _, terminated, truncated, info = env.step(action)

            steps += 1
            after = [
                {
                    "location": tuple(agent.location),
                    "target": tuple(agent.target_location),
                    "station": tuple(agent.current_station_location),
                    "route_index": agent.route_index,
                    "carrying": agent.carrying_document,
                    "finished": agent.finished,
                }
                for agent in office_env._agent_states
            ]
            recent_steps.append((steps, np.asarray(action).reshape(-1).copy(), before, after))
        
        if terminated:
            successes += 1
            successful_steps += steps
        else:
            unfinished_agents = [
                agent_id
                for agent_id, finished in info["agents_finished"].items()
                if not finished
            ]
            print(f"Failed seed {episode}; unfinished agents: {unfinished_agents}")
            print("Final worker states:")
            for agent in office_env._agent_states:
                print(
                    f"  {agent.agent_id}: position={tuple(agent.location)}, "
                    f"target={tuple(agent.target_location)}, "
                    f"station={tuple(agent.current_station_location)}, "
                    f"route_index={agent.route_index}/{len(agent.task_route)}, "
                    f"carrying={agent.carrying_document}, finished={agent.finished}"
                )
            print("Last 12 simultaneous ticks:")
            for tick, actions, before, after in recent_steps:
                transitions = []
                for index, (old, new) in enumerate(zip(before, after)):
                    action_name = Action(int(actions[index])).name
                    transitions.append(
                        f"{office_env._agent_states[index].agent_id} {action_name}: "
                        f"{old['location']} -> {new['location']}; "
                        f"target {old['target']} -> {new['target']}; "
                        f"station={new['station']} route={new['route_index']} "
                        f"carrying={new['carrying']}"
                    )
                print(f"  tick {tick}: " + " | ".join(transitions))

        completed_agents = sum(info["agents_finished"].values())
        print(
            f"Episode {episode + 1}: all agents finished: {terminated}; "
            f"agents: {completed_agents}/{len(DEFAULT_AGENT_CONFIGS)}; "
            f"simultaneous ticks: {steps}"
        )
    print(f"Success rate: {successes}/{run_amount}")

    if successes:
        average_steps = successful_steps / successes
        print(f"Average steps: {average_steps:.1f}")
    else:
        print("No successful episodes.")

    env.close()


if __name__ == "__main__":
    main()
