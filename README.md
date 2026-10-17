# Autonomous NPC Behaviour

A small reinforcement learning project for teaching office workers to navigate a shared workspace and complete assigned tasks. It combines a Gymnasium environment, a shared MaskablePPO policy, training and evaluation scripts, and a Unity visualization.

## How the simulation works

The office is represented as a 10 by 10 grid. Each worker has a desk and an ordered route of stations. At the beginning of an episode, a worker interacts with their desk to pick up a document, travels to each assigned station, interacts there, and returns to their desk to finish. The default routes are:

- Worker 1: desk → printer → snack area → desk.
- Worker 2: desk → coffee area → bathroom → desk.

The environment chooses two obstacle locations and random worker starting positions for each episode, while checking that the routes remain reachable. Episodes end when both workers finish or the 400 tick limit is reached.

At each tick, the policy receives one joint observation containing both workers’ positions, previous positions, targets, document carrying state, completion state, and the obstacle locations. It chooses one action per worker at the same time. Actions are north, east, south, west, interact, or wait. Action masks remove moves that are currently invalid and allow controlled detours when the direct route is blocked. Collision handling resolves shared destinations and prevents a worker from moving into a cell occupied by a worker who stays there.

The reward encourages progress toward each worker’s current target, penalizes collisions and unnecessary backtracking, and rewards task completion. A shared policy is used for both workers, so it learns the same navigation behavior while conditioning on each worker’s position and task state.

## Project structure

- `src/autonomous_npc/envs/office_env.py`: grid layout, station interactions, rewards, action masks, and simultaneous movement/collision rules.
- `src/autonomous_npc/envs/multi_agent_env.py`: Gymnasium wrapper that combines worker state into a joint observation and action space.
- `scripts/train_ppo.py`: train the shared MaskablePPO policy and save checkpoints.
- `scripts/evaluate_ppo.py`: run seeded episodes and report completions, episode lengths, and failure traces.
- `scripts/demo_unity.py`: run the policy and stream worker state to Unity over local TCP.
- `unity-demo/unity-office-demo`: Unity scene and scripts for visualizing the workers and station interactions.

## Setup

Create a virtual environment and install the project dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-dev.txt
```

Run project scripts from the repository root with `PYTHONPATH=src`.

## Train

Train a policy and save it to the path used by the Unity demo:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/train_ppo.py \
  --total-timesteps 200000 \
  --checkpoint-freq 10000 \
  --model-path models/multi_agent_fixed
```

The final model is saved to `models/multi_agent_fixed.zip`; periodic checkpoints are written under `models/checkpoints/`.

## Evaluate

Evaluate the trained policy over 100 seeded episodes:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/evaluate_ppo.py \
  --model-path models/multi_agent_fixed \
  --episodes 100
```

Use `--seed-start` to choose the first evaluation seed. Failed episodes print the final worker states and recent actions to help diagnose stalls.

## Unity demo

Open `unity-demo/unity-office-demo` in Unity and open `Assets/Scenes/SampleScene`. Press **Play**, then start the Python runner from the repository root:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/demo_unity.py
```

The runner loads `models/multi_agent_fixed.zip` by default and connects to Unity over local TCP port `5005`. See [unity-demo/README.md](unity-demo/README.md) for replaying a seeded episode or using another model.

## Tests

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest
```

## License

This project's original code is provided under the MIT License; see [LICENSE](LICENSE). Third party Unity assets and character models may have separate license terms.
