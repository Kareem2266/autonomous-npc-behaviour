# Autonomous NPC Behaviour

A grid based office simulation where two autonomous workers share an environment, complete individual task routes, and avoid blocking one another. A shared MaskablePPO policy selects both workers' actions at each simulation tick. The same policy can drive the Unity office demo.

## Features

- Simultaneous actions for multiple workers, with separate desks, routes, targets, and task state.
- Action masking and collision handling for obstacles, occupied cells, and shared destinations.
- PPO training with periodic checkpoints and a multi episode evaluation script.
- A Unity demo that displays both workers and plays station interaction animations.

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
