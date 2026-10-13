# Autonomous NPC Project Handoff

Updated: 2026-10-12

Use this note to resume work in a new chat. It describes the repository state checked on this date and carries forward the project context and collaboration preferences.

## Project goal

Build a goal-conditioned autonomous NPC in a Unity office. The near-term demo is one agent navigating a 10×10 office, receiving a task route, and completing it while avoiding obstacles. The longer-term goal includes multiple agents working at the same time, each handling tasks from a shared set. The target demo date discussed was October 18, 2026.

The planned order is to keep the single-agent task loop reliable, polish its presentation with 3D models and a better office scene, then expand to multiple agents. No character or office asset models have been added yet.

## Current repository state

- Current branch: `clean-up-demo`.
- HEAD: `bb6cced`, the same commit as local `main` and `origin/main` when checked.
- Working tree was clean when checked.
- Unity project: `unity-demo/unity-office-demo/`.
- Unity editor version: `6000.6.5f1`.
- The active Python environment is the Gymnasium environment in `src/autonomous_npc/envs/office_env.py`.

## Python environment

`OfficeEnv` defaults to a 10×10 grid. Coordinates are `[x, y]`.

For the 10×10 office, the stations are:

- Desk: `[1, 5]`
- Printer: `[8, 2]`
- Bathroom: `[8, 8]`
- Snack Area: `[3, 2]`
- Coffee Area: `[3, 8]`

Each reset chooses one of five routes at random:

- Desk → Printer → Desk
- Desk → Bathroom → Desk
- Desk → Snack Area → Coffee Area → Desk
- Desk → Snack Area → Desk
- Desk → Coffee Area → Desk

The agent starts in a randomly selected reachable cell. Its first target is the desk. It must use `INTERACT` at the desk to pick up the document, then visits each selected route stop. The episode ends when it finishes the route and returns to the desk.

Two distinct obstacles are selected randomly on each reset. They cannot overlap a station, and candidate layouts are rejected if a route leg or the agent’s path to the desk is blocked. A breadth-first search in `_get_path_distance()` calculates shortest walkable distance around obstacles.

The observation is a dictionary containing `agent`, `previous_agent`, `target`, and two `obstacle` coordinates. The six actions are north, east, south, west, interact, and wait.

`action_masks()` masks moves outside the grid or into obstacles. It allows `INTERACT` only when the agent is at the desk before pickup and masks `WAIT`. Rewards give progress toward the current target, apply a per-step cost, penalize reversing and obstacle impacts, and reward completing the route. The exact reward logic is in `step()` in `office_env.py`.

## PPO model and reported evaluation

- Training uses `MaskablePPO` with `MultiInputPolicy` in `scripts/train_ppo.py`.
- Training is currently configured for 50,000 timesteps.
- The model saves as `office_10x10_masked_ppo.zip` in the repository root.
- `scripts/evaluate_ppo.py` runs 100 deterministic episodes with action masks and fixed seeds.
- The last reported evaluation was **100/100 success**, **21.8 average steps**, **21.5 average shortest route**, and **0.3 average extra steps**. This is a previously reported result; training/evaluation were not rerun while preparing this handoff.

## Unity integration

`scripts/demo_unity.py` loads the trained model and sends state to Unity over TCP at `127.0.0.1:5005`. Each line contains eight comma-separated integers: agent coordinates, target coordinates, and both obstacle coordinates. With no `--seed`, the demo selects a random seed; pass `--seed N` to replay a run.

In Unity:

- `PythonCommandReceiver.cs` listens on port 5005, creates the colored station tiles and two obstacle markers, and applies received state.
- `GridAgentController.cs` maps grid coordinates to Unity positions and moves the agent smoothly between cells.
- The station labels now lie flat on their colored tiles. Kareem confirmed the current labels look right in Unity. They are generated in `Start()`, so stop Play and restart after script changes.
- The target transform must be assigned to the receiver’s `Target` field in the Inspector.
- The current scene is still a basic prototype with primitive geometry; 3D models and visual polish are future work.

## Commands

From the repository root, with the virtual environment activated or addressed directly:

```bash
PYTHONPATH=src ./.venv/bin/python -m pytest -v
PYTHONPATH=src ./.venv/bin/python scripts/train_ppo.py
PYTHONPATH=src ./.venv/bin/python scripts/evaluate_ppo.py
```

For the Unity demo, first open `unity-demo/unity-office-demo` in Unity and press Play. Wait for “Waiting for python on port 5005,” then run:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/demo_unity.py
```

To replay a particular seed:

```bash
PYTHONPATH=src ./.venv/bin/python scripts/demo_unity.py --seed 42
```

Tests were reported passing after the environment, task-route, and action-mask work. They were not rerun for this handoff.

## Next work

1. Continue polishing the visible single-agent demo. The user said adding character and office models is a later session, not the immediate task.
2. Once the presentation is in good shape, plan and implement multiple simultaneous agents, including task assignment and avoiding agent collisions.

## Collaboration preferences

- Do not write or change code unless Kareem explicitly asks for implementation. When he asks what to do, guide him through the change instead of doing it.
- Explain clearly and confidently. Give the exact file, location, and replacement instructions. Keep related code together and avoid excessive, fragmented code blocks.
- Kareem is still learning Unity. Give concrete panel names and click-by-click directions when needed.
- For progress notes, use concise bullets such as “Added X” and “Fixed Y”; do not phrase them as “we did X.”
- There are two vaults: `.vault-staging` and the actual knowledge vault. If a vault update is requested, send it to the actual vault, not staging.
