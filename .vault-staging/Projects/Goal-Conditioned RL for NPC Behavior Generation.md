# Goal-Conditioned RL for NPC Behavior Generation

## Project

- Repository: `/Users/kareemmaamoun/Coding/autonomous-npc-behaviour`
- Design version: v0.1
- Status: M1 in progress; October 18 Unity demo in progress

## Current checkpoint (2026-10-10)

- Repository branch: `unity-demo-creation`, based on `main` at `e084d3f`. Current worktree changes are `.gitignore` and the untracked `unity-demo/` Unity project; neither has been committed yet.
- Python baseline: PPO trained for 50,000 requested timesteps (51,200 collected due to 2,048-step rollouts), saved as `office_ppo_50k.zip`. Evaluation on seeds 0–99 reached the target in 100/100 episodes; average steps and shortest Manhattan route both print as 3.4, with average excess rounded to 0.0.
- Tests: reward expectations were updated for distance-progress shaping, with new closer/farther cases and boundary no-progress coverage. Latest run: `.venv/bin/python -m pytest -v` — 40 passed.
- Unity: Kareem created a Universal 3D project at `unity-demo/unity-office-demo`. `Assets/Scenes/SampleScene.unity` and default URP assets exist; no custom floor, agent, target, or bridge scripts have been added yet.
- `.gitignore` now excludes Unity-generated `Library`, `Temp`, `Obj`, build, logs, user settings, IDE files, and generated project/solution files. `git check-ignore` confirmed generated Unity folders are ignored.
- Immediate task: in the Unity Editor, make a simple 5×5 floor with a visible agent and target, save it as an office scene, then map grid coordinates to Unity X/Z and send one Python movement action to Unity.
- October 18 demo remains: Python PPO chooses actions in the 2D Gymnasium environment; Unity visualizes the same actions in real time in a simple 3D office. Multi-role behavior remains long-term scope.

## Historical checkpoint (2026-09-26)

## Current checkpoint (2026-09-26)

- Environment: `src/autonomous_npc/envs/office_env.py` defines the six-action `Action` enum, configurable square-grid size with a minimum of 2, Dict observation space for agent/target coordinates, Discrete action space, `[x, y]` movement vectors, observation/info helpers, and seeded `reset()` that places agent and target in distinct cells.
- Tests: Codex maintains `tests/test_office_env.py`; it covers valid/invalid sizes, reset observation validity, distinct start positions across seeds, seeded reproducibility, and Manhattan distance in `info`.
- Rendering: `OfficeEnv` now supports `human` and `rgb_array`, renders the target as red and agent as blue, uses `[x, y]` coordinates without reversing axes, draws corrected circles/grid lines, updates the human display, and returns an RGB NumPy array in `(height, width, channels)` order.
- Dependencies: `pygame==2.6.1` added to `requirements.txt` (already present in the local venv at verification time).
- Verification: all 33 tests pass, including render-mode validation, disabled-render behavior, frame shape/dtype, and pixels confirming x/y orientation.
- CI: `.github/workflows/tests.yml` runs pytest on GitHub Actions for pushes and pull requests; dependency pins are in `requirements.txt` and `requirements-dev.txt`.
- Current `step()` work: Kareem corrected the updated position to apply to the agent, preserved the target, handled `INTERACT` and `WAIT` as no-op movement, and calls `_get_obs()`/`_get_info()` to return current state. The basic target-reaching reward and termination behavior is covered and passing.
- Tests: parameterized cases now have descriptive IDs (e.g. `north`, `north-edge`, `seed=0`, `wait`) to make failures easier to identify.
- Random runner: `scripts/run_random_agent.py` imports `OfficeEnv`, creates it with `render_mode="human"`, captures/prints the reset observation, samples actions, tracks total reward, exits on termination/truncation, and closes in `finally`. It has successfully reached the target; this validates mechanics/rendering, not RL. Launch from the repository root with `PYTHONPATH=src python scripts/run_random_agent.py`.
- Current work: branch `creation-of-agent-policy`; uncommitted changes in `scripts/run_random_agent.py` and new `tests/test_random_agent.py`. Policy `choose_action(observation, env)` computes target-minus-agent, uses coordinate signs to move x-first then y, and reverse-searches `env._action_to_direction` to return the matching Action key. Runner is import-safe via `main()` and `if __name__ == "__main__"` guard. Added five parameterized policy cases (four cardinal directions plus x-first diagonal). Latest verification: `.venv/bin/python -m pytest -v` — 38 passed.
- Next: optionally run the visible demo again after the main guard change, then review what the deterministic hand-coded policy demonstrates and how it differs from a learned policy. Before the next new task, ask Kareem for a brief teach-back; don't over-quiz routine setup details. Later move toward PPO after environment behavior/reward are understood. Kareem writes production policy code; Codex maintains tests.
- Learning arrangement: Kareem writes the production implementation; Codex writes and maintains tests. Guide production work as assignments with explanations and analogous examples; inspect the workspace when Kareem says a step is done.

Treat this checkpoint as a snapshot. Update it after substantive code, tests, dependency, CI, or design changes, and after each verified milestone. Record what changed, the latest test result, Git branch/commit plus any worktree status, and the next concrete step so another session can resume without reconstructing context.

## Core question

Can one policy, conditioned on a structured role, produce plausible and varied autonomous NPC behavior—including sensible behavior for interpolated or held-out roles?

This is goal-conditioned / multi-task reinforcement learning, not an LLM-agent system. The intended mapping is `(state, role_vector) → action`.

## Success criteria

1. An office worker produces believable, non-repetitive task sequences.
2. One flat PPO policy produces distinct, correct behavior for 3–4 trained roles.
3. Stretch/headline result: useful zero-shot behavior on an interpolated or held-out role.

## V1 boundaries

No 3D animation or rigging, real-time multiplayer, natural-language interface, or photorealistic environment.

## Environment direction

- Custom Gymnasium 2D office grid world.
- Zones/objects include desks, coffee machine, meeting room, and break room.
- State includes position, nearby objects/zones and distances, episode time, needs, and role vector.
- Initial needs: `[energy, social, hunger, work_backlog]`.
- Needs evolve over time; stochastic events such as meetings create purposeful episode variation.
- Initial discrete actions: movement in four directions, `interact`, and `wait`.
- Unity ML-Agents is deferred until a later visual/portfolio phase.

## Role-conditioning path

1. Start with one-hot roles to validate the full pipeline.
2. A learned embedding is possible, though less interpretable.
3. Target role-as-reward-template conditioning: role weights over fixed objectives.

For the target representation:

`reward = role_weights · objective_vector`

Example objectives include desk work, social interaction, breaks, meetings, and idling. Interpolating role weights provides a clean generalization experiment.

## Behavior variety

- MVP: needs-driven task selection plus stochastic episode events.
- V2 stretch: hierarchical RL with high-level task selection and shared low-level navigation/interaction.

## Reward-design note

Reward shaping is expected to be the most iterative part. Begin with role-specific rewards such as completed desk-work cycles, timely meeting attendance, and mild penalties for excessive idling. Do not introduce reward-template conditioning until the environment and training loop are stable.

## Tooling

- PPO
- Stable-Baselines3, unless implementing PPO from scratch specifically for learning
- TensorBoard or Weights & Biases from day one

## Evaluation

- Inspect rendered or logged trajectories for plausibility.
- Track per-role completion rates, idle percentage, and convergence.
- Evaluate an unseen interpolated or held-out role zero-shot for the primary research result.

## Milestones

| Milestone | Deliverable |
|---|---|
| M0 | Gymnasium skeleton and scripted/random-policy mechanics validation |
| M1 | Plausible converged office-worker PPO behavior |
| M2 | Distinct one-hot-conditioned behavior across 3–4 roles |
| M3 | Reward-template conditioning |
| M4 | Generalization evaluation and write-up |
| M5 | Hierarchical RL stretch goal |

Expected M4 timeline at 5–8 hours/week: roughly three months.

## Collaboration preference

The owner intends to implement milestones personally. Codex should serve as a reviewer, debugging partner, and design collaborator when given code, errors, or decisions—not automatically implement entire milestones unless explicitly asked.

## Product direction

The long-term goal is a reusable system that companies or other developers can adapt to their own NPC roles, objectives, environments, and game use cases. The initial 2D office is a research and validation environment, not a permanent product limitation.

Keep the learned-behavior system independent from rendering and engine-specific code. Future 3D support should be provided through environment/engine adapters (for example Unity or Unreal integrations) that translate game observations and actions into the policy's stable interface. Do not allow office-specific concepts, grid coordinates, or 2D movement assumptions to become inseparable from the core role-conditioning and policy APIs.

## October 18 demo goal

By 2026-10-18, deliver a live demo in which a Python RL agent trains in the existing 2D grid environment and its chosen actions drive an agent moving through a simple 3D office scene in real time. Keep the first demo to one learned navigation behavior, one room, and a small number of visible objects. The 3D scene is a visualization/adapter for the Python agent; multi-role behavior, needs, and convincing office routines remain future work.

At the beginning of each work session, compare the current repo and demo against that day's target, note what is done or behind, and adjust the remaining targets. Keep progressing toward this goal until the demo is delivered. These are daily checkpoints, not a promise that every feature must be completed on its listed date.

### Replanned daily targets (October 1–18, 2026; reviewed October 6, accelerated)

| Date | Mini target |
|---|---|
| Oct 1 | **Done:** lock the minimum demo scope, choose the 3D engine/viewer, and confirm the Python environment and current repository state. |
| Oct 2–3 | **Missed; rolled forward:** set up SB3/PyTorch and start PPO training. |
| Oct 4 | **Done:** Kareem installed the declared ML dependencies and created the training-script bootstrap. |
| Oct 5 | **Missed; rolled forward:** add PPO construction, training, and model saving. |
| Oct 6 | Finish the PPO trainer, run the first 10,000-step training job, save its checkpoint, and do a quick evaluation. |
| Oct 7 | Complete evaluation over fresh seeds; record success rate/episode length, fix any serious training issue, and freeze the Python behavior/output. |
| Oct 8 | Set up the Unity project and build a simple office room, agent, and target. |
| Oct 9 | Map `[x, y]` grid coordinates to Unity world coordinates and confirm reset/action event fields. |
| Oct 10 | Send one Python test action/state to Unity and see the expected move. |
| Oct 11 | Map all four movement actions plus WAIT/INTERACT to visible Unity behavior; run a complete scripted episode and reset. |
| Oct 12 | Connect the saved PPO policy so its live actions drive the Unity agent. |
| Oct 13 | Verify synchronization across repeated episodes and fix demo-blocking connection issues. |
| Oct 14 | Run end-to-end PPO demonstrations; collect results and compare against scripted/random behavior. |
| Oct 15 | Improve scene readability and prepare the walkthrough of what PPO learned. |
| Oct 16 | Rehearse from a clean start; fix only demo-blocking problems. |
| Oct 17 | Final rehearsal, freeze features, and preserve a known-good model/build plus fallback recording. |
| Oct 18 | Deliver and present the working demo; record the next project milestone. |

### Starting point (October 1)

- Current checkout is on branch `gymnasium-rl` at `9c5139d` (merge PR #4); `requirements-dev.txt` has an uncommitted addition of `stable-baselines3==2.9.0`.
- The Python environment remains the small Gymnasium office grid with four cardinal movement actions, `INTERACT`, `WAIT`, Pygame rendering, and a deterministic target-seeking scripted runner. PPO training and the 3D adapter/viewer are not yet implemented.
- First-session decisions (2026-10-01): use Unity for the first 3D adapter. Kareem has Unity experience, and Unity is already one of the long-term target engine integrations. PPO will choose actions in Python against the Gymnasium environment; Unity receives action/state updates and visualizes them. The existing scripted runner remains a baseline and bridge-debugging tool, not the learned demo policy. Keep the policy and Gymnasium environment engine-independent; Unity does not choose actions or calculate environment rewards/state.
- Replan check (2026-10-04): no project changes since October 1. The checkout remains on `gymnasium-rl` at `9c5139d`, `requirements-dev.txt` has an uncommitted Stable-Baselines3 pin, and the active Python 3.13.9 `.venv` has Gymnasium/Pygame/NumPy but not PyTorch or Stable-Baselines3. Oct 2–3 training targets are rolled into Oct 4–6. No Unity project or PPO trainer exists yet.
- Next: Kareem handles dependency installation and implements the minimal PPO trainer; Codex guides and reviews. Unity scene/bridge work starts after the Python policy has been evaluated so rendering work does not obscure RL issues.
- Replan check (2026-10-06): Kareem installed dependencies; `.venv` imports PyTorch 2.14.1, Stable-Baselines3 2.9.0, and Gymnasium 1.3.0. `scripts/train_ppo.py` currently only instantiates `OfficeEnv`, resets it, prints observation/info, and closes it. No PPO model/training call or saved checkpoint is present. Accelerated target: complete the trainer, first 10,000-step run, and quick check today; finish multi-seed evaluation tomorrow. Keep the ML-first sequence and protect the engine-independent Python learning layer.
- Session 1 environment check (2026-10-01): the active `.venv` is Python 3.13.9 and has Gymnasium, Pygame, and NumPy, but not PyTorch or Stable-Baselines3. No Godot, Unity, or Blender executable was found on PATH, and none of those apps appeared in `/Applications`. This is a discovery check only; PPO training and 3D runtime have not been verified. Next: Kareem's teach-back on the scripted policy, then confirm dependency installation/compatibility and choose a 3D viewer.

## Sources to consult

- Gymnasium custom-environment documentation
- Stable-Baselines3 PPO, custom-policy, and `VecEnv` documentation
- Sutton & Barto on policy gradients
- Goal-conditioned RL surveys
- Sutton, Precup, and Singh on the options framework
- Multi-task RL research relevant to role conditioning

## Learning log

- 2026-09-12 — Initial project design captured.
- 2026-09-13 — Migrated durable project context into the shared Codex knowledge vault.
- 2026-09-13 — Clarified the product goal: reusable and configurable for external companies, with 3D engine support planned after the core RL system is validated in 2D.
- 2026-09-22 — M0 checkpoint recorded: Gymnasium skeleton and seeded reset are implemented; CI and initial environment tests are in place; movement in `step()` is next.
- 2026-09-23 — PR #1 merged to `main`; checkpoint advanced to merged commit `723b6dc`, with `step()` movement and boundary handling as the next task.
- 2026-09-23 — Began `step()` work on `implementing-step`; added 11 movement/boundary/no-op/target tests. Latest run: 19 passed, 11 failed, revealing the production issues recorded above. Rendering deferred until tomorrow.
- 2026-09-23 — Kareem fixed missing calls to the observation/info helpers; latest test run: 23 passed, 7 failed. Remaining failures are movement state assignment and handling `INTERACT`/`WAIT` as no-ops.
- 2026-09-23 — Kareem fixed `step()` movement/state updates and non-movement actions. Added descriptive IDs to parameterized pytest cases. Latest suite: all 30 tests pass. Rendering remains the next feature when resumed.
- 2026-09-24 — PR #2 merged step behavior. Clean `main` at `ffe7f08`; all 30 tests passed before merge. Next: terminal-style ANSI grid rendering, followed by using the renderer in the random-agent runner.
- 2026-09-24 — Implemented Pygame `human`/`rgb_array` rendering, corrected `[x, y]` mapping, circle/grid-line calls, display update, and RGB array conversion. Added `pygame==2.6.1` and three render tests. All 33 tests pass. Next: launch human rendering from the runner and inspect the live window.
- 2026-09-24 — Added and ran a random-action Pygame runner; it reached the target and exited, confirming rendering and episode termination work together. This is not RL. Kareem requests a teach-back before every new step: he will explain what the previous step did and why, then Codex will correct gaps and only afterward introduce the next assignment.
