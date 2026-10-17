# Run the trained policy in Unity

The Unity scene listens for the Python policy runner on `127.0.0.1:5005`.
The runner controls both workers using `models/multi_agent_fixed.zip` by default.
Station interactions trigger the existing snack, coffee, printer, and bathroom
animations in the Unity scene. Worker 2 uses the Secretary character prefab while
Worker 1 keeps the Office Worker model. Policy steps continue at the same pace
during station animations so one worker's interaction does not pause both.

1. Open `unity-demo/unity-office-demo` in Unity and open `Assets/Scenes/SampleScene`.
2. Press **Play**. The Console should say it is waiting for Python on port 5005.
3. From the repository root, run:

   ```bash
   PYTHONPATH=src ./.venv/bin/python scripts/demo_unity.py
   ```

Use `--seed 14` to replay a fixed environment layout, or `--model-path PATH`
to use a different MaskablePPO model. Python prints the episode seed and each
simulation tick; the Unity scene displays both workers, their current targets,
and the episode's obstacles. Stop Play mode to close the Unity listener.
