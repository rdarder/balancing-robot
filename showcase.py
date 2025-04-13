# /home/rdarder/dev/balancing-robot/showcase.py
import pybullet as p
from stable_baselines3 import SAC
import numpy as np
from segway_env import SegwayEnv  # Your env, now manages connection
import os
import glob
import random
import time

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "segway_sac"
# --- End Configuration ---

# --- Environment Setup ---
print("Creating SegwayEnv with human rendering...")
try:
    env = SegwayEnv(render_mode="human")
except ConnectionError as e:
    print(f"Fatal Error: Could not initialize SegwayEnv. {e}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred during environment creation: {e}")
    if "env" in locals() and hasattr(env, "client") and p.isConnected(env.client):
        p.disconnect(physicsClientId=env.client)
    exit()

# --- Simulation Mode Setup ---
env.is_real_time = False
print("Using discrete simulation stepping (like training).")
# ---------------------------------------------------------

# --- Find and Load the Policy Model ---
# ... (model loading code remains the same) ...
checkpoint_pattern = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*.zip")
list_of_files = glob.glob(checkpoint_pattern)
if not list_of_files:
    list_of_files = glob.glob(os.path.join(CHECKPOINT_DIR, "*.zip"))
    if not list_of_files:
        print(f"Error: No checkpoint files (.zip) found in {CHECKPOINT_DIR}")
        env.close()
        exit()
    else:
        print(
            f"Warning: No checkpoints found with prefix '{CHECKPOINT_PREFIX}'. Loading latest generic checkpoint."
        )
latest_checkpoint = max(list_of_files, key=os.path.getmtime)
print(f"Loading latest checkpoint: {latest_checkpoint}")
try:
    model = SAC.load(latest_checkpoint, env=env)
except Exception as e:
    print(f"Error loading SAC model from {latest_checkpoint}: {e}")
    env.close()
    exit()
# --- End Model Loading ---


# --- Simulation Loop (Single Episode Run with Discrete Steps) ---
sleep_interval = 1.0 / 100

try:
    # Initial reset for the single run
    print("Resetting environment for the single run...")
    obs, info = env.reset()  # Gets random targets now by default

    step_counter = 0
    episode_reward = 0.0

    while True:
        start_time = time.time()

        # --- Get Base Position for Camera ONLY ---
        # We only need this for the camera target now
        try:
            pos, _ = p.getBasePositionAndOrientation(
                env.segway_id, physicsClientId=env.client
            )
        except p.error as e:
            print(f"PyBullet error getting base position for camera: {e}")
            # Use a default position if error occurs
            pos = [0, 0, 0.1]
        # ---

        # Update camera view
        if env.render_mode == "human":
            try:
                p.resetDebugVisualizerCamera(
                    cameraDistance=0.6,
                    cameraYaw=45,
                    cameraPitch=-25,
                    cameraTargetPosition=pos,  # Use fetched position
                    physicsClientId=env.client,
                )
            except p.error as e:
                print(f"Warning: Could not reset debug visualizer camera: {e}")

        # Get action from the loaded policy
        action, _ = model.predict(obs, deterministic=True)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        step_counter += 1

        # --- MODIFIED Console Printing ---
        if step_counter % 24 == 0:
            try:
                # Call the new methods on the env object
                pitch = env.get_pitch()
                ground_speed = env.get_ground_speed()
                yaw_rate = env.get_yaw_rate()

                print("-" * 20)
                print(f"Step: {step_counter}")
                # Access targets directly from env attributes
                print(f"  Target: Thr={env.throttle:.2f}, Turn={env.turn:.2f}")
                print(f"  Action: L={action[0]:.2f}, R={action[1]:.2f}")
                # Use the values obtained from the env methods
                print(
                    f"  State: Pitch={pitch:.2f}, Speed={ground_speed:.2f}, YawRate={yaw_rate:.2f}"
                )
                print(f"  Step Reward: {reward:.3f} (Total: {episode_reward:.3f})")
            except Exception as e:  # Catch potential errors during printing
                print(f"Error during state printing: {e}")
        # --- END MODIFIED Console Printing ---

        # Check for episode end conditions
        if terminated or truncated:
            print("\n--- EPISODE FINISHED ---")
            reason = []
            if terminated:
                reason.append("Terminated (Fell Over/Out of Bounds)")
            if truncated:
                reason.append("Truncated (Max Episode Steps Reached)")
            print(f"Reason(s): {', '.join(reason)}")
            print(f"Total Steps: {step_counter}")
            print(f"Total Reward: {episode_reward:.3f}")
            break

        # Pacing for Visualization
        time.sleep(sleep_interval)

except KeyboardInterrupt:
    print("\nSimulation interrupted by user.")
except Exception as e:
    print(f"\nAn unexpected error occurred during the simulation loop: {e}")
    import traceback

    traceback.print_exc()
finally:
    # --- Cleanup ---
    # ... (cleanup code remains the same) ...
    print("Closing environment and disconnecting PyBullet...")
    if "env" in locals():
        env.close()
    else:
        try:
            if p.getConnectionInfo()["isConnected"]:
                p.disconnect()
        except NameError:
            pass
        except p.error:
            pass
    print("Cleanup complete.")
