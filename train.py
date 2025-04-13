import os
import glob
import argparse  # Import argparse
from stable_baselines3 import SAC
from stable_baselines3.common.callbacks import CheckpointCallback
from segway_env import SegwayEnv

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Train or resume training a SAC model for Segway."
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to a specific checkpoint file (.zip) to resume training from. If not provided, tries to resume from the latest checkpoint.",
)
parser.add_argument(
    "--total_timesteps",
    type=int,
    default=100000, 
    help="Total number of timesteps to train for.",
)
args = parser.parse_args()
# --- End Argument Parsing ---

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints"
TENSORBOARD_LOG_DIR = "tensorboard_logs"
CHECKPOINT_FREQ = 10000
CHECKPOINT_PREFIX = "segway_sac"
FINAL_MODEL_NAME = "segway_sac_final"
TB_LOG_NAME = "segway_sac"
# --- End Configuration ---

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# Environment
env = SegwayEnv()

# --- Determine Model Loading ---
load_path = None
reset_num_timesteps = True

if args.resume:
    if os.path.isfile(args.resume):
        load_path = args.resume
        print(f"Resuming training from specified checkpoint: {load_path}")
        reset_num_timesteps = False  # Continue timestep count from loaded model
    else:
        print(
            f"Warning: Specified checkpoint '{args.resume}' not found. Starting new training."
        )
else:
    # Find the latest checkpoint automatically (matching the prefix)
    checkpoint_pattern = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*.zip")
    list_of_files = glob.glob(checkpoint_pattern)
    if list_of_files:
        latest_checkpoint = max(list_of_files, key=os.path.getmtime)
        load_path = latest_checkpoint
        print(f"Resuming training from latest checkpoint: {load_path}")
        reset_num_timesteps = False  # Continue timestep count from loaded model
    else:
        print("No suitable checkpoints found. Starting new training.")
# --- End Determine Model Loading ---


# --- Load or Create Model ---
if load_path:
    model = SAC.load(load_path, env=env, tensorboard_log=TENSORBOARD_LOG_DIR)
    print(f"Loaded model from {load_path}")
    # You might want to adjust learning rate or other hyperparameters when resuming
    # model.learning_rate = 1e-4 # Example
else:
    # Define SAC model parameters if creating a new one
    model = SAC(
        "MlpPolicy",
        env,
        policy_kwargs={"net_arch": [64, 64]},
        learning_rate=3e-4,
        buffer_size=100000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )
    print("Created a new SAC model.")
# --- End Load or Create Model ---


# Checkpoint callback: save every N steps
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix=CHECKPOINT_PREFIX,
    save_replay_buffer=True,  # Save replay buffer for smoother continuation
    save_vecnormalize=True,  # Save VecNormalize stats if used (not in this case, but good practice)
)

# Train the model
print(f"Starting training for {args.total_timesteps} total timesteps...")
model.learn(
    total_timesteps=args.total_timesteps,
    callback=checkpoint_callback,
    tb_log_name=TB_LOG_NAME,
    reset_num_timesteps=reset_num_timesteps,  # Crucial for correct timestep counting and logging
)

# Save the final model
final_model_path = os.path.join(CHECKPOINT_DIR, FINAL_MODEL_NAME)
model.save(final_model_path)
print(f"Saved final model to {final_model_path}")

# Close env
env.close()
print("Training complete.")
