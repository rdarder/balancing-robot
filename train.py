# /home/rdarder/dev/balancing-robot/train.py
import argparse  # Import argparse
import glob
import os

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback

from segway_env import SegwayEnv

# --- Argument Parsing ---
parser = argparse.ArgumentParser(
    description="Train or resume training an PPO model for Segway."  # <--- Updated description
)
parser.add_argument(
    "--resume",
    type=str,
    default=None,
    help="Path to a specific checkpoint file (.zip) to resume training from. If not provided, tries to resume from the latest checkpoint (unless --no_resume is specified).",
)
parser.add_argument(
    "--no_resume",
    action="store_true",  # Makes this a boolean flag
    help="Force start training from scratch, ignoring any existing checkpoints and the --resume argument.",
)
parser.add_argument(
    "--total_timesteps",
    type=int,
    default=1000000,  # <--- Increased default total timesteps (PPO might need more samples)
    help="Total number of timesteps to train for (or additional timesteps if resuming).",
)
args = parser.parse_args()
# --- End Argument Parsing ---

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints"
TENSORBOARD_LOG_DIR = "tensorboard_logs"
CHECKPOINT_FREQ = 20000  # <--- Adjusted frequency, depends on n_steps
CHECKPOINT_PREFIX = "segway_ppo"  # <--- Changed prefix
FINAL_MODEL_NAME = "segway_ppo_final"  # <--- Changed final name
TB_LOG_NAME = "segway_ppo"  # <--- Changed TB log name
# --- End Configuration ---

# Create directories if they don't exist
os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)

# Environment
env = SegwayEnv()  # No changes needed for the environment itself

# --- Determine Model Loading ---
load_path = None
reset_num_timesteps = True  # Default to starting fresh


def make_PPO_model():
    return PPO(
        "MlpPolicy",
        env,
        learning_rate=7e-4,
        n_steps=1024,  # Collect a good number of timesteps before updating
        batch_size=64,  # Process data in small batches
        n_epochs=10,  # Multiple passes through the data
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,  # Standard PPO clipping
        ent_coef=0.05,  # Some entropy for exploration
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs={
            "net_arch": [dict(pi=[64, 64], vf=[64, 64])],  # Separate networks
        },
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )


# Highest priority: --no_resume forces a fresh start
if args.no_resume:
    print("Starting new training from scratch (--no_resume specified).")
    # Keep load_path = None and reset_num_timesteps = True

# Second priority: --resume tries to load a specific file
elif args.resume:
    if os.path.isfile(args.resume):
        load_path = args.resume
        print(f"Resuming training from specified checkpoint: {load_path}")
        reset_num_timesteps = False  # Continue timestep count from loaded model
    else:
        print(
            f"Warning: Specified checkpoint '{args.resume}' not found. Starting new training instead."
        )
        # Keep load_path = None and reset_num_timesteps = True

# Default behavior: Try to find and resume from the latest checkpoint
else:
    # Use the NEW prefix to find checkpoints
    checkpoint_pattern = os.path.join(CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*.zip")
    list_of_files = glob.glob(checkpoint_pattern)
    if list_of_files:
        try:
            # Sort by modification time to find the latest
            latest_checkpoint = max(list_of_files, key=os.path.getmtime)
            load_path = latest_checkpoint
            print(f"Resuming training from latest checkpoint: {load_path}")
            reset_num_timesteps = False  # Continue timestep count from loaded model
        except Exception as e:
            print(f"Error finding latest checkpoint: {e}. Starting new training.")
            # Keep load_path = None and reset_num_timesteps = True
    else:
        print(
            f"No suitable '{CHECKPOINT_PREFIX}' checkpoints found. Starting new training."
        )
        # Keep load_path = None and reset_num_timesteps = True
# --- End Determine Model Loading ---


# --- Load or Create Model ---
if load_path:
    try:
        # Use PPO.load
        model = PPO.load(load_path, env=env, tensorboard_log=TENSORBOARD_LOG_DIR)
        print(f"Successfully loaded PPO model from {load_path}")
        # Optional: Adjust learning rate or other hyperparameters when resuming
        # current_lr = model.learning_rate(1.0) # Need to pass progress_remaining=1 for Schedule
        # print(f"  Current learning rate: {current_lr}")
        # model.learning_rate = lambda p: 3e-4 * p # Example: Set a new learning rate schedule
        # print(f"  Set new learning rate schedule.")
    except Exception as e:
        print(f"Error loading model from {load_path}: {e}")
        print("Attempting to start new training instead.")
        load_path = None  # Ensure we create a new model below
        reset_num_timesteps = True
        # Define the PPO model again as if starting fresh
        model = make_PPO_model()
        print("Created a new PPO model after load failure.")

else:
    # Define PPO model parameters if creating a new one
    model = make_PPO_model()
    print("Created a new PPO model.")
# --- End Load or Create Model ---


# Checkpoint callback: save every N steps
checkpoint_callback = CheckpointCallback(
    save_freq=CHECKPOINT_FREQ,
    save_path=CHECKPOINT_DIR,
    name_prefix=CHECKPOINT_PREFIX,  # Use the new prefix
    # save_replay_buffer=False, # Not needed for PPO (on-policy)
    save_vecnormalize=True,  # Keep this if you wrap env with VecNormalize later
)

# Train the model
print("-" * 30)
if reset_num_timesteps:
    print(f"Starting training from step 0 for {args.total_timesteps} timesteps.")
    # For PPO, when starting fresh, total_timesteps is the target.
    learn_steps = args.total_timesteps
else:
    current_steps = model.num_timesteps
    # When resuming, total_timesteps in learn() is cumulative.
    # We still pass the *target* total timesteps.
    learn_steps = args.total_timesteps
    remaining_steps = learn_steps - current_steps
    if remaining_steps <= 0:
        print(
            f"Model already trained for {current_steps} steps. Target {args.total_timesteps} reached or exceeded."
        )
        print("To train further, increase --total_timesteps.")
        learn_steps = (
            current_steps  # Prevent negative learn calls by setting target to current
        )
    else:
        print(f"Resuming training from step {current_steps}.")
        print(
            f"Training until {learn_steps} total timesteps are reached ({remaining_steps} more steps)."
        )


if learn_steps > model.num_timesteps if not reset_num_timesteps else learn_steps > 0:
    try:
        model.learn(
            total_timesteps=learn_steps,  # Target total steps
            callback=checkpoint_callback,
            tb_log_name=TB_LOG_NAME,  # Use new TB log name
            reset_num_timesteps=reset_num_timesteps,  # Crucial for correct timestep counting and logging
        )
        print("Training loop finished.")
    except Exception as e:
        print(f"\nAn error occurred during model.learn(): {e}")
        import traceback

        traceback.print_exc()
        print("Attempting to save model state before exiting...")

    # Save the final model (or intermediate state if learn errored)
    final_model_path = os.path.join(
        CHECKPOINT_DIR, FINAL_MODEL_NAME
    )  # Use new final name
    try:
        model.save(final_model_path)
        print(
            f"Saved final model to {final_model_path}.zip"
        )  # .zip is added automatically
    except Exception as e:
        print(f"Error saving final model: {e}")
else:
    print("Skipping training loop (target timesteps already met or zero).")


# Close env
env.close()
print("Training script finished.")
