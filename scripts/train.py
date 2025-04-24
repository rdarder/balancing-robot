# /home/rdarder/dev/balancing-robot/train.py
import argparse  # Import argparse
import glob
import os
import numpy as np
import pkg_resources

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.torch_layers import MlpExtractor
from gymnasium.spaces import Box

from balancing_robot.rl.segway_env import make_segway_env
from balancing_robot.rl.custom_feature_extractor import GRUFeatureExtractor
from balancing_robot.rl.utils import load_model_from_latest_checkpoint_or_new, CHECKPOINTS_DIR, CHECKPOINT_PREFIX, ensure_checkpoint_and_log_directories_exist, check_and_log_remaining_train_steps, train_and_save, make_argument_parser

# --- Argument Parsing ---
parser = make_argument_parser()
args = parser.parse_args()

# Environment
env = make_segway_env()  # No changes needed for the environment itself
ensure_checkpoint_and_log_directories_exist()
resuming_from_path, model = load_model_from_latest_checkpoint_or_new(env, args.from_scratch, args.resume)


# Train the model
remaining_steps = check_and_log_remaining_train_steps(args.total_timesteps, resuming_from_path, model)

if remaining_steps > 0:
    train_and_save(model, args.total_timesteps, resuming_from_path is not None)
else:
    print("Skipping training loop (target timesteps already met or zero).")

# Close env
env.close()
print("Training script finished.")
