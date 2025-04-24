import glob
import pkg_resources
import os
import argparse

from typing import Optional
from stable_baselines3 import PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import CheckpointCallback
from gymnasium import Env

CHECKPOINT_PREFIX = "segway_ppo"
TB_LOG_NAME = "segway_PPO"
TENSORBOARD_LOG_DIR = pkg_resources.resource_filename('balancing_robot', 'tensorboard_logs')
CHECKPOINTS_DIR = pkg_resources.resource_filename('balancing_robot', 'checkpoints')
FINAL_MODEL_NAME = "segway_PPO_final"  # <--- Changed final name


def get_latest_checkpoint_path() -> Optional[str]:
    checkpoint_pattern = os.path.join(CHECKPOINTS_DIR, f"{CHECKPOINT_PREFIX}_*.zip")
    list_of_files = glob.glob(checkpoint_pattern)
    if not list_of_files:
        list_of_files = glob.glob(os.path.join(CHECKPOINTS_DIR, "*.zip"))
        if not list_of_files:
            return None
        else:
            return max(list_of_files, key=os.path.getmtime)

def load_model_from_latest_checkpoint(env):
    latest_checkpoint = get_latest_checkpoint_path()
    if latest_checkpoint is None:
        return None
    return PPO.load(latest_checkpoint, env=env)

def make_model(env: Env) -> BaseAlgorithm:
    return PPO(
        "MlpPolicy",  # Still using MlpPolicy as the base, but customizing feature extraction
        env,
        learning_rate=3e-4,
        n_epochs=10,
        n_steps=2048,
        batch_size=64,
        gamma=0.99,
        policy_kwargs={
            # "features_extractor_class": GRUFeatureExtractor, # <--- Use our custom feature extractor
            # "features_extractor_kwargs": {'features_dim': 32, 'hidden_size': 16}, # <--- Pass arguments to the feature extractor
            "net_arch": dict(pi=[32, 32], vf=[32, 32]),  # Policy and Value network architecture AFTER feature extraction
            "share_features_extractor": True,
        },
        verbose=1,
        tensorboard_log=TENSORBOARD_LOG_DIR,
    )

def load_model_from_latest_checkpoint_or_new(env, try_resume: bool, resume_from: Optional[str] = None) -> tuple[Optional[str], BaseAlgorithm]:
    if not try_resume:
        return None, make_model(env)
    checkpoint_path = resume_from if resume_from is not None else get_latest_checkpoint_path()
    if checkpoint_path is None:
        return None, make_model(env)
    return checkpoint_path, PPO.load(checkpoint_path, env=env)

def ensure_checkpoint_and_log_directories_exist() -> None:
    # Create directories if they don't exist
    os.makedirs(CHECKPOINTS_DIR, exist_ok=True)
    os.makedirs(TENSORBOARD_LOG_DIR, exist_ok=True)


def check_and_log_remaining_train_steps(total_timesteps, resuming_from_path, model) -> int:
    print("-" * 30)
    if not resuming_from_path:
        print(f"Starting training from step 0 for {total_timesteps} timesteps.")
        # For PPO, when starting fresh, total_timesteps is the target.
        return total_timesteps
    else:
        current_steps = model.num_timesteps
        # When resuming, total_timesteps in learn() is cumulative.
        # We still pass the *target* total timesteps.
        learn_steps = total_timesteps
        remaining_steps = learn_steps - current_steps
        if remaining_steps <= 0:
            print(
                f"Model already trained for {current_steps} steps. Target {args.total_timesteps} reached or exceeded."
            )
            print("To train further, increase --total_timesteps.")
            return 0
        else:
            print(f"Resuming training from step {current_steps}.")
            print(
                f"Training until {learn_steps} total timesteps are reached ({remaining_steps} more steps)."
            )
            return remaining_steps


def make_checkpoint_callback(frequency: int = 20_000):
    # Checkpoint callback: save every N steps
    checkpoint_callback = CheckpointCallback(
        save_freq=frequency,
        save_path=CHECKPOINTS_DIR,
        name_prefix=CHECKPOINT_PREFIX,  # Use the new prefix
        # save_replay_buffer=False, # Not needed for PPO (on-policy)
        save_vecnormalize=True,  # Keep this if you wrap env with VecNormalize later
    )

def train_and_save(model: BaseAlgorithm, total_timesteps: int, reset_timesteps: bool, checkpoint_callback: Optional[CheckpointCallback]=None):
    try:
        model.learn(
            total_timesteps=total_timesteps,  # Target total steps
            callback=checkpoint_callback,
            tb_log_name=TB_LOG_NAME,  # Use new TB log name
            reset_num_timesteps=reset_timesteps,  # Crucial for correct timestep counting and logging
        )
        print("Training loop finished.")
    except Exception as e:
        print(f"\nAn error occurred during model.learn(): {e}")
        import traceback

        traceback.print_exc()
        print("Attempting to save model state before exiting...")

    finally:
        final_model_path = os.path.join(
            CHECKPOINTS_DIR, FINAL_MODEL_NAME
        )  # Use new final name
        try:
            model.save(final_model_path)
            print(
                f"Saved final model to {final_model_path}.zip"
            )
        except Exception as e:
            print(f"Error saving final model: {e}")


def make_argument_parser() -> argparse.ArgumentParser:
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
        "--from_scratch",
        action="store_true",  # Makes this a boolean flag
        help="Force start training from scratch, ignoring any existing checkpoints and the --resume argument.",
    )
    parser.add_argument(
        "--total_timesteps",
        type=int,
        default=1000000,  # <--- Increased default total timesteps (PPO might need more samples)
        help="Total number of timesteps to train for (or additional timesteps if resuming).",
    )
    return parser
