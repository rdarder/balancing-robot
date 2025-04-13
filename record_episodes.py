# /home/rdarder/dev/balancing-robot/record_episodes.py
import os
import glob
import re
import subprocess
import shutil
import time
import pybullet as p
from stable_baselines3 import SAC
from segway_env import SegwayEnv
import numpy as np
import cv2  # <--- Import OpenCV

# --- Configuration ---
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PREFIX = "segway_sac"
VIDEO_OUTPUT_DIR = "recorded_videos"
FINAL_VIDEO_NAME = "training_progress.mp4"
FFMPEG_PATH = "ffmpeg"

# Video recording parameters
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FRAME_RATE = 30  # Target FPS for the output video
CAMERA_DISTANCE = 0.8
CAMERA_YAW = 45
CAMERA_PITCH = -25
CAMERA_TARGET_HEIGHT = 0.1  # Approx height of robot center
# --- End Configuration ---


# ... (get_steps_from_filename, check_ffmpeg, run_ffmpeg_command functions remain the same) ...
def get_steps_from_filename(filename):
    """Extracts the step number from a checkpoint filename."""
    match = re.search(rf"{CHECKPOINT_PREFIX}_(\d+)_steps\.zip", filename)
    if match:
        return int(match.group(1))
    return None


def check_ffmpeg():
    """Checks if ffmpeg is available."""
    ffmpeg_executable = shutil.which(FFMPEG_PATH)
    if ffmpeg_executable:
        print(f"Found ffmpeg at: {ffmpeg_executable}")
        return ffmpeg_executable
    else:
        print(f"Warning: '{FFMPEG_PATH}' not found in PATH.")
        print("Video stitching and text overlay will be skipped.")
        print("Please install ffmpeg and ensure it's in your PATH.")
        return None


def run_ffmpeg_command(command):
    """Runs an ffmpeg command using subprocess."""
    print(f"Running ffmpeg command: {' '.join(command)}")
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        # Limit printing potentially huge ffmpeg output unless debugging
        # print("ffmpeg stdout:", result.stdout)
        # print("ffmpeg stderr:", result.stderr)
        print("ffmpeg command successful.")
        return True
    except FileNotFoundError:
        print(
            f"Error: '{command[0]}' command not found. Is ffmpeg installed and in PATH?"
        )
        return False
    except subprocess.CalledProcessError as e:
        print(f"Error executing ffmpeg command: {e}")
        print("ffmpeg stdout:", e.stdout)
        print("ffmpeg stderr:", e.stderr)
        return False
    except Exception as e:
        print(f"An unexpected error occurred while running ffmpeg: {e}")
        return False


# --- Main Script ---
if __name__ == "__main__":
    ffmpeg_executable = check_ffmpeg()

    os.makedirs(VIDEO_OUTPUT_DIR, exist_ok=True)
    print(f"Ensured video output directory exists: {VIDEO_OUTPUT_DIR}")
    print("Cleaning previous individual videos...")
    cleaned_count = 0
    for f in glob.glob(os.path.join(VIDEO_OUTPUT_DIR, f"{CHECKPOINT_PREFIX}_*.mp4")):
        try:
            os.remove(f)
            cleaned_count += 1
        except OSError as e:
            print(f"  Warning: Could not remove {f}: {e}")
    print(f"  Removed {cleaned_count} previous video files.")

    # Find and sort checkpoints
    checkpoint_pattern = os.path.join(
        CHECKPOINT_DIR, f"{CHECKPOINT_PREFIX}_*_steps.zip"
    )
    list_of_files = glob.glob(checkpoint_pattern)
    if not list_of_files:
        print(f"No checkpoint files found matching '{checkpoint_pattern}'. Exiting.")
        exit()
    checkpoints_with_steps = []
    for f in list_of_files:
        steps = get_steps_from_filename(f)
        if steps is not None:
            checkpoints_with_steps.append((steps, f))
    checkpoints_with_steps.sort(key=lambda x: x[0])

    recorded_video_files = []

    # --- Loop through Checkpoints and Record ---
    for steps, checkpoint_path in checkpoints_with_steps:
        print("-" * 30)
        print(
            f"Processing checkpoint: {os.path.basename(checkpoint_path)} ({steps} steps)"
        )

        video_filename = os.path.join(
            VIDEO_OUTPUT_DIR, f"{CHECKPOINT_PREFIX}_{steps}_steps.mp4"
        )
        print(f"  Target video file: {video_filename}")

        env = None
        video_writer = None  # Initialize video writer outside try
        try:
            # --- Environment Setup (DIRECT mode) ---
            env = SegwayEnv(render_mode="direct")
            p.setTimeStep(env.time_step, physicsClientId=env.client)

            # --- Load Model ---
            model = SAC.load(checkpoint_path, env=env)

            # --- Setup Video Writer (OpenCV) ---
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Codec for MP4
            video_writer = cv2.VideoWriter(
                video_filename, fourcc, FRAME_RATE, (FRAME_WIDTH, FRAME_HEIGHT)
            )
            if not video_writer.isOpened():
                print(f"  ERROR: Could not open video writer for {video_filename}")
                continue  # Skip to next checkpoint
            print(f"  Video writer opened successfully for {video_filename}")

            # --- Run and Record ---
            obs, info = env.reset()
            print(f"  Starting episode recording frame by frame...")

            step_counter = 0
            frames_written = 0
            while True:
                # --- Get Action & Step ---
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                step_counter += 1

                # --- Capture Frame ---
                try:
                    # 1. Get robot position for camera targeting
                    base_pos, _ = p.getBasePositionAndOrientation(
                        env.segway_id, physicsClientId=env.client
                    )
                    cam_target_pos = [base_pos[0], base_pos[1], CAMERA_TARGET_HEIGHT]

                    # 2. Calculate camera position based on yaw, pitch, distance
                    cam_dist = CAMERA_DISTANCE
                    cam_yaw = CAMERA_YAW
                    cam_pitch = CAMERA_PITCH
                    # Calculate eye position using spherical coordinates relative to target
                    cam_x = cam_target_pos[0] - cam_dist * np.cos(
                        np.radians(cam_pitch)
                    ) * np.cos(np.radians(cam_yaw))
                    cam_y = cam_target_pos[1] - cam_dist * np.cos(
                        np.radians(cam_pitch)
                    ) * np.sin(np.radians(cam_yaw))
                    cam_z = cam_target_pos[2] + cam_dist * np.sin(np.radians(cam_pitch))
                    cam_eye_pos = [cam_x, cam_y, cam_z]

                    # 3. Compute matrices
                    view_matrix = p.computeViewMatrix(
                        cameraEyePosition=cam_eye_pos,
                        cameraTargetPosition=cam_target_pos,
                        cameraUpVector=[0, 0, 1],
                        physicsClientId=env.client,
                    )
                    projection_matrix = p.computeProjectionMatrixFOV(
                        fov=60,
                        aspect=float(FRAME_WIDTH) / FRAME_HEIGHT,
                        nearVal=0.1,
                        farVal=100.0,
                        physicsClientId=env.client,
                    )

                    # 4. Get image data
                    img_arr = p.getCameraImage(
                        width=FRAME_WIDTH,
                        height=FRAME_HEIGHT,
                        viewMatrix=view_matrix,
                        projectionMatrix=projection_matrix,
                        renderer=p.ER_BULLET_HARDWARE_OPENGL,  # Try OpenGL first
                        # renderer=p.ER_TINY_RENDERER, # Fallback if OpenGL fails
                        physicsClientId=env.client,
                    )

                    # 5. Process image
                    width = img_arr[0]
                    height = img_arr[1]
                    rgb_pixels = img_arr[2]  # RGBA pixels

                    # Reshape the pixel data
                    np_img_arr = np.reshape(rgb_pixels, (height, width, 4))

                    # --- FIX: Convert data type to uint8 ---
                    # OpenCV expects uint8 for color conversions
                    np_img_arr_uint8 = np_img_arr.astype(np.uint8)
                    # --- END FIX ---

                    # Convert RGBA (first 3 channels) to BGR for OpenCV VideoWriter
                    # Use the converted array!
                    frame_bgr = cv2.cvtColor(
                        np_img_arr_uint8[:, :, :3], cv2.COLOR_RGB2BGR
                    )

                    # 6. Write frame
                    video_writer.write(frame_bgr)
                    frames_written += 1

                except p.error as e:
                    print(f"  Warning: PyBullet error during getCameraImage: {e}")
                except Exception as e:
                    print(f"  Warning: Error processing frame: {e}")

                # --- Check for episode end ---
                if terminated or truncated:
                    reason = "Terminated" if terminated else "Truncated"
                    print(
                        f"  Episode finished ({reason} at step {step_counter}). Wrote {frames_written} frames."
                    )
                    break

            # --- Finalize Video ---
            print("  Releasing video writer...")
            video_writer.release()
            video_writer = None  # Reset for the finally block check

            # Verify file existence
            if os.path.exists(video_filename) and os.path.getsize(video_filename) > 0:
                print(f"  Verified video file saved: {video_filename}")
                recorded_video_files.append((steps, video_filename))
            else:
                print(
                    f"  ERROR: Video file {video_filename} was NOT created or is empty."
                )

        except Exception as e:
            print(f"  Error processing checkpoint {checkpoint_path}: {e}")
            import traceback

            traceback.print_exc()
        finally:
            # --- Cleanup for this checkpoint ---
            if video_writer is not None and video_writer.isOpened():
                print("  Closing video writer in finally block...")
                video_writer.release()  # Ensure writer is released on error
            if env:
                print("  Closing environment...")
                env.close()
                print("  Environment closed.")
            time.sleep(0.5)  # Short pause between checkpoints

    print("-" * 30)
    print("Finished recording loop.")
    print(
        f"Recorded video file paths: {[p for s, p in recorded_video_files]}"
    )  # Show paths

    # --- Stitch Videos with FFMPEG (Optional) ---
    # This part should work as before, operating on the generated MP4 files
    if ffmpeg_executable and recorded_video_files:
        print("Attempting to stitch videos and add text overlays...")

        # 1. Get durations (needed for text timing)
        print("  Getting video durations...")
        segment_durations = []
        valid_videos_for_stitching = []  # Keep track of videos with valid durations
        for i, (steps, video_path) in enumerate(recorded_video_files):
            try:
                probe_command = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ]
                result = subprocess.run(
                    probe_command, check=True, capture_output=True, text=True
                )
                duration = float(result.stdout.strip())
                if duration <= 0:  # Check for zero or negative duration
                    print(
                        f"  Warning: Invalid duration ({duration:.2f}s) for {os.path.basename(video_path)}. Skipping."
                    )
                    continue
                segment_durations.append(duration)
                valid_videos_for_stitching.append(
                    (steps, video_path)
                )  # Add to valid list
                print(
                    f"    Duration of {os.path.basename(video_path)}: {duration:.2f}s"
                )
            except Exception as e:
                print(
                    f"  Error probing {video_path}: {e}. Skipping this video for stitching."
                )

        # Proceed only if there are valid videos to stitch
        if valid_videos_for_stitching and len(valid_videos_for_stitching) == len(
            segment_durations
        ):
            filelist_path = os.path.join(VIDEO_OUTPUT_DIR, "filelist.txt")
            filter_complex_parts = []
            input_args = []
            last_segment_end_time = 0.0

            with open(filelist_path, "w") as f:
                for i, (steps, video_path) in enumerate(valid_videos_for_stitching):
                    input_args.extend(["-i", video_path])
                    f.write(f"file '{os.path.abspath(video_path)}'\n")

                    text = f"Checkpoint: {steps} steps"
                    start_time = last_segment_end_time
                    end_time = start_time + segment_durations[i]
                    filter_complex_parts.append(
                        f"[{i}:v]drawtext=text='{text}':"
                        f"fontcolor=white:fontsize=24:box=1:boxcolor=black@0.5:boxborderw=5:"
                        f"x=(w-text_w)/2:y=10:"
                        f"enable='between(t,{start_time},{end_time})'[v{i}]"
                    )
                    last_segment_end_time = end_time

            filter_graph = ";".join(filter_complex_parts)
            concat_inputs = "".join(
                [f"[v{i}]" for i in range(len(valid_videos_for_stitching))]
            )
            filter_graph += f";{concat_inputs}concat=n={len(valid_videos_for_stitching)}:v=1:a=0[outv]"

            final_output_path = os.path.join(VIDEO_OUTPUT_DIR, FINAL_VIDEO_NAME)
            ffmpeg_command = [
                ffmpeg_executable,
                *input_args,
                "-filter_complex",
                filter_graph,
                "-map",
                "[outv]",
                "-c:v",
                "libx264",
                "-preset",
                "medium",
                "-crf",
                "23",
                "-y",
                final_output_path,
            ]

            if run_ffmpeg_command(ffmpeg_command):
                print(f"Successfully stitched video saved to: {final_output_path}")
                # Optional cleanup (still commented out by default)
                # print("  Cleaning up individual video files and filelist.txt...")
                # try: os.remove(filelist_path)
                # except OSError: pass
                # for _, video_path in valid_videos_for_stitching:
                #     try: os.remove(video_path)
                #     except OSError: pass
                print("  Cleanup skipped (manual cleanup recommended).")
            else:
                print("Stitching failed. Individual videos are kept.")
        elif valid_videos_for_stitching:
            print("Skipping stitching due to errors getting some video durations.")
        else:
            print("No valid videos found to stitch.")

    elif not recorded_video_files:
        print("No videos were recorded.")
    else:
        print("ffmpeg not found, cannot stitch videos.")

    print("Script finished.")
