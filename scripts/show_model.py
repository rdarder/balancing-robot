import math
import sys
import time
import pkg_resources

import numpy as np
import pybullet as p
from rich.console import Console
from rich.layout import Layout
from rich.table import Table
import numpy as np

from balancing_robot.rl.pybullet_utils import add_debug_lines, get_non_fixed_joint_ids
from balancing_robot.rl.segway_env import SegwayEnv, make_segway_env
from balancing_robot.rl.train_utils import load_model_from_latest_checkpoint


class DebugSegwayEnv(SegwayEnv):
    def __init__(self, output_video_filename=None):
        super().__init__(render_mode="human", is_real_time=True)
        self._video_filename = output_video_filename
        self._console = Console()
        self._log_id = None

    def reset(self, *, seed=None, options=None):
        for joint_id in get_non_fixed_joint_ids(self.segway_id, self._client_id):
            add_debug_lines(joint_id, self.segway_id, self._client_id)

        self._camera_follows_segway()

        if self._video_filename is not None:
            self._log_id = p.startStateLogging(
                p.STATE_LOGGING_VIDEO_MP4, self._video_filename
            )

        ret = super().reset(seed=seed, options=options)
        time.sleep(1.0)
        return ret

    def step(self, action):
        observation, reward, terminated, done, info = super().step(action)
        self._camera_follows_segway()
        if self.step_count % self.STEPS_PER_SECOND == 0:
            self._print_debug_info(info)
        return observation, reward, terminated, done, info

    def _camera_follows_segway(self):
        camera_target = p.getBasePositionAndOrientation(
            self.segway_id, physicsClientId=self._client_id
        )[0]
        p.resetDebugVisualizerCamera(
            cameraDistance=0.3,
            cameraYaw=100,
            cameraPitch=-20,
            cameraTargetPosition=camera_target,
            physicsClientId=self._client_id,
        )

    def close(self):
        if self._log_id is not None:
            p.stopStateLogging(self._log_id)

    def _print_debug_info(self, info):
        rewards = Table(show_header=True)
        rewards.add_column("metric", style="dim", width=10)
        rewards.add_column("w_reward", style="bold", width=8, justify="right")
        rewards.add_column("r_reward", style="dim", width=8, justify="right")
        rewards.add_column("weight", style="dim", width=6, justify="right")
        rewards.add_column("value", style="bold", width=12, justify="right")
        rewards.add_column("target", style="bold", width=12, justify="right")

        rewards.add_row("total", fmt(info["total_reward"]))
        rewards.add_row(
            "speed",
            fmt(info["speed_reward"]),
            fmt(info["speed_reward_raw"]),
            fmt(self.W_SPEED),
            fmt(info["forward_speed"]),
            fmt(self.target_speed),
        )
        rewards.add_row(
            "turn",
            fmt(info["turn_reward"]),
            fmt(info["turn_reward_raw"]),
            fmt(self.W_TURN),
            fmt(np.rad2deg(info["turn"])),
            fmt(np.rad2deg(self.target_turn)),
        )
        rewards.add_row(
            "upright",
            fmt(info["upright_reward"]),
            fmt(info["upright_reward_raw"]),
            fmt(self.W_UPRIGHT),
            fmt(np.rad2deg(info["upright_angle"])),
        )
        rewards.add_row(
            "balance",
            fmt(info["balance_reward"]),
            fmt(info["balance_reward_raw"]),
            fmt(self.W_BALANCE),
            fmt(np.rad2deg(info["roll_rate"])),
            fmt(np.rad2deg(self.MAX_ROLL_RATE)),
        )
        rewards.add_row(
            "torque",
            fmt(info["torque_reward"]),
            fmt(info["torque_reward_raw"]),
            fmt(self.W_TORQUE),
            fmt(np.linalg.norm([info["torque_L"], info["torque_R"]])/math.sqrt(2*self.MOTOR_STALL_TORQUE)),
        )

        pose = Table(title="Extra", show_header=True)
        pose.add_column("left", style="dim", width=7, justify="right")
        pose.add_column("right", style="dim", width=7, justify="right")

        pose.add_row(fmt(info["pwm_L"]), fmt(info["pwm_R"]))
        pose.add_row(fmt(info["torque_L"]), fmt(info["torque_R"]))

        layout = Layout()
        layout.split_row(Layout(rewards), Layout(pose))
        self._console.clear()
        self._console.print(layout)


def fmt(value):
    if isinstance(value, (float, np.floating)):
        return f"{value:.4f}"
    elif isinstance(value, int):
        return str(value) + "....."
    else:
        return str(value)




if __name__ == "__main__":
    video_filename = sys.argv[1] if len(sys.argv) > 1 else None
    env = make_segway_env(base_env_cls=lambda: DebugSegwayEnv(video_filename)) #ugh
    model = load_model_from_latest_checkpoint(env)
    obs, _ = env.reset()
    terminated = False

    try:
        while True:
            action, _ = model.predict(obs, deterministic=True)
            # Step the environment - NOW info CONTAINS REWARD COMPONENTS
            obs, reward, terminated, truncated, info = env.step(action)
            if video_filename is not None and truncated:
                break # ugh
    finally:
        env.close()
