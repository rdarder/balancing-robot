# /home/rdarder/dev/balancing-robot/segway_env.py
import gymnasium as gym
import pybullet as p
import pybullet_data
import numpy as np
from gymnasium import spaces
import time

from pybullet_utils import (
    get_joint_ids_by_name,
)


class SegwayEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "direct"],
        "render_fps": 60,
    }

    MAX_SPEED = 1.5
    MAX_TORQUE = 0.02

    def __init__(self, render_mode=None):
        super().__init__()

        self.render_mode = render_mode
        self.is_real_time = False

        # --- PyBullet Connection Setup ---
        # ... (rest of connection setup remains the same) ...
        if self.render_mode == "human":
            try:
                self.client = p.connect(p.GUI)
                p.configureDebugVisualizer(
                    p.COV_ENABLE_GUI, 0, physicsClientId=self.client
                )
                p.configureDebugVisualizer(
                    p.COV_ENABLE_SHADOWS, 1, physicsClientId=self.client
                )
                # ... other visualizer settings ...
            except p.error as e:
                print(
                    f"Warning: Could not connect via GUI, falling back to DIRECT. Error: {e}"
                )
                self.render_mode = "direct"
                self.client = p.connect(p.DIRECT)
        else:
            self.render_mode = "direct"
            self.client = p.connect(p.DIRECT)

        if not p.isConnected(self.client):
            raise ConnectionError("Failed to connect to PyBullet simulation.")
        print(
            f"SegwayEnv connected to PyBullet client {self.client} in {self.render_mode} mode."
        )

        # --- Simulation Setup ---
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self.client
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self.client)
        self.time_step = 1 / 240
        p.setTimeStep(self.time_step, physicsClientId=self.client)

        # --- Load URDF and Get Joints ---
        self.plane_id = p.loadURDF("plane.urdf", physicsClientId=self.client)
        self._initial_pos = [0, 0, 0.1]
        self.segway_id = p.loadURDF(
            "segway.urdf", self._initial_pos, physicsClientId=self.client
        )
        self.joint_ids_by_name = get_joint_ids_by_name(
            self.segway_id, physicsClientId=self.client
        )  # Pass client ID

        self.left_wheel_joint = self.joint_ids_by_name["left_drive"]
        self.right_wheel_joint = self.joint_ids_by_name["right_drive"]

        # --- Unlock Motors ---
        p.setJointMotorControl2(
            self.segway_id,
            self.left_wheel_joint,
            p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=self.client,
        )
        p.setJointMotorControl2(
            self.segway_id,
            self.right_wheel_joint,
            p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=self.client,
        )

        # --- Spaces and State ---
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(8,), dtype=np.float32
        )
        self.action_space = spaces.Box(low=-1, high=1, shape=(2,), dtype=np.float32)
        self.prev_vel = np.zeros(3)
        self.throttle = 0.0
        self.turn = 0.0
        self.step_count = 0

    # --- NEW STATE CALCULATION METHODS ---

    def get_pitch(self):
        """Returns the current pitch angle in radians."""
        try:
            _, ori = p.getBasePositionAndOrientation(
                self.segway_id, physicsClientId=self.client
            )
            return p.getEulerFromQuaternion(ori)[0]
        except p.error as e:
            print(f"Warning: PyBullet error getting pitch: {e}")
            return 0.0

    def get_yaw_rate(self):
        """Returns the current yaw rate (angular velocity around Z) in rad/s."""
        try:
            _, ang_vel = p.getBaseVelocity(self.segway_id, physicsClientId=self.client)
            return ang_vel[2]
        except p.error as e:
            print(f"Warning: PyBullet error getting yaw rate: {e}")
            return 0.0

    def get_ground_speed(self):
        """Calculates ground speed based on the average XY velocity of wheel links."""
        try:
            left_link_state = p.getLinkState(
                self.segway_id,
                self.left_wheel_joint,
                computeLinkVelocity=1,
                physicsClientId=self.client,
            )
            right_link_state = p.getLinkState(
                self.segway_id,
                self.right_wheel_joint,
                computeLinkVelocity=1,
                physicsClientId=self.client,
            )
            left_link_vel = left_link_state[6]
            right_link_vel = right_link_state[6]
            avg_vx = (left_link_vel[0] + right_link_vel[0]) / 2.0
            avg_vy = (left_link_vel[1] + right_link_vel[1]) / 2.0
            return np.sqrt(avg_vx**2 + avg_vy**2)
        except p.error as e:
            print(f"Warning: PyBullet error getting link states for ground speed: {e}")
            return 0.0

    # --- END NEW STATE CALCULATION METHODS ---

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # --- Reset Base State ---
        reset_orientation = p.getQuaternionFromEuler([1.8, 0, 0])
        reset_position = [0, 0, 0.022]
        p.resetBasePositionAndOrientation(
            self.segway_id,
            reset_position,
            reset_orientation,
            physicsClientId=self.client,
        )
        for j in range(p.getNumJoints(self.segway_id, physicsClientId=self.client)):
            p.resetJointState(
                self.segway_id,
                j,
                targetValue=0,
                targetVelocity=0,
                physicsClientId=self.client,
            )
        p.resetBaseVelocity(
            self.segway_id,
            linearVelocity=[0, 0, 0],
            angularVelocity=[0, 0, 0],
            physicsClientId=self.client,
        )

        # --- Unlock Motors ---
        p.setJointMotorControl2(
            self.segway_id,
            self.left_wheel_joint,
            p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=self.client,
        )
        p.setJointMotorControl2(
            self.segway_id,
            self.right_wheel_joint,
            p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=self.client,
        )

        # --- Reset Internal State ---
        self.step_count = 0
        self.prev_vel = np.zeros(3)

        # --- Set Targets ---
        if options and "throttle" in options:
            self.throttle = options["throttle"]
        else:
            self.throttle = self.np_random.uniform(-self.MAX_SPEED, self.MAX_SPEED)
        if options and "turn" in options:
            self.turn = options["turn"]
        else:
            self.turn = self.np_random.uniform(-1.0, 1.0)

        # --- Return Initial Observation ---
        observation = self._get_obs()
        info = {}
        return observation, info

    def _get_obs(self):
        # Get base velocity (linear, angular)
        # Note: We still need the full velocity here for the accelerometer calculation
        try:
            vel, ang_vel = p.getBaseVelocity(
                self.segway_id, physicsClientId=self.client
            )
        except p.error as e:
            print(f"Warning: PyBullet error getting base velocity for obs: {e}")
            vel, ang_vel = ([0, 0, 0], [0, 0, 0])  # Fallback

        # Gyroscope data (angular velocities)
        gyro_x, gyro_y, gyro_z = ang_vel  # rad/s

        # Accelerometer data (approximate linear acceleration)
        curr_vel = np.array(vel)
        if not hasattr(self, "prev_vel"):
            self.prev_vel = np.zeros(3)
        accel = (
            (curr_vel - self.prev_vel) / self.time_step
            if self.time_step > 0
            else np.zeros(3)
        )
        self.prev_vel = curr_vel

        # Combine into 6-DOF IMU reading
        imu = np.array(
            [accel[0], accel[1], accel[2], gyro_x, gyro_y, gyro_z], dtype=np.float32
        )

        # Concatenate IMU data with current control targets
        observation = np.concatenate(
            [imu, [self.throttle, self.turn]], dtype=np.float32
        )
        return observation

    def step(self, action):
        # --- Apply Action ---
        action = np.clip(action, self.action_space.low, self.action_space.high)
        left_pwm, right_pwm = action
        left_torque = self.MAX_TORQUE * left_pwm
        right_torque = self.MAX_TORQUE * right_pwm
        p.setJointMotorControl2(
            bodyUniqueId=self.segway_id,
            jointIndex=self.left_wheel_joint,
            controlMode=p.TORQUE_CONTROL,
            force=left_torque,
            physicsClientId=self.client,
        )
        p.setJointMotorControl2(
            bodyUniqueId=self.segway_id,
            jointIndex=self.right_wheel_joint,
            controlMode=p.TORQUE_CONTROL,
            force=right_torque,
            physicsClientId=self.client,
        )

        # --- Step Simulation ---
        if not self.is_real_time:
            p.stepSimulation(physicsClientId=self.client)
        self.step_count += 1

        # --- Get Results ---
        observation = self._get_obs()
        reward = self._compute_reward()  # Now uses the new methods internally
        terminated = self._check_termination()  # Now uses get_pitch() internally
        truncated = self.step_count >= 1000
        info = {}

        return observation, reward, terminated, truncated, info

    def _check_termination(self):
        """Checks if the episode should terminate."""
        # Use the new method to get pitch
        pitch = self.get_pitch()

        # Check position for falling through floor
        try:
            pos, _ = p.getBasePositionAndOrientation(
                self.segway_id, physicsClientId=self.client
            )
            below_ground = pos[2] < -0.1
        except p.error as e:
            print(
                f"Warning: PyBullet error getting position for termination check: {e}"
            )
            below_ground = False  # Assume not below ground if error occurs

        # Termination conditions
        fall_threshold = np.pi / 2.0
        grace_period_steps = 200
        fell_over = self.step_count > grace_period_steps and abs(pitch) > fall_threshold

        return fell_over or below_ground

    def _compute_reward(self):
        """Calculates the reward for the current state using helper methods."""
        # --- Use helper methods to get state values ---
        pitch = self.get_pitch()
        yaw_rate = self.get_yaw_rate()
        ground_speed = self.get_ground_speed()
        # ---

        # --- Reward Components ---
        # 1. Uprightness Reward:
        max_angle = np.pi / 2.0
        scaled_pitch = abs(pitch) * (np.pi / 2.0) / max_angle
        upright_reward = np.cos(scaled_pitch) if abs(pitch) < max_angle else -1.0

        # 2. Speed Matching Reward:
        target_speed = np.clip(self.throttle, -self.MAX_SPEED, self.MAX_SPEED)
        speed_error = abs(ground_speed - target_speed)
        speed_reward = np.exp(-2.0 * speed_error / self.MAX_SPEED)

        # 3. Turn Matching Reward:
        target_turn_rate = np.clip(self.turn, -2.0, 2.0)
        turn_error = abs(yaw_rate - target_turn_rate)
        max_expected_turn_rate = 1.0
        turn_reward = np.exp(-1.0 * turn_error / max_expected_turn_rate)

        # --- Combine Rewards ---
        w_upright = 1.0
        w_speed = 1.0
        w_turn = 1.0
        total_reward = (
            w_upright * upright_reward + w_speed * speed_reward + w_turn * turn_reward
        )

        return total_reward

    def render(self):
        # ... (render method remains the same) ...
        pass

    def close(self):
        # ... (close method remains the same) ...
        if hasattr(self, "client") and p.isConnected(self.client):
            print(f"Disconnecting SegwayEnv from PyBullet client {self.client}.")
            try:
                p.disconnect(physicsClientId=self.client)
            except p.error as e:
                print(f"Error during PyBullet disconnect: {e}")
        else:
            print(
                "SegwayEnv: No active PyBullet connection to close or already disconnected."
            )
        self.client = -1
