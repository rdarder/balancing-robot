import math
import time
# /home/rdarder/dev/balancing-robot/segway_env.py
import gymnasium as gym
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces

from pybullet_utils import JointsByName


class SegwayEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "direct"],
        "render_fps": 60,
    }

    MAX_SPEED = 1.5  # m/s

    # --- Motor & Gearbox Parameters ---
    MOTOR_MAX_VOLTAGE = 3.7
    MOTOR_RESISTANCE = 3.0  # ohms per phase
    MOTOR_STALL_TORQUE = 0.00196  # 20 g/cm
    GEAR_RATIO = 10.0
    MOTOR_KE = 0.00707
    MOTOR_KT = 0.00159
    OUTPUT_STALL_TORQUE = MOTOR_STALL_TORQUE * GEAR_RATIO

    # --- Weights for reward components ---
    W_UPRIGHT = 0.2
    W_SPEED = 0.6
    W_TURN = 0.2

    # --- Noise Parameters (for Domain Randomization) ---
    ANGLE_NOISE_STD_DEV_PITCH_ROLL = np.radians(0.25)
    ANGLE_NOISE_STD_DEV_YAW = np.radians(0.5)
    # ---
    #
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0

    def __init__(self, render_mode: str="direct", is_real_time: bool=False):
        super().__init__()

        self._client_id = self._connect_pybullet(render_mode)
        self._is_real_time = is_real_time
        self._setup_pybullet()
        self._load_model()
        # --- Spaces and State ---
        # Observation: [ax, ay, az, gx, gy, gz, pitch_noisy, roll_noisy, yaw_noisy, target_speed, target_turn]
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(11,), dtype=np.float32
        )
        # Action: [left_pwm, right_pwm] (normalized between -1 and 1)
        self.action_space = spaces.Box(low=self.ACTION_LOW, high=self.ACTION_HIGH, shape=(2,), dtype=np.float32)

        # Internal state variables
        self._prev_vel = np.zeros(3)
        self.target_speed = 0.0  # Target forward/backward speed command
        self.target_turn = 0.0  # Target turning rate command
        self.step_count = 0

    def _load_model(self):
        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client_id)
        self._initial_pos = [0, 0, 0.022]  # Slightly above ground
        self._initial_orientation = p.getQuaternionFromEuler(
            [-1.8, 0, 0]
        )  # Start upright

        self.segway_id = p.loadURDF(
            "segway.urdf",
            self._initial_pos,
            self._initial_orientation,
            physicsClientId=self._client_id,
        )
        self.joints = JointsByName(self.segway_id, self._client_id)
        self.left_wheel_joint = self.joints.by_name("left_drive")
        self.right_wheel_joint = self.joints.by_name("right_drive")

        self._unlock_motors()

    def _unlock_motors(self):
        p.setJointMotorControl2(
            self.segway_id,
            self.left_wheel_joint,
            p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=self._client_id,
        )
        p.setJointMotorControl2(
            self.segway_id,
            self.right_wheel_joint,
            p.VELOCITY_CONTROL,
            force=0,
            physicsClientId=self._client_id,
        )

    def _setup_pybullet(self):
        if not p.isConnected(self._client_id):
            raise ConnectionError("Failed to connect to PyBullet simulation.")

        # --- Simulation Setup ---
        p.setAdditionalSearchPath(
            pybullet_data.getDataPath(), physicsClientId=self._client_id
        )
        p.setGravity(0, 0, -9.81, physicsClientId=self._client_id)
        self.time_step = 1 / 240.0  # Ensure it's float
        p.setTimeStep(self.time_step, physicsClientId=self._client_id)
        # Optional: Improve physics simulation stability
        p.setPhysicsEngineParameter(
            numSolverIterations=10, physicsClientId=self._client_id
        )
        p.setPhysicsEngineParameter(numSubSteps=4, physicsClientId=self._client_id)

    def _connect_pybullet(self, render_mode):
        """Connects to PyBullet and returns the client ID."""
        if render_mode == "human":
            client = p.connect(p.GUI)
            p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0, physicsClientId=client)
            p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 1, physicsClientId=client)
        else:
            render_mode = "direct"
            client = p.connect(p.DIRECT)
        return client

    def _get_orientation(self):
        """Returns the current pitch angle in radians (rotation around world Y)."""
        _, ori_quat = p.getBasePositionAndOrientation(
            self.segway_id, physicsClientId=self._client_id
        )
        euler_angles = p.getEulerFromQuaternion(ori_quat)
        return euler_angles

    def get_rotation_velocities(self):
        """Returns the current yaw rate (angular velocity around world Z) in rad/s."""
        _, ang_vel = p.getBaseVelocity(self.segway_id, physicsClientId=self._client_id)
        return ang_vel  # Index 2 is angular velocity around Z

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        p.resetBasePositionAndOrientation(
            self.segway_id,
            self._initial_pos,
            self._initial_orientation,
            physicsClientId=self._client_id,
        )
        self._reset_joints()
        self._unlock_motors()

        # Reset internal state variables
        self.step_count = 0
        self.prev_vel = np.zeros(3)

        # Set control targets (speed, turn)
        self.target_speed = self._get_initial_target_speed(options)
        self.target_turn = self._get_initial_target_turn(options)
        self._last_step_timestamp = time.time()
        observation = self._get_obs()  # Get initial info dictionary
        return observation, {}

    def _get_initial_target_turn(self, options):
        if options and "target_turn" in options:
            return np.clip(
                options["target_turn"], -1.0, 1.0
            )  # Assuming turn target is normalized
        else:
            # Random turn target (e.g., normalized rate)
            return self.np_random.uniform(-1.0, 1.0)

    def _get_initial_target_speed(self, options):
        if options and "target_speed" in options:
            return np.clip(options["target_speed"], -self.MAX_SPEED, self.MAX_SPEED)
        else:
            # Random target speed within MAX_SPEED limits
            return self.np_random.uniform(-self.MAX_SPEED, self.MAX_SPEED)

    def _reset_joints(self):
        for j in range(p.getNumJoints(self.segway_id, physicsClientId=self._client_id)):
            p.resetJointState(
                self.segway_id,
                j,
                targetValue=0,
                targetVelocity=0,
                physicsClientId=self._client_id,
            )
            p.resetBaseVelocity(
                self.segway_id,
                linearVelocity=[0, 0, 0],
                angularVelocity=[0, 0, 0],
                physicsClientId=self._client_id,
            )

    def _get_obs(self):
        """Constructs the observation vector, adding noise to orientation."""
        sim_pos, sim_orientation_quat = p.getBasePositionAndOrientation(
            self.segway_id, physicsClientId=self._client_id
        )
        sim_velocity, sim_angular_velocity = p.getBaseVelocity(
            self.segway_id, physicsClientId=self._client_id
        )

        imu_angular_velocity = sim_angular_velocity
        imu_velocity = np.array(sim_velocity)

        # Calculate acceleration, handle potential division by zero if time_step is invalid
        if self.time_step > 1e-9:
            imu_accel = (imu_velocity - self.prev_vel) / self.time_step
        else:
            imu_accel = np.zeros(3)
        self.prev_vel = (
            imu_velocity  # Store current velocity for next step's calculation
        )

        # Add gravity component to accelerometer Z reading (IMUs measure proper acceleration)
        # Assuming Z is up, gravity is -9.81. IMU measures reaction force, so +9.81.
        imu_accel[2] += 9.81
        imu_orientation = p.getEulerFromQuaternion(sim_orientation_quat)

        # --- Control Targets ---
        # Ensure targets are clipped just in case they were set externally without clipping
        clipped_target_speed = np.clip(self.target_speed, -self.MAX_SPEED, self.MAX_SPEED)
        clipped_target_turn = np.clip(self.target_turn, -1.0, 1.0)
        targets = np.array([clipped_target_speed, clipped_target_turn], dtype=np.float32)

        # --- Concatenate Observation ---
        # Order: IMU accel (3), IMU angular velocity (3), Noisy Fused Angles (Roll, Pitch, Yaw) (3), Targets (2) = 11 elements
        ideal_observation = np.concatenate(
            [imu_accel, imu_angular_velocity, imu_orientation, targets],
            dtype=np.float32,
        )

        noisy_observation = ideal_observation * np.random.normal(
            1.0, 0.04, size=ideal_observation.shape
        )
        return noisy_observation

    def _calculate_single_motor_torque(
        self,
        pwm_signal,
        joint_id,
    ):
        """
        Calculate the final torque for a single motor based on PWM signal and current state.
        While in the simulation we can only apply a torque, the robot controller can only output a PWM duty cycle.
        Torque will decrease as the motor spins faster due to back-emf. This method simulates back-emf.
        """
        # Applied voltage
        voltage_applied = self.MOTOR_MAX_VOLTAGE * pwm_signal
        joint_state = p.getJointState(
            self.segway_id, joint_id, physicsClientId=self._client_id
        )
        wheel_speed_rad_s = joint_state[1]  # [1] is angular velocity over y

        # Calculate motor speed and back-EMF
        motor_speed_rad_s = wheel_speed_rad_s * self.GEAR_RATIO
        back_emf = self.MOTOR_KE * motor_speed_rad_s

        # Calculate motor torque
        effective_voltage = voltage_applied - back_emf
        motor_torque = self.MOTOR_KT * (effective_voltage / self.MOTOR_RESISTANCE)

        # Clip motor torque
        motor_torque = np.clip(
            motor_torque, -self.MOTOR_STALL_TORQUE, self.MOTOR_STALL_TORQUE
        )

        # Calculate and clip output torque
        output_torque = motor_torque * self.GEAR_RATIO
        final_torque = np.clip(
            output_torque, -self.OUTPUT_STALL_TORQUE, self.OUTPUT_STALL_TORQUE
        )

        return final_torque

    def _calculate_motor_torques(self, action):
        """
        Calculates the final torque to apply to each wheel based on the action (PWM signal),
        current wheel speeds, and motor characteristics (including back-EMF).
        Action components are expected to be in [-1, 1].
        """
        left_pwm_signal, right_pwm_signal = action

        left_final_torque = self._calculate_single_motor_torque(
            left_pwm_signal, self.left_wheel_joint
        )
        right_final_torque = self._calculate_single_motor_torque(
            right_pwm_signal, self.right_wheel_joint
        )

        return left_final_torque, right_final_torque

    def _apply_torque(self, joint_index, torque):
        p.setJointMotorControl2(
            bodyUniqueId=self.segway_id,
            jointIndex=joint_index,
            controlMode=p.TORQUE_CONTROL,
            force=torque,
            physicsClientId=self._client_id,
        )


    def step(self, action):
        # Clip action just in case it's outside the [-1, 1] range
        action = np.clip(action, self.ACTION_LOW, self.ACTION_HIGH)

        # Calculate torques based on action and current state
        left_final_torque, right_final_torque = self._calculate_motor_torques(action)

        self._apply_torque(self.left_wheel_joint, left_final_torque)
        self._apply_torque(self.right_wheel_joint, right_final_torque)
        self.step_count += 1
        observation = self._get_obs()  # Gets observation with noisy angles

        velocities, angular_velocities = p.getBaseVelocity(
            self.segway_id, physicsClientId=self._client_id
        )
        position_quat, orientation_quat = p.getBasePositionAndOrientation(
            self.segway_id, physicsClientId=self._client_id
        )

        reward_components = self._compute_reward(velocities, angular_velocities, position_quat, orientation_quat)
        total_reward = reward_components["total_reward"]
        terminated = self._check_termination(orientation_quat)
        truncated = self.step_count >= 2400
        info = self._make_info(reward_components, action, [left_final_torque, right_final_torque])
        p.stepSimulation(self._client_id)
        self._sleep_if_real_time()

        return observation, total_reward, terminated, truncated, info

    def _sleep_if_real_time(self):
        if self._is_real_time:
            elapsed = time.time() - self._last_step_timestamp
            if elapsed < self.time_step:
                time.sleep(self.time_step - elapsed)
            self._last_step_timestamp = time.time()


    def _check_termination(self, orientation_quat):
        """Checks if the episode should terminate due to falling."""
        ax, ay, az = p.getEulerFromQuaternion(orientation_quat)

        # Check pitch angle for falling over
        fall_threshold = np.radians(100.0)  # e.g., 100 degrees
        # Add a grace period at the start of the episode before checking fall condition
        grace_period_steps = 30
        fell_over = self.step_count > grace_period_steps and abs(ax) > fall_threshold
        return fell_over

    def _compute_reward(self, velocities, angular_velocities, position_quat, orientation_quat):
        """Computes the reward based on uprightness, speed tracking, and turn tracking."""

        tx, ty, tz  = angular_velocities
        ox, oy, oz = p.getEulerFromQuaternion(orientation_quat)
        vx, vy, vz = velocities

        orientation_inv = p.invertTransform([0,0,0], orientation_quat)[1]
        local_vel = p.multiplyTransforms([0,0,0], orientation_inv, velocities, [0,0,0,1])[0]
        forward_speed = local_vel[1]  # y component is forward

        target_speed = np.clip(self.target_speed, -self.MAX_SPEED, self.MAX_SPEED)

        # First check if the signs match (going in the right direction)
        same_direction = (forward_speed * target_speed >= 0) or (abs(forward_speed) < 0.05)

        # Base error is still the absolute difference
        speed_error = abs(forward_speed - target_speed)

        # Apply additional penalty for wrong direction
        if not same_direction:
            # Scale up the error when moving in wrong direction
            direction_penalty = 1.0 + abs(forward_speed) / self.MAX_SPEED * 2.0
            speed_error *= direction_penalty

        # Exponential reward: 1 for zero error, decays as error increases
        speed_reward_raw = np.exp(-2.0 * speed_error / self.MAX_SPEED)

        # --- Upright Reward ---
        # Reward for staying upright (cosine function of pitch)
        # Max angle considered 'stable' for reward calculation (e.g., +/- 90 degrees)
        horizontal_pitch = np.pi / 2.0
        # Cosine reward: 1 when upright (pitch=0), 0 at max_stable_pitch, negative beyond
        upright_reward_raw = (
            np.cos(ox) if abs(ox) < horizontal_pitch else -1.0
        )

        # --- Turn Tracking Reward ---
        # Target turn rate is self.target_turn (normalized, e.g., [-1, 1])
        # We need to scale self.target_turn to an actual expected yaw rate if it's normalized
        # Let's assume max turn rate corresponds to 1.0 (e.g., 2 rad/s)
        max_expected_turn_rate = (
            2.0  # rad/s corresponding to turn command of 1.0 or -1.0
        )
        target_turn_rate = np.clip(self.target_turn, -1.0, 1.0) * max_expected_turn_rate

        turn_error = abs(tz - target_turn_rate)
        # Exponential reward for turn tracking
        # Adjust scaling factor (e.g., 1.0) and normalization
        turn_reward_raw = np.exp(-1.0 * turn_error / max_expected_turn_rate)

        # --- Combine Rewards with Weights ---
        upright_reward = self.W_UPRIGHT * upright_reward_raw
        speed_reward = self.W_SPEED * speed_reward_raw
        turn_reward = self.W_TURN * turn_reward_raw

        total_reward = upright_reward + speed_reward + turn_reward

        # Return dictionary including components for logging/info
        return {
            "total_reward": total_reward,
            "speed_reward_raw": speed_reward_raw,
            "speed_reward": speed_reward,
            "turn_reward_raw": turn_reward_raw,
            "turn_reward": turn_reward,
            "upright_reward_raw": upright_reward_raw,
            "upright_reward": upright_reward,
            "forward_speed": forward_speed,
            "turn": tz,
            "upright_angle": ox
        }

    def _make_info(self, reward_components, pwm, torque):
        """Gathers supplementary information about the environment state."""
        info = {
            "step_count": self.step_count,
            "target_speed": self.target_speed,
            "target_turn": self.target_turn,
            "torque_L": torque[0],
            "torque_R": torque[1],
            "pwm_L": pwm[0],
            "pwm_R": pwm[1]
        }

        # Add reward components if available
        if reward_components:
            info.update(reward_components)  # Add all keys from the reward dict
        return info


    def close(self):
        """Closes the environment and disconnects from PyBullet."""
        if hasattr(self, "client") and self._client_id >= 0:
            try:
                if p.isConnected(self._client_id):
                    print(
                        f"Disconnecting SegwayEnv from PyBullet client {self._client_id}."
                    )
                    p.disconnect(physicsClientId=self._client_id)
                else:
                    print("SegwayEnv: PyBullet client already disconnected.")
            except p.error as e:
                print(f"Error during PyBullet disconnect: {e}")
            finally:
                self._client_id = -1  # Mark as disconnected
        else:
            print(
                "SegwayEnv: No active PyBullet connection to close or already disconnected."
            )
