import math
import time
from collections import deque
import pkg_resources

# /home/rdarder/dev/balancing-robot/segway_env.py
import gymnasium as gym
from gymnasium.wrappers.stateful_observation import FrameStackObservation
import numpy as np
import pybullet as p
import pybullet_data
from gymnasium import spaces
from gymnasium.wrappers import NormalizeObservation, FrameStackObservation

from balancing_robot.rl.pybullet_utils import JointsByName, LinksByName


class SegwayEnv(gym.Env):
    metadata = {
        "render_modes": ["human", "direct"],
        "render_fps": 60,
    }

    MAX_SPEED = 1.5  # m/s
    MAX_ROLL_RATE = 1.0  # rad/s

    # --- Motor & Gearbox Parameters ---
    MOTOR_MAX_VOLTAGE = 3.7
    MOTOR_RESISTANCE = 3.0  # ohms per phase
    MOTOR_STALL_TORQUE = 0.00196  # 20 g/cm
    GEAR_RATIO = 37.0 / 8
    MOTOR_KE = 0.00707
    MOTOR_KT = 0.00159
    OUTPUT_STALL_TORQUE = MOTOR_STALL_TORQUE * GEAR_RATIO

    # --- Weights for reward components ---
    W_UPRIGHT = 0.2
    W_SPEED = 0.2
    W_TURN = 0.2
    W_BALANCE = 0.2
    W_TORQUE = 0.2

    # --- Noise Parameters (for Domain Randomization) ---
    ANGLE_NOISE_STD_DEV_PITCH_ROLL = np.radians(0.1)
    ANGLE_NOISE_STD_DEV_YAW = np.radians(0.2)
    # ---
    #
    ACTION_LOW = -1.0
    ACTION_HIGH = 1.0
    FRAME_STACK_SIZE = 8


    STEPS_PER_SECOND = 50
    TRUNCATE_AT_SECONDS = 20.0
    TIME_STEP = 1.0 / STEPS_PER_SECOND
    TRUNCATE_AT_STEPS = TRUNCATE_AT_SECONDS * STEPS_PER_SECOND

    GRACE_PERIOD_SECONDS = 0.4
    GRACE_PERIOD_STEPS = int(GRACE_PERIOD_SECONDS * STEPS_PER_SECOND)

    CHASSIS_CENTER_TO_AXLE_DISTANCE = 0.021 #meters, we should extract this from the model for single source of truth
    WHEEL_RADIUS = 0.027

    def __init__(self, render_mode: str = "direct", is_real_time: bool = False):
        super().__init__()

        self._client_id = self._connect_pybullet(render_mode)
        self._is_real_time = is_real_time
        self._setup_pybullet()
        self._load_model()
        # --- Spaces and State ---
        # Observation: [ax, ay, az, gx, gy, gz, pitch_noisy, roll_noisy, yaw_noisy, target_speed, target_turn]
        self.observation_space = spaces.Box(
            low=-1.0, high=-1.0, shape=(8,), dtype=np.float32
        )
        # Action: [left_pwm, right_pwm] (normalized between -1 and 1)
        self.action_space = spaces.Box(
            low=self.ACTION_LOW, high=self.ACTION_HIGH, shape=(2,), dtype=np.float32
        )

        # Internal state variables
        self._prev_vel = np.zeros(3)
        self.target_speed = 0.0  # Target forward/backward speed command
        self.target_turn = 0.0  # Target turning rate command
        self.step_count = 0
        self.last_upright_timestep = 0
        self.last_action = [0.0, 0.0]
        # the stack should behave like a ring buffer of size self.FRAME_STACK_SIZE
        # self._observation_stack = deque(maxlen=self.FRAME_STACK_SIZE)

    def _load_model(self):
        self._plane_id = p.loadURDF("plane.urdf", physicsClientId=self._client_id)
        model_path = pkg_resources.resource_filename("balancing_robot", "assets/segway.urdf")
        self.segway_id = p.loadURDF(
            model_path,
            [0,0,0],
            [0,0,0,1],
            physicsClientId=self._client_id,
        )
        self.joints = JointsByName(self.segway_id, self._client_id)
        self.links = LinksByName(self.segway_id, self._client_id)
        self.left_wheel_joint = self.joints.by_name("left_drive")
        self.right_wheel_joint = self.joints.by_name("right_drive")
        self.imu_link = self.links.by_name("imu")

        self._unlock_motors()

    def _make_initial_position_and_orientation(self) -> tuple[np.ndarray, np.ndarray]:
        x_angle = self.np_random.uniform(-1.8, 1.8)
        z_angle = self.np_random.uniform(-math.pi, math.pi)
        euler = np.array([x_angle, 0.0, z_angle], dtype=np.float32)
        orientation = p.getQuaternionFromEuler(euler)
        position_z = self.WHEEL_RADIUS + self.CHASSIS_CENTER_TO_AXLE_DISTANCE * math.cos(x_angle)
        position = np.array([0, 0, position_z], dtype=np.float32)
        return position, orientation

    def _unlock_motors(self):
        p.setJointMotorControl2(
            self.segway_id,
            self.left_wheel_joint,
            p.VELOCITY_CONTROL,
            force=self.MOTOR_STALL_TORQUE/10,
            physicsClientId=self._client_id,
        )
        p.setJointMotorControl2(
            self.segway_id,
            self.right_wheel_joint,
            p.VELOCITY_CONTROL,
            force=self.MOTOR_STALL_TORQUE/10,
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
        p.setTimeStep(self.TIME_STEP, physicsClientId=self._client_id)
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
        position, orientation = self._make_initial_position_and_orientation()
        p.resetBasePositionAndOrientation(
            self.segway_id,
            position,
            orientation,
            physicsClientId=self._client_id,
        )
        self._reset_joints()
        self._unlock_motors()

        # Reset internal state variables
        self.step_count = 0
        self.prev_vel = np.zeros(3)

        self.last_action = self.np_random.uniform(low=-1.0, high=1.0, size=(2,))
        # Set control targets (speed, turn)
        self.target_speed = self._get_initial_target_speed(options)
        self.target_turn = self._get_initial_target_turn(options)
        self._last_step_timestamp = time.time()
        observation = self._get_obs()  # Get initial info dictionary
        # self._observation_stack.clear()  # Clear any previous data
        # for _ in range(self.FRAME_STACK_SIZE):
        #     self._observation_stack.append(observation)

        # return np.array(list(self._observation_stack)), {}
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
            return np.clip(options["target_speed"], -1, 1)
        else:
            # Random target speed within MAX_SPEED limits
            return self.np_random.uniform(-1, 1)

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
        """Constructs the observation vector using data from the IMU link."""
        # Get IMU link state
        imu_state = p.getLinkState(
            self.segway_id,
            self.imu_link,
            computeLinkVelocity=1,  # Need to compute velocities
            physicsClientId=self._client_id,
        )

        # Extract position, orientation, and velocities
        sim_orientation_quat = imu_state[1]  # world orientation
        sim_velocity = imu_state[6]  # linear velocity
        sim_angular_velocity = imu_state[7]  # angular velocity

        # Convert to IMU readings
        imu_angular_velocity = sim_angular_velocity
        imu_velocity = np.array(sim_velocity)

        # Calculate acceleration
        imu_accel = (imu_velocity - self.prev_vel) / self.TIME_STEP
        self.prev_vel = (
            imu_velocity  # Store current velocity for next step's calculation
        )

        # Add gravity component to accelerometer Z reading
        # Transform gravity vector from world frame to IMU frame
        gravity_world = np.array([0, 0, 9.81])  # Gravity in world frame (positive Z up)

        # Convert orientation quaternion to rotation matrix
        rot_matrix = p.getMatrixFromQuaternion(sim_orientation_quat)
        rot_matrix = np.array(rot_matrix).reshape(3, 3)

        # Transpose rotation matrix to convert from world to IMU frame
        rot_matrix_transposed = rot_matrix.T

        # Transform gravity to IMU frame and add to acceleration
        gravity_imu = rot_matrix_transposed.dot(gravity_world)
        imu_accel += gravity_imu

        # Stack imu data
        imu_data = np.concatenate((imu_accel, imu_angular_velocity))

        # Add sensor noise
        noisy_imu_data = imu_data + np.random.normal(0, 0.05, size=imu_data.shape) # Assuming standard deviation of 0.05

        normalized_imu_data = normalize_imu_data(noisy_imu_data)

        normalized_noisy_observation = np.concatenate(
            [normalized_imu_data, np.array([self.target_speed, self.target_turn])],
            dtype=np.float32,
        )

        clipped_normalized_noisy_obs = np.clip(normalized_noisy_observation, -1.0, 1.0)
        return clipped_normalized_noisy_obs


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
        observation = self._get_obs()
        # self._observation_stack.append()

        velocities, angular_velocities = p.getBaseVelocity(
            self.segway_id, physicsClientId=self._client_id
        )
        position_quat, orientation_quat = p.getBasePositionAndOrientation(
            self.segway_id, physicsClientId=self._client_id
        )

        reward_components = self._compute_reward(
            velocities, angular_velocities, position_quat, orientation_quat, action
        )
        total_reward = reward_components["total_reward"]
        terminated = self._check_termination(orientation_quat)
        truncated = self.step_count >= self.TRUNCATE_AT_STEPS
        info = self._make_info(
            reward_components, action, [left_final_torque, right_final_torque]
        )
        p.stepSimulation(self._client_id)
        self._sleep_if_real_time()
        self.last_action = action

        # observation = np.array(list(self._observation_stack))
        return observation, total_reward, terminated, truncated, info

    def _sleep_if_real_time(self):
        if self._is_real_time:
            elapsed = time.time() - self._last_step_timestamp
            if elapsed < self.TIME_STEP:
                time.sleep(self.TIME_STEP - elapsed)
            self._last_step_timestamp = time.time()

    def _check_termination(self, orientation_quat):
        """Checks if the episode should terminate due to falling."""
        ax, ay, az = p.getEulerFromQuaternion(orientation_quat)

        # Check pitch angle for falling over
        fall_threshold = np.radians(90.0)
        if abs(ax) > fall_threshold:
            return self.step_count - self.last_upright_timestep > self.GRACE_PERIOD_STEPS
        else:
            self.last_upright_timestep = self.step_count
            return False

    def _compute_reward(
        self, velocities, angular_velocities, position_quat, orientation_quat, action
    ):
        """Computes the reward based on uprightness, speed tracking, turn tracking, and balance stability."""

        tx, ty, tz = angular_velocities
        ox, oy, oz = p.getEulerFromQuaternion(orientation_quat)
        vx, vy, vz = velocities

        orientation_inv = p.invertTransform([0, 0, 0], orientation_quat)[1]
        local_vel = p.multiplyTransforms(
            [0, 0, 0], orientation_inv, velocities, [0, 0, 0, 1]
        )[0]
        forward_speed = local_vel[1]  # y component is forward

        # Transform angular velocities to local frame
        local_angular_vel = p.multiplyTransforms(
            [0, 0, 0], orientation_inv, angular_velocities, [0, 0, 0, 1]
        )[0]
        local_tx = local_angular_vel[
            0
        ]  # Local x-axis angular velocity (rolling oscillation)

        upright_reward_raw = self._get_upright_reward_raw(abs(ox))

        speed_reward_raw = self._get_speed_reward_raw(forward_speed)

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
        # Turn reward is subject to being at the right speed. if we're turning but we're not moving or not near the right speed,
        # there should be less reward.
        turn_reward_raw = np.exp(-6 * turn_error / max_expected_turn_rate)

        # --- Balance Stability Reward ---
        # Penalize oscillations in roll (angular velocity around local x-axis)  # rad/s - adjust based on expected behavior
        roll_rate = abs(local_tx)
        balance_reward_raw = np.exp(-4 * roll_rate / self.MAX_ROLL_RATE)

        torque_reward_raw = 0.0

        # --- Combine Rewards with Weights ---
        # Besides just weighting the rewards, we're composing them to form
        # a hierarchical or progressive set of goals. For example, we only
        # consider the speed reward proportionally to how well we're achieving the
        # upright reward. Turning reward makes no sense if we're not upright or
        # if we're not going at the speed we're intending (that's debatable though.)
        upright_reward = self.W_UPRIGHT * upright_reward_raw
        torque_reward = self.W_TORQUE * torque_reward_raw
        speed_reward = self.W_SPEED * speed_reward_raw * upright_reward_raw
        turn_reward = self.W_TURN * turn_reward_raw * upright_reward_raw
        balance_reward = self.W_BALANCE * balance_reward_raw * upright_reward_raw

        total_reward = upright_reward + speed_reward + turn_reward + balance_reward

        # Return dictionary including components for logging/info
        return {
            "total_reward": total_reward,
            "speed_reward_raw": speed_reward_raw,
            "speed_reward": speed_reward,
            "turn_reward_raw": turn_reward_raw,
            "turn_reward": turn_reward,
            "upright_reward_raw": upright_reward_raw,
            "upright_reward": upright_reward,
            "balance_reward_raw": balance_reward_raw,
            "balance_reward": balance_reward,
            "torque_reward_raw": torque_reward_raw,
            "torque_reward": torque_reward,
            "forward_speed": forward_speed,
            "turn": tz,
            "upright_angle": ox,
            "roll_rate": roll_rate,
        }

    def _get_upright_reward_raw(self, ox):
        min_tilt=np.radians(15)
        max_tilt=np.radians(80)
        tilt_range = max_tilt - min_tilt

        if ox < min_tilt:
            return 1.0
        elif ox > max_tilt:
            return 0.0
        else:
            return (math.cos((ox - min_tilt) * math.pi / tilt_range) + 1)/2


    def _get_speed_reward_raw(self, forward_speed):
        target_speed_ratio = np.clip(self.target_speed, -1.0, 1.0)

        target_speed  = target_speed_ratio * self.MAX_SPEED

        # Calculate the speed error (difference)
        speed_error = abs(forward_speed - target_speed)

        # Define the maximum possible error magnitude.
        # If target is +MAX_SPEED, worst case is -MAX_SPEED, difference is 2*MAX_SPEED.
        # Add a small epsilon for numerical stability if MAX_SPEED could be 0.
        max_possible_error = 2.0 * self.MAX_SPEED + 1e-9

        speed_reward_raw = math.exp(-6 * (speed_error / max_possible_error))
        return speed_reward_raw


    def _make_info(self, reward_components, pwm, torque):
        """Gathers supplementary information about the environment state."""
        info = {
            "step_count": self.step_count,
            "target_speed": self.target_speed,
            "target_turn": self.target_turn,
            "torque_L": torque[0],
            "torque_R": torque[1],
            "pwm_L": pwm[0],
            "pwm_R": pwm[1],
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


def normalize_imu_data(imu_data: np.ndarray) -> np.ndarray:
    """Normalizes IMU data to the range [-1, 1].

    Args:
        imu_data: A numpy array of shape (6,) containing the IMU data.
                  The first 3 elements are accelerometer readings (m/s^2),
                  and the last 3 elements are gyro readings (rad/s).

    Returns:
        A numpy array of shape (6,) containing the normalized IMU data.
    """
    acc_range = 4 * 9.81  # +-4g, where g = 9.81 m/s^2
    gyro_range = 500  # +-500 deg/s
    gyro_range_rad = np.deg2rad(gyro_range)  # Convert deg/s to rad/s

    normalized_acc = imu_data[:3] / acc_range
    normalized_gyro = imu_data[3:] / gyro_range_rad

    return np.concatenate((normalized_acc, normalized_gyro))

STACK_SIZE=16
def make_segway_env(base_env_cls=SegwayEnv):
    env = base_env_cls() # awful, show_model should be an env wrapper.
    stacked = FrameStackObservation(env,stack_size=STACK_SIZE)
    return stacked
