import pybullet as p
import pybullet_data
import numpy as np

from pybullet_utils import get_joint_ids_by_name


p.connect(p.DIRECT)
p.setAdditionalSearchPath(pybullet_data.getDataPath())  # used by loadURDF
p.setGravity(0, 0, -9.81)
p.loadURDF("plane.urdf")
segway_id = p.loadURDF("./segway.urdf", [0, 0, 0.1])
joints = get_joint_ids_by_name(segway_id)
left_drive = joints["left_drive"]
right_drive = joints["right_drive"]

p.setJointMotorControl2(segway_id, left_drive, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(segway_id, right_drive, p.VELOCITY_CONTROL, force=0)
max_torque = 0.02
torque_per_step = max_torque / 2000
# Run 2s
for step in range(5000):  # 2s at 240 Hz
    torque = min(max_torque, max_torque / 2 + torque_per_step * step)
    # Max torque
    p.setJointMotorControl2(segway_id, left_drive, p.TORQUE_CONTROL, force=torque)
    p.setJointMotorControl2(segway_id, right_drive, p.TORQUE_CONTROL, force=torque)
    p.stepSimulation()
vel, ang_vel = p.getBaseVelocity(segway_id)
speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
print(f"Max speed: {speed} m/s")
print(f"Angular velocity: {ang_vel} rad/s")
p.disconnect()
