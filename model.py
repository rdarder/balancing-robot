import pybullet as p
import pybullet_data
import time
import numpy as np

from pybullet_utils import get_joint_ids_by_name

# Connect
p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.setGravity(0, 0, -9.81)
p.setTimeStep(1 / 240)

# Load
plane_id = p.loadURDF("plane.urdf")
segway_id = p.loadURDF("segway.urdf", [0, 0, 0.1])

# Debug info
print(f"Plane ID: {plane_id}, Segway ID: {segway_id}")
num_joints = p.getNumJoints(segway_id)
print(f"Number of Joints: {num_joints}")
for i in range(num_joints):
    joint_info = p.getJointInfo(segway_id, i)
    print(f"Joint {i}: Name={joint_info[1].decode('utf-8')}, Type={joint_info[2]}")
pos, ori = p.getBasePositionAndOrientation(segway_id)
print(f"Base Pos: {pos}, Ori: {p.getEulerFromQuaternion(ori)}")

p.addUserDebugLine([0, 0, 0], [0.1, 0, 0], [1, 0, 0])  # x-axis, red
p.addUserDebugLine([0, 0, 0], [0, 0.1, 0], [0, 1, 0])  # y-axis, green
p.addUserDebugLine([0, 0, 0], [0, 0, 0.1], [0, 0, 1])  # z-axis, blue

for i in range(num_joints):
    # y axis
    p.addUserDebugLine(
        [0, 0, 0],
        [0, 0.1, 0],
        [0, 1, 0],
        parentObjectUniqueId=segway_id,
        parentLinkIndex=i,
    )
    # z axis
    p.addUserDebugLine(
        [0, 0, 0],
        [0, 0, 0.1],
        [0, 0, 1],
        parentObjectUniqueId=segway_id,
        parentLinkIndex=i,
    )
    # x axis
    p.addUserDebugLine(
        [0, 0, 0],
        [0.1, 0, 0],
        [1, 0, 0],
        parentObjectUniqueId=segway_id,
        parentLinkIndex=i,
    )

orientation = p.getQuaternionFromEuler([-1.8, 0, 0])
p.resetBasePositionAndOrientation(segway_id, [0, 0, 0.022], orientation)

joints = get_joint_ids_by_name(segway_id)
left_drive = joints["left_drive"]
right_drive = joints["right_drive"]

# Control
max_torque = 0.02  # 20 g·cm * 10
torque_per_step = max_torque / 2000

p.setJointMotorControl2(segway_id, left_drive, p.VELOCITY_CONTROL, force=0)
p.setJointMotorControl2(segway_id, right_drive, p.VELOCITY_CONTROL, force=0)

# max_velocity = 5.0  #
# p.setJointMotorControl2(segway_id, 0, p.VELOCITY_CONTROL, targetVelocity=max_velocity)
# p.setJointMotorControl2(segway_id, 1, p.VELOCITY_CONTROL, targetVelocity=max_velocity)


step = 0
# Simulate
try:
    while True:
        step += 1
        torque = min(max_torque, torque_per_step * step)

        p.setJointMotorControl2(segway_id, left_drive, p.TORQUE_CONTROL, force=torque)
        p.setJointMotorControl2(
            segway_id, right_drive, p.TORQUE_CONTROL, force=torque * 0.95
        )
        p.stepSimulation()
        pos, ori = p.getBasePositionAndOrientation(segway_id)

        p.resetDebugVisualizerCamera(
            cameraDistance=0.3,  # Close view
            cameraYaw=60,  # Behind robot
            cameraPitch=-20,  # Slight angle down
            cameraTargetPosition=pos,  # Follow chassis
        )
        joint_states = [p.getJointState(segway_id, j) for j in range(num_joints)]
        orientation_angles = p.getEulerFromQuaternion(ori)

        if step % 100 == 0:
            vel, angle_vel = p.getBaseVelocity(segway_id)
            speed = np.sqrt(vel[0] ** 2 + vel[1] ** 2)
            print(
                f"Step: {step}, pos: {pos}, Speed: {speed}, Angle Vel: {angle_vel}, Orientation: {orientation_angles}"
            )
        if p.isConnected() == 0:
            break
        time.sleep(1 / 240)
finally:
    p.disconnect()
