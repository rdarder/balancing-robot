import pybullet as p
import pybullet_data
import time

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
    p.addUserDebugLine(
        [0, 0, 0],
        [0, 0.1, 0],
        [0, 1, 0],
        parentObjectUniqueId=segway_id,
        parentLinkIndex=i,
    )

# Control
max_torque = 0.00196133  # 20 g·cm / 10
p.setJointMotorControl2(segway_id, 0, p.TORQUE_CONTROL, force=max_torque)
p.setJointMotorControl2(segway_id, 1, p.TORQUE_CONTROL, force=max_torque)

# Simulate
try:
    while True:
        p.stepSimulation()
        pos, ori = p.getBasePositionAndOrientation(segway_id)
        joint_states = [p.getJointState(segway_id, j) for j in range(num_joints)]
        time.sleep(1 / 240)
        if p.isConnected() == 0:
            break
finally:
    p.disconnect()
