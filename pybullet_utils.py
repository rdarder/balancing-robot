import pybullet as p
from typing import Iterable

class JointsByName:
    """Utility for accessing joint IDs by name in a PyBullet model."""

    def __init__(self, model_id, client_id: int = 0):
        self._model_id = model_id
        self._client_id = client_id
        self.joint_ids_by_name = self._make_joint_ids_by_name()

    def by_name(self, joint_name: str) -> int:
        try:
            return self.joint_ids_by_name[joint_name]
        except KeyError:
            raise ValueError(
                f"Joint '{joint_name}' not found in model {self._model_id}"
            )

    def _make_joint_ids_by_name(self):
        joint_map = {}
        for i in range(p.getNumJoints(self._model_id, self._client_id)):
            joint_info = p.getJointInfo(self._model_id, i, self._client_id)
            joint_name = joint_info[1].decode("utf-8")  # Name as string
            joint_id = joint_info[0]  # ID
            joint_map[joint_name] = joint_id
        return joint_map


def get_non_fixed_joint_ids(model_id: int, client_id: int) -> Iterable[int]:
    for i in range(p.getNumJoints(model_id, client_id)):
        joint_info = p.getJointInfo(model_id, i, client_id)
        if joint_info[2] != p.JOINT_FIXED:
            yield joint_info[0]

def add_debug_lines(joint_id: int, model_id: int, client_id: int):
    print(joint_id)
    p.addUserDebugLine([0, 0, 0], [1, 0, 0], [1, 0, 0], parentLinkIndex=joint_id , parentObjectUniqueId=model_id, physicsClientId=client_id)  # X axis (red)
    p.addUserDebugLine([0, 0, 0], [0, 1, 0], [0, 1, 0], parentLinkIndex=joint_id, parentObjectUniqueId=model_id, physicsClientId=client_id)  # Y axis (green)
    p.addUserDebugLine([0, 0, 0], [0, 0, 1], [0, 0, 1], parentLinkIndex=joint_id, parentObjectUniqueId=model_id, physicsClientId=client_id)  # Z axis (blue)
