import pybullet as p


def get_joint_ids_by_name(model_id, physicsClientId=0):
    """Returns a dictionary mapping joint names to their IDs for a given model.

    Args:
        model_id: The ID of the model in the PyBullet simulation.
        physicsClientId: The ID of the physics client. Defaults to 0.

    Returns:
        A dictionary where keys are joint names (strings) and values are joint IDs (integers).
    """
    joint_map = {}
    for i in range(p.getNumJoints(model_id, physicsClientId)):
        joint_info = p.getJointInfo(model_id, i, physicsClientId)
        joint_name = joint_info[1].decode("utf-8")  # Name as string
        joint_id = joint_info[0]  # ID
        joint_map[joint_name] = joint_id
    return joint_map
