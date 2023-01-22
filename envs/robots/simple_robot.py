from typing import Optional

import numpy as np
from gymnasium import spaces

from panda_gym.envs.core import PyBulletRobot
from panda_gym.pybullet import PyBullet


class SimpleRobot(PyBulletRobot):
    """SimpleRobot robot in PyBullet.
    Args:
        sim (PyBullet): Simulation instance.
        block_gripper (bool, optional): Whether the gripper is blocked. Defaults to False.
        base_position (np.ndarray, optionnal): Position of the base base of the robot, as (x, y, z). Defaults to (0, 0, 0).
        control_type (str, optional): "ee" to control end-effector displacement or "joints" to control joint angles.
            Defaults to "ee".
    """

    def __init__(
        self,
        sim: PyBullet,
        block_gripper: bool = False,
        base_position: Optional[np.ndarray] = None,
        control_type: str = "joints",
    ) -> None:
        base_position = base_position if base_position is not None else np.zeros(3)
        self.control_type = control_type
        n_action = 5   # 5 joints
        action_space = spaces.Box(-1.0, 1.0, shape=(n_action,), dtype=np.float32)
        super().__init__(
            sim,
            body_name="simple_robot",
            file_name="C:/Users/crypt/Dokumente/GitHub/panda-gym-tutorial/urdf/simple_robot.urdf",
            base_position=base_position,
            action_space=action_space,
            joint_indices=np.array([0, 1, 3, 5, 7]),
            joint_forces=np.array([40.0, 40.0, 40.0, 40.0, 40.0]),
        )

        self.neutral_joint_values = np.array([0.00, 0.00, 0.00, 0.00, 0.00])
        self.ee_link = 8

    def set_action(self, action: np.ndarray) -> None:
        # clip action to action space
        action = np.clip(action, self.action_space.low, self.action_space.high)
        # get current joint angles
        current_arm_joint_angles = np.array([self.get_joint_angle(joint=i) for i in range(5)])
        # move joints 
        target_arm_angles = current_arm_joint_angles + action
        # set joint angles
        self.control_joints(target_angles=target_arm_angles)

  
    def get_obs(self) -> np.ndarray:
        # end-effector position and velocity
        ee_position = np.array(self.get_ee_position())
        ee_velocity = np.array(self.get_ee_velocity())
        # fingers opening

        observation = np.concatenate((ee_position, ee_velocity))
        return observation

    def reset(self) -> None:
        self.set_joint_neutral()

    def set_joint_neutral(self) -> None:
        """Set the robot to its neutral pose."""
        self.set_joint_angles(self.neutral_joint_values)

    def get_ee_position(self) -> np.ndarray:
        """Returns the position of the ned-effector as (x, y, z)"""
        return self.get_link_position(self.ee_link)

    def get_ee_velocity(self) -> np.ndarray:
        """Returns the velocity of the end-effector as (vx, vy, vz)"""
        return self.get_link_velocity(self.ee_link)