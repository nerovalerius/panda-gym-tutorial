from panda_gym.envs.core import RobotTaskEnv
from envs.robots.simple_robot import SimpleRobot
from envs.tasks.simple_reach import SimpleReach
from panda_gym.pybullet import PyBullet


class SimpleEnv(RobotTaskEnv):
    """Pick and Place task wih SimpleRobot.
    Args:
        render_mode (str, optional): Render mode. Defaults to "rgb_array".
        reward_type (str, optional): "sparse" or "dense". Defaults to "sparse".
        control_type (str, optional): "ee" to control end-effector position or "joints" to control joint values.
            Defaults to "ee".
        render (bool, optional): Deprecated: This argument is deprecated and will be removed in a future
            version. Use the render_mode argument instead.
    """

    def __init__(self, render_mode):
        sim = PyBullet(render_mode=render_mode)
        robot = SimpleRobot(sim, control_type="joints", base_position=[-0.45, 0, 0])
        task = SimpleReach(sim, robot.get_ee_position)
        super().__init__(robot, task)