import os
from baselines.gail.vrep_ur_env import UR5VrepEnv, tipcoor
import numpy as np


env = UR5VrepEnv()
env.reset()
env.set_joints(env.target_joint_pos)
goal_tip = tipcoor(np.concatenate((env.target_joint_pos, np.zeros(6-env.dof))))
env.obj_set_position(env.tip, goal_tip)
env.close()
