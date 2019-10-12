from vrep_env import vrep_env, vrep
import os
import time
import pickle
import gym
from baselines.gail.vrep_env_base import UR5VrepEnvBase
from baselines.gail.vrep_ur_env import UR5VrepEnv, UR5VrepEnvConcat, tipcoor
from gym import spaces
from gym.utils import seeding
import numpy as np
import autograd.numpy as anp
from autograd import jacobian

pi = np.pi


def ur5fk(thetas):
    thetas_0 = anp.array([0, pi / 2, 0, pi / 2, pi])
    thetas = thetas + thetas_0
    #thetas = thetas._value
    d0 = 0.3
    d1 = 8.92e-2
    d2 = 0.11
    d5 = 9.475e-2
    #d6 = 7.495e-2
    d6 = 1.1495e-1
    a2 = 4.251e-1
    a3 = 3.9215e-1
    #All = np.zeros((6, 4, 4))
    #All[:, 3, 3] = 1
    A1 = anp.array([[anp.cos(thetas[0]), -anp.sin(thetas[0]), 0, 0],
                   [anp.sin(thetas[0]), anp.cos(thetas[0]), 0, 0],
                   [0, 0, 1, d1], [0, 0, 0, 1]])
    A2 = anp.array([[anp.cos(thetas[1]), -anp.sin(thetas[1]), 0, 0],
                   [0, 0, -1, -d2],
                   [anp.sin(thetas[1]), anp.cos(thetas[1]), 0, 0],
                   [0, 0, 0, 1]])
    A3 = anp.array([[anp.cos(thetas[2]), -anp.sin(thetas[2]), 0, a2],
                   [anp.sin(thetas[2]), anp.cos(thetas[2]), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    A4 = anp.array([[anp.cos(thetas[3]), -anp.sin(thetas[3]), 0, a3],
                   [anp.sin(thetas[3]), anp.cos(thetas[3]), 0, 0],
                   [0, 0, 1, 0], [0, 0, 0, 1]])
    A5 = anp.array([[anp.cos(thetas[4]), -anp.sin(thetas[4]), 0, 0],
                   [0, 0, -1, -d5],
                   [anp.sin(thetas[4]), anp.cos(thetas[4]), 0, 0],
                   [0, 0, 0, 1]])
    A6 = anp.array([[1, 0, 0, 0], [0, 1, 0, -d6], [0, 0, 1, 0], [0, 0, 0, 1]])

    A0 = anp.zeros((4, 4))
    A0[0, 1] = 1
    A0[1, 0] = -1
    A0[2, 2] = 1
    A0[2, 3] = d0
    A0[3, 3] = 1
    #A0[2, 3] = 0
    A = A0@A1@A2@A3@A4@A5@A6
    eular = anp.array([anp.arctan2(A[2, 1], A[2, 2]), anp.arctan2(-A[2, 0], anp.sqrt(A[2, 1]**2+A[2, 2]**2)),
                      anp.arctan2(A[1, 0], A[0, 0])])
    return anp.concatenate([A[:3, 3], eular])


class UR5VrepDisEnv(UR5VrepEnvConcat):
    def __init__(self,
                 server_addr='127.0.0.1',
                 server_port=19997,
                 scene_path=None,
                 l2_thresh=0.1,
                 random_seed=0,
                 dof=5,
                 enable_cameras=False,):
        super(UR5VrepDisEnv, self).__init__(server_addr, server_port, scene_path, l2_thresh, random_seed, dof)
        self.action_space = spaces.Discrete(self.dof)

    def _config(self):
        return self.observation['joint']

    def _make_action(self, a):
        """Send action to v-rep
        """
        cfg = self._config()
        newa = a + cfg
        self.set_joints(newa)
        if (self.observation_space.spaces['joint'].low < newa).all() and (
                self.observation_space.spaces['joint'].high > newa).all():
            return True
        else:
            return False


class UR5VrepEnvKine(UR5VrepEnv):
    def __init__(self,
                 server_addr='127.0.0.1',
                 server_port=19997,
                 scene_path=None,
                 l2_thresh=0.1,
                 random_seed=0,
                 dof=5,
                 enable_cameras=False,):
        super(UR5VrepEnvKine, self).__init__(server_addr, server_port, scene_path, l2_thresh, random_seed, dof, enable_cameras)
        self.jacob = jacobian(ur5fk)
        #self.action_space = spaces.Box(low=-0.12 * joint_space, high=0.12 * joint_space)
        self._make_obs_space()

    def _make_action(self, a):
        """Send action to v-rep
        """
        cfg = self._config()
        newa = a + cfg
        self.set_joints(newa)
        if (self.observation_space.low[:self.dof] < newa).all() and (
                self.observation_space.high[:self.dof] > newa).all():
            return True
        else:
            return False

    def _make_obs_space(self):
        joint_lbound = np.array([-2 * pi / 3, -pi / 2, -pi, -pi / 2, 0])
        joint_hbound = np.array([2 * pi / 3, pi / 6, 0, pi / 2, pi])
        obstacle_pos_lbound = np.array([-5, -5, 0])
        obstalce_pos_hbound = np.array([5, 5, 2])
        pos_lbound = np.array([-1.2, -1.2, -0.2]*5)
        pos_hbound = np.array([1.2, 1.2, 1.3] * 6)
        self.observation_space = gym.spaces.Box(low=np.concatenate([joint_lbound, joint_lbound, obstacle_pos_lbound, pos_lbound, 0]),
                                                high=np.concatenate([joint_hbound, joint_hbound, obstalce_pos_hbound, pos_hbound, 2]))

    def _make_observation(self):
        joint_angles = [self.obj_get_joint_angle(joint) for joint in self.oh_joint]
        self.distance = self.read_distance(self.distance_handle)
        ps = tipcoor(self._config())
        self.observation = np.concatenate([np.array(joint_angles).astype('float32'),
                                           self.target_joint_pos,
                                           self.obstacle_pos,
                                           ps,
                                           np.array([self.distance])
                                           ])

        return self.observation

    def _action_process(self, ac):
        J = self.jacob(ac)
        ac = np.linalg.pinv(J)@ac
        ac = np.clip(ac, self.action_space.low, self.action_space.high)
        return ac