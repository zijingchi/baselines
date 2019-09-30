import gym
from baselines.gail.vrep_ur_env import UR5VrepEnvConcat
import numpy as np

pi = np.pi


class UR5VrepEnvDis(UR5VrepEnvConcat):
    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            l2_thresh=0.1,
            random_seed=0,
            dof=5,
            num_per_dim=5,
            discrete_stepsize=0.04
    ):
        super(UR5VrepEnvDis, self).__init__(
            server_addr,
            server_port,
            scene_path,
            l2_thresh,
            random_seed,
            dof)
        #self.action_space = gym.spaces.MultiDiscrete(num_per_dim*np.ones(dof))
        self.num = num_per_dim
        self.action_space = gym.spaces.Discrete(num_per_dim**dof)
        self.step_size = discrete_stepsize

    def _fiv(self, nind):
        res = [0 for _ in range(self.dof)]
        for i in range(self.dof)[::-1]:
            res[i] = nind%self.num
            nind = nind//self.num
        return res

    def _action_process(self, ac):
        ac = self._fiv(ac)
        discrete_action = (ac - (self.num-1)/2 * np.ones(self.dof))
        newa = np.concatenate([discrete_action, np.zeros(5 - self.dof)])*self.step_size
        cfg = self._config()
        newa += self.l2_thresh * (self.target_joint_pos - cfg) / np.linalg.norm(self.target_joint_pos - cfg)
        return newa

    def step(self, ac):
        # self._make_observation()
        ac = self._action_process(ac)
        invalid = not self._make_action(ac)
        self.step_simulation()
        self._make_observation()
        self.collision_check = self.read_collision(self.collision_handle) or abs(self.observation[2])>5*pi/6

        cfg = self._config()
        done = self._angle_dis(cfg, self.target_joint_pos,
                               5) < 1.5 * self.l2_thresh or self.collision_check or invalid

        reward = self.compute_reward(cfg, ac)
        info = {}
        if reward > 0.8 and done:
            info["status"] = 'reach'
        elif reward < 0.8 and done:
            info["status"] = 'collide'
        else:
            info["status"] = 'running'
        return self.observation, reward, done, info


class UR5VrepEnvKin(UR5VrepEnvConcat):
    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            l2_thresh=0.1,
            random_seed=0,
            dof=5,
            num_per_dim=5,
            discrete_stepsize=0.04
    ):
        super(UR5VrepEnvKin, self).__init__(
            server_addr,
            server_port,
            scene_path,
            l2_thresh,
            random_seed,
            dof)
