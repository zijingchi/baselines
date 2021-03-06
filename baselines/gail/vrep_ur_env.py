from vrep_env import vrep_env, vrep
import os
import time
import pickle
import gym
from baselines.gail.vrep_env_base import UR5VrepEnvBase
from gym import spaces
from gym.utils import seeding
import numpy as np

pi = np.pi


class UR5VrepEnv(UR5VrepEnvBase):
    metadata = {'render.modes': ['human', 'rgb_array'], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            l2_thresh=0.1,
            random_seed=0,
            dof=5,
            enable_cameras=False,
    ):

        UR5VrepEnvBase.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
            l2_thresh,
            random_seed,
            dof,
        )
        self.enable_cameras = enable_cameras
        if enable_cameras:
            self.camera1 = self.get_object_handle('camera1')
            self.camera2 = self.get_object_handle('camera2')
            self.camera3 = self.get_object_handle('camera3')
            self.zfar2 = self.get_obj_float_parameter(self.camera2,
                                                      vrep.sim_visionfloatparam_far_clipping)
            self.znear2 = self.get_obj_float_parameter(self.camera2,
                                                       vrep.sim_visionfloatparam_near_clipping)

    def _config(self):
        return self.observation['joint']

    def _make_action(self, a):
        """Send action to v-rep
        """
        cfg = self._config()
        newa = a + cfg
        self.set_joints(newa)
        if (self.observation_space.spaces['joint'].low<newa).all() and (self.observation_space.spaces['joint'].high>newa).all():
            return True
        else:
            return False

    def _make_obs_space(self):
        joint_lbound = np.array([-2*pi/3, -pi/2, -pi, -pi/2, 0])
        joint_hbound = np.array([2 * pi / 3, pi / 6, 0, pi / 2, pi])
        obstacle_pos_lbound = np.array([-5, -5, 0])
        obstalce_pos_hbound = np.array([5, 5, 2])
        oob = 0.3
        obstacle_ori_lbound = np.array([-oob, -oob, pi / 2 - oob])
        obstacle_ori_hbound = np.array([oob, oob, pi / 2 + oob])
        self.observation_space = spaces.Dict(dict(joint=spaces.Box(low=joint_lbound, high=joint_hbound),
                                                  target=spaces.Box(low=joint_lbound, high=joint_hbound),
                                                  obstacle_pos1=spaces.Box(low=obstacle_pos_lbound,
                                                                          high=obstalce_pos_hbound),
                                                  obstacle_ori=spaces.Box(low=obstacle_ori_lbound,
                                                                          high=obstacle_ori_hbound)))

    def _make_observation(self):
        """Get observation from v-rep and stores in self.observation
        """
        joint_angles = [self.obj_get_joint_angle(joint) for joint in self.oh_joint]

        self.observation = {'joint': np.array(joint_angles).astype('float32'),
                            'target': self.target_joint_pos,
                            'obstacle_ori': self.obstacle_ori,
                            'obstacle_pos1': self.obstacle_pos}
        if isinstance(self.observation_space, spaces.Box):
            self.observation = np.concatenate((np.array(joint_angles), self.target_joint_pos, self.observation_space), axis=-1)
        return self.observation

    def compute_reward(self, state, action):
        config_dis = self._angle_dis(state, self.target_joint_pos, 5)
        #pre_config_dis = self._angle_dis(state-action, self.target_joint_pos, 5)
        approach = 2 if config_dis < self.l2_thresh else 0
        collision = -1 if self.collision_check else 0
        danger = -0.2 if self.distance < 2e-2 else 0
        valid = (self.observation_space.spaces['joint'].low<state).all() and (self.observation_space.spaces['joint'].high>state).all()
        invalid = -1 if not valid else 0
        return approach + collision + invalid

    def _action_process(self, ac):
        ac = np.clip(ac, self.action_space.low, self.action_space.high)
        if isinstance(self.observation_space, spaces.Box):
            cfg = self.observation[:self.dof]
        else:
            cfg = self.observation['joint']
        return ac + self.l2_thresh * (self.target_joint_pos - cfg) / np.linalg.norm(self.target_joint_pos - cfg)

    def step(self, ac):
        # self._make_observation()
        ac = self._action_process(ac)
        invalid = not self._make_action(ac)
        self.step_simulation()
        self._make_observation()
        self.collision_check = self.read_collision(self.collision_handle) or abs(self.observation['joint'][2])>5*pi/6
        self.distance = self.read_distance(self.distance_handle)
        cfg = self._config()

        done = self._angle_dis(cfg, self.target_joint_pos, self.dof) < self.l2_thresh or self.collision_check or invalid
        reward = self.compute_reward(cfg, ac)
        info = {}
        if reward > 0.5 and done:
            info["status"] = 'reach'
        else:
            info["status"] = 'collide'
        return self.observation, reward, done, info

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        init_w = [0.1, 0.1, 0.1, 0.2, 0.2][:self.dof]
        self.init_joint_pos = np.array([0, -pi / 6, -3 * pi / 4, 0, pi / 2])[:self.dof]
        self.target_joint_pos = np.array([0, - pi / 3, - pi / 3, 0, pi / 2])[:self.dof]
        init_joint_pos = self.init_joint_pos + np.multiply(init_w, np.random.randn(self.dof))
        target_joint_pos = self.target_joint_pos + np.multiply(init_w, np.random.randn(self.dof))
        self.start_simulation()
        while abs(init_joint_pos[2])>5*pi/6 or abs(target_joint_pos[2])>5*pi/6:
            init_joint_pos[2] = self.init_joint_pos[2] + init_w[2]*np.random.randn()
            target_joint_pos[2] = self.target_joint_pos[2] + init_w[2]*np.random.randn()
        self.set_joints(target_joint_pos)
        self.init_joint_pos = init_joint_pos
        self.target_joint_pos = target_joint_pos
        self.init_goal_dis = self._angle_dis(init_joint_pos, target_joint_pos, self.dof)
        self.reset_obstacle()

        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)

        while self._clear_obs_col():
            self.reset_obstacle()

        self.step_simulation()
        ob = self._make_observation()

        return ob

    def reset_obstacle(self):
        init_tips = tipcoor(np.concatenate((self.init_joint_pos, np.zeros(5-self.dof))))[:3]
        goal_tips = tipcoor(np.concatenate((self.target_joint_pos, np.zeros(5-self.dof))))[:3]
        init_tip = init_tips[-1]
        goal_tip = goal_tips[-1]
        alpha = 0.5
        obs_pos = alpha*init_tip + (1-alpha)*goal_tip
        obs_pos += np.concatenate((0.15*np.random.randn(2), np.array([0.25*np.random.rand()+0.28])))
        obs_ori = 0.2*np.random.randn(3)
        obs_ori[2] += pi/2

        self.obstacle_pos = np.clip(obs_pos, self.observation_space.spaces['obstacle_pos1'].low,
                                    self.observation_space.spaces['obstacle_pos1'].high)
        self.obstacle_ori = np.clip(obs_ori, self.observation_space.spaces['obstacle_ori'].low,
                                    self.observation_space.spaces['obstacle_ori'].high)

    def _clear_obs_col(self):
        self.step_simulation()
        self.obj_set_position(self.obstacle, self.obstacle_pos)
        self.obj_set_orientation(self.obstacle, self.obstacle_ori)
        col1 = self._check_collision(self.target_joint_pos, self.collision_handle)
        col2 = self._check_collision(self.init_joint_pos, self.collision_handle)
        return col1 or col2

    def _set_path_draw(self, path, obj, icolor=3):
        l = len(path)
        inFloats = path.flatten()
        inInts = [self.dof, l, icolor]
        emptyBuff = bytearray()
        retInts, retFloats, retStrings, retBuffer = self.call_childscript_function(obj, 'drawPath_remote', [inInts, inFloats, [], emptyBuff])
        draw_handle = retInts[0]
        return draw_handle

    def draw_policy_path(self, path):
        policy_handle = self._set_path_draw(path, 'Dummy', 1)
        self.directly_towards(40)
        linear_path = np.array(self.linear_sub)
        linear_handle = self._set_path_draw(linear_path, 'UR5', 3)
        n_path, expert_path, _ = self._calPathThroughVrep(self.cID, 50, self.init_joint_pos.tolist() + self.target_joint_pos.tolist(), bytearray())
        if n_path>0:
            self._set_path_draw(np.array(expert_path), 'wall4', 2)
        time.sleep(1)
        #self.call_childscript_function('Dummy', 'rmvDraw_remote', [[linear_handle, policy_handle], [], [], bytearray()])

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


def ur5fk(thetas):
    d0 = 0.3
    d1 = 8.92e-2
    d2 = 0.11
    d5 = 9.475e-2
    #d6 = 7.495e-2
    d6 = 1.1495e-1
    a2 = 4.251e-1
    a3 = 3.9215e-1
    All = np.zeros((6, 4, 4))
    All[:, 3, 3] = 1
    for i in range(6):
        All[i, 0, 0] = np.cos(thetas[i])
        All[i, 0, 1] = -np.sin(thetas[i])
    All[0, 1, 0] = np.sin(thetas[0])
    All[0, 1, 1] = np.cos(thetas[0])
    All[0, 2, 3] = d1
    All[0, 2, 2] = 1

    All[1, 2, 0] = np.sin(thetas[1])
    All[1, 2, 1] = np.cos(thetas[1])
    All[1, 1, 2] = -1
    All[1, 1, 3] = -d2

    All[2, 1, 0] = np.sin(thetas[2])
    All[2, 1, 1] = np.cos(thetas[2])
    All[2, 0, 3] = a2
    All[2, 2, 2] = 1

    All[3, 1, 0] = np.sin(thetas[3])
    All[3, 1, 1] = np.cos(thetas[3])
    All[3, 0, 3] = a3
    All[3, 2, 2] = 1

    All[4, 2, 0] = np.sin(thetas[4])
    All[4, 2, 1] = np.cos(thetas[4])
    All[4, 1, 3] = -d5
    All[4, 1, 2] = -1

    All[5, :, :] = np.eye(4)
    All[5, 1, 3] = -d6

    A0 = np.zeros((4, 4))
    A0[0, 1] = 1
    A0[1, 0] = -1
    A0[2, 2] = 1
    A0[2, 3] = d0
    A0[3, 3] = 1
    #A0[2, 3] = 0
    return All, A0


def tipcoor(thetas):
    thetas_0 = np.array([0, pi / 2, 0, pi / 2, pi])
    thetas = thetas + thetas_0
    All, A0 = ur5fk(thetas)
    ps = []
    for A in All:
        A0 = A0 @ A
        ps.extend(A0[:3, 3])
    return np.array(ps)


class UR5VrepEnvMultiObstacle(UR5VrepEnv):
    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            l2_thresh=0.1,
            random_seed=0,
            dof=5,
            enable_cameras=False,):
        super(UR5VrepEnvMultiObstacle, self).__init__(server_addr, server_port, scene_path, l2_thresh, random_seed,
                                                      dof, enable_cameras)
        self.obstacle2 = self.get_object_handle('Obstacle2')
        self.obstacle_pos2 = np.ones(3)
        self.collision_handle = self.get_collision_handle('Collision')

    def _make_obs_space(self):
        joint_lbound = np.array([-2 * pi / 3, -pi / 2, -pi, -pi / 2, 0])
        joint_hbound = np.array([2 * pi / 3, pi / 6, 0, pi / 2, pi])
        obstacle_pos_lbound = np.array([-5, -5, 0])
        obstalce_pos_hbound = np.array([5, 5, 2])
        self.observation_space = spaces.Dict(dict(joint=spaces.Box(low=joint_lbound, high=joint_hbound),
                                                  target=spaces.Box(low=joint_lbound, high=joint_hbound),
                                                  obstacle_pos1=spaces.Box(low=obstacle_pos_lbound,
                                                                           high=obstalce_pos_hbound),
                                                  obstacle_pos2=spaces.Box(low=obstacle_pos_lbound,
                                                                           high=obstalce_pos_hbound)))

    def _make_observation(self):
        """Get observation from v-rep and stores in self.observation
        """
        joint_angles = [self.obj_get_joint_angle(joint) for joint in self.oh_joint]
        #self.distance = self.read_distance(self.distance_handle)
        if self.enable_cameras:
            img1 = self.obj_get_vision_image(self.camera1)
            img2 = self.obj_get_vision_image(self.camera2)
            img3 = self.obj_get_vision_image(self.camera3)
            img1 = np.flip(img1, 2)
            img2 = np.flip(img2, 2)
            img3 = np.flip(img3, 2)
            self.observation = {'joint': np.array(joint_angles).astype('float32'),
                                'target': self.target_joint_pos,
                                'obstacle_pos1': self.obstacle_pos,
                                'obstacle_pos2': self.obstacle_pos2,
                                'image1': img1, 'image2': img2, 'image3': img3}
        else:
            self.observation = {'joint': np.array(joint_angles).astype('float32'),
                                'target': self.target_joint_pos,
                                'obstacle_pos1': self.obstacle_pos,
                                'obstacle_pos2': self.obstacle_pos2}

        return self.observation

    def reset_obstacle(self):
        init_tip = tipcoor(np.concatenate((self.init_joint_pos, np.zeros(5 - self.dof))))[-3:]
        goal_tip = tipcoor(np.concatenate((self.target_joint_pos, np.zeros(5 - self.dof))))[-3:]
        alpha = 0.5
        obs_pos = alpha * init_tip + (1 - alpha) * goal_tip
        obs_pos += np.concatenate((0.15 * np.random.randn(2), np.array([0.15 * np.random.rand() + 0.24])))

        self.obstacle_pos = np.clip(obs_pos, self.observation_space.spaces['obstacle_pos1'].low,
                                    self.observation_space.spaces['obstacle_pos1'].high)

        offset = np.concatenate((np.random.randn(2), np.random.rand(1)))
        self.obstacle_pos2 = self.obstacle_pos + 0.2*offset/np.linalg.norm(offset)

    def _clear_obs_col(self):
        self.step_simulation()
        self.obj_set_position(self.obstacle, self.obstacle_pos)
        self.obj_set_position(self.obstacle2, self.obstacle_pos2)
        col1 = self._check_collision(self.target_joint_pos, self.collision_handle)
        col2 = self._check_collision(self.init_joint_pos, self.collision_handle)
        return col1 or col2


class UR5VrepEnvConcat(UR5VrepEnv):
    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            l2_thresh=0.1,
            random_seed=0,
            dof=5,
    ):

        super(UR5VrepEnvConcat, self).__init__(
            server_addr,
            server_port,
            scene_path,
            l2_thresh,
            random_seed,
            dof,
        )
        self._make_obs_space()

    def _make_obs_space(self):
        joint_lbound = np.array([-2 * pi / 3, -pi / 2, -pi, -pi / 2, 0])
        joint_hbound = np.array([2 * pi / 3, pi / 6, 0, pi / 2, pi])
        obstacle_pos_lbound = np.array([-5, -5, 0])
        obstalce_pos_hbound = np.array([5, 5, 2])
        dis_lb = np.array([-5])
        dis_hb = np.array([5])
        self.observation_space = gym.spaces.Box(low=np.concatenate([joint_lbound, joint_lbound, obstacle_pos_lbound]),
                                                high=np.concatenate([joint_hbound, joint_hbound, obstalce_pos_hbound]))

    def _make_observation(self):
        """Get observation from v-rep and stores in self.observation
        """
        joint_angles = [self.obj_get_joint_angle(joint) for joint in self.oh_joint]
        self.distance = self.read_distance(self.distance_handle)
        self.observation = np.concatenate([np.array(joint_angles).astype('float32'),
                                           self.target_joint_pos,
                                           self.obstacle_pos,
                                           #np.array([self.distance])
                                           ])

        return self.observation

    def reset_obstacle(self):
        init_tip = tipcoor(self.init_joint_pos)
        goal_tip = tipcoor(self.target_joint_pos)
        alpha = 0.5
        #obs_pos = alpha * init_tip + (1 - alpha) * goal_tip
        #obs_pos += np.concatenate((0.15 * np.random.randn(2), np.array([0.15 * np.random.rand() + 0.24])))
        obs_pos = alpha * init_tip + (1 - alpha) * goal_tip
        obs_pos += np.concatenate((0.15 * np.random.randn(2), np.array([0.25 * np.random.rand() + 0.28])))
        self.obstacle_pos = np.clip(obs_pos, self.observation_space.low[5*2:5*2+3],
                                    self.observation_space.high[5*2:5*2+3])

        offset = np.concatenate((np.random.randn(2), np.random.rand(1)))
        self.obstacle_pos2 = self.obstacle_pos + 0.2*offset/np.linalg.norm(offset)

    def _config(self):
        return self.observation[:5]

    def _make_action(self, a):
        """Send action to v-rep
        """
        cfg = self._config()
        newa = a + cfg
        self.set_joints(newa)
        if (self.observation_space.low[:5]<newa).all() and (self.observation_space.high[:5]>newa).all():
            return True
        else:
            return False

    def step(self, ac):
        # self._make_observation()
        ac = self._action_process(ac)
        ac = np.clip(ac, self.action_space.low, self.action_space.high)
        invalid = not self._make_action(ac)
        self.step_simulation()

        self._make_observation()
        self.collision_check = self.read_collision(self.collision_handle) or abs(self.observation[2])>5*pi/6

        cfg = self.observation[:self.dof]
        done = self._angle_dis(cfg, self.target_joint_pos,
                               5) < 1.5 * self.l2_thresh or self.collision_check or invalid

        reward = self.compute_reward(cfg, ac)
        info = {}
        if reward > 0.5 and done:
            info["status"] = 'reach'
        elif reward < 0.5 and done:
            info["status"] = 'collide'
        else:
            info["status"] = 'running'
        return self.observation, reward, done, info

    def compute_reward(self, state, action):
        config_dis = self._angle_dis(state, self.target_joint_pos, 5)
        # pre_config_dis = self._angle_dis(state-action, self.target_joint_pos, 5)
        approach = 2 if config_dis < 1.5 * self.l2_thresh else 0
        collision = -1 if self.collision_check else 0
        danger = -0.01 if self.distance < 2e-2 else 0
        valid = (self.observation_space.low[:5] < state).all() and (
                    self.observation_space.high[:5] > state).all()
        invalid = -1 if not valid else 0
        return approach + collision + invalid


class UR5VrepEnvDrtn(UR5VrepEnv):
    def __init__(self,
                 server_addr='127.0.0.1',
                 server_port=19997,
                 scene_path=None,
                 l2_thresh=0.1,
                 random_seed=0,
                 dof=5,):
        super(UR5VrepEnvDrtn, self).__init__(server_addr, server_port, scene_path, l2_thresh, random_seed,
                                             dof, False)
        self.action_space = spaces.Box(low=-2*pi*np.ones(dof-1), high=2*pi*np.ones(dof-1))

    def _action_process(self, ac):
        assert len(ac) == self.dof - 1
        tg = self.target_joint_pos - self._config()
        newa = ac + self._vec2dir(tg)
        return self._dir2vec(newa, self.l2_thresh)

    @staticmethod
    def _vec2dir(v):
        v = v / np.linalg.norm(v)  # 指向目标的单位角向量
        a1 = np.arccos(v[4])
        a2 = np.arcsin(v[3] / np.sin(a1))
        a3 = np.arcsin(v[2] / (np.sin(a1) * np.cos(a2)))
        b = np.sin(a1) * np.cos(a2) * np.cos(a3)
        a4 = np.arctan2(v[1] / b, v[0] / b)  # [a1, a2, a3, a4]是tg的方向向量
        return np.array([a1, a2, a3, a4])

    def _dir2vec(self, d, l2):
        t = [np.cos(d[0]), np.sin(d[0])]
        for i in range(1, self.dof):
            last = t[-1]
            t[-1] = last * np.sin(d[i])
            t.append(last * np.cos(d[i]))
        t = np.array(t[::-1])  # 偏移后的方向对应的单位角向量
        return l2*t
