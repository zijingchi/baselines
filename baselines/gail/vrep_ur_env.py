from vrep_env import vrep_env, vrep
import os
import time
import pickle
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np

pi = np.pi


class UR5VrepEnv(vrep_env.VrepEnv):
    metadata = {'render.modes': [], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            l2_thresh=0.1,
            random_initial=True,
            obs_space_type='dict',
            dof=5,
            enable_cameras=False,
    ):

        vrep_env.VrepEnv.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        # Settings
        self.init_joint_pos = np.array([0, -pi / 12, -3 * pi / 4, 0, pi / 2])
        self.target_joint_pos = np.array([0, - pi / 3, - pi / 3, 0, pi / 2])
        self.obstacle_pos = 5*np.ones(3)
        self.obstacle_ori = np.zeros(3)
        # All joints
        ur5_joints = ['UR5_joint1', 'UR5_joint2', 'UR5_joint3', 'UR5_joint4', 'UR5_joint5']
        # All shapes
        # ur5_links_visible = ['UR5_link1_visible','UR5_link2_visible','UR5_link3_visible',
        #	'UR5_link4_visible','UR5_link5_visible','UR6_link1_visible','UR5_link7_visible']

        # Getting object handles
        self.obstacle = self.get_object_handle('Obstacle')
        # Meta
        self.camera1 = self.get_object_handle('camera1')
        self.camera2 = self.get_object_handle('camera2')
        self.camera3 = self.get_object_handle('camera3')
        self.zfar2 = self.get_obj_float_parameter(self.camera2,
                                                  vrep.sim_visionfloatparam_far_clipping)
        self.znear2 = self.get_obj_float_parameter(self.camera2,
                                                   vrep.sim_visionfloatparam_near_clipping)
        self.goal_viz = self.get_object_handle('Cuboid')
        self.tip = self.get_object_handle('tip')
        self.distance_handle = self.get_distance_handle('Distance')
        self.distance = -1

        self.random_init = random_initial
        self.enable_cameras = enable_cameras
        self.dof = dof
        h = 256
        w = 256
        c = 3
        self.img_size = [h, w, c]
        # Actuators
        self.oh_joint = list(map(self.get_object_handle, ur5_joints))
        self.l2_thresh = l2_thresh
        self.collision_handle = self.get_collision_handle('Collision1')
        joint_space = np.ones(self.dof)
        self.action_space = spaces.Box(low=-0.08*joint_space, high=0.08*joint_space)
        joint_bound = 2*pi*joint_space
        obstacle_pos_lbound = np.array([-5, -5, 0])
        obstalce_pos_hbound = np.array([5, 5, 2])
        oob = 0.3
        obstacle_ori_lbound = np.array([-oob, -oob, pi/2-oob])
        obstacle_ori_hbound = np.array([oob, oob, pi/2+oob])

        if obs_space_type == 'concat':
            concat_bound_l = np.concatenate((-joint_bound, obstacle_pos_lbound, obstacle_ori_lbound), axis=-1)
            concat_bound_h = np.concatenate((joint_bound, obstalce_pos_hbound, obstacle_ori_hbound), axis=-1)
            self.observation_space = spaces.Box(low=-concat_bound_l, high=concat_bound_h)
        elif obs_space_type == 'dict':
            self.observation_space = spaces.Dict(dict(joint=spaces.Box(low=-joint_bound, high=joint_bound),
                                                      target=spaces.Box(low=-joint_bound, high=joint_bound),
                                                      obstacle_pos=spaces.Box(low=obstacle_pos_lbound,
                                                                              high=obstalce_pos_hbound),
                                                      obstacle_ori=spaces.Box(low=obstacle_ori_lbound,
                                                                              high=obstacle_ori_hbound)))

        self.seed()
        #print('UR5VrepEnv: initialized')

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
                                'image1': img1, 'image2': img2, 'image3': img3}
        else:
            self.observation = {'joint': np.array(joint_angles).astype('float32'),
                                'target': self.target_joint_pos,
                                'obstacle_pos': self.obstacle_pos,
                                'obstacle_ori': self.obstacle_ori}
        return self.observation

    def _calPathThroughVrep(self, clientID, minConfigsForPathPlanningPath, inFloats, emptyBuff):
        """send the signal to v-rep and retrieve the path tuple calculated by the v-rep script"""
        maxConfigsForDesiredPose = 10  # we will try to find 10 different states corresponding to the goal pose and order them according to distance from initial state
        maxTrialsForConfigSearch = 300  # a parameter needed for finding appropriate goal states
        searchCount = 2  # how many times OMPL will run for a given task
        # minConfigsForPathPlanningPath = 50  # interpolation states for the OMPL path
        minConfigsForIkPath = 100  # interpolation states for the linear approach path
        collisionChecking = 1  # whether collision checking is on or off
        inInts = [collisionChecking, minConfigsForIkPath, minConfigsForPathPlanningPath,
                  maxConfigsForDesiredPose, maxTrialsForConfigSearch, searchCount]
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID,
                            'Dummy', vrep.sim_scripttype_childscript, 'findPath_goalIsState',
                            inInts, inFloats, [], emptyBuff, vrep.simx_opmode_oneshot_wait)
        """retInts, path, retStrings, retBuffer = self.call_childscript_function('Dummy', 'findPath_goalIsState', 
                                                                              [inInts, inFloats, [], emptyBuff])"""
        #length = 0
        if (res == 0) and len(path) > 0:
            n_path = retInts[0]
            final = np.array(path[-5:])
            tar = np.array(inFloats[-5:])
            if np.linalg.norm(final - tar) > 0.01:
                n_path = 0
                path = []
            #length = retInts[2]
        else:
            n_path = 0
        return n_path, path, res

    def _checkInitCollision(self, clientID, emptyBuff):
        """returns 1 if collision occurred, 0 otherwise"""
        # todo: use get_collision_handle or read_collision instead
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID,
                            'Dummy', vrep.sim_scripttype_childscript, 'checkCollision',
                            [], [], [], emptyBuff, vrep.simx_opmode_oneshot_wait)
        if res == 0:
            return retInts[0]
        else:
            return -1

    def _make_action(self, a):
        """Send action to v-rep
        """
        newa = a + self.observation['joint']
        self.set_joints(newa)

    def set_joints(self, angles):
        for j, a in zip(self.oh_joint, angles):
            self.obj_set_joint_position(j, a)

    def compute_reward(self, state, action):
        approach = 5 if self._angle_dis(state, self.target_joint_pos, 5) < self.l2_thresh else 0
        collision = -3 if self.collision_check else 0
        #danger = -1 if self.distance < 3e-2 else 0
        return approach+collision

    def step(self, ac):
        self._make_observation()

        self._make_action(ac)
        self.step_simulation()
        self.collision_check = self.read_collision(self.collision_handle)
        done = self._angle_dis(self.observation['joint'], self.target_joint_pos, 5) < self.l2_thresh or self.collision_check
        reward = self.compute_reward(self.observation['joint'], ac)
        return self.observation, reward, done, {}

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        init_w = [0.6, 0.3, 0.3, 0.5, 0.5]
        if self.random_init:
            for i in range(self.dof):
                self.init_joint_pos[i] += init_w[i]*np.random.randn()
                self.target_joint_pos[i] += init_w[i]*np.random.randn()
        self.start_simulation()
        self.set_joints(self.init_joint_pos)
        self.reset_obstacle()
        self.obj_set_position(self.obstacle, self.obstacle_pos)
        self.obj_set_orientation(self.obstacle, self.obstacle_ori)
        self.step_simulation()
        ob = self._make_observation()
        '''concat_array = np.concatenate((self.observation['joint'],
                                       self.target_joint_pos,
                                       self.obstacle_pos,
                                       self.obstacle_ori), axis=-1)
        return concat_array'''
        return ob

    def _angle_dis(self, a1, a2, dof):
        return np.linalg.norm(a1[:dof]-a2[:dof])

    def reset_obstacle(self):
        pass

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


class PathPlanDset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True):
        #self.datafromindex = DataFromIndex(expert_path, rad2deg=False, load_img=False)
        all_files = [f for f in os.listdir(expert_path) if f.isdigit()]
        all_files.sort(key=lambda s: int(s[0]))
        self.obs_list = []
        self.acs_list = []
        for f in all_files:
            with open(os.path.join(expert_path, f, 'data.pkl'), 'rb') as pf:
                data = pickle.load(pf)
                inits = data['inits']
                acs = data['actions']
                obs = data['observations']
                obstacle_pos = inits['obstacle_pos']
                obstacle_ori = inits['obstacle_ori']
                for t in range(len(obs)):
                    inp = obs[t]
                    inp = np.append(inp, inits['target_joint_pos'])
                    inp = np.append(inp, obstacle_pos)
                    inp = np.append(inp, obstacle_ori)
                    self.obs_list.append(inp)
                    self.acs_list.append(acs[t])

        if traj_limitation < 0:
            self.obs_list = self.obs_list[:traj_limitation]
            self.acs_list = self.acs_list[:traj_limitation]
        self.obs_list = np.array(self.obs_list)
        self.acs_list = np.array(self.acs_list)
        train_size = int(train_fraction*len(self.obs_list))
        self.dset = VDset(self.obs_list, self.acs_list, randomize)
        # for behavior cloning
        self.train_set = VDset(self.obs_list[:train_size, :],
                               self.acs_list[:train_size, :],
                               randomize)
        self.val_set = VDset(self.obs_list[train_size:, :],
                             self.acs_list[train_size:, :],
                             randomize)
        self.pointer = 0
        self.train_pointer = 0
        self.test_pointer = 0

    def get_next_batch(self, batch_size, split=None):
        if split is None:
            return self.dset.get_next_batch(batch_size)
        elif split == 'train':
            return self.train_set.get_next_batch(batch_size)
        elif split == 'val':
            return self.val_set.get_next_batch(batch_size)
        else:
            raise NotImplementedError


class VDset(object):
    def __init__(self, inputs, labels, randomize):
        self.inputs = inputs
        self.labels = labels
        assert len(self.inputs) == len(self.labels)
        self.randomize = randomize
        self.num_pairs = len(inputs)
        self.init_pointer()

    def init_pointer(self):
        self.pointer = 0
        if self.randomize:
            idx = np.arange(self.num_pairs)
            np.random.shuffle(idx)
            self.inputs = self.inputs[idx, :]
            self.labels = self.labels[idx, :]

    def get_next_batch(self, batch_size):
        # if batch_size is negative -> return all
        if batch_size < 0:
            return self.inputs, self.labels
        if self.pointer + batch_size >= self.num_pairs:
            self.init_pointer()
        end = self.pointer + batch_size
        inputs = self.inputs[self.pointer:end, :]
        labels = self.labels[self.pointer:end, :]
        self.pointer = end
        return inputs, labels