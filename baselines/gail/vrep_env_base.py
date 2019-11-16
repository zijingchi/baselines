from vrep_env import vrep_env, vrep
import time
from gym import spaces
from gym.utils import seeding
import numpy as np

pi = np.pi


class UR5VrepEnvBase(vrep_env.VrepEnv):
    metadata = {'render.modes': ['human', 'rgb_array'], }

    def __init__(
            self,
            server_addr='127.0.0.1',
            server_port=19997,
            scene_path=None,
            l2_thresh=0.1,
            random_seed=0,
            dof=5,
    ):

        vrep_env.VrepEnv.__init__(
            self,
            server_addr,
            server_port,
            scene_path,
        )

        # Settings

        self.obstacle_pos = 5*np.ones(3)
        self.obstacle_ori = np.zeros(3)
        # All joints
        ur5_joints = ['UR5_joint1', 'UR5_joint2', 'UR5_joint3', 'UR5_joint4', 'UR5_joint5', 'UR5_joint6'][:5]

        # Getting object handles
        self.obstacle = self.get_object_handle('Obstacle')
        # Meta
        self.goal_viz = self.get_object_handle('Cuboid')
        self.tip = self.get_object_handle('tip')
        self.distance_handle = self.get_distance_handle('Distance')
        self.distance = -1

        self.dof = dof

        # Actuators
        self.oh_joint = list(map(self.get_object_handle, ur5_joints))
        self.init_joint_pos = np.array([0, -pi / 6, -3 * pi / 4, 0, pi / 2, 0])
        self.target_joint_pos = np.array([0, - pi / 6, - pi / 3, 0, pi / 2, 0])
        self.l2_thresh = l2_thresh
        self.collision_handle = self.get_collision_handle('Collision1')
        #self.self_col_handle = self.get_collision_handle('SelfCollision')
        joint_space = np.ones(self.dof)
        self.action_space = spaces.Box(low=-0.1*joint_space, high=0.1*joint_space)
        #self._make_obs_space()

        self.seed(random_seed)

    def _calPathThroughVrep(self, clientID, minConfigsForPathPlanningPath, inFloats, emptyBuff):
        """send the signal to v-rep and retrieve the path tuple calculated by the v-rep script"""
        dof = self.dof
        maxTrialsForConfigSearch = 300  # a parameter needed for finding appropriate goal states
        searchCount = 1  # how many times OMPL will run for a given task
        # minConfigsForPathPlanningPath = 50  # interpolation states for the OMPL path
        minConfigsForIkPath = 100  # interpolation states for the linear approach path
        collisionChecking = 1  # whether collision checking is on or off
        inInts = [collisionChecking, minConfigsForIkPath, minConfigsForPathPlanningPath,
                  dof, maxTrialsForConfigSearch, searchCount]
        res, retInts, path, retStrings, retBuffer = vrep.simxCallScriptFunction(clientID,
                            'Dummy', vrep.sim_scripttype_childscript, 'findPath_goalIsState',
                            inInts, inFloats, [], emptyBuff, vrep.simx_opmode_oneshot_wait)
        """retInts, path, retStrings, retBuffer = self.call_childscript_function('Dummy', 'findPath_goalIsState', 
                                                                              [inInts, inFloats, [], emptyBuff])"""
        if (res == 0) and len(path) > 0:
            n_path = retInts[0]
            final = np.array(path[-self.dof:])
            tar = np.array(inFloats[-self.dof:])
            if np.linalg.norm(final - tar) > 0.01:
                n_path = 0
                path = []
        else:
            n_path = 0
        return n_path, path, res

    def _make_obs_space(self):
        raise NotImplementedError

    def _checkInitCollision(self, clientID, emptyBuff):
        """returns 1 if collision occurred, 0 otherwise"""
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
        cfg = self._config()
        newa = a + cfg
        self.set_joints(newa)

    def _make_observation(self):
        """Get observation from v-rep and stores in self.observation
        """
        raise NotImplementedError

    def set_joints(self, angles):
        for j, a in zip(self.oh_joint, angles):
            self.obj_set_joint_position(j, a)

    def compute_reward(self, state, action):
        raise NotImplementedError

    def _action_process(self, ac):
        raise NotImplementedError

    def _config(self):
        raise NotImplementedError

    def step(self, ac):
        # self._make_observation()
        ac = self._action_process(ac)
        ac = np.clip(ac, self.action_space.low, self.action_space.high)
        self._make_action(ac)
        self.step_simulation()
        self._make_observation()
        self.collision_check = self.read_collision(self.collision_handle) or abs(self._config()[2])>5*pi/6
        self.distance = self.read_distance(self.distance_handle)
        cfg = self._config()
        done = self._angle_dis(cfg, self.target_joint_pos, self.dof) < self.l2_thresh or self.collision_check
        reward = self.compute_reward(cfg, ac)
        return self.observation, reward, done, {}

    def reset(self):
        raise NotImplementedError

    def reset_expert(self):
        emptybuff = bytearray()
        thresh = 0.09
        while True:
            ob = self.reset()
            final_path = None
            n_mid = int(self._angle_dis(self.target_joint_pos, self.init_joint_pos, self.dof) / thresh)
            linear_res = self.directly_towards(n_mid)
            self.set_joints(self.init_joint_pos)
            self.step_simulation()
            if linear_res == 0:
                final_path = self.linear_sub
                n_path = n_mid
            else:
                inFloats = self.init_joint_pos.tolist() + self.target_joint_pos.tolist()
                n_path, path, res = self._calPathThroughVrep(self.cID, 30, inFloats, emptybuff)
                if res==3:
                    time.sleep(3)
                    break
                np_path = np.array(path)
                re_path = np_path.reshape((n_path, self.dof))
                final_path = re_path
                '''c0 = self.init_joint_pos
                final_path = [c0]
                for c in re_path:
                    if self._angle_dis(c, c0, self.dof) > thresh:
                        final_path.append(c)
                        c0 = c
                # if c0.any() != np.array(self.target_joint_pos).any():
                #    final_path.append(np.array(self.target_joint_pos))
                n_path = len(final_path)'''
            if n_path:
                break
        return n_path, final_path

    def directly_towards(self, n):
        init_jp = self.init_joint_pos
        sub = (self.target_joint_pos - init_jp) / n
        self.linear_sub = []
        for i in range(n):
            next_state = init_jp + sub * (i + 1)
            self.linear_sub.append(next_state)
            self.set_joints(next_state)
            self.step_simulation()
            colcheck = self.read_collision(self.collision_handle)
            if colcheck == 1:
                #print('colliding during direct path')
                return 1
        #print('reaching by direct path')
        return 0

    def _angle_dis(self, a1, a2, dof):
        return np.linalg.norm(a1[:dof]-a2[:dof])

    def reset_obstacle(self):
        raise NotImplementedError

    def _check_collision(self, pos, handle):
        self.set_joints(pos)
        self.step_simulation()
        return self.read_collision(handle)

    def _clear_obs_col(self):
        raise NotImplementedError

    def render(self, mode='human', close=False):
        pass

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]


