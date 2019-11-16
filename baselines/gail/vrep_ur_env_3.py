import gym
from baselines.gail.vrep_ur_env import tipcoor2, UR5VrepEnvConcat, tipcoor
import autograd.numpy as anp
import numpy as np
import os
import pickle
import time
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
        self.prev_config = cfg
        return self.observation, reward, done, info


class UR5VrepEnvKine(UR5VrepEnvConcat):
    def __init__(self,
                 server_addr='127.0.0.1',
                 server_port=19997,
                 scene_path=None,
                 l2_thresh=0.1,
                 random_seed=0,
                 dof=5,
                 ):
        super(UR5VrepEnvKine, self).__init__(server_addr, server_port, scene_path, l2_thresh, random_seed, dof)
        #self.jacob = jacobian(ur5fk)
        ac_space_bound = np.array([0.05, 0.05, 0.05, 0.12, 0.12, 0.12])
        #self.action_space = gym.spaces.Box(low=-ac_space_bound, high=ac_space_bound)
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
        pos_lbound = np.array([-1.2, -1.2, -0.2]*4)
        pos_hbound = np.array([1.2, 1.2, 1.3] * 4)
        self.observation_space = gym.spaces.Box(low=np.concatenate([joint_lbound, joint_lbound, obstacle_pos_lbound, pos_lbound]),
                                                high=np.concatenate([joint_hbound, joint_hbound, obstalce_pos_hbound, pos_hbound]))

    def _make_observation(self):
        joint_angles = [self.obj_get_joint_angle(joint) for joint in self.oh_joint]
        self.distance = self.read_distance(self.distance_handle)
        self.tip_pos = self.obj_get_position(self.tip)
        ps = tipcoor(joint_angles)
        # todo: replace the exact distance by a rough interval
        self.observation = np.concatenate([np.array(joint_angles).astype('float32'),
                                           self.target_joint_pos,
                                           self.obstacle_pos,
                                           ps[3:-3],
                                           #np.array([self.distance])
                                           ])

        return self.observation

    def _action_process(self, ac):
        #J = self.jacob(self._config())
        #ac = np.clip(ac, self.action_space.low, self.action_space.high)
        #ac = np.linalg.pinv(J)@ac
        '''cfg = self._config()
        norm = np.linalg.norm(self.target_joint_pos - cfg)
        if norm<self.l2_thresh*2:
            c = 0.5
        else:
            c = 1
        ac += c*self.l2_thresh * (self.target_joint_pos - cfg) / norm'''
        ac = ac/np.linalg.norm(ac)*self.l2_thresh
        ac = np.clip(ac, self.action_space.low, self.action_space.high)
        return ac


class UR5VrepEnvKineLoad(UR5VrepEnvKine):
    def __init__(self,
                 expert_path,
                 server_addr='127.0.0.1',
                 server_port=19997,
                 scene_path=None,
                 l2_thresh=0.1,
                 random_seed=0,
                 dof=5,
                 ):
        super(UR5VrepEnvKineLoad, self).__init__(server_addr, server_port, scene_path, l2_thresh, random_seed, dof)
        self.expert_data_path = expert_path
        self.i = 0
        self._load_expert_data()
        self.expert_path = []

    def _load_expert_data(self):
        dirlist = os.listdir(self.expert_data_path)
        self.pkllist = []
        for d in dirlist:
            pkl = os.path.join(self.expert_data_path, d, 'data.pkl')
            if os.path.exists(pkl):
                self.pkllist.append(pkl)

    def reset(self):
        if self.sim_running:
            self.stop_simulation()
        while self.sim_running:
            self.stop_simulation()

        pklfile = self.pkllist[self.i]
        self.i = (self.i+1)%len(self.pkllist)
        with open(pklfile, 'rb') as f:
            data = pickle.load(f)
            inits = data['inits']
            self.init_joint_pos = inits['init_joint_pos']
            self.target_joint_pos = inits['target_joint_pos']
            self.obstacle_pos = inits['obstacle_pos']
            self.expert_path = data['observations']
        self.obstacle_ori = np.zeros(3)
        self.start_simulation()
        self.set_joints(self.init_joint_pos)
        self.init_goal_dis = self._angle_dis(self.init_joint_pos, self.target_joint_pos, 5)
        self.obj_set_position(self.obstacle, self.obstacle_pos)

        tip_pos = self.obj_get_position(self.tip)
        tip_ori = self.obj_get_orientation(self.tip)
        self.obj_set_position(self.goal_viz, tip_pos)
        self.obj_set_orientation(self.goal_viz, tip_ori)

        self.step_simulation()
        ob = self._make_observation()
        self.last_dis = self.distance
        self.target_tip_pos = tip_pos
        self.prev_ac = -1
        self.t = 0
        return ob


class ColStateEnv(UR5VrepEnvConcat):
    def __init__(self, random_seed=0):
        super(ColStateEnv, self).__init__(random_seed=random_seed, dof=3)
        theta1_left = -1.4
        theta1_right = 1.4
        theta2_left = -1.9
        theta2_right = 0.5
        theta3_left = -2.7
        theta3_right = 0.0
        theta1_sample = np.linspace(theta1_left, theta1_right, 32)
        theta2_sample = np.linspace(theta2_left, theta2_right, 32)
        theta3_sample = np.linspace(theta3_left, theta3_right, 32)
        thetas_sample = []
        alpha = 1.2
        for t1 in theta1_sample:
            for t2 in theta2_sample:
                for t3 in theta3_sample:
                    if (alpha * t2 + t3 < -6 * pi / 5) or (alpha * t2 + t3 > 0):
                        continue
                    thetas_sample.append([t1, t2, t3, 0, pi / 2])
        print(len(thetas_sample))
        # self.collision_handle = self.get_collision_handle('Collision1')
        self.thetas_sample = np.array(thetas_sample)

    @staticmethod
    def pldis(p1, p2, q):
        p1q = q - p1
        p1p2 = p2 - p1
        return np.linalg.norm(np.cross(p1q, p1p2)) / np.linalg.norm(p1p2)

    def check_col_states(self, obs):
        emptybuff = bytearray()
        n_states = len(self.thetas_sample)
        col_states = np.zeros(n_states, dtype=np.int8)

        for i in range(n_states):
            theta = self.thetas_sample[i]
            ps = tipcoor2(theta)
            lmin = 10
            for i in range(1, 7):
                p = ps[i*3:i*3+3]
                l = np.linalg.norm(p-obs)
                lmin = l if l<lmin else lmin
            lmin = min(lmin, self.pldis(ps[9:12], ps[15:18], obs))
            if lmin>0.2:
                continue
            self.set_joints(theta)
            #self.step_simulation()
            colcheck = self._checkInitCollision(self.cID, emptybuff)
            #colcheck = self._check_collision(theta, self.collision_handle)
            col_states[i] = colcheck

        return col_states


def main():
    workpath = os.path.expanduser('~/colstate')
    datapath1 = workpath + '/0'
    # examine_states(datapath1)
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist if os.path.isdir(os.path.join(workpath, s))]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    next_dir = str(maxdir + 1)
    os.mkdir(next_dir)

    env = ColStateEnv(100)
    #env.reset()
    for i in range(600):
        print(i)
        start_time = time.time()
        env.reset()
        col_states_per_obs = env.check_col_states(env.obstacle_pos)
        col_states_pkl = str(i) + 'col_states.pkl'
        with open(os.path.join(next_dir, col_states_pkl), 'wb') as f2:
            pickle.dump(col_states_per_obs, f2)
        end_time = time.time()
        print('cost %d s' % (end_time - start_time))

    states_pkl = 'states.pkl'
    obs_pkl = 'obs.pkl'
    with open(os.path.join(next_dir, states_pkl), 'wb') as f1:
        pickle.dump(env.thetas_sample, f1)
    with open(os.path.join(next_dir, obs_pkl), 'wb') as f3:
        pickle.dump(env.obstacle, f3)


if __name__ == '__main__':
    main()
