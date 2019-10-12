import os
import cv2
import time
import pickle
from baselines.gail.vrep_ur_env import UR5VrepEnv
import numpy as np
pi = np.pi


def main(args):
    workpath = 'dataset/ur5expert1/'
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    env = UR5VrepEnv(dof=3, enable_cameras=True)
    i = maxdir + 1
    while i < maxdir + 1000:
        print('iter:', i)
        n_path, path = env.reset_expert()
        init = {'init_joint_pos': env.init_joint_pos, 'target_joint_pos': env.target_joint_pos,
                'obstable_pos': env.obstacle_pos}
        os.mkdir(str(i))
        os.mkdir(str(i) + "/img1")
        os.mkdir(str(i) + "/img2")
        os.mkdir(str(i) + "/img3")
        obs = []
        acs = []
        for t in range(n_path-1):
            action = path[t+1] - path[t]
            observation, _, done, _ = env.step(action)
            obs.append(observation['joint'])
            acs.append(action)
            img1_path = str(i) + "/img1/" + str(t) + ".jpg"
            img2_path = str(i) + "/img2/" + str(t) + ".jpg"
            img3_path = str(i) + "/img3/" + str(t) + ".jpg"
            cv2.imwrite(img1_path, observation['image1'])
            cv2.imwrite(img2_path, observation['image2'])
            cv2.imwrite(img3_path, observation['image3'])
        data = {'inits': init, 'observations': obs, 'actions': acs}
        with open(str(i) + '/data.pkl', 'wb') as f:
            pickle.dump(data, f)
        i = i + 1
    # print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))


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
                for t in range(1, len(obs)):
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
        # self.dset = VDset(self.obs_list, self.acs_list, randomize)
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