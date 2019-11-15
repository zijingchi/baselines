import os
import cv2
import re
import pickle
from baselines.gail.vrep_ur_env import UR5VrepEnv, tipcoor
import numpy as np
pi = np.pi


class PathPlanDset(object):
    def __init__(self, expert_path, train_fraction=0.7, traj_limitation=-1, randomize=True, thresh=0.1):
        #self.datafromindex = DataFromIndex(expert_path, rad2deg=False, load_img=False)
        all_files = [f for f in os.listdir(expert_path) if f.isdigit()]
        #all_files.sort(key=lambda s: int(s[0]))
        np.random.shuffle(all_files)
        self.obs_list = []
        self.acs_list = []
        for f in all_files:
            pklpath = os.path.join(expert_path, f, 'data.pkl')
            if not os.path.exists(pklpath): continue
            with open(pklpath, 'rb') as pf:
                data = pickle.load(pf)
                inits = data['inits']
                acs = data['actions']
                obs = data['observations']
                obstacle_pos = inits['obstacle_pos']
                #obstacle_ori = inits['obstacle_ori']
                for t in range(1, len(obs)):
                    inp = obs[t]
                    inp = np.append(inp, inits['target_joint_pos'])
                    inp = np.append(inp, obstacle_pos)
                    #inp = np.append(inp, obstacle_ori)
                    self.obs_list.append(inp)
                    avo = acs[t] - thresh * (inits['target_joint_pos'] - obs[t]) / np.linalg.norm(inits['target_joint_pos'] - obs[t])
                    self.acs_list.append(avo)

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


class ExpertDataset(object):
    def __init__(self, expert_path, train_fraction=0.7, listpkl='list0.pkl', traj_limitation=-1, load_img=False):
        self.path = expert_path
        self.load_img = load_img
        self.listpkl = listpkl
        self.traj_limit = traj_limitation
        self.split_train(train_fraction)
        self.n_train = len(self.train_list)
        self.n_val = len(self.vali_list)
        self.train_pointer = 0
        self.val_pointer = 0

    def split_train(self, fraction):
        if os.path.exists(os.path.join(self.path, self.listpkl)):
            with open(os.path.join(self.path, self.listpkl), 'rb') as f:
                data = pickle.load(f)
                self.train_list = data['train']
                self.vali_list = data['test']
        else:
            #self.listpkl = 'list0.pkl'
            dirlist = os.listdir(self.path)
            id_list = []
            np.random.shuffle(dirlist)
            for d in dirlist:
                subdir = os.path.join(self.path, d)
                if os.path.isdir(subdir):
                    datapkl = os.path.join(subdir, 'data.pkl')
                    if os.path.exists(datapkl):
                        with open(datapkl, 'rb') as dataf:
                            data = pickle.load(dataf)
                            '''a0 = data['actions'][0]
                            at = data['actions'][-1]
                            if np.linalg.norm(a0-at)<0.02 and np.random.rand()<0.8:
                                #shutil.rmtree(subdir)
                                continue'''
                            for i in range(len(data['actions'])):
                                id_list.append(d + '-' + str(i))
            if self.traj_limit>0:
                id_list = id_list[:self.traj_limit]
            id_size = len(id_list)
            train_size = int(fraction * id_size)
            # np.random.shuffle(id_list)
            self.train_list = id_list[:train_size]
            self.vali_list = id_list[train_size:]

            np.random.shuffle(self.train_list)
            np.random.shuffle(self.vali_list)

            with open(os.path.join(self.path, self.listpkl), 'wb') as f1:
                pickle.dump({'train': self.train_list, 'test': self.vali_list}, f1)

    def read_single_index(self, index):
        reindex = re.split('\W', index)
        dirname = reindex[0]
        datadir = os.path.join(self.path, dirname)

        t = int(reindex[1])
        pkl_path = os.path.join(datadir, 'data.pkl')
        img1_path = os.path.join(datadir, 'img1/' + reindex[1] + '.jpg')
        img2_path = os.path.join(datadir, 'img2/' + reindex[1] + '.jpg')
        observation = {}
        if self.load_img:
            img1 = cv2.imread(img1_path)
            img2 = cv2.imread(img2_path)
            observation['img1'] = img1
            observation['img2'] = img2
        with open(pkl_path, 'rb') as pkl_file:
            pkl_data = pickle.load(pkl_file)
            inits = pkl_data['inits']
            obs = pkl_data['observations']
            # config = obs[t]['joint']

            config = obs[t]
            tar_pos = inits['target_joint_pos']
            obstacle_pos = inits['obstacle_pos']
            xyzs = tipcoor(config)[3:-3]
            obs_final = np.concatenate((config, tar_pos, obstacle_pos, xyzs))

            actions = pkl_data['actions']
            action = actions[t]
            if np.linalg.norm(action) > 0.5:
                action = actions[-1]

        observation['config'] = obs_final
        return observation, action

    def read_indexes(self, indexes):
        obs = []
        acs = []
        for index in indexes:
            ob, ac = self.read_single_index(index)
            obs.append(ob['config'])
            acs.append(ac)
        return np.array(obs), np.array(acs)

    def get_next_batch(self, batch_size, split='train'):
        if split == 'train':
            cur_list = self.train_list
            cur_pointer = self.train_pointer
            cur_len = self.n_train
        elif split == 'val':
            cur_list = self.vali_list
            cur_pointer = self.val_pointer
            cur_len = self.n_val
        else:
            raise NotImplementedError
        #cur_len = len(cur_list)
        end = cur_pointer + batch_size
        if batch_size<0:
            return self.read_indexes(cur_list)
        if end<cur_len:
            if split=='train':
                self.train_pointer = end
            else:
                self.val_pointer = end
            return self.read_indexes(cur_list[cur_pointer:end])
        else:
            if split=='train':
                self.train_pointer = end - cur_len
            else:
                self.val_pointer = end - cur_len
            return self.read_indexes(cur_list[cur_pointer:]+cur_list[:end-cur_len])


class Recorder(object):
    def __init__(self, datapath, *extras, expert=False):
        self.path = datapath
        self.exp = expert
        self.pklpath = os.path.join(datapath, 'traj.pkl')
        self.extras = {}
        for name in extras:
            self.extras[name] = []
        self.reset()

    def record(self, ob, ac, rew, done, *args):
        self.obs.append(ob)
        self.acs.append(ac)
        self.rews.append(rew)
        self.dones.append(done)
        for name, value in args:
            self.extras[name].append(value)

    def reset(self):
        self.obs = []
        self.acs = []
        self.rews = []
        self.dones = []
        for key in self.extras.keys():
            self.extras[key] = []

    def save(self):
        with open(self.pklpath, 'wb') as f:
            pickle.dump({'ob': self.obs, 'ac': self.acs, 'rew': self.rews,
                         'done': self.dones, 'extra': self.extras}, f)
        self.reset()


def main(args):
    workpath = 'dataset/ur5expert4/'
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    env = UR5VrepEnv(dof=5, enable_cameras=True, l2_thresh=0.08, random_seed=maxdir)
    i = maxdir + 1
    while i < maxdir + 2000:
        print('iter:', i)
        n_path, path = env.reset_expert()
        if n_path==0: continue
        init = {'init_joint_pos': env.init_joint_pos, 'target_joint_pos': env.target_joint_pos,
                'obstacle_pos': env.obstacle_pos}
        os.mkdir(str(i))
        os.mkdir(str(i) + "/img1")
        os.mkdir(str(i) + "/img2")
        os.mkdir(str(i) + "/img3")
        obs = []
        dis = []
        acs = []
        for t in range(n_path-1):
            action = path[t+1] - path[t]
            observation, rew, done, info = env.step(action)
            obs.append(observation['joint'])
            dis.append(env.distance)
            #ac_expert[i] - thresh * (tar[i] - cur[i]) / np.linalg.norm(tar[i] - cur[i])
            #action = np.clip(action, env.action_space.low, env.action_space.high)
            acs.append(action)
            img1_path = str(i) + "/img1/" + str(t) + ".jpg"
            img2_path = str(i) + "/img2/" + str(t) + ".jpg"
            img3_path = str(i) + "/img3/" + str(t) + ".jpg"
            cv2.imwrite(img1_path, observation['image1'])
            cv2.imwrite(img2_path, observation['image2'])
            cv2.imwrite(img3_path, observation['image3'])
        data = {'inits': init, 'observations': [obs, dis], 'actions': acs}
        with open(str(i) + '/data.pkl', 'wb') as f:
            pickle.dump(data, f)
        i = i + 1
    # print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys

    sys.exit(main(sys.argv))