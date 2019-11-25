import os
import cv2
import re
import pickle
from baselines.gail.vrep_ur_env import UR5VrepEnvConcat, tipcoor
import time
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
        action = action - 0.1*(tar_pos-config)/np.linalg.norm(tar_pos-config)
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


class ExpertDisDataset(ExpertDataset):
    def __init__(self, expert_path, dof=3, interval=0.08, train_fraction=0.7, listpkl='list0.pkl', traj_limitation=-1):
        super(ExpertDisDataset, self).__init__(expert_path, train_fraction=train_fraction, listpkl=listpkl,
                                               traj_limitation=traj_limitation)
        self.dof = dof
        self.interval = interval
        self.data = {'obs':[], 'acs':[]}

    @staticmethod
    def fiv(numa):
        acnum = 0
        for i, a in enumerate(reversed(numa)):
            acnum += (5**i)*a
        return acnum

    @staticmethod
    def defiv(acnum, dof):
        res = [0 for _ in range(dof)]
        for i in range(dof)[::-1]:
            res[i] = acnum % 5
            acnum = acnum // 5
        return res

    def ac_process(self, acs):
        final_acs = []
        acs /= self.interval
        acs = np.round(acs)+2.0
        for ac in acs:
            final_acs.append(self.fiv(ac))
        return final_acs

    def split_train(self, fraction):
        if os.path.exists(os.path.join(self.path, self.listpkl)):
            with open(os.path.join(self.path, self.listpkl), 'rb') as f:
                data = pickle.load(f)
                self.data = data['data']
                self.train_list = data['train']
                self.vali_list = data['test']
        else:
            dirlist = os.listdir(self.path)
            np.random.shuffle(dirlist)
            for d in dirlist:
                subdir = os.path.join(self.path, d)
                if os.path.isdir(subdir):
                    datapkl = os.path.join(subdir, 'data.pkl')
                    if os.path.exists(datapkl):
                        with open(datapkl, 'rb') as dataf:
                            pkl_data = pickle.load(dataf)
                            inits = pkl_data['inits']
                            obs = pkl_data['observations']
                            tar_pos = inits['target_joint_pos']
                            obstacle_pos = inits['obstacle_pos']
                            for t in range(len(obs)):
                                config = obs[t]
                                xyzs = tipcoor(config)[3:-3]
                                self.data['obs'].append(np.concatenate((config, tar_pos, obstacle_pos, xyzs)))
                            self.data['acs'].extend(self.ac_process(pkl_data['actions']))
            self.data['obs'] = np.array(self.data['obs'])
            self.data['acs'] = np.array(self.data['acs'])
            id_list = np.arange(len(self.data['acs']))
            train_size = int(fraction*len(id_list))
            self.train_list = id_list[:train_size]
            self.vali_list = id_list[train_size:]
            np.random.shuffle(self.train_list)
            np.random.shuffle(self.vali_list)
            with open(os.path.join(self.path, self.listpkl), 'wb') as f:
                pickle.dump({'data':self.data, 'train': self.train_list, 'test': self.vali_list}, f)

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
        end = cur_pointer + batch_size
        if batch_size<0:
            return self.data['obs'][cur_list], self.data['acs'][cur_list]
        if end<cur_len:
            if split=='train':
                self.train_pointer = end
            else:
                self.val_pointer = end
            return self.data['obs'][cur_list[cur_pointer:end]], self.data['acs'][cur_list[cur_pointer:end]]
        else:
            if split=='train':
                self.train_pointer = end - cur_len
            else:
                self.val_pointer = end - cur_len
            return np.concatenate((self.data['obs'][cur_list[cur_pointer:]], self.data['obs'][cur_list[:end-cur_len]]), 0), \
                   np.concatenate(
                       (self.data['acs'][cur_list[cur_pointer:]], self.data['acs'][cur_list[:end - cur_len]]), 0)


class RecordLoader(object):
    def __init__(self, datapath, sort=False):
        self.path = datapath
        self.sort = sort
        self.samples = {'ob': [], 'ac': [], 'rew': [], 'new': []}
        self.load()
        self.total = len(self.samples['ob'])
        self.pointer = 0

    def load(self):
        dirlist = os.listdir(self.path)
        if self.sort:
            dirlist.sort()
        for d in dirlist:
            with open(os.path.join(self.path, d), 'rb') as f:
                data = pickle.load(f)
                ob = data['ob']
                ac = data['ac']
                rew = data['rew']
            self.samples['ob'].extend(ob)
            self.samples['ac'].extend(ac)
            self.samples['rew'].extend(rew)
            self.samples['new'].extend([0 for _ in range(len(rew)-1)] + [1])

    def get_next_batch(self, batch_size):
        end = self.pointer + batch_size
        pointer = self.pointer
        if end>self.total:
            self.pointer = end - self.total
            data = {'ob': np.array(self.samples['ob'][pointer:]+self.samples['ob'][:self.pointer]),
                    'ac': np.array(self.samples['ac'][pointer:]+self.samples['ac'][:self.pointer]),
                    'rew': np.array(self.samples['rew'][pointer:]+self.samples['rew'][:self.pointer]),
                    'new': np.array(self.samples['new'][pointer:] + self.samples['new'][:self.pointer]),
                    'nextob': np.expand_dims(self.samples['ob'][self.pointer], 0)}
        else:
            self.pointer = end
            data = {'ob': np.array(self.samples['ob'][pointer:end]),
                    'ac': np.array(self.samples['ac'][pointer:end]),
                    'rew': np.array(self.samples['rew'][pointer:end]),
                    'new': np.array(self.samples['new'][pointer:end]),
                    'nextob': np.expand_dims(self.samples['ob'][end%(self.total-1)], 0)}
        return data

class Recorder(object):
    def __init__(self, datapath, *extras, begin=0):
        self.path = datapath
        if not os.path.exists(datapath):
            os.mkdir(datapath)
        self.extras = {}
        self.episode = begin
        for name in extras:
            self.extras[name] = []
        self.reset()

    def record(self, ob, ac, rew, done, *args):
        self.obs.append(ob)
        self.acs.append(ac)
        self.rews.append(rew)
        self.dones.append(done)
        for extra in args:
            self.extras[extra[0]].append(extra[1])
        if done:
            pklfile = os.path.join(self.path, str(self.episode)+'traj.pkl')
            with open(pklfile, 'wb') as f:
                data = {'ob': self.obs, 'ac': self.acs, 'rew': self.rews, 'extra': self.extras}
                pickle.dump(data, f)
            while os.path.getsize(pklfile)==0:
                pickle.dump(data, f)
            print('save {}'.format(self.episode))
            self.reset()
            self.episode += 1

    def reset(self):
        self.obs = []
        self.acs = []
        self.rews = []
        self.dones = []
        for key in self.extras.keys():
            self.extras[key] = []

def analyze_var(path):
    loader = RecordLoader(path, True)
    trajs = []
    ex = 0
    for i in range(loader.total):
        if loader.samples['new'][i]:
            trajs.append([np.array(loader.samples['ob'])[ex:i+1, 0:5], np.array(loader.samples['ac'][ex:i+1])])
            ex = i+1
    print(len(trajs))


def main(args):
    workpath = 'dataset/ur5expert5/'
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    '''dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)'''
    maxdir = 0
    #os.chdir(workpath)
    env = UR5VrepEnvConcat(dof=5, l2_thresh=0.1, random_seed=maxdir)
    rec = Recorder(workpath+'record_var3', 'dis', begin=0)
    i = maxdir + 1
    while i < maxdir + 101:
        print('iter:', i)
        n_path, path = env.reset_expert()
        if n_path==0:
            print(i, 'not found')
            continue
        init = {'init_joint_pos': env.init_joint_pos, 'target_joint_pos': env.target_joint_pos,
                'obstacle_pos': env.obstacle_pos}
        '''os.mkdir(str(i))
        os.mkdir(str(i) + "/img1")
        os.mkdir(str(i) + "/img2")
        os.mkdir(str(i) + "/img3")
        obs = []
        dis = []
        acs = []'''
        observation = env.observation
        for t in range(n_path-1):
            action = path[t+1] - path[t]
            next_observation, rew, done, info = env.step(action)
            rec.record(observation, action, rew, done, ['dis', env.distance])
            observation = next_observation
            if done:
                print(i, 'done')
                break

            '''obs.append(observation['joint'])
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
            pickle.dump(data, f)'''
        print(observation[5:10])
        i = i + 1
    # print("Episode finished after {} timesteps.\tTotal reward: {}".format(t+1,total_reward))
    env.close()
    return 0


if __name__ == '__main__':
    import sys
    analyze_var('dataset/ur5expert5/record_var2')
    #sys.exit(main(sys.argv))