import os
import cv2
import time
import pickle
from baselines.gail.vrep_ur_env import UR5VrepEnv
import numpy as np
pi = np.pi


def main(args):
    workpath = 'dataset/ur5expert3/'
    if not os.path.exists(workpath):
        os.mkdir(workpath)
    dirlist = os.listdir(workpath)
    numlist = [int(s) for s in dirlist]
    if len(numlist) == 0:
        maxdir = -1
    else:
        maxdir = max(numlist)
    os.chdir(workpath)
    env = UR5VrepEnv(dof=5, enable_cameras=True, l2_thresh=0.08, random_seed=3)
    i = maxdir + 1
    while i < maxdir + 1000:
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
        acs = []
        for t in range(n_path-1):
            action = path[t+1] - path[t]
            observation, _, done, _ = env.step(action)
            obs.append(observation['joint'])
            #ac_expert[i] - thresh * (tar[i] - cur[i]) / np.linalg.norm(tar[i] - cur[i])
            #action = np.clip(action, env.action_space.low, env.action_space.high)
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