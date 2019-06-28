'''
The code is used to train BC imitator, or pretrained GAIL imitator
'''

import argparse
import tempfile
import os.path as osp
import gym
import logging
from tqdm import tqdm

import tensorflow as tf
import numpy as np
from baselines.gail import mlp_policy
from baselines import bench
from baselines import logger
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines.common.mpi_adam import MpiAdam
from baselines.gail.vrep_ur_env import UR5VrepEnv, PathPlanDset


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of Behavior Cloning")
    parser.add_argument('--env_id', help='environment ID', default='Hopper-v1')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='/home/ubuntu/vdp/5_2/')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='BC/log')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=200)
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=1e4)
    return parser.parse_args()


def learn(env, policy_func, dataset, optim_batch_size=100, max_iters=1e4,
          adam_epsilon=1e-5, optim_stepsize=1e-3,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new polic
    dof = 5
    # placeholder
    ob_config = U.get_placeholder_cached(name="ob")
    ob_target = U.get_placeholder_cached(name="goal")
    obs_pos = U.get_placeholder_cached(name="obs_pos")
    obs_ori = U.get_placeholder_cached(name="obs_ori")
    ac = pi.pdtype.sample_placeholder([None])

    stochastic = U.get_placeholder_cached(name="stochastic")
    loss = tf.reduce_mean(tf.square(ac-pi.ac))
    # loss = tf.reduce_mean(pi.pd.neglogp(ac))
    #var_list = pi.get_trainable_variables()

    all_var_list = pi.get_variables()
    var_list = [v for v in all_var_list if
                v.name.startswith("pi/pol") or v.name.startswith("pi/logstd") or v.name.startswith("pi/obs")]
    adam = MpiAdam(var_list, epsilon=adam_epsilon)
    lossandgrad = U.function([ob_config, ob_target, obs_pos, obs_ori, ac, stochastic], [loss]+[U.flatgrad(loss, var_list)])

    U.initialize()
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, 'model')
    if osp.exists(savedir_fname):
        try:
            U.load_variables(savedir_fname, pi.get_variables())
        except:
            print("size of the pretrained model does not match the current model")
    adam.sync()
    logger.log("Pretraining with Behavior Cloning...")
    thresh = 0.1
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        tar = ob_expert[:, dof:2*dof]
        cur = ob_expert[:, :dof]
        avo = np.zeros_like(cur)
        for i in range(len(avo)):
            avo[i] = ac_expert[i] - thresh * (tar[i] - cur[i]) / np.linalg.norm(tar[i] - cur[i])
        # avo = ac_expert - thresh * (tar - cur) / np.linalg.norm(tar - cur)
        train_loss, g = lossandgrad(cur, tar, ob_expert[:, -6:-3], ob_expert[:, -3:], avo, True)
        adam.update(g, optim_stepsize)
        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            tar = ob_expert[:, dof:2 * dof]
            cur = ob_expert[:, :dof]
            avo = np.zeros_like(cur)
            for i in range(len(avo)):
                avo[i] = ac_expert[i] - thresh * (tar[i] - cur[i]) / np.linalg.norm(tar[i] - cur[i])
            val_loss, _ = lossandgrad(cur, tar, ob_expert[:, -6:-3], ob_expert[:, -3:], avo, True)
            logger.log("Training loss: {}, Validation loss: {}".format(np.rad2deg(np.sqrt(train_loss)), np.rad2deg(np.sqrt(val_loss))))

    U.save_variables(savedir_fname, variables=pi.get_variables())
    return savedir_fname


def runner(env, policy_func, load_model_path, timesteps_per_batch, number_trajs,
           stochastic_policy, save=False, reuse=False):

    # Setup network
    # ----------------------------------------
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space, reuse=reuse)
    U.initialize()
    # Prepare for rollouts
    # ----------------------------------------
    U.load_variables(load_model_path, pi.get_variables())

    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    for _ in tqdm(range(number_trajs)):
        traj = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
    if stochastic_policy:
        print('stochastic policy:')
    else:
        print('deterministic policy:')
    if save:
        filename = load_model_path.split('/')[-1] + '.' + env.spec.id
        np.savez(filename, obs=np.array(obs_list), acs=np.array(acs_list),
                 lens=np.array(len_list), rets=np.array(ret_list))
    avg_len = sum(len_list)/len(len_list)
    avg_ret = sum(ret_list)/len(ret_list)
    print("Average length:", avg_len)
    print("Average return:", avg_ret)
    return avg_len, avg_ret


def traj_1_generator(pi, env, horizon, stochastic):

    t = 0
    ac = env.action_space.sample()  # not used, just so we have the datatype
    new = True  # marks if we're on first timestep of an episode

    ob = env.reset()
    cur_ep_ret = 0  # return in current episode
    cur_ep_len = 0  # len of current episode

    # Initialize history arrays
    obs = []
    rews = []
    news = []
    acs = []

    while True:
        ac, vpred = pi.act(stochastic, ob)
        obs.append(ob)
        news.append(new)
        acs.append(ac)

        ob, rew, new, _ = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1

    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj


def get_task_name(args):
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = UR5VrepEnv(obs_space_type='dict')
    env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MlpPolicy4Dict(name=name, ob_space=ob_space, ac_space=ac_space,
                                    reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=3)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    task_name = get_task_name(args)
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)
    dataset = PathPlanDset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
    savedir_fname = learn(env,
                          policy_fn,
                          dataset,
                          max_iters=args.BC_max_iter,
                          ckpt_dir=args.checkpoint_dir,
                          log_dir=args.log_dir,
                          task_name=task_name,
                          verbose=True)
    #savedir_fname = osp.join(args.checkpoint_dir, task_name)
    avg_len, avg_ret = runner(env,
                              policy_fn,
                              savedir_fname,
                              timesteps_per_batch=200,
                              number_trajs=50,
                              stochastic_policy=args.stochastic_policy,
                              save=args.save_sample,
                              reuse=True)
    env.close()


if __name__ == '__main__':
    args = argsparser()
    main(args)
