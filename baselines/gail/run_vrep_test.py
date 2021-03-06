'''
Disclaimer: this code is highly based on trpo_mpi at @openai/baselines and @openai/imitation
'''

import argparse
import os.path as osp
import logging
from mpi4py import MPI
from tqdm import tqdm
from baselines.gail.vrep_ur_env import UR5VrepEnv, UR5VrepEnvMultiObstacle, UR5VrepEnvDrtn
import numpy as np
import gym

from baselines.gail import mlp_policy
from baselines.common import set_global_seeds, tf_util as U
from baselines.common.misc_util import boolean_flag
from baselines import bench
from baselines import logger
from baselines.gail.adversary import TransitionClassifier4Dict


def argsparser():
    parser = argparse.ArgumentParser("Tensorflow Implementation of GAIL")
    parser.add_argument('--env_id', help='environment ID', default='VREP-UR')
    parser.add_argument('--seed', help='RNG seed', type=int, default=0)
    parser.add_argument('--expert_path', type=str, default='/home/czj/vrep_path_dataset/4_7/')
    parser.add_argument('--checkpoint_dir', help='the directory to save model', default='checkpoint')
    parser.add_argument('--log_dir', help='the directory to save log file', default='log')
    parser.add_argument('--load_model_path', help='if provided, load the model', type=str, default=None)
    # Task
    parser.add_argument('--task', type=str, choices=['train', 'evaluate', 'sample'], default='train')
    # for evaluatation
    boolean_flag(parser, 'stochastic_policy', default=True, help='use stochastic/deterministic policy to evaluate')
    boolean_flag(parser, 'save_sample', default=False, help='save the trajectories or not')
    #  Mujoco Dataset Configuration
    parser.add_argument('--traj_limitation', type=int, default=-1)
    # Optimization Configuration
    parser.add_argument('--g_step', help='number of steps to train policy in each epoch', type=int, default=3)
    parser.add_argument('--d_step', help='number of steps to train discriminator in each epoch', type=int, default=2)
    # Network Configuration (Using MLP Policy)
    parser.add_argument('--policy_hidden_size', type=int, default=256)
    parser.add_argument('--adversary_hidden_size', type=int, default=256)
    # Algorithms Configuration
    parser.add_argument('--algo', type=str, choices=['trpo', 'ppo'], default='trpo')
    parser.add_argument('--max_kl', type=float, default=0.02)
    parser.add_argument('--policy_entcoeff', help='entropy coefficiency of policy', type=float, default=1e-3)
    parser.add_argument('--adversary_entcoeff', help='entropy coefficiency of discriminator', type=float, default=1e-3)
    # Traing Configuration
    parser.add_argument('--save_per_iter', help='save model every xx iterations', type=int, default=100)
    parser.add_argument('--num_timesteps', help='number of timesteps per episode', type=int, default=2e5)
    # Behavior Cloning
    boolean_flag(parser, 'pretrained', default=False, help='Use BC to pretrain')
    parser.add_argument('--BC_max_iter', help='Max iteration for training BC', type=int, default=8e3)
    return parser.parse_args()


def get_task_name(args):
    task_name = args.algo + "_gail."
    if args.pretrained:
        task_name += "with_pretrained."
    if args.traj_limitation != np.inf:
        task_name += "transition_limitation_%d." % args.traj_limitation
    task_name += args.env_id.split("-")[0]
    task_name = task_name + ".g_step_" + str(args.g_step) + ".d_step_" + str(args.d_step) + \
        ".policy_entcoeff_" + str(args.policy_entcoeff) + ".adversary_entcoeff_" + str(args.adversary_entcoeff)
    task_name += ".seed_" + str(args.seed)
    return task_name


def main(args):
    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    #env = UR5VrepEnvMultiObstacle(l2_thresh=0.12)
    env = UR5VrepEnvDrtn(l2_thresh=0.08)
    from baselines.common.wrappers import TimeLimit
    env = TimeLimit(env, max_episode_steps=100)

    def policy_fn(name, ob_space, ac_space, reuse=False):
        return mlp_policy.MLPD(name=name, ob_space=ob_space, ac_space=ac_space,
                               reuse=reuse, hid_size=args.policy_hidden_size, num_hid_layers=3)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    gym.logger.setLevel(logging.WARN)
    #task_name = get_task_name(args)
    task_name = "UR5PathPlanTRPO3"
    args.checkpoint_dir = osp.join(args.checkpoint_dir, task_name)
    args.log_dir = osp.join(args.log_dir, task_name)

    if args.task == 'train':
        #dataset = PathPlanDset(expert_path=args.expert_path, traj_limitation=args.traj_limitation)
        dataset = {}
        #reward_giver = TransitionClassifier4Dict(env, args.adversary_hidden_size, hidden_layers=3,
        #                                        ob_shape=16, entcoeff=args.adversary_entcoeff)
        reward_giver = {}
        train(env,
              args.seed,
              policy_fn,
              reward_giver,
              dataset,
              args.algo,
              args.g_step,
              args.d_step,
              args.policy_entcoeff,
              args.num_timesteps,
              args.save_per_iter,
              args.checkpoint_dir,
              args.log_dir,
              args.pretrained,
              args.BC_max_iter,
              task_name
              )
    elif args.task == 'evaluate':
        runner(env,
               policy_fn,
               args.load_model_path,
               timesteps_per_batch=100,
               number_trajs=30,
               stochastic_policy=args.stochastic_policy,
               save=args.save_sample
               )
    else:
        raise NotImplementedError
    env.close()


def train(env, seed, policy_fn, reward_giver, dataset, algo,
          g_step, d_step, policy_entcoeff, num_timesteps, save_per_iter,
          checkpoint_dir, log_dir, pretrained, BC_max_iter, task_name=None):

    pretrained_weight = None
    if pretrained and (BC_max_iter > 0):
        # Pretrain with behavior cloning
        from baselines.gail import behavior_clone_vrep
        pretrained_weight = behavior_clone_vrep.learn(env, policy_fn, dataset,
                                                      max_iters=BC_max_iter, ckpt_dir='checkpoint/BC', verbose=True)
    path = 'UR5PathPlanTRPO2/UR5PathPlanTRPO2'
    #pretrained_weight = "checkpoint/" + path
    pretrained_weight = None
    if algo == 'trpo':
        #from baselines.gail import trpo_mpi4vrep
        from baselines.trpo_mpi import trpo_vrep
        # Set up for MPI seed
        rank = MPI.COMM_WORLD.Get_rank()
        if rank != 0:
            logger.set_level(logger.DISABLED)
        workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
        set_global_seeds(workerseed)
        env.seed(workerseed)
        trpo_vrep.learn(env, policy_fn, rank,
                        pretrained=pretrained, pretrained_weight=pretrained_weight,
                        g_step=1, d_step=d_step,
                        entcoeff=policy_entcoeff,
                        max_timesteps=num_timesteps,
                        ckpt_dir=checkpoint_dir, log_dir=log_dir,
                        save_per_iter=save_per_iter,
                        timesteps_per_batch=1024,
                        max_kl=0.01, cg_iters=15, cg_damping=0.5,
                        gamma=0.995, lam=0.97,
                        vf_iters=5, vf_stepsize=1e-2,
                        task_name=task_name)
    else:
        raise NotImplementedError


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
    #U.load_state(load_model_path)
    import tensorflow as tf
    saver = tf.train.Saver()
    saver.restore(tf.get_default_session(), load_model_path)
    obs_list = []
    acs_list = []
    len_list = []
    ret_list = []
    suc_count = 0
    for _ in tqdm(range(number_trajs)):
        traj, suc = traj_1_generator(pi, env, timesteps_per_batch, stochastic=stochastic_policy)
        if suc:
            suc_count += 1
        obs, acs, ep_len, ep_ret = traj['ob'], traj['ac'], traj['ep_len'], traj['ep_ret']
        obs_list.append(obs)
        acs_list.append(acs)
        len_list.append(ep_len)
        ret_list.append(ep_ret)
        env.env.env.draw_policy_path(np.array(obs))

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
    print("Success rate: {}/{}".format(suc_count, number_trajs))
    return avg_len, avg_ret


# Sample one trajectory (until trajectory end)
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
        obs.append(ob['joint'])
        news.append(new)
        acs.append(ac)

        ob, rew, new, info = env.step(ac)
        rews.append(rew)

        cur_ep_ret += rew
        cur_ep_len += 1
        if new or t >= horizon:
            break
        t += 1
    if info['status']=='reach':
        suc = True
    else:
        suc = False
    obs.append(ob['joint'])
    obs = np.array(obs)
    rews = np.array(rews)
    news = np.array(news)
    acs = np.array(acs)
    traj = {"ob": obs, "rew": rews, "new": news, "ac": acs,
            "ep_ret": cur_ep_ret, "ep_len": cur_ep_len}
    return traj, suc


if __name__ == '__main__':
    args = argsparser()
    main(args)
