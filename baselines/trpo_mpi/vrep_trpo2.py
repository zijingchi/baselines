import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
import datetime
import gym
import baselines.common.tf_util as U
from baselines.gail.vrep_ur_env import UR5VrepEnvConcat, UR5VrepEnvDrtn
from baselines.gail.vrep_ur_env_3 import UR5VrepEnvKine
from baselines.common.distributions import make_pdtype
from baselines.acktr.utils import dense
import tensorflow as tf
from baselines.common import explained_variance, set_global_seeds
from baselines.common.wrappers import TimeLimit
from baselines.bench.monitor import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ppo2.runner import CustomRunner
from baselines.trpo_mpi.trpo_mpi import learn


def constfn(val):
    def f(_):
        return val
    return f


def runner(env, times, load_path=None):
    suc = 0
    for i in range(times):
        env.reset()
        n_mid = 50
        linear_res = env.directly_towards(n_mid)
        if linear_res == 0:
            suc += 1
    print('{}/{}'.format(suc, times))


def main():
    if MPI is None or MPI.COMM_WORLD.Get_rank() == 0:
        rank = 0
        logger.configure(osp.join('logdir', datetime.datetime.now().__str__()))
    else:
        logger.configure(format_strs=[])
        rank = MPI.COMM_WORLD.Get_rank()

    #env = UR5VrepEnv(server_port=19997)
    #env = Monitor(TimeLimit(env, max_episode_steps=100), logger.get_dir() and
    #              osp.join(logger.get_dir(), "monitor.json"))

    def env_fn():
        env = UR5VrepEnvConcat(server_port=19997, l2_thresh=0.1, random_seed=16)
        #env = Monitor(TimeLimit(env, max_episode_steps=120), logger.get_dir() and
        #              osp.join(logger.get_dir(), "monitor.json"))
        env = TimeLimit(env, max_episode_steps=50)
        return env

    env = DummyVecEnv([env_fn])
    model = learn(env=env,
                  seed=16, network='mlp', gamma=0.95, lam=0.95,
                  total_timesteps=1e5, ent_coef=0.002, max_kl=0.002, cg_iters=25, cg_damping=0.01, vf_stepsize=1e-4,
                  num_hidden=128, num_layers=4, activation=tf.tanh, vf_iters=8, timesteps_per_batch=1600,
                  load_path='./best/cpt_avo1', data_path=None)
    save_path = './cpt0'
    if save_path is not None and rank == 0:
        save_path = osp.expanduser(save_path)
        model.save(save_path)
    #runner(env.envs[0], 200)
    env.close()


if __name__ == '__main__':
    main()
