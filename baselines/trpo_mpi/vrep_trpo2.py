import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
import datetime
import gym
import baselines.common.tf_util as U
from baselines.gail.vrep_ur_env import UR5VrepEnvConcat
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
        env = UR5VrepEnvConcat(server_port=19997, l2_thresh=0.08)
        #env = Monitor(TimeLimit(env, max_episode_steps=120), logger.get_dir() and
        #              osp.join(logger.get_dir(), "monitor.json"))
        env = TimeLimit(env, max_episode_steps=120)
        return env

    env = DummyVecEnv([env_fn])
    model = learn(env=env,
        seed=1, network='mlp',
        total_timesteps=2e5, ent_coef=0.0, max_kl=0.01, cg_iters=15, cg_damping=0.1, vf_stepsize=3e-4,
        num_hidden=128, num_layers=3, timesteps_per_batch=1024)
    save_path = './cpt'
    if save_path is not None and rank == 0:
        save_path = osp.expanduser(save_path)
        model.save(save_path)
    env.close()


if __name__ == '__main__':
    main()
