import os.path as osp
from baselines import logger

import datetime

from baselines.gail.vrep_ur_env import UR5VrepEnvConcat
from baselines.common.wrappers import TimeLimit
from baselines.bench.monitor import Monitor
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
try:
    from mpi4py import MPI
except ImportError:
    MPI = None

from baselines.ppo2.ppo2 import learn


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
        env = Monitor(TimeLimit(env, max_episode_steps=120), logger.get_dir() and
                      osp.join(logger.get_dir(), "monitor.json"))
        return env

    env = DummyVecEnv([env_fn])
    model = learn(env=env,
        seed=1, network='mlp',
        total_timesteps=2e5, ent_coef=0.01, lr=9e-5, vf_coef=0.5,  max_grad_norm=0.5, cliprange=0.2, save_interval=4,
        log_interval=1, num_hidden=128, num_layers=3, nsteps=1024)
    save_path = './cpt'
    if save_path is not None and rank == 0:
        save_path = osp.expanduser(save_path)
        model.save(save_path)
    env.close()


if __name__ == '__main__':
    main()