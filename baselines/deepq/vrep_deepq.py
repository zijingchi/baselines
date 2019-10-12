from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.wrappers import TimeLimit
from baselines.gail.vrep_ur_env_3 import UR5VrepEnvDis
import os.path as osp
from baselines import logger
import datetime
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.deepq.deepq import learn

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
        env = UR5VrepEnvDis(server_port=19997, l2_thresh=0.08, num_per_dim=5, dof=3, discrete_stepsize=0.03)
        #env = Monitor(TimeLimit(env, max_episode_steps=120), logger.get_dir() and
        #              osp.join(logger.get_dir(), "monitor.json"))
        env = TimeLimit(env, max_episode_steps=120)
        return env

    env = DummyVecEnv([env_fn])
    model = learn(env, 'mlp', seed=1,
                  total_timesteps=50000, buffer_size=10000,
                  exploration_fraction=0.2,
                  exploration_final_eps=0.1,
                  train_freq=1,
                  batch_size=64,
                  print_freq=50,
                  checkpoint_freq=10000,
                  checkpoint_path=None,
                  learning_starts=1000,
                  gamma=0.95,
                  target_network_update_freq=50,
                  prioritized_replay=True,
                  prioritized_replay_alpha=0.6,
                  prioritized_replay_beta0=0.4,
                  prioritized_replay_eps=1e-6,)
    save_path = './cpt'
    if save_path is not None and rank == 0:
        save_path = osp.expanduser(save_path)
        model.save(save_path)
    env.close()


if __name__ == '__main__':
    main()