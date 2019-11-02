from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.wrappers import TimeLimit
from baselines.gail.vrep_ur_env_3 import UR5VrepEnvKine
import os.path as osp
from baselines import logger
import datetime
try:
    from mpi4py import MPI
except ImportError:
    MPI = None
from baselines.ddpg.ddpg import learn

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
        env = UR5VrepEnvKine(server_port=19997, l2_thresh=0.08, random_seed=2)
        #env = Monitor(TimeLimit(env, max_episode_steps=120), logger.get_dir() and
        #              osp.join(logger.get_dir(), "monitor.json"))
        env = TimeLimit(env, max_episode_steps=120)
        return env

    env = DummyVecEnv([env_fn])
    model = learn('mlp', env,
                  seed=None,
                  total_timesteps=None,
                  nb_epochs=None, # with default settings, perform 1M steps total
                  nb_epoch_cycles=20,
                  nb_rollout_steps=100,
                  reward_scale=1.0,
                  render=False,
                  render_eval=False,
                  noise_type='adaptive-param_0.2',
                  normalize_returns=False,
                  normalize_observations=True,
                  critic_l2_reg=1e-2,
                  actor_lr=1e-4,
                  critic_lr=1e-3,
                  popart=False,
                  gamma=0.99,
                  clip_norm=None,
                  nb_train_steps=50, # per epoch cycle and MPI worker,
                  nb_eval_steps=100,
                  batch_size=64, # per MPI worker
                  tau=0.01,
                  eval_env=None,
                  param_noise_adaption_interval=50,
                  num_layers=3, num_hidden=256,
                  )
    save_path = './cpt'
    if save_path is not None and rank == 0:
        save_path = osp.expanduser(save_path)
        model.save(save_path)
    env.close()


if __name__ == '__main__':
    main()
