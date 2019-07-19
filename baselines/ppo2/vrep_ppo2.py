import os
import time
import numpy as np
import os.path as osp
from baselines import logger
from collections import deque
import datetime
import gym
import baselines.common.tf_util as U
from baselines.gail.vrep_ur_env import UR5VrepEnv
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


def constfn(val):
    def f(_):
        return val
    return f


def build_policy(ob_space, ac_space, hid_size, num_hid_layers):
    def polc(nbatch=None, nsteps=None, sess=None):
        return PolicyBuilder(nbatch, ob_space, ac_space, hid_size, num_hid_layers, sess)
    return polc


def learn(env, total_timesteps, eval_env=None, seed=None, nsteps=512, ent_coef=0.0, lr=3e-4,
          vf_coef=0.5, max_grad_norm=0.5, gamma=0.99, lam=0.95, num_hidden=100, num_layers=3,
          log_interval=2, nminibatches=4, noptepochs=4, cliprange=0.2,
          save_interval=0, load_path=None, model_fn=None, update_fn=None, init_fn=None, mpi_rank_weight=1,
          comm=None):
    set_global_seeds(seed)

    if isinstance(lr, float):
        lr = constfn(lr)
    else:
        assert callable(lr)
    if isinstance(cliprange, float):
        cliprange = constfn(cliprange)
    else:
        assert callable(cliprange)
    total_timesteps = int(total_timesteps)
    nenvs = 1
    nbatch = nenvs * nsteps
    nbatch_train = nbatch // nminibatches
    is_mpi_root = (MPI is None or MPI.COMM_WORLD.Get_rank() == 0)
    ob_space = env.observation_space
    ac_space = env.action_space
    # Instantiate the model object (that creates act_model and train_model)
    if model_fn is None:
        from baselines.ppo2.model import CustomModel
        model_fn = CustomModel
    policy = build_policy(ob_space, ac_space, num_hidden, num_layers)
    model = model_fn(policy=policy, nbatch_act=nenvs, nbatch_train=nbatch_train,
                     nsteps=nsteps, ent_coef=ent_coef, vf_coef=vf_coef,
                     max_grad_norm=max_grad_norm, comm=comm, mpi_rank_weight=mpi_rank_weight)
    if load_path is not None:
        model.load(load_path)
    # Instantiate the runner object
    runner = CustomRunner(env=env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)
    if eval_env is not None:
        eval_runner = CustomRunner(env=eval_env, model=model, nsteps=nsteps, gamma=gamma, lam=lam)

    epinfobuf = deque(maxlen=100)
    if eval_env is not None:
        eval_epinfobuf = deque(maxlen=100)

    if init_fn is not None:
        init_fn()

    # Start total timer
    tfirststart = time.perf_counter()

    nupdates = total_timesteps // nbatch
    for update in range(1, nupdates + 1):
        print('{}/{}'.format(update, nupdates))
        assert nbatch % nminibatches == 0
        # Start timer
        tstart = time.perf_counter()
        frac = 1.0 - (update - 1.0) / nupdates
        # Calculate the learning rate
        lrnow = lr(frac)
        # Calculate the cliprange
        cliprangenow = cliprange(frac)

        if update % log_interval == 0 and is_mpi_root: logger.info('Stepping environment...')

        # Get minibatch
        obs, returns, masks, actions, values, neglogpacs, states, epinfos = runner.run()  # pylint: disable=E0632
        if eval_env is not None:
            eval_obs, eval_returns, eval_masks, eval_actions, eval_values, eval_neglogpacs, eval_states, eval_epinfos = eval_runner.run()  # pylint: disable=E0632

        if update % log_interval == 0 and is_mpi_root: logger.info('Done.')

        epinfobuf.extend(epinfos)
        if eval_env is not None:
            eval_epinfobuf.extend(eval_epinfos)

        # Here what we're going to do is for each minibatch calculate the loss and append it.
        mblossvals = []
        if states is None:  # nonrecurrent version
            # Index of each element of batch_size
            # Create the indices array
            inds = np.arange(nbatch)
            for _ in range(noptepochs):
                # Randomize the indexes
                np.random.shuffle(inds)
                # 0 to batch_size with batch_train_size step
                for start in range(0, nbatch, nbatch_train):
                    end = start + nbatch_train
                    mbinds = inds[start:end]
                    slices = (arr[mbinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices))
        else:  # recurrent version
            assert nenvs % nminibatches == 0
            envsperbatch = nenvs // nminibatches
            envinds = np.arange(nenvs)
            flatinds = np.arange(nenvs * nsteps).reshape(nenvs, nsteps)
            for _ in range(noptepochs):
                np.random.shuffle(envinds)
                for start in range(0, nenvs, envsperbatch):
                    end = start + envsperbatch
                    mbenvinds = envinds[start:end]
                    mbflatinds = flatinds[mbenvinds].ravel()
                    slices = (arr[mbflatinds] for arr in (obs, returns, masks, actions, values, neglogpacs))
                    mbstates = states[mbenvinds]
                    mblossvals.append(model.train(lrnow, cliprangenow, *slices, mbstates))

        # Feedforward --> get losses --> update
        lossvals = np.mean(mblossvals, axis=0)
        # End timer
        tnow = time.perf_counter()
        # Calculate the fps (frame per second)
        fps = int(nbatch / (tnow - tstart))

        if update_fn is not None:
            update_fn(update)

        if update % log_interval == 0 or update == 1:
            # Calculates if value function is a good predicator of the returns (ev > 1)
            # or if it's just worse than predicting nothing (ev =< 0)
            ev = explained_variance(values, returns)
            logger.logkv("misc/serial_timesteps", update * nsteps)
            logger.logkv("misc/nupdates", update)
            logger.logkv("misc/total_timesteps", update * nbatch)
            logger.logkv("fps", fps)
            logger.logkv("misc/explained_variance", float(ev))
            logger.logkv('eprewmean', safemean([epinfo['r'] for epinfo in epinfobuf]))
            logger.logkv('eplenmean', safemean([epinfo['l'] for epinfo in epinfobuf]))
            if eval_env is not None:
                logger.logkv('eval_eprewmean', safemean([epinfo['r'] for epinfo in eval_epinfobuf]))
                logger.logkv('eval_eplenmean', safemean([epinfo['l'] for epinfo in eval_epinfobuf]))
            logger.logkv('misc/time_elapsed', tnow - tfirststart)
            for (lossval, lossname) in zip(lossvals, model.loss_names):
                logger.logkv('loss/' + lossname, lossval)

            logger.dumpkvs()
        if save_interval and (update % save_interval == 0 or update == 1) and logger.get_dir() and is_mpi_root:
            checkdir = osp.join(logger.get_dir(), 'checkpoints')
            os.makedirs(checkdir, exist_ok=True)
            savepath = osp.join(checkdir, '%.5i' % update)
            print('Saving to', savepath)
            model.save(savepath)

    return model


# Avoid division error when calculate the mean (in our case if epinfo is empty returns np.nan, not return an error)
def safemean(xs):
    return np.nan if len(xs) == 0 else np.mean(xs)


class PolicyBuilder(object):

    def __init__(self, n_batch, ob_space, ac_space, hid_size, num_hid_layers, sess=None, gaussian_fixed_var=True):
        sequence_length = n_batch
        self.sess = sess or tf.get_default_session()
        self.ob_config = tf.placeholder(name="ob", dtype=tf.float32,
                                   shape=[sequence_length] + list(ob_space.spaces['joint'].shape))
        self.ob_target = tf.placeholder(name="goal", dtype=tf.float32,
                                      shape=[sequence_length] + list(ob_space.spaces['target'].shape))

        self.obs_pos = tf.placeholder(name="obs_pos", dtype=tf.float32,
                                    shape=[sequence_length] + list(ob_space.spaces['obstacle_pos'].shape))
        self.pdtype = pdtype = make_pdtype(ac_space)

        last_out = self.ob_config
        goal_last_out = self.ob_target
        obs_last_out = self.obs_pos

        for i in range(num_hid_layers):
            last_out = dense(last_out, hid_size, "vfcfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                             weight_loss_dict={})
            # last_out = tf.layers.batch_normalization(last_out, training=is_training, name="vfcbn%i"%(i+1))
            last_out = tf.nn.tanh(last_out)
            goal_last_out = dense(goal_last_out, hid_size, "vfgfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                  weight_loss_dict={})
            goal_last_out = tf.nn.tanh(goal_last_out)
            obs_last_out = dense(obs_last_out, hid_size, "vfobsfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                 weight_loss_dict={})
            # obs_last_out = tf.layers.batch_normalization(obs_last_out, training=is_training, name="vfobn%i"%(i+1))
            obs_last_out = tf.nn.tanh(obs_last_out)
        vpred = tf.concat([last_out, goal_last_out, obs_last_out], -1)
        vpred = dense(vpred, 1, "vffinal", weight_init=U.normc_initializer(1.0))
        self.vf = vpred[:, 0]
        # construct policy probability distribution model
        last_out = self.ob_config
        goal_last_out = self.ob_target
        obs_last_out = self.obs_pos

        for i in range(num_hid_layers):
            last_out = dense(last_out, hid_size, "pol_cfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                             weight_loss_dict={})
            last_out = tf.nn.tanh(last_out)
            goal_last_out = dense(goal_last_out, hid_size, "pol_gfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                  weight_loss_dict={})
            goal_last_out = tf.nn.tanh(goal_last_out)
            obs_last_out = dense(obs_last_out, hid_size, "pol_obsfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                 weight_loss_dict={})
            obs_last_out = tf.nn.tanh(obs_last_out)
        last_out = tf.concat([last_out, goal_last_out, obs_last_out], -1)
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.constant_initializer(-3))
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))
        #self.pdparam = pdparam
        self.pd = pdtype.pdfromflat(pdparam)
        self.action = self.pd.sample()
        self.initial_state = None
        self.state = tf.constant([])
        self.neglogp = self.pd.neglogp(self.action)
        self._act = U.function([self.ob_config, self.ob_target, self.obs_pos], [self.action, self.vf, self.state, self.neglogp])

    def step(self, ob_config, ob_target, obs_pos):
        a, v, state, neglop = self._act(ob_config, ob_target, obs_pos)
        if state.size == 0:
            state = None
        return a, v, state, neglop

    def value(self, ob_config, ob_target, obs_pos):
        _, v, _, _ = self.step(ob_config, ob_target, obs_pos)
        return v

    def save(self, save_path):
        U.save_state(save_path, sess=self.sess)

    def load(self, load_path):
        U.load_state(load_path, sess=self.sess)


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
        env = UR5VrepEnv(server_port=19997)
        env = Monitor(TimeLimit(env, max_episode_steps=100), logger.get_dir() and
                      osp.join(logger.get_dir(), "monitor.json"))
        return env

    env = DummyVecEnv([env_fn])
    model = learn(env, 2e6, ent_coef=0.0, lr=3e-4, vf_coef=0.5,  max_grad_norm=0.5, cliprange=0.2)
    save_path = './cpt'
    if save_path is not None and rank == 0:
        save_path = osp.expanduser(save_path)
        model.save(save_path)
    env.close()


if __name__ == '__main__':
    main()
