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
from keras.utils import to_categorical
from baselines.gail.vrep_ur_env_3 import UR5VrepEnvKine
from baselines.gail.expert_demo import ExpertDataset, RecordLoader


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


def learn(env, policy_func, dataset, optim_batch_size=256, max_iters=5e4,
          adam_epsilon=1e-7, optim_stepsize=1e-3,
          ckpt_dir=None, log_dir=None, task_name=None,
          verbose=False):

    val_per_iter = int(max_iters/10)
    ob_space = env.observation_space
    ac_space = env.action_space
    pi = policy_func("pi", ob_space, ac_space)  # Construct network for new policy
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

    all_var_list = pi.get_trainable_variables()
    var_list = [v for v in all_var_list if
                v.name.startswith("pi/pol") or v.name.startswith("pi/logstd") or v.name.startswith("pi/obs")]
    AdamOp = tf.train.AdamOptimizer(learning_rate=optim_stepsize, epsilon=adam_epsilon).minimize(loss, var_list=var_list)

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

        U.get_session().run(AdamOp, feed_dict={ob_config: cur, ob_target: tar, obs_pos: ob_expert[:, -6:-3],
                                    obs_ori:ob_expert[:, -3:], ac: avo, stochastic: True})

        if verbose and iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(-1, 'val')
            tar = ob_expert[:, dof:2 * dof]
            cur = ob_expert[:, :dof]
            avo = np.zeros_like(cur)
            for i in range(len(avo)):
                avo[i] = ac_expert[i] - thresh * (tar[i] - cur[i]) / np.linalg.norm(tar[i] - cur[i])
            val_loss = U.get_session().run(loss, feed_dict={ob_config: cur, ob_target: tar, obs_pos: ob_expert[:, -6:-3],
                                obs_ori:ob_expert[:, -3:], ac: avo, stochastic: True})
            logger.log("Validation loss: {}".format(np.rad2deg(np.sqrt(val_loss))))
    allvar = pi.get_variables()
    savevar = [v for v in allvar if "Adam" not in v.name]
    U.save_variables(savedir_fname, variables=savevar)
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


def add_vtarg_and_adv(seg, gamma, lam):
    new = np.append(seg["new"], 0) # last element is only used for last vtarg, but we already zeroed it if last new = 1
    vpred = np.append(seg["vpred"], seg["nextvpred"])
    T = len(seg["rew"])
    seg["adv"] = gaelam = np.empty(T, 'float32')
    rew = seg["rew"]
    lastgaelam = 0
    for t in reversed(range(T)):
        nonterminal = 1-new[t+1]
        delta = rew[t] + gamma * vpred[t+1] * nonterminal - vpred[t]
        gaelam[t] = lastgaelam = delta + gamma * lam * nonterminal * lastgaelam
    seg["tdlamret"] = seg["adv"] + seg["vpred"]


def add_vtarg(seg, gamma):
    T = len(seg['rew'])
    gts = np.empty(T, 'float32')
    lastg = 0
    new = seg['new']
    rew = seg['rew']
    for t in reversed(range(T)):
        gts[t] = lastg = rew[t] + gamma*(1-new[t])*lastg
    seg['vpred'] = gts

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


def bc_train(expert_path, scope_name, pol, lr, cpt_path, batch_size,  max_iter, dof, thresh, val_per_iter):
    all_var = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope_name)
    all_var = [v for v in all_var if v.name.find('/vf')==-1]
    acs = pol.pdtype.sample_placeholder([None])
    obs = pol.X
    #loss = pol.pd.neglogp(acs)
    metrics = tf.constant([1.5, 1.5, 1.5, 0.5, 0.5])[:dof]
    loss2 = tf.reduce_mean(tf.square(tf.multiply(metrics, acs-pol.action)))
    AdamOp = tf.train.AdamOptimizer(learning_rate=lr, epsilon=1e-9).minimize(loss2, var_list=all_var)
    grad = tf.gradients(loss2, all_var)
    savedir_fname = osp.join(cpt_path, 'bcmodel')
    savevar = [v for v in all_var if "Adam" not in v.name]
    U.initialize()
    if osp.exists(savedir_fname):
        U.load_variables(savedir_fname, variables=savevar)
    dataset = ExpertDataset(expert_path=expert_path, traj_limitation=-1)
    # dof += 1
    for iter_so_far in tqdm(range(int(max_iter))):
        ob_expert, ac_expert = dataset.get_next_batch(batch_size, 'train')
        tar = ob_expert[:, dof:2 * dof]
        cur = ob_expert[:, :dof]
        avo = np.zeros_like(cur)
        for i in range(len(avo)):
            avo[i] = ac_expert[i] - thresh * (tar[i] - cur[i]) / np.linalg.norm(tar[i] - cur[i])
        g = U.get_session().run([grad, AdamOp], feed_dict={obs: ob_expert, acs: avo})

        if iter_so_far % val_per_iter == 0:
            ob_expert, ac_expert = dataset.get_next_batch(batch_size, 'val')
            tar = ob_expert[:, dof:2 * dof]
            cur = ob_expert[:, :dof]
            avo = np.zeros_like(cur)
            for i in range(len(avo)):
                avo[i] = ac_expert[i] - thresh * (tar[i] - cur[i]) / np.linalg.norm(tar[i] - cur[i])
            val_loss = U.get_session().run(loss2, feed_dict={obs: ob_expert, acs: avo})
            val_loss = np.mean(val_loss)

            logger.log("Validation loss: {}".format(np.rad2deg(np.sqrt(val_loss))))

    U.save_variables(savedir_fname, variables=savevar)

def bc_val(env, pol, max_iter):
    count = 0
    ob = env.reset()
    for i in range(max_iter):

        print(i)
        done = False
        state = None
        for j in range(120):
            action, value, state, neglogpac = pol.step(ob[0], S=state, M=done)
            ob, r, done, infos = env.step(action[0])
            if done:
                if infos[0]['status'] == 'reach':
                    count += 1
                break
    print('success rate: ', count/max_iter)

def vec2dir(v):
    v = v / np.linalg.norm(v)  # 指向目标的单位角向量
    a1 = np.arccos(v[0])
    a2 = np.arcsin(v[1] / np.sin(a1))
    a3 = np.arcsin(v[2] / (np.sin(a1) * np.cos(a2)))
    b = np.sin(a1) * np.cos(a2) * np.cos(a3)
    a4 = np.arctan2(v[3] / b, v[4] / b)  # [a1, a2, a3, a4]是tg的方向向量
    return np.array([a1, a2, a3, a4])

def get_task_name(args):
    task_name = 'BC'
    task_name += '.{}'.format(args.env_id.split("-")[0])
    task_name += '.traj_limitation_{}'.format(args.traj_limitation)
    task_name += ".seed_{}".format(args.seed)
    return task_name


def bc_learn(pi, data_path, ob, ac, var_list, optim_batch_size=256, max_iters=5e4,
             adam_epsilon=1e-7, optim_stepsize=1e-3, ckpt_dir=None, verbose=False):
    val_per_iter = int(max_iters / 10)
    #norm_ac = tf.transpose(tf.transpose(ac)/tf.norm(ac, axis=1))
    #norm_piac = tf.transpose(pi.action)/tf.norm(pi.action, axis=1)
    norm_ac = tf.nn.l2_normalize(ac, axis=1)
    norm_piac = tf.nn.l2_normalize(pi.action, axis=1)
    loss = -tf.reduce_mean(tf.reduce_sum(tf.multiply(norm_ac, norm_piac), axis=1))
    #loss = -tf.reduce_mean(batch_dot(norm_ac, norm_piac))
    AdamOp = tf.train.AdamOptimizer(learning_rate=optim_stepsize, epsilon=adam_epsilon).minimize(loss,
                                                                                                 var_list=var_list)
    U.initialize()
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, 'model4x128')
    if osp.exists(savedir_fname):
        U.load_variables(savedir_fname, var_list)
        #return savedir_fname
    dataset = ExpertDataset(data_path, 0.8)
    logger.log("Pretraining with Behavior Cloning...")
    niter = dataset.n_val // optim_batch_size
    sum_val_loss = 0
    for _ in range(niter):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'val')
        val_loss = U.get_session().run(loss, feed_dict={ob: ob_expert, ac: ac_expert})
        sum_val_loss += val_loss
    logger.log("Init Validation loss: {}".format(sum_val_loss/niter))
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        U.get_session().run(AdamOp, feed_dict={ob: ob_expert, ac: ac_expert})
        if verbose and iter_so_far % val_per_iter == 0:
            niter = dataset.n_val//optim_batch_size
            sum_val_loss = 0
            for _ in range(niter):
                ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'val')
                val_loss = U.get_session().run(loss, feed_dict={ob: ob_expert, ac: ac_expert})
                sum_val_loss += val_loss
            logger.log("Validation loss: {}".format(sum_val_loss/niter))
    U.save_variables(savedir_fname, variables=var_list)
    return savedir_fname


def vf_bc(pi, ob, ret, data_path, vferr, var_list, batch_size=256, max_iters=200, lam=0.95, gamma=0.95,
          adam_epsilon=1e-7, optim_stepsize=1e-4, ckpt_dir=None, verbose=False):
    AdamOp = tf.train.AdamOptimizer(learning_rate=optim_stepsize, epsilon=adam_epsilon).minimize(vferr,
                                                                                                 var_list=var_list)
    loader = RecordLoader(data_path)
    U.initialize()
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, 'vf_bc')
    #if osp.exists(savedir_fname):
        #U.load_variables(savedir_fname, var_list)
        #return savedir_fname
    for i in range(int(max_iters)):
        record_data = loader.get_next_batch(batch_size)
        record_data['vpred'] = U.get_session().run(pi.vf, {ob: record_data['ob']}).flatten()
        record_data['nextvpred'] = U.get_session().run(pi.vf, {ob: record_data['nextob']})
        add_vtarg_and_adv(record_data, gamma, lam)
        U.get_session().run(AdamOp, {ob: record_data['ob'], ret: record_data['tdlamret']})
        if verbose and i%50==0:
            err = U.get_session().run(vferr, {ob: record_data['ob'], ret: record_data['tdlamret']})
            logger.log('iter: {}, tderr: {}'.format(i, err))
    U.save_variables(savedir_fname, variables=var_list)
    return savedir_fname


def vf_mc_bc(pi, ob, ret, data_path, vferr, var_list, batch_size=256, max_iters=200, lam=0.95, gamma=0.95,
          adam_epsilon=1e-7, optim_stepsize=1e-4, ckpt_dir=None, verbose=False):
    AdamOp = tf.train.AdamOptimizer(learning_rate=optim_stepsize, epsilon=adam_epsilon).minimize(vferr,
                                                                                                 var_list=var_list)
    loader = RecordLoader(data_path)
    U.initialize()
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, 'vf_bc')
    if osp.exists(savedir_fname):
        U.load_variables(savedir_fname, var_list)
        #return savedir_fname
    for i in range(int(max_iters)):
        record_data = loader.get_next_batch(batch_size)
        record_data['vpred'] = U.get_session().run(pi.vf, {ob: record_data['ob']}).flatten()
        add_vtarg(record_data, gamma)
        U.get_session().run(AdamOp, {ob: record_data['ob'], ret: record_data['vpred']})
        if verbose and i%50==0:
            err = U.get_session().run(vferr, {ob: record_data['ob'], ret: record_data['vpred']})
            logger.log('iter: {}, tderr: {}'.format(i, err))
    U.save_variables(savedir_fname, variables=var_list)
    return savedir_fname


def bc_dis(act, data_path, ob, var_list, optim_batch_size=256, max_iters=5e4,
             adam_epsilon=1e-7, optim_stepsize=1e-3, ckpt_dir=None, verbose=False):
    val_per_iter = int(max_iters / 10)
    ac_true = tf.placeholder(dtype=tf.float32, shape=(None, 125), name='label')
    ac_pred = tf.placeholder(dtype=tf.float32, shape=(None, 125), name='pi_ac')
    loss = tf.reduce_sum(tf.nn.softmax_cross_entropy_with_logits(ac_pred, ac_true))
    AdamOp = tf.train.AdamOptimizer(learning_rate=optim_stepsize, epsilon=adam_epsilon).minimize(loss,
                                                                                                 var_list=var_list)

    def categorical_accuracy(y_true, y_pred):
        return tf.cast(tf.equal(tf.argmax(y_true, axis=-1),
                                tf.argmax(y_pred, axis=-1)),
                       'float32')
    acc = categorical_accuracy(ac_true, ac_pred)
    U.initialize()
    if ckpt_dir is None:
        savedir_fname = tempfile.TemporaryDirectory().name
    else:
        savedir_fname = osp.join(ckpt_dir, 'model')
    if osp.exists(savedir_fname):
        U.load_variables(savedir_fname, var_list)
        # return savedir_fname
    dataset = ExpertDataset(data_path, 0.8)
    logger.log("Pretraining with Behavior Cloning...")
    niter = dataset.n_val // optim_batch_size
    sum_val_loss = 0
    for _ in range(niter):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'val')
        val_loss = U.get_session().run(loss, feed_dict={ob: ob_expert, ac_true: to_categorical(ac_expert),
                                                        ac_pred: act(ob_expert)})
        sum_val_loss += val_loss
    logger.log("Init Validation loss: {}".format(sum_val_loss / niter))
    for iter_so_far in tqdm(range(int(max_iters))):
        ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'train')
        U.get_session().run(AdamOp, feed_dict={ob: ob_expert, ac_true: to_categorical(ac_expert),
                                               ac_pred: act(ob_expert)})
        if verbose and iter_so_far % val_per_iter == 0:
            niter = dataset.n_val // optim_batch_size
            sum_val_loss = 0
            for _ in range(niter):
                ob_expert, ac_expert = dataset.get_next_batch(optim_batch_size, 'val')
                val_loss, val_acc = U.get_session().run([loss, acc], feed_dict={ob: ob_expert, ac_true: to_categorical(ac_expert),
                                                                ac_pred: act(ob_expert)})
                sum_val_loss += val_loss
            logger.log("Validation loss: {}".format(sum_val_loss / niter))
    U.save_variables(savedir_fname, variables=var_list)
    return savedir_fname

def main(args):
    from baselines.common.policies import build_policy

    U.make_session(num_cpu=1).__enter__()
    set_global_seeds(args.seed)
    env = UR5VrepEnvKine(l2_thresh=0.08, random_seed=11)
    #env = gym.wrappers.TimeLimit(env, max_episode_steps=200)

    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "monitor.json"))
    env.seed(args.seed)
    policy_fn = build_policy(env, 'mlp', normalize_observations=False,
                             num_layers=4, num_hidden=256)

    ob_ph = tf.placeholder(tf.float32, (None, 25), 'ob')
    ac_ph = tf.placeholder(tf.float32, (None, 5), 'ac')
    with tf.variable_scope("ppo2_model"):
        pi = policy_fn(observ_placeholder=ob_ph)
    var_list = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, 'ppo2_model')
    savedir_fname = bc_learn(pi, '/home/czj/Downloads/ur5expert', ob_ph, ac_ph, var_list, max_iters=240, verbose=True,
                             ckpt_dir='bc', optim_stepsize=1e-4)
    gym.logger.setLevel(logging.WARN)

    '''avg_len, avg_ret = runner(env,
                              policy_fn,
                              savedir_fname,
                              timesteps_per_batch=200,
                              number_trajs=100,
                              stochastic_policy=args.stochastic_policy,
                              save=args.save_sample,
                              reuse=True)'''
    suc = 0
    for i in range(200):
        ob = env.reset()
        while True:
            ac, v, _, negpa = pi.step(ob)
            ob, rew, new, info = env.step(ac[0])
            if new:
                if info['status']=='reach':
                    suc += 1
                break
    print('{}/{}'.format(suc, 200))
    env.close()


if __name__ == '__main__':
    args = argsparser()
    main(args)
