'''
Reference: https://github.com/openai/imitation
I follow the architecture from the official repository
'''
import tensorflow as tf
import numpy as np

from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common import tf_util as U

def logsigmoid(a):
    '''Equivalent to tf.log(tf.sigmoid(a))'''
    return -tf.nn.softplus(-a)

""" Reference: https://github.com/openai/imitation/blob/99fbccf3e060b6e6c739bdf209758620fcdefd3c/policyopt/thutil.py#L48-L51"""
def logit_bernoulli_entropy(logits):
    ent = (1.-tf.nn.sigmoid(logits))*logits - logsigmoid(logits)
    return ent


class TransitionClassifier(object):
    def __init__(self, env, hidden_size, entcoeff=0.001, lr_rate=1e-3, scope="adversary"):
        self.scope = scope
        self.observation_shape = env.observation_space.shape
        self.actions_shape = env.action_space.shape
        self.input_shape = tuple([o+a for o, a in zip(self.observation_shape, self.actions_shape)])
        self.num_actions = env.action_space.shape[0]
        self.hidden_size = hidden_size
        self.build_ph()
        # Build grpah
        generator_logits = self.build_graph(self.generator_obs_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obs_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits, labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff*entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1-tf.nn.sigmoid(generator_logits)+1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function([self.generator_obs_ph, self.generator_acs_ph, self.expert_obs_ph, self.expert_acs_ph],
                                      self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        self.generator_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="observations_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
        self.expert_obs_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape, name="expert_observations_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert_actions_ph")

    def build_graph(self, obs_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, acs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obs_ph: obs, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward


class TransitionClassifier4Dict(object):
    def __init__(self, env, hidden_size, hidden_layers=2, entcoeff=0.001, ob_shape=16, lr_rate=1e-3, scope="adversary"):
        self.scope = scope
        self.observation_shape = env.observation_space.spaces['joint'].shape
        self.obstacle_pos_shape = env.observation_space.spaces['obstacle_pos'].shape
        self.obstacle_ori_shape = env.observation_space.spaces['obstacle_ori'].shape
        self.actions_shape = env.action_space.shape
        #self.input_shape = tuple([o+a for o, a in zip(ob_shape, self.actions_shape)])
        self.num_actions = env.action_space.shape[0]
        self.ob_shape = ob_shape
        self.hidden_size = hidden_size
        self.hidden_layers = hidden_layers
        self.build_ph()
        # Build grpah
        generator_logits = self.build_graph(self.generator_obser_config_ph, self.generator_goal_ph,
                                            self.generator_obs_pos_ph, self.generator_obs_ori_ph, self.generator_acs_ph, reuse=False)
        expert_logits = self.build_graph(self.expert_obser_config_ph, self.expert_goal_ph, self.expert_obs_pos_ph,
                                         self.expert_obs_ori_ph, self.expert_acs_ph, reuse=True)
        # Build accuracy
        generator_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(generator_logits) < 0.5))
        expert_acc = tf.reduce_mean(tf.to_float(tf.nn.sigmoid(expert_logits) > 0.5))
        # Build regression loss
        # let x = logits, z = targets.
        # z * -log(sigmoid(x)) + (1 - z) * -log(1 - sigmoid(x))
        generator_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=generator_logits,
                                                                 labels=tf.zeros_like(generator_logits))
        generator_loss = tf.reduce_mean(generator_loss)
        expert_loss = tf.nn.sigmoid_cross_entropy_with_logits(logits=expert_logits, labels=tf.ones_like(expert_logits))
        expert_loss = tf.reduce_mean(expert_loss)
        # Build entropy loss
        logits = tf.concat([generator_logits, expert_logits], 0)
        entropy = tf.reduce_mean(logit_bernoulli_entropy(logits))
        entropy_loss = -entcoeff * entropy
        # Loss + Accuracy terms
        self.losses = [generator_loss, expert_loss, entropy, entropy_loss, generator_acc, expert_acc]
        self.loss_name = ["generator_loss", "expert_loss", "entropy", "entropy_loss", "generator_acc", "expert_acc"]
        self.total_loss = generator_loss + expert_loss + entropy_loss
        # Build Reward for policy
        self.reward_op = -tf.log(1 - tf.nn.sigmoid(generator_logits) + 1e-8)
        var_list = self.get_trainable_variables()
        self.lossandgrad = U.function(
            [self.generator_obser_config_ph, self.generator_goal_ph, self.generator_obs_pos_ph,
             self.generator_obs_ori_ph, self.generator_acs_ph, self.expert_obser_config_ph, self.expert_goal_ph,
             self.expert_obs_pos_ph, self.expert_obs_ori_ph, self.expert_acs_ph],
            self.losses + [U.flatgrad(self.total_loss, var_list)])

    def build_ph(self):
        self.generator_obser_config_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape,
                                                        name="generator_obser_config_ph")
        self.generator_goal_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                                        name="generator_obser_config_ph")
        self.generator_obs_pos_ph = tf.placeholder(tf.float32, (None,) + (3,), name="generator_obs_pos_ph")
        self.generator_obs_ori_ph = tf.placeholder(tf.float32, (None,) + (3,), name="generator_obs_ori_ph")
        self.generator_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="actions_ph")
        self.expert_obser_config_ph = tf.placeholder(tf.float32, (None, ) + self.observation_shape,
                                                     name="expert_observations_config_ph")
        self.expert_goal_ph = tf.placeholder(tf.float32, (None,) + self.observation_shape,
                                                        name="expert_obser_config_ph")
        self.expert_obs_pos_ph = tf.placeholder(tf.float32, (None,) + (3,), name="expert_obs_pos_ph")
        self.expert_obs_ori_ph = tf.placeholder(tf.float32, (None,) + (3,), name="expert_obs_ori_ph")
        self.expert_acs_ph = tf.placeholder(tf.float32, (None, ) + self.actions_shape, name="expert_actions_ph")

    def build_graph(self, obs_ph, goal_ph, obs_pos_ph, obs_ori_ph, acs_ph, reuse=False):
        with tf.variable_scope(self.scope):
            if reuse:
                tf.get_variable_scope().reuse_variables()

            '''with tf.variable_scope("obfilter"):
                self.obs_rms = RunningMeanStd(shape=self.observation_shape)
            obs = (obs_ph - self.obs_rms.mean) / self.obs_rms.std'''
            obs_config = obs_ph
            obs_goal = goal_ph
            op_last_out = tf.layers.batch_normalization(obs_pos_ph, True)
            oo_last_out = tf.layers.batch_normalization(obs_ori_ph, True)
            op_last_out = tf.contrib.layers.fully_connected(op_last_out, self.hidden_size, activation_fn=tf.nn.tanh)
            oo_last_out = tf.contrib.layers.fully_connected(oo_last_out, self.hidden_size, activation_fn=tf.nn.tanh)
            obs_last_out = tf.concat([op_last_out, oo_last_out], axis=-1)
            for i in range(self.hidden_layers):
                obs_config = tf.contrib.layers.fully_connected(obs_config, self.hidden_size*2**(self.hidden_layers-1-i), activation_fn=tf.nn.tanh)
                obs_goal = tf.contrib.layers.fully_connected(obs_goal, self.hidden_size*2**(self.hidden_layers-1-i), activation_fn=tf.nn.tanh)
                obs_last_out = tf.contrib.layers.fully_connected(obs_last_out, self.hidden_size*2**(self.hidden_layers-1-i), activation_fn=tf.nn.tanh)
            obs = tf.concat([obs_config, obs_goal, obs_last_out], axis=-1)
            obs = tf.contrib.layers.fully_connected(obs, self.ob_shape)
            _input = tf.concat([obs, acs_ph], axis=1)  # concatenate the two input -> form a transition
            p_h1 = tf.contrib.layers.fully_connected(_input, self.hidden_size, activation_fn=tf.nn.tanh)
            p_h2 = tf.contrib.layers.fully_connected(p_h1, self.hidden_size, activation_fn=tf.nn.tanh)
            logits = tf.contrib.layers.fully_connected(p_h2, 1, activation_fn=tf.identity)
        return logits

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_reward(self, obs, goal, obs_pos, obs_ori, acs):
        sess = tf.get_default_session()
        if len(obs.shape) == 1:
            obs = np.expand_dims(obs, 0)
        if len(goal.shape) == 1:
            goal = np.expand_dims(goal, 0)
        if len(obs_pos.shape) == 1:
            obs_pos = np.expand_dims(obs_pos, 0)
        if len(obs_ori.shape) == 1:
            obs_ori = np.expand_dims(obs_ori, 0)
        if len(acs.shape) == 1:
            acs = np.expand_dims(acs, 0)
        feed_dict = {self.generator_obser_config_ph: obs, self.generator_goal_ph: goal,
                     self.generator_obs_pos_ph: obs_pos, self.generator_obs_ori_ph: obs_ori, self.generator_acs_ph: acs}
        reward = sess.run(self.reward_op, feed_dict)
        return reward