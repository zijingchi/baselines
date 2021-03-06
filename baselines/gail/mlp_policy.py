'''
from baselines/ppo1/mlp_policy.py and add simple modification
(1) add reuse argument
(2) cache the `stochastic` placeholder
'''
import tensorflow as tf
import gym

import baselines.common.tf_util as U
from baselines.common.mpi_running_mean_std import RunningMeanStd
from baselines.common.distributions import make_pdtype
from baselines.acktr.utils import dense


class MlpPolicy(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Box)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.shape))

        with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space.shape)

        obz = tf.clip_by_value((ob - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vffc%i" % (i+1), weight_init=U.normc_initializer(1.0), weight_loss_dict={}))
        self.vpred = dense(last_out, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        last_out = obz
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "polfc%i" % (i+1), weight_init=U.normc_initializer(1.0), weight_loss_dict={}))

        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.zeros_initializer())
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob[None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []


class MlpPolicy4Dict(object):
    recurrent = False

    def __init__(self, name, reuse=False, *args, **kwargs):
        with tf.variable_scope(name):
            if reuse:
                tf.get_variable_scope().reuse_variables()
            self._init(*args, **kwargs)
            self.scope = tf.get_variable_scope().name

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Dict)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob_config = U.get_placeholder(name="ob", dtype=tf.float32, shape=[sequence_length] + list(ob_space.spaces['joint'].shape))
        ob_target = U.get_placeholder(name="goal", dtype=tf.float32, shape=[sequence_length] + list(ob_space.spaces['target'].shape))
        obs_pos1 = U.get_placeholder(name="obs_pos", dtype=tf.float32,
                                    shape=[sequence_length] + list(ob_space.spaces['obstacle_pos1'].shape))
        obs_pos2 = U.get_placeholder(name="obs_pos2", dtype=tf.float32,
                                    shape=[sequence_length] + list(ob_space.spaces['obstacle_pos2'].shape))
        # construct v function model
        '''with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space['joint'].shape)

        obz = tf.clip_by_value((ob_config - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        goal_last_out = tf.clip_by_value((ob_target - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)'''
        last_out = ob_config
        goal_last_out = ob_target
        '''op_last_out = dense(obs_pos1, hid_size, "obs_pos_pre", weight_init=U.normc_initializer(1.0),
                            weight_loss_dict={})
        oo_last_out = dense(obs_pos2, hid_size, "obs_ori_pre", weight_init=U.normc_initializer(1.0),
                            weight_loss_dict={})
        #op_last_out = tf.layers.batch_normalization(op_last_out, training=True, name="obs_pos_bn")
        #oo_last_out = tf.layers.batch_normalization(oo_last_out, training=True, name="obs_ori_bn")
        op_last_out = tf.nn.tanh(op_last_out)
        oo_last_out = tf.nn.tanh(oo_last_out)'''
        obs_last_out = tf.concat([obs_pos1, obs_pos2], axis=-1)
        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "vfcfc%i" % (i+1), weight_init=U.normc_initializer(1.0),
                                              weight_loss_dict={}))
            goal_last_out = tf.nn.tanh(dense(goal_last_out, hid_size, "vfgfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                                   weight_loss_dict={}))
            obs_last_out = tf.nn.tanh(dense(obs_last_out, hid_size, "vfobsfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                                  weight_loss_dict={}))
        vpred = tf.concat([last_out, goal_last_out, obs_last_out], -1)
        self.vpred = dense(vpred, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        # construct policy probability distribution model
        last_out = ob_config
        goal_last_out = ob_target
        obs_last_out = tf.concat([obs_pos1, obs_pos2], axis=-1)

        for i in range(num_hid_layers):
            last_out = tf.nn.tanh(dense(last_out, hid_size, "pol_cfg_fc%i" % (i+1), weight_init=U.normc_initializer(1.0),
                                              weight_loss_dict={}))
            goal_last_out = tf.nn.tanh(
                dense(goal_last_out, hid_size, "pol_g_fc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                      weight_loss_dict={}))
            obs_last_out = tf.nn.tanh(
                dense(obs_last_out, hid_size, "pol_obs_fc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                      weight_loss_dict={}))
        last_out = tf.concat([last_out, goal_last_out, obs_last_out], -1)
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0]//2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0]//2], initializer=tf.constant_initializer(-3))
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob_config, ob_target, obs_pos1, obs_pos2], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob['joint'][None], ob['target'][None],
                                ob['obstacle_pos1'][None], ob['obstacle_pos2'][None])
        return ac1[0], vpred1[0]

    def get_variables(self):
        return tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, self.scope)

    def get_trainable_variables(self):
        return tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, self.scope)

    def get_initial_state(self):
        return []

class MLPD(MlpPolicy):
    def _init__(self, name, reuse=False, *args, **kwargs):
        super(MLPD, self).__init__(name, reuse, *args, **kwargs)

    def _init(self, ob_space, ac_space, hid_size, num_hid_layers, gaussian_fixed_var=True):
        assert isinstance(ob_space, gym.spaces.Dict)

        self.pdtype = pdtype = make_pdtype(ac_space)
        sequence_length = None

        ob_config = U.get_placeholder(name="ob", dtype=tf.float32,
                                      shape=[sequence_length] + list(ob_space.spaces['joint'].shape))
        ob_target = U.get_placeholder(name="goal", dtype=tf.float32,
                                      shape=[sequence_length] + list(ob_space.spaces['target'].shape))
        obs_pos = U.get_placeholder(name="obs_pos", dtype=tf.float32,
                                    shape=[sequence_length] + list(ob_space.spaces['obstacle_pos1'].shape))
        #is_training = U.get_placeholder(name="bn_training", dtype=tf.bool, shape=())
        # construct v function model
        '''with tf.variable_scope("obfilter"):
            self.ob_rms = RunningMeanStd(shape=ob_space['joint'].shape)

        obz = tf.clip_by_value((ob_config - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)
        last_out = obz
        goal_last_out = tf.clip_by_value((ob_target - self.ob_rms.mean) / self.ob_rms.std, -5.0, 5.0)'''
        last_out = ob_config
        goal_last_out = ob_target
        obs_last_out = obs_pos
        for i in range(num_hid_layers):
            last_out = dense(last_out, hid_size, "vfcfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                             weight_loss_dict={})
            #last_out = tf.layers.batch_normalization(last_out, training=is_training, name="vfcbn%i"%(i+1))
            last_out = tf.nn.tanh(last_out)
            goal_last_out = dense(goal_last_out, hid_size, "vfgfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                  weight_loss_dict={})
            #goal_last_out = tf.layers.batch_normalization(goal_last_out, training=is_training, name="vfgbn%i" % (i + 1))
            goal_last_out = tf.nn.tanh(goal_last_out)
            obs_last_out = dense(obs_last_out, hid_size, "vfobsfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                 weight_loss_dict={})
            #obs_last_out = tf.layers.batch_normalization(obs_last_out, training=is_training, name="vfobn%i"%(i+1))
            obs_last_out = tf.nn.tanh(obs_last_out)
        vpred = tf.concat([last_out, goal_last_out, obs_last_out], -1)
        self.vpred = dense(vpred, 1, "vffinal", weight_init=U.normc_initializer(1.0))[:, 0]

        # construct policy probability distribution model
        last_out = ob_config
        goal_last_out = ob_target
        obs_last_out = obs_pos

        for i in range(num_hid_layers):
            last_out = dense(last_out, hid_size, "pol_cfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                             weight_loss_dict={})
            #last_out = tf.layers.batch_normalization(last_out, training=is_training, name="pol_cbn%i"%(i+1))
            last_out = tf.nn.tanh(last_out)
            goal_last_out = dense(goal_last_out, hid_size, "pol_gfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                  weight_loss_dict={})
            #goal_last_out = tf.layers.batch_normalization(goal_last_out, training=is_training, name="pol_gbn%i" % (i + 1))
            goal_last_out = tf.nn.tanh(goal_last_out)
            obs_last_out = dense(obs_last_out, hid_size, "pol_obsfc%i" % (i + 1), weight_init=U.normc_initializer(1.0),
                                 weight_loss_dict={})
            #obs_last_out = tf.layers.batch_normalization(obs_last_out, training=is_training, name="pol_obn%i"%(i+1))
            obs_last_out = tf.nn.tanh(obs_last_out)
        last_out = tf.concat([last_out, goal_last_out, obs_last_out], -1)
        if gaussian_fixed_var and isinstance(ac_space, gym.spaces.Box):
            mean = dense(last_out, pdtype.param_shape()[0] // 2, "polfinal", U.normc_initializer(0.01))
            logstd = tf.get_variable(name="logstd", shape=[1, pdtype.param_shape()[0] // 2],
                                     initializer=tf.constant_initializer([0.2, 0.2, -1., -1.]))
            pdparam = tf.concat([mean, mean * 0.0 + logstd], axis=1)
        else:
            pdparam = dense(last_out, pdtype.param_shape()[0], "polfinal", U.normc_initializer(0.01))

        self.pd = pdtype.pdfromflat(pdparam)

        self.state_in = []
        self.state_out = []

        # change for BC
        stochastic = U.get_placeholder(name="stochastic", dtype=tf.bool, shape=())
        ac = U.switch(stochastic, self.pd.sample(), self.pd.mode())
        self.ac = ac
        self._act = U.function([stochastic, ob_config, ob_target, obs_pos], [ac, self.vpred])

    def act(self, stochastic, ob):
        ac1, vpred1 = self._act(stochastic, ob['joint'][None], ob['target'][None], ob['obstacle_pos1'][None])
        return ac1[0], vpred1[0]