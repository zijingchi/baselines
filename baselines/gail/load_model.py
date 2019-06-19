import tensorflow as tf
import os

path = 'trpo_gail.with_pretrained.transition_limitation_-1.HalfCheetah.g_step_3.d_step_2.policy_entcoeff_0.001.adversary_entcoeff_0.001.seed_0'
#ckpt = tf.train.get_checkpoint_state('./'+path)
#saver=tf.train.import_meta_graph(ckpt.model_checkpoint_path+'.meta')
os.chdir(os.path.join("checkpoint", path))
saver = tf.train.import_meta_graph(os.path.join(os.getcwd(), path+".meta"))
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./'))
    print(sess.run('pi/obs_pos_pre/w:0'))