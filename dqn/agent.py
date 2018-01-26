import cv2
import joblib
import time
import random

import numpy as np
import tensorflow as tf
import tensorlayer as tl

from collections import deque
# from tensorflow.contrib.slim.python.slim.nets.inception_v3 import inception_v3, inception_v3_arg_scope
                                                                   
# slim = tf.contrib.slim


class Agent(object):
    def __init__(self, env):
        # get environment's information
        self.ob_shape = env.observation_space.shape
        self.ac_shape = env.action_space.shape
        # build hyper parameter
        self.act_step = 0
        self.train_step = 0
        self.epsilon = 1.
        self.gamma = 0.99
        self.image_size = 84
        self.ob_step = 1000
        self.batch_size = 64
        # build tensorflow session
        graph = tf.get_default_graph()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True 
        self.sess = tf.Session(graph=graph, config=config)
        # build tensorflow placeholder
        self._build_ph()
        # build network
        self._build_net()
        # build training
        self._build_training()
        # build memory buffer
        self.memory_buffer = deque()
        self.memory_size = 1000000
        # build tensorboard
        self.merge_all = tf.summary.merge_all()
        self.writer = tf.summary.FileWriter('./tb/{}'.format(
            time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))),
            self.sess.graph)

        # init tensorflow session
        self.sess.run(tf.global_variables_initializer())

    def _build_ph(self):
        self.ob_ph = tf.placeholder(tf.float32, [None, self.image_size, self.image_size, 1], 'ob_ph')
        self.ac_ph = tf.placeholder(tf.float32, [None, self.ac_shape[0]], 'ac_ph')
        self.ret_ph = tf.placeholder(tf.float32, [None, ], 'ret_ph')
        self.epsilon_tb = tf.placeholder(tf.float32, name='epsilon_tb')
        self.memory_size_tb = tf.placeholder(tf.int32, name='memory_size_tb')
        self.score_tb = tf.placeholder(tf.float32, name='score_tb')

    def _build_net(self):
        def build_q_net(name):
            net = tl.layers.InputLayer(self.ob_ph, name='{}_input'.format(name))
            net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=[3, 3, 1, 32], 
                strides=[1, 2, 2, 1], name='{}_cnn1'.format(name))
            net = tl.layers.Conv2dLayer(net, act=tf.nn.relu, shape=[3, 3, 32, 32], 
                strides=[1, 1, 1, 1], name='{}_cnn2'.format(name))
            net = tl.layers.FlattenLayer(net, name='{}_flat'.format(name))
            net = tl.layers.DenseLayer(net, n_units=32, act=tf.nn.relu, name='{}_h1'.format(name))
            net = tl.layers.DenseLayer(net, n_units=32, act=tf.nn.relu, name='{}_h2'.format(name))
            net = tl.layers.DenseLayer(net, n_units=self.ac_shape[0], name='{}_output'.format(name))
            return net

        self.net_eval = build_q_net('eval')
        self.net_target = build_q_net('target')

    def _build_training(self):
        q_action = self.net_eval.outputs
        # compute Q value
        q_eval = tf.reduce_sum(tf.multiply(q_action, self.ac_ph), axis=1)
        # compute td error
        with tf.variable_scope('mse'):
            td_error = tf.reduce_mean(tf.square(q_eval - self.ret_ph))
        # build a optimizer
        self.opt = tf.train.AdamOptimizer(3e-4).minimize(td_error)
        # build a operation to update target network
        vars_eval = self.net_eval.all_params
        vars_target = self.net_target.all_params
        update_target = []
        for var, var_target in zip(vars_eval, vars_target):
            update_target.append(var_target.assign(var))
        self.update_target = tf.group(*update_target)
        # build summary
        tf.summary.scalar('loss', td_error)
        tf.summary.scalar('mean_q', tf.reduce_mean(q_action))
        tf.summary.scalar('epsilon', self.epsilon_tb)
        tf.summary.scalar('memory_size', self.memory_size_tb)
        tf.summary.scalar('score', self.score_tb)

    def memory(self, ob, ac, next_ob, rew, done):
        self.act_step += 1
        one_hot_ac = np.zeros(self.ac_shape[0])
        one_hot_ac[ac] = 1
        next_ob = self.preproc(next_ob)
        self.memory_buffer.append((ob, one_hot_ac, next_ob, rew, done))
        if len(self.memory_buffer) > self.memory_size:
            self.memory_buffer.popleft()

        if len(self.memory_buffer) > self.ob_step and self.act_step % 4 == 0:
            self.train()

        return next_ob

    def train(self):
        self.train_step += 1
        # get minibatch from memory buffer
        mb = random.sample(self.memory_buffer, self.batch_size)
        mb_ob = np.asarray([data[0] for data in mb])
        mb_ac = np.asarray([data[1] for data in mb])
        mb_next_ob = np.asarray([data[2] for data in mb])
        mb_rew = [data[3] for data in mb]
        mb_done = [data[4] for data in mb]

        q_target = self.sess.run(self.net_target.outputs, 
            feed_dict={self.ob_ph: mb_next_ob})

        # compute return
        mb_ret = []
        for i in range(self.batch_size):
            if mb_done[i]:
                mb_ret.append(mb_rew[i])
            else:
                mb_ret.append(mb_rew[i] + self.gamma*np.max(q_target[i]))
        mb_ret = np.asarray(mb_ret)
        # opt q eval network and visualize it
        summary, _ = self.sess.run([self.merge_all, self.opt], feed_dict={
            self.ob_ph: mb_ob, self.ac_ph:mb_ac, self.ret_ph: mb_ret,
            self.score_tb: self.score, self.memory_size_tb: len(self.memory_buffer),
            self.epsilon_tb: self.epsilon})
        self.writer.add_summary(summary, self.train_step)
        # update q target network
        if self.train_step % 500 == 0:
            self.sess.run(self.update_target)

        if self.train_step % 10000 == 0:
            self.save_net('./tb/checkpoints/{}'.format(self.train_step))

    def act(self, ob, is_training):
        if is_training:
            # observation stage
            if self.act_step < self.ob_step:
                return random.randint(0, self.ac_shape[0] - 1)
            # epsilon-greedy exloration
            self.epsilon -= (1. - .1) / 100000
            if self.epsilon < .1:
                self.epsilon = .1
            if random.random() < self.epsilon:
                return random.randint(0, self.ac_shape[0] - 1)
            else:
                q_eval = self.sess.run(self.net_eval.outputs, 
                    feed_dict={self.ob_ph: [ob]})[0]
                return np.argmax(q_eval)
        else:
            q_eval = self.sess.run(self.net_eval.outputs, 
                feed_dict={self.ob_ph: [ob]})[0]
            return np.argmax(q_eval)

    # preprocess observation
    def preproc(self, ob):
        dst = ob[:430, 38:, :]
        dst = cv2.cvtColor(dst, cv2.COLOR_BGR2GRAY)
        dst = cv2.resize(dst, (self.image_size, self.image_size))
        dst = np.expand_dims(np.asarray(dst/255., dtype=np.float32), axis=2)
        return dst

    def save_net(self, save_path):
        params = self.net_eval.all_params
        ps = self.sess.run(params)
        joblib.dump(ps, save_path)

    def load_net(self, load_path):
        loaded_params = joblib.load(load_path)
        restores = []
        params = self.net_eval.all_params
        for p, loaded_p in zip(params, loaded_params):
            restores.append(p.assign(loaded_p))
        self.sess.run(restores)

    def get_score(self, score):
        self.score = score
