
import os
import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt

from tensor_utils import *
import params

class model():
    def __init__(self, state_shape, n_actions, recover=False):

        tf.set_random_seed(params.seed)
        np.random.seed(params.seed)
        self.n_actions = n_actions
        
        with tf.name_scope('input'):
            self.state_in = tf.placeholder(tf.float32, 
                    (None,) + state_shape)
            self.action_in = tf.placeholder(tf.float32, [None, n_actions])
            self.reward_in = tf.placeholder(tf.float32, [None])
            self.nextstate_in = tf.placeholder(tf.float32, 
                    (None,) + state_shape)
            self.keep_prob = tf.placeholder(tf.float32)

            #this is just for summary stats
            self.episode_reward_in = tf.placeholder(tf.float32)
            episode_count = tf.Variable(0, trainable=False)
            self.episode_count = tf.assign_add(episode_count, 1)

        with tf.name_scope('model'):
            state = tf.contrib.layers.flatten(self.state_in)
            next_state = tf.contrib.layers.flatten(self.nextstate_in)
            
            #shared weights
            n = 512
            w1 = weight([state.get_shape()[-1].value, n])
            b1 = bias([n])
            
            w2 = weight([n, n_actions])
            b2 = bias([n_actions])


            #computes q values for input
                #if max_a, returns max q value
            def q_val(s, a, max_a=False):
                #x = conv_layer(x, (3,3), 16, 'elu')
                dense = tf.nn.elu(tf.matmul(s, w1) + b1)
                logits = tf.matmul(dense, w2) + b2
                if max_a:
                    q = tf.reduce_max(logits, axis=[1])
                else:
                    q = tf.reduce_sum(logits * a, axis=[1])
                return q, logits
        
            #calc q values
            q_curr, logits = q_val(state, self.action_in)
            q_next, _ = q_val(next_state, self.action_in, max_a=True)

            #target value
            y = self.reward_in + params.reward_decay * q_next

        with tf.name_scope('policy'):
            #epsilon greedy choice is in self.act()
            self.action = tf.argmax(logits, axis=1)

        with tf.name_scope('loss'):
            #squared difference error
            self.loss = 0.5 * tf.reduce_sum(tf.square(tf.subtract(
                    q_curr, y)))

        with tf.name_scope('optimize'):
            self.optimize, self.step = minimize(self.loss, 
                    params.learn_rate)

        with tf.name_scope('summary'):
            batch_size = tf.shape(self.state_in)[0]
            tf.summary.scalar('1_total_loss', self.loss)
            self.summaries = tf.summary.merge_all()
            self.episode_rewards = tf.summary.scalar('5_episode_reward', 
                    self.episode_reward_in)

        self.sess = tf.Session()
        self.writer = tf.summary.FileWriter(os.path.join(params.out_dir, 
                'model'), self.loss.graph)
        self.saver = tf.train.Saver()
        self.model_vars = tf.trainable_variables()
        self.sess.run(tf.global_variables_initializer())

        #load weights from disk if specified
        self.status = 'initialized'
        if recover:
            self.load()
            self.status = 'recovered'

    def add_episode_stat(self, reward):
        ep, summ = self.sess.run([self.episode_count, self.episode_rewards],
                feed_dict={self.episode_reward_in: np.float32(reward),})
        self.writer.add_summary(summ, ep)
        self.writer.flush()

    def act(self, state, explore=True):
        action = self.sess.run([self.action], 
                feed_dict={
                    self.state_in: [state],
                    self.keep_prob: 1.0,
                })[0]

        #epsilon greedy exploration
        if explore and np.random.random() < params.explore:
            action = np.random.choice(self.n_actions, 1)

        return action[0]

    def learn(self, batch, sample=False):
        states, actions, rewards, dones, next_states = batch
        loss, _, step, summ = self.sess.run([self.loss, 
                self.optimize, self.step, self.summaries],
                feed_dict={
                    self.state_in: states,
                    self.action_in: actions,
                    self.reward_in: rewards,
                    self.nextstate_in: next_states,
                    self.keep_prob: 0.5,
                })
        self.writer.add_summary(summ, step)
        self.writer.flush()
        return loss

    def save(self):
        self.saver.save(self.sess, os.path.join(params.out_dir, 
                'model', 'model.ckpt'), global_step=self.step)

    def load(self):
        #load sess, trained variables
        self.saver.restore(self.sess, tf.train.latest_checkpoint(
                os.path.join(params.out_dir, 'model')))
        recovered = tf.trainable_variables()

        #assign recovered variables over the initialized ones
        restore = [self.model_vars[i].assign(recovered[i]) for i in range(
                len(self.model_vars))]
        self.sess.run([restore])















