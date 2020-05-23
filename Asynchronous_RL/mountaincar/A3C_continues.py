import gym
import multiprocessing
import threading
import numpy as np
import os
import shutil
import matplotlib.pyplot as plt
import tensorflow as tf
from tqdm import tqdm

no_of_workers = multiprocessing.cpu_count()
max_ep_steps = 1000
no_of_episodes = 100
global_net_scope = 'Global_Net'
gamma = 0.90
entropy_beta = 0.01
lr_a = 0.0001
lr_c = 0.001
render=False
log_dir = 'logs'

env = gym.make('MountainCarContinuous-v0')
env.reset()
no_of_states = env.observation_space.shape[0]
no_of_actions = env.action_space.shape[0]
action_bound = [env.action_space.low, env.action_space.high]
class ActorCritic(object):
     def __init__(self, scope, sess, globalAC=None):
        self.sess=sess
        self.actor_optimizer = tf.train.RMSPropOptimizer(lr_a, name='RMSPropA')
        self.critic_optimizer = tf.train.RMSPropOptimizer(lr_c, name='RMSPropC')
        if scope == global_net_scope:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, no_of_states], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, no_of_states], 'S')
                self.a_his = tf.placeholder(tf.float32, [None, no_of_actions], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')
                mean, var, self.v, self.a_params, self.c_params = self._build_net(scope)
                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('critic_loss'):
                    self.critic_loss = tf.reduce_mean(tf.square(td))
                with tf.name_scope('wrap_action'):
                    mean, var = mean * action_bound[1], var + 1e-4
                normal_dist = tf.contrib.distributions.Normal(mean, var)
                with tf.name_scope('actor_loss'):
                    log_prob = normal_dist.log_prob(self.a_his)
                    exp_v = log_prob * td
                    entropy = normal_dist.entropy()
                    self.exp_v = exp_v + entropy_beta * entropy
                    self.actor_loss = tf.reduce_mean(-self.exp_v)
                with tf.name_scope('choose_action'):
                    self.A = tf.clip_by_value(tf.squeeze(normal_dist.sample(1), axis=0), action_bound[0], action_bound[1])
                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.actor_loss, self.a_params)
                    self.c_grads = tf.gradients(self.critic_loss, self.c_params)
            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = self.actor_optimizer.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = self.critic_optimizer.apply_gradients(zip(self.c_grads, globalAC.c_params))
     def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            mean = tf.layers.dense(l_a, no_of_actions, tf.nn.tanh,kernel_initializer=w_init, name='mean')
            var = tf.layers.dense(l_a, no_of_actions, tf.nn.softplus, kernel_initializer=w_init, name='var')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return mean, var, v, a_params, c_params
     def update_global(self, feed_dict):
        self.sess.run([self.update_a_op, self.update_c_op], feed_dict)
     def pull_global(self):
        self.sess.run([self.pull_a_params_op, self.pull_c_params_op])
     def choose_action(self, s):
        s = s[np.newaxis, :]
        return self.sess.run(self.A, {self.s: s})[0]
class Worker(object):
    def __init__(self, name, globalAC, sess, prop=False): #prop=False for worker and True for boss(main agent)
        self.env = gym.make('MountainCarContinuous-v0').unwrapped
        self.name = name
        self.AC = ActorCritic(name, sess, globalAC)
        self.sess=sess
        self.prop = prop

    def update(self):
        self.AC.pull_global()

    def work(self):
        global global_rewards, global_episodes
        buffer_s, buffer_a, buffer_r = [], [], []
        s = self.env.reset()
        ep_r = 0
        step = 0
        while True:
            a = self.AC.choose_action(s)
            s_, r, done, info = self.env.step(a)
            # ep_r += r
            if done:
                r = 100
            else:
                r = 0
            buffer_s.append(s)
            buffer_a.append(a)
            buffer_r.append(r)
            if done:
                print(self.name, 'success to solve')
            if step > max_ep_steps - 1 or done:
                if done:
                    v_s_ = 0
                else:
                    v_s_ = self.sess.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                buffer_v_target = []
                for r in buffer_r[::-1]:
                    v_s_ = r + gamma * v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()
                buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.vstack(buffer_a), np.vstack(buffer_v_target)
                feed_dict = {
                             self.AC.s: buffer_s,
                             self.AC.a_his: buffer_a,
                             self.AC.v_target: buffer_v_target,
                             }
                if self.prop == False:
                    self.AC.update_global(feed_dict)
                    buffer_s, buffer_a, buffer_r = [], [], []
                    self.AC.pull_global()
            s = s_
            step += 1
            if done or step > max_ep_steps - 1:
                if self.prop:
                    print('main agent:', step)
                    global_rewards.append(step)
                break

global_rewards = []
global_episodes = 0
sess = tf.Session()
with tf.device("/cpu:0"):
    global_ac = ActorCritic(global_net_scope,sess)
    workers = []
    for i in range(no_of_workers):
        i_name = 'W_%i' % i
        workers.append(Worker(i_name, global_ac,sess))
boss = Worker('boss', global_ac,sess, True)
coord = tf.train.Coordinator()
sess.run(tf.global_variables_initializer())

for ep in tqdm(range(no_of_episodes)):
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    coord.join(worker_threads)
    boss.update()
    boss.work()
    print('finish a round')

plt.plot(range(len(global_rewards)), global_rewards)
plt.show()
