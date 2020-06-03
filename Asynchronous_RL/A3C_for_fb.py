import gym
from gym.wrappers import Monitor
import gym_ple
import numpy as np
import tensorflow as tf
import time
import multiprocessing
import threading
import shutil
from unit import *
import matplotlib.pyplot as plt

N_WORKERS = multiprocessing.cpu_count()
#game screen = [288, 512]
GAME = 'FlappyBird-v0'
LR_A = 1e-3
LR_C = 1e-3
GAMA = 0.99
ENTROPY_BETA = 0.01
MAX_GLOBAL_TIME = 3
GLOBAL_TIME = 0
GLOBAL_NET_SCOPE = 'Global_Net'
OPT_A = tf.train.RMSPropOptimizer(LR_A, decay=0.99, epsilon=0.1, name='RMSFORA')
OPT_C = tf.train.RMSPropOptimizer(LR_C, decay=0.99, epsilon=0.1, name='RMSFORC')

env = gym.make(GAME)
N_S_w = 80
N_S_h = 80
N_A = env.action_space.n
GLOBAL_R = []

class ACNet(object):
    def __init__(self, scope, global_AC=None):
        if scope == GLOBAL_NET_SCOPE:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S_w, N_S_h, 4], name='S')
                self.a_params, self.c_params = self.build_model(scope)[:2]
        else:
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S_w, N_S_h, 4], name='S')
                self.action = tf.placeholder(tf.int32, [None, ], name='A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], name='v_target')

                self.a_params, self.c_params, self.a_prob, self.v = self.build_model(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_ERROR')
                with tf.name_scope('actor_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.action, 2, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5), axis=1, keep_dims=True)
                    self.exp_v = ENTROPY_BETA*entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('critic_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('local_gradient'):
                    self.a_grad = tf.gradients(self.a_loss, self.a_params)
                    self.c_grad = tf.gradients(self.c_loss, self.c_params)
            with tf.name_scope('syn'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, global_AC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, global_AC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grad, self.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grad, self.c_params))

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.01)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.01, shape = shape)
        return tf.Variable(initial)

    def build_model(self, scope):
        w_init = tf.random_normal_initializer(0., .1)

        with tf.variable_scope('ACNETWORK'):

            with tf.variable_scope('SHARE_CNN'):
                w_cov1 = self.weight_variable([8,8,4,16])
                b_cov1 = self.bias_variable([16])

                w_cov2 = self.weight_variable([4,4,16,32])
                b_cov2 = self.bias_variable([32])

                w_cov3 = self.weight_variable([2,2,32,64])
                b_cov3 = self.bias_variable([64])

                hy_cov1 = tf.nn.conv2d(self.s, w_cov1, strides=[1,4,4,1], padding='SAME') # [1,20,20,16]
                hy_cov1 = tf.nn.relu(hy_cov1)
                pool1 = tf.nn.max_pool(hy_cov1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # [1,10,10,16]

                hy_cov2 = tf.nn.conv2d(pool1, w_cov2, strides=[1,2,2,1], padding='SAME') # [1,5,5,32]
                hy_cov2 = tf.nn.relu(hy_cov2)
                # pool2 = tf.nn.max_pool(hy_cov2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # [1,8,8,32]

                hy_cov3 = tf.nn.conv2d(hy_cov2, w_cov3, strides=[1,2,2,1], padding='SAME') # [1,4,4,64]
                hy_cov3 = tf.nn.relu(hy_cov3)

                w_init = tf.truncated_normal_initializer(stddev=0.01)
                hy_cov3_falt = tf.reshape(hy_cov3, [-1, 3*3*64])

            with tf.variable_scope('actor'):
                afc1 = tf.layers.dense(hy_cov3_falt, 256, tf.nn.relu6, kernel_initializer=w_init, name='fc1')
                # afc2 = tf.layers.dense(afc1, 256, tf.nn.relu6, kernel_initializer=w_init, name='fc2')
                a_prob = tf.layers.dense(afc1, N_A, tf.nn.softmax, kernel_initializer=w_init, name='a_out')
            with tf.variable_scope('critic'):
                cfc1 = tf.layers.dense(hy_cov3_falt, 256, tf.nn.relu6, kernel_initializer=w_init, name='fc1')
                # cfc2 = tf.layers.dense(cfc1, 256, tf.nn.relu6, kernel_initializer=w_init, name='fc2')
                v = tf.layers.dense(cfc1, 1, kernel_initializer=w_init, name='v')

        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/ACNETWORK/SHARE_CNN') \
                + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/ACNETWORK/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/ACNETWORK/SHARE_CNN') \
                + tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/ACNETWORK/critic')
        return a_params, c_params, a_prob, v

    def update_global(self, feed_dict):
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)

    def pull_global(self):
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):
        a_prob = SESS.run(self.a_prob, {self.s: s})
        action = np.random.choice(range(a_prob.shape[1]), p=a_prob.ravel())
        return action

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.AC = ACNet(name, globalAC)
        self.name = name
    def work(self):
        global REC
        buffer_s, buffer_a, buffer_r, buffer_v_target = [], [], [], []
        self.env.seed(0)
        s = self.env.reset()
        ep_r = 0
        s_tmp = []
        while True:
            # self.env.render()
            s = preprocess(s)

            s_tmp.append(s)
            if len(s_tmp) == 1:
                _s = np.concatenate((s_tmp[0],s_tmp[0], s_tmp[0], s_tmp[0]), axis=3)
                action = self.AC.choose_action(_s)
            if len(s_tmp) == 4:
                s_tmp = []

            s_, r, done, info = self.env.step(action)
            if done != True:
                ep_r += r
            if r > 0:
                r = +5
            if r < 0:
                r = -2
            # if r == 0:
            #     r = 0.0025
            buffer_s.append(s)
            buffer_a.append(action)
            buffer_r.append(r)

            if done or ep_r > 999:
                buffer_s = state_transform(buffer_s)
                buffer_a = action_transform(buffer_a)
                buffer_r = reward_transform(buffer_r)
                # if self.name == 'agent_0':
                #     # if len(GLOBAL_R) == 0 or REC == True:
                #     #     GLOBAL_R.append(ep_r)
                #     #     REC = False
                #     GLOBAL_R.append(ep_r)
                #     print('main agent get the total reward:', ep_r)
                ALL_p.append(ep_r)
                if done:
                    v_s_ = 0
                else:
                    v_s_ = SESS.run(self.AC.v, {self.AC.s: preprocess(s_)})
                for r in buffer_r[::-1]:
                    v_s_ = r + GAMA*v_s_
                    buffer_v_target.append(v_s_)
                buffer_v_target.reverse()
                buffer_s, buffer_a, buffer_v_target = np.vstack(buffer_s), np.array(buffer_a), np.vstack(buffer_v_target)
                feed_dict = {
                    self.AC.s: buffer_s,
                    self.AC.action: buffer_a,
                    self.AC.v_target: buffer_v_target,
                }
                self.AC.update_global(feed_dict)
                self.AC.pull_global()
                break
            s = s_


SESS = tf.Session()
COORD = tf.train.Coordinator()

GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
workers = []
for i in range(N_WORKERS):
    name = 'agent_%i' % i
    workers.append(Worker(name, GLOBAL_AC))

test = Worker('test', GLOBAL_AC)

SESS.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=0)
check = tf.train.get_checkpoint_state('./')
if check:
    print('load existed model')
    load_model(saver, SESS, check)
else:
    print('create new model')

ttmp = []
long = 2 # 1 long = 1h
opt = time_count(long)
REC = False
ALL_p = [] # save the point in all agent each episode
P_REC = [] # save all the point
c = 3.0
while opt != 3:
    if opt == 2:
        save_model(saver, SESS)
        np.save('A3C_normal_%.1fH' % c, ttmp) #保存一个小时内的数据
        ttmp = []
        c += 1.0
        REC = True
    worker_threads = []
    for worker in workers:
        job = lambda: worker.work()
        t = threading.Thread(target=job)
        t.start()
        worker_threads.append(t)
    COORD.join(worker_threads)
    max_p = max(ALL_p)
    GLOBAL_R.append(max_p)
    ttmp.append(max_p)
    print('this round get point', max_p)
    # P_REC.append(ALL_p)
    ALL_p = []
    opt = time_count(long)


plt.plot(range(len(GLOBAL_R)), GLOBAL_R)
plt.show()
