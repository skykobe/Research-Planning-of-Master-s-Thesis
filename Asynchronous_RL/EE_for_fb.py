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
from KNN import *
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

Fail_EXP = []
F_size = 2000
Succ_EXP = []
GLOBAL_EXCHANGE = False

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

            hy_cov1 = tf.nn.conv2d(self.s, w_cov1, strides=[1,4,4,1], padding='SAME') # [1,32,32,16]
            hy_cov1 = tf.nn.relu(hy_cov1)
            pool1 = tf.nn.max_pool(hy_cov1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME') # [1,16,16,16]

            hy_cov2 = tf.nn.conv2d(pool1, w_cov2, strides=[1,2,2,1], padding='SAME') # [1,8,8,32]
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

    def get_c_v(self, s):
        v = SESS.run(self.v, {self.s: s})
        return v

class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME)
        self.AC = ACNet(name, globalAC)
        self.name = name

    def work(self):
        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_v_target = [], [], [], []
        self.env.seed(0)
        s = self.env.reset()
        s = preprocess(s)
        t = 0
        ep_r = 0
        s_tmp = []
        step_record = []
        s_pre = None
        while True:
            # self.env.render()
            _s = np.concatenate((s,s,s,s), axis=3)
            s_pre = _s
            action = self.AC.choose_action(_s)
            step_record.append(onehot(action))
            for _ in range(4):
                s_, r, done, info = self.env.step(action)
                self.buffer_s.append(s)
                self.buffer_a.append(action)
                if r > 0:
                    r = +5
                if r < 0:
                    r = -2
                self.buffer_r.append(r)
                s = preprocess(s_)
                if done:
                    break
                ep_r += r
            if KNN.count >= KNN.memory_size:
                if t >= 4 and t < KNN.minus_stepnum - 1:
                    # print('knn predict')
                    predict = KNN.predict_modified(t, step_record)
                    s_in = np.concatenate((s,s,s,s), axis=3)
                    pbrs = GAMA*self.AC.get_c_v(s_in) - self.AC.get_c_v(s_pre)
                    # if pbrs > np.exp(0.5):
                    #     pbrs = np.exp(0.5)
                    # if pbrs < (-np.exp(0.5)):
                    #     pbrs = (-np.exp(0.5))
                    self.buffer_r[-1] += 0.2*predict + 0.1*pbrs

            if done or ep_r > 999:
                self.buffer_s = state_transform(self.buffer_s)
                self.buffer_a = action_transform(self.buffer_a)
                self.buffer_r = reward_transform(self.buffer_r)

                step_record.append(ep_r)
                KNN.memory_in(step_record)

                if done:
                    re_step = 2
                    pre = 0
                    for i in range(re_step):
                        i = - (i + 1)
                        tmp = []
                        tmp.append(self.buffer_s[i])
                        tmp.append(self.buffer_a[i])
                        tmp.append(self.buffer_r[i] + 0.99*pre)
                        Fail_EXP.append(tmp)
                        pre = self.buffer_r[i] + 0.99*pre
                    GLOBAL_EXCHANGE = True
                ALL_p.append(ep_r)
                if done:
                    v_s_ = 0
                else:
                    v_s_ = SESS.run(self.AC.v, {self.AC.s: preprocess(s_)})
                for r in self.buffer_r[::-1]:
                    v_s_ = r + GAMA*v_s_
                    self.buffer_v_target.append(v_s_)
                self.buffer_v_target.reverse()
                self.exchange_succ()
                break
            t += 1

    def solve_way(self):
        global Fail_EXP
        for f in Fail_EXP:
            index = get_state_index(f[0], self.buffer_s)
            if index != -1:
                if self.buffer_r[index] >= 0 and self.buffer_a[index] != f[1]:
                    tmp = []
                    tmp.append(f[0])
                    tmp.append(self.buffer_a[index])
                    tmp.append(self.buffer_v_target)
                    Succ_EXP.append(tmp)

    def exchange_succ(self):
        global Succ_EXP
        re_step = 8
        index = 0
        for r in self.buffer_r:
            if r > 0:
                for i in range(re_step):
                    # print('share succ exp')
                    tmp = []
                    tmp.append(self.buffer_s[index - i])
                    tmp.append(self.buffer_a[index - i])
                    tmp.append(self.buffer_v_target[index - i])
                    Succ_EXP.append(tmp)
            index += 1

    def study(self):
        global Succ_EXP, Fail_EXP
        EE = Succ_EXP + Fail_EXP
        for item in EE:
            self.buffer_s.append(item[0])
            self.buffer_a.append(item[1])
            self.buffer_v_target.append(item[2])
        buffer_s, buffer_a, buffer_v_target = np.vstack(self.buffer_s), np.array(self.buffer_a), np.vstack(self.buffer_v_target)
        feed_dict = {
            self.AC.s: buffer_s,
            self.AC.action: buffer_a,
            self.AC.v_target: buffer_v_target,
        }
        self.AC.update_global(feed_dict)
        self.AC.pull_global()

SESS = tf.Session()
COORD = tf.train.Coordinator()

ALL_p = []
GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)
workers = []
for i in range(N_WORKERS):
    name = 'agent_%i' % i
    workers.append(Worker(name, GLOBAL_AC))

SESS.run(tf.global_variables_initializer())
saver = tf.train.Saver(max_to_keep=0)
check = tf.train.get_checkpoint_state('./')
if check:
    print('load existed model')
    load_model(saver, SESS, check)
else:
    print('create new model')

ttmp = []
long = 4 # 1 long = 1h
opt = time_count(long)
REC = False
c = 1.0
KNN = KNN_Predict(10)
KNN.load_memory()
while opt != 3:
    if opt == 2:
        save_model(saver, SESS)
        np.save('EE_date_%.1fH' % c, ttmp)
        KNN.save_knn_memory()
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

    EE_threads = []
    for worker in workers:
        job = lambda: worker.solve_way()
        t = threading.Thread(target=job)
        t.start()
        EE_threads.append(t)
    COORD.join(EE_threads)
    # for worker in workers:
    #     worker.solve_way()

    study_threads = []
    for worker in workers:
        job = lambda: worker.study()
        t = threading.Thread(target=job)
        t.start()
        study_threads.append(t)
    COORD.join(study_threads)

    GLOBAL_R.append(max(ALL_p)/5)
    ttmp.append(max(ALL_p)/5)
    print('the', len(GLOBAL_R) , 'episode get point', max(ALL_p)/5)
    ALL_p = []
    Succ_EXP = []
    # if len(Fail_EXP) > F_size:
    Fail_EXP = []
    opt = time_count(long)

# np.save('test_date_1H', GLOBAL_R)
plt.plot(range(len(GLOBAL_R)), GLOBAL_R)
plt.show()
