import multiprocessing
import threading
import tensorflow as tf
import numpy as np
import gym
import os
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm

# np.random.seed(1)
# tf.set_random_seed(1)

GAME = 'MountainCar-v0'
OUTPUT_GRAPH = False
LOG_DIR = './log'
N_WORKERS = multiprocessing.cpu_count()
MAX_STEP = 2000
MAX_GLOBAL_EP = 400
GLOBAL_NET_SCOPE = 'Global_Net'
# UPDATE_GLOBAL_ITER = 10
GAMMA = 0.99
ENTROPY_BETA = 0.1
LR_A = 0.01    # learning rate for actor
LR_C = 0.01    # learning rate for critic
GLOBAL_RUNNING_R = []
GLOBAL_EP = 0
GLOBAL_ES_fail = []
GLOBAL_ES_succ = []
GLOBAL_EXCHANGE = False
# GLOBAL_solve_EXCHANGE = False

env = gym.make(GAME)
N_S = env.observation_space.shape[0]
N_A = env.action_space.n


class ACNet(object):
    def __init__(self, scope, globalAC=None):

        if scope == GLOBAL_NET_SCOPE:   # get global network
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_params, self.c_params = self._build_net(scope)[-2:]
        else:   # local net, calculate losses
            with tf.variable_scope(scope):
                self.s = tf.placeholder(tf.float32, [None, N_S], 'S')
                self.a_his = tf.placeholder(tf.int32, [None, ], 'A')
                self.v_target = tf.placeholder(tf.float32, [None, 1], 'Vtarget')

                self.a_prob, self.v, self.a_params, self.c_params = self._build_net(scope)

                td = tf.subtract(self.v_target, self.v, name='TD_error')
                with tf.name_scope('c_loss'):
                    self.c_loss = tf.reduce_mean(tf.square(td))

                with tf.name_scope('a_loss'):
                    log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
                    exp_v = log_prob * tf.stop_gradient(td)
                    entropy = -tf.reduce_sum(self.a_prob * tf.log(self.a_prob + 1e-5),
                                             axis=1, keep_dims=True)  # encourage exploration
                    self.exp_v = ENTROPY_BETA * entropy + exp_v
                    self.a_loss = tf.reduce_mean(-self.exp_v)

                with tf.name_scope('local_grad'):
                    self.a_grads = tf.gradients(self.a_loss, self.a_params)
                    self.c_grads = tf.gradients(self.c_loss, self.c_params)

            with tf.name_scope('sync'):
                with tf.name_scope('pull'):
                    self.pull_a_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.a_params, globalAC.a_params)]
                    self.pull_c_params_op = [l_p.assign(g_p) for l_p, g_p in zip(self.c_params, globalAC.c_params)]
                with tf.name_scope('push'):
                    self.update_a_op = OPT_A.apply_gradients(zip(self.a_grads, globalAC.a_params))
                    self.update_c_op = OPT_C.apply_gradients(zip(self.c_grads, globalAC.c_params))

    def _build_net(self, scope):
        w_init = tf.random_normal_initializer(0., .1)
        with tf.variable_scope('actor'):
            l_a = tf.layers.dense(self.s, 200, tf.nn.relu6, kernel_initializer=w_init, name='la')
            a_prob = tf.layers.dense(l_a, N_A, tf.nn.softmax, kernel_initializer=w_init, name='ap')
        with tf.variable_scope('critic'):
            l_c = tf.layers.dense(self.s, 100, tf.nn.relu6, kernel_initializer=w_init, name='lc')
            v = tf.layers.dense(l_c, 1, kernel_initializer=w_init, name='v')  # state value
        a_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/actor')
        c_params = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope=scope + '/critic')
        return a_prob, v, a_params, c_params

    def update_global(self, feed_dict):  # run by a local
        SESS.run([self.update_a_op, self.update_c_op], feed_dict)  # local grads applies to global net

    def pull_global(self):  # run by a local
        SESS.run([self.pull_a_params_op, self.pull_c_params_op])

    def choose_action(self, s):  # run by a local
        prob_weights = SESS.run(self.a_prob, feed_dict={self.s: s[np.newaxis, :]})
        action = np.random.choice(range(prob_weights.shape[1]),
                                  p=prob_weights.ravel())  # select action w.r.t the actions prob
        return action


class Worker(object):
    def __init__(self, name, globalAC):
        self.env = gym.make(GAME).unwrapped
        self.name = name
        self.AC = ACNet(name, globalAC)

    def work(self, w=True):
        global GLOBAL_RUNNING_R, GLOBAL_EP
        total_step = 1
        self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_v_target = [], [], [], []
        # while not COORD.should_stop() and GLOBAL_EP < MAX_GLOBAL_EP:
        if not COORD.should_stop():
            s = self.env.reset()
            step = 0
            while True:
                global GLOBAL_ES_fail, GLOBAL_EXCHANGE
                a = self.AC.choose_action(s)
                s_, r, done, info = self.env.step(a)
                if done:
                    r = +100
                else:
                    r = 0
                step += 1
                self.buffer_s.append(s)
                self.buffer_a.append(a)
                self.buffer_r.append(r)
                if done:    #As without the fail experience, so we store the success experience
                    pre = 0
                    for i in range(15):
                        i = - (i + 1)
                        tmp = []
                        tmp.append(self.buffer_s[i])
                        tmp.append(self.buffer_a[i])
                        vs = self.buffer_r[i] + GAMMA*pre
                        tmp.append(vs)
                        GLOBAL_ES_succ.append(tmp)
                        # pre = self.buffer_r[i]
                        pre = vs
                    GLOBAL_EXCHANGE = True
                    print(self.name, 'agent is success to climb the hill with', step)

                if done or step > MAX_STEP - 1:   # update global and assign to local net
                    if w:
                        if done:
                            v_s_ = 0   # terminal
                        else:
                            v_s_ = SESS.run(self.AC.v, {self.AC.s: s_[np.newaxis, :]})[0, 0]
                        # buffer_v_target = []
                        for r in self.buffer_r[::-1]:    # reverse buffer r
                            v_s_ = r + GAMMA * v_s_
                            self.buffer_v_target.append(v_s_)
                        self.buffer_v_target.reverse()

                        # buffer_s, buffer_a, buffer_v_target = np.vstack(self.buffer_s), np.array(self.buffer_a), np.vstack(self.buffer_v_target)
                        # feed_dict = {
                        #     self.AC.s: buffer_s,
                        #     self.AC.a_his: buffer_a,
                        #     self.AC.v_target: buffer_v_target,
                        # }
                        #
                        # self.AC.update_global(feed_dict)
                        # self.AC.pull_global()
                        # print(self.name, 'is finish')
                        if self.name == 'W_0':
                            GLOBAL_RUNNING_R.append(step)
                            print('main agent:', step)
                    else:
                        print('main agent:', step)
                        GLOBAL_RUNNING_R.append(step)
                    # if not GLOBAL_EXCHANGE:
                    #     self.buffer_s, self.buffer_a, self.buffer_r, self.buffer_v_target = [], [], [], []
                    break
                s = s_

    def update(self):
        self.AC.pull_global()

    def study(self):
        global GLOBAL_ES_succ
        # bs, ba, btv = [], [], []
        if len(GLOBAL_ES_succ) > 0:
            for item in GLOBAL_ES_succ:
                self.buffer_s.append(item[0])
                self.buffer_a.append(item[1])
                self.buffer_v_target.append(item[2])
        buffer_s, buffer_a, buffer_v_target = np.vstack(self.buffer_s), np.array(self.buffer_a), np.vstack(self.buffer_v_target)
        feed_dict = {
            self.AC.s: buffer_s,
            self.AC.a_his: buffer_a,
            self.AC.v_target: buffer_v_target,
        }
        self.AC.update_global(feed_dict)
        self.AC.pull_global()

if __name__ == "__main__":
    SESS = tf.Session()
    OPT_A = tf.train.RMSPropOptimizer(LR_A, name='RMSPropA')
    OPT_C = tf.train.RMSPropOptimizer(LR_C, name='RMSPropC')
    GLOBAL_AC = ACNet(GLOBAL_NET_SCOPE)  # we only need its params
    workers = []
    # Create worker
    for i in range(N_WORKERS):
        i_name = 'W_%i' % i   # worker name
        workers.append(Worker(i_name, GLOBAL_AC))

    # m = Worker('main', GLOBAL_AC)
    COORD = tf.train.Coordinator()
    SESS.run(tf.global_variables_initializer())

    for x in tqdm(range(MAX_GLOBAL_EP)):
        worker_threads = []
        for worker in workers:               # running stage
            job = lambda: worker.work()
            t = threading.Thread(target=job)
            t.start()
            worker_threads.append(t)
        COORD.join(worker_threads)

        #   discussion stage
        #   first: search the solved experience among all the agents
        study_threads = []
        for worker in workers:
            job = lambda: worker.study()
            t = threading.Thread(target=job)
            t.start()
            study_threads.append(t)
        COORD.join(study_threads)
        # GLOBAL_EP += 1
        # m.update()
        # m.work(w=False)
        GLOBAL_ES_succ = []
        # GLOBAL_EXCHANGE = False
        print('finish', x, 'round')

    np.save('EE_mc_400_step15', GLOBAL_RUNNING_R)
    plt.plot(np.arange(len(GLOBAL_RUNNING_R)), GLOBAL_RUNNING_R)
    plt.xlabel('episode')
    plt.ylabel('Total steps')
    plt.show()
