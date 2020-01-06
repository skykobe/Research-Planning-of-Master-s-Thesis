import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm

np.random.seed(2)
tf.set_random_seed(2)

OUTPUT_GRAPH = False
MAX_EPISODE = 1000
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 200   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
# LR_A = 0.001    # learning rate for actor
# LR_C = 0.01     # learning rate for critic


env = gym.make('CartPole-v1')
env.seed(2)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

import numpy as np

def onehot(action):
    if(action == 0):
        return [1,0]
    if(action == 1):
        return [0,1]


class KNN_Predict:
    def __init__(self, size):
        self.memory_size = size
        self.memory = []
        self.count = 0
        self.minus_stepnum = 999

    def memory_in(self, data):
        if(self.count >= self.memory_size):
            # print('out of size')
            # memory满了以后，进行数据交换，交换原创是？ 1，根据分数的高到低？ 2，类似与缓存的机制？关注局所性 3， LRU ?
            # 先尝试最简单的, 保存最高的数据
            num = len(data)
            reward_for_memory = []
            memory = np.array(self.memory)
            # print('all data', memory)
            # reward_for_memory = memory[:, -1]
            for x in memory:
                reward_for_memory.append(x[-1])
            index = np.argmin(reward_for_memory)
            data_r = data[-1]
            if(reward_for_memory[index] <= data_r):
                # print(data_r, reward_for_memory[index])
                self.memory[index] = data
                self.minus_stepnum = len(self.memory[1])-1
                for index in self.memory:
                    if(len(index) - 1 < self.minus_stepnum):
                        self.minus_stepnum = len(index) - 1
        else:
            self.memory.append(data)
            self.count += 1
            if(len(data) - 1 < self.minus_stepnum):
                self.minus_stepnum = len(data) - 1

        # print('knn memory size', len(self.memory))

    def predict(self, step_nums, data):
        # if(self.count >= self.memory_size):
        memory = np.array(self.memory)
        # memory_step = memory[:, :step_nums] - data + 0.1
        memory_step = []
        memory_reward = []
        # print(step_nums, x[:step_nums])
        for x in memory:
            # print(x[-1], x[:step_nums - 1])
            memory_reward.append(x[-1])
            memory_step.append(x[:step_nums+1])
        # print(memory_step, memory_reward, data[-1])
        memory_step = np.array(memory_step) - np.array(data) + 0.001
        #先求Pn
        # print(memory_step)
        distance = np.array(memory_step)**2
        distance = distance.sum(1)
        distance = distance / distance.sum()
        dis_step = 1 / distance
        dis_step = dis_step / dis_step.sum()

        #再求Xn
        max_r = np.max(memory_reward)
        min_r = np.min(memory_reward)
        r_step = (memory_reward - min_r + 1e-8) / (max_r - min_r + 1e-8)
        # print(dis_step)
        dis_step = dis_step.reshape(self.memory_size, 1)

        res = np.dot(r_step, dis_step)
        res = round(res[0], 4)
        return np.exp(res) - np.exp(0.5)

    def predict_modified(self, step_nums, data):
        memory = self.memory
        memory_step = []
        memory_reward = []
        for x in memory:
            memory_reward.append(x[-1])
            memory_step.append(np.array(x[:step_nums+1]) - np.array(data) + 0.001)
        # print(memory_step)
        # memory_step = np.array(memory_step) - np.array(data) + 0.001
        distance = np.array(memory_step)**2
        distance = distance.sum(2)
        distance = distance.sum(1)
        distance = distance / distance.sum()
        dis_step = 1 / distance
        dis_step = dis_step / dis_step.sum()

        min_reward = np.min(memory_reward)
        max_reward = np.max(memory_reward)
        close_reward = []
        close_record = []

        # close_reward.append(memory_reward[np.argmax(memory_reward)])
        # close_record.append(memory_step[np.argmax(memory_reward)])
        for i in range(20):
            index = np.argmax(dis_step)
            close_reward.append(memory_reward[index])
            close_record.append(dis_step[index])
            dis_step[index] = -1

        r_step = (close_reward - min_reward + 1e-8) / (max_reward - min_reward + 1e-8)
        close_record = np.array(close_record)
        close_record = close_record.reshape(20, 1)

        res = np.dot(r_step, close_record)
        res = round(res[0], 4)
        return np.exp(res) - np.exp(0.5)

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.001):
        self.sess = sess

        self.g = tf.constant(1e-8, dtype=tf.float32)
        self.epsilon = tf.placeholder(tf.float32, None, "e_greedy")
        self.epsilon_v = 0.3
        self.lr = tf.placeholder(tf.float32, None, "lr")
        self.lr_v = lr
        self.probility = tf.placeholder(tf.float32, [1, n_actions], "res")
        # self.metar = meta_td(value=300)

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.a = tf.placeholder(tf.int32, None, "act")
        self.td_error = tf.placeholder(tf.float32, None, "td_error")  # TD_error

        with tf.variable_scope('Actor'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,    # number of hidden units
                activation=tf.nn.relu,
                kernel_initializer=tf.random_normal_initializer(0., .1),    # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.acts_prob = tf.layers.dense(
                inputs=l1,
                units=n_actions,    # output units
                activation=tf.nn.softmax,   # get action probabilities
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='acts_prob'
            )

        with tf.variable_scope('epsilon_action'):
            prob = self.acts_prob[0] / self.epsilon
            self.res = tf.nn.softmax(prob)

        with tf.variable_scope('exp_v'):
            log_prob = tf.log(self.acts_prob[0, self.a] + 1e-8)
            # log_prob = tf.add(log_prob, self.g)
            # log_prob = tf.log(self.acts_prob[0, self.a] + 1e-5)
            # exp = log_prob * self.td_error
            # entropy = -tf.reduce_sum(self.acts_prob * tf.log(self.acts_prob + 1e-5), keep_dims=True)
            # log_prob = tf.reduce_sum(tf.log(self.a_prob + 1e-5) * tf.one_hot(self.a_his, N_A, dtype=tf.float32), axis=1, keep_dims=True)
            self.exp_v = tf.reduce_mean(log_prob * self.td_error)  # advantage (TD_error) guided loss

        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(-self.exp_v)  # minimize(-exp_v) = maximize(exp_v)

    def boltzmann(self, x, temperature):
        if(temperature < 0.00001):
            temperature = 0.00001
        exponent = np.true_divide(x - np.max(x), temperature)
        return np.exp(exponent)/np.sum(np.exp(exponent))

    def learn(self, s, a, td):
        s = s[np.newaxis, :]
        feed_dict = {self.s: s, self.a: a, self.td_error: td, self.lr: self.lr_v}
        _, exp_v = self.sess.run([self.train_op, self.exp_v], feed_dict)

        return exp_v

    # def cal_probs(self, s):
    #     prob = self.acts_prob[0] / self.epsilon
    #     a_e = tf.exp(prob)
    #     a_t = tf.reduce_sum(a_e)
    #     res_probs = self.sess.run(a_e / a_tm , {self.s: s, self.epsilon: self.epsilon_v})
    #     return

    def choose_action(self, s):
        s = s[np.newaxis, :]
        # res, probs = self.sess.run([self.res, self.acts_prob], {self.s: s, self.epsilon: self.epsilon_v})
        probs = self.sess.run(self.acts_prob, {self.s: s})
        # res = self.boltzmann(probs[0], self.epsilon_v)
        # return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
        return np.random.choice(np.arange(probs.shape[1]), p=probs[0])   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.01):
        self.sess = sess

        self.s = tf.placeholder(tf.float32, [1, n_features], "state")
        self.v_ = tf.placeholder(tf.float32, [1, 1], "v_next")
        self.r = tf.placeholder(tf.float32, None, 'r')

        self.gama = tf.placeholder(tf.float32, None, 'gama')
        self.gama_v = 0.9

        self.lr = tf.placeholder(tf.float32, None, "lr")
        self.lr_v = lr

        with tf.variable_scope('Critic'):
            l1 = tf.layers.dense(
                inputs=self.s,
                units=20,  # number of hidden units
                activation=tf.nn.relu,  # None
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='l1'
            )

            self.v = tf.layers.dense(
                inputs=l1,
                units=1,  # output units
                activation=None,
                kernel_initializer=tf.random_normal_initializer(0., .1),  # weights
                bias_initializer=tf.constant_initializer(0.1),  # biases
                name='V'
            )

        with tf.variable_scope('squared_TD_error'):
            self.td_error = self.r + self.gama * self.v_ - self.v
            self.loss = tf.square(self.td_error)    # TD_error = (r+gamma*V_next) - V_eval
        with tf.variable_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)

    def learn(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]

        v_ = self.sess.run(self.v, {self.s: s_})
        td_error, _ = self.sess.run([self.td_error, self.train_op],
                                          {self.s: s, self.v_: v_, self.r: r, self.gama: self.gama_v, self.lr: self.lr_v})


        return td_error

    def value(self, s):
        s = s[np.newaxis, :]
        v = self.sess.run(self.v, {self.s: s})
        return v

    def errorfortd(self, s, r, s_):
        s, s_ = s[np.newaxis, :], s_[np.newaxis, :]
        v_ = self.sess.run(self.v, {self.s: s_})
        td_error = self.sess.run(self.td_error,
                                          {self.s: s, self.v_: v_, self.r: r, self.gama: self.gama_v})

        return td_error

sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))

actor = Actor(sess, n_features=N_F, n_actions=N_A)
critic = Critic(sess, n_features=N_F)     # we need a good teacher, so the teacher should learn faster than the actor
# metar = meta_td(value=200)
knn = KNN_Predict(30)
record = []
step_record = []
sess.run(tf.global_variables_initializer())

if OUTPUT_GRAPH:
    tf.summary.FileWriter("logs/", sess.graph)

reward = []
running_reward_label = 0
cost = []
success = False

def trian():
    global running_reward_label, success, step_record
    repeat = 0.8
    for i_episode in range(MAX_EPISODE):
        env.seed(2)
        s = env.reset()
        t = 0
        track_r = []
        tmp = 0
        while True:
            if i_episode > 990:
                env.render()
            a = actor.choose_action(s)
            # print(a, onehot(a))
            s_, r, done, info = env.step(a)
            track_r.append(r)
            # x, x_dot, theta, theta_dot = s_
            # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
            # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
            # r = r1 + r2
            step_record.append(onehot(a))
            # KNN_BASED_REALTIME_REWARD_SHAPING
            # if(knn.memory_size <= knn.count):
            #     if(t < knn.minus_stepnum - 1 and t >= 8):
            #         po_s = 0.9*critic.value(s_) - critic.value(s)
            #         predict = knn.predict_modified(t, step_record)
            #         r += 0.8*(predict + po_s)

            # PBRS
            # if(i_episode > 5):
            # r += 0.9*critic.value(s_) - critic.value(s)
            if t >= MAX_EP_STEPS: r = 1
            if done: r = -1
            # r += 0.9*critic.value(s_) - critic.value(s)
            # meta
            error = critic.errorfortd(s, r, s_)
            tde = float(error)
            # lr, gama, epsilon = metar.learning(error=tde)
#             actor.epsilon_v = epsilon
#             critic.gama_v = gama
#             critic.lr_v = lr
            # actor.lr_v = abs(np.exp(r/10)-1)

#             print('state', s, 's value', critic.value(s), 's_ value', critic.value(s_), 'td_error', tde)
            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)

            if done != True:
                cost.append(abs(tde)*0.95+tmp*0.05)
                tmp = abs(tde)
#             print('after learning', 's value', critic.value(s), 's_ value', critic.value(s_))
            s = s_
            t += 1
            if done or t >= MAX_EP_STEPS:
                ep_rs_sum = sum(track_r)

                step_record.append(ep_rs_sum)
                knn.memory_in(step_record)
                step_record = []
                # if 'running_reward' not in globals():
                #     running_reward = ep_rs_sum
                if running_reward_label == 0:
                    running_reward = ep_rs_sum
                    running_reward_label = 1
                else:
                    running_reward = running_reward * 0.95 + ep_rs_sum * 0.05
                if running_reward > DISPLAY_REWARD_THRESHOLD: RENDER = True  # rendering
                print("episode:", i_episode, "  reward:", int(running_reward), 'paramter', actor.lr_v)
                reward.append(running_reward)
                record.append(ep_rs_sum)
                # repeat = repeat / 2
#                 critic.lr_v = 0.1
                break
#             print('------------------')

        if(len(record) >= 100):
            task_avg = sum(record[-100:]) / 100
            # print(task_avg)
            if(task_avg >= 195):
                print('game task finish in :', i_episode, task_avg)
                # success = True
                actor.lr_v = 1e-5
                # break

    running_reward_label = 1

trian()

# ax1 = plt.subplot(1,2,1)
# ax2 = plt.subplot(1,2,2)

# plt.sca(ax1)
# plt.plot(range(1, len(record) + 1), record)
# plt.sca(ax2)
np.save('v2_PBRS_AC', reward)

plt.plot(range(1, len(reward) + 1), reward)

plt.show()
