import numpy as np
import tensorflow as tf
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm
# %matplotlib inline

np.random.seed(2)
tf.set_random_seed(2)

OUTPUT_GRAPH = False
MAX_EPISODE = 100
DISPLAY_REWARD_THRESHOLD = 200  # renders environment if total episode reward is greater then this threshold
MAX_EP_STEPS = 250   # maximum time step in one episode
RENDER = False  # rendering wastes time
GAMMA = 0.9     # reward discount in TD error
# LR_A = 0.001    # learning rate for actor
# LR_C = 0.01     # learning rate for critic


env = gym.make('MountainCar-v0')
# env.seed(2)  # reproducible
env = env.unwrapped

N_F = env.observation_space.shape[0]
N_A = env.action_space.n

def get_state(s):
    state = np.zeros(64)
    state[s] = 1
    # state = state.reshape(8,8)
    return state

class meta_td:
    def __init__(self, value):
        self.x = 1
        self.meta_paramter = 1/value
        self.error_set = 0
        self.count = 0
        self.cache = 100

    def learning(self, error):
        er = abs(error) * self.meta_paramter
        self.x = (1 - self.meta_paramter) * self.x + er

        lr = (2 / (1 + np.exp(-self.x/10))) - 1
        gamma = (2 / (1 + np.exp(self.x)))
        epsilon = np.exp(self.x) - 1

        return round(lr, 4), round(gamma, 3), round(epsilon, 3)

    def learning2(self, error):
        if(self.count >= self.cache):
            td_avg = self.error_set / self.cache
            er = td_avg * self.meta_paramter
            self.x = (1 - self.meta_paramter) * self.x + er

            lr = (2 / (1 + np.exp(-self.x/10))) - 1
            gamma = (2 / (1 + np.exp(self.x)))
            epsilon = np.exp(self.x) - 1

            self.count = 0
            self.error_set = 0

            return lr, gamma, epsilon, True

        else:
            self.error_set += abs(error)
            self.count += 1
            return 0, 0, 0, False

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
                self.minus_stepnum = len(self.memory[1]) - 1
                for index in self.memory:
                    if(len(index) < self.minus_stepnum):
                        self.minus_stepnum = len(index)
        else:
            self.memory.append(data)
            self.count += 1
            if(len(data) < self.minus_stepnum):
                self.minus_stepnum = len(data)

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
        r_step = (memory_reward - min_r) / (max_r - min_r)
        # print(dis_step)
        dis_step = dis_step.reshape(self.memory_size, 1)

        res = np.dot(r_step, dis_step)
        res = round(res[0], 4)
        return 10*(np.exp(res) - np.exp(0.5))

    def predict_modified(self, step_nums, data):
        memory = np.array(self.memory)
        memory_step = []
        memory_reward = []
        for x in memory:
            memory_reward.append(x[-1])
            memory_step.append(x[:step_nums+1])
        memory_step = np.array(memory_step) - np.array(data)
        distance = np.array(memory_step)**2
#         distance = abs(memory_step)
        distance = distance.sum(1)

        close_reward = memory_reward[np.argmin(distance)]
        min_reward = np.min(memory_reward)
        max_reward = np.max(memory_reward)
        value = (close_reward - min_reward) / (max_reward - min_reward + 1e-4)

        if(value >=1):
            value = 1
        if(min_reward == max_reward):
            value = 1
#         print(value)
        return round(np.exp(value)-np.exp(0.5), 3)

class Actor(object):
    def __init__(self, sess, n_features, n_actions, lr=0.0001):
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
                units=10,    # number of hidden units
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
            log_prob = tf.log(self.acts_prob[0, self.a] + 1e-10)
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

    def choose_action(self, s):
        s = s[np.newaxis, :]
        # res, probs = self.sess.run([self.res, self.acts_prob], {self.s: s, self.epsilon: self.epsilon_v})
        probs = self.sess.run(self.acts_prob, {self.s: s})
        # res = self.boltzmann(probs[0], self.epsilon_v)
        # return np.random.choice(np.arange(probs.shape[1]), p=probs.ravel())   # return a int
        return np.random.choice(np.arange(probs.shape[1]), p=probs[0])   # return a int


class Critic(object):
    def __init__(self, sess, n_features, lr=0.001):
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
                units=10,  # number of hidden units
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
metar = meta_td(value=200)
knn = KNN_Predict(10)
record = []
step_record = []
sess.run(tf.global_variables_initializer())

# if OUTPUT_GRAPH:
#     tf.summary.FileWriter("logs/", sess.graph)

reward = []
running_reward_label = 0
cost = []
success = 0

def trian():
    global running_reward_label, success, step_record
    for i_episode in tqdm(range(MAX_EPISODE)):
        env.seed(1)
        s = env.reset()
        t = 0
        tmp = 0
        ep_r = 0
        while True:
            a = actor.choose_action(s)
            s_, r, done, info = env.step(a)
#             ep_r += r
#             step_record.append(a)
            td_error = critic.learn(s, r, s_)
            actor.learn(s, a, td_error)
            s = s_
            t += 1
            if done:
                reward.append(t)
                print('epi:', i_episode, 'steps:', t)
                break




trian()

# ax1 = plt.subplot(1,2,1)
# ax2 = plt.subplot(1,2,2)

# plt.sca(ax1)
# plt.plot(range(1, len(cost) + 1), cost)
# plt.sca(ax2)
plt.plot(range(1, len(reward) + 1), reward)

plt.show()
