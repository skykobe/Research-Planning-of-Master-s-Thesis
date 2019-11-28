from RL_brain import DeepQNetwork
from maze_env import Maze
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from knn_reward_model import KNN_Predict

MEMORY_SIZE = 10000
epsilon = 400
sess = tf.Session()
np.random.seed(1)
tf.set_random_seed(1)

env = Maze()
with tf.variable_scope('natural_DQN'):
    RL_natural = DeepQNetwork(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        # e_greedy_increment=0.0001
    )
knn = KNN_Predict(3)
times = []
real = []

def train(RL, model=False):
    step = 0
    first = 0
    r_r = 0
    for i_epsilon in tqdm(range(epsilon)):
        ob = env.reset()
        # print(ob)
        t = 0
        step_record = []
        while True:
            action = RL.choose_action(np.array(ob))
            step_record.append(action)
            ob_, r, done = env.step(action)
            # print(ob, ob_)
            r_r = r
            if model:
                if(knn.memory_size <= knn.count):
                    if(t < knn.minus_stepnum and t >= 5):
                        r += knn.predict(t, step_record)
            RL.store_transition(ob, action, r, ob_)
            if(step > RL.memory_size):
                RL.learn()
            if done:
                if(first != 0):
                    time = int(0.9*time + 0.1*t)
                else:
                    first = 1
                    time = t
                times.append(time)
                real.append(t)
                print('success in steps:', i_epsilon, 'with steps:', time)
                step_record.append(r_r*1000/t)
                knn.memory_in(step_record)
                break

            ob_ = ob
            t += 1
            step += 1


train(RL_natural)


env.destroy()
import matplotlib.pyplot as plt

np.save('maze_dqn_rewardmodel_run400_size=3', times)

plt.plot(range(1, len(times) + 1), times)
plt.show()
