from RL_brain import DeepQNetwork
from maze_env import Maze
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from knn_reward_model import KNN_Predict

MEMORY_SIZE = 20000
epsilon = 10000
sess = tf.Session()
np.random.seed(1)
tf.set_random_seed(1)

env = Maze()
with tf.variable_scope('natural_DQN'):
    RL_natural = DeepQNetwork(
        n_actions=env.n_actions, n_features=env.n_features, memory_size=MEMORY_SIZE,
        replace_target_iter=200
        # e_greedy_increment=0.0001
    )
knn = KNN_Predict(2)
times = []
real = []

def check_append(list, data):
    if data not in list:
        list.append(data)
    if data in list:
        index = list.index(data)
        list = list[:index+1]
    return list

def train(RL, model=False):
    step = 0
    first = 0
    r_r = 0
    for i_epsilon in tqdm(range(epsilon)):
        ob = env.reset()
        # print(ob)
        t = 0
        state_record = []
        # state_record = check_append(state_record, ob)
        while True:
            # if knn.suptrain:
            #     action = knn.get_sup_action(t)
            # else:
            action = RL.choose_action(np.array(ob))
            ob_, r, done = env.step(action)
            # print(ob, ob_)
            state_record.append(action)
            # state_record = check_append(state_record, ob_)
            if model:
                # tp = len(state_record) - 1
                if(knn.memory_size <= knn.count):
                    if(t >= 5 and t < knn.max_stepnum):
                        po_s = 0.9*RL.qtarget_value(np.array(ob_)) - RL.qeval_value(np.array(ob))
                        r += knn.predict_modified(t, state_record) + 0.3*po_s
            # if i_epsilon > 3:
            po_s = 0.9*RL.qtarget_value(np.array(ob_)) - RL.qeval_value(np.array(ob))
            # r += po_s
            # if done:
            #     r = 1
            r_r = r
            if knn.suptrain:
                r = 1*0.99**t
                RL.store_transition_maml(ob, action, r, ob_)
            else:
                RL.store_transition(ob, action, r, ob_)
            if(step > RL.memory_size):
                RL.learn(knn.suptrain)
            if done:
                if(first != 0):
                    time = int(0.9*time + 0.1*t)
                else:
                    first = 1
                    time = t
                times.append(time)
                real.append(t)
                print('success in steps:', i_epsilon, 'with steps:', time)
                state_record.append(r_r*100/t)
                knn.memory_in(state_record)
                break

            ob_ = ob
            t += 1
            step += 1


train(RL_natural)

print(np.min(real))
# print(np.sum(real[:500]), np.sum(real[-500:]))
print(np.sum(real))

env.destroy()
import matplotlib.pyplot as plt
#
# np.save('maze_dqn_PBRS', times)
# np.save('PBRS_500_times', times)


plt.plot(range(1, len(times) + 1), times)
plt.show()
