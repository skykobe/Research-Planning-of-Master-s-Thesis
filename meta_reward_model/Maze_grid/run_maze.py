from RL_brain import DeepQNetwork
from maze_env import Maze
import tensorflow as tf
from tqdm import tqdm
import numpy as np
from knn_reward_model import KNN_Predict

MEMORY_SIZE = 20000
epsilon = 200
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
knn = KNN_Predict(3)
times = []
real = []

def onehot(action):
    x = [0,0,0,0]
    x[action] = 1
    return x

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
            action = RL.choose_action(np.array(ob))
            ob_, r, done = env.step(action)
            # print(ob, ob_)
            state_record.append(onehot(action))
            # state_record = check_append(state_record, ob_)
            r_r = r
            if model:
                if(knn.memory_size <= knn.count):
                    if(t < knn.minus_stepnum - 1 and  t >= 8): # 8 for best in 200 episodes , 10 for 300
                        po_s = 0.9*RL.qtarget_value(np.array(ob_)) - RL.qeval_value(np.array(ob))
                        r += 0.8*knn.predict(t, state_record) + 0.3*po_s
            # if i_epsilon > 3:
            # po_s = 0.9*RL.qtarget_value(np.array(ob_)) - RL.qeval_value(np.array(ob))
            # r += po_s
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
                state_record.append(r_r*100/t)
                knn.memory_in(state_record)
                break

            ob_ = ob
            t += 1
            step += 1


train(RL_natural, model=True)

print(np.min(real))
# print(np.sum(real[:500]), np.sum(real[-500:]))
print(np.sum(real))

env.destroy()
import matplotlib.pyplot as plt
#
# np.save('V2_pbrs', times)
# np.save('maze_dqn_PBRS_real', real)


plt.plot(range(1, len(times) + 1), times)
plt.show()
