import gym
from RL_brain import DQNPrioritizedReplay
from RL_brain_doubledqn import DoubleDQN
from RL_brain_dueling import DuelingDQN
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
from knn_reward_model import KNN_Predict
from tqdm import tqdm

env = gym.make('MountainCar-v0')
env = env.unwrapped
env.seed(21)
MEMORY_SIZE = 50000

sess = tf.Session()

with tf.variable_scope('natural_DQN'):
    RL_natural = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=False,
    )
with tf.variable_scope('DQN_with_prioritized_replay'):
    RL_prio = DQNPrioritizedReplay(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, prioritized=True,
    )
with tf.variable_scope('double_DQN'):
    RL_doubleq = DoubleDQN(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess,
    )
with tf.variable_scope('dueling_DQN'):
    RL_dueling = DuelingDQN(
        n_actions=3, n_features=2, memory_size=MEMORY_SIZE,
        e_greedy_increment=0.00005, sess=sess, dueling=True, output_graph=False
    )

knn = KNN_Predict(8) #best for 3
sess.run(tf.global_variables_initializer())
x = []

def train(RL, model=False):
    total_steps = 0
    steps = []
    episodes = []
    for i_episode in tqdm(range(200)):
        observation = env.reset()
        t = 0
        sc = 0
        record = []
        while True:
            # env.render()

            action = RL.choose_action(observation, knn, t)
            record.append(action)
            observation_, reward, done, info = env.step(action)
            # print(RL.qtarget_value(observation_))
            # position, velocity = observation_
            # reward = abs(position - (-0.5))
            sc += reward
            # my research
            if model:
                if(knn.memory_size <= knn.count):
                    if(t < knn.minus_stepnum and t >=5):
                        # print('reward model')
                        po_s = 0.9*RL.qtarget_value(observation_) - RL.qeval_value(observation)
                        reward += -knn.predict(t, record) - po_s
            # PBRS
            # if(i_episode > 40):
            # reward += (0.9*RL.qtarget_value(observation_) - RL.qeval_value(observation))

            if done: reward = 5

            RL.store_transition(observation, action, reward, observation_)

            if total_steps > MEMORY_SIZE:
                # print('learn')
                RL.learn()

            if done:
                print('episode ', i_episode, ' finished')
                steps.append(total_steps)
                episodes.append(i_episode)
                record.append(sc)
                knn.memory_in(record)
                x.append(t)
                break

            observation = observation_
            total_steps += 1
            t += 1
    return np.vstack((episodes, steps))

# test_model = train(RL_natural2)
# his_natural = train(RL_natural)
# his_prio = train(RL_prio, model=True)
# his_natural = train(RL_natural, model=True)
test_model = train(RL_natural, model=True)
print(np.min(x))
# test_model = train(RL_dueling)

# np.save('normal_dqn_500', test_model[1, :] - test_model[1, 0])

# x = [     0,  29943,  40541,  43858,  46815,  49193,  49832,  54098,  61024,  63578,
#   66517,  69013,  72690,  75689,  77382,  84137,  88604,  90779,  96911,  98781,
#  101096, 106680, 108060, 109607, 111223, 112416, 113377, 114570, 115411, 116028]
# y = [    0, 29943, 40541, 47802, 58559, 60212, 61759, 63622, 64255, 65208, 65873, 66844,
#  67468, 68460, 69313, 69910, 70404, 71402, 73044, 73561, 75278, 75792, 76309, 76863,
#  77667, 78530, 79026, 79428, 80283, 80664]

# plt.plot(his_natural[0, :], his_natural[1, :] - his_natural[1, 0], c='b', label='natural DQN')
# plt.plot(his_prio[0, :], his_prio[1, :] - his_prio[1, 0], c='y', label='DQN with prioritized replay')
plt.plot(test_model[0, :], test_model[1, :] - test_model[1, 0])
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
# plt.grid()
plt.show()
