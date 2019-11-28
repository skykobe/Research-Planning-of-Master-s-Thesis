import gym
from DQN_modified import DeepQNetwork
import matplotlib.pyplot as plt
import numpy as np

env = gym.make('CartPole-v0')
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

RL = DeepQNetwork(n_actions=env.action_space.n,
                  n_features=env.observation_space.shape[0],
                  learning_rate=0.01, e_greedy=0.9,
                  replace_target_iter=100, memory_size=1000,
                  )

total_steps = 0
reward_c = []
show = []
running_reward = 0
for i_episode in range(1000):
    t = 0
    observation = env.reset()
    ep_r = 0
    while True:
        # env.render()

        action = RL.choose_action(observation)

        observation_, reward, done, info = env.step(action)

        # the smaller theta and closer to center the better
        # x, x_dot, theta, theta_dot = observation_
        # r1 = (env.x_threshold - abs(x))/env.x_threshold - 0.8
        # r2 = (env.theta_threshold_radians - abs(theta))/env.theta_threshold_radians - 0.5
        # reward = r1 + r2
        if done:
            reward = -1

        RL.store_transition(observation, action, reward, observation_)

        ep_r += reward
        if total_steps > 1000:
            RL.learn()

        if done or t >= 200:
            if(running_reward == 0):
                running_reward = ep_r
            else:
                running_reward = running_reward*0.95 + ep_r*0.05
            print('episode: ', i_episode,
                  'ep_r: ', int(running_reward),
                  ' epsilon: ', round(RL.epsilon, 2))
            reward_c.append(ep_r)
            show.append(running_reward)
            break

        observation = observation_
        total_steps += 1
        t += 1
    if(len(reward_c) >= 100):
        avg = sum(reward_c[-100:]) / 100
        if(avg >= 195):
            print('finish in :', i_episode, 'avg:', avg)
            break

# RL.plot_cost()
np.save('dqn_cartpole', show)
plt.plot(range(1, len(show) + 1), show)
plt.show()
