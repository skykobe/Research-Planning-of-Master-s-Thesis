import matplotlib.pyplot as plt
import numpy as np

dqn = np.load('maze_dqn.npy')
dqn_model = np.load('maze_dqn_x.npy')
dqn_pbrs = np.load('maze_dqn_PBRS.npy')

dqnreal = np.load('maze_dqn_real.npy')
dqnmodel = np.load('maze_dqn_x_real.npy')
print(sum(dqnreal))
print(sum(dqnmodel))

# print(sum(dqn), sum(dqn_model))
plt.plot(range(1, len(dqn)+1), dqn, label='nature dqn')
plt.plot(range(1, len(dqn_model)+1), dqn_model, label='dqn with reward model')
plt.plot(range(1, len(dqn_pbrs)+1), dqn_pbrs, label='dqn with PBRS')
plt.legend(loc='best')
plt.ylabel('episode solved steps')
plt.xlabel('episode')
plt.show()
