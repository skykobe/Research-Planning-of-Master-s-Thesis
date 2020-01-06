import matplotlib.pyplot as plt
import numpy as np

AC = np.load('v2_natural_AC.npy')
AC_PBRS = np.load('v2_PBRS_AC.npy')
AC_KNN = np.load('v2_metarewardmodel_AC.npy')
# DQN = np.load('dqn_cartpole.npy')

ac = []
ac_pbrs = []
ac_knn = []
dqn = []

step = 50
mark = np.ones(10)

for i in range(0, 1000, step):
    ac.append(sum(AC[i:i+step])/step)
    ac_pbrs.append(sum(AC_PBRS[i:i+step])/step)
    ac_knn.append(sum(AC_KNN[i:i+step])/step)
    # dqn.append(sum(DQN[i:i+step])/step)

plt.plot(range(1, len(ac)+1, 1), ac, label='A2C')
plt.plot(range(1, len(ac_pbrs)+1, 1), ac_pbrs, label='A2C with PBRS')
plt.plot(range(1, len(ac_knn)+1, 1), ac_knn, c='r', label='A2C with meta reward model')
# plt.plot(range(1, len(dqn)+1, 1), dqn, label='dqn')
# plt.plot(range(len(mark)), 195*mark, linewidth=2, linestyle=":")
plt.legend(loc='best')
plt.ylabel('episode reward')
plt.xlabel('*100 episode')
# plt.grid()
plt.show()
