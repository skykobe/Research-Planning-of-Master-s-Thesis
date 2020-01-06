import matplotlib.pyplot as plt
import numpy as np

dqn = np.load('V2_normal.npy')
dqn_model = np.load('V2_metarewardmodel.npy')
dqn_pbrs = np.load('PBRS_500_times.npy')
dqn_pbrs = dqn_pbrs[:200]
step = 1
d = []
dm = []
pbrs = []
for i in range(0, 200, step):
    d.append(sum(dqn[i:i+step])/step)
    dm.append(sum(dqn_model[i:i+step])/step)
    pbrs.append(sum(dqn_pbrs[i:i+step])/step)
    # ac_knn.append(sum(AC_KNN[i:i+step])/step)
# dqnreal = np.load('maze_dqn_real.npy')
# dqnmodel = np.load('maze_dqn_x_real.npy')
# print(sum(dqnreal))
# print(sum(dqnmodel))

# print(sum(dqn), sum(dqn_model))
plt.plot(range(1, len(d)+1), d, label='nature dqn')
plt.plot(range(1, len(dm)+1), dm, c='r', label='dqn with reward model')
plt.plot(range(1, len(pbrs)+1), pbrs, label='dqn with PBRS')
plt.legend(loc='best')
plt.ylabel('episode solved steps')
plt.xlabel('episode')
plt.show()
