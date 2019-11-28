import matplotlib.pyplot as plt
import numpy as np

dqn = np.load('normal_dqn_500.npy')
# dqn_rewardmodel = np.load('data_dqn_rewardmodel_3.npy')
dqn_rewardmodel_2 = np.load('meta_reward_model_s=8.npy')
# dqn_pri = np.load('data_pri_dqn.npy')
# dqn_pbsr = np.load('data_dqn_PBRS2.npy')
# print(dqn[-1] - dqn[-2])
# # print(dqn_rewardmodel[-1] - dqn_rewardmodel[-2])
# print(dqn_rewardmodel_2[-1] - dqn_rewardmodel_2[-2])
# print(dqn_pri[-1] - dqn_pri[-2])
# print(dqn_rewardmodel)
# dqn_rewardmodel10 = np.load('data_dqn_rewardmodel_10.npy')
plt.plot(range(len(dqn)), dqn, label='normal DQN')
# plt.plot(range(len(dqn_rewardmodel)), dqn_rewardmodel, c='green', label='DQN with reward model')
plt.plot(range(len(dqn_rewardmodel_2)), dqn_rewardmodel_2, c='r', label='DQN with reward model')
# plt.plot(range(len(dqn_pri)), dqn_pri, c='orange', label='DQN with prioritized replay')
# plt.plot(range(len(dqn_pbsr)), dqn_pbsr, c='black', label='DQN with PBSR')
plt.legend(loc='best')
plt.ylabel('total training time')
plt.xlabel('episode')
# plt.grid()
plt.show()
