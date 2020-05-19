import numpy as np
import matplotlib.pyplot as plt
import os,sys
os.chdir(os.path.dirname(sys.argv[0]))

A3C_normal = np.load('normal_A3C_200.npy')
normal = []
A3C_EE_2 = np.load('A3C_experience_exchange_step=2.npy')
A3C_EE_3 = np.load('A3C_experience_exchange_step3.npy')
A3C_EE_4 = np.load('A3C_experience_exchange_step=4.npy')
A3C_EE_5 = np.load('A3C_EE_step5_2.npy')
A3C_EE_4_new = np.load('A3C_EE_step4.npy')

ES_5 = np.load('ES_200_2.npy')

def goodlook(original):
    renew = []
    pre = original[0]
    for i in original:
        if i == pre:
            renew.append(i)
        else:
            renew.append(0.1*i + 0.9*renew[-1])
    return renew

normal = goodlook(A3C_normal)
plt.plot(range(1, len(A3C_normal)+1), A3C_normal, alpha=.1, c='blue')
plt.plot(range(1, len(A3C_normal)+1), normal, c='blue', label='A3C')
EE_5 = goodlook(A3C_EE_5)
plt.plot(range(1, len(A3C_EE_5)+1), A3C_EE_5, alpha=.1, c='red')
plt.plot(range(1, len(A3C_EE_5)+1), EE_5, c='red', label='A3C with Full Experience Exchange')
ES5 = goodlook(ES_5)
plt.plot(range(1, len(ES_5)+1), ES_5, alpha=.2, c='green')
plt.plot(range(1, len(ES5)+1), ES5, c='green', label='A3C with ES')
# EE_3 = goodlook(A3C_EE_3)
# plt.plot(range(1, len(A3C_EE_3)+1), A3C_EE_3, alpha=.2, c='purple')
# plt.plot(range(1, len(A3C_EE_3)+1), EE_3, c='purple', label='A3C with full experience exchange steo 5')
plt.legend(loc='best', fontsize=13)
plt.ylabel('Reward', fontsize=20)
plt.xlabel('Episode', fontsize=20)
plt.show()
