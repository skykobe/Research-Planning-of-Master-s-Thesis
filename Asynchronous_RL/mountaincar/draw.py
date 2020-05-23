import matplotlib.pyplot as plt
import numpy as np

def goodlook(original):
    renew = []
    pre = original[0]
    for i in original:
        if i == pre:
            renew.append(i)
        else:
            renew.append(0.1*i + 0.9*renew[-1])
    return renew

def goodlook2(original):
    total = 0
    renew = []
    for i in original:
        if total == 0:
            renew.append(i)
            total = i
        else:
            total += i
            renew.append(total)
    return renew

A3C_400 = goodlook2(np.load('A3C_400_mc2.npy'))
EE_400 = goodlook2(np.load('EE_mc_400_step15.npy'))

plt.plot(range(1, len(A3C_400)+1), A3C_400, label='A3C with ES')
plt.plot(range(1, len(EE_400)+1), EE_400, c='red', label='A3C with EE')
plt.legend(loc='best')
plt.xlabel('Episode')
plt.ylabel('total training time')
plt.show()
