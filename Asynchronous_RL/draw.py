import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


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

def goodlook3(ary):
    half = int(len(ary) / 2)
    p1 = np.array(ary[half - 10 : half + 10]).max()
    p2 = np.array(ary[-20:]).max()
    return [p1, p2]

def goodlook4(ary):
    mean = []
    std = []
    inter = int(len(ary)/2)
    # print(len(ary))
    index = list(range(0, len(ary), inter))
    # print(index)
    # for i in range(2):
    #     v = ary[i:(i+inter)]
    #     mean.append(np.mean(v))
    #     std.append(np.std(v))
    mean.append(np.mean(ary[:inter]))
    std.append(np.mean(ary[:inter]))
    mean.append(np.mean(ary[-inter:]))
    std.append(np.mean(ary[-inter:]))
    print(mean)
    return mean, std


# d1 = np.load('EE_date_1.0H.npy')
d1 = np.load('EE_date_0.0H.npy')
d2 = np.load('EE_date_0.0-2.0H.npy')
d2 = d2[len(d1):]
d3 = np.load('EE_date_3.0H.npy')
d4 = np.load('EE_date_4.0H.npy')
d5 = np.load('EE_date_5.0H.npy')
d6 = np.load('EE_date_6.0H.npy')
d7 = np.load('EE_date_7.0H.npy')
d8 = np.load('EE_date_8.0H.npy')
# print(len(d2))
# l = len(d1)
# d2 = d2[l:]
m1, s1 = goodlook4(d1)
m2, s2 = goodlook4(d2)
m3, s3 = goodlook4(d3)
m4, s4 = goodlook4(d4)
m5, s5 = goodlook4(d5)
m6, s6 = goodlook4(d6)
m7, s7 = goodlook4(d7)
m8, s8 = goodlook4(d8)
m = np.concatenate((m1,m2,m3,m4,m5,m6,m7,m8))
s = np.concatenate((s1,s2,s3,s4,s5,s6,s7,s8))
mean = np.insert(m, 0, 0)
std = np.insert(s, 0, 0)
# m, s = goodlook4(d)
# goodlook4(d3)
# print(m)
plt.plot(np.arange(0, len(mean)/2, 0.5), mean)
plt.fill_between(np.arange(0, len(mean)/2, 0.5), mean - std, mean + std,alpha=0.05)
plt.show()
