import tensorflow as tf
import numpy as np

class KNN_Predict:
    def __init__(self, size):
        self.memory_size = size
        self.memory = []
        self.count = 0
        self.minus_stepnum = 999
        self.K = 3
        self.suptrain = False  # 当出现新的最大值，就把agent的随机率加大
        self.subindex = [] # 记录被选中的数据条
        self.num_maml = 5

    def memory_in(self, data):
        if(self.count >= self.memory_size):
            # print('out of size')
            # memory满了以后，进行数据交换，交换原创是？ 1，根据分数的高到低？ 2，类似与缓存的机制？关注局所性 3， LRU ?
            # 先尝试最简单的, 保存最高的数据
            num = len(data)
            reward_for_memory = []
            memory = np.array(self.memory)
            # print('all data', memory)
            # reward_for_memory = memory[:, -1]
            for x in memory:
                reward_for_memory.append(x[-1])
            index = np.argmin(reward_for_memory)
            max_i = np.argmax(reward_for_memory)
            data_r = data[-1]
            if(reward_for_memory[index] <= data_r):
                # print(data_r, reward_for_memory[index])
                self.memory[index] = data
                self.minus_stepnum = len(self.memory[0]) - 1
                for index in self.memory:
                    if(len(index) - 1 < self.minus_stepnum):
                        self.minus_stepnum = len(index) - 1
            if(reward_for_memory[max_i] <= data_r):
                self.newmax = True
                self.subindex = index
        else:
            self.memory.append(data)
            self.count += 1
            if(len(data) - 1 < self.minus_stepnum):
                self.minus_stepnum = len(data) - 1

        # print('knn memory size', len(self.memory))
    def maml(self):
        memory = np.array(self.memory)
        for x in memory:
            reward_for_memory.append(x[-1])
        for i in range(self.num_maml):
            index = np.argmax(reward_for_memory)
            if index not in self.subindex:
                self.subindex.append(index)
            else:
                self.subindex.append(np.random.randint(0, self.memory_size))
            reward_for_memory[index] = np.min(reward_for_memory)
        self.suptrain = True

    def predict(self, step_nums, data):
        memory = self.memory
        memory_step = []
        memory_reward = []
        for x in memory:
            memory_reward.append(x[-1])
            memory_step.append(np.array(x[:step_nums+1]) - np.array(data) + 0.001)
        # print(memory_step)
        # memory_step = np.array(memory_step) - np.array(data) + 0.001
        distance = np.array(memory_step)**2
        distance = distance.sum(2)
        distance = distance.sum(1)
        distance = distance / distance.sum()
        dis_step = 1 / distance
        dis_step = dis_step / dis_step.sum()
        #再求Xn
        max_r = np.max(memory_reward)
        min_r = np.min(memory_reward)
        r_step = (memory_reward - min_r + 1e-8) / (max_r - min_r + 1e-8)
        # print(dis_step)
        dis_step = dis_step.reshape(self.memory_size, 1)

        res = np.dot(r_step, dis_step)
        res = round(res[0], 4)
        return np.exp(res) - np.exp(0.5)

    def predict_modified(self, step_nums, data):
        memory = self.memory
        memory_step = []
        memory_reward = []
        for x in memory:
            memory_reward.append(x[-1])
            memory_step.append(np.array(x[:step_nums+1]) - np.array(data) + 0.001)
        # print(memory_step)
        # memory_step = np.array(memory_step) - np.array(data) + 0.001
        distance = np.array(memory_step)**2
        distance = distance.sum(2)
        distance = distance.sum(1)
        distance = distance / distance.sum()
        dis_step = 1 / distance
        dis_step = dis_step / dis_step.sum()

        min_reward = np.min(memory_reward)
        max_reward = np.max(memory_reward)
        close_reward = []
        close_record = []

        # close_reward.append(memory_reward[np.argmax(memory_reward)])
        # close_record.append(memory_step[np.argmax(memory_reward)])
        for i in range(20):
            index = np.argmax(dis_step)
            close_reward.append(memory_reward[index])
            close_record.append(dis_step[index])
            dis_step[index] = -1

        r_step = (close_reward - min_reward + 1e-8) / (max_reward - min_reward + 1e-8)
        close_record = np.array(close_record)
        close_record = close_record.reshape(20, 1)

        res = np.dot(r_step, close_record)
        res = round(res[0], 4)
        return np.exp(res) - np.exp(0.5)

    def pull_action(self, step_nums):
        memory = np.array(self.memory)
        memory_step = []
        memory_reward = []
        for x in memory:
            memory_reward.append(x[-1])
            memory_step.append(x[:-1])
        # index = np.argmax(np.array(memory_reward))
        index = self.subindex
        # print(memory_step[index][step_nums])
        return int(memory_step[index][step_nums])

# #
# #
# test = KNN_Predict(size=5)
# #
# test.memory_in([1, 0, 0 , 1 , 1, 2, 1])
# test.memory_in([1, 1, 0 , 2 , 1, 2, 2])
# test.memory_in([1, 1, 2 , 2 , 1, 2, 3])
# test.memory_in([2, 1, 0 , 1 , 1, 2, 4])
# test.memory_in([1, 1, 2 , 1 , 2, 1, 0])
# # # print(test.memory)
# test.memory_in([2 ,2 ,1 ,0 ,0 ,1 , 5])
# # # print(test.memory)
# print(test.predict(step_nums=4, data=[2, 2, 1, 1, 0]))
