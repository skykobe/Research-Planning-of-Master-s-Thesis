import tensorflow as tf
import numpy as np

class KNN_Predict:
    def __init__(self, size):
        self.memory_size = size
        self.memory = []
        self.count = 0
        self.minus_stepnum = 999

    def memory_in(self, data):
        if(self.count >= self.memory_size):
            # print('out of size')
            # memory满了以后，进行数据交换，交换原创是？ 1，根据分数的高到低？ 2，类似与缓存的机制？关注局所性 3， LRU ?
            # 先尝试最简单的, 保存最高的数据
            num = len(data)
            reward_for_memory = []
            memory = self.memory
            # print('all data', memory)
            # reward_for_memory = memory[:, -1]
            for x in memory:
                reward_for_memory.append(x[-1])
            index = np.argmin(reward_for_memory)
            data_r = data[-1]
            if(reward_for_memory[index] <= data_r):
                # print(data_r, reward_for_memory[index])
                self.memory[index] = data
                self.minus_stepnum = len(self.memory[0]) - 1
                for index in self.memory:
                    if(len(index) < self.minus_stepnum):
                        self.minus_stepnum = len(index) - 1
        else:
            self.memory.append(data)
            self.count += 1
            if(len(data) < self.minus_stepnum):
                self.minus_stepnum = len(data) - 1

        # print('knn memory size', len(self.memory))

    def predict(self, step_nums, data):
        # if(self.count >= self.memory_size):
        memory = np.array(self.memory)
        # memory_step = memory[:, :step_nums] - data + 0.1
        memory_step = []
        memory_reward = []
        # print(step_nums, x[:step_nums])
        for x in memory:
            # print(x[-1], x[:step_nums - 1])
            memory_reward.append(x[-1])
            memory_step.append(x[:step_nums+1])
        # print(memory_step, memory_reward, data[-1])
        memory_step = np.array(memory_step) - np.array(data) + 0.001
        #先求Pn
        # print(memory_step)
        distance = np.array(memory_step)**2
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

    def predict_maze(self, step_nums, data):
        # if(self.count >= self.memory_size):
        memory = self.memory
        # memory_step = memory[:, :step_nums] - data + 0.1
        memory_step = []
        memory_reward = []
        # print(step_nums, x[:step_nums])
        for x in memory:
            # print(x[-1], x[:step_nums - 1])
            memory_reward.append(x[-1])
            memory_step.append(x[:step_nums+1])
        # print(memory_step, memory_reward, data[-1])
        memory_step = np.array(memory_step) - np.array(data) + 0.001
        #先求Pn
        distance = np.array(memory_step)**2
        distance = distance.sum(2)
        # print(distance)
        distance = distance.sum(1)
        distance = distance / distance.sum()
        dis_step = 1 / distance
        dis_step = dis_step / dis_step.sum()
        # print(dis_step)
        #再求Xn
        max_r = np.max(memory_reward)
        min_r = np.min(memory_reward)
        r_step = (memory_reward - min_r + 1e-8) / (max_r - min_r + 1e-8)
        # print(dis_step)
        dis_step = dis_step.reshape(self.memory_size, 1)

        res = np.dot(r_step, dis_step)
        res = round(res[0], 4)
        return np.exp(res) - np.exp(0.5)

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
