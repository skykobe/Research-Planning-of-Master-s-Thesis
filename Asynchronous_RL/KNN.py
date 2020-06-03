import numpy as np

def onehot(action):
    if(action == 0):
        return [1,0]
    if(action == 1):
        return [0,1]

class KNN_Predict:
    def __init__(self, size):
        self.memory_size = size
        self.memory = []
        self.count = 0
        self.minus_stepnum = 999

    def memory_in(self, data):
        if(self.count >= self.memory_size):
            num = len(data)
            reward_for_memory = []
            memory = np.array(self.memory)
            # print('all data', memory)
            # reward_for_memory = memory[:, -1]
            for x in memory:
                reward_for_memory.append(x[-1])
            index = np.argmin(reward_for_memory)
            data_r = data[-1]
            if(reward_for_memory[index] <= data_r):
                # print(data_r, reward_for_memory[index])
                self.memory[index] = data
                self.minus_stepnum = len(self.memory[1])-1
                for index in self.memory:
                    if(len(index) - 1 < self.minus_stepnum):
                        self.minus_stepnum = len(index) - 1
        else:
            if data[-1] > 0:
                print('store new data')
                self.memory.append(data)
                self.count += 1
                if(len(data) - 1 < self.minus_stepnum):
                    self.minus_stepnum = len(data) - 1

        # print('knn memory size', len(self.memory))
    def save_knn_memory(self):
        np.save('knn_reward_model_memory', self.memory)

    def load_memory(self):
        self.memory = list(np.load('knn_reward_model_memory.npy', allow_pickle=True))
        self.count = len(self.memory)

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
        for i in range(10):
            index = np.argmax(dis_step)
            close_reward.append(memory_reward[index])
            close_record.append(dis_step[index])
            dis_step[index] = -1

        r_step = (close_reward - min_reward + 1e-8) / (max_reward - min_reward + 1e-8)
        close_record = np.array(close_record)
        close_record = close_record.reshape(10, 1)

        res = np.dot(r_step, close_record)
        res = round(res[0], 4)
        return np.exp(res) - np.exp(0.5)
