import numpy as np
import time
import sys
if sys.version_info.major == 2:
    import Tkinter as tk
else:
    import tkinter as tk

np.random.seed(1)
UNIT = 40   # pixels
MAZE_H = 8  # grid height
MAZE_W = 8  # grid width


class Maze(tk.Tk, object):
    def __init__(self):
        super(Maze, self).__init__()
        self.action_space = ['u', 'd', 'l', 'r']
        self.wall_coords = []
        self.n_actions = len(self.action_space)
        self.n_features = 4
        self.step_count = 0
        self.title('maze')
        self.geometry('{0}x{1}'.format(MAZE_H * UNIT, MAZE_H * UNIT))
        self._build_maze()

    def _build_maze(self):
        self.canvas = tk.Canvas(self, bg='white',
                           height=MAZE_H * UNIT,
                           width=MAZE_W * UNIT)

        # create grids
        for c in range(0, MAZE_W * UNIT, UNIT):
            x0, y0, x1, y1 = c, 0, c, MAZE_H * UNIT
            self.canvas.create_line(x0, y0, x1, y1)
        for r in range(0, MAZE_H * UNIT, UNIT):
            x0, y0, x1, y1 = 0, r, MAZE_W * UNIT, r
            self.canvas.create_line(x0, y0, x1, y1)

        # create origin
        origin = np.array([20, 20])

        # hell                      (x，y)
        self.hell1 = self.black_holl(5, 0)
        self.hell2 = self.black_holl(1, 1)
        self.hell3 = self.black_holl(0, 2)
        self.hell4 = self.black_holl(1, 2)
        self.hell5 = self.black_holl(4, 2)
        self.hell6 = self.black_holl(7, 2)
        self.hell7 = self.black_holl(6, 2)
        self.hell8 = self.black_holl(7, 1)
        self.hell9 = self.black_holl(0, 4)
        self.hell10 = self.black_holl(0, 5)
        self.hell11 = self.black_holl(2, 4)
        self.hell12 = self.black_holl(3, 5)
        self.hell13 = self.black_holl(6, 4)
        self.hell14 = self.black_holl(6, 5)
        self.hell15 = self.black_holl(7, 5)
        self.hell16 = self.black_holl(6, 6)
        self.hell17 = self.black_holl(3, 7)
        self.hell18 = self.black_holl(4, 7)
        # self.hell19 = self.black_holl(6, 5)
        # self.hell20 = self.black_holl(0, 6)
        # self.hell21 = self.black_holl(3, 6)
        # self.hell22 = self.black_holl(4, 6)
        # self.hell23 = self.black_holl(6, 6)
        # self.hell24 = self.black_holl(4, 7)
        # self.hell25 = self.black_holl(6, 7)
        self.oval = self.goal_holl()
        self.rect = self.start_holl()
        s = np.array(self.canvas.coords(self.rect))
        s_ = np.array(self.canvas.coords(self.oval))
        self.dis = sum(abs(s_ - s))

        # print(self.dis)
        # pack all

        # self.canvas.pack()


    def reset(self):
        self.update()
        time.sleep(0.01)
        self.canvas.delete(self.rect)
        origin = np.array([20, 20])
        origin = origin + np.array([0, UNIT * 7])
        self.rect = self.canvas.create_rectangle(
            origin[0] - 18, origin[1] - 18,
            origin[0] + 18, origin[1] + 18,
            fill='red')
        # return observation
        return self.canvas.coords(self.rect)

    def step(self, action):
        self.step_count += 1
        s = self.canvas.coords(self.rect)
        base_action = np.array([0, 0])
        if action == 0:   # up
            if s[1] > UNIT:
                base_action[1] -= UNIT
        elif action == 1:   # down
            if s[1] < (MAZE_H - 1) * UNIT:
                base_action[1] += UNIT
        elif action == 2:   # right
            if s[0] < (MAZE_W - 1) * UNIT:
                base_action[0] += UNIT
        elif action == 3:   # left
            if s[0] > UNIT:
                base_action[0] -= UNIT

        s[0] += base_action[0]
        s[1] += base_action[1]
        s[2] += base_action[0]
        s[3] += base_action[1]

        if s not in self.wall_coords:
            self.canvas.move(self.rect, base_action[0], base_action[1])  # move agent

        s_ = self.canvas.coords(self.rect)  # next state
        # reward function
        if s_ == self.canvas.coords(self.oval):
            reward = 1
            done = True
            self.step_count = 0
        else:
            reward = 0
            # reward = self.cal_reward(s_)
            # print(reward)
            done = False

        if(self.step_count > 10000):
            reward = -1
            done = True
            self.step_count = 0

        return s_, reward, done

    def cal_reward(self, state):
        s_ = self.canvas.coords(self.oval)
        tmp = abs(np.array(state) - np.array(s_))
        dis = sum(tmp)
        cha = (self.dis - dis)/20
        if(cha < 0):
            reward = -1
        if(cha > 0):
            reward = 1
        else:
            reward = -1

        self.dis = dis
        return reward

    def black_holl(self, x, y): # for wall
        origin = np.array([20, 20])
        hell1_center = origin + np.array([UNIT * x, UNIT * y])
        hell = self.canvas.create_rectangle(
            hell1_center[0] - 18, hell1_center[1] - 18,
            hell1_center[0] + 18, hell1_center[1] + 18,
            fill='black')
        self.wall_coords.append(self.canvas.coords(hell))
        return hell

    def start_holl(self):
        origin = np.array([20, 20])
        origin = origin + np.array([0, UNIT * 7])
        rect = self.canvas.create_rectangle(
            origin[0] - 18, origin[1] - 18,
            origin[0] + 18, origin[1] + 18,
            fill='red')
        return rect

    def goal_holl(self):
        origin = np.array([20, 20])
        hell1_center = origin + np.array([UNIT * 7, 0])
        hell = self.canvas.create_rectangle(
            hell1_center[0] - 18, hell1_center[1] - 18,
            hell1_center[0] + 18, hell1_center[1] + 18,
            fill='yellow')
        return hell

    def render(self):
        time.sleep(0.01)
        self.update()

    def result_draw(self, array):
        self.canvas.create_line(array, fill='red')


def update():
    for t in range(10):
        s = env.reset()
        while True:
            env.render()
            a = 1
            s, r, done = env.step(a)
            if done:
                break

if __name__ == '__main__':
    env = Maze()
    env.after(100, update)
    env.mainloop()
