# Research-Planning-of-Master‘s-Thesis
「Asynchronous multi meta-reinforcement learning for game AI」
now, the first part of research is completed - using the reward shaping to the deep reinforcement learning. In the future work, I plan to improve the Asynchronous DRL (A3C), and at last, combineing the reward model and Asynchronous method to finish my master research

### part 1
Reward shaping is an efficient method to speed up the process of reinforcement learning. However, designing reward shaping functions usually require much expert demonstrations and hand-engineering. Moreover, using the potential function to shape the training rewards, RL agent had a good performance in Q-learning for converging faster in Q-table without using the expert data, but in DRL, it will sometime slow the learning of networks, especially in long horizon environment. In this paper, we propose a reward model to shape the training rewards in real-time for deep reinforcement learning, which uses the agent self-demonstrations and the potential-based reward shaping (PBRS), to make the neural networks converge faster and it can be used in either deep Q-learning or actor-critic methods. We also experimentally show that our proposed method could speed up the DRL in standard problems.
paper：[Meta-Reward Model Based on Trajectory Data with k-Nearest Neighbors Method]
