from collections import deque
import numpy as np
import random
from hyperparameters import *

class ReplayMemory(object):
    def __init__(self):
    	self.memory = deque(maxlen=memory_capacity)

    def push(self, history, action, reward, done):
        self.memory.append((history, action, reward, done))

    def sample_mini_batch(self, frame):
        mini_batch = []
        if frame >= memory_capacity:
            sample_range = memory_capacity
        else:
            sample_range = frame
        sample_range -= (frames_history + 1)

        idx_sample = random.sample(range(sample_range), mini_batch_size)
        for i in idx_sample:
            sample = []
            for j in range(frames_history + 1):
                sample.append(self.memory[i + j])

            sample = np.array(sample)
            mini_batch.append((np.stack(sample[:, 0], axis=0), sample[3, 1], sample[3, 2], sample[3, 3]))

        return mini_batch

    def __len__(self):
        return len(self.memory)
