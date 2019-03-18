import random
import numpy as np
from collections import deque
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from replay_memory import ReplayMemory
from model import *
from hyperparameters import *

class Agent():
    def __init__(self, num_actions):
        self.total_actions = num_actions

        # These are hyper parameters for the DQN
        self.discount_factor = 0.99
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.explore_step = 1000000
        self.epsilon_decay = (self.epsilon - self.epsilon_min) / self.explore_step
        self.train_start = 100000
        self.update_target = 1000

        # Generate the memory
        self.replay_memory = ReplayMemory()

        # Create the policy net and the target net
        self.action_net = QNet(num_actions)
        self.target_action_net = QNet(num_actions)

        self.optimizer = optim.Adam(params=self.action_net.parameters(), lr=learning_rate)

        # initialize target net
        self.update_target_action_net()

    # after some time interval update the target net to be same with policy net
    def update_target_action_net(self):
        self.target_action_net.load_state_dict(self.action_net.state_dict())

    def get_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = random.randrange(self.total_actions)
        else:
            Q_values = self.action_net.forward(convert_to_variable(state)).data.numpy()
            action = np.argmax(Q_values)
        return action

    # pick samples randomly from replay memory (with batch_size)
    def train_action_net(self, frame):
        if self.epsilon > self.epsilon_min:
            self.epsilon -= self.epsilon_decay

        mini_batch = self.memory.sample_mini_batch(frame)
        mini_batch = np.array(mini_batch).transpose()

        history = np.stack(mini_batch[0], axis=0)
        states = np.float32(history[:, :4, :, :]) / 255.
        actions = list(mini_batch[1])
        rewards = list(mini_batch[2])
        next_states = np.float32(history[:, 1:, :, :]) / 255.
        dones = mini_batch[3] 

        Q_values = self.action_net.forward(convert_to_variable(states)).data.numpy()
        computed_q_val = np.amax(Q_values)

        Q_values_target = self.target_action_net.forward(convert_to_variable(next_states)).data.numpy()
        computed_q_val = np.amax(Q_values_target)
        target_q_val = rewards
        if not dones:
            target_q_val+=gamma*computed_q_val
        
        target_q_val = convert_to_variable(target_q_val)
        computed_q_val = convert_to_variable(computed_q_val)
        loss = SmoothL1Loss(computed_q_val, target_q_val)
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        # Optimize the model 
        ### CODE ####