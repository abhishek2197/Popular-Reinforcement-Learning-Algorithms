import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from itertools import count
import math
import torch.nn.functional as F


class QNet(nn.Module):
    def __init__(self, num_actions):
        super(QNet, self).__init__()
        self.total_actions = num_actions
        self.build_model()

    def build_model(self):
        self.conv1 = nn.Conv2d(4, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.fc1 = nn.Linear(7 * 7 * 64, 512)
        self.fc2 = nn.Linear(512, self.total_actions)
        self.relu = F.relu
        self.flatten = lambda x: x.view(x.size(0), -1)
        
        nn.init.xavier_uniform_(self.conv1.weight)
        nn.init.xavier_uniform_(self.conv2.weight)
        nn.init.xavier_uniform_(self.conv3.weight)
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x) 
        return x


