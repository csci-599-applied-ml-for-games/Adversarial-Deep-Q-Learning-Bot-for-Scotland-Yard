import gym
import math
import random
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple
from itertools import count
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T
import os
from torch import optim


# Initialize Replay memory capacity
class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self):
        super(DQN, self).__init__()
        # self.ll1 = nn.Linear(1213, 708)
        # self.ll2 = nn.Linear(708, 708)
        # self.ll3 = nn.Linear(708, 354)
        # self.ll4 = nn.Linear(354, 354)
        # self.oll = nn.Linear(354, 8)

        # self.t1 = nn.Linear(2, 2)
        self.t2 = nn.Linear(2, 1)


    def forward(self, x):
        # x = F.relu(self.ll1(x))
        # x = F.relu(self.ll2(x))
        # x = F.relu(self.ll3(x))
        # x = F.relu(self.ll4(x))
        # x = F.relu(self.ll1(x))
        # x = F.softmax(self.oll(x))

        x = F.sigmoid(self.t2(x))
        # x = F.sigmoid(self.t2(x))
        return x




memory = ReplayMemory(10000)
policy_net = DQN()
# input = torch.randn(4, 2)
# print(input)
input = [[0,0], [0,1], [1,0], [1,1]]
input = torch.FloatTensor(input)
expd = torch.FloatTensor([1,0,0,0])
# loss_function = nn.CrossEntropyLoss()


loss_function = nn.BCELoss()

optimizer = optim.SGD(policy_net.parameters(), lr = 0.001)

for epoch in range(1000):
	policy_net.train()
	optimizer.zero_grad()
	out = policy_net(torch.FloatTensor(input))
	loss = loss_function(out,expd)
	loss.backward()
	optimizer.step()
policy_net.eval()
print(out)


