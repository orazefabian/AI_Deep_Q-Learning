# AI for self driving car

# import all libraries
from typing import Any

import numpy as np
import random
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.autograd as autograd
from torch.autograd import Variable


# architecture for neural network

class Network(nn.Module):

    def __init__(self, input_size, nb_actions):
        super(Network, self).__init__()
        self.input_size = input_size
        self.nb_actions = nb_actions
        self.fc1 = nn.Linear(input_size, 30)  # second parameter is the amount of hidden layer nodes
        self.fc2 = nn.Linear(30, nb_actions)  # fc2 is connection between hidden layer and output layer

    def forward(self, state):
        x = F.relu(self.fc1(state))
        q_values = self.fc2(x)
        return q_values


# replay experience class

class MemoryReplay(object):

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []

    # method to push a new transition or event to our memory storage
    def push(self, transition):
        self.memory.append(transition)
        if len(self.memory) > self.capacity:
            del self.memory[0]

    # method which reshapes the events and maps the samples to a torch type variable
    def sample(self, batch_size):
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)

