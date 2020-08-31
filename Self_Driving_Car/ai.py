# AI for self driving car

# import all libraries

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


# deep q-learning algorithm

class Dqn:

    def __init__(self, input_size, nb_actions, gamma, capacity):
        self.gamma = gamma
        self.rewards = []
        self.model = Network(input_size, nb_actions)
        self.memory = MemoryReplay(capacity)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)  # needing a fake dimension for a tensor type
        self.last_action = 0
        self.last_reward = 0

    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volelile=True)) * 7)  # T=7
        action = probs.multinomial(num_samples=1)
        return action.data[0, 0]

    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma * next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)  # best cost function for q-learning
        self.optimizer.zero_grad()
        td_loss.backward(retain_graph=True)
        self.optimizer.step()  # update weights with this step

    def update(self, reward, new_signal):
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push(
            (self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_reward, batch_action = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action
        self.last_state = new_state
        self.last_reward = reward
        self.rewards.append(reward)
        if len(self.rewards) > 1000:
            del self.rewards[0]
        return action

