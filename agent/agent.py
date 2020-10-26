import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
import random
from agent.dqn import DQN, ReplayMemory

class Agent():
    def __init__(self, h,w,outputs):

        self.qNetwork = DQN(h,w,outputs)
        self.targetNetwork = DQN(h,w,outputs)
        self.optimizer = optim.RMSprop(self.qNetwork.parameters(),
                                        lr = 1e-4,
                                        momentum=0.9)
        self.memory = ReplayMemory(100000)

    def update_targetNet(self):
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())
    
    def update_qNet(self):
        state = 0
        action = 0
        action = 0
        terminal = 0
        reward = 0
