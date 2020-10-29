import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from collections import namedtuple
import random
from agent.dqn import DQN
import math
from utils.utils import Transition
# Transition = namedtuple('Transition',('state', 'action', 'next_state', 'reward', 'terminal'))

class ReplayMemory:
    '''

    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)\
                
        # TODO find a way to use list as a queue
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Selects a random batch of transitions for training."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent():
    def __init__(self, action_set):

        h,w = 84, 84
        self.qNetwork = DQN(h,w, len(action_set))
        self.targetNetwork = DQN(h,w, action_set)
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())
        self.targetNetwork.eval()
        
        self.optimizer = optim.RMSprop(self.qNetwork.parameters(),
                                        lr = 1e-4,
                                        momentum=0.9)
        self.loss_func = nn.MSELoss()
                   
        self.memory = ReplayMemory(100000)
        
        self.EPS_START = 0.9
        self.EPS_END = 0.05
        self.EPS_DECAY = 200
        self.eps_threshold = self.EPS_START
        self.BATCH_SIZE = 32

        self.n_actions = len(action_set) # 2
        # ! TODO set device
        self.device = 'cpu'
    
    def updateTargetNet(self):
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())    


    def getAaction(self, state):
        state = torch.from_numpy(state).float() / 255.0
        # state = state.cuda()
        sample = random.random()

        if sample > self.eps_threshold:
            with torch.no_grad():
                # t.max(1) will return largest column value of each row.
                # second column on max result is index of where max element was
                # found, so we pick action with the larger expected reward.
                estimate = self.qNetwork(state).max(1)[1]
                return estimate.data[0]
        else:
            return random.randint(0, self.n_actions - 1)

    def updateQnet(self, state, action, reward, state_new, terminal):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=self.device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = self.qNetwork(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(self.BATCH_SIZE, device=self.device)
        next_state_values[non_final_mask] = self.targetNetwork(non_final_next_states).max(1)[0].detach()

        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = self.loss_func(state_action_values, expected_state_action_values.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()

        for param in self.qNetwork.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

        self.updateEPS(1000)

        return loss.data[0]
        
    def updateEPS(self, steps_done):
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * steps_done / self.EPS_DECAY)
        steps_done += 1