import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable

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
            self.memory.append(None)
            if len(self.memory)%100 == 0:
                print('{}/{}  ({:.5f})'.format(len(self.memory),self.capacity, len(self.memory)/self.capacity) )
        # TODO find a way to use list as a queue
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        """Selects a random batch of transitions for training."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class Agent():
    def __init__(self, action_set, hParam):

        h,w = 84, 84
        self.qNetwork = DQN(h,w, len(action_set))
        self.targetNetwork = DQN(h,w,len(action_set))
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())
        
        self.optimizer = optim.RMSprop(self.qNetwork.parameters(),
                                        lr = 1e-4,
                                        momentum=0.9)
        self.loss_func = nn.MSELoss()
                   
        self.memory = ReplayMemory(50000)
        
        self.DISCOUNT_FACTOR = 0.99

        self.steps_done = 0
        self.EPS_START = 0.1
        self.EPS_END = 0.0001
        self.EPS_DECAY = 1e-7
        self.eps_threshold = self.EPS_START
        self.BATCH_SIZE = hParam['BATCH_SIZE']

        self.n_actions = len(action_set) # 2
        # ! TODO set device
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.qNetwork.to(self.device)
        self.targetNetwork.to(self.device)
        self.qNetwork.train()

    def updateTargetNet(self):
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())    


    def getAaction(self, state):
        state = torch.from_numpy(state).float() / 255.0
        sample = random.random()
        state = state.to(self.device)

        if sample > self.eps_threshold:
            # with torch.no_grad():
            # t.max(1) will return largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            estimate = self.qNetwork(state).max(1)[1].cpu()
            del state
            
            return estimate.data[0]
        else:
            return random.randint(0, self.n_actions - 1)

    def updateQnet(self):
        if len(self.memory) < self.BATCH_SIZE:
            return

        transitions = self.memory.sample(self.BATCH_SIZE)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state).to(self.device)
        action_batch = torch.cat(batch.action).to(self.device)
        next_state_batch = torch.cat(batch.next_state).to(self.device)
        reward_batch = torch.cat(batch.reward).to(self.device)
        done_batch = torch.cat(batch.done).to(self.device)


        with torch.no_grad():
            self.targetNetwork.eval()
            next_state_values = self.targetNetwork(next_state_batch)

        y_batch = torch.cat(tuple(reward if done else reward + self.DISCOUNT_FACTOR * torch.max(value) 
                                for reward, done, value in zip(reward_batch, done_batch, next_state_values)))

        state_action_values = torch.sum(self.qNetwork(state_batch) * action_batch, dim=1)

        self.optimizer.zero_grad()
        loss = self.loss_func(state_action_values, y_batch.detach())
        loss.backward()
        # for param in self.qNetwork.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.updateEPS()
        return loss.data
        
    def updateEPS(self):
        self.eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.steps_done / self.EPS_DECAY)
        self.steps_done += 1

    def save(self):
        print('save')
        torch.save({
            'state_dict': self.qNetwork.state_dict(),
        }, 'checkpoint.pth.tar')