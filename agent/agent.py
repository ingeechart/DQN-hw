import numpy as np
import random

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from utils.utils import Transition
from agent.dqn import DQN

def convertToTensor(*args):

    state = torch.from_numpy(args[0]).float() / 255.0
    
    action_onehot = np.zeros(2)
    action_onehot[args[1]] = 1
    action_onehot = np.expand_dims(action_onehot, axis=0)
    action = torch.from_numpy(action_onehot).float()
    
    next_state = torch.from_numpy(args[2]).float() / 255.0
    reward = torch.tensor([[args[3]]]).float()
    done = torch.tensor([[args[4]]])

    return (state, action, next_state, reward, done)

class ReplayMemory:
    '''
        pytorch DQN tutorial official code
    '''
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        args = convertToTensor(*args)
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)

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
                   
        self.memory = ReplayMemory(30000)
        
        self.DISCOUNT_FACTOR = 0.99  # hParam['DISCOUNT_FACTOR']

        self.steps_done = 0
        self.EPS_START = 0.1  # hParam['EPS_START']
        self.EPS_END = 0.0001  # hParam['EPS_END']
        self.EPS_DECAY = 1e-7  # hParam['EPS_DECAY']
        self.MAX_ITER = 2000000 # hParam['MAX_ITER']
        self.eps_threshold = self.EPS_START
        self.BATCH_SIZE = hParam['BATCH_SIZE']


        self.n_actions = len(action_set) # 2

        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.qNetwork.to(self.device)
        self.targetNetwork.to(self.device)
        self.qNetwork.train()

    def updateTargetNet(self):
        self.targetNetwork.load_state_dict(self.qNetwork.state_dict())    


    def getAction(self, state):
        state = torch.from_numpy(state).float() / 255.0
        sample = random.random()
        state = state.to(self.device)

        if sample > self.eps_threshold or self.steps_done > 1000000:
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

        loss = self.loss_func(state_action_values, y_batch.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        # for param in self.qNetwork.parameters():
        #     param.grad.data.clamp_(-1, 1)
        self.optimizer.step()
        self.updateEPS()
        return loss.data
        
    def updateEPS(self):
        self.steps_done += 1

        self.eps_threshold = self.EPS_END \
                        + ((self.MAX_ITER - self.steps_done) \
                            * (self.EPS_START - self.EPS_END) / self.MAX_ITER)

    def save(self, path='checkpoint.pth.tar'):
        print('save')
        torch.save({
            'state_dict': self.qNetwork.state_dict(),
        }, path)