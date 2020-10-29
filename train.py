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

from utils.env import Environment
from agent.agent import Agent


# * incase using GPU * #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

episode_durations = []

def convertToTensor(state, action, next_state, reward, done):
    state = torch.from_numpy(state).float() / 255.0
    action = torch.from_numpy(action).float()
    next_state = torch.from_numpy(next_state).float() / 255.0
    reward = torch.from_numpy(reward).float()
    done = torch.from_numpy(done).float()

    # state = state.cuda()
    # action = action.cuda()
    # state_new = state_new.cuda()
    # terminal = terminal.cuda()
    # reward = reward.cuda()
    
    return state, action, next_state, reward, done

def train(hyerParam, env, agent):
    num_episodes = 50


    for i_episode in range(num_episodes):

        # Initialize the environment and state
        env.reset()
        state = env.start()

        while not env.game_over():
            # Select and perform an action
            action = agent.getAaction(state)
            next_state, reward, done = env.step(action.item()) # next_state, reward, done
            # reward = torch.tensor([reward], device=device)


            # Store the transition in memory
            state_, action_, next_state, reward_, done_ = convertToTensor(
                                                        state, action, next_state, reward, done)
            agent.memory.push(state_, action_, next_state, reward_, done_ )

            # Move to the next state
            state = next_state

            # Perform one step of the optimization (on the target network)
            # memory에 있는 data를 통해 data를 sample 하여 qNetwork를 training 한다.
            optimize_model()
            if done:
                episode_durations.append(t + 1)
                plot_durations()
                break
        # Update the target network, copying all weights and biases in DQN
        if i_episode % hyerParam.TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

        loss = agent.updateQnet()

        
if __name__=='main':
    hyerParam = {
        'BATCH_SIZE' = 64,
        'GAMMA' = 0.999,
        'TARGET_UPDATE' = 10
    }
    env = Environment(device)
    chulsoo = Agent(env.action_set)
    train(hyerParam, env, chulsoo)