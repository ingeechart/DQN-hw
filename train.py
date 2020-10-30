import math
import random
import numpy as np
# import matplotlib
# import matplotlib.pyplot as plt
from collections import namedtuple
# from itertools import count
from PIL import Image

import cv2

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils.env import Environment
from agent.agent import Agent


# * incase using GPU * #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
episode_durations = []


def convertToTensor(state, action, next_state, reward, done):
    state = torch.from_numpy(state).float() / 255.0

    action_onehot = np.zeros(2)
    action_onehot[action] = 1
    action_onehot = np.expand_dims(action_onehot, axis=0)
    action = torch.from_numpy(action_onehot).float()

    next_state = torch.from_numpy(next_state).float() / 255.0
    reward = torch.tensor([[reward]]).float()
    done = torch.tensor([[done]])

    return state, action, next_state, reward, done


def train(hParam, env, agent):
    num_episodes = 1000000
    best = 0

    for i_episode in range(num_episodes):

        # Initialize the environment and state
        env.reset()
        state = env.start()

        while not env.game_over():
            # Select and perform an action
            action = agent.getAaction(state)
            next_state, reward, done = env.step(action) # next_state, reward, done
            # print(type(state), type(action), type(next_state), type(reward), type(done))
            # state(ndarray), action(int), next_state(ndarray), reward(float), done(bool)

            frame = env.get_screen()
            frame = np.rot90(frame, k=1)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            frame = frame[::-1]
            cv2.imshow('frame', frame)

            # Store the transition in memory
            state_, action_, next_state_, reward_, done_ = convertToTensor(
                                                        state, action, next_state, reward, done)
            # print(state_.shape, action_.shape, next_state_.shape, reward_.shape, done_.shape)
            # torch.Size([1, 4, 84, 84]) torch.Size([1, 1]) torch.Size([1, 4, 84, 84]) torch.Size([1, 1]) torch.Size([1, 1])

            agent.memory.push(state_, action_, next_state_, reward_, done_ )
            loss = agent.updateQnet()

            # Move to the next state
            state = next_state

        cv2.destroyAllWindows()

        # Update the target network, copying all weights and biases in DQN
        if i_episode > 100:
            if i_episode % hParam['TARGET_UPDATE'] == 0:
                agent.updateTargetNet()

            if (i_episode % 10) == 1:
                print('Episode: {} Reward: {:.3f} Loss: {:.3f}'.format(
                    i_episode, env.total_reward, loss))
                if env.total_reward > best:
                    agent.save()
                    best = env.total_reward
                    # env.total_reward = best
        # loss = agent.updateQnet()


if __name__ == '__main__':
    hParam = {
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'TARGET_UPDATE': 5
    }
    env = Environment(device, display=True)
    chulsoo = Agent(env.action_set, hParam)
    train(hParam, env, chulsoo)
