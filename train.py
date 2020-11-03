import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from utils.env import Environment
from agent.agent import Agent


# * incase using GPU * #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
episode_durations = []


def train(hParam, env, agent):
    num_episodes = 1000000
    best = 0

    for i_episode in range(num_episodes):

        # Initialize the environment and state
        env.reset()
        state = env.start()

        while not env.game_over():
            # Select and perform an action
            action = agent.getAction(state)
            next_state, reward, done = env.step(action) # next_state, reward, done

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


            if global_steps > 50000:
                if global_steps % hParam['TARGET_UPDATE'] == 0:
                    agent.updateTargetNet()

                # Update the target network, copying all weights and biases in DQN
                if env.game_over():
                    print('Episode: {} Episode Total Reward: {:.3f} Loss: {:.3f}'.format(
                        i_episode, env.total_reward, loss))
                    if env.total_reward > best:
                        agent.save()
                        best = env.total_reward
                        

            loss = agent.updateQnet()
            # Move to the next state
            state = next_state

        cv2.destroyAllWindows()


if __name__ == '__main__':
    hParam = {
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'TARGET_UPDATE': 5,
        'EPS_START': 0.1,
        'EPS_END': 0.0001,
        'EPS_DECAY': 1e-7,
        'MAX_ITER': 2000000,
        'DISCOUNT_FACTOR': 0.99,
        'LR': 1e-4,
        'MOMENTUM': 0.9,
        'BUFFER_SIZE': 50000
    }
    env = Environment(device, display=True)
    chulsoo = Agent(env.action_set, hParam)
    train(hParam, env, chulsoo)
