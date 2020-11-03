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
    best = 0
    global_steps = 0
    i_episode = 0

    print('TRAIN STARTS')

    while(hParam['MAX_ITER'] > global_steps ):
        # Initialize the environment and state
        env.reset()
        state = env.start()
        i_episode += 1


        while not env.game_over():
            global_steps += 1

            # Select and perform an action
            action = agent.getAction(state)

            # make an action.
            next_state, reward, done = env.step(action) # next_state, reward, done

            # frame = env.get_screen()
            # frame = np.rot90(frame, k=1)
            # frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            # frame = frame[::-1]
            # cv2.imshow('frame', frame)

            # Store the state, action, next_state, reward, done in memory
            agent.memory.push(state, action, next_state, reward, done)


            if global_steps > 50000:
                if global_steps % hParam['TARGET_UPDATE'] == 0:
                    agent.updateTargetNet()

                # Update the target network, copying all weights and biases in DQN
                if env.game_over():
                    print('Episode: {} Score: {:.4f} Episode Total Reward: {:.4f} Loss: {:.4f}'.format(
                       i_episode, env.score(), env.total_reward, loss))
                    if env.total_reward > best:
                        agent.save()
                        best = env.total_reward
            elif global_steps%500 == 0: 
                print('steps {}/{}'.format(global_steps, hParam['MAX_ITER']))

            # update Qnetwork
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
    sungjun = Agent(env.action_set, hParam)
    train(hParam, env, sungjun)
