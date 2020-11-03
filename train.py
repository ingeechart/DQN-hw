import math
import numpy as np
import cv2

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T

from utils.env import Environment
from agent.agent import Agent

if torch.__version__ <= '1.1.0':
    from tensorboardX import SummaryWriter
else:
    from torch.utils.tensorboard import SummaryWriter

# * incase using GPU * #
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
episode_durations = []


def train(hParam, env, agent):
    best = 0
    global_steps = 0
    i_episode = 0

    writer = SummaryWriter()

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


            if global_steps > hParam['BUFFER_SIZE']:
                if global_steps % hParam['TARGET_UPDATE'] == 0:
                    agent.updateTargetNet()

                # Update the target network, copying all weights and biases in DQN
                if env.game_over():
                    print('Episode: {}  Global Step: {}  Episode Total Reward: {:.4f} Loss: {:.4f}'.format(
                       i_episode, global_steps, env.total_reward, loss))

                    writer.add_scalar('Episode_total_reward', env.total_reward, i_episode)

                    if env.total_reward > best:
                        agent.save()
                        best = env.total_reward

                # update Qnetwork
                loss = agent.updateQnet()
                writer.add_scalar('train_loss', loss, global_steps)

            elif global_steps%500 == 0: 
                print('steps {}/{}'.format(global_steps, hParam['MAX_ITER']))

            # Move to the next state
            state = next_state

        cv2.destroyAllWindows()


if __name__ == '__main__':
    hParam = {
        'BATCH_SIZE': 32,
        'GAMMA': 0.99,
        'TARGET_UPDATE': 10000,
        'EPS_START': 0.1,
        'EPS_END': 0.0001,
        'MAX_ITER': 2000000,
        'DISCOUNT_FACTOR': 0.99,
        'LR': 1e-6,
        'MOMENTUM': 0.9,
        'BUFFER_SIZE': 30000
    }

    env = Environment(device, display=True)
    sungjun = Agent(env.action_set, hParam)
    train(hParam, env, sungjun)
