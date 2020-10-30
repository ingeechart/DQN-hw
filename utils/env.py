import os
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

from ple import PLE
from ple.games.flappybird import FlappyBird

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from utils.utils import convert, make_video
# from utils import convert, make_video


class Environment():
    def __init__(self, device, display=False):
        # Design reward
        reward_values = {
            "positive": 1,
            "tick": 0.1,
            "loss": -10,
        }
        self.env = PLE(FlappyBird(),
                        display_screen=display,
                        reward_values=reward_values)
        self.device = device
        self.action_set = self.env.getActionSet()
        
        self.frames = []


    def reset(self):
        self.env.reset_game()

    def start(self):
        self.env.act(0)
        obs = convert(self.env.getScreenGrayscale())
        self.state = np.stack([[obs for _ in range(4)]], axis=0)
        self.t_alive = 0
        self.total_reward = 0

        return self.state

    def game_over(self):
        return self.env.game_over()

    def step(self, action):
        
        reward = self.env.act(self.action_set[action])

        # make next state
        obs = convert(self.env.getScreenGrayscale())
        obs = np.reshape(obs, [1, 1, obs.shape[0], obs.shape[1]])
        next_state = np.append(self.state[:, 1:, ...], obs, axis=1)
        
        self.t_alive += 1
        self.total_reward += reward
        self.state = next_state

        return self.state, reward, self.env.game_over()

    def record(self):
        self.frames.append(self.env.getScreenRGB())

    def saveVideo(self, episode, video_path):

        os.makedirs(video_path, exist_ok=True)
        clip = make_video(self.frames, fps=60).rotate(-90)
        clip.write_videofile(os.path.join(
            video_path, 'env_{}.mp4'.format(episode)), fps=60)
        # agent.restore_epsilon()
        print('Episode: {} t: {} Reward: {:.3f}'.format(
            episode, self.t_alive, self.total_reward))

if __name__=='__main__':
    env = Environment(device='cpu')
    print(env.action_set)
    # env.reset()
    # plt.figure()
    # plt.imshow(env.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
    #     interpolation='none')
    # plt.title('Example extracted screen')
    # plt.show()