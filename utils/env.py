import gym
import matplotlib
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

resize = T.Compose([T.ToPILImage(),
                    T.Resize(40, interpolation=Image.CUBIC),
                    T.ToTensor()])


class Environment():
    def __init__(self, device):
        # * import environment * #
        self.env = gym.make('CartPole-v0').unwrapped
        self.device = device

    def reset(self):
        self.env.reset()

    def get_cart_location(self, screen_width):
        world_width = self.env.x_threshold * 2
        scale = screen_width / world_width
        return int(self.env.state[0] * scale + screen_width / 2.0)  # MIDDLE OF CART

    def get_screen(self):
        # gym이 요청한 화면은 400x600x3 이지만, 가끔 800x1200x3 처럼 큰 경우가 있습니다.
        # 이것을 Torch order (CHW)로 변환한다.
        screen = self.env.render(mode='rgb_array').transpose((2, 0, 1))
        # 카트는 아래쪽에 있으므로 화면의 상단과 하단을 제거하십시오.
        _, screen_height, screen_width = screen.shape
        screen = screen[:, int(screen_height*0.4):int(screen_height * 0.8)]
        view_width = int(screen_width * 0.6)
        cart_location = self.get_cart_location(screen_width)
        if cart_location < view_width // 2:
            slice_range = slice(view_width)
        elif cart_location > (screen_width - view_width // 2):
            slice_range = slice(-view_width, None)
        else:
            slice_range = slice(cart_location - view_width // 2,
                                cart_location + view_width // 2)
        # 카트를 중심으로 정사각형 이미지가 되도록 가장자리를 제거하십시오.
        screen = screen[:, :, slice_range]
        # float 으로 변환하고,  rescale 하고, torch tensor 로 변환하십시오.
        # (이것은 복사를 필요로하지 않습니다)
        screen = np.ascontiguousarray(screen, dtype=np.float32) / 255
        screen = torch.from_numpy(screen)
        # 크기를 수정하고 배치 차원(BCHW)을 추가하십시오.
        return resize(screen).unsqueeze(0).to(self.device)

if __name__=='__main__':
    env = Environment(device='cpu')
    env.reset()
    plt.figure()
    plt.imshow(env.get_screen().cpu().squeeze(0).permute(1, 2, 0).numpy(),
        interpolation='none')
    plt.title('Example extracted screen')
    plt.show()