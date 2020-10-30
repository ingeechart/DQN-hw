import os
import cv2
import json
import shutil
import logging
import math
import random
from collections import namedtuple

import torch
import numpy as np

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward', 'done'))


def convert(image):
    """
    Convert images to 84 * 84.
    Args:
        image: PLE game screen.
    """
    image = cv2.resize(image, (84, 84))
    _, image = cv2.threshold(image, 100, 255, cv2.THRESH_BINARY_INV)

    return image


def make_video(images, fps):
    """
    Make videos.
    Args:
        images:
        fps:
    """
    import moviepy.editor as mpy
    duration = len(images) / fps

    def make_frame(t):
        """A function `t-> frame at time t` where frame is a w*h*3 RGB array."""
        try:
            x = images[int(len(images) / duration * t)]
        except:
            x = images[-1]
        return x.astype(np.uint8)

    clip = mpy.VideoClip(make_frame, duration=duration)
    clip.fps = fps

    return clip