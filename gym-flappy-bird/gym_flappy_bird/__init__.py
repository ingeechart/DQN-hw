#!/usr/local/bin/python3
# -*- coding: utf-8 -*-
from gym.envs.registration import register

register(
    id='flappy-bird-v0',
    entry_point='gym_flappy_bird.envs:FlappyBirdEnv'
)
