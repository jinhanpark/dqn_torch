import gym

import numpy as np

import cv2

from .utils import rgb2gray

import random


class Environment:

    def __init__(self, config):
        self.make()
        self.config = config

    def initialize_game(self):
        self._state = self.env.reset()
        return self.state, 0, 0, self.done

    def start_randomly(self):
        self.initialize_game()
        self.render()
        return self.state, 0, 0, self.done

    def _step(self, action):
        self._state, self.reward, self.done, self.info = self.env.step(action)

    def _random_step(self):
        action = self.env.action_space.sample()
        self._step(action)

    @property
    def action_space_size(self):
        return self.env.action_space.n

    @property
    def state(self):
        return np.expand_dims(self._state, axis=0)

    def render(self):
        if self.config.render:
            self.env.render()

    def after_act(self):
        self.render()

    def act(self, action):
        self._step(action)
        self.after_act()
        return self.state, self.reward, self.done

    def make(self):
        self.env = gym.make('CartPole-v0')
        self._screen = None
        self.reward = 0
        self.done = True
        self.info = None
