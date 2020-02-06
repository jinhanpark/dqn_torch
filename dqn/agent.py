import random

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .model import ConvNet


class BaseAgent:

    def __init__(self, config):
        self.config = config


class DQNAgent(BaseAgent):

    def __init__(self, config):
        super(DQNAgent, self).__init__(config)
        self.model = ConvNet(self.config).to(device)
        self.fixed_net = ConvNet(self.config).to(device)
        self.update_fixed_target()

    def update_fixed_target(self):
        self.fixed_net.load_state_dict(self.model.state_dict())

    def choose_action(self, state, eps):
        if random.random() < eps:
            return random.randrange(self.config.output_size)
        else:
            with torch.no_grad():
                a = int(self.model(state).max(1)[1][0])
                return a