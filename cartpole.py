import numpy as np

from configs.cartpole_config import Config
from envs.cartpole_env import Environment
from dqn.agent import DQNAgent
from dqn.trainer import Trainer


def main():
    config = Config()
    env = Environment(config)
    agent = DQNAgent(config)
    trainer = Trainer(config, env, agent)
    trainer.train()
    trainer.play()


if __name__ == "__main__":
    main()
