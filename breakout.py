import numpy as np

from configs.breakout_config import Config
from envs.breakout_env import Environment
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
