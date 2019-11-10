import numpy as np

import random

import time

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

from .replay_memory import ReplayMemory
from .short_term_memory import ShortTermMemory

from .utils import makedir_if_there_is_no

from tqdm import tqdm


class Trainer:

    def __init__(self, config, env, agent):
        self.config = config
        self.env = env
        self.agent = agent
        self.memory = ReplayMemory(config)
        self.short_term = ShortTermMemory(config)

        self.loss = F.smooth_l1_loss
        self.optim = optim.RMSprop(self.agent.q_net.parameters(), lr=config.lr)

    def train(self):

        screen, action, reward, done = self.env.start_randomly()

        for _ in range(self.config.history_length):
            self.short_term.add(screen)

        start_step = 0
        for self.step in tqdm(range(start_step, self.config.max_step),
                              ncols=70,
                              total=self.config.max_step,
                              initial=start_step):
            if self.step == start_step or self.step == self.config.replay_start_size:
                self.update_cnt = 0
                ep_reward = 0.
                total_reward = 0.
                ep_rewards = []
                max_avg_record = 0.
                self.total_loss = 0.
                self.total_q = 0.

            action = self.agent.choose_action(
                torch.from_numpy(self.short_term.frames).to(device),
                self.get_eps())
            screen, reward, done = self.env.act(action)
            self.after_act(action, screen, reward, done)

            if done:
                screen, action, reward, done = self.env.start_randomly()
                ep_rewards.append(ep_reward)
                ep_reward = 0.
            else:
                ep_reward += reward

            total_reward += reward

            if self.step > self.config.replay_start_size and self.step % self.config.summarize_step == 0:
                # TODO: tensorboard summarization
                avg_reward = total_reward / self.config.summarize_step
                avg_loss = self.total_loss / self.update_cnt
                avg_q = self.total_q / self.update_cnt
                try:
                    max_ep_reward = np.max(ep_rewards)
                    avg_ep_reward = np.mean(ep_rewards)
                except:
                    max_ep_reward = 0
                    avg_ep_reward = 0

                print(
                    "\navg_r: {:.4f}, avg_l: {:.6f}, avg_q: {:3.6f}, avg_ep_r: {:.4f}, max_ep_r: {:.4f}"
                    .format(avg_reward, avg_loss, avg_q, avg_ep_reward,
                            max_ep_reward))

                if max_avg_record * 0.9 <= avg_ep_reward:
                    max_avg_record = max(max_avg_record, avg_ep_reward)

                self.update_cnt = 0
                ep_reward = 0.
                total_reward = 0.
                ep_rewards = []
                self.total_loss = 0.
                self.total_q = 0.

    def get_eps(self):
        return 0.5
        if self.step < self.config.replay_start_size:
            return 1.
        elif self.step < self.config.final_exploration_step:
            return 1 - 0.9 * ((self.step - self.config.replay_start_size) /
                              (self.config.final_exploration_step -
                               self.config.replay_start_size))
        else:
            return self.config.final_exploration

    def reward_clipping(self, r):
        return max(self.config.min_reward, min(self.config.max_reward, r))

    def experience_replay(self):
        batch = list(self.memory.get_batch())
        for i in range(len(batch)):
            batch[i] = torch.from_numpy(batch[i]).to(device)
        S, A, R, NS, Done = batch
        S = S.to(torch.float)
        A = A.to(torch.long)
        NS = NS.to(torch.float)

        Q = self.agent.q_net(S).gather(1, A)

        next_state_values = torch.zeros(self.config.batch_size, device=device)
        next_state_values = self.agent.fixed_net(NS).max(1)[0].detach()

        targets = (next_state_values * self.config.df) + R

        loss = self.loss(Q, targets.unsqueeze(1))

        self.optim.zero_grad()
        loss.backward()
        for param in self.agent.q_net.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optim.step()

        self.total_loss += loss
        self.total_q += torch.mean(Q)
        self.update_cnt += 1

    def after_act(self, action, screen, reward, done):
        # reward = self.reward_clipping(reward)

        self.short_term.add(screen)
        self.memory.memorize(action, screen, reward, done)

        if self.step > self.config.replay_start_size:
            if self.step % self.config.replay_frequency == 0:
                self.experience_replay()
            if self.step % self.config.fixed_net_update_frequency == 0:
                self.agent.update_fixed_target()
            # TODO: rl decaying
            # TODO: model saving

    def play(self, test=False):
        try:
            self.load()
        except:
            print("****FAILED to load weights. Can't play anymore")
            return None

        screen, action, reward, done = self.env.initialize_game()

        for _ in range(self.config.history_length):
            self.short_term.add(screen)

        ep_rewards = []

        for i in range(self.config.test_play_num):
            ep_reward = 0
            while not done:
                if test:
                    action = int(
                        input("Enter the action (0 ~ {}): ".format(
                            self.env.action_space_size - 1)))
                else:
                    action = self.agent.choose_action(
                        self.short_term.frames, self.config.test_exploration)
                screen, reward, done = self.env.act(action)
                time.sleep(1 / 240)
                self.short_term.add(screen)
                ep_reward += reward
            ep_rewards.append(ep_reward)
            print("game #: {}, reward: {}".format(i + 1, ep_reward))
            screen, action, reward, done = self.env.initialize_game()

        print("Evaluation Done.\n mean reward: {}, max reward: {}".format(
            np.mean(ep_rewards), np.max(ep_rewards)))
