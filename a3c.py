# -*- coding: utf-8 -*-
"""A3C.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1KIDahkxKNnTuNuYK3z1faa6jxX8L5aqs
"""

# Commented out IPython magic to ensure Python compatibility.
import numpy as np
import sys
import time
import os

os.environ["OMP_NUM_THREADS"] = "1"
MAX_EP = 150000
EVAL_FREQ = 150
LSTM_SIZE = 128
ENV_NAME = "KungFuMasterDeterministic-v0"

import cv2
import numpy as np
import gym
from gym.core import Wrapper
from gym.spaces.box import Box
from preprocess_atari import make_env

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class AC_Net(nn.Module):
    def __init__(self, obs_shape, n_actions, lstm_size=128):
        """A simple actor-critic agent"""
        super(self.__class__, self).__init__()
        self.obs_shape = obs_shape
        self.lstm_size = lstm_size
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, stride=2),
            nn.ReLU(),
        )

        self.flatten = Flatten()

        self.hid = nn.Linear(self.feature_size(), self.lstm_size)
        self.rnn = nn.LSTMCell(self.lstm_size, self.lstm_size)

        self.logits = nn.Linear(self.lstm_size, n_actions)
        self.state_value = nn.Linear(self.lstm_size, 1)

    def feature_size(self):
        return self.conv(torch.zeros(1, *self.obs_shape)).view(1, -1).size(1)

    def forward(self, prev_state, obs_t):
        """
        Takes agent's previous hidden state and a new observation,
        returns a new hidden state and whatever the agent needs to learn
        """

        h = self.conv(obs_t)
        h = self.flatten(h)
        h = self.hid(h)
        h = F.relu(h)

        new_state = h_new, c_new = self.rnn(h, prev_state)
        logits = self.logits(h_new)
        state_value = self.state_value(h_new)

        return new_state, (logits, state_value)

    def get_initial_state(self, batch_size):
        """Return a list of agent memory states at game start. Each state is a np array of shape [batch_size, ...]"""
        return torch.zeros((batch_size, self.lstm_size)), \
         torch.zeros((batch_size, self.lstm_size))

    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_values = agent_outputs
        probs = F.softmax(logits, dim=1)
        return torch.multinomial(probs, 1)[:, 0].data.numpy()

    def step(self, prev_state, obs_t):
        """ like forward, but obs_t is a numpy array """
        obs_t = torch.tensor(np.asarray(obs_t), dtype=torch.float32)
        (h, c), (l, s) = self.forward(prev_state, obs_t)
        return (h.detach(), c.detach()), (l.detach(), s.detach())

    def compute_rollout_loss(self, states, actions, rewards, is_done, prev_memory_states, gamma=0.99):

        states = torch.tensor(np.asarray(states), dtype=torch.float32) # shape: [time, c, h, w]
        actions = torch.tensor(np.array(actions), dtype=torch.int64)  # shape: [time]
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)  # shape: [time]
        is_done = torch.tensor(np.array(is_done, dtype=int), dtype=torch.float32)  # shape: [time]
        is_not_done = 1 - is_done
        rollout_length = rewards.shape[0] - 1

        # predict logits, probas and log-probas using an agent.
        memory = [m.detach() for m in prev_memory_states]

        logits = []  # append logit sequence here
        state_values = []  # append state values here
        for t in range(rewards.shape[0]):
            obs_t = states[t]

            # use agent to comute logits_t and state values_t.
            # append them to logits and state_values array

            memory, (logits_t, values_t) = self.forward(memory, obs_t.unsqueeze(0))

            logits.append(logits_t.view(logits_t.size(0), -1))
            state_values.append(values_t.view(values_t.size(0), -1))

        logits = torch.stack(logits, dim=1)
        logits = logits.view(logits.size(1), -1)
        state_values = torch.stack(state_values, dim=1)
        state_values = torch.squeeze(state_values)

        probas = F.softmax(logits, dim=1)
        logprobas = F.log_softmax(logits, dim=1)
        # select log-probabilities for chosen actions, log pi(a_i|s_i)

        logprobas_for_actions = logprobas[range(len(actions)), actions]

        J_hat = 0  # policy objective as in the formula for J_hat

        value_loss = 0

        cumulative_returns = state_values[-1].detach()

        for t in reversed(range(rollout_length)):
            r_t = rewards[t]                                # current rewards
            # current state values
            V_t = state_values[t]
            V_next = state_values[t+1].detach()           # next state values
            is_alive = is_not_done[t]
            # log-probability of a_t in s_t
            logpi_a_s_t = logprobas_for_actions[t]

            # update G_t = r_t + gamma * G_{t+1} as we did in week6 reinforce

            cumulative_returns = G_t = r_t + gamma * cumulative_returns * is_alive

            # Compute temporal difference error (MSE for V(s))
            value_loss += (r_t + gamma * V_next * is_alive - V_t)**2

            # compute advantage A(s_t, a_t) using cumulative returns and V(s_t) as baseline
            advantage = cumulative_returns - V_t
            advantage = advantage.detach()

            # compute policy pseudo-loss aka -J_hat.
            J_hat += logpi_a_s_t * advantage

        # regularize with entropy
        entropy_reg = -(logprobas * probas).sum(-1).mean()

        # add-up three loss components and average over time
        loss = -J_hat / rollout_length +\
            value_loss / rollout_length +\
              -0.02 * entropy_reg

        return loss

def evaluate(agent, env, n_games=1):
    """Plays an entire game start to end, returns session rewards."""

    game_rewards = []
    for _ in range(n_games):
        # initial observation and memory
        observation = env.reset()
        prev_memories = agent.get_initial_state(1)

        total_reward = 0
        while True:
            new_memories, readouts = agent.step(
                prev_memories, observation[None, ...])
            action = agent.sample_actions(readouts)

            observation, reward, done, info = env.step(action[0])

            total_reward += reward
            prev_memories = new_memories
            if done:
                break

        game_rewards.append(total_reward)
    return game_rewards

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-5):
        super(SharedAdam, self).__init__(params, lr=lr)
        # State initialization
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                state['step'] = 0
                state['exp_avg'] = torch.zeros_like(p.data)
                state['exp_avg_sq'] = torch.zeros_like(p.data)

                # share in memory
                state['exp_avg'].share_memory_()
                state['exp_avg_sq'].share_memory_()

class Worker(mp.Process):
    def __init__(self, master, opt, process_id):
      super(Worker, self).__init__()
      self.process_id = process_id
      #self.g_ep, self.g_ep_r, self.res_queue = global_ep, global_ep_r, res_queue
      self.opt = opt
      self.master = master
      self.env = make_env(ENV_NAME)
      obs_shape = env.observation_space.shape
      n_actions = env.action_space.n
      self.lnet = AC_Net(obs_shape, n_actions, lstm_size=LSTM_SIZE)
      self.prev_observation = self.env.reset()
      self.prev_memories = self.lnet.get_initial_state(1)

    def _sync_local_with_global(self):
        self.lnet.load_state_dict(self.master.state_dict())

    def work(self, n_iter):
      obs = []
      actions = []
      rewards = []
      is_done = []
      prev_memories = self.prev_memories

      for _ in range(n_iter-1):

        new_memories, readouts = self.lnet.step(prev_memories, [self.prev_observation])
        action = self.lnet.sample_actions(readouts)

        new_observation, reward, done, _ = self.env.step(action[0])
        if done:
          new_observation = self.env.reset()
          new_memories = self.lnet.get_initial_state(1)

        obs.append(self.prev_observation)
        actions.append(action[0])
        rewards.append(reward)
        is_done.append(done)

        prev_memories = new_memories
        self.prev_observation = new_observation


      obs.append(self.prev_observation)
      actions.append(0)
      rewards.append(0)
      is_done.append(False)
      initial_memory = self.prev_memories
      self.prev_memories = prev_memories

      return obs, actions, rewards, is_done, initial_memory

    def train(self, opt, states, actions, rewards, is_done,
              prev_memory_states, gamma=0.99):
      loss = self.lnet.compute_rollout_loss(states, actions, rewards, is_done,
                                            prev_memory_states, gamma)
      opt.zero_grad()
      loss.backward()
      for lp, mp in zip(self.lnet.parameters(), self.master.parameters()):
          mp._grad = lp.grad
      opt.step()

      #self.data[0] = dumps(self.master.state_dict())

    def run(self):
        time.sleep(int(np.random.rand() * (self.process_id + 5)))
        while self.master.train_step.value < self.master.steps:
            self._sync_local_with_global()
            if self.master.train_step.value % EVAL_FREQ == 0 and self.process_id == 0:
                reward = np.mean(evaluate(self.master, make_env(ENV_NAME), n_games=3))
                torch.save(self.master.state_dict(), 'a3c.weights')
                print(self.master.train_step.value, reward)
            obs, actions, rewards, is_done, prev_memories = self.work(20)
            self.train(self.opt, obs, actions, rewards, is_done, prev_memories)
            self.master.train_step.value += 1

if __name__ == "__main__":

    env = make_env(ENV_NAME)
    obs_shape = env.observation_space.shape
    n_actions = env.action_space.n

    master = AC_Net(obs_shape, n_actions, lstm_size=LSTM_SIZE)
    master.train_step = mp.Value('l', 0)
    master.steps = MAX_EP
    master.share_memory()
    shared_opt = SharedAdam(master.parameters())

    print('Workers count:', mp.cpu_count())

    # parallel training
    workers = [Worker(master, shared_opt, i) for i in range(mp.cpu_count())]
    for worker in workers:
      worker.start()
    for worker in workers:
      worker.join()
