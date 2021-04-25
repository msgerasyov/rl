# -*- coding: utf-8 -*-

import numpy as np
import networkx as nx
import sys
import time
import os
import pickle
import math

os.environ["OMP_NUM_THREADS"] = "1"
MAX_EP = 100000
NUM_PROCESSES = 16
NUM_STEPS = 10
EVAL_FREQ = 150
LSTM_SIZE = 128
MAX_GRAD = 40
GCN_ALPHA = 0.8
SCALE = 1
CFG = "my_way_home.cfg"

import cv2
import numpy as np
from gym.core import Wrapper
from gym.spaces.box import Box

from preprocess_doom import make_env

def crop_func(img):
    return img[20:-20, 60:-60]

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp

import networkx as nx
from pygcn import update_graph
from pygcn import GCN

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

class AC_Net(nn.Module):
    def __init__(self, obs_shape, n_actions, lstm_size=128):
        """A simple actor-critic agent"""
        super(self.__class__, self).__init__()
        self.obs_shape = obs_shape
        self.n_actions = n_actions
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

        self.rnn = nn.LSTMCell(self.feature_size(), self.lstm_size)

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

        new_state = h_new, c_new = self.rnn(h, prev_state)
        logits = self.logits(h_new)
        state_value = self.state_value(h_new)

        return new_state, (logits, state_value), h

    def get_initial_state(self, batch_size):
        """Return a list of agent memory states at game start. Each state is a np array of shape [batch_size, ...]"""
        return torch.zeros((batch_size, self.lstm_size)), torch.zeros((batch_size, self.lstm_size))

    def sample_actions(self, agent_outputs):
        """pick actions given numeric agent outputs (np arrays)"""
        logits, state_values = agent_outputs
        probs = F.softmax(logits, dim=1)
        actions = torch.multinomial(probs, 1)[:, 0].data.numpy()
        return actions

    def step(self, prev_state, obs_t):
        """ like forward, but obs_t is a numpy array """
        obs_t = torch.tensor(np.asarray(obs_t), dtype=torch.float32)
        (h, c), (l, s), hid = self.forward(prev_state, obs_t)
        return (h, c), (l, s), hid.detach()


class EnvPool(object):
    def __init__(self, make_env, n_parallel_games=1):
        # Create Atari games.
        self.make_env = make_env
        self.envs = [self.make_env(CFG, SCALE, crop = crop_func) for _ in range(n_parallel_games)]

        # Initial observations.
        self.prev_observations = self.reset()


    def step(self, actions):

        obs = []
        rewards = []
        dones = []

        for i in range(len(self.envs)):
            next_obs, r, done, _ = self.envs[i].step(actions[i])
            if done:
                next_obs = self.envs[i].reset()
            obs.append(next_obs)
            rewards.append(r)
            dones.append(done)

        self.prev_observations = np.array(obs)

        return self.prev_observations, np.array(rewards), np.array(dones)

    def reset(self):
        obs = []
        for env in self.envs:
            obs.append(env.reset())
        return np.array(obs)


def to_one_hot(y, n_dims=None):
    """ Take an integer tensor and convert it to 1-hot matrix. """
    y_tensor = y.to(dtype=torch.int64).reshape(-1, 1)
    n_dims = n_dims if n_dims is not None else int(torch.max(y_tensor)) + 1
    y_one_hot = torch.zeros(y_tensor.size()[0], n_dims).scatter_(1, y_tensor, 1)
    return y_one_hot


def train_on_rollout(actions, rewards, done, logits, state_values, hidden_states,
                        gcn, opt, gamma=0.99, alpha = 0.7):

    actions = torch.tensor(np.array(actions).swapaxes(0, 1), dtype=torch.int64)  # shape: [batch_size, time]
    rewards = torch.tensor(np.array(rewards).swapaxes(0, 1), dtype=torch.float32)  # shape: [batch_size, time]
    done = torch.tensor(np.array(done).swapaxes(0, 1), dtype=torch.float32)  # shape: [batch_size, time]
    mask = 1 - done
    logits = torch.stack(logits, dim=1)
    state_values = torch.stack(state_values, dim=1)
    hidden_states = torch.stack(hidden_states, dim=1)
    total_len = hidden_states.size(0) * hidden_states.size(1)
    adj = torch.eye(total_len)
    rollout_length = rewards.shape[1]

    gcn_phi = torch.exp(gcn(hidden_states.reshape(total_len, hidden_states.size(2) ), adj))
    gcn_phi = gcn_phi[:,1].reshape(hidden_states.size(0), hidden_states.size(1), 1).detach()

    probas = F.softmax(logits, dim=2)
    logprobas = F.log_softmax(logits, dim=2)

    # select log-probabilities for chosen actions, log pi(a_i|s_i)
    actions_one_hot = to_one_hot(actions, n_actions).view(
        actions.shape[0], actions.shape[1], n_actions)

    logprobas_for_actions = torch.sum(logprobas * actions_one_hot, dim=-1)

    J_hat = 0  # policy objective as in the formula for J_hat
    value_loss = 0
    adv_phi = 0
    cumulative_returns = state_values[:, -1].squeeze().detach()

    for t in reversed(range(rollout_length)):
        r_t = rewards[:, t]                             # current rewards
        # current state values
        V_t = state_values[:, t].squeeze()
        V_next = state_values[:, t + 1].squeeze().detach()           # next state values
        mask_t = mask[:, t]

        # log-probability of a_t in s_t
        logpi_a_s_t = logprobas_for_actions[:, t].squeeze()

        # update G_t = r_t + gamma * G_{t+1} as we did in week6 reinforce
        cumulative_returns = G_t = r_t + mask_t * gamma * cumulative_returns

        # Compute temporal difference error (MSE for V(s))

        value_loss += torch.mean((r_t + mask_t * gamma * V_next - V_t)**2)


        # compute advantage A(s_t, a_t) using cumulative returns and V(s_t) as baseline
        advantage = cumulative_returns - V_t

        if t == (rollout_length-1):
            delta = r_t + gamma * V_t - gcn_phi[:, t].squeeze()
        else:
            delta = r_t + mask_t * gamma * gcn_phi[:, t+1].squeeze()  - gcn_phi[:, t].squeeze()
        adv_phi = delta + gamma * mask_t * adv_phi

        advantage = ((1-alpha) * adv_phi + alpha * advantage).detach()
        #print(advantage)

        # compute policy pseudo-loss aka -J_hat.
        J_hat += torch.mean(logpi_a_s_t * advantage)

    # regularize with entropy
    entropy_reg = -(logprobas * probas).sum(-1).mean()

    # add-up three loss components and average over time
    loss = -J_hat / rollout_length +\
        value_loss / rollout_length +\
           -0.01 * entropy_reg

    # Gradient descent step
    opt.zero_grad()
    loss.backward()
    opt.step()


    return loss.data.numpy()

def evaluate(agent, env, n_games=1):
    """Plays an entire game start to end, returns session rewards."""

    game_rewards = []
    for _ in range(n_games):
        # initial observation and memory
        observation = env.reset()
        prev_memories = agent.get_initial_state(1)

        total_reward = 0
        while True:
            new_memories, readouts, h = agent.step(
                prev_memories, observation[None, ...])
            action = agent.sample_actions(readouts)

            observation, reward, done, info = env.step(action[0])

            total_reward += reward
            prev_memories = new_memories
            if done:
                break

        game_rewards.append(total_reward)
    return game_rewards

if __name__ == "__main__":

    rewards_history = []

    env = make_env(CFG, SCALE, crop = crop_func, show_window=False)
    obs_shape = env.observation_space.shape
    n_actions = env.a_size

    agent = AC_Net(obs_shape, n_actions, lstm_size=LSTM_SIZE)
    opt = torch.optim.Adam(agent.parameters(), lr=1e-5)
    pool = EnvPool(make_env, NUM_PROCESSES)

    gcn_model = GCN(nfeat=agent.feature_size(), nhid=64)
    gcn_optimizer = torch.optim.Adam(gcn_model.parameters(),
                           lr=1e-3, weight_decay=0e-4)

    gcn_loss = nn.NLLLoss()
    gcn_states = [[] for _ in range(NUM_PROCESSES)]
    Gs = [nx.Graph() for _ in range(NUM_PROCESSES)]
    node_ptrs = [ 0 for _ in range(NUM_PROCESSES)]
    rew_states = [ [] for _ in range(NUM_PROCESSES)]

    obs = pool.prev_observations
    memory = agent.get_initial_state(len(obs))

    for i in range(MAX_EP):

        memory = (memory[0].detach(), memory[1].detach())

        logits = []
        state_values = []
        hidden = []
        rewards = []
        actions = []
        dones = []

        for t in range(NUM_STEPS):
            memory, (logits_t, state_values_t), hidden_states = agent.step(memory, obs)
            action = agent.sample_actions((logits_t, state_values_t))
            obs, reward, done = pool.step(action)
            for idx,(hid, eps_done) in enumerate(zip(hidden_states,done)):
                if GCN_ALPHA < 1.0:
                    gcn_states[idx].append(hid)
                    node_ptrs[idx]+=1
                    if not eps_done:
                        Gs[idx].add_edge(node_ptrs[idx]-1,node_ptrs[idx])
                    if reward[idx] > 0. or eps_done:
                        #print(idx, reward[idx])
                        rew_states[idx].append([node_ptrs[idx]-1,reward[idx]])
                    if eps_done:
                        adj = nx.adjacency_matrix(Gs[idx]) if len(Gs[idx].nodes)\
                                        else sp.csr_matrix(np.eye(1,dtype='int64'))
                        update_graph(gcn_model,gcn_optimizer,
                            torch.stack(gcn_states[idx]),adj,
                            rew_states[idx], gcn_loss)

                        gcn_states[idx]=[]
                        Gs[idx]=nx.Graph()
                        node_ptrs[idx]=0
                        rew_states[idx] =[]

            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            logits.append(logits_t)
            state_values.append(state_values_t)
            hidden.append(hidden_states)

        _, (logits_t, next_value), hidden_states = agent.step(memory, obs)
        state_values.append(next_value)

        l = train_on_rollout(actions, rewards, dones, logits, state_values,
                                hidden, gcn_model, opt, gamma=0.99, alpha=GCN_ALPHA)
        if i % 1000 == 0:
            rewards_history.append(np.mean(evaluate(agent, env, n_games=1)))
            print(rewards_history[-1])
        #break
