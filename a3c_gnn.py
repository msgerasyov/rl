# -*- coding: utf-8 -*-

import numpy as np
import sys
import time
import os
import pickle
import math

os.environ["OMP_NUM_THREADS"] = "1"
MAX_EP = 1000000
N_WORKERS = 16
EVAL_FREQ = 5000
LSTM_SIZE = 128
MAX_GRAD = 40
SCALE = 1
GCN_ALPHA = 0.7
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
from pygcn import update_graph, compute_graph_loss
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
        return torch.multinomial(probs, 1)[:, 0].data.numpy()

    def step(self, prev_state, obs_t):
        """ like forward, but obs_t is a numpy array """
        obs_t = torch.tensor(np.asarray(obs_t), dtype=torch.float32)
        (h, c), (l, s), hid = self.forward(prev_state, obs_t)
        return (h, c), (l, s), hid.detach()

    def compute_rollout_loss(self, actions, rewards, logits, state_values,
                            hidden_states, gcn, alpha=0.7, gamma=0.99):

        actions = torch.tensor(np.array(actions), dtype=torch.int64)  # shape: [time]
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)  # shape: [time]
        rollout_length = rewards.shape[0]

        hidden_states = torch.stack(hidden_states, dim=1)
        hidden_states = hidden_states.view(hidden_states.size(1), -1)
        logits = torch.stack(logits, dim=1)
        logits = logits.view(logits.size(1), -1)
        state_values = torch.stack(state_values, dim=1)
        state_values = torch.squeeze(state_values)

        adj = torch.eye(hidden_states.size(0))
        if hidden_states.size(0) > 1:
            gcn_phi = torch.exp(gcn(hidden_states, adj))
            gcn_phi = gcn_phi[:,1].detach()
        else:
            gcn_phi = torch.zeros(rewards.shape)

        probas = F.softmax(logits, dim=1)
        logprobas = F.log_softmax(logits, dim=1)
        # select log-probabilities for chosen actions, log pi(a_i|s_i)

        logprobas_for_actions = logprobas[range(len(actions)), actions]

        J_hat = 0  # policy objective as in the formula for J_hat

        value_loss = 0
        adv_phi = 0

        cumulative_returns = state_values[-1].detach()

        for t in reversed(range(rollout_length)):
            r_t = rewards[t]                                # current rewards
            V_t = state_values[t]
            V_next = state_values[t+1].detach()           # next state values
            logpi_a_s_t = logprobas_for_actions[t]

            cumulative_returns = G_t = r_t + gamma * cumulative_returns

            # Compute temporal difference error (MSE for V(s))
            value_loss += (r_t + gamma * V_next - V_t)**2

            # compute advantage A(s_t, a_t) using cumulative returns and V(s_t) as baseline
            advantage = cumulative_returns - V_t

            if t == (rollout_length-1):
                delta = r_t + gamma * V_t - gcn_phi[t]
            else:
                delta = r_t + gamma * gcn_phi[t+1] - gcn_phi[t]

            adv_phi = delta + gamma * adv_phi

            adv_comb = ((1-alpha) * adv_phi + alpha * advantage).detach()

            # compute policy pseudo-loss aka -J_hat.
            J_hat += logpi_a_s_t * adv_comb

        # regularize with entropy
        entropy_reg = -(logprobas * probas).sum(-1).mean()

        # add-up three loss components and average over time
        loss = -J_hat / rollout_length +\
            value_loss / rollout_length +\
              -0.01 * entropy_reg

        return loss

class SharedAdam(torch.optim.Adam):
    def __init__(self, params, lr=1e-4):
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
    def __init__(self, master, opt, gcn, gcn_opt, process_id):
        super(Worker, self).__init__()
        self.process_id = process_id
        self.opt = opt
        self.master = master
        self.gcn = gcn
        self.gcn_opt = gcn_opt
        obs_shape = self.master.obs_shape
        n_actions = self.master.n_actions
        self.lnet = AC_Net(obs_shape, n_actions, lstm_size=LSTM_SIZE)
        self.lgcn = GCN(nfeat=self.master.feature_size(), nhid=64)
        self.gcn_loss = nn.NLLLoss()
        self.gcn_states = []
        self.Gs = nx.Graph()
        self.node_ptr = 0
        self.rew_states = []
        self.memories = self.lnet.get_initial_state(1)

    def _sync_local_with_global(self):
        self.lnet.load_state_dict(self.master.state_dict())
        self.lgcn.load_state_dict(self.gcn.state_dict())

    def work(self, n_iter):
        self.memories = (self.memories[0].detach(),
                            self.memories[1].detach())

        actions = []
        rewards = []
        logits = []
        state_values = []
        hidden_states = []

        for _ in range(n_iter):
            self.memories, (logits_t, value_t), hid = self.lnet.step(
                self.memories, self.observation[None, ...])
            action = self.lnet.sample_actions((logits_t, value_t))

            self.observation, reward, done, _ = self.env.step(action[0])

            actions.append(action[0])
            rewards.append(reward)
            logits.append(logits_t)
            state_values.append(value_t)
            hidden_states.append(hid)

            self.gcn_states.append(hid.squeeze())
            self.node_ptr += 1
            if not done:
                self.Gs.add_edge(self.node_ptr-1, self.node_ptr)
            if reward > 0. or done:
                self.rew_states.append([self.node_ptr-1, reward])
            if done:
                if len(self.gcn_states) > 1:
                    adj = nx.adjacency_matrix(self.Gs) if len(self.Gs.nodes)\
                                    else sp.csr_matrix(np.eye(1,dtype='int64'))

                    graph_loss = compute_graph_loss(self.lgcn,
                        torch.stack(self.gcn_states), adj,
                        self.rew_states, self.gcn_loss)

                    graph_loss.backward()

                    self.gcn_opt.zero_grad()
                    for lp, mp in zip(self.lgcn.parameters(), self.gcn.parameters()):
                        mp._grad = lp.grad
                    self.gcn_opt.step()

                self.gcn_states=[]
                self.Gs=nx.Graph()
                self.node_ptr=0
                self.rew_states = []

                self.observation = self.env.reset()
                self.memories = self.lnet.get_initial_state(1)
                break

        _, (logits_t, value_t), hid = self.lnet.step(
            self.memories, self.observation[None, ...])

        state_values.append(value_t * (1 - done))

        return actions, rewards, logits, state_values, hidden_states

    def train(self, actions, rewards, logits, state_values, hidden_states, gamma=0.99):
        loss = self.lnet.compute_rollout_loss(actions, rewards, logits,
                state_values, hidden_states, self.lgcn, GCN_ALPHA, gamma)
        self.opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.lnet.parameters(), MAX_GRAD)
        for lp, mp in zip(self.lnet.parameters(), self.master.parameters()):
            mp._grad = lp.grad
        self.opt.step()

    def run(self):
        self.env = make_env(CFG, SCALE, crop=crop_func)
        self.observation = self.env.reset()
        time.sleep(int(np.random.rand() * (self.process_id + 5)))
        while self.master.train_step.value < self.master.steps:
            self._sync_local_with_global()
            actions, rewards, logits, state_values, hidden_states = self.work(20)
            self.train(actions, rewards, logits, state_values, hidden_states)
            self.master.train_step.value += 1

class Tester(mp.Process):
    def __init__(self, master, gcn, process_id, eval_freq, n_games=1):
        super(Tester, self).__init__()
        self.gcn = gcn
        self.process_id = process_id
        self.master = master
        obs_shape = self.master.obs_shape
        n_actions = self.master.n_actions
        self.lnet = AC_Net(obs_shape, n_actions, lstm_size=LSTM_SIZE)
        self.rewards = []
        self.entropy = []
        self.n_games = n_games
        self.eval_freq = eval_freq
        self.step = 0

    def _sync_local_with_global(self):
        self.lnet.load_state_dict(self.master.state_dict())

    def evaluate(self):
        """Plays an entire game start to end, returns session rewards."""

        game_rewards = []
        logits = []
        for _ in range(self.n_games):
            # initial observation and memory
            observation = self.env.reset()
            prev_memories = self.lnet.get_initial_state(1)

            total_reward = 0
            while True:
                new_memories, (logits_t, value_t), _ = self.lnet.step(
                    prev_memories, observation[None, ...])
                action = self.lnet.sample_actions((logits_t, value_t))

                observation, reward, done, info = self.env.step(action[0])

                logits.append(logits_t)
                total_reward += reward
                prev_memories = new_memories
                if done:
                    break

            game_rewards.append(total_reward)

        logits = torch.stack(logits, dim=1)
        logits = logits.view(logits.size(1), -1)
        probas = F.softmax(logits, dim=1)
        logprobas = F.log_softmax(logits, dim=1)
        entropy_reg = -(logprobas * probas).sum(-1).mean()
        return np.mean(game_rewards), entropy_reg.item()

    def run(self):
        self.env = make_env(CFG, 1, crop=crop_func)
        while self.master.train_step.value < self.master.steps:
            if self.master.train_step.value >= self.step:
                eval_step = self.master.train_step.value
                self._sync_local_with_global()
                torch.save(self.lnet.state_dict(),
                            'a3c-{0}.weights'.format(CFG[0:3]))
                torch.save(self.gcn.state_dict(),
                            'gcn-{0}.weights'.format(CFG[0:3]))
                mean_reward, entropy_reg = self.evaluate()
                print(eval_step, "reward:", mean_reward, "entropy:", entropy_reg)
                self.rewards.append((eval_step, mean_reward))
                self.entropy.append((eval_step, entropy_reg))
                self.step += self.eval_freq
                with open('a3c-{0}_rewards.pkl'.format(CFG[0:3]), 'wb') as f:
                    pickle.dump(self.rewards, f)
                with open('a3c-{0}_entropy.pkl'.format(CFG[0:3]), 'wb') as f:
                    pickle.dump(self.entropy, f)


if __name__ == "__main__":

    env = make_env(CFG, SCALE, crop = crop_func)
    obs_shape = env.observation_space.shape
    n_actions = env.a_size

    master = AC_Net(obs_shape, n_actions, lstm_size=LSTM_SIZE)
    master.steps = MAX_EP
    master.train_step = mp.Value('l', 0)
    if os.path.exists('a3c-{0}.weights'.format(CFG[0:3])):
        master.load_state_dict(torch.load('a3c-{0}.weights'.format(CFG[0:3])))
        print('Successfully loaded weights')
    master.share_memory()
    shared_opt = SharedAdam(master.parameters())
    gcn_model = GCN(nfeat=master.feature_size(), nhid=64)
    gcn_opt = SharedAdam(gcn_model.parameters(), lr=1e-3)
    if os.path.exists('gcn-{0}.weights'.format(CFG[0:3])):
        gcn_model.load_state_dict(torch.load('gcn-{0}.weights'.format(CFG[0:3])))
        print('Successfully loaded weights')
    gcn_model.share_memory()
    print('Workers count:', N_WORKERS)

    # parallel training
    processes = [Worker(master, shared_opt, gcn_model, gcn_opt, i) for i in range(N_WORKERS)]
    processes.append(Tester(master, gcn_model, len(processes), EVAL_FREQ, n_games=3))
    for p in processes:
        p.start()
    for p in processes:
        p.join()


