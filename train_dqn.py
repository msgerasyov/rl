from dqn import DQNAgent, ReplayBuffer, play_and_record, compute_td_loss, evaluate
from preprocess_atari import make_env

from tqdm import trange
import numpy as np
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import psutil

def is_enough_ram(min_available_gb=0.1):
    mem = psutil.virtual_memory()
    return mem.available >= min_available_gb * (1024 ** 3)

def linear_decay(init_val, final_val, cur_step, total_steps):
    if cur_step >= total_steps:
        return final_val
    return (init_val * (total_steps - cur_step) +
            final_val * cur_step) / total_steps

seed = 42

timesteps_per_epoch = 1
batch_size = 16
total_steps = 2 * 10**6
decay_steps = 10**6

init_epsilon = 0.9
final_epsilon = 0.1

loss_freq = 50
refresh_target_network_freq = 5000
eval_freq = 5000

max_grad_norm = 50

n_lives = 3

mean_rw_history = []
td_loss_history = []
grad_norm_history = []
initial_state_v_history = []
step = 0

if __name__ == '__main__':

    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        device  = torch.device('cuda:1')
    else:
        device = torch.device('cpu')
    print(device)
    env = make_env(seed)
    state_shape = env.observation_space.shape
    n_actions = env.action_space.n
    state = env.reset()

    agent = DQNAgent(state_shape, n_actions, epsilon=0.9).to(device)
    #agent.load_state_dict(torch.load('dqn.weights'))
    target_network = DQNAgent(state_shape, n_actions).to(device)
    target_network.load_state_dict(agent.state_dict())
    opt = torch.optim.Adam(agent.parameters(), lr=1e-4)
    exp_replay = ReplayBuffer(10**3)

    for i in range(100):
        play_and_record(state, agent, env, exp_replay, n_steps=10**2)
        if len(exp_replay) == 10**3:
            break
    print(len(exp_replay))

    state = env.reset()
    for step in trange(step, total_steps + 1):
       
        agent.epsilon = linear_decay(init_epsilon, final_epsilon, step, decay_steps)

        # play
        _, state = play_and_record(state, agent, env, exp_replay, timesteps_per_epoch)

        # train
        obs_batch, act_batch, reward_batch, next_obs_batch, is_done_batch = exp_replay.sample(batch_size)

        loss = compute_td_loss(obs_batch, act_batch, reward_batch, next_obs_batch,
                               is_done_batch, agent, target_network, device=device)

        loss.backward()
        grad_norm = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
        opt.step()
        opt.zero_grad()

        if step % loss_freq == 0:
            td_loss_history.append(loss.data.cpu().item())
            grad_norm_history.append(grad_norm)

        if step % refresh_target_network_freq == 0:
            # Load agent weights into target_network
            target_network.load_state_dict(agent.state_dict())

        if step % eval_freq == 0:
            torch.save(agent.state_dict(), 'dqn.weights')
            mean_rw_history.append(np.mean(evaluate(
                make_env(clip_rewards=True, seed=step), agent, n_games=3 * n_lives, greedy=True))
            )
            #print(mean_rw_history[-1])

            with open('dqn_rewards.pkl', 'wb') as f:
                pickle.dump(mean_rw_history, f)
