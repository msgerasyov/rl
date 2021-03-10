import pickle
import numpy as np

if __name__ == '__main__':
    with open('dqn_rewards.pkl', 'rb') as f:
        rewards = pickle.load(f)
        print(rewards)
