import random
import numpy as np
import torch
import cv2

import gym
from gym.core import Wrapper
from gym.spaces import Box

class PreprocessAtari(Wrapper):
    def __init__(self, env, height=42, width=42, color=False,
                 crop=lambda img: img, n_frames=4, dim_order='pytorch', reward_scale=1):
        """A gym wrapper that reshapes, crops and scales image into the desired shapes"""
        super(PreprocessAtari, self).__init__(env)
        self.img_size = (height, width)
        self.crop = crop
        self.color = color
        self.dim_order = dim_order

        self.reward_scale = reward_scale
        n_channels = (3 * n_frames) if color else n_frames

        obs_shape = {
            'theano': (n_channels, height, width),
            'pytorch': (n_channels, height, width),
            'tensorflow': (height, width, n_channels),
        }[dim_order]

        self.observation_space = Box(0.0, 1.0, obs_shape)
        self.framebuffer = np.zeros(obs_shape, 'float32')

    def reset(self):
        """Resets the game, returns initial frames"""
        self.framebuffer = np.zeros_like(self.framebuffer)
        self.update_buffer(self.env.reset())
        return self.framebuffer

    def step(self, action):
        """Plays the game for 1 step, returns frame buffer"""
        new_img, r, done, info = self.env.step(action)
        self.update_buffer(new_img)

        return self.framebuffer, r * self.reward_scale, done, info

    def update_buffer(self, img):
        img = self.preproc_image(img)
        offset = 3 if self.color else 1
        if self.dim_order == 'tensorflow':
            axis = -1
            cropped_framebuffer = self.framebuffer[:, :, :-offset]
        else:
            axis = 0
            cropped_framebuffer = self.framebuffer[:-offset, :, :]
        self.framebuffer = np.concatenate([img, cropped_framebuffer], axis=axis)

    def preproc_image(self, img):
        """what happens to the observation"""
        img = self.crop(img)
        img = cv2.resize(img / 255, self.img_size, interpolation=cv2.INTER_LINEAR)
        if not self.color:
            img = img.mean(-1, keepdims=True)
        if self.dim_order != 'tensorflow':
            img = img.transpose([2, 0, 1])  # [h, w, c] to [c, h, w]
        return img

class ClipRewardEnv(gym.RewardWrapper):
    def __init__(self, env):
        gym.RewardWrapper.__init__(self, env)

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)


class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so it's important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
        self.lives = lives
        return obs, reward, done, info

    def reset(self, **kwargs):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset(**kwargs)
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs


def make_env(env_name, crop = lambda img: img, n_frames=1, clip_rewards=False, seed=None):
    env = gym.make(env_name)  # create raw env
    if seed is not None:
        env.seed(seed)

    #env = EpisodicLifeEnv(env)

    if clip_rewards:
        env = ClipRewardEnv(env)

    env = PreprocessAtari(env, height=80, width=80, crop=crop, n_frames=n_frames)

    return env
