from vizdoom import *
import random
import numpy as np
import time
import cv2
from gym.spaces import Box

def preprocess_frame(frame, crop):
    img_size = (84, 84)
    img = crop(frame)
    img = cv2.resize(img / 255, img_size, interpolation=cv2.INTER_LINEAR)
    img = img[None, :, :]
    return img

class DoomEnv():
    def __init__(self, scenario, reward_scale, crop):
        self.crop = crop
        self.observation_space = Box(0.0, 1.0, (1, 84, 84))
        self.a_size = 3
        self.actions = np.identity(self.a_size, dtype=bool).tolist()
        self.reward_scale = reward_scale
        self.game = DoomGame()
        self.game.set_doom_scenario_path(scenario) 
        self.game.set_doom_map("map01")
        self.game.set_screen_resolution(ScreenResolution.RES_160X120)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        self.game.set_render_decals(False)
        self.game.set_render_particles(False)
        self.game.add_available_button(Button.MOVE_LEFT)
        self.game.add_available_button(Button.MOVE_RIGHT)
        self.game.add_available_button(Button.ATTACK)
        self.game.add_available_game_variable(GameVariable.AMMO2)
        self.game.add_available_game_variable(GameVariable.POSITION_X)
        self.game.add_available_game_variable(GameVariable.POSITION_Y)
        self.game.set_episode_timeout(300)
        self.game.set_episode_start_time(10)
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        self.game.set_living_reward(-1)
        self.game.set_mode(Mode.PLAYER)
        self.game.init()

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        done = self.game.is_episode_finished()
        if done:
            next_state = None
        else:
            next_state = preprocess_frame(self.game.get_state().screen_buffer, self.crop)
        return next_state, reward * self.reward_scale, done, None

    def reset(self):
        self.game.new_episode()
        return preprocess_frame(self.game.get_state().screen_buffer, self.crop)

def make_env(scenario, reward_scale, crop = lambda img: img):
    return DoomEnv(scenario, reward_scale, crop)
