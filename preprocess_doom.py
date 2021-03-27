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
    def __init__(self, cfg, reward_scale, crop):
        self.crop = crop
        self.observation_space = Box(0.0, 1.0, (1, 84, 84))
        self.reward_scale = reward_scale
        self.game = DoomGame()
        self.game.load_config(cfg)
        self.a_size = self.game.get_available_buttons_size()
        self.actions = np.identity(self.a_size, dtype=bool).tolist()
        #self.game.set_doom_scenario_path(scenario)
        self.game.set_doom_map("map01")
        self.game.set_screen_resolution(ScreenResolution.RES_320X240)
        self.game.set_screen_format(ScreenFormat.GRAY8)
        self.game.set_render_hud(False)
        self.game.set_render_crosshair(False)
        self.game.set_render_weapon(True)
        #self.game.set_render_decals(False)
        #self.game.set_render_particles(False)
        #self.game.add_available_button(Button.MOVE_LEFT)
        #self.game.add_available_button(Button.MOVE_RIGHT)
        #self.game.add_available_button(Button.ATTACK)
        #self.game.add_available_game_variable(GameVariable.AMMO2)
        #self.game.add_available_game_variable(GameVariable.POSITION_X)
        #self.game.add_available_game_variable(GameVariable.POSITION_Y)
        #self.game.set_episode_timeout(300)
        #self.game.set_episode_start_time(10)
        self.game.set_window_visible(False)
        self.game.set_sound_enabled(False)
        #self.game.set_living_reward(-1)
        self.game.set_mode(Mode.PLAYER)
        self.game.init()
        self.health = self.game.get_game_variable(GameVariable.HEALTH)
        self.armor = self.game.get_game_variable(GameVariable.ARMOR)

    def step(self, action):
        reward = self.game.make_action(self.actions[action])
        new_health = self.game.get_game_variable(GameVariable.HEALTH)
        new_armor = self.game.get_game_variable(GameVariable.ARMOR)
        reward += (new_health - self.health) * 0.01
        reward += (new_armor - self.armor) * 0.01
        self.health = new_health
        self.armor = new_armor
        done = self.game.is_episode_finished()
        if done:
            next_state = None
        else:
            next_state = preprocess_frame(self.game.get_state().screen_buffer, self.crop)
        return next_state, reward * self.reward_scale, done, None



    def reset(self):
        self.game.new_episode()
        return preprocess_frame(self.game.get_state().screen_buffer, self.crop)

def make_env(cfg, reward_scale, crop = lambda img: img):
    return DoomEnv(cfg, reward_scale, crop)
