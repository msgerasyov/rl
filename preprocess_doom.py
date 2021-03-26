from vizdoom import *
import random
import numpy as np
import time
import cv2
from gym.spaces import Box

def preprocess_frame(frame):
    img_size = (84, 84)
    img = frame[10:-10,30:-30]
    img = cv2.resize(img / 255, img_size, interpolation=cv2.INTER_LINEAR)
    img = img[None, :, :]
    return img

class DoomEnv():
    def __init__(self, scenario, reward_scale, preprocess_frame):
        self.preprocess_frame = preprocess_frame
        self.observation_space = Box(0.0, 1.0, (1, 84, 84))
        self.a_size = 3
        self.actions = np.identity(self.a_size, dtype=bool).tolist()
        self.reward_scale = reward_scale
        self.game = DoomGame()
        self.game.set_doom_scenario_path(scenario) #This corresponds to the simple task we will pose our agent
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
            next_state = self.preprocess_frame(self.game.get_state().screen_buffer)
        return next_state, reward * self.reward_scale, done, None

    def reset(self):
        self.game.new_episode()
        return self.preprocess_frame(self.game.get_state().screen_buffer)

def make_env(scenario, reward_scale, crop = lambda img: img,
            preprocess_frame=preprocess_frame):
    return DoomEnv(scenario, reward_scale, preprocess_frame)

if __name__ == '__main__':
    episodes = 10
    env = DoomEnv("basic.wad", 1, preprocess_frame=preprocess_frame)
    for i in range(episodes):
        total_reward = 0
        env.reset()
        while True:
            next_s, reward, done, _ = env.step(random.choice(range(3)))
            print("\treward:",reward)
            total_reward += reward
            time.sleep(0.02)
            if done:
                break

        print("Result:", total_reward)
        time.sleep(2)
