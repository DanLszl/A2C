from envs.base_env import BaseEnv
from envs.monitor import Monitor
from envs.wrappers import ActionBox2Discrete, MovingActionWrapper, GreyChannelWrapper

import click_button

import gym

class WobEnv(BaseEnv):
    def __init__(self, env_name, id, seed):
        super().__init__(env_name, id)
        self.seed = seed
        self.make()

    def make(self):
        env = gym.make(self.env_name)
        self.gym_env = env

        env = Monitor(env, self.rank)
        self.monitor = env

        env.seed(self.seed + self.rank)


        env = ActionBox2Discrete(env)
        env = MovingActionWrapper(env)
        env = GreyChannelWrapper(env)

        self.env = env

        width = 80
        height = 80
        button_width = 20
        button_height = 10
        mouse_width = 10
        mouse_height = 10
        max_steps = 100
        randomize_mouse = 'xy'
        randomize_button = 'xy'
        decrease_reward_with_time = True
        penalty_if_unsolved = False


        env.configure(width=width, height=height,
                      button_width=button_width, button_height=button_height,
                      mouse_width=mouse_width, mouse_height=mouse_height,
                      max_steps=max_steps,
                      randomize_mouse=randomize_mouse,
                      randomize_button=randomize_button,
                      decrease_reward_with_time=decrease_reward_with_time,
                      penalty_if_unsolved=penalty_if_unsolved)

        return env

    def step(self, data):
        o, r, d, i = self.env.step(data)
        return o, r, d, i

    def reset(self):
        return self.env.reset()

    def get_action_space(self):
        return self.env.action_space

    def get_observation_space(self):
        return self.env.observation_space

    def render(self):
        self.gym_env.render()