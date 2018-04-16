import time
import logging

import numpy as np

from gym import Wrapper, ActionWrapper, ObservationWrapper
import tensorflow as tf

from gym.spaces import Discrete, Box

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Filter:
    def after_reset(self, observation):
        return observation

    def after_step(self, observation, reward, done, info):
        return observation, reward, done, info


class ApplyFilter(Wrapper):
    def __init__(self, env, filter_factory, *args, **kwargs):
        super().__init__(env)
        self.filter = filter_factory(*args, **kwargs)

    def _reset(self):
        observation = self.env.reset()
        observation = self.filter.after_reset(observation)
        return observation

    def _step(self, action):
        o, r, d, i = self.env.step(action)
        observation, reward, done, info = self.filter.after_step(o, r, d, i)
        return observation, reward, done, info

    @property
    def action_space(self):
        return self.env.action_space

    @action_space.setter
    def action_space(self, value):
        pass

    @property
    def observation_space(self):
        return self.env.observation_space

    @observation_space.setter
    def observation_space(self, value):
        pass

def Diagnostics(env, *args, **kwargs):
    return ApplyFilter(env, DiagnosticsInfo, *args, **kwargs)

class DiagnosticsInfo(Filter):
    def __init__(self, summary_writer):
        super().__init__()

        self._episode_time = time.time()
        self._last_time = time.time()
        self._episode_count = 0
        self._episode_reward = 0
        self._episode_length = 0
        self._last_episode_id = -1

        self.summary_writer = summary_writer

    def after_reset(self, observation):
        logger.info('Resetting environment')
        # TODO check this
        self.summary_writer.add_graph(tf.get_default_graph())
        self._episode_reward = 0
        self._episode_length = 0
        return observation

    def after_step(self, observation, reward, done, info):
        to_log = {}
        if self._episode_length == 0:
            self._episode_time = time.time()

        self._episode_length += 1
        self._episode_reward += reward

        if done:
            self._episode_count += 1
            logger.info('Episode terminating: episode_reward=%s episode_length=%s', self._episode_reward, self._episode_length)
            total_time = time.time() - self._episode_time
            to_log["global/episode_reward"] = self._episode_reward
            to_log["global/episode_length"] = self._episode_length
            to_log["global/episode_time"] = total_time
            to_log["global/reward_per_time"] = self._episode_reward / total_time
            to_log["diagnostics/fps"] = total_time / self._episode_length

            # Writing a summary for tensorboard
            summary = tf.Summary()
            for k, v in to_log.items():
                summary.value.add(tag=k, simple_value=float(v))

            self.summary_writer.add_summary(summary, self._episode_count)
            self.summary_writer.flush()

        return observation, reward, done, to_log


class GreyChannelWrapper(ObservationWrapper):
    def __init__(self, env):
        super().__init__(env)

        # TODO hardcoded 160
        o = np.zeros((160, 160, 1))
        self.observation_space = Box(o, o + 1)

    @property
    def observation_space(self):
        if self.env.observation_space is not None:
            o = self.env.observation_space
            low = np.expand_dims(o.low, -1)
            high = np.expand_dims(o.high, -1)
            return Box(low, high)
        else:
            return None


    @observation_space.setter
    def observation_space(self, value):
        pass

    @property
    def action_space(self):
        return self.env.action_space

    @action_space.setter
    def action_space(self, value):
        pass

    def _observation(self, observation):
        # So that this becomes an image with one channel
        return observation[:, :, np.newaxis]


class ActionBox2Discrete(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def action_space(self):
        a_s = self.env.action_space
        if a_s is not None:
            high = a_s.high
            # Flattening the action space
            return Discrete(int(np.prod(high)))
        else:
            return None

    @action_space.setter
    def action_space(self, value):
        pass

    @property
    def observation_space(self):
        return self.env.observation_space

    @observation_space.setter
    def observation_space(self, value):
        pass

    def action(self, action):
        b = self.env.action_space
        x = action // b.high[0]
        y = action % b.high[1]
        a = np.array([x, y])
        return a


class MovingActionWrapper(ActionWrapper):
    def __init__(self, env):
        super().__init__(env)

    @property
    def width(self):
        return self.unwrapped.page_size[0]

    @property
    def height(self):
        return self.unwrapped.page_size[1]

    @property
    def mouse_pos(self):
        return self.unwrapped.mouse_start

    def action(self, action):
        current_idx = self.mouse_pos[0] * self.width + self.mouse_pos[1]

        if action == 0:
            next_idx = current_idx + 1
        elif action == 1:
            next_idx = current_idx - 1
        elif action == 2:
            next_idx = current_idx + self.width
        elif action == 3:
            next_idx = current_idx - self.width

        if self.env.action_space.contains(next_idx):
            return next_idx
        else:
            return current_idx


    @property
    def action_space(self):
        return Discrete(4)

    @action_space.setter
    def action_space(self, value):
        pass

    @property
    def observation_space(self):
        return self.env.observation_space

    @observation_space.setter
    def observation_space(self, value):
        pass

