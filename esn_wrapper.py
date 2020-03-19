import gym
from gym import spaces
from gym import Wrapper
import numpy as np

def xavier_normal(ni, no):
  std = np.sqrt(1 / (ni + no))
  return np.random.normal(size=[ni, no]) * std

def sparse(shape, density):
  return (np.random.random(shape) < density).astype("float32")

class ESNWrapper(Wrapper):
  def __init__(self, env, size, density=0.1, reset_on_done=True):
    super().__init__(env)
    self.observation_space = spaces.Box(shape=[size], low=-1, high=1)
    self.size = size
    self.in_size = self.env.observation_space.shape[0]
    self.in_weights = xavier_normal(self.in_size, size) * sparse([self.in_size, size], density)
    self.recur_weights = xavier_normal(size, size) * sparse([size, size], density)
    self.bias_weights = xavier_normal(1, size) * sparse([1, size], density)
    self.reset_on_done = reset_on_done

    self.state = np.zeros(self.size)

  def update_esn(self, obs):
    a = np.dot(obs, self.in_weights)
    b = np.dot(self.state, self.recur_weights)
    c = np.dot([1], self.bias_weights)
    self.state = a + b + c

  def step(self, action):
    obs, reward, done, info = self.env.step(action)

    self.update_esn(obs)

    return self.state, reward, done, info

  def reset(self, **kwargs):
    if self.reset_on_done:
      self.state = np.zeros(self.size)

    obs = self.env.reset(**kwargs)
    self.update_esn(obs)

    return self.state
