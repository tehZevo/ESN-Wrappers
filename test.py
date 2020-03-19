import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO2
from esn_wrapper import ESNWrapper

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import tensorflow as tf
tf.get_logger().setLevel('INFO')
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)  # or any {DEBUG, INFO, WARN, ERROR, FATAL}

esn_size = 16
reset = False
env = gym.make("CartPole-v0")
env = ESNWrapper(env, esn_size, reset_on_done=reset)

# multiprocess environment
env = DummyVecEnv([lambda: env])

model = PPO2(MlpPolicy, env, verbose=1, tensorboard_log="./logs/esn-{}-{}/".format(esn_size, reset))
model.learn(total_timesteps=50000)
