import gym

import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

from rl.agents.dqn import DQNAgent
from rl.policy import EpsGreedyQPolicy
from rl.memory import SequentialMemory


env = gym.make('SpaceInvaders-v0')

tensorboard = TensorBoard(log_dir='/home/vivnp/Python Projects/Gama/log', histogram_freq=0,
                          write_graph=True, write_images=False)

np.random.seed(123)
env.seed(123)
nb_actions = env.action_space.n
model = Sequential()
model.add(Flatten(input_shape=(1,) + env.observation_space.shape))
model.add(Dense(16))
model.add(Activation('relu'))
model.add(Dense(nb_actions))
model.add(Activation('linear'))

policy = EpsGreedyQPolicy()
memory = SequentialMemory(limit=200000, window_length=1)
dqn = DQNAgent(model=model, nb_actions=nb_actions, memory=memory, nb_steps_warmup=50,
               target_model_update=1e-2, policy=policy)
dqn.compile(Adam(lr=1e-3), metrics=['mae'])

dqn.fit(env, nb_steps=5000, visualize=False, callbacks=[tensorboard], verbose=2)
dqn.save_weights("./spaceinvaders.model", overwrite=True)