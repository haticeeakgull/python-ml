import gym
import random
import numpy as np


environment = gym.make("FrozenLake-v1",is_slippery=False,render_mode="ansi")
environment.reset()
nb_states=environment.observation_space.n
nb_action =environment.action_space.n
qtable= np.zeros((nb_states,nb_action))

print("q table:")
print(qtable)

action = environment.action_space.sample()
new_state , reward ,done , info, _= environment.step(action)