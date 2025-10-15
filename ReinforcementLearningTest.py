import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt


environment = gym.make("FrozenLake-v1",is_slippery=False,render_mode="ansi")
environment.reset()
nb_states=environment.observation_space.n
nb_action =environment.action_space.n
qtable= np.zeros((nb_states,nb_action))

print("q table:")
print(qtable)

# action = environment.action_space.sample()
# new_state , reward ,done , info, _= environment.step(action)

episodes= 1000
alpha=0.5       #learning rate
gamma= 0.9      #discount rate
nb_success=0

outcomes=[]

for _ in range(episodes):
    state,_=environment.reset()
    done=False
    
    outcomes.append("Failure")

    while not done:
        if np.max(qtable[state])>0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
        new_state , reward ,done , info, _= environment.step(action)

        qtable[state,action]=qtable[state,action]+alpha*(reward+gamma*np.max(qtable[new_state])-qtable[state,action])
        state=new_state

        if reward:
            outcomes[-1]="Success"
print("Qtable After Training:")
print(qtable)


for _ in range(episodes):
    state,_=environment.reset()
    done=False
    


    while not done:
        if np.max(qtable[state])>0:
            action = np.argmax(qtable[state])
        else:
            action = environment.action_space.sample()
        new_state , reward ,done , info, _= environment.step(action)

        state=new_state
        nb_success+=reward

print("success rate : " ,100*nb_success/episodes)

        