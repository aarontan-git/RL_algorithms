import numpy as np
from GridWorld import *
# display output
from random import uniform
import time
from IPython.display import display, clear_output

#initialization of Grid
grid = Gridworld(5)
#Q matrix of zeros
Q_values = np.zeros((grid.size*grid.size, len(actions)))
# initialize other parameters
epsilon = 0.2
lr = 0.1
gamma = 0.99
# Define random initial policy

def choose_action(state):
	action=0
	if np.random.uniform(0, 1) < epsilon:
		action = env.action_space.sample()
	else:
		action = np.argmax(Q[state, :])
	return action

def learn(state, state2, reward, action, action2):
	predict = Q[state, action]
	target = reward + gamma * Q[state2, action2]
	Q[state, action] = Q[state, action] + lr_rate * (target - predict)

def choose_action(state):
	 # choose an action type: explore or exploit
    action_type = int(np.random.choice(2, 1, p=[(len(actions)-1)*(epsilon/len(actions)),1-(len(actions)-1)*(epsilon/len(actions))]))
    
    best_action_index = np.argmax(Q_values[state]) # find best action based on Q values
    
    action_index = best_action_index #assigns best action index to the action, this will hold only if action type is exploit

    if action_type = 0: #if action type is explore, then choses a different action than argmax
		while action_index == best_action_index: 
            action_index = np.random.choice(4, 1)

        
    return action_index

for episode in range(500):

    # initialize state (output: [4, 4])
    state = grid.initial_state([4,4])

    # iterate over 200 steps within each episode
    for step in range(200):

        # get state index (output: 24)
        state_index = grid.states.index(state)

        # choose an action based on epsilon-greedy (output: action index ie. 0)
        action_index = choose_action(state_index, epsilon)
        action_vector = actions[action_index] # convert action_index (0) to action_vector ([-1, 0])

        # get the next state and reward after taking the chosen action in the current state
        next_state_vector, reward = grid.transition_reward(state, action_vector)
        next_state_index = grid.states.index(list(next_state_vector))
        next_action_index = choose_action(next_state_index)

        # update Q value
        Q_values[state_index][action_index] = Q_values[state_index][action_index] + lr*(reward + gamma*Q_values[next_state_index][next_action_index]-Q_values[state_index][action_index])

        # set the next state as the current state
        state = list(next_state_vector)
print('training finished')
print(Q_values)

# FIND ARGMAX POLICY 

import pandas as pd
# define column and index
columns=range(grid.size)
index = range(grid.size)
# define dataframe to represent policy table
policy_table = pd.DataFrame(index = index, columns=columns)

# iterate through Q matrix to find best action
# as action name (eg. left, right, up, down)
for state in range(grid.size):
  
    # find the best action at each state
    best_action = np.argmax(Q_values[state])

    # get action name
    if best_action == 0:
        action_name = 'up'
    elif best_action == 1:
        action_name = 'right'
    elif best_action == 2:
        action_name = 'down'
    else:
        action_name = 'left'

    # calculate the row and column coordinate of the current state number
    row = int(state/grid.size)
    column = round((state/grid.size - int(state/grid.size))*grid.size)
            
    # assign action name
    policy_table.loc[row][column] = action_name

print("Policy Table: ")
print(policy_table)
print()
