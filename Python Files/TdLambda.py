# Find the value function of policy
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
# display output
import random
from random import uniform
import time
from IPython.display import display, clear_output
from Gridworld import Gridworld
import pickle

actions = [[-1, 0], [0, 1], [1, 0], [0, -1]] #up, right, down, left = (clockwise from up) 
action_count = len(actions) # total number of actions
gridSize = 5 # create a square grid of gridSize by gridSize
state_count = gridSize*gridSize # total number of states

# define a function that chooses action based on epsilon-greedy
def choose_action(state, epsilon):
    
    # choose an action type: explore or exploit
    action_type = int(np.random.choice(2, 1, p=[epsilon,1-epsilon]))

    # find best action based on Q values
    best_action_index = np.argmax(Q_values[state])

    # pick a random action
    random_action_index = random.choice(range(4))

    # choose an action based on exploit or explore
    if action_type == 0:
        # explore
        # while random action is the same as the best action, pick a new action
        while random_action_index == best_action_index:
            random_action_index = random.choice(range(4))
        action_index = random_action_index
    else:
        # exploit
        # print("exploit")
        action_index = best_action_index
    
    # if Q_values is all zero, randomly pick an action
    if np.count_nonzero(Q_values[state]) == 0:
        action_index = random.randint(0,3)
        
    return action_index

    # define average function
def Average(lst): 
    return sum(lst) / len(lst) 

def generate_episode(steps):

    # set initial state
    state_vector = grid.initial_state()

    # initialize state (with iniitial state), action list and reward list
    state_list = [state_vector]
    action_list = []
    reward_list = []

    # generate an episode
    for i in range(steps):

        # pick an action based on categorical distribution in policy
        action_index = int(np.random.choice(action_count, 1, p=policy[grid.states.index(state_vector)])) 
        action_vector = actions[action_index] # convert the integer index (ie. 0) to action (ie. [-1, 0])

        # get new state and reward after taking action from current state
        new_state_vector, reward = grid.transition_reward(state_vector, action_vector)
        state_vector = list(new_state_vector)

        # save state, action chosen and reward to list
        state_list.append(state_vector)
        action_list.append(action_vector)
        reward_list.append(reward)
        
    return state_list, action_list, reward_list

# create a grid object
grid = Gridworld(5)

# intialize parameters
gamma = 0.99
epsilon = [0.01, 0.1, 0.25]
runs = 20
lamda = 0.9
lr = 0.1
episode_length = 500
window_length = int(episode_length/20)

for eps in epsilon:
    average_test_reward_list = []
    Q_values_list = []

    for run in range(1, runs+1):

        # initialize q values for all state action pairs
        Q_values = np.zeros((state_count, action_count))

        # initialize list for plots
        average_reward_list = []
        cumulative_reward_list = []
        cumulative_reward = 0
        delta_list = []
        episode_test_reward_list=[]

        for episode in range(episode_length):
            
            # initialize delta for eligibility trace
            delta_ = 0
            
            # delta for change in Q values
            delta = 0
            
            # initialize S,A (? should i choose an Action using epsilon-greedy here or just select an Action?)
            state_vector = grid.initial_state()
            state_index = grid.states.index(state_vector)
            
            # initialize  eligibility traces for all state action pairs of all states to 0
            z_values = np.zeros((state_count, action_count))
            
            action_index = choose_action(state_index, eps)
            action_vector = actions[action_index]
            
            reward_list = []
            
            # iteration 200 steps of the episode
            for i in range(200):

                # Take action A, oberserve R, S'
                next_state_vector, reward = grid.transition_reward(state_vector, action_vector)
                next_state_index = grid.states.index(list(next_state_vector))
                
                reward_list.append(reward)

                # Choose A' from S' using policy derived from Q (eg. epsilon-greedy)
                next_action_index = choose_action(next_state_index, eps)
                next_action_vector = actions[next_action_index]

                # update the action-value form of the TD error
                delta_ = reward + gamma*Q_values[next_state_index][next_action_index] - Q_values[state_index][action_index]
                
                # accumulate traces (? big S and big A?)
                z_values[state_index][action_index] +=1
                
                # calculate max Q_value change for plotting max delta
                Q_value = Q_values[state_index][action_index] + lr*delta_*z_values[state_index][action_index]
                delta = max(delta, np.abs(Q_value - Q_values[state_index][action_index]))   
                
                # update Q value
                Q_values[state_index][action_index] = Q_values[state_index][action_index] + lr*delta_*z_values[state_index][action_index]
                
                # update z value
                z_values[state_index][action_index] = gamma*lamda*z_values[state_index][action_index]
                
                # update state and action vector
                state_vector = list(next_state_vector)
                state_index = grid.states.index(state_vector)
                action_vector = list(next_action_vector)
                action_index = next_action_index
            
            # append delta
            delta_list.append(delta)
            
            # append average rewards
            average_reward_list.append(Average(reward_list))
            
            # append cumulative rewards
            cumulative_reward = cumulative_reward + sum(reward_list)
            cumulative_reward_list.append(cumulative_reward)
            
            # initialize q values for all state action pairs
            policy = np.zeros((state_count, action_count))
            
            # Generate Greedy policy based on Q_values after each episode
            for state in range(len(Q_values)):
                # find the best action at each state
                best_action = np.argmax(Q_values[state])
                # write deterministic policy based on Q_values
                policy[state][best_action] = 1
            
            # Generate test trajectory with the greedy policy
            state_list, action_list, test_reward_list = generate_episode(200)
            
            # sum up all the rewards obtained during test trajectory and append to list
            episode_test_reward_list.append(sum(test_reward_list))
            
            # print current episode
            clear_output(wait=True)
            display('Epsilon: ' + str(eps) + ' Run: ' + str(run) + ' Episode: ' + str(episode))

        # get average test reward
        average_test_reward_list.append(Average(episode_test_reward_list))

        Q_values_list.append(Q_values)

        # test reward of each episode, where delta is the change in Q values
        plt.plot(episode_test_reward_list)
        plt.title('Testing: TdLambda Reward after Run: ' + str(int(run)) + ', Epsilon: ' + str(float(eps)))
        plt.xlabel('Episode')
        plt.ylabel('Reward')
        # plot moving average
        delta_frame = pd.DataFrame(episode_test_reward_list)
        rolling_mean = delta_frame.rolling(window=window_length).mean()
        plt.plot(rolling_mean, label='Moving Average', color='orange')
        plt.savefig('Graphs/TdLambda/test_reward/test_reward_run_' + str(int(run)) + '_epsilon_' + str(float(eps)) + '.png')
        plt.clf()
        time.sleep(0.1)

        # max delta of each episode, where delta is the change in Q values
        plt.plot(delta_list)
        plt.title('Training: TdLambda Max Delta for Run: ' + str(int(run)) + ', Epsilon: ' + str(float(eps)))
        plt.xlabel('Episode')
        plt.ylabel('Max Delta')
        # plot moving average
        delta_frame = pd.DataFrame(delta_list)
        rolling_mean = delta_frame.rolling(window=window_length).mean()
        plt.plot(rolling_mean, label='Moving Average', color='orange')
        plt.savefig('Graphs/TdLambda/delta/delta_run_'+str(int(run))+'_epsilon_' + str(float(eps)) + '.png')
        plt.clf()
        time.sleep(0.1)

        # average reward per episode
        plt.plot(average_reward_list)
        plt.title('Training: TdLambda Avg. Reward for Run: ' + str(int(run)) + ', Epsilon: ' + str(float(eps)))
        plt.xlabel('Episode')
        plt.ylabel('Average Reward')
        # plot moving average
        reward_frame = pd.DataFrame(average_reward_list)
        rolling_mean = reward_frame.rolling(window=window_length).mean()
        plt.plot(rolling_mean, label='Moving Average', color='orange')
        plt.savefig('Graphs/TdLambda/average_reward/avg_reward_run_'+str(int(run))+'_epsilon_' + str(float(eps)) + '.png')
        plt.clf()
        time.sleep(0.1)

        # cumulative reward per episode
        plt.plot(cumulative_reward_list)
        plt.title('Training: TdLambda Cumulative Reward for Run: '+ str(int(run)) + ', Epsilon: ' + str(float(eps)))
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.savefig('Graphs/TdLambda/cumulative_reward/cumulative_reward_run_'+str(int(run))+'_epsilon_' + str(float(eps)) + '.png')
        plt.clf()
        time.sleep(0.1)
    
    # test reward of each episode, where delta is the change in Q values
    plt.plot(average_test_reward_list)
    plt.title('Testing: TdLambda Avg. Reward, Epsilon: ' + str(float(eps)))
    plt.xlabel('Run')
    plt.ylabel('Reward')
    plt.xticks(np.arange(0, runs, step=1))
    plt.savefig('Graphs/TdLambda/average_test_rewards/avg_test_reward_epsilon_' + str(float(eps)) + '.png')
    plt.clf()
    time.sleep(0.1)

    # save Q value tables to a pickle
    with open('Graphs/TdLambda/Qvalues/TdLambda_Qvalues_' + str(eps) + '.pkl', 'wb') as f:
        pickle.dump(Q_values_list, f)