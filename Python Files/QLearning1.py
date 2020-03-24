import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import random
import time
from Gridworld import Gridworld

actions = [[-1, 0], [0, 1], [1, 0], [0, -1]] #up, right, down, left = (clockwise from up) 
action_count = len(actions) # total number of actions
gridSize = 5 # create a square grid of gridSize by gridSize
state_count = gridSize*gridSize # total number of states

global Q_values

# define a function that chooses action based on epsilon-greedy
def choose_action(state, epsilon):
    
    # choose an action type: explore or exploit
    action_type = int(np.random.choice(2, 1, p=[epsilon,1-epsilon]))

    # find best action based on Q values
    best_action_index = np.argmax(Q_values[state])

    # pick a random action
    random_action_index = random.choice(range(4))

    # while random action is the same as the best action, pick a new action
    while random_action_index == best_action_index:
        random_action_index = random.choice(range(4))

    # choose an action based on exploit or explore
    if action_type == 0:
        # explore
        # print("explore")
        action_index = random_action_index
    else:
        # exploit
        # print("exploit")
        action_index = best_action_index
        
    return action_index

# define average function
def Average(lst): 
    return sum(lst) / len(lst) 

# define a function that generates an episode
def generate_episode(steps, grid, policy):

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

def q_learning(gamma, lr, epsilon, runs):

    # create a grid object
    grid = Gridworld(5)

    # # initialize other parameters
    # gamma = 0.99
    # lr = 0.1
    # epsilon = [0.01, 0.1, 0.25]
    # runs = 20

    for eps in epsilon:
        average_test_reward_list = []
        for run in range(runs):
                
            # initialize q values for all state action pairs
            global Q_values
            Q_values = np.zeros((state_count, action_count))

            # define lists for plots
            average_reward_list = []
            cumulative_reward_list = []
            cumulative_reward = 0
            delta_list = []
            episode_test_reward_list = []
            
            # iterate over 500 episodes
            for episode in range(500):

                # initialize state (output: [4, 4])
                state = grid.initial_state()

                reward_list = []
                delta = 0
                
                # iterate over 200 steps within each episode
                for step in range(200):

                    # get state index (output: 24)
                    state_index = grid.states.index(state)

                    # choose an action based on epsilon-greedy (output: action index ie. 0)
                    action_index = choose_action(state_index, eps)
                    action_vector = actions[action_index] # convert action_index (0) to action_vector ([-1, 0])

                    # get the next state and reward after taking the chosen action in the current state
                    next_state_vector, reward = grid.transition_reward(state, action_vector)
                    next_state_index = grid.states.index(list(next_state_vector))

                    # add reward to list
                    reward_list.append(reward)
                    
                    # calculate max delta change for plotting max q value change
                    Q_value = Q_values[state_index][action_index] + lr*(reward + gamma*np.max(Q_values[next_state_index])-Q_values[state_index][action_index])
                    delta = max(delta, np.abs(Q_value - Q_values[state_index][action_index]))   
                    
                    # update Q value
                    Q_values[state_index][action_index] = Q_values[state_index][action_index] + lr*(reward + gamma*np.max(Q_values[next_state_index])-Q_values[state_index][action_index])

                    # set the next state as the current state
                    state = list(next_state_vector)
                
                # append max change in Q value to list
                delta_list.append(delta)
                
                # average rewards
                average_reward_list.append(Average(reward_list))
                
                # add cumulative reward
                cumulative_reward = cumulative_reward + sum(reward_list)
                cumulative_reward_list.append(cumulative_reward)
                
                # initialize q values for all state action pairs
                policy = np.zeros((state_count, action_count))
                
                # Generate Greedy policy based on Q_values after each episode
                for state in range(len(Q_values)):
                    # find the best action at each state
                    best_action = np.argmax(Q_values[state])
                    policy[state][best_action] = 1
                
                # Generate test trajectory with the greedy policy
                state_list, action_list, test_reward_list = generate_episode(200, grid, policy)
                
                # sum up all the rewards obtained during test trajectory and append to list
                episode_test_reward_list.append(sum(test_reward_list))

            # get average test reward
            average_test_reward_list.append(Average(episode_test_reward_list))

            # test reward of each episode, where delta is the change in Q values
            plt.plot(episode_test_reward_list)
            plt.title('Testing: Reward with Trained Policy after Run: ' + str(int(run)) + ', Epsilon: ' + str(float(eps)))
            plt.xlabel('Episode')
            plt.ylabel('Reward')
            plt.savefig('Graphs/QLearning/test_reward/test_reward_run_' + str(int(run)) + '_epsilon_' + str(float(eps)) + '.png')
            plt.clf()
            time.sleep(0.1)

            # max delta of each episode, where delta is the change in Q values
            plt.plot(delta_list)
            plt.title('Training: Max Delta per Episode for Run: ' + str(int(run)) + ', Epsilon: ' + str(float(eps)))
            plt.xlabel('Episode')
            plt.ylabel('Max Delta')
            # plot moving average
            delta_frame = pd.DataFrame(delta_list)
            rolling_mean = delta_frame.rolling(window=10).mean()
            plt.plot(rolling_mean, label='Moving Average', color='orange')
            plt.savefig('Graphs/QLearning/delta/delta_'+str(int(run))+'_epsilon_' + str(float(eps)) + '.png')
            plt.clf()
            time.sleep(0.1)

            # average reward per episode
            plt.plot(average_reward_list)
            plt.title('Training: Average Reward per Episode for Run: ' + str(int(run)) + ', Epsilon: ' + str(float(eps)))
            plt.xlabel('Episode')
            plt.ylabel('Average Reward')
            # plot moving average
            reward_frame = pd.DataFrame(average_reward_list)
            rolling_mean = reward_frame.rolling(window=10).mean()
            plt.plot(rolling_mean, label='Moving Average', color='orange')
            plt.savefig('Graphs/QLearning/average_reward/average_reward_'+str(int(run))+'_epsilon_' + str(float(eps)) + '.png')
            plt.clf()
            time.sleep(0.1)

            # cumulative reward per episode
            plt.plot(cumulative_reward_list)
            plt.title('Training: Cumulative Reward per Episode for Run: '+ str(int(run)) + ', Epsilon: ' + str(float(eps)))
            plt.xlabel('Episode')
            plt.ylabel('Cumulative Reward')
            plt.savefig('Graphs/QLearning/cumulative_reward/cumulative_reward_'+str(int(run))+'_epsilon_' + str(float(eps)) + '.png')
            plt.clf()
            time.sleep(0.1)
        
        # test reward of each episode, where delta is the change in Q values
        plt.plot(average_test_reward_list)
        plt.title('Testing: Average Reward per Run, Epsilon: ' + str(float(eps)))
        plt.xlabel('Run')
        plt.ylabel('Reward')
        plt.xticks(np.arange(1, runs+1, step=1))
        plt.savefig('Graphs/QLearning/average_test_rewards/average_test_reward_epsilon_' + str(float(eps)) + '.png')
        plt.clf()
        time.sleep(0.1)