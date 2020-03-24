for epsilon

    for 20 runs

        for 500 episodes

            for 200 steps
                (training)
            
            extract policy

            reward_list = generate_episode(200) = [0, 0, -1, 0, 5, 0, ... etc] < - add discounted rewards here

            episode_reward_list.append(sum(reward_list)) = 500 x 1 = [325, 325, 330, 380 ... etc]
        
        average_reward_list = average(episode_reward_list) = 20 x 1