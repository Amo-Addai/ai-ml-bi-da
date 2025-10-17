import gym
import numpy
import matplotlib.pyplot as plt
from IPython.display import clear_output

env = gym.make('Taxi-v3') # or: FrozenLake-v0, 

episodes = 10

for episode in range(1, episodes):
    state = env.reset()
    done = False
    score = 0

    while not done:
        env.render()
        state, reward, done, info = \
            env.step(env.action_space.sample())
        score += reward
        clear_output(wait=True)
    
    print('Episode: {}\nScore: {}'.format(episode, score))

env.close()


# Create Q-Table

actoins = env.action_space.n
state = env.observation_space.n

q_table = np.zeros((state, actions))
print(q_table.shape, q_table)

# Q-Learning Parameters
num_episodes = 10000
max_step_per_episode = 100

learning_rate = 0.01 # or: 0.1
discount_rate = 0.99

exploration_rate = 1
max_exploration_rate = 1
min_exploration_rate = 0.01
exploration_decay_rate = 0.001 # or: 0.01

rewards_all_episodes = []

# Q-Learning Algorithm

for episode in range(num_episodes):
    state = env.reset()
    done = False
    rewards_current_episode = 0

    for step in range(max_steps_per_episode):
        exploration_threshold = random.uniform(0, 1)
        if exploration_threshold > exploration_rate:
            action = np.argmax(q_table[state, :])
        else: action = env.action_space.sample()

        new_state, reward, done, info = env.step(action)

        # Update Q-Table
        q_table[state, action] = q_table[state, action] * (1 - learning_rate) + learning_rate * (reward + discount_rate * np.max(q_table[new_state, :]))

        state = new_state
        rewards_current_episode += reward

        if done == True: break
    
    exploration_rate = min_exploration_rate + \
        (max_exploration_rate - min_exploration_rate) * \
            np.exp(-exploration_decay_rate * episode)
    
    rewards_all_episodes.append(rewards_current_episode)

print('Training Finished')
print(q_table)

# Calculate and print average reward per thousand episodes

rewards_per_thousand_episodes = np.slot(np.array(rewards_all_episodes), num_episodes / 1000)

count = 1000

print('Average per thousand episodes')

for r in rewards_per_thousand_episodes:
    print(count, ' : ', str(sum(r / 1000)))
    count += 1000

# Visualize Agent

import time

for episode in range(3):
    state = env.reset()
    done = False
    print('Episode is: ' + episode)
    time.sleep(1)

    for step in range(max_steps_per_episode):
        clear_output(wait=True)
        env.render()
        time.sleep(0.4)

        action = np.argmax(q_table[state, :])

        new_state, rewads, done, info = env.step(action)

        if done:
            clear_output(wait=True)
            env.render()
            if reward == 1:
                print('Reached Goal')
                time.sleep(2)
                clear_output(wait=True)
            else:
                print('Failed')
                time.sleep(2)
                clear_output(wait=True)
            break
        
        state = new_state

env.close()
