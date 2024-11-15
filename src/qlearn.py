import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

def run(episodes, render=False, training=True):
    env = gym.make('FrozenLake-v1',
                   map_name='4x4',
                   # when is_slippery, the env is stochastic (an action behaves as a different one) so it needs more episodes
                   is_slippery=False,
                   render_mode='human' if render else None)
    if training:
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('qlearn.pkl', 'rb')
        q = pickle.load(f)
        f.close()
    learning_rate = 0.9
    discount_rate = 0.9
    epsilon = 1 # exploration rate (1: full exploration, 0: no exploration)
    epsilon_decay = 0.0001 # we need to train at least for 10k episodes to reach epsilon 0 (no exploration)
    rng = np.random.default_rng()
    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        terminated = False # agent reached a goal (good or bad)
        truncated = False # agent did not reach a goal (an external signal not part of the env)
        
        state = env.reset()[0]
        while (not terminated) and (not truncated):
            if training and rng.random() < epsilon:
                # take a random action
                action = env.action_space.sample()
            else:
                # take the best action
                action = np.argmax(q[state, :])
            new_state, reward, terminated, truncated, _ = env.step(action)
            if training:
                q[state, action] = (q[state, action] +
                                learning_rate * (
                                    reward + discount_rate * np.max(q[new_state, :]) - q[state, action]
                                ))
            state = new_state

        # decrease epsilon/exploration (less random action over time) 
        epsilon = max(epsilon - epsilon_decay, 0)
        if epsilon == 0:
            # adjust to low learning rate as there is no exploration anymore
            learning_rate = 0.0001
        if reward == 1:
            # used for the plot
            rewards_per_episode[i] = 1

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0,t-100):(t+1)])
    plt.plot(sum_rewards)
    plt.xlabel('Episode')
    plt.ylabel('Sum of rewards (win goal)')
    plt.savefig('qlearn.png')

    if training:
        f = open('qlearn.pkl', 'wb')
        pickle.dump(q, f)
        f.close()

if __name__ == '__main__':
    run(15000, training=True, render=None)