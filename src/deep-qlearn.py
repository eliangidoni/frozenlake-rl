import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt
import torch
import random
from torch import nn
import torch.nn.functional as F
from collections import deque

# a DQN approximates the Q-value instead of using a table to store them.
class DQN(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super().__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return x

# Queue with limited capacity, it discards the oldest element (FIFO)
class ReplayMemory:
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def append(self, transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQLAgent:
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    network_sync_rate = 10
    replay_memory_size = 1000
    mini_batch_size = 32
    loss_fn = nn.MSELoss()    
    optimizer = None
    
    def train(self, episodes, render=False, is_slippery=False, double_dqn=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n

        epsilon = 1.0
        epsilon_decay = 1.0/episodes
        memory = ReplayMemory(self.replay_memory_size)

        policy_dqn = DQN(num_states, num_states, num_actions)
        target_dqn = DQN(num_states, num_states, num_actions)
        target_dqn.load_state_dict(policy_dqn.state_dict())
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)
        
        rewards_per_episode = np.zeros(episodes)
        epsilon_history = []
        step_count = 0

        for i in range(episodes):
            state = env.reset()[0]
            terminated = False
            truncated = False
            while (not terminated and not truncated):
                # Take an action from the policy network or randomly (exploration)
                if random.random() < epsilon:
                    action = env.action_space.sample()
                else:
                    with torch.no_grad():
                        action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                new_state, reward, terminated, truncated, _ = env.step(action)
                # Keep the results in memory/experience (used to train the policy network)
                memory.append((state, action, new_state, reward, terminated))
                state = new_state
                step_count += 1
            if reward == 1:
                rewards_per_episode[i] = 1
    
            # When we have enough experience (mini batch) train the policy network to minimize 
            # the loss between the current policy Q-value and the target policy Q-value. 
            # The target policy network is static and synced from the policy after X episodes.
            if len(memory) > self.mini_batch_size and np.sum(rewards_per_episode) > 0:
                mini_batch = memory.sample(self.mini_batch_size)
                self.optimize(mini_batch, policy_dqn, target_dqn, double_dqn)
                epsilon = max(epsilon - epsilon_decay, 0)
                epsilon_history.append(epsilon)
                # Sync the policy networks, copy the current policy to the target policy.
                if step_count > self.network_sync_rate:
                    target_dqn.load_state_dict(policy_dqn.state_dict())
                    step_count = 0
        env.close()

        torch.save(policy_dqn.state_dict(), 'dqn.pt')
        plt.figure(1)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):x+1])
        plt.subplot(121) # plot 1 row x 2 columns grid, at cell 1
        plt.plot(sum_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Sum of rewards (win goal)')
        plt.subplot(122) # plot 1 row x 2 columns grid, at cell 2
        plt.plot(epsilon_history)
        plt.xlabel('Episode Mini Batch')
        plt.ylabel('Epsilon (exploration rate)')
        plt.savefig('dqn.png')

    def state_to_dqn_input(self, state, num_states):
        encoded = torch.zeros(num_states)
        encoded[state] = 1
        return encoded
    
    def optimize(self, mini_batch, policy_dqn, target_dqn, double_dqn):
        num_states = policy_dqn.fc1.in_features
        current_q_list = []
        target_q_list = []
        
        for state, action, new_state, reward, terminated in mini_batch:
            if terminated:
                # reward can be 0 or 1 (lose or win)
                target = torch.FloatTensor([reward])
            elif double_dqn:
                # calculate the target Q-value from the experience/memory using the action from the policy network
                with torch.no_grad():
                    tmp_action = policy_dqn(self.state_to_dqn_input(new_state, num_states)).argmax().item()
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states))[tmp_action])
            else:
                # calculate the target Q-value from the experience/memory
                with torch.no_grad():
                    target = torch.FloatTensor(
                        reward + self.discount_factor_g * target_dqn(self.state_to_dqn_input(new_state, num_states)).max())
            
            # gather current policy Q-value
            current_q = policy_dqn(self.state_to_dqn_input(state, num_states))
            current_q_list.append(current_q)
            # gather target policy Q-value, adjusted to the calculated value
            target_q = target_dqn(self.state_to_dqn_input(state, num_states))
            target_q[action] = target
            target_q_list.append(target_q)
        
        # train the current policy one step towards the target (learnt experience)
        loss = self.loss_fn(torch.stack(current_q_list), torch.stack(target_q_list))
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def test(self, episodes, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        policy_dqn = DQN(num_states, num_states, num_actions)
        policy_dqn.load_state_dict(torch.load('dqn.pt'))
        policy_dqn.eval() # switch to evaluation mode

        for _ in range(episodes):
            state = env.reset()[0]
            terminated = False # whether the episode is terminated (win or lose)
            truncated = False # whether the episode is truncated (action limit)
            while (not terminated and not truncated):
                with torch.no_grad():
                    action = policy_dqn(self.state_to_dqn_input(state, num_states)).argmax().item()
                new_state, _, terminated, truncated, _ = env.step(action)
                state = new_state
        env.close()

if __name__ == '__main__':
    agent = DQLAgent()
    is_slippery = True # stochastic environment, no guarantee the action will reach the goal.
    double_dqn = True # use two networks to avoid overestimation of Q-values.
    agent.train(episodes=35000, is_slippery=is_slippery, double_dqn=double_dqn)
    agent.test(episodes=4, is_slippery=is_slippery)