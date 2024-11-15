import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

# Policy based in NN
class Policy(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.out(x)
        return F.softmax(x, dim=-1)

    def sample_action(self, state: torch.Tensor):
        state = state.float()
        probs = self.forward(state)
        distribution = Categorical(probs) # for a continuous action space it could be Normal instead
        action = distribution.sample()
        return action.item(), distribution.log_prob(action)

class ReinforceAgent:
    learning_rate_a = 0.001
    discount_factor_g = 0.9
    
    def train(self, episodes, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        policy = Policy(num_states, num_states, num_actions)        
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate_a)
        rewards_per_episode = np.zeros(episodes)

        for i in range(episodes):
            state = env.reset()[0]
            episode_rewards = []
            episode_log_probs = []
            terminated = False
            truncated = False
            steps = 0
            while (not terminated and not truncated):
                # Take an action from the policy network
                action, log_prob = policy.sample_action(self.state_to_input(state, num_states))
                new_state, reward, terminated, truncated, _ = env.step(action)
                state = new_state
                # save rewards and log_probs for training
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)
                steps += 1

            rewards_per_episode[i] = sum(episode_rewards)

            # use dynamic programming to calculate the returns (from T-1 to 0)
            discounted_returns = deque(maxlen=steps)
            for t in range(steps)[::-1]:
                discounted_return_t = (discounted_returns[0] if len(discounted_returns) > 0 else 0)
                discounted_returns.appendleft(episode_rewards[t] + self.discount_factor_g * discounted_return_t)
            discounted_returns = torch.tensor(discounted_returns)
            
            # optimize, the action and update are both executed on the policy (REINFORCE is an on-policy algorithm)
            policy_loss = []
            for log_prob, disc_return in zip(episode_log_probs, discounted_returns):
                # loss is negative because we want gradient ascent
                policy_loss.append(-log_prob * disc_return)
            policy_loss = torch.stack(policy_loss).sum()
            optimizer.zero_grad()
            policy_loss.backward()
            optimizer.step()

        env.close()

        torch.save(policy.state_dict(), 'reinforce.pt')
        plt.figure(1)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):x+1])
        plt.subplot(111) # plot 1 row x 1 columns grid, at cell 1
        plt.plot(sum_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Sum of rewards (win goal)')
        plt.savefig('reinforce.png')

    def state_to_input(self, state, num_states):
        encoded = torch.zeros(num_states)
        encoded[state] = 1
        return encoded
    
    def test(self, episodes, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        policy = Policy(num_states, num_states, num_actions)
        policy.load_state_dict(torch.load('reinforce.pt'))
        policy.eval() # switch to evaluation mode

        for _ in range(episodes):
            state = env.reset()[0]
            terminated = False # whether the episode is terminated (win or lose)
            truncated = False # whether the episode is truncated (action limit)
            while (not terminated and not truncated):
                with torch.no_grad():
                    action, _ = policy.sample_action(self.state_to_input(state, num_states))
                new_state, _, terminated, truncated, _ = env.step(action)
                state = new_state
        env.close()

if __name__ == '__main__':
    agent = ReinforceAgent()
    is_slippery = False # stochastic environment, no guarantee the action will reach the goal (hard for REINFORCE to solve).
    agent.train(episodes=10000, is_slippery=is_slippery)
    agent.test(episodes=4, is_slippery=is_slippery)