import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

# Policy based in A2C (Actor-Critic)
# Shared head for action and value
class Policy(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Policy, self).__init__()
        self.fc1 = nn.Linear(in_size, hidden_size)
        self.action_head = nn.Linear(hidden_size, out_size)
        self.critics_value_head = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        return F.softmax(self.action_head(x), dim=-1), self.critics_value_head(x)

    def sample_action_value(self, state: torch.Tensor):
        state = state.float()
        probs, value = self.forward(state)
        distribution = Categorical(probs) # for a continuous action space it could be Normal instead
        action = distribution.sample()
        return action.item(), distribution.log_prob(action), value

class ActorCriticAgent:
    learning_rate_a = 0.001
    discount_factor_g = 0.99
    hidden_size = 128

    def train(self, episodes, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        policy = Policy(num_states, self.hidden_size, num_actions)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate_a)
        rewards_per_episode = np.zeros(episodes)

        for i in range(episodes):
            if i % 100 == 0:
                print(f'Episodes trained %{i*100.0/episodes}')
            state = env.reset()[0]
            episode_rewards = []
            episode_log_probs = []
            episode_values = []
            terminated = False
            truncated = False
            steps = 0
            while (not terminated and not truncated):
                # Take an action from the policy network
                action, log_prob, value = policy.sample_action_value(self.state_to_input(state, num_states))
                new_state, reward, terminated, truncated, _ = env.step(action)
                state = new_state
                # save rewards and log_probs for training
                episode_rewards.append(reward)
                episode_log_probs.append(log_prob)
                episode_values.append(value)
                steps += 1

            rewards_per_episode[i] = sum(episode_rewards)

            # use dynamic programming to calculate the returns (from T-1 to 0)
            discounted_returns = deque(maxlen=steps)
            for t in range(steps)[::-1]:
                discounted_return_t = (discounted_returns[0] if len(discounted_returns) > 0 else 0)
                discounted_returns.appendleft(episode_rewards[t] + self.discount_factor_g * discounted_return_t)
            discounted_returns = torch.tensor(discounted_returns)
            
            # optimize, the action and update are both executed on the policy (a2c is an on-policy algorithm)
            policy_loss = []
            value_loss = []
            for log_prob, disc_return, value in zip(episode_log_probs, discounted_returns, episode_values):
                advantage = disc_return - value.item()
                # loss is negative because we want gradient ascent
                policy_loss.append(-log_prob * advantage)
                value_loss.append(F.smooth_l1_loss(value, torch.tensor([disc_return])))

            policy_value_loss = torch.stack(policy_loss).sum() + torch.stack(value_loss).sum()
            optimizer.zero_grad()
            policy_value_loss.backward()
            optimizer.step()

        env.close()

        torch.save(policy.state_dict(), 'a2c.pt')
        plt.figure(1)
        sum_rewards = np.zeros(episodes)
        for x in range(episodes):
            sum_rewards[x] = np.sum(rewards_per_episode[max(0, x-100):x+1])
        plt.subplot(111) # plot 1 row x 1 columns grid, at cell 1
        plt.plot(sum_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Sum of rewards (win goal)')
        plt.savefig('a2c.png')

    def state_to_input(self, state, num_states):
        encoded = torch.zeros(num_states)
        encoded[state] = 1
        return encoded
    
    def test(self, episodes, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human')
        num_states = env.observation_space.n
        num_actions = env.action_space.n
        policy = Policy(num_states, self.hidden_size, num_actions)
        policy.load_state_dict(torch.load('a2c.pt'))
        policy.eval() # switch to evaluation mode
        rewards = np.zeros(episodes)
        for i in range(episodes):
            state = env.reset()[0]
            terminated = False # whether the episode is terminated (win or lose)
            truncated = False # whether the episode is truncated (action limit)
            while (not terminated and not truncated):
                with torch.no_grad():
                    action, _, _ = policy.sample_action_value(self.state_to_input(state, num_states))
                new_state, reward, terminated, truncated, _ = env.step(action)
                rewards[i] += reward
                state = new_state
        print(f'Average rewards: {np.mean(rewards)}')
        env.close()

if __name__ == '__main__':
    agent = ActorCriticAgent()
    is_slippery = True # stochastic environment, no guarantee the action will reach the goal (A2C can solve it!).
    agent.train(episodes=20000, is_slippery=is_slippery)
    agent.test(episodes=4, is_slippery=is_slippery)
    