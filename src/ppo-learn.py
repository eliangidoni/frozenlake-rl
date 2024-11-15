import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import torch
from torch import nn
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque

# Policy based in PPO (Proximal Policy Optimization)
# Different heads for action and value
class Policy(nn.Module):
    def __init__(self, in_size, hidden_size, out_size):
        super(Policy, self).__init__()
        self.action_head = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, out_size)
        )
        self.critics_value_head = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, 1)
        )
    def forward(self, x):
        return F.softmax(self.action_head(x), dim=-1), self.critics_value_head(x)

    def sample_action_value(self, state: torch.Tensor, action = None):
        state = state.float()
        probs, value = self.forward(state)
        distribution = Categorical(probs) # for a continuous action space it could be Normal instead
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), distribution.entropy(), value

class ProximalPolicyOptimizationAgent:
    learning_rate_a = 0.001
    clip_coef = 0.2
    entropy_coef = 0.01
    value_coef = 0.5
    discount_factor_g = 0.99
    hidden_size = 128
    mini_batch_size = 16
    updates_per_iteration = 4
    timesteps_per_batch = 4096
    target_kl = 0.02
    anneal_lr = True

    def train(self, total_steps, render=False, is_slippery=False):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode='human' if render else None)
        num_observations = env.observation_space.n
        num_actions = env.action_space.n
        policy = Policy(num_observations, self.hidden_size, num_actions)
        optimizer = torch.optim.Adam(policy.parameters(), lr=self.learning_rate_a)
        rewards_per_iteration = []
        steps = 0
        while steps < total_steps:
            print(f'Executed Steps %{steps*100.0/total_steps}')

            batch_log_probs = []
            batch_observations = []
            batch_actions = []            
            # Collect trajectories
            batch_rewards = []
            batch_values = []
            t = 0
            while t < self.timesteps_per_batch:
                episode_rewards = []
                episode_values = []
                terminated = False
                truncated = False
                state = env.reset()[0]
                with torch.no_grad():
                    while (not terminated and not truncated):
                        # Take an action from the policy network
                        action, log_prob, _, value = policy.sample_action_value(
                            self.state_to_input(state, num_observations))
                        new_state, reward, terminated, truncated, _ = env.step(action.item())
                        batch_observations.append(state)
                        batch_actions.append(action)
                        batch_log_probs.append(log_prob)
                        episode_rewards.append(reward)
                        episode_values.append(value)
                        state = new_state
                        t += 1
                batch_rewards.append(episode_rewards)
                batch_values.append(episode_values)
            # Transform into tensors
            batch_log_probs = torch.tensor(batch_log_probs)
            batch_observations = torch.tensor(batch_observations)
            batch_actions = torch.tensor(batch_actions)
            steps += t

            # use dynamic programming to calculate the returns (from T-1 to 0)            
            with torch.no_grad():            
                batch_returns = []
                for episode_rewards in batch_rewards:
                    discounted_returns = deque(maxlen=len(episode_rewards))
                    for t in range(len(episode_rewards))[::-1]:
                        discounted_return_t = (discounted_returns[0] if len(discounted_returns) > 0 else 0)
                        discounted_returns.appendleft(episode_rewards[t] + self.discount_factor_g * discounted_return_t)
                    batch_returns.extend(discounted_returns)
                batch_returns = torch.tensor(batch_returns)
                values = []
                for value in batch_values:
                    values.extend(value)
                values = torch.tensor(values)
                batch_advantages = batch_returns - values                

            # optimize, the action and update are both executed on the policy (ppo is an on-policy algorithm)
            indices = np.arange(len(batch_observations))
            for _ in range(self.updates_per_iteration):
                # Learning Rate Annealing
                if self.anneal_lr:
                    frac = (steps - 1.0) / total_steps
                    new_lr = self.learning_rate_a * (1.0 - frac)
                    new_lr = max(new_lr, 0.0) # Make sure learning rate doesn't go below 0
                    optimizer.param_groups[0]["lr"] = new_lr
                np.random.shuffle(indices)
                for i in range(0, len(batch_observations), self.mini_batch_size):
                    mini_indices = indices[i:i+self.mini_batch_size]
                    mini_log_probs = batch_log_probs[mini_indices]
                    mini_observations = batch_observations[mini_indices]
                    mini_actions = batch_actions[mini_indices]
                    mini_returns = batch_returns[mini_indices]
                    mini_advantages = batch_advantages[mini_indices]
                    # Calculate the policy loss
                    _, new_log_prob, entropy, new_values = policy.sample_action_value(
                        self.state_batch_to_input(mini_observations, num_observations), mini_actions)
                    logratio = new_log_prob.flatten() - mini_log_probs
                    ratio = torch.exp(logratio)
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipped_ratio = torch.clamp(ratio, 1-self.clip_coef, 1+self.clip_coef)
                    policy_loss = -torch.min(ratio*mini_advantages, clipped_ratio*mini_advantages)
                    policy_loss = policy_loss.mean()
                    # Calculate the value loss
                    value_loss = F.mse_loss(new_values.flatten(), mini_returns)
                    # Calculate the entropy loss
                    entropy_loss = entropy.mean()
                    # Calculate the total loss
                    loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy_loss
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                if self.target_kl and approx_kl > self.target_kl:
                    break

            torch.save(policy.state_dict(), 'ppo.pt')
            episodes = 100
            rewards_per_episode = self.test(episodes, is_slippery=is_slippery, render_mode=None)
            rewards_per_iteration.extend(rewards_per_episode)

        env.close()

        # plot the rewards
        sum_rewards = np.zeros(len(rewards_per_iteration))
        for x in range(len(rewards_per_iteration)):
            sum_rewards[x] = np.sum(rewards_per_iteration[max(0, x-100):x+1])
        plt.figure(1)
        plt.subplot(111) # plot 1 row x 1 columns grid, at cell 1
        plt.plot(sum_rewards)
        plt.xlabel('Episode')
        plt.ylabel('Sum of rewards (win goal)')
        plt.savefig('ppo.png')

    def state_to_input(self, state, num_observations):
        encoded = torch.zeros(num_observations)
        encoded[state] = 1
        return encoded

    def state_batch_to_input(self, state_batch, num_observations):
        encoded = torch.zeros(state_batch.shape[0], num_observations)
        for i, state in enumerate(state_batch):
            encoded[i, state] = 1
        return encoded

    def test(self, episodes, is_slippery=False, render_mode='human'):
        env = gym.make('FrozenLake-v1', map_name="4x4", is_slippery=is_slippery,
                       render_mode=render_mode)
        num_observations = env.observation_space.n
        num_actions = env.action_space.n
        policy = Policy(num_observations, self.hidden_size, num_actions)
        policy.load_state_dict(torch.load('ppo.pt'))
        policy.eval() # switch to evaluation mode
        rewards = np.zeros(episodes)
        for i in range(episodes):
            state = env.reset()[0]
            terminated = False # whether the episode is terminated (win or lose)
            truncated = False # whether the episode is truncated (action limit)
            while (not terminated and not truncated):
                with torch.no_grad():
                    action, _, _, _ = policy.sample_action_value(self.state_to_input(state, num_observations))
                new_state, reward, terminated, truncated, _ = env.step(action.item())
                rewards[i] += reward
                state = new_state
        env.close()
        return rewards

if __name__ == '__main__':
    agent = ProximalPolicyOptimizationAgent()
    is_slippery = False # stochastic environment, no guarantee the action will reach the goal (PPO can solve it!).
    agent.train(total_steps=200000, is_slippery=is_slippery)
    agent.test(episodes=4, is_slippery=is_slippery)
