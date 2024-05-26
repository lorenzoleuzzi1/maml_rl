import numpy as np
import torch
from torch import optim
from utils import compute_discounted_rewards, select_action, plot_reinforce_training, set_env_task, get_env_dims
from utils import  CartPoleMetaLoader, AcrobotMetaLoader, HighwayMetaLoader
from policy import PolicyNetwork
import gymnasium as gym
import argparse
import pprint

AVAILABLE_ENVS = ['cartpole', 'acrobot', 'highway']

def reinforce(env, policy, task, lr, num_episodes, return_threshold, gamma=0.99):
    print(f"Task: {task}")

    returns = []
    losses = []
    optimizer = optim.Adam(policy.parameters(), lr=lr)
    
    # Training loop
    for episode in range(num_episodes):
        
        set_env_task(env, task)
        #env = gym.make(env_id)
        state, _ = env.reset(seed=task['seed'])

        log_probs = []
        rewards = []
        done = truncated = False

        while True:
            state = torch.from_numpy(state.flatten()).float()
            probs = policy(state)
            action = select_action(probs.detach())
            log_prob = torch.log(probs[action])
            next_state, reward, done, truncated, _ = env.step(action)
            state = next_state

            log_probs.append(log_prob)
            rewards.append(reward)

            if done or truncated:
                break 

        discounted_rewards = compute_discounted_rewards(rewards, gamma)
        discounted_rewards = torch.tensor(discounted_rewards)
        # Normalize rewards
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

        loss = []
        for log_prob, reward in zip(log_probs, discounted_rewards):
            loss.append(-log_prob * reward)
        
        loss = torch.stack(loss).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        returns.append(np.sum(rewards)) # Store rewards for plotting
        losses.append(loss.item()) # Store policy loss for plotting
        
        if np.sum(rewards) >= return_threshold:
            print(f"Solved (reward {np.sum(rewards):.2f}) in {episode} gradient updates.")
            break
        # Print episode statistics
        if episode % 50 == 0:
            print(f'Episode {episode}\t Reward sum: {np.sum(rewards):.2f}, policy loss: {loss:.2f}')

    return returns, losses

def main(env_id, lr, num_episodes, seed):
    env = gym.make(env_id)
    state_dim, action_dim = get_env_dims(env)
    policy = PolicyNetwork(state_dim, action_dim, 100)
    
    if env_id == 'CartPole-v1':
        task = CartPoleMetaLoader(1, seed).__iter__().__next__()
        return_threshold = 500
    elif env_id == 'Acrobot-v1':
        task = AcrobotMetaLoader(1, seed).__iter__().__next__()
        return_threshold = -100
    elif env_id == 'highway-fast-v0':
        task = HighwayMetaLoader(1, seed).__iter__().__next__()
        return_threshold = 22
    else:
        raise ValueError("Invalid environment id")

    returns, losses = reinforce(env, policy, task, lr=lr, num_episodes=num_episodes, return_threshold=return_threshold)

    plot_reinforce_training(returns, losses, title=env_id, save=True)

if __name__ == "__main__":
    # Parse args
    parser = argparse.ArgumentParser(
                    prog='REINFORCE',
                    description='Train a policy using REINFORCE')
    parser.add_argument('--env', type=str, default='cartpole', help='Environment id')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()

    if args.env not in AVAILABLE_ENVS:
        raise ValueError(f"Invalid environment id. Available envs: {AVAILABLE_ENVS}")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if args.num_episodes <= 0:
        raise ValueError("Number of episodes must be positive")
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")
    
    if args.env == 'cartpole':
        env_id = 'CartPole-v1'
    elif args.env == 'acrobot':
        env_id = 'Acrobot-v1'
    else:
        env_id = 'highway-fast-v0'

    main(env_id, args.lr, args.num_episodes, args.seed)