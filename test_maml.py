import gymnasium as gym
from policy import PolicyNetwork
from utils import CartPoleMetaLoader, AcrobotMetaLoader, HighwayMetaLoader
from utils import plot_init_comparison, get_env_dims
from reinforce import reinforce
import torch
import numpy as np
import argparse
import imageio

AVAILABLE_ENVS = ['cartpole', 'acrobot', 'highway']

def run_reinforce_random_maml_init(env_id, lr, num_episodes, num_tasks, path_checkpoint, seed):
    env = gym.make(env_id)
    env.reset()
    state_dim, action_dim = get_env_dims(env)
    if env_id == 'CartPole-v1':
        random_init_task_loader = CartPoleMetaLoader(num_tasks, seed=seed)
        maml_init_task_loader = CartPoleMetaLoader(num_tasks, seed=seed)
        return_threshold = 500
    elif env_id == 'Acrobot-v1':
        random_init_task_loader = AcrobotMetaLoader(num_tasks, seed=seed)
        maml_init_task_loader = AcrobotMetaLoader(num_tasks, seed=seed)
        return_threshold = -100
    else:
        random_init_task_loader = HighwayMetaLoader(num_tasks, seed=seed)
        maml_init_task_loader = HighwayMetaLoader(num_tasks, seed=seed)
        return_threshold = 22 
    
    print("--- Random init ---")
    random_init_returns = []
    random_init_losses = []
    random_init_num_updates_to_complete = []
    for task in random_init_task_loader:
        random_init_policy = PolicyNetwork(state_dim, action_dim, 100)
        returns, losses = reinforce(env, random_init_policy, task, lr=lr, num_episodes=num_episodes, return_threshold=return_threshold)
        random_init_returns.append(returns)
        random_init_losses.append(losses)
        random_init_num_updates_to_complete.append(len(returns))

    print("\n--- MAML init --- ")
    maml_init_returns = []
    maml_init_losses = []
    maml_init_num_updates_to_complete = []
    for task in maml_init_task_loader:
        maml_init_policy = PolicyNetwork(state_dim, action_dim, 100)
        maml_init_params = torch.load(path_checkpoint)['model_state_dict']
        maml_init_policy.load_state_dict(maml_init_params)
        returns, losses = reinforce(env, maml_init_policy, task, lr=lr, num_episodes=num_episodes, return_threshold=return_threshold)
        maml_init_returns.append(returns)
        maml_init_losses.append(losses)
        maml_init_num_updates_to_complete.append(len(returns))
    
    data_random_init = {
        'random_init_returns': random_init_returns,
        'random_init_losses': random_init_losses,
        'random_init_num_updates_to_complete': random_init_num_updates_to_complete
    }

    data_maml_init = {
        'maml_init_returns': maml_init_returns,
        'maml_init_losses': maml_init_losses,
        'maml_init_num_updates_to_complete': maml_init_num_updates_to_complete
    }
    
    return data_random_init, data_maml_init

def compare_init(experiment_name, num_tasks):
    data_random_init = np.load(f'data/{experiment_name}_random_init.npy', allow_pickle=True).item()
    data_maml_init = np.load(f'data/{experiment_name}_maml_init.npy', allow_pickle=True).item()

    random_returns = data_random_init['random_init_returns']
    maml_returns = data_maml_init['maml_init_returns']

    plot_init_comparison(random_returns, maml_returns, num_tasks, 5, title=experiment_name, save=True)

    random_returns = data_random_init['random_init_returns']
    maml_returns = data_maml_init['maml_init_returns']
    plot_init_comparison(random_returns, maml_returns, num_tasks, 300, title=experiment_name, save=True)

    random_updates_to_complete = data_random_init['random_init_num_updates_to_complete']
    maml_updates_to_complete = data_maml_init['maml_init_num_updates_to_complete']
    random_updates_to_complete_mean = np.mean(random_updates_to_complete)
    random_updates_to_complete_std = np.std(random_updates_to_complete)
    maml_updates_to_complete_mean = np.mean(maml_updates_to_complete)
    maml_updates_to_complete_std = np.std(maml_updates_to_complete)
    print(f"Random init: {random_updates_to_complete_mean:.2f} +/- {random_updates_to_complete_std:.2f}")
    print(f"MAML init: {maml_updates_to_complete_mean:.2f} +/- {maml_updates_to_complete_std:.2f}")

def visualize_policy(env_id, path_checkpoint):
    env = gym.make(env_id)
    env.reset()
    state_dim, action_dim = get_env_dims(env)
    policy = PolicyNetwork(state_dim, action_dim, 100)
    params = torch.load(path_checkpoint)['model_state_dict']
    policy.load_state_dict(params)
    frames = []

    state, _ = env.reset()
    frames.append(env.render(mode='rgb_array'))
    done = truncated = False
    while True:
        state = torch.from_numpy(state.flatten()).float()
        probs = policy(state)
        action = torch.argmax(probs).item()
        state, _, done, truncated, _ = env.step(action)
        
        frames.append(env.render(mode='rgb_array'))
        
        if done or truncated:
            break
    
    # Save the frames as a GIF
    imageio.mimsave('episode.gif', frames, fps=30)
        
def main(env_id, model_name, run, lr, num_episodes, num_tasks, seed):
    experiment_name = f'{model_name}_{env_id}'
    path_checkpoint = f'checkpoints/{experiment_name}_model.pth'

    if run:
        data_random_init, data_maml_init = run_reinforce_random_maml_init(env_id, lr, num_episodes, num_tasks, path_checkpoint, seed)

        # Save data
        np.save(f'data/{experiment_name}_random_init.npy', data_random_init)
        np.save(f'data/{experiment_name}_maml_init.npy', data_maml_init)

    compare_init(experiment_name, num_tasks)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Test MAML',
                    description='Test a Model Agnostic Meta Learning for Reinforcement Learning initizalization against a random one.') 
    parser.add_argument('--env', type=str, default='cartpole', help='Environment id')
    parser.add_argument('--model_name', type=str, default='vpg', help='Path to the model checkpoint.')
    parser.add_argument('--run', type=bool, default=False, help='Run the test or just visualize the results.')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_episodes', type=int, default=1000, help='Number of episodes')
    parser.add_argument('--num_tasks', type=int, default=10, help='Batch tasks size')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()
    
    if args.env not in AVAILABLE_ENVS:
        raise ValueError(f"Invalid environment id. Available envs: {AVAILABLE_ENVS}")
    if args.lr <= 0:
        raise ValueError("Learning rate must be positive")
    if args.num_episodes <= 0:
        raise ValueError("Number of episodes must be positive")
    if args.num_tasks <= 0:
        raise ValueError("Number of tasks must be positive")
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")

    if args.env == 'cartpole':
        env_id = 'CartPole-v1'
    elif args.env == 'acrobot':
        env_id = 'Acrobot-v1'
    else:
        env_id = 'highway-fast-v0'
    
    main(env_id, args.model_name, args.run, args.lr, args.num_episodes, args.num_tasks, args.seed)