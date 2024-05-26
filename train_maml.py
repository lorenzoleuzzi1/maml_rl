import gymnasium as gym
from policy import PolicyNetwork
from maml import MAML
from utils import plot_maml_training, get_env_dims
from utils import CartPoleMetaLoader, AcrobotMetaLoader, HighwayMetaLoader
import argparse
import numpy as np

AVAILABLE_ENVS = ['cartpole', 'acrobot', 'highway']

def main(env_string,
        first_order, 
        inner_lr, 
        meta_lr, 
        num_iterations, 
        num_trajectories, 
        horizon, 
        batch_tasks_size, 
        model_name, 
        seed,
        return_threshold
        ):
    
    experiment_name = model_name + '_' + env_string 
    
    env = gym.make(env_string)
    state_dim, action_dim = get_env_dims(env)

    policy = PolicyNetwork(state_dim, action_dim, 100)

    maml_model = MAML( env, 
                    policy, 
                    first_order=first_order,
                    inner_lr=inner_lr, 
                    meta_lr=meta_lr, 
                    num_iterations=num_iterations, 
                    num_trajectories=num_trajectories, 
                    horizon=horizon, 
                    save_path=experiment_name,
                    return_threshold=return_threshold
                    )
    if env_string == 'CartPole-v1':
        train_loader = CartPoleMetaLoader(batch_tasks_size, seed=seed)
        eval_loader = CartPoleMetaLoader(batch_tasks_size, seed=seed+1)
    elif env_string == 'Acrobot-v1':
        train_loader = AcrobotMetaLoader(batch_tasks_size, seed=seed)
        eval_loader = AcrobotMetaLoader(batch_tasks_size, seed=seed+1)
    elif env_string == 'highway-fast-v0':
        train_loader = HighwayMetaLoader(batch_tasks_size, seed=seed)
        eval_loader = HighwayMetaLoader(batch_tasks_size, seed=seed+1)

    log = maml_model.train_and_evaluate(train_loader, eval_loader)

    plot_maml_training(log['vl_return'], log['tr_loss'], title=experiment_name, save=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
                    prog='Train MAML',
                    description='Train Model Agnostic Meta Learning for Reinforcement Learning') 
    parser.add_argument('--env', type=str, default='cartpole', help='Environment id')
    parser.add_argument('--fo', type=bool, default=True, help='First order MAML')
    parser.add_argument('--inner_lr', type=float, default=1e-2, help='Inner Learning rate')
    parser.add_argument('--meta_lr', type=float, default=1e-3, help='Meta Learning rate')
    parser.add_argument('--num_iterations', type=int, default=100, help='Number of iterations')
    parser.add_argument('--num_trajectories', type=int, default=10, help='Number of sampled trajectories')
    parser.add_argument('--horizon', type=int, default=500, help='Horizon')
    parser.add_argument('--batch_tasks_size', type=int, default=10, help='Batch tasks size')
    parser.add_argument('--model_name', type=str, default='vpg', help='Model name')
    parser.add_argument('--seed', type=int, default=42, help='Seed')
    args = parser.parse_args()

    if args.env not in AVAILABLE_ENVS:
        raise ValueError(f"Invalid environment id. Available envs: {AVAILABLE_ENVS}")
    if args.inner_lr <= 0:
        raise ValueError("Inner learning rate must be positive")
    if args.meta_lr <= 0:
        raise ValueError("Meta learning rate must be positive")
    if args.num_iterations <= 0:
        raise ValueError("Number of iterations must be positive")
    if args.num_trajectories <= 0:
        raise ValueError("Number of trajectories must be positive")
    if args.horizon <= 0:
        raise ValueError("Horizon must be positive")
    if args.batch_tasks_size <= 0:
        raise ValueError("Batch tasks size must be positive")
    if args.seed < 0:
        raise ValueError("Seed must be non-negative")
    if args.env == 'cartpole':
        env_id = 'CartPole-v1'
        return_threshold = 500
    elif args.env == 'acrobot':
        env_id = 'Acrobot-v1'
        return_threshold = -100
    elif args.env == 'highway':
        env_id = 'highway-fast-v0'
        return_threshold = 22
    
    # Print training information
    print(f"Training MAML on {env_id} with following hyperparameters:")
    print(f"First order: {args.fo}")
    print(f"Inner learning rate: {args.inner_lr}")
    print(f"Meta learning rate: {args.meta_lr}")
    print(f"Number of iterations: {args.num_iterations}")
    print(f"Number of trajectories: {args.num_trajectories}")
    print(f"Horizon: {args.horizon}")
    print(f"Batch tasks size: {args.batch_tasks_size}")
    print(f"Model name: {args.model_name}")
    print(f"Seed: {args.seed}")

    main(env_id,
        args.fo, 
        args.inner_lr, 
        args.meta_lr, 
        args.num_iterations, 
        args.num_trajectories, 
        args.horizon, 
        args.batch_tasks_size, 
        args.model_name, 
        args.seed,
        return_threshold
        )