from torch.distributions import Categorical
import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym

class CartPoleMetaLoader:
    def __init__(self, num_tasks, seed=0):
        self.num_tasks = num_tasks
        np.random.seed(seed)

    def __iter__(self):
        for _ in range(self.num_tasks):
            gravity = np.random.uniform(9.0, 11.0)
            cart_mass = np.random.uniform(0.5, 2.0)
            pole_mass = np.random.uniform(0.05, 0.5)
            pole_length = np.random.uniform(0.5, 2.0)
            seed = np.random.randint(0, 1000)
            yield {
                "gravity": gravity,
                "cart_mass": cart_mass,
                "pole_mass": pole_mass,
                "pole_length": pole_length,
                "seed": seed
            }
    
    def __len__(self):
        return self.num_tasks
    
class AcrobotMetaLoader:
    def __init__(self, num_tasks, seed=0):
        self.num_tasks = num_tasks
        np.random.seed(seed)

    def __iter__(self):
        for _ in range(self.num_tasks):
            link_length_1 = np.random.uniform(0.5, 1.5)
            link_length_2 = np.random.uniform(0.5, 1.5)
            link_mass_1 = np.random.uniform(0.5, 1.5)
            link_mass_2 = np.random.uniform(0.5, 1.5)
            link_com_pos_1 = np.random.uniform(0.1, 0.9)
            link_com_pos_2 = np.random.uniform(0.1, 0.9)
            link_moi = np.random.uniform(0.5, 2.0)
            seed = np.random.randint(0, 1000)
            yield {
                "link_length_1": link_length_1,
                "link_length_2": link_length_2,
                "link_mass_1": link_mass_1,
                "link_mass_2": link_mass_2,
                "link_com_pos_1": link_com_pos_1,
                "link_com_pos_2": link_com_pos_2,
                "link_moi": link_moi,
                "seed": seed
            }

    def __len__(self):
        return self.num_tasks
    
class MountainCarMetaLoader:
    def __init__(self, num_tasks, seed=0):
        self.num_tasks = num_tasks
        np.random.seed(seed)

    def __iter__(self):
        for _ in range(self.num_tasks):
            force = np.random.uniform(0.0005, 0.0015)
            gravity = np.random.uniform(0.001, 0.003)
            seed = np.random.randint(0, 1000)
            yield {
                "force": force,
                "gravity": gravity,
                "seed": seed
            }

    def __len__(self):
        return self.num_tasks

class HighwayMetaLoader:
    def __init__(self, num_tasks, seed=0):
        self.num_tasks = num_tasks
        np.random.seed(seed)

    def __iter__(self):
        env = gym.make("highway-fast-v0")
        config = env.unwrapped.default_config()
        for _ in range(self.num_tasks):
            lanes_count = np.random.randint(3, 5)
            vehicles_count = np.random.randint(20, 30)
            vehicles_density = np.random.uniform(1, 1.5)

            seed = np.random.randint(0, 1000)
            config["lanes_count"] = lanes_count
            config["vehicles_count"] = vehicles_count
            config["vehicles_density"] = vehicles_density

            yield {
                "config": config, 
                "seed": seed
            }

    def __len__(self):
        return self.num_tasks

def get_env_dims(env):
    if len(env.observation_space.shape) > 0:
        state_dim = np.prod(env.observation_space.shape)
    else:
        state_dim = env.observation_space.shape[0]
    if env.action_space.shape is None:
        raise ValueError("Action space must be discrete")
    action_dim = env.action_space.n
    
    return state_dim, action_dim

def inject_attribute_into_base_env(env, attribute_name, attribute_value):
    if hasattr(env, "env"):
        return inject_attribute_into_base_env(env.env, attribute_name, attribute_value)
    setattr(env, attribute_name, attribute_value)

def set_env_task(env, task):
    for key, value in task.items():
        if key == 'seed':
            continue
        inject_attribute_into_base_env(env, key, value)
    
def compute_discounted_rewards(rewards, gamma):
    # Function to compute the discounted rewards
    discounted_rewards = []
    R = 0
    for r in reversed(rewards):
        R = r + gamma * R
        discounted_rewards.insert(0, R)
    return discounted_rewards

def select_action(probs):
    m = Categorical(probs)
    return m.sample().item()

def plot_reinforce_training(returns, losses, title = None, save = False):
    if title is None:
        title = ""
    plt.plot(returns)
    plt.ylabel('Return')
    plt.xlabel('Episode')
    fig_title_r = title + ' - Return over episodes'
    title_r = title + '_return'
    plt.title(f'{fig_title_r}')
    if save:
        plt.savefig(f'plots/{title_r}.png')
    plt.show()

    plt.plot(losses)
    plt.ylabel('Policy Loss')
    plt.xlabel('Episode')
    fig_title_l = title + ' - Policy Loss over episodes'
    title_l = title + '_loss'
    plt.title(f'{fig_title_l}')
    if save:
        plt.savefig(f'plots/{title_l}.png')
    plt.show()

def plot_maml_training(returns, meta_losses, title = None, save = False):
    if title is None:
        title = ""
    plt.plot(returns)
    plt.ylabel('Return')
    plt.xlabel('Iteration')
    fig_title_r = title + ' - Return over iterations'
    title_r = title + '_return'
    plt.title(f'{fig_title_r}')
    if save:
        plt.savefig(f'plots/{title_r}.png')
    plt.show()

    plt.plot(meta_losses)
    plt.ylabel('Meta Loss')
    plt.xlabel('Iteration')
    fig_title_l = title + ' - Meta Loss over iterations'
    title_l = title + '_loss'
    plt.title(f'{fig_title_l}')
    if save:
        plt.savefig(f'plots/{title_l}.png')
    plt.show()

def adjust_list_length(lst, target_length, fill_value):
    if len(lst) > target_length:
        return lst[:target_length]
    else:
        return lst + [fill_value] * (target_length - len(lst))

def plot_init_comparison(random_init_returns, maml_init_returns, num_tasks, update_steps, title = None, save = False):
    # Adjust lists length
    random_init_returns = [adjust_list_length(lst, update_steps, lst[-1]) for lst in random_init_returns]
    maml_init_returns = [adjust_list_length(lst, update_steps, lst[-1]) for lst in maml_init_returns]
    if maml_init_returns[0][-1] > 0:
        start_value = 0
    else:
        start_value = -500
    # Plot all of them
    for i, rand_first in enumerate(random_init_returns):
        rand_first = np.insert(rand_first, 0, start_value)
        if i == 0:
            plt.plot(rand_first, color='blue', alpha=0.2, label='Random Init')
        else:
            plt.plot(rand_first, color='blue', alpha=0.2)
    for i, maml_first in enumerate(maml_init_returns):
        maml_first = np.insert(maml_first, 0, start_value)
        if i == 0:
            plt.plot(maml_first, color='green', alpha=0.2, label='MAML Init')
        else:
            plt.plot(maml_first, color='green', alpha=0.2)
    # Plot the mean
    random_mean = np.mean(random_init_returns, axis=0)
    random_mean = np.insert(random_mean, 0, start_value)
    maml_mean = np.mean(maml_init_returns, axis=0)
    maml_mean = np.insert(maml_mean, 0, start_value)
    plt.plot(random_mean, color='blue', label='Random Init (Mean)')
    plt.plot(maml_mean, color='green', label='MAML Init (Mean)')
    plt.xlabel('Number of Updates')
    plt.ylabel('Return')
    plt.legend()
    plt.title(f'{title} - Initializations Comparison over {num_tasks} task (up to {update_steps} updates)')
    if save:
        plt.savefig(f'plots/{title}_init_comparison_{update_steps}.png')
    plt.show()
