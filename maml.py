import torch
import gymnasium as gym
from policy import PolicyNetwork
from utils import compute_discounted_rewards, select_action, set_env_task
import numpy as np
from torch.func import vmap, grad, functional_call
import torch.optim as optim

class MAML():
    def __init__(self, 
                env : gym.Env,
                policy : PolicyNetwork,
                first_order = False,
                inner_lr = 1e-2, 
                meta_lr = 1e-3, 
                num_iterations = 100, 
                num_trajectories = 10, 
                horizon = 200, 
                return_threshold = 475,
                save_path = "model.pth"):
        
        self.env = env
        self.policy = policy

        self.first_order = first_order # First order MAML

        self.inner_lr = inner_lr # alfa learning rate in inner loop
        self.meta_lr = meta_lr   # beta learning rate in outer loop
        self.meta_optimizer = optim.Adam(self.policy.parameters(), lr=self.meta_lr)

        self.num_iterations = num_iterations  # epochs
        self.num_trajectories = num_trajectories # K
        self.horizon = horizon # H
        self.return_threshold = return_threshold # Threshold for task completed, no more training needed
        self.save_path = save_path # Model checkpoint path for saving

        # Store performance history
        self.best_return = -np.inf  # Maximizing return, trackng for model checkpointing
        self.log = {"tr_loss" : [], "vl_return": []}
    
    def _inner_loss(self, params, buffers, task):
        total_inner_loss = []
        
        for _ in range(self.num_trajectories):
            
            set_env_task(self.env, task)
            state, _ = self.env.reset(seed=task['seed'])
            log_probs = []
            rewards = []
            done = truncated = False

            for _ in range(self.horizon):
                state = torch.from_numpy(state.flatten()).float()
                probs = functional_call(self.policy, (params, buffers), state)
                action = select_action(probs.detach())
                log_prob = torch.log(probs[action])
                next_state, reward, done, truncated, _ = self.env.step(action)
                state = next_state

                log_probs.append(log_prob)
                rewards.append(reward)

                if done or truncated:
                    break 

            discounted_rewards = compute_discounted_rewards(rewards, 0.99)
            discounted_rewards = torch.tensor(discounted_rewards)
            # Normalize rewards
            discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-9)

            loss = []
            for log_prob, reward in zip(log_probs, discounted_rewards):
                loss.append(-log_prob * reward)
            
            total_inner_loss.append(torch.stack(loss).sum())
        
        return torch.stack(total_inner_loss).mean()

    def train_step(self, train_tasks):
        meta_losses = []
        self.policy.train()
        self.meta_optimizer.zero_grad()  # Reset meta-gradients
        
        for task in train_tasks:
            params = dict(self.policy.named_parameters())
            buffers = dict(self.policy.named_buffers())
            
            # Inner Loop
            if self.first_order:
                inner_loss = self._inner_loss(params, buffers, task)
                grads = torch.autograd.grad(inner_loss, params.values(), create_graph=False) # create_graph=False for first order
                # Update the model parameters, inner SGD step
                params = {k: v - self.inner_lr * g for (k, v), g in zip(params.items(), grads)}
            else:
                grads = grad(self._inner_loss)(params, buffers, task)
                # Update the model parameters, inner SGD step
                params = {k: params[k] - g * self.inner_lr for k, g, in grads.items()}

            # Outer Loop loss accumulation
            meta_loss = self._inner_loss(params, buffers, task)
            meta_losses.append(meta_loss)
        
        # Outer Loop update
        meta_loss = torch.stack(meta_losses).mean()
        meta_loss.backward()
        self.meta_optimizer.step()

        return meta_loss.item()
            
    def evaluate(self, eval_tasks):
        eval_returns = []
        self.policy.eval()

        for task in eval_tasks:
            set_env_task(self.env, task)
            state, _ = self.env.reset(seed=task['seed'])
            
            done = truncated = False
            total_reward = 0

            for _ in range(self.horizon):
                state = torch.from_numpy(state.flatten()).float()
                probs = self.policy(state)
                action = select_action(probs.detach())
                next_state, reward, done, truncated, _ = self.env.step(action)
                state = next_state
                total_reward += reward
                if done or truncated:
                    break

            eval_returns.append(total_reward)

        return np.mean(eval_returns)
        
    def train_and_evaluate(self, train_loader, eval_loader):
        
        for iteration in range(self.num_iterations):
            tr_meta_loss = self.train_step(train_loader)
            vl_returns = self.evaluate(eval_loader)

            self.log["tr_loss"].append(tr_meta_loss)
            self.log["vl_return"].append(vl_returns)

            # Print performance
            print(
                f'[Epoch {iteration}] | ',
                f'[TR] Meta Loss: {tr_meta_loss:.2f} -',
                f'[VL] Returns: {vl_returns:.2f} -  ')

            # Model checkpointing
            if vl_returns > self.best_return:  
                self.best_return = vl_returns
                torch.save({
                    'epoch': iteration,
                    'model_state_dict': self.policy.state_dict(),
                    'optimizer_state_dict': self.meta_optimizer.state_dict(),
                    'train_loss': tr_meta_loss,
                    'val_return': vl_returns,
                }, f"checkpoints/{self.save_path}_model.pth")
            
            # Early stopping if task is solved
            if vl_returns >= self.return_threshold:
                print(f"Task solved in {iteration} iterations.")
                break

        return self.log