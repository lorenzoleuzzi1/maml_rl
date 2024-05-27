# maml_rl

This repository contains an implementation of Model-Agnostic Meta-Learning (MAML) for Reinforcement Learning (RL). The primary focus is on applying MAML to three control task environments: CartPole, Acrobot, and Highway. This implementation is designed to work with any observation space, but the action space must be discrete. This project was developed as part of the Continual Learning course (23/24) at the University of Pisa.

## Overview

Model-Agnostic Meta-Learning (MAML) is a meta-learning algorithm designed to adapt quickly to new tasks with a small number of training examples. In reinforcement learning, MAML enables agents to quickly adapt to new environment configurations.

This implementation uses the vanilla policy gradient method (REINFORCE) as the base RL algorithm. The environments used for testing the implementation are:
- CartPole ('cartpole')
- Acrobot ('acrobot')
- Highway ('highway')

Presentation of the project can be found [here](https://docs.google.com/presentation/d/1YgyzeKkvxEmcL7m0gFkcneBGap2cjnwaOhP9mqCEqDw/edit?usp=sharing)

## Installation

To run this code, you need to have Python 3.x installed along with the necessary dependencies. You can install the required packages using the following command:

```bash
pip install -r requirements.txt
```

## Usage

### Training with REINFORCE

To train a policy using the vanilla policy gradient REINFORCE algorithm, you can use the `reinforce.py` script. This script also allows you to specify various parameters for the training process. Below is an example command to start training:

```bash
python reinforce.py --env cartpole --lr 1e-3 --num_episodes 1000 --seed 42
```
**reinforce.py**:
- `--env`: Environment ID (default: 'cartpole')
- `--lr`: Learning rate (default: 1e-3)
- `--num_episodes`: Number of episodes (default: 1000)
- `--seed`: Seed (default: 42)

### Training with MAML

To train the model using MAML, you can use the `train_maml.py` script. This script allows you to specify various parameters for the training process. Below is an example command to start training:

```bash
python train_maml.py --env cartpole --model_name vpg --inner_lr 1e-2 --meta_lr 1e-3 --num_iterations 100 --num_trajectories 10 --horizon 500 --batch_tasks_size 10  --seed 42
```
**train_maml.py**:
- `--env`: Environment ID (default: 'cartpole')
- `--model_name`: Model name (default: 'vpg')
- `--fo`: First order MAML (default: True)
- `--inner_lr`: Inner learning rate (default: 1e-2)
- `--meta_lr`: Meta learning rate (default: 1e-3)
- `--num_iterations`: Number of iterations (default: 100)
- `--num_trajectories`: Number of sampled trajectories (default: 10)
- `--horizon`: Horizon (default: 500)
- `--batch_tasks_size`: Batch tasks size (default: 10)
- `--seed`: Seed (default: 42)

### Testing with MAML

To test a model trained with MAML, you can use the `test_maml.py` script. This script allows you to evaluate the performance of the trained model on the specified environment. Below is an example command for testing:

```bash
python test_maml.py --env cartpole --model_name vpg --run True --compare False --visualize False --lr 1e-3 --num_episodes 1000 --num_tasks 10 --seed 42
```
**test_maml.py**:
- `--env`: Environment ID (default: 'cartpole')
- `--model_name`: Path to the model checkpoint (default: 'vpg')
- `--run`: Run the test and store data results (default: False)
- `--compare`: Compare the results of the test (default: False)
- `--visualize`: Visualize the policy (default: False)
- `--lr`: Learning rate (default: 1e-3)
- `--num_episodes`: Number of episodes (default: 1000)
- `--num_tasks`: Batch tasks size (default: 10)
- `--seed`: Seed (default: 42)

## Results

Results from the training and evaluation processes can be found in following folders:
- `plots`: Training curves and initialization comparisons.
- `policy_visualizations`: GIFs of the policy visualizations in the environment.