import stable_baselines3
import gymnasium as gym
import numpy as np
import torch
import torchvision
import argparse
import os
import random
from environment import EvasionEnv
from ppo_agent import PPOAgent
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure

# Create a parser to parse command line arguments
parser = argparse.ArgumentParser(description='Perform a black-box evasion attack with reinforcement learning.')

# Dataset Argument
parser.add_argument('--dataset', type=str, default='cifar10', help='The dataset to use.')

# Environment Arguments
parser.add_argument('--rl_attack_type', type=str, default='RLMaxLoss', help='The type of RL attack to perform.')
parser.add_argument('--epsilon', type=float, default=0.35, help='The epsilon value for the RL Max Loss attack.')
parser.add_argument('--c', type=float, default=1.0, help='The c value for the RL Min Norm attack.')
parser.add_argument('--N', type=int, default=5, help='The number of actions to take in the environment.')
parser.add_argument('--theta', type=float, default=0.05, help='The theta value for the RL attack.')

# Training Arguments
parser.add_argument('--training_steps', type=int, default=5_000_000, help='The number of steps to run the attack for.')
parser.add_argument('--n_envs', type=int, default=32, help='The number of environments to use.')

# Label (for saving the trained adversarial agent)
parser.add_argument('--label', type=str, default='agent0', help='The label for saving the trained adversarial agent.')

# Parse the arguments
args = parser.parse_args()

# Set the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Set the random seed for reproducibility
random.seed(42)
torch.manual_seed(42)
np.random.seed(42)

# Make a directory called learning_results if it does not exist (for logging)
if not os.path.exists('learning_results'):
    os.makedirs('learning_results')

# Load the victim model (ResNet50)
victim_model = torchvision.models.resnet50(weights=None)
victim_model.fc = torch.nn.Linear(victim_model.fc.in_features, 10)  # Assuming 10 classes for CIFAR-10/SVHN
victim_model.to(device)

# Load the victim model weights (from the ./victim_models/ directory)
state_dict = torch.load('./victim_models/{}+resnet50.pth'.format(args.dataset), map_location=device)
victim_model.load_state_dict(state_dict)

# Set the victim model to evaluation mode
victim_model.eval()

# Load the RL training dataset D (50% of the original test dataset)
raw_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
])
if args.dataset == 'cifar10':
    dataset = torchvision.datasets.CIFAR10(root='./dataset', train=False, download=True, transform=raw_transform)
elif args.dataset == 'svhn':
    dataset = torchvision.datasets.SVHN(root='./dataset', split='test', download=True, transform=raw_transform)
else:
    raise ValueError(f"Dataset {dataset_name} not supported.")
dataset = [(datapoint[0], datapoint[1]) for datapoint in dataset] # Convert to list of tuples (image, label)
split_index = int(len(dataset) * 0.5)
D = dataset[:split_index]

# Create the attack configuration
rl_attack_config = {
    'type': args.rl_attack_type,
    'epsilon': args.epsilon,
    'c': args.c
}

# Create the environment(s) for the RL attack (n_envs parallel environments)
def make_env():
    """
    Utility function for multiprocessed env.

    :param env_id: the environment ID
    :param num_env: the number of environments you wish to have in subprocesses
    :param seed: the initial seed for RNG
    :param rank: index of the subprocess
    """
    def _init():
        env = EvasionEnv(victim_model, D, args.N, args.theta, rl_attack_config)
        env = Monitor(env)
        return env
    return _init

training_env = DummyVecEnv([make_env() for i in range(args.n_envs)])

# Create the logger and the attacker
logger = configure("./learning_results/", ["stdout", "csv"])
adversarial_agent = PPOAgent(feature_extractor='efficientnet_b0',
                            policy_head='default',
                            env=training_env,
                            learning_rate=lambda x: x * 0.0025, # Learning rate schedule
                            n_steps=128, # Number of steps to run in each environment before updating the policy (128 * n_envs)
                            batch_size=64,
                            n_epochs=10,
                            gamma=0.99,
                            clip_range=lambda x: x * 0.1 # Clip range schedule
                            )
adversarial_agent.model.set_logger(logger)

# Write the episodic data file header
with open('learning_results/episodic.csv', 'w') as f:
    f.write('queries,f_start,f_end,cumulative_reward,discounted_reward,label,prediction,distortion\n')

# Train the adversarial agent
adversarial_agent.train(steps=args.training_steps)

# Make sure the agents directory exists
if not os.path.exists('agents'):
    os.makedirs('agents')

# Save the attacker policy to a file - {args.label}.pth
torch.save(adversarial_agent.model.policy.state_dict(), './agents/{}.pth'.format(args.label))