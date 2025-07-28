import stable_baselines3
import torch
import torchvision
import random
import numpy as np
import gymnasium as gym
import torchvision.models as models
import torchvision.transforms as transforms
import torch.nn as nn
from gymnasium import spaces

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


'''
Class: EfficientNet
Description: This class implements a feature extractor using the EfficientNet architecture.
It processes images and other observation spaces, extracting features for reinforcement learning tasks.
Parameters:
    observation_space (gym.spaces.Dict): The observation space containing different types of observations.
Returns:
    None
'''
class EfficientNet(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict):
        # We do not know features-dim here before going over all the items,
        # so put something dummy for now. PyTorch requires calling
        # nn.Module.__init__ before adding modules
        super().__init__(observation_space, features_dim=1)

        extractors = {}

        total_concat_size = 0
        # We need to know size of the output of this extractor,
        # so go over all the spaces and compute output feature sizes
        for key, subspace in observation_space.spaces.items():
            if key == 'image':
                # Use a randomly initialized EfficientNet model
                net_arch = models.efficientnet_b0(weights=None)
                net_arch.classifier[1] = nn.Linear(net_arch.classifier[1].in_features, 512)

                # Pass the image through the EfficientNet model
                extractors[key] = nn.Sequential(
                    net_arch,
                    nn.ReLU(),
                    nn.Flatten()
                )

                total_concat_size += 512
            elif key == 'outputs':
                # Run through a simple MLP
                extractors[key] = nn.Linear(subspace.shape[0], 10)
                total_concat_size += 10
            elif key == 'source_class':
                # Run through the Embedding layer
                extractors[key] = nn.Linear(10, 10)
                total_concat_size += 10

        self.extractors = nn.ModuleDict(extractors)

        # Update the features dim manually
        self._features_dim = total_concat_size

    def forward(self, observations) -> torch.Tensor:
        encoded_tensor_list = []

        # self.extractors contain nn.Modules that do all the processing.
        for key, extractor in self.extractors.items():
            if key == 'image':
                encoded_tensor_list.append(extractor(observations[key]))
            elif key == 'outputs':
                encoded_tensor_list.append(extractor(observations[key]))
            elif key == 'source_class':
                if len(observations[key].shape) == 3:
                    observations[key] = observations[key].squeeze(1)
                encoded_tensor_list.append(extractor(observations[key]))
        # Return a (B, self._features_dim) PyTorch tensor, where B is batch dimension.
        return torch.cat(encoded_tensor_list, dim=1)

'''
Class: PPOAgent
Description: This class implements a PPO agent for adversarial attacks using reinforcement learning. A policy head and feature extractor is implemented with training and evaluation methods.
It uses the Stable Baselines3 library for the PPO implementation and PyTorch for the neural network architecture.
'''
class PPOAgent:
    def __init__(self, feature_extractor, policy_head, env, **kwargs):
        self.feature_extractor = feature_extractor
        self.policy_head = policy_head
        self.env = env

        # Define the policy head architecture and activation function (defined as MLPs, refer to StableBaselines3 documentation)
        if self.policy_head == 'default':
            net_arch = dict(pi=[256,256], vf=[256,256])
            activation_fn = torch.nn.ReLU
        else:
            raise NotImplementedError("Policy head {} not valid.".format(self.policy_head))
        
        # Define the feature extractor architecture (refer to StableBaselines3 documentation)
        if self.feature_extractor == 'efficientnet_b0':
            if net_arch is not None:
                policy_kwargs = dict(
                    features_extractor_class=EfficientNet,
                    features_extractor_kwargs=dict(),
                    net_arch=net_arch,
                    activation_fn=activation_fn
                )
            else:
                policy_kwargs = dict(
                    features_extractor_class=EfficientNet,
                    features_extractor_kwargs=dict()
                )
            self.model = stable_baselines3.PPO('MultiInputPolicy', env, policy_kwargs=policy_kwargs, **kwargs)
        else:
            raise NotImplementedError("Feature extractor {} not valid.".format(self.feature_extractor))

    '''
    Method: train
    Description: This method trains the PPO agent using the specified environment and parameters.
    Parameters:
        steps (int): The number of training steps.
        callback (callable): A callback function to be called at each step.
        log_interval (int): The interval for logging training progress.
        reset_num_timesteps (bool): Whether to reset the number of timesteps.
        progress_bar (bool): Whether to show a progress bar during training.
    '''
    def train(self, steps: int, callback=None, log_interval=1, reset_num_timesteps=False, progress_bar=False):
        # Train the Model using the stable_baselines3 PPO implementation
        self.model.learn(total_timesteps=steps, callback=callback, log_interval=log_interval, reset_num_timesteps=reset_num_timesteps, progress_bar=progress_bar)

    '''
    Method: predict
    Description: This method predicts the action to take given an observation.
    Parameters:
        observation (np.ndarray): The observation from the environment.
        deterministic (bool): Whether to use a deterministic policy.
    Returns:
        action (np.ndarray): The predicted action.
    '''
    def predict(self, observation: np.ndarray, deterministic: bool = False):
        # Get the action from the model
        action, _ = self.model.predict(observation, deterministic=deterministic)
        return action
    
    '''
    Method: save
    Description: This method saves the trained PPO agent to a specified file.
    Parameters:
        filename (str): The name of the file to save the agent to.
    '''
    def save(self, filename: str):
        self.model.save(filename)