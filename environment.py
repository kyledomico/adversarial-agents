import stable_baselines3
import gymnasium as gym
import numpy as np
import random
import torch
import torchvision
from PIL import Image
import math
import os
import torch

class EvasionEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self, victim_model, dataset, N, theta, rl_attack_config: dict):

        super(EvasionEnv, self).__init__()
        self.victim_model = victim_model
        self.dataset = dataset
        self.N = N
        self.theta = theta
        self.rl_attack_config = rl_attack_config
        self.victim_model.eval()

        # Get device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.victim_model.to(self.device)

        # PIL for visualization
        self.to_pil = torchvision.transforms.ToPILImage()

        # Set logging directory
        self.log_dir = 'learning_results'

        # Initialize the preprocess transform for the pytorch vision models
        self.preprocess = torch.nn.Sequential(
            torchvision.transforms.Resize(256, antialias=True),
            torchvision.transforms.CenterCrop(224),
            torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        )
        
        # Initialize the observation and action spaces 
        self.observation_space = gym.spaces.Dict({'image': gym.spaces.Box(low=-np.inf, high=np.inf, shape=(3, 224, 224), dtype=np.float32), 'outputs': gym.spaces.Box(low=0.0, high=1.0, shape=(10,), dtype=np.float32), 'source_class': gym.spaces.Discrete(10)})
        self.action_space = gym.spaces.Box(low=-2.0, high=2.0, shape=(self.N*(5),), dtype=np.float32) # Each N has (x, y, c1, c2, c3) where x and y are the coordinates and c1, c2, c3 are the perturbations
        
    def project(self, perturbations):
        l2_norm = torch.norm(perturbations, p=2)
        perturbations = perturbations / l2_norm * self.rl_attack_config['epsilon']
        self.sample = self.sample_point[0] + perturbations
        
    def generate_query(self, action):
        # (N,\theta) Configuration
        for i in range(0, self.N*(5), 5): # Each action consists of (x, y, c1, c2, c3)
            x = min(int((action[i] + 2.0) * 32 / 4.0), 31) # Assuming 32x32 input size for CIFAR-10 and SVHN
            y = min(int((action[i + 1] + 2.0) * 32 / 4.0), 31) # Assuming 32x32 input size for CIFAR-10 and SVHN
            for c in range(3): # Assuming RGB channels
                distortion = action[i + 2 + c] * self.theta / 2.0
                self.sample[c, x, y] = min(1.0, max(self.sample[c, x, y] + distortion, 0.0))

        # Project the perturbations if the attack is RL Max Loss (to epsilon distortion)
        if self.rl_attack_config['type'] == 'RLMaxLoss':
            perturbations = torch.clone(self.sample - self.sample_point[0])
            lp_norm = torch.norm(perturbations, p=2)
            if lp_norm > self.rl_attack_config['epsilon']:
                self.project(perturbations)
                self.movement = self.rl_attack_config['epsilon']
            else:
                self.movement = torch.norm(self.sample - self.sample_point[0], p=2).item()
        
        return self.sample
    
    def dynamics(self, x):
        # Get the model outputs, apply softmax if necessary
        outputs = self.victim_model(self.preprocess(x).to(self.device).unsqueeze(0))[0].detach()
        if torch.sum(outputs).item() != 1:
            outputs = torch.nn.functional.softmax(outputs, dim=0)

        # Get the value indexed by the sample point, and the classification
        confidence = outputs[self.sample_point[1]].item()
        prediction = torch.argmax(outputs).item()

        # Get the entropy loss (negative log of the confidence)
        loss = -torch.log(torch.tensor(confidence + 1e-10)).item()  # Adding a small value to avoid log(0)

        # Return the loss, prediction, and outputs
        return loss, prediction, outputs

    def step(self, action):
        # Generate the query
        self.sample = self.generate_query(action)

        # Get the movement
        self.movement = torch.norm(self.sample - self.sample_point[0], p=2).item()

        # Get the model loss, prediction, and outputs
        model_loss, prediction, outputs = self.dynamics(self.sample)
        self.queries += 1

        # Calculate the reward
        if self.rl_attack_config['type'] == 'RLMinNorm':
            reward = self.rl_attack_config['c'] * (self.cumulative_lp - self.movement) + 20.0 * (model_loss - self.model_loss) # Scale the model loss to avoid negligible differences in confidence for effective RL
        elif self.rl_attack_config['type'] == 'RLMaxLoss':
            reward = 20.0 * (model_loss - self.model_loss) # Scale the model loss to avoid negligible differences in confidence for effective RL
        else:
            raise ValueError('Invalid reward type.')

        # Employ Hill Climbing
        if reward < 0:
            reward = 0.0
            self.sample = torch.clone(self.s_tminus1)
        else:
            self.model_loss = model_loss
            self.cumulative_lp = self.movement
            self.s_tminus1 = torch.clone(self.sample)

        # Determine the done condition
        if prediction != self.sample_point[1]:
            done = True
        else:
            done = False

        # Determine the truncated condition
        if self.queries >= 1000: # Assuming a maximum of 1000 queries per episode
            truncated = True
        else:
            truncated = False

        # Update global variables
        self.prediction = prediction
        self.outputs = outputs
        self.cumulative_reward += reward
        self.discounted_reward += reward * (0.99 ** (self.queries - 2)) # Discounted reward with gamma = 0.99

        # If done or truncated, write to the episodic data file
        if done or truncated:
            with open('{}/episodic.csv'.format(self.log_dir), 'a') as f:
                f.write('{},{},{},{},{},{},{},{}\n'.format(self.queries, self.start_state_model_loss, self.model_loss, self.cumulative_reward, self.discounted_reward, self.sample_point[1], self.prediction, self.cumulative_lp))
        
        # Get the info
        info = self.get_info()

        # Return the observation, reward, done, truncated, and info
        final_state = self.get_obs()

        return final_state, reward, done, truncated, info

    def get_obs(self):
        return {'image': self.preprocess(self.sample), 'outputs': self.outputs.to('cpu'), 'source_class': self.sample_point[1]}

    def get_info(self):
        return {"image_size": self.sample.shape, "movement": self.movement, 'prediction': self.prediction, 'original_label': self.sample_point[1]}

    def reset(self, seed=42):
        # Reset the state of the environment to an initial state
        super().reset(seed=seed)
        
        # Randomly sample a sample from the dataset
        self.sample_point = None
        while self.sample_point is None:
            self.index = np.random.randint(0, len(self.dataset))
            x = self.dataset[self.index][0]
            y = self.dataset[self.index][1]
            if self.victim_model(torch.clone(x).to(self.device).unsqueeze(0)).argmax().item() == y:
                self.sample_point = (x, y)
        self.sample = torch.clone(self.sample_point[0])

        # Initialize Cumulative Lp Distance
        self.cumulative_lp = 0.0

        # Initialize log confidence of the original sample
        self.model_loss, self.prediction, self.outputs = self.dynamics(self.sample)

        # Record Original model loss
        self.start_state_model_loss = self.model_loss
        self.misclass = False

        # Initialize the Timestep
        self.queries = 1
        
        # Initialize the movement and the previous state
        self.movement = 0.0
        self.s_tminus1 = torch.clone(self.sample)

        # Initialize the Cumulative Reward
        self.cumulative_reward = 0.0
        self.discounted_reward = 0.0

        # Return the initial observation and info
        return self.get_obs(), self.get_info()
    
    def setImage(self, sample_point):
        # Set the sample point to the given image and reset the environment
        self.sample_point = sample_point
        self.sample = torch.clone(self.sample_point[0])

        # Initialize log confidence of the original sample
        self.model_loss, self.prediction, self.outputs = self.dynamics(self.sample)

        # Initialize Cumulative Lp Distance
        self.cumulative_lp = 0.0

        # Record Original model loss
        self.start_state_model_loss = self.model_loss
        self.misclass = False

        # Initialize the Timestep
        self.queries = 1
        
        # Initialize the movement and the previous state
        self.movement = 0.0
        self.s_tminus1 = torch.clone(self.sample)

        # Initialize the Cumulative Reward
        self.cumulative_reward = 0.0
        self.discounted_reward = 0.0

        self.log_dir = 'evaluation_results'

        # Return the initial observation and info
        return self.get_obs(), self.get_info()

    def render(self, mode='human'):
        # convert the normalized torch tensor to an integer numpy array corresponding to the image
        image = (self.sample_point[0] * 255).squeeze(0).type(torch.uint8)
        game_image = (self.sample * 255).squeeze(0).type(torch.uint8)

        return image, game_image
    
    def close(self):
        pass