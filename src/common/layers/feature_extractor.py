from typing import Tuple
import gymnasium as gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import torch.nn as nn



class CustomFeaturesExtractor(BaseFeaturesExtractor):
    def __init__(
        self,
        observation_space: gym.Space,
        features_dim: int,
        model: nn.Module, 
        ):
        
        super().__init__(
            observation_space=observation_space,
            features_dim=features_dim
        )
        
        self.model = model(
            in_shape=self._observation_space.shape,
            features_dim=features_dim,
        )
               
    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)
    
    
    