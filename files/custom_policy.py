# custom_policy.py
import torch
import torch.nn as nn
from gymnasium import spaces
from typing import Dict

from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

class NatureCNN(BaseFeaturesExtractor):
    """이미지 입력을 처리하는 CNN 부분"""
    def __init__(self, observation_space: spaces.Box, features_dim: int = 512):
        super().__init__(observation_space, features_dim)
        n_input_channels = observation_space.shape[0]
        self.cnn = nn.Sequential(
            nn.Conv2d(n_input_channels, 32, kernel_size=8, stride=4, padding=0),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=0),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=0),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            n_flatten = self.cnn(torch.as_tensor(observation_space.sample()[None]).float()).shape[1]
        self.linear = nn.Sequential(nn.Linear(n_flatten, features_dim), nn.ReLU())

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.linear(self.cnn(observations))

class CombinedExtractor(BaseFeaturesExtractor):
    """
    Dict 관측 공간을 처리하기 위한 특징 추출기.
    이미지는 CNN으로, 상태 벡터는 MLP로 처리한 뒤 결과를 합칩니다.
    """
    def __init__(self, observation_space: spaces.Dict):
        # 합쳐진 특징 벡터의 최종 차원을 계산합니다.
        cnn_output_dim = 512
        mlp_output_dim = 64
        total_features_dim = cnn_output_dim + mlp_output_dim
        
        super().__init__(observation_space, features_dim=total_features_dim)

        # 각 관측 공간에 맞는 추출기를 생성합니다.
        self.cnn = NatureCNN(observation_space["image"], features_dim=cnn_output_dim)
        
        state_space_shape = observation_space["state"].shape[0]
        self.mlp = nn.Sequential(
            nn.Linear(state_space_shape, 128),
            nn.ReLU(),
            nn.Linear(128, mlp_output_dim),
            nn.ReLU(),
        )

    def forward(self, observations: Dict[str, torch.Tensor]) -> torch.Tensor:
        # 각 데이터를 해당 추출기로 처리합니다.
        image_features = self.cnn(observations["image"])
        state_features = self.mlp(observations["state"])
        
        # 두 특징 벡터를 하나로 합쳐서 반환합니다.
        return torch.cat([image_features, state_features], dim=1)