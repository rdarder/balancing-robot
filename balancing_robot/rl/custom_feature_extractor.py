# /home/rdarder/dev/balancing-robot/custom_feature_extractor.py
import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class GRUFeatureExtractor(BaseFeaturesExtractor):
    """
    Feature extractor that uses a GRU to process sequential observations.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 64, hidden_size: int = 64, num_layers: int = 1):
        super().__init__(observation_space, features_dim=features_dim)

        # Assuming observation space is (FRAME_STACK_SIZE, observation_dimension)
        # Extract the dimension of a single observation frame (e.g., 8 in your case)
        input_dim = observation_space.shape[-1]  # Last dimension of the shape

        self.gru = th.nn.GRU(
            input_size=input_dim,  # Dimension of each input frame
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,  # Input and output tensors are provided as (batch, seq, feature)
        )
        self.fc = th.nn.Linear(hidden_size, features_dim)  # Fully connected layer after GRU
        self.relu = th.nn.ReLU()

    def forward(self, observations: th.Tensor) -> th.Tensor:
        # observations shape: (batch_size, FRAME_STACK_SIZE, observation_dimension)

        # Pass the observations through the GRU
        gru_out, _ = self.gru(observations)  # gru_out: (batch_size, FRAME_STACK_SIZE, hidden_size)

        # We can take the output from the last time step of the GRU
        # or you could experiment with averaging or max-pooling over the time steps.
        # Here we take the last time step:
        last_timestep_output = gru_out[:, -1, :]  # Shape: (batch_size, hidden_size)

        # Pass through a fully connected layer and ReLU activation
        features = self.relu(self.fc(last_timestep_output))  # Shape: (batch_size, features_dim)
        return features


if __name__ == '__main__':
    # Example Usage and Testing (optional - you can run this file to test)
    # Create a dummy observation space (replace with your actual observation space)
    dummy_observation_space = gym.spaces.Box(low=-np.inf, high=np.inf, shape=(8, 8), dtype=np.float32)

    # Instantiate the GRU feature extractor
    gru_extractor = GRUFeatureExtractor(dummy_observation_space, features_dim=128, hidden_size=64, num_layers=2)

    # Generate a dummy batch of observations (batch_size=2, FRAME_STACK_SIZE=8, obs_dim=8)
    dummy_observations = th.randn(2, 8, 8)

    # Pass the dummy observations through the feature extractor
    extracted_features = gru_extractor(dummy_observations)

    # Print the output shape to verify
    print("Input observations shape:", dummy_observations.shape)
    print("Extracted features shape:", extracted_features.shape) # Expected: torch.Size([2, 128])