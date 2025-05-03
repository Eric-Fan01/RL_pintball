# models/muzero_network.py

import torch
import torch.nn as nn
from utils.network import Representation, Dynamics, Prediction

class MuZeroNet(nn.Module):
    def __init__(self, observation_shape=(4, 84, 84), hidden_dim=16, action_space_size=9, support_size=21):
        super().__init__()
        self.representation_net = Representation(in_ch=observation_shape[0], hidden_ch=hidden_dim)
        self.dynamics_net = Dynamics(ch=hidden_dim, action_dim=action_space_size)
        self.prediction_net = Prediction(ch=hidden_dim, action_dim=action_space_size, support_size=support_size)

    def representation(self, obs):
        """
        obs: tensor (batch_size, C, H, W)
        """
        return self.representation_net(obs)

    def dynamics(self, state, action_plane):
        """
        state: tensor (batch_size, C, H, W)
        action_plane: tensor (batch_size, 1, H, W)
        """
        return self.dynamics_net(state, action_plane)

    def prediction(self, state):
        """
        state: tensor (batch_size, C, H, W)
        """
        return self.prediction_net(state)

if __name__ == "__main__":

    network = MuZeroNet(observation_shape=(4, 84, 84), hidden_dim=16)
