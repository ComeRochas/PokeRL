"""
Neural network architectures for RL agents.

This module provides various deep learning architectures that can be easily swapped.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNetwork(ABC, nn.Module):
    """Base class for neural networks."""
    
    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        pass


class DQNNetwork(BaseNetwork):
    """
    Deep Q-Network architecture.
    
    Simple feedforward network for Q-learning.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [256, 256, 128],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ):
        super(DQNNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm
        
        # Choose activation function
        self.activation = self._get_activation(activation)
        
        # Build network layers
        layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))
            
            layers.append(self.activation)
            
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        # Output layer
        layers.append(nn.Linear(prev_size, output_size))
        
        self.network = nn.Sequential(*layers)
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.network(x)


class DuelingDQNNetwork(BaseNetwork):
    """
    Dueling DQN architecture.
    
    Separates value and advantage streams for better learning.
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ):
        super(DuelingDQNNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation = self._get_activation(activation)
        
        # Shared feature layers
        shared_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            shared_layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                shared_layers.append(nn.BatchNorm1d(hidden_size))
            
            shared_layers.append(self.activation)
            
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Value stream
        self.value_stream = nn.Sequential(
            nn.Linear(prev_size, 128),
            self.activation,
            nn.Linear(128, 1)
        )
        
        # Advantage stream
        self.advantage_stream = nn.Sequential(
            nn.Linear(prev_size, 128),
            self.activation,
            nn.Linear(128, output_size)
        )
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with dueling streams."""
        features = self.shared_network(x)
        
        value = self.value_stream(features)
        advantage = self.advantage_stream(features)
        
        # Combine value and advantage
        # Q(s,a) = V(s) + (A(s,a) - mean(A(s,a)))
        q_values = value + (advantage - advantage.mean(dim=1, keepdim=True))
        
        return q_values


class ActorCriticNetwork(BaseNetwork):
    """
    Actor-Critic network for policy gradient methods (PPO, A2C).
    
    Outputs both policy (actor) and value estimate (critic).
    """
    
    def __init__(
        self,
        input_size: int,
        output_size: int,
        hidden_layers: List[int] = [256, 256],
        activation: str = "relu",
        dropout: float = 0.0,
        use_batch_norm: bool = False
    ):
        super(ActorCriticNetwork, self).__init__()
        
        self.input_size = input_size
        self.output_size = output_size
        self.activation = self._get_activation(activation)
        
        # Shared feature layers
        shared_layers = []
        prev_size = input_size
        
        for hidden_size in hidden_layers:
            shared_layers.append(nn.Linear(prev_size, hidden_size))
            
            if use_batch_norm:
                shared_layers.append(nn.BatchNorm1d(hidden_size))
            
            shared_layers.append(self.activation)
            
            if dropout > 0:
                shared_layers.append(nn.Dropout(dropout))
            
            prev_size = hidden_size
        
        self.shared_network = nn.Sequential(*shared_layers)
        
        # Actor head (policy)
        self.actor = nn.Sequential(
            nn.Linear(prev_size, 128),
            self.activation,
            nn.Linear(128, output_size)
        )
        
        # Critic head (value)
        self.critic = nn.Sequential(
            nn.Linear(prev_size, 128),
            self.activation,
            nn.Linear(128, 1)
        )
    
    def _get_activation(self, activation: str):
        """Get activation function."""
        activations = {
            "relu": nn.ReLU(),
            "tanh": nn.Tanh(),
            "elu": nn.ELU(),
            "leaky_relu": nn.LeakyReLU()
        }
        return activations.get(activation, nn.ReLU())
    
    def forward(self, x: torch.Tensor) -> tuple:
        """
        Forward pass.
        
        Returns:
            policy_logits: Logits for action probabilities
            value: State value estimate
        """
        features = self.shared_network(x)
        
        policy_logits = self.actor(features)
        value = self.critic(features)
        
        return policy_logits, value
    
    def get_action_and_value(self, x: torch.Tensor, action=None):
        """
        Get action probabilities and value estimate.
        
        Used during training for PPO.
        """
        policy_logits, value = self.forward(x)
        probs = F.softmax(policy_logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        if action is None:
            action = dist.sample()
        
        return action, dist.log_prob(action), dist.entropy(), value


def get_network(
    network_type: str,
    input_size: int,
    output_size: int,
    config: Dict[str, Any]
) -> BaseNetwork:
    """
    Factory function to create network from config.
    
    Args:
        network_type: Type of network to create
        input_size: Input dimension
        output_size: Output dimension (number of actions)
        config: Configuration dictionary
        
    Returns:
        Network instance
    """
    hidden_layers = config.get("hidden_layers", [256, 256, 128])
    activation = config.get("activation", "relu")
    dropout = config.get("dropout", 0.0)
    use_batch_norm = config.get("use_batch_norm", False)
    
    if network_type == "dqn":
        return DQNNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
    elif network_type == "dueling_dqn":
        return DuelingDQNNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
    elif network_type in ["ppo", "a2c"]:
        return ActorCriticNetwork(
            input_size=input_size,
            output_size=output_size,
            hidden_layers=hidden_layers,
            activation=activation,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
    else:
        raise ValueError(f"Unknown network type: {network_type}")
