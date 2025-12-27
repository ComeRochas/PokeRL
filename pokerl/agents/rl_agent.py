"""
RL Agent implementation for Pokemon battles.

This module wraps the poke-env Player class with RL capabilities.
"""

from typing import Dict, Any, Optional
import numpy as np
import torch
from collections import deque
import random

from poke_env.player import Player
from poke_env.environment import Battle

from pokerl.observations.observation_space import ObservationSpace
from pokerl.rewards.reward_functions import RewardFunction
from pokerl.networks.architectures import BaseNetwork


class RLAgent(Player):
    """
    Reinforcement Learning agent for Pokemon battles.
    
    This class extends poke-env's Player to add RL capabilities.
    """
    
    def __init__(
        self,
        observation_space: ObservationSpace,
        reward_function: RewardFunction,
        network: BaseNetwork,
        battle_format: str = "gen8randombattle",
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.05,
        device: str = "cpu",
        **kwargs
    ):
        super().__init__(battle_format=battle_format, **kwargs)
        
        self.observation_space = observation_space
        self.reward_function = reward_function
        self.network = network
        self.device = device
        
        # Epsilon-greedy exploration
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Training state
        self.training_mode = True
        self.previous_battle_states = {}  # Store previous battle state for reward calculation
        
        # Move network to device
        self.network.to(self.device)
    
    def choose_move(self, battle: Battle) -> str:
        """
        Choose a move based on current policy.
        
        This is called by poke-env during battles.
        """
        # Get observation
        observation = self.observation_space.get_observation(battle)
        observation_tensor = torch.FloatTensor(observation).unsqueeze(0).to(self.device)
        
        # Epsilon-greedy action selection
        if self.training_mode and random.random() < self.epsilon:
            # Random action
            action_idx = random.randint(0, len(battle.available_moves) + len(battle.available_switches) - 1)
        else:
            # Greedy action
            with torch.no_grad():
                q_values = self.network(observation_tensor)
                action_idx = q_values.argmax().item()
        
        # Convert action index to battle action
        action = self._action_to_move(battle, action_idx)
        
        # Store battle state for reward calculation
        self.previous_battle_states[battle.battle_tag] = self._copy_battle_state(battle)
        
        return action
    
    def _action_to_move(self, battle: Battle, action_idx: int) -> str:
        """
        Convert action index to poke-env move string.
        
        Action space:
        - First 4 actions: moves (if available)
        - Next 5 actions: switches (if available)
        """
        n_moves = len(battle.available_moves)
        n_switches = len(battle.available_switches)
        
        if action_idx < n_moves:
            # Use a move
            return self.create_order(list(battle.available_moves)[action_idx])
        elif action_idx < n_moves + n_switches:
            # Switch Pokemon
            switch_idx = action_idx - n_moves
            return self.create_order(list(battle.available_switches)[switch_idx])
        else:
            # Invalid action, default to random valid move
            if battle.available_moves:
                return self.create_order(random.choice(list(battle.available_moves)))
            elif battle.available_switches:
                return self.create_order(random.choice(list(battle.available_switches)))
            else:
                return self.choose_default_move()
    
    def _copy_battle_state(self, battle: Battle) -> Battle:
        """Store a reference to battle state."""
        # Note: In practice, we'd want to store only necessary info
        # For now, just store the battle reference
        return battle
    
    def compute_reward(self, battle: Battle, done: bool) -> float:
        """
        Compute reward for current battle state.
        
        Args:
            battle: Current battle state
            done: Whether episode is done
            
        Returns:
            Reward value
        """
        # Get previous state
        previous_battle = self.previous_battle_states.get(battle.battle_tag, battle)
        
        # Compute reward using reward function
        reward = self.reward_function.calculate_reward(
            current_battle=battle,
            previous_battle=previous_battle,
            action=0,  # Action not used in current reward functions
            done=done
        )
        
        return reward
    
    def decay_epsilon(self):
        """Decay epsilon for exploration."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def set_training_mode(self, training: bool):
        """Set training mode."""
        self.training_mode = training
        if training:
            self.network.train()
        else:
            self.network.eval()
    
    def save_model(self, path: str):
        """Save model weights."""
        torch.save(self.network.state_dict(), path)
    
    def load_model(self, path: str):
        """Load model weights."""
        self.network.load_state_dict(torch.load(path, map_location=self.device))


class ReplayBuffer:
    """
    Experience replay buffer for off-policy algorithms like DQN.
    """
    
    def __init__(self, capacity: int):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer."""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int):
        """Sample a batch of experiences."""
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        return (
            np.array(states),
            np.array(actions),
            np.array(rewards),
            np.array(next_states),
            np.array(dones)
        )
    
    def __len__(self):
        return len(self.buffer)
