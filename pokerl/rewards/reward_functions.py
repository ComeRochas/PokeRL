"""
Reward functions for Pokemon battle RL agent.

This module provides various reward shaping strategies that can be easily swapped.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from poke_env.environment import Battle


class RewardFunction(ABC):
    """Base class for reward functions."""
    
    @abstractmethod
    def calculate_reward(
        self, 
        current_battle: Battle, 
        previous_battle: Battle,
        action: int,
        done: bool
    ) -> float:
        """
        Calculate reward for a battle state transition.
        
        Args:
            current_battle: Current battle state
            previous_battle: Previous battle state
            action: Action taken
            done: Whether episode is done
            
        Returns:
            Reward value
        """
        pass


class DefaultRewardFunction(RewardFunction):
    """
    Default reward function based on battle outcome and state changes.
    
    Rewards:
    - Win: +1.0
    - Loss: -1.0
    - Faint opponent: +0.1
    - Lose own Pokemon: -0.1
    - HP delta: small weight on HP changes
    """
    
    def __init__(
        self,
        win_reward: float = 1.0,
        lose_reward: float = -1.0,
        faint_reward: float = 0.1,
        faint_penalty: float = -0.1,
        hp_delta_weight: float = 0.01,
        turn_penalty: float = -0.001
    ):
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.faint_reward = faint_reward
        self.faint_penalty = faint_penalty
        self.hp_delta_weight = hp_delta_weight
        self.turn_penalty = turn_penalty
    
    def calculate_reward(
        self, 
        current_battle: Battle, 
        previous_battle: Battle,
        action: int,
        done: bool
    ) -> float:
        """Calculate reward based on battle state changes."""
        reward = 0.0
        
        # Terminal rewards
        if done:
            if current_battle.won:
                reward += self.win_reward
            else:
                reward += self.lose_reward
            return reward
        
        # Intermediate rewards based on state changes
        # Count fainted Pokemon
        current_fainted = sum(1 for p in current_battle.team.values() if p.fainted)
        previous_fainted = sum(1 for p in previous_battle.team.values() if p.fainted)
        
        current_opponent_fainted = sum(
            1 for p in current_battle.opponent_team.values() if p.fainted
        )
        previous_opponent_fainted = sum(
            1 for p in previous_battle.opponent_team.values() if p.fainted
        )
        
        # Reward for fainting opponent's Pokemon
        if current_opponent_fainted > previous_opponent_fainted:
            reward += self.faint_reward * (current_opponent_fainted - previous_opponent_fainted)
        
        # Penalty for losing own Pokemon
        if current_fainted > previous_fainted:
            reward += self.faint_penalty * (current_fainted - previous_fainted)
        
        # Small penalty per turn to encourage efficiency
        reward += self.turn_penalty
        
        return reward


class SparseRewardFunction(RewardFunction):
    """Sparse reward function - only rewards at end of battle."""
    
    def __init__(self, win_reward: float = 1.0, lose_reward: float = -1.0):
        self.win_reward = win_reward
        self.lose_reward = lose_reward
    
    def calculate_reward(
        self, 
        current_battle: Battle, 
        previous_battle: Battle,
        action: int,
        done: bool
    ) -> float:
        """Only give reward at end of battle."""
        if not done:
            return 0.0
        
        return self.win_reward if current_battle.won else self.lose_reward


class HPDeltaRewardFunction(RewardFunction):
    """
    Reward function based on HP delta.
    
    Focuses on maximizing team HP while minimizing opponent HP.
    """
    
    def __init__(
        self,
        win_reward: float = 1.0,
        lose_reward: float = -1.0,
        hp_weight: float = 0.1
    ):
        self.win_reward = win_reward
        self.lose_reward = lose_reward
        self.hp_weight = hp_weight
    
    def calculate_reward(
        self, 
        current_battle: Battle, 
        previous_battle: Battle,
        action: int,
        done: bool
    ) -> float:
        """Calculate reward based on HP changes."""
        if done:
            return self.win_reward if current_battle.won else self.lose_reward
        
        # Calculate total HP percentages
        def total_hp_percentage(battle, team_type='team'):
            team = battle.team if team_type == 'team' else battle.opponent_team
            if not team:
                return 0.0
            total = sum(p.current_hp_fraction for p in team.values())
            return total / len(team)
        
        # Current HP deltas
        current_team_hp = total_hp_percentage(current_battle, 'team')
        current_opp_hp = total_hp_percentage(current_battle, 'opponent')
        
        # Previous HP deltas  
        previous_team_hp = total_hp_percentage(previous_battle, 'team')
        previous_opp_hp = total_hp_percentage(previous_battle, 'opponent')
        
        # Reward = increase in our HP - decrease in opponent HP
        team_delta = current_team_hp - previous_team_hp
        opp_delta = previous_opp_hp - current_opp_hp
        
        reward = (team_delta + opp_delta) * self.hp_weight
        
        return reward


def get_reward_function(config: Dict[str, Any]) -> RewardFunction:
    """
    Factory function to create reward function from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        RewardFunction instance
    """
    reward_type = config.get("type", "default")
    
    if reward_type == "default":
        return DefaultRewardFunction(
            win_reward=config.get("win_reward", 1.0),
            lose_reward=config.get("lose_reward", -1.0),
            faint_reward=config.get("faint_reward", 0.1),
            faint_penalty=config.get("faint_penalty", -0.1),
            hp_delta_weight=config.get("hp_delta_weight", 0.01),
            turn_penalty=config.get("turn_penalty", -0.001)
        )
    elif reward_type == "sparse":
        return SparseRewardFunction(
            win_reward=config.get("win_reward", 1.0),
            lose_reward=config.get("lose_reward", -1.0)
        )
    elif reward_type == "hp_delta":
        return HPDeltaRewardFunction(
            win_reward=config.get("win_reward", 1.0),
            lose_reward=config.get("lose_reward", -1.0),
            hp_weight=config.get("hp_delta_weight", 0.1)
        )
    else:
        raise ValueError(f"Unknown reward type: {reward_type}")
