"""
Opponent strategies for Pokemon battles.

This module provides different opponent strategies for training and evaluation.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any
from poke_env.player import Player, RandomPlayer, MaxBasePowerPlayer, SimpleHeuristicsPlayer


class OpponentStrategy(ABC):
    """Base class for opponent strategies."""
    
    @abstractmethod
    def get_opponent(self, battle_format: str) -> Player:
        """
        Get an opponent player instance.
        
        Args:
            battle_format: Battle format (e.g., "gen8randombattle")
            
        Returns:
            Player instance
        """
        pass


class RandomOpponent(OpponentStrategy):
    """Random action opponent."""
    
    def get_opponent(self, battle_format: str) -> Player:
        """Get random opponent."""
        return RandomPlayer(battle_format=battle_format)


class MaxDamageOpponent(OpponentStrategy):
    """Opponent that always chooses max base power move."""
    
    def get_opponent(self, battle_format: str) -> Player:
        """Get max damage opponent."""
        return MaxBasePowerPlayer(battle_format=battle_format)


class SmartOpponent(OpponentStrategy):
    """Opponent using simple heuristics."""
    
    def get_opponent(self, battle_format: str) -> Player:
        """Get smart heuristic opponent."""
        return SimpleHeuristicsPlayer(battle_format=battle_format)


class TrainedOpponent(OpponentStrategy):
    """
    Opponent using a trained RL agent.
    
    This allows self-play or training against previous versions.
    """
    
    def __init__(self, model_path: str):
        self.model_path = model_path
    
    def get_opponent(self, battle_format: str) -> Player:
        """Get trained opponent."""
        # This would load a trained agent
        # For now, fallback to smart opponent
        # TODO: Implement loading trained model
        return SimpleHeuristicsPlayer(battle_format=battle_format)


class MixedOpponent(OpponentStrategy):
    """
    Mix of different opponent strategies.
    
    Randomly selects from a pool of opponents for more diverse training.
    """
    
    def __init__(self, strategies: list):
        self.strategies = strategies
        self.current_idx = 0
    
    def get_opponent(self, battle_format: str) -> Player:
        """Get next opponent from the mix."""
        import random
        strategy = random.choice(self.strategies)
        return strategy.get_opponent(battle_format)


def get_opponent_strategy(config: Dict[str, Any]) -> OpponentStrategy:
    """
    Factory function to create opponent strategy from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        OpponentStrategy instance
    """
    opponent_type = config.get("type", "random")
    
    if opponent_type == "random":
        return RandomOpponent()
    elif opponent_type == "max_damage":
        return MaxDamageOpponent()
    elif opponent_type == "smart_switch":
        return SmartOpponent()
    elif opponent_type == "trained":
        model_path = config.get("model_path", "models/opponent.pth")
        return TrainedOpponent(model_path)
    elif opponent_type == "mixed" or config.get("mix_opponents", False):
        # Create mixed opponents
        opponent_mix = config.get("opponent_mix", ["random", "max_damage"])
        strategies = []
        for opp_type in opponent_mix:
            if opp_type == "random":
                strategies.append(RandomOpponent())
            elif opp_type == "max_damage":
                strategies.append(MaxDamageOpponent())
            elif opp_type == "smart_switch":
                strategies.append(SmartOpponent())
        return MixedOpponent(strategies)
    else:
        raise ValueError(f"Unknown opponent type: {opponent_type}")
