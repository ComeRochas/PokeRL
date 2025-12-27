"""
PokeRL: Reinforcement Learning for Pokemon Battles

A modular framework for training RL agents to play Pokemon battles using poke-env.
"""

__version__ = "0.1.0"

from pokerl.agents.rl_agent import RLAgent
from pokerl.rewards.reward_functions import RewardFunction
from pokerl.observations.observation_space import ObservationSpace
from pokerl.networks.architectures import DQNNetwork

__all__ = [
    "RLAgent",
    "RewardFunction",
    "ObservationSpace",
    "DQNNetwork",
]
