"""
Observation space for Pokemon battle RL agent.

This module converts poke-env battle states into observation vectors.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, List
import numpy as np
from poke_env.environment import Battle, Pokemon, Move, PokemonType


class ObservationSpace(ABC):
    """Base class for observation space."""
    
    @abstractmethod
    def get_observation(self, battle: Battle) -> np.ndarray:
        """
        Convert battle state to observation vector.
        
        Args:
            battle: Current battle state
            
        Returns:
            Observation vector as numpy array
        """
        pass
    
    @abstractmethod
    def observation_shape(self) -> tuple:
        """Return the shape of the observation space."""
        pass


class EmbeddingObservationSpace(ObservationSpace):
    """
    Observation space using embeddings for Pokemon, moves, types, etc.
    
    Creates a fixed-size observation vector containing:
    - Active Pokemon stats (HP, Attack, Defense, Speed, etc.)
    - Active Pokemon types
    - Active Pokemon moves
    - Active Pokemon status
    - Opponent active Pokemon info (what's visible)
    - Team and opponent team remaining Pokemon count
    - Weather, terrain, and field effects
    """
    
    def __init__(
        self,
        include_stats: bool = True,
        include_types: bool = True,
        include_moves: bool = True,
        include_status: bool = True,
        include_weather: bool = True,
        include_terrain: bool = True,
        normalize: bool = True
    ):
        self.include_stats = include_stats
        self.include_types = include_types
        self.include_moves = include_moves
        self.include_status = include_status
        self.include_weather = include_weather
        self.include_terrain = include_terrain
        self.normalize = normalize
        
        # Pre-calculate observation size
        self._obs_size = self._calculate_obs_size()
    
    def _calculate_obs_size(self) -> int:
        """Calculate total observation size."""
        size = 0
        
        # Stats: HP, Attack, Defense, Sp.Atk, Sp.Def, Speed (normalized)
        if self.include_stats:
            size += 6 * 2  # Own and opponent active Pokemon
        
        # Types: One-hot encoding for 18 types * 2 (dual type)
        if self.include_types:
            size += 18 * 2 * 2  # Own and opponent, dual typing
        
        # Moves: 4 moves with properties (type, power, accuracy, pp)
        if self.include_moves:
            size += 4 * (18 + 3)  # 18 for type one-hot, 3 for power/acc/pp
        
        # Status: One-hot for status conditions (7 states)
        if self.include_status:
            size += 7 * 2  # Own and opponent
        
        # Weather: One-hot for weather (6 states)
        if self.include_weather:
            size += 6
        
        # Terrain: One-hot for terrain (5 states)
        if self.include_terrain:
            size += 5
        
        # Team info: Remaining Pokemon count
        size += 2  # Own team and opponent team
        
        return size
    
    def observation_shape(self) -> tuple:
        """Return the shape of the observation space."""
        return (self._obs_size,)
    
    def _encode_type(self, pokemon_type: PokemonType) -> np.ndarray:
        """Encode Pokemon type as one-hot vector."""
        type_encoding = np.zeros(18)
        if pokemon_type is not None:
            type_idx = pokemon_type.value - 1  # Types are 1-indexed
            if 0 <= type_idx < 18:
                type_encoding[type_idx] = 1
        return type_encoding
    
    def _encode_pokemon_types(self, pokemon: Pokemon) -> np.ndarray:
        """Encode Pokemon's dual types."""
        type1 = self._encode_type(pokemon.type_1) if pokemon.type_1 else np.zeros(18)
        type2 = self._encode_type(pokemon.type_2) if pokemon.type_2 else np.zeros(18)
        return np.concatenate([type1, type2])
    
    def _encode_stats(self, pokemon: Pokemon) -> np.ndarray:
        """Encode Pokemon stats."""
        if pokemon is None:
            return np.zeros(6)
        
        stats = np.array([
            pokemon.current_hp_fraction,  # Already normalized
            pokemon.base_stats.get("atk", 100) / 255.0,  # Normalize by max stat
            pokemon.base_stats.get("def", 100) / 255.0,
            pokemon.base_stats.get("spa", 100) / 255.0,
            pokemon.base_stats.get("spd", 100) / 255.0,
            pokemon.base_stats.get("spe", 100) / 255.0,
        ])
        
        return stats
    
    def _encode_status(self, pokemon: Pokemon) -> np.ndarray:
        """Encode Pokemon status condition."""
        # Status: none, burn, freeze, paralysis, poison, badly poison, sleep
        status_encoding = np.zeros(7)
        if pokemon and pokemon.status:
            status_map = {
                None: 0,
                'brn': 1,
                'frz': 2,
                'par': 3,
                'psn': 4,
                'tox': 5,
                'slp': 6
            }
            status_idx = status_map.get(pokemon.status.name if hasattr(pokemon.status, 'name') else None, 0)
            status_encoding[status_idx] = 1
        else:
            status_encoding[0] = 1  # No status
        return status_encoding
    
    def _encode_moves(self, pokemon: Pokemon) -> np.ndarray:
        """Encode Pokemon's moves."""
        moves_encoding = []
        
        for i in range(4):
            if pokemon and pokemon.moves and i < len(pokemon.moves):
                move = list(pokemon.moves.values())[i]
                # Move type (one-hot)
                move_type = self._encode_type(move.type) if move.type else np.zeros(18)
                # Move power, accuracy, pp (normalized)
                move_power = (move.base_power / 250.0) if move.base_power else 0.0
                move_acc = (move.accuracy if move.accuracy else 100) / 100.0
                move_pp = (move.current_pp / move.max_pp) if move.max_pp else 0.0
                
                move_encoding = np.concatenate([
                    move_type,
                    [move_power, move_acc, move_pp]
                ])
            else:
                # Empty move slot
                move_encoding = np.zeros(18 + 3)
            
            moves_encoding.append(move_encoding)
        
        return np.concatenate(moves_encoding)
    
    def _encode_weather(self, battle: Battle) -> np.ndarray:
        """Encode weather condition."""
        # Weather: none, sun, rain, sandstorm, hail, snow
        weather_encoding = np.zeros(6)
        weather_map = {
            None: 0,
            'sunnyday': 1,
            'raindance': 2,
            'sandstorm': 3,
            'hail': 4,
            'snow': 5
        }
        weather_name = battle.weather.name if battle.weather else None
        weather_idx = weather_map.get(weather_name, 0)
        weather_encoding[weather_idx] = 1
        return weather_encoding
    
    def _encode_terrain(self, battle: Battle) -> np.ndarray:
        """Encode terrain condition."""
        # Terrain: none, electric, grassy, misty, psychic
        terrain_encoding = np.zeros(5)
        terrain_map = {
            None: 0,
            'electricterrain': 1,
            'grassyterrain': 2,
            'mistyterrain': 3,
            'psychicterrain': 4
        }
        terrain_name = battle.fields.name if battle.fields else None
        terrain_idx = terrain_map.get(terrain_name, 0)
        terrain_encoding[terrain_idx] = 1
        return terrain_encoding
    
    def get_observation(self, battle: Battle) -> np.ndarray:
        """Convert battle state to observation vector."""
        observation_parts = []
        
        # Get active Pokemon
        active_pokemon = battle.active_pokemon
        opponent_pokemon = battle.opponent_active_pokemon
        
        # Stats
        if self.include_stats:
            own_stats = self._encode_stats(active_pokemon)
            opp_stats = self._encode_stats(opponent_pokemon)
            observation_parts.extend([own_stats, opp_stats])
        
        # Types
        if self.include_types:
            own_types = self._encode_pokemon_types(active_pokemon) if active_pokemon else np.zeros(36)
            opp_types = self._encode_pokemon_types(opponent_pokemon) if opponent_pokemon else np.zeros(36)
            observation_parts.extend([own_types, opp_types])
        
        # Moves (only for our Pokemon)
        if self.include_moves:
            moves = self._encode_moves(active_pokemon)
            observation_parts.append(moves)
        
        # Status
        if self.include_status:
            own_status = self._encode_status(active_pokemon)
            opp_status = self._encode_status(opponent_pokemon)
            observation_parts.extend([own_status, opp_status])
        
        # Weather
        if self.include_weather:
            weather = self._encode_weather(battle)
            observation_parts.append(weather)
        
        # Terrain
        if self.include_terrain:
            terrain = self._encode_terrain(battle)
            observation_parts.append(terrain)
        
        # Team info
        remaining_mons = sum(1 for mon in battle.team.values() if not mon.fainted)
        opponent_remaining = sum(1 for mon in battle.opponent_team.values() if not mon.fainted)
        team_info = np.array([remaining_mons / 6.0, opponent_remaining / 6.0])
        observation_parts.append(team_info)
        
        # Concatenate all parts
        observation = np.concatenate(observation_parts)
        
        return observation.astype(np.float32)


class SimplifiedObservationSpace(ObservationSpace):
    """
    Simplified observation space with just essential information.
    
    Good for faster training and simpler models.
    """
    
    def __init__(self):
        self._obs_size = 20  # Simplified feature count
    
    def observation_shape(self) -> tuple:
        """Return the shape of the observation space."""
        return (self._obs_size,)
    
    def get_observation(self, battle: Battle) -> np.ndarray:
        """Create simplified observation."""
        obs = np.zeros(self._obs_size)
        
        # Own Pokemon HP
        if battle.active_pokemon:
            obs[0] = battle.active_pokemon.current_hp_fraction
        
        # Opponent Pokemon HP
        if battle.opponent_active_pokemon:
            obs[1] = battle.opponent_active_pokemon.current_hp_fraction
        
        # Remaining Pokemon count
        obs[2] = sum(1 for mon in battle.team.values() if not mon.fainted) / 6.0
        obs[3] = sum(1 for mon in battle.opponent_team.values() if not mon.fainted) / 6.0
        
        # Type advantage (simplified)
        if battle.active_pokemon and battle.opponent_active_pokemon:
            # This is a placeholder - would need proper type effectiveness calculation
            obs[4] = 0.5  # Neutral by default
        
        # Number of available moves
        if battle.active_pokemon and battle.active_pokemon.moves:
            obs[5] = len(battle.active_pokemon.moves) / 4.0
        
        return obs.astype(np.float32)


def get_observation_space(config: Dict[str, Any]) -> ObservationSpace:
    """
    Factory function to create observation space from config.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        ObservationSpace instance
    """
    obs_type = config.get("type", "embeddings")
    
    if obs_type == "embeddings":
        return EmbeddingObservationSpace(
            include_stats=config.get("include_stats", True),
            include_types=config.get("include_types", True),
            include_moves=config.get("include_moves", True),
            include_status=config.get("include_status", True),
            include_weather=config.get("include_weather", True),
            include_terrain=config.get("include_terrain", True),
            normalize=config.get("normalize", True)
        )
    elif obs_type == "simplified":
        return SimplifiedObservationSpace()
    else:
        raise ValueError(f"Unknown observation type: {obs_type}")
