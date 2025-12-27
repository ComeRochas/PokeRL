"""
Testing/Evaluation script for RL agent.

This script evaluates a trained agent against various opponents.
"""

import asyncio
import argparse
import os
import json
from datetime import datetime
from typing import Dict, List
import numpy as np
from tqdm import tqdm

from pokerl.utils.helpers import load_config, get_device
from pokerl.observations.observation_space import get_observation_space
from pokerl.rewards.reward_functions import get_reward_function
from pokerl.networks.architectures import get_network
from pokerl.opponents.strategies import get_opponent_strategy
from pokerl.agents.rl_agent import RLAgent


class AgentEvaluator:
    """Evaluator for trained RL agents."""
    
    def __init__(self, config, model_path: str):
        self.config = config
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Create components
        self.observation_space = get_observation_space(config["observation"])
        self.reward_function = get_reward_function(config["reward"])
        
        # Get observation and action dimensions
        obs_shape = self.observation_space.observation_shape()
        self.obs_dim = obs_shape[0]
        self.action_dim = 9  # 4 moves + 5 switches (max)
        
        # Create network
        self.network = get_network(
            config["network"]["type"],
            self.obs_dim,
            self.action_dim,
            config["network"]
        )
        
        # Create agent
        self.agent = RLAgent(
            observation_space=self.observation_space,
            reward_function=self.reward_function,
            network=self.network,
            battle_format=config["environment"]["battle_format"],
            epsilon=0.0,  # No exploration during evaluation
            device=self.device
        )
        
        # Load model
        if os.path.exists(model_path):
            self.agent.load_model(model_path)
            print(f"Model loaded from {model_path}")
        else:
            print(f"Warning: Model file {model_path} not found. Using untrained model.")
        
        # Set to evaluation mode
        self.agent.set_training_mode(False)
    
    async def evaluate_against_opponent(
        self,
        opponent_type: str,
        num_battles: int
    ) -> Dict[str, float]:
        """
        Evaluate agent against a specific opponent type.
        
        Args:
            opponent_type: Type of opponent
            num_battles: Number of battles to run
            
        Returns:
            Dictionary with evaluation metrics
        """
        print(f"\nEvaluating against {opponent_type} opponent ({num_battles} battles)...")
        
        # Create opponent config
        opponent_config = {
            "type": opponent_type,
            "mix_opponents": False
        }
        opponent_strategy = get_opponent_strategy(opponent_config)
        
        # Run battles
        wins = 0
        total_rewards = []
        battle_lengths = []
        
        for i in tqdm(range(num_battles), desc=f"vs {opponent_type}"):
            opponent = opponent_strategy.get_opponent(
                self.config["environment"]["battle_format"]
            )
            
            # Run battle
            await self.agent.battle_against(opponent, n_battles=1)
            
            # Get results
            battle = list(self.agent.battles.values())[-1]
            
            if battle.won:
                wins += 1
            
            # Compute reward
            reward = self.agent.compute_reward(battle, done=True)
            total_rewards.append(reward)
            
            # Battle length
            battle_lengths.append(battle.turn)
        
        # Compute metrics
        win_rate = wins / num_battles
        avg_reward = np.mean(total_rewards)
        std_reward = np.std(total_rewards)
        avg_battle_length = np.mean(battle_lengths)
        
        results = {
            "opponent_type": opponent_type,
            "num_battles": num_battles,
            "wins": wins,
            "win_rate": win_rate,
            "avg_reward": avg_reward,
            "std_reward": std_reward,
            "avg_battle_length": avg_battle_length
        }
        
        print(f"Results vs {opponent_type}:")
        print(f"  Win Rate: {win_rate:.2%}")
        print(f"  Avg Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Avg Battle Length: {avg_battle_length:.1f} turns")
        
        return results
    
    async def full_evaluation(self) -> Dict[str, any]:
        """
        Run full evaluation against all configured opponents.
        
        Returns:
            Dictionary with all evaluation results
        """
        print("\n" + "="*60)
        print("Starting Full Evaluation")
        print("="*60)
        
        num_episodes = self.config["evaluation"]["num_episodes"]
        opponent_types = self.config["evaluation"]["opponent_types"]
        
        all_results = {}
        
        for opponent_type in opponent_types:
            results = await self.evaluate_against_opponent(
                opponent_type,
                num_episodes
            )
            all_results[opponent_type] = results
        
        # Compute overall statistics
        overall_win_rate = np.mean([r["win_rate"] for r in all_results.values()])
        overall_avg_reward = np.mean([r["avg_reward"] for r in all_results.values()])
        
        all_results["overall"] = {
            "overall_win_rate": overall_win_rate,
            "overall_avg_reward": overall_avg_reward
        }
        
        print("\n" + "="*60)
        print("Overall Evaluation Results")
        print("="*60)
        print(f"Overall Win Rate: {overall_win_rate:.2%}")
        print(f"Overall Avg Reward: {overall_avg_reward:.2f}")
        
        # Save results if configured
        if self.config["evaluation"]["save_results"]:
            results_dir = self.config["evaluation"]["results_path"]
            os.makedirs(results_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = os.path.join(results_dir, f"evaluation_{timestamp}.json")
            
            with open(results_file, 'w') as f:
                json.dump(all_results, f, indent=2)
            
            print(f"\nResults saved to {results_file}")
        
        return all_results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="Evaluate trained RL agent")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="models/model_final.pth",
        help="Path to trained model"
    )
    parser.add_argument(
        "--num-battles",
        type=int,
        default=None,
        help="Number of battles per opponent (overrides config)"
    )
    parser.add_argument(
        "--opponent",
        type=str,
        default=None,
        help="Specific opponent to test against (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override num battles if specified
    if args.num_battles:
        config["evaluation"]["num_episodes"] = args.num_battles
    
    # Override opponent types if specified
    if args.opponent:
        config["evaluation"]["opponent_types"] = [args.opponent]
    
    # Create evaluator
    evaluator = AgentEvaluator(config, args.model)
    
    # Run evaluation
    asyncio.run(evaluator.full_evaluation())


if __name__ == "__main__":
    main()
