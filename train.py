"""
Training script for RL agent to play Pokemon battles.

This script trains an agent using DQN or other RL algorithms.
"""

import asyncio
import argparse
import os
from datetime import datetime
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pokerl.utils.helpers import load_config, create_directories, get_device, count_parameters
from pokerl.observations.observation_space import get_observation_space
from pokerl.rewards.reward_functions import get_reward_function
from pokerl.networks.architectures import get_network
from pokerl.opponents.strategies import get_opponent_strategy
from pokerl.agents.rl_agent import RLAgent, ReplayBuffer


class DQNTrainer:
    """Trainer for DQN-based agents."""
    
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        print(f"Using device: {self.device}")
        
        # Create components
        self.observation_space = get_observation_space(config["observation"])
        self.reward_function = get_reward_function(config["reward"])
        self.opponent_strategy = get_opponent_strategy(config["opponent"])
        
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
        self.target_network = get_network(
            config["network"]["type"],
            self.obs_dim,
            self.action_dim,
            config["network"]
        )
        self.target_network.load_state_dict(self.network.state_dict())
        
        print(f"Network parameters: {count_parameters(self.network):,}")
        
        # Create agent
        self.agent = RLAgent(
            observation_space=self.observation_space,
            reward_function=self.reward_function,
            network=self.network,
            battle_format=config["environment"]["battle_format"],
            epsilon=config["training"]["epsilon_start"],
            epsilon_decay=config["training"]["epsilon_decay"],
            epsilon_min=config["training"]["epsilon_end"],
            device=self.device
        )
        
        # Training components
        self.optimizer = optim.Adam(
            self.network.parameters(),
            lr=config["training"]["learning_rate"]
        )
        self.replay_buffer = ReplayBuffer(config["training"]["memory_size"])
        
        # Training settings
        self.batch_size = config["training"]["batch_size"]
        self.gamma = config["training"]["gamma"]
        self.target_update_freq = config["training"]["target_update_freq"]
        
        # Logging
        create_directories(config)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.writer = SummaryWriter(f"{config['paths']['logs']}/run_{timestamp}")
        self.save_path = config["paths"]["models"]
        
    def train_step(self):
        """Perform one training step."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)
        
        # Compute Q values
        q_values = self.network(states)
        q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # Compute target Q values
        with torch.no_grad():
            next_q_values = self.target_network(next_states)
            next_q_value = next_q_values.max(1)[0]
            target_q_value = rewards + (1 - dones) * self.gamma * next_q_value
        
        # Compute loss
        loss = nn.MSELoss()(q_value, target_q_value)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    async def train(self, total_episodes: int):
        """
        Train the agent.
        
        Args:
            total_episodes: Total number of episodes to train
        """
        print(f"\nStarting training for {total_episodes} episodes...")
        
        episode_rewards = []
        episode_wins = []
        
        for episode in tqdm(range(total_episodes), desc="Training"):
            # Create opponent
            opponent = self.opponent_strategy.get_opponent(
                self.config["environment"]["battle_format"]
            )
            
            # Run battle
            await self.agent.battle_against(opponent, n_battles=1)
            
            # Get battle results
            battle = list(self.agent.battles.values())[-1]
            
            # Compute episode reward
            episode_reward = self.agent.compute_reward(battle, done=True)
            episode_rewards.append(episode_reward)
            episode_wins.append(1 if battle.won else 0)
            
            # Store experience (simplified - in practice, store each step)
            # This is a placeholder for the full experience collection
            
            # Train
            loss = self.train_step()
            
            # Decay epsilon
            self.agent.decay_epsilon()
            
            # Update target network
            if episode % self.target_update_freq == 0:
                self.target_network.load_state_dict(self.network.state_dict())
            
            # Logging
            if episode % self.config["training"]["log_freq"] == 0 and episode > 0:
                avg_reward = sum(episode_rewards[-100:]) / min(100, len(episode_rewards))
                win_rate = sum(episode_wins[-100:]) / min(100, len(episode_wins))
                
                self.writer.add_scalar("Reward/Average", avg_reward, episode)
                self.writer.add_scalar("WinRate/100Episodes", win_rate, episode)
                self.writer.add_scalar("Training/Epsilon", self.agent.epsilon, episode)
                self.writer.add_scalar("Training/Loss", loss, episode)
                
                print(f"\nEpisode {episode}: Avg Reward: {avg_reward:.2f}, "
                      f"Win Rate: {win_rate:.2%}, Epsilon: {self.agent.epsilon:.3f}")
            
            # Save model
            if episode % self.config["training"]["save_freq"] == 0 and episode > 0:
                save_file = os.path.join(self.save_path, f"model_episode_{episode}.pth")
                self.agent.save_model(save_file)
                print(f"Model saved to {save_file}")
        
        # Save final model
        final_save_file = os.path.join(self.save_path, "model_final.pth")
        self.agent.save_model(final_save_file)
        print(f"\nTraining complete! Final model saved to {final_save_file}")
        
        self.writer.close()


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train RL agent for Pokemon battles")
    parser.add_argument(
        "--config",
        type=str,
        default="config.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=None,
        help="Number of episodes to train (overrides config)"
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Override episodes if specified
    total_episodes = args.episodes if args.episodes else config["training"]["total_timesteps"]
    
    # Create trainer
    trainer = DQNTrainer(config)
    
    # Train
    asyncio.run(trainer.train(total_episodes))


if __name__ == "__main__":
    main()
