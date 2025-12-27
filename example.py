"""
Quick example demonstrating the PokeRL framework.

This script shows how to:
1. Create components from config
2. Set up an agent
3. Run a simple battle
"""

import asyncio
from pokerl.utils.helpers import load_config, get_device
from pokerl.observations.observation_space import get_observation_space
from pokerl.rewards.reward_functions import get_reward_function
from pokerl.networks.architectures import get_network
from pokerl.opponents.strategies import get_opponent_strategy
from pokerl.agents.rl_agent import RLAgent


async def run_example():
    """Run a simple example battle."""
    
    print("PokeRL Framework Example")
    print("=" * 60)
    
    # Load configuration
    print("\n1. Loading configuration...")
    config = load_config("config.yaml")
    device = get_device()
    print(f"   Using device: {device}")
    
    # Create observation space
    print("\n2. Creating observation space...")
    observation_space = get_observation_space(config["observation"])
    obs_shape = observation_space.observation_shape()
    print(f"   Observation shape: {obs_shape}")
    
    # Create reward function
    print("\n3. Creating reward function...")
    reward_function = get_reward_function(config["reward"])
    print(f"   Reward type: {config['reward']['type']}")
    
    # Create neural network
    print("\n4. Creating neural network...")
    obs_dim = obs_shape[0]
    action_dim = 9  # 4 moves + 5 switches
    network = get_network(
        config["network"]["type"],
        obs_dim,
        action_dim,
        config["network"]
    )
    print(f"   Network type: {config['network']['type']}")
    print(f"   Input dim: {obs_dim}, Output dim: {action_dim}")
    
    # Create agent
    print("\n5. Creating RL agent...")
    agent = RLAgent(
        observation_space=observation_space,
        reward_function=reward_function,
        network=network,
        battle_format=config["environment"]["battle_format"],
        epsilon=1.0,  # Full exploration for demo
        device=device
    )
    print(f"   Agent created with battle format: {config['environment']['battle_format']}")
    
    # Create opponent
    print("\n6. Creating opponent...")
    opponent_strategy = get_opponent_strategy(config["opponent"])
    opponent = opponent_strategy.get_opponent(config["environment"]["battle_format"])
    print(f"   Opponent type: {config['opponent']['type']}")
    
    # Run a single battle
    print("\n7. Running a demonstration battle...")
    print("   (This may take a moment...)")
    
    try:
        await agent.battle_against(opponent, n_battles=1)
        
        # Get battle results
        battle = list(agent.battles.values())[-1]
        
        print(f"\n   Battle complete!")
        print(f"   Result: {'WIN' if battle.won else 'LOSS'}")
        print(f"   Battle length: {battle.turn} turns")
        
        # Compute reward
        reward = agent.compute_reward(battle, done=True)
        print(f"   Final reward: {reward:.2f}")
        
    except Exception as e:
        print(f"\n   Error during battle: {e}")
        print("   This is normal if Pokemon Showdown server is not accessible.")
    
    print("\n" + "=" * 60)
    print("Example complete!")
    print("\nNext steps:")
    print("  - Modify config.yaml to customize training")
    print("  - Run 'python train.py' to train an agent")
    print("  - Run 'python test.py --model models/model_final.pth' to evaluate")


if __name__ == "__main__":
    asyncio.run(run_example())
