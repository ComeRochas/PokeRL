# PokeRL

A modular reinforcement learning framework for training agents to play Pokemon battles using [poke-env](https://github.com/hsahovic/poke-env).

## Features

- 🎮 **Modular Architecture**: Easy to swap reward functions, observation spaces, neural networks, and opponent strategies
- 🧠 **Multiple RL Algorithms**: Support for DQN, Dueling DQN, PPO, and A2C
- 🎯 **Flexible Reward Shaping**: Choose from default, sparse, or HP-delta reward functions, or create your own
- 👁️ **Configurable Observations**: Full embeddings or simplified observation spaces
- 🤖 **Various Opponent Strategies**: Train against random, max damage, smart heuristic, or trained opponents
- ⚙️ **Easy Configuration**: All hyperparameters in a single YAML config file
- 📊 **Training Monitoring**: TensorBoard integration for tracking training progress
- 🧪 **Comprehensive Testing**: Evaluation script with detailed metrics

## Project Structure

```
PokeRL/
├── pokerl/                      # Main package
│   ├── agents/                  # RL agent implementations
│   │   └── rl_agent.py         # Main RL agent class
│   ├── observations/            # Observation space modules
│   │   └── observation_space.py # State representations
│   ├── rewards/                 # Reward function modules
│   │   └── reward_functions.py # Various reward shaping strategies
│   ├── networks/                # Neural network architectures
│   │   └── architectures.py    # DQN, Dueling DQN, Actor-Critic
│   ├── opponents/               # Opponent strategy modules
│   │   └── strategies.py       # Different opponent types
│   └── utils/                   # Utility functions
│       └── helpers.py          # Config loading, device setup, etc.
├── train.py                     # Training script
├── test.py                      # Evaluation script
├── config.yaml                  # Configuration file
└── requirements.txt             # Dependencies

```

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ComeRochas/PokeRL.git
cd PokeRL
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Quick Start

### Training an Agent

Train with default configuration:
```bash
python train.py
```

Train with custom config:
```bash
python train.py --config my_config.yaml
```

Train for specific number of episodes:
```bash
python train.py --episodes 10000
```

### Testing/Evaluating an Agent

Evaluate trained model:
```bash
python test.py --model models/model_final.pth
```

Evaluate against specific opponent:
```bash
python test.py --model models/model_final.pth --opponent max_damage
```

Custom number of evaluation battles:
```bash
python test.py --model models/model_final.pth --num-battles 500
```

## Configuration

All training and evaluation settings are in `config.yaml`. Key sections:

### Environment Settings
```yaml
environment:
  battle_format: "gen8randombattle"  # Battle format
```

### Reward Function
Choose from: `default`, `sparse`, `hp_delta`, or create your own
```yaml
reward:
  type: "default"
  win_reward: 1.0
  lose_reward: -1.0
  faint_reward: 0.1
  # ... more options
```

### Observation Space
Choose from: `embeddings`, `simplified`
```yaml
observation:
  type: "embeddings"
  include_stats: true
  include_types: true
  # ... more options
```

### Neural Network Architecture
Choose from: `dqn`, `dueling_dqn`, `ppo`, `a2c`
```yaml
network:
  type: "dqn"
  hidden_layers: [256, 256, 128]
  activation: "relu"
  # ... more options
```

### Training Settings
```yaml
training:
  algorithm: "dqn"
  total_timesteps: 100000
  learning_rate: 0.0001
  batch_size: 32
  gamma: 0.99
  # ... more options
```

### Opponent Strategy
Choose from: `random`, `max_damage`, `smart_switch`, `trained`, or mix multiple
```yaml
opponent:
  type: "random"
  mix_opponents: false
  opponent_mix: ["random", "max_damage"]
```

## Customization Guide

### Creating a Custom Reward Function

1. Open `pokerl/rewards/reward_functions.py`
2. Create a new class inheriting from `RewardFunction`:

```python
class MyCustomReward(RewardFunction):
    def calculate_reward(self, current_battle, previous_battle, action, done):
        # Your reward logic here
        return reward
```

3. Update the `get_reward_function()` factory to include your reward
4. Set `reward.type: "my_custom"` in config.yaml

### Creating a Custom Observation Space

1. Open `pokerl/observations/observation_space.py`
2. Create a new class inheriting from `ObservationSpace`:

```python
class MyCustomObservation(ObservationSpace):
    def observation_shape(self):
        return (my_size,)
    
    def get_observation(self, battle):
        # Your observation logic here
        return observation_vector
```

3. Update the `get_observation_space()` factory
4. Set `observation.type: "my_custom"` in config.yaml

### Using a Different Neural Network

1. Open `pokerl/networks/architectures.py`
2. Create a new network class or modify existing ones
3. Update `get_network()` factory if needed
4. Set `network.type` and `network.hidden_layers` in config.yaml

### Adding a New Opponent Strategy

1. Open `pokerl/opponents/strategies.py`
2. Create a new class inheriting from `OpponentStrategy`:

```python
class MyCustomOpponent(OpponentStrategy):
    def get_opponent(self, battle_format):
        # Return a poke-env Player instance
        return MyPlayer(battle_format=battle_format)
```

3. Update the `get_opponent_strategy()` factory
4. Set `opponent.type: "my_custom"` in config.yaml

## Monitoring Training

Training logs are saved to TensorBoard. View them with:
```bash
tensorboard --logdir logs/
```

Metrics tracked:
- Average reward per episode
- Win rate (last 100 episodes)
- Epsilon (exploration rate)
- Training loss

## Model Checkpoints

Models are saved to the `models/` directory:
- Periodic checkpoints: `model_episode_X.pth`
- Final model: `model_final.pth`

## Evaluation Results

Evaluation results are saved to `results/` directory as JSON files with:
- Win rates per opponent type
- Average rewards
- Battle statistics

## Advanced Usage

### Training Against Multiple Opponent Types

Set in config.yaml:
```yaml
opponent:
  mix_opponents: true
  opponent_mix: ["random", "max_damage", "smart_switch"]
```

### Using Different Devices

The framework automatically detects and uses the best available device (CUDA, MPS, or CPU).

### Hyperparameter Tuning

Modify `config.yaml` to experiment with:
- Learning rates
- Network architectures
- Batch sizes
- Exploration strategies
- Reward shaping

## Requirements

- Python 3.8+
- PyTorch 2.0+
- poke-env 0.9.0+
- See `requirements.txt` for full list

## Contributing

Contributions are welcome! Areas for improvement:
- Additional reward functions
- More sophisticated observation encodings
- Advanced RL algorithms (Rainbow DQN, SAC, etc.)
- Self-play training
- Opponent modeling

## License

This project is open source and available under the MIT License.

## Acknowledgments

- Built on [poke-env](https://github.com/hsahovic/poke-env) for Pokemon battle simulation
- Inspired by reinforcement learning research in competitive games
