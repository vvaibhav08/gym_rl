## gym_rl

A classical Q-Learning algorithm applied on various OpenAI Gym environments.

### Presentation and article
An article was written for this work and can be found [here](https://polyester-flier-216.notion.site/Q-learning-on-simple-discrete-and-continuous-environments-1407273d8c64814ea894d5981c60e540).
A presentation was written for this work and can be found [here](https://vvaibhav08.github.io/RL_qlearning_presentation_082024.html).

### Code overview

This repository implements a Q-Learning agent for multiple gym environments, including:

- LunarLander-v2
- MountainCar-v0
- FrozenLake-v0

The agent is capable of training in these environments using the Q-Learning algorithm with customizable hyperparameters. The code supports saving and loading trained models, updating exploration rates, and visualizing performance metrics.

### Repository Structure
```
gym_rl/
├── README.md # This file. 
├── pyproject.toml # dependencies file. 
├── poetry.lock # lock file. 
└── src/ 
    ├── agents.py # Contains the QLearningAgent class with training, action selection, 
    │ # Q-value update, and model saving/loading functionalities. 
    ├── environments.py # Custom wrappers for gym environments. 
    └── train.py # Main script to run training sessions with command-line arguments for # hyperparameters and environment selection.
```

### Getting Started

#### Prerequisites

- Python 3.11+
- Required libraries:
  - numpy
  - pandas
  - plotly (for data visualization)
  - OpenAI Gym (compatible with the environments used)

The full list of dependencies can be found in `pyproject.toml`.

## Installation
Clone the repository:
```
git clone https://github.com/<your_username>/gym_rl.git
cd gym_rl
```

## Install dependencies via poetry:
```bash
poetry install
```

## Running the Training
To train a Q-Learning agent on a Gym environment (e.g., MountainCar-v0), run:
```
poetry run python train.py --env MountainCar-v0 --num_episodes 15000 --alpha 0.9 --gamma 0.9 --epsilon 1 --decay 0.00008
```

Command line arguments:

`--env`: Gym environment to use (e.g., LunarLander-v2, MountainCar-v0, FrozenLake-v0)
`--num_episodes`: Number of episodes for training (e.g., 15000)
`--alpha`: Learning rate
`--gamma`: Discount factor
`--epsilon`: Initial exploration rate
`--decay`: Epsilon decay rate

## Saving and Loading Models
The trained Q-values can be saved using the `save_trained` method in `QLearningAgent`. The model is serialized using Python’s pickle library.
To load a previously trained model, specify the file path with `--trained_weights_path` when running train.py.

## Project Details
**QLearningAgent**: Implements the Q-Learning algorithm with methods to select actions, update Q-values, and adjust exploration (epsilon).

**Training Script**: Runs episodes in the specified gym environment, collects rewards, and updates the QLearningAgent accordingly. Performance is logged and can be visualized using Plotly.

**Environments**: This project includes support for multiple gym environments, making it versatile for different types of reinforcement learning tasks.