import pickle
import numpy as np


class RandomAgent:
    def getAction(self, actions):
        return np.random.choice(actions)

    def update(self, s, a, r, ns, *args, **kwargs):
        pass


class QLearningAgent:
    def __init__(
        self,
        num_actions: int,
        num_states: list[int],
        is_training: bool = True,
        trained_path: str | None = None,
        alpha: float = 0.1,
        gamma: float = 0.9,
        epsilon: float = 1,
        epsilon_decay: float = 0.0001,
    ):
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.num_actions = num_actions
        self.num_states = num_states
        self.is_training = is_training
        self.random_rng = np.random.default_rng()

        if self.is_training:
            self.Q_size = self.num_states
            self.Q_size.append(self.num_actions)
            self.Q = np.zeros(self.Q_size)
        else:
            self.Q = self.load_trained(trained_path)

    @staticmethod
    def load_trained(filepath: str):
        with open(filepath, "rb") as f:
            data = pickle.load(f)
        return data

    @staticmethod
    def save_trained(filepath: str, data) -> None:
        with open(filepath, "wb") as f:
            pickle.dump(data, f)

    def update_epsilon(self) -> None:
        self.epsilon = max(self.epsilon - self.epsilon_decay, 0)

    def getAction(self, state, actions):
        if self.is_training and self.random_rng.random() < self.epsilon:
            return actions.sample()
        else:
            if isinstance(state, tuple):
                state_tuple = state + (slice(None),)
                # print(f"all actions: {self.Q[state_tuple]}")
                return np.argmax(self.Q[state_tuple])
            else:
                return np.argmax(self.Q[state, :])

    def update(self, action, state, reward, new_state):
        if self.is_training:
            if isinstance(state, tuple) and isinstance(new_state, tuple):
                current_state = state + (action,)
                new_state_all_actions = new_state + (slice(None),)
                self.Q[current_state] = self.Q[current_state] + self.alpha * (
                    reward + self.gamma * np.max(self.Q[new_state_all_actions]) - self.Q[current_state]
                )
            else:
                self.Q[state, action] = self.Q[state, action] + self.alpha * (
                    reward + self.gamma * np.max(self.Q[new_state, :]) - self.Q[state, action]
                )
