import gymnasium as gym
import numpy as np


# Discrete environments
class FrozenLake:
    def __init__(
        self, map_name: str = "4x4", is_slippery: bool = True, render_mode: str = "human", render_fps: int = 80
    ):
        self.gym_env = gym.make(
            "FrozenLake-v1", desc=None, map_name=map_name, is_slippery=is_slippery, render_mode=render_mode
        )
        self.gym_env.metadata["render_fps"] = render_fps

    def get_num_discrete_states(self) -> list[int]:
        return [self.gym_env.observation_space.n]


# Continous environments
class MountainCar:
    def __init__(self, render_mode: str = "human", render_fps: int = 80, discretize_grid_length: int = 20):
        self.discretize_grid_length = discretize_grid_length
        self.gym_env = gym.make("MountainCar-v0", render_mode=render_mode)
        self.gym_env.metadata["render_fps"] = render_fps

        self.state_discrete_space = [
            np.linspace(
                self.gym_env.observation_space.low[i],
                self.gym_env.observation_space.high[i],
                self.discretize_grid_length,
            )
            for i in range(len(self.gym_env.observation_space.high))
        ]

    def get_num_discrete_states(self) -> list[int]:
        return [len(state_element) for state_element in self.state_discrete_space]


class LunarLander:
    def __init__(
        self,
        gravity: int = -10,
        continuous: bool = False,
        render_mode: str = "human",
        coordinate_grid_length: int = 16,
        velocity_grid_length: int = 16,
        tilt_grid_length: int = 16,
        render_fps: int = 60,
    ):
        self.gym_env = gym.make("LunarLander-v2", continuous=continuous, gravity=gravity, render_mode=render_mode)
        self.gym_env.metadata["render_fps"] = render_fps
        self.coordinate_grid_length = coordinate_grid_length
        self.velocity_grid_length = velocity_grid_length
        self.tilt_grid_length = tilt_grid_length
        k = [
            self.coordinate_grid_length,
            self.coordinate_grid_length,
            self.velocity_grid_length,
            self.velocity_grid_length,
            self.tilt_grid_length,
            self.tilt_grid_length,
            2,
            2,
        ]
        # we add small number for the binary foot-on-ground space because digitise index goes out of bound
        # this error:
        # >>> a = np.linspace(-0., 1., 2, endpoint=True)
        # >>> b = np.digitize(1., a)
        # >>> b
        # 2
        position_i = [0, 1, 4]
        velocity_i = [2, 3, 5]
        self.state_discrete_space = [
            np.linspace(
                (self.gym_env.observation_space.low[i] - 0.2)
                if (i in position_i)
                else (
                    (self.gym_env.observation_space.low[i] - 0.2)
                    if (i in velocity_i)
                    else (self.gym_env.observation_space.low[i])
                ),
                (self.gym_env.observation_space.high[i] + 0.2)
                if (i in position_i)
                else (
                    (self.gym_env.observation_space.high[i] + 0.2)
                    if (i in velocity_i)
                    else (self.gym_env.observation_space.high[i] + 1e-6)
                ),
                k[i],
            )
            for i in range(len(self.gym_env.observation_space.high))
        ]

    def get_num_discrete_states(self) -> list[int]:
        return [len(state_element) for state_element in self.state_discrete_space]
